//! QuickSilver consistency check.

use std::mem;

use blake3::Hasher;
use cfg_if::cfg_if;
use mpz_core::{
    Block,
    bitvec::{BitSlice, BitVec},
};
use serde::{Deserialize, Serialize};
use zerocopy::IntoBytes;

use crate::vole::{vole_receiver, vole_sender};

// Chi-bridge import for chi_pool feature on wasm32
#[cfg(all(target_arch = "wasm32", feature = "chi_pool"))]
#[wasm_bindgen::prelude::wasm_bindgen(raw_module = "./chi-bridge.js")]
extern "C" {
    /// Request chi computation from JS coordinator.
    /// This function BLOCKS via Atomics.wait() until result is ready.
    ///
    /// Parameters:
    ///   memory - WASM memory object (from wasm_bindgen::memory())
    ///   chi_ptr, count, result_ptr - raw pointers into WASM linear memory
    fn request_chi_computation(
        memory: wasm_bindgen::JsValue,
        chi_ptr: u32,
        count: u32,
        result_ptr: u32,
    );
}

type Result<T> = core::result::Result<T, CheckError>;

/// Values sent from the prover to the verifier for the consistency check.
#[derive(Debug, Serialize, Deserialize)]
pub struct UV {
    u: Block,
    v: Block,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Triple {
    pub(crate) x: Block,
    pub(crate) y: Block,
    pub(crate) z: Block,
}

#[derive(Debug, Default)]
pub(crate) struct Check {
    triples: Vec<Triple>,
    adjust: BitVec,
}

impl Check {
    /// Reserves capacity for at least `n` AND gates, returns the starting
    /// index.
    pub(crate) fn reserve(&mut self, n: usize) -> usize {
        let idx = self.triples.len();
        self.triples.resize_with(idx + n, Default::default);
        self.adjust.resize_with(idx + n, |_| Default::default());
        idx
    }

    pub(crate) fn write(&mut self, idx: usize, triples: &[Triple], adjust: &BitSlice) {
        self.triples[idx..idx + triples.len()].copy_from_slice(triples);
        self.adjust[idx..idx + triples.len()].copy_from_bitslice(adjust);
    }

    /// Returns `true` if there are gates to check.
    #[inline]
    pub(crate) fn wants_check(&self) -> bool {
        !self.triples.is_empty()
    }

    fn compute_chis(&self, chi: Block) -> Vec<Block> {
        let n = self.triples.len();
        if n == 0 {
            return Vec::new();
        }

        cfg_if! {
            if #[cfg(all(target_arch = "wasm32", feature = "chi_pool"))] {
                // Use chi worker pool via chi-bridge
                Self::compute_chis_via_chi_pool(chi, n)
            } else if #[cfg(feature = "rayon")] {
                Self::compute_chis_parallel(chi, n)
            } else {
                Self::compute_chis_sequential(chi, n)
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "chi_pool"))]
    fn compute_chis_via_chi_pool(chi: Block, n: usize) -> Vec<Block> {
        // Allocate memory for chi input (16 bytes) and result (n * 16 bytes)
        let chi_bytes = chi.to_bytes();
        let mut result_bytes = vec![0u8; n * 16];

        // Get pointers - these are addresses in WASM linear memory
        let chi_ptr = chi_bytes.as_ptr() as u32;
        let result_ptr = result_bytes.as_mut_ptr() as u32;

        // Call chi-bridge - blocks until result is ready
        // Pass wasm_bindgen::memory() so it works in any context (main thread or workers)
        request_chi_computation(
            wasm_bindgen::memory(),
            chi_ptr,
            n as u32,
            result_ptr,
        );

        // Convert result bytes to Vec<Block>
        result_bytes
            .chunks_exact(16)
            .map(|chunk| Block::try_from(chunk).expect("chunk is 16 bytes"))
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn compute_chis_parallel(chi: Block, n: usize) -> Vec<Block> {
        use rayon::prelude::*;

        const PARALLELISM: usize = 16;
        let segment_size = n.div_ceil(PARALLELISM);
        let starts = Self::compute_chi_starts(chi, segment_size);

        let segments: Vec<Vec<Block>> = starts
            .into_par_iter()
            .enumerate()
            .map(|(i, start)| {
                let seg_start = i * segment_size;
                let seg_end = ((i + 1) * segment_size).min(n);
                let seg_len = seg_end - seg_start;
                if seg_len == 0 {
                    return Vec::new();
                }
                let mut segment = Vec::with_capacity(seg_len);
                let mut current = start;
                segment.push(current);
                for _ in 1..seg_len {
                    current = current.gfmul(current);
                    segment.push(current);
                }
                segment
            })
            .collect();

        segments.into_iter().flatten().collect()
    }

    #[cfg(not(feature = "rayon"))]
    fn compute_chis_sequential(chi: Block, n: usize) -> Vec<Block> {
        const PARALLELISM: usize = 16;
        let segment_size = n.div_ceil(PARALLELISM);
        let starts = Self::compute_chi_starts(chi, segment_size);

        let mut chis = Vec::with_capacity(n);
        for (i, start) in starts.into_iter().enumerate() {
            let seg_start = i * segment_size;
            let seg_end = ((i + 1) * segment_size).min(n);
            let mut current = start;
            for _ in seg_start..seg_end {
                chis.push(current);
                current = current.gfmul(current);
            }
        }
        chis.truncate(n);
        chis
    }

    /// Computes independent starting points for parallel chi computation.
    /// Bootstrap 16 values via squaring, hash each to get independent starts.
    fn compute_chi_starts(chi: Block, segment_size: usize) -> [Block; 16] {
        use blake3::Hasher;

        // Bootstrap 16 values via squaring
        let mut bootstrapped = [Block::ZERO; 16];
        let mut current = chi;
        for b in &mut bootstrapped {
            *b = current;
            current = current.gfmul(current);
        }

        // Hash each to get independent starting points
        let mut starts = [Block::ZERO; 16];
        for (i, boot) in bootstrapped.iter().enumerate() {
            let mut hasher = Hasher::new();
            hasher.update(&boot.to_bytes());
            hasher.update(&(i as u64).to_le_bytes());
            hasher.update(&(segment_size as u64).to_le_bytes());
            let hash = hasher.finalize();
            starts[i] =
                Block::try_from(&hash.as_bytes()[..16]).expect("hash should be at least 16 bytes");
        }

        starts
    }

    /// Executes the prover check, returning `U` and `V` defined in Figure 5,
    /// Step 7.b.
    pub(crate) fn check_prover(
        &mut self,
        transcript: &mut Hasher,
        svole_choices: &[bool],
        svole_ev: &[Block],
    ) -> Result<UV> {
        #[inline]
        fn compute_terms(triple: Triple, chi: Block) -> (Block, Block) {
            let Triple { x, y, z } = triple;

            let u = x.gfmul(y).gfmul(chi);

            // (Note that the LSB of a MAC contains the authenticated bit).
            let a_10 = if x.lsb() { y } else { Block::ZERO };
            let a_11 = if y.lsb() { x } else { Block::ZERO };
            let v = (a_10 ^ a_11 ^ z).gfmul(chi);

            (u, v)
        }

        let adjust_len = self.adjust.len();
        transcript.update(&self.adjust.as_raw_slice().as_bytes()[..adjust_len.div_ceil(8)]);

        let chi = Block::try_from(&transcript.finalize().as_bytes()[..16])
            .expect("block should be 16 bytes");
        let chis = self.compute_chis(chi);
        let macs = mem::take(&mut self.triples);
        cfg_if! {
            if #[cfg(feature = "rayon")] {
                use rayon::prelude::*;

                let (mut u, mut v) = macs
                    .into_par_iter()
                    .zip(chis)
                    .map(|(macs, chi)| compute_terms(macs, chi))
                    .reduce(
                        || (Block::ZERO, Block::ZERO),
                        |(u_acc, v_acc), (u, v)| (u_acc ^ u, v_acc ^ v),
                    );
            } else {
                let (mut u, mut v) = macs
                    .into_iter()
                    .zip(chis)
                    .map(|(macs, chi)| compute_terms(macs, chi))
                    .fold(
                        (Block::ZERO, Block::ZERO),
                        |(u_acc, v_acc), (u, v)| (u_acc ^ u, v_acc ^ v),
                    );
            }
        }

        let (a_0, a_1) = vole_receiver(
            svole_choices.try_into().map_err(|_| CheckError::SVole)?,
            svole_ev.try_into().map_err(|_| CheckError::SVole)?,
        );

        u ^= a_0;
        v ^= a_1;

        transcript.update(&u.to_bytes());
        transcript.update(&v.to_bytes());

        self.adjust.clear();

        Ok(UV { u, v })
    }

    /// Executes the verifier check, returning `W` defined in Figure 5, Step
    /// 7.c.
    pub(crate) fn check_verifier(
        &mut self,
        transcript: &mut Hasher,
        delta: &Block,
        svole_keys: &[Block],
        uv: UV,
    ) -> Result<()> {
        #[inline]
        fn compute_term(triple: Triple, chi: Block, delta: &Block) -> Block {
            let Triple { x, y, z } = triple;
            let b = x.gfmul(y) ^ delta.gfmul(z);
            b.gfmul(chi)
        }

        let adjust_len = self.adjust.len();
        transcript.update(&self.adjust.as_raw_slice().as_bytes()[..adjust_len.div_ceil(8)]);

        let chi = Block::try_from(&transcript.finalize().as_bytes()[..16])
            .expect("block should be 16 bytes");
        let chis = self.compute_chis(chi);
        let keys = mem::take(&mut self.triples);
        cfg_if! {
            if #[cfg(feature = "rayon")] {
                use rayon::prelude::*;

                let mut w = keys
                    .into_par_iter()
                    .zip(chis)
                    .map(|(keys, chi)| compute_term(keys, chi, delta))
                    .reduce(
                        || Block::ZERO,
                        |w_acc, w| w_acc ^ w,
                    );
            } else {
                let mut w = keys
                    .into_iter()
                    .zip(chis)
                    .map(|(keys, chi)| compute_term(keys, chi, delta))
                    .fold(
                        Block::ZERO,
                        |w_acc, w| w_acc ^ w,
                    );
            }
        }

        let b = vole_sender(svole_keys.try_into().map_err(|_| CheckError::SVole)?);

        w ^= b;

        let UV { u, v } = uv;
        transcript.update(&u.to_bytes());
        transcript.update(&v.to_bytes());

        self.adjust.clear();

        if w != u ^ delta.gfmul(v) {
            // Invalid! Call the police.
            return Err(CheckError::Invalid);
        }

        Ok(())
    }

    /// Returns the total number of triples that need to be checked.
    pub(crate) fn total(&self) -> usize {
        self.triples.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckError {
    #[error("incorrect number of sVOLE instances provided")]
    SVole,
    #[error("invalid consistency check")]
    Invalid,
}
