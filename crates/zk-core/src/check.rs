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

// Chi-bridge import for chi_pool feature on wasm32 (blocking version for rayon
// workers)
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

// Terms-bridge import for terms_pool feature on wasm32 (blocking version for
// rayon workers)
#[cfg(all(target_arch = "wasm32", feature = "terms_pool"))]
#[wasm_bindgen::prelude::wasm_bindgen(raw_module = "./terms-bridge.js")]
extern "C" {
    /// Request terms computation from JS coordinator.
    /// This function BLOCKS via Atomics.wait() until result is ready.
    ///
    /// Parameters:
    ///   memory - WASM memory object (from wasm_bindgen::memory())
    ///   triples_ptr - pointer to triples in WASM memory (48 bytes per triple)
    ///   chis_ptr - pointer to chis in WASM memory (16 bytes per chi)
    ///   count - number of triples
    ///   result_ptr - where to write 32-byte result (u: 16 bytes, v: 16 bytes)
    fn request_terms_computation(
        memory: wasm_bindgen::JsValue,
        triples_ptr: u32,
        chis_ptr: u32,
        count: u32,
        result_ptr: u32,
    );
}

// Async worker-based imports for wasm32 (no rayon, main thread orchestration)
#[cfg(all(target_arch = "wasm32", feature = "wasm_workers"))]
#[wasm_bindgen::prelude::wasm_bindgen(raw_module = "./check-workers.js")]
extern "C" {
    /// Request chi computation from JS worker pool (async, returns Promise).
    ///
    /// Parameters:
    ///   chi - 16 byte chi seed
    ///   count - number of chi values to compute
    /// Returns: Uint8Array of count * 16 bytes
    #[wasm_bindgen(js_name = "computeChisAsync")]
    async fn compute_chis_async(chi: &[u8], count: u32) -> wasm_bindgen::JsValue;

    /// Request compute_terms from JS worker pool (async, returns Promise).
    ///
    /// Parameters:
    ///   triples - flattened triples as bytes (48 bytes per triple: x, y, z
    /// each 16 bytes)   chis - chi values as bytes (16 bytes each)
    /// Returns: Uint8Array of 32 bytes (u: 16 bytes, v: 16 bytes)
    #[wasm_bindgen(js_name = "computeTermsAsync")]
    async fn compute_terms_async(triples: &[u8], chis: &[u8]) -> wasm_bindgen::JsValue;
}

type Result<T> = core::result::Result<T, CheckError>;

/// Values sent from the prover to the verifier for the consistency check.
#[derive(Debug, Serialize, Deserialize)]
pub struct UV {
    u: Block,
    v: Block,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
        // Pass wasm_bindgen::memory() so it works in any context (main thread or
        // workers)
        request_chi_computation(wasm_bindgen::memory(), chi_ptr, n as u32, result_ptr);

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

    /// Compute terms via terms worker pool (blocking call via Atomics.wait).
    /// This offloads the expensive gfmul operations to private memory workers
    /// using fast polyval soft64 implementation.
    #[cfg(all(target_arch = "wasm32", feature = "terms_pool"))]
    fn compute_terms_via_pool(triples: &[Triple], chis: &[Block]) -> (Block, Block) {
        let n = triples.len();
        if n == 0 {
            return (Block::ZERO, Block::ZERO);
        }

        // Zero-copy cast triples to bytes (48 bytes per triple: x, y, z)
        let triples_bytes: &[u8] = bytemuck::cast_slice(triples);

        // Convert chis to bytes (16 bytes per chi)
        let chis_bytes: Vec<u8> = chis.iter().flat_map(|c| c.to_bytes()).collect();

        // Allocate result buffer (32 bytes: u, v)
        let mut result_bytes = [0u8; 32];

        // Get pointers - these are addresses in WASM linear memory
        let triples_ptr = triples_bytes.as_ptr() as u32;
        let chis_ptr = chis_bytes.as_ptr() as u32;
        let result_ptr = result_bytes.as_mut_ptr() as u32;

        // Call terms-bridge - blocks until result is ready
        request_terms_computation(
            wasm_bindgen::memory(),
            triples_ptr,
            chis_ptr,
            n as u32,
            result_ptr,
        );

        // Parse result (u: 16 bytes, v: 16 bytes)
        let u = Block::try_from(&result_bytes[0..16]).expect("u should be 16 bytes");
        let v = Block::try_from(&result_bytes[16..32]).expect("v should be 16 bytes");

        (u, v)
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
            if #[cfg(all(target_arch = "wasm32", feature = "terms_pool"))] {
                // Use terms worker pool via terms-bridge (blocking call)
                // This offloads the expensive gfmul operations to private memory workers
                // using fast polyval soft64 implementation
                let (mut u, mut v) = Self::compute_terms_via_pool(&macs, &chis);
            } else if #[cfg(feature = "rayon")] {
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

    /// Async version of check_prover for WASM using worker pools.
    /// Uses private memory workers for both chi computation and compute_terms.
    /// No rayon/SharedArrayBuffer contention.
    #[cfg(all(target_arch = "wasm32", feature = "wasm_workers"))]
    pub(crate) async fn check_prover_async(
        &mut self,
        transcript: &mut Hasher,
        svole_choices: &[bool],
        svole_ev: &[Block],
    ) -> Result<UV> {
        use js_sys::Uint8Array;
        use wasm_bindgen::JsCast;

        let adjust_len = self.adjust.len();
        transcript.update(&self.adjust.as_raw_slice().as_bytes()[..adjust_len.div_ceil(8)]);

        let chi = Block::try_from(&transcript.finalize().as_bytes()[..16])
            .expect("block should be 16 bytes");

        let n = self.triples.len();
        if n == 0 {
            let (a_0, a_1) = vole_receiver(
                svole_choices.try_into().map_err(|_| CheckError::SVole)?,
                svole_ev.try_into().map_err(|_| CheckError::SVole)?,
            );
            transcript.update(&a_0.to_bytes());
            transcript.update(&a_1.to_bytes());
            self.adjust.clear();
            return Ok(UV { u: a_0, v: a_1 });
        }

        // 1. Compute chis via worker pool (async, no blocking)
        let chi_bytes = chi.to_bytes();
        let chis_js = compute_chis_async(&chi_bytes, n as u32).await;
        let chis_array: Uint8Array = chis_js.unchecked_into();
        let chis_bytes = chis_array.to_vec();

        // 2. Zero-copy cast triples to bytes (48 bytes per triple: x, y, z)
        let macs = mem::take(&mut self.triples);
        let triples_bytes: &[u8] = bytemuck::cast_slice(&macs);

        // 3. Compute terms via worker pool (async, no blocking)
        let uv_js = compute_terms_async(triples_bytes, &chis_bytes).await;
        let uv_array: Uint8Array = uv_js.unchecked_into();
        let uv_bytes = uv_array.to_vec();

        // 4. Parse result (u: 16 bytes, v: 16 bytes)
        let mut u = Block::try_from(&uv_bytes[0..16]).expect("u should be 16 bytes");
        let mut v = Block::try_from(&uv_bytes[16..32]).expect("v should be 16 bytes");

        // 5. Apply sVOLE (cheap, main thread)
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
