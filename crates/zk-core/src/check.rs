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

type Result<T> = core::result::Result<T, CheckError>;

/// Bootstrap 16 independent random starting points from a single chi.
fn compute_chi_starts(chi: Block) -> [Block; 16] {
    // Bootstrap: compute χ, χ², χ⁴, ..., χ^(2^15)
    let mut bootstrap = [Block::ZERO; 16];
    bootstrap[0] = chi;
    for i in 1..16 {
        bootstrap[i] = bootstrap[i - 1].gfmul(bootstrap[i - 1]);
    }

    // Hash each to get independent random starting points
    std::array::from_fn(|i| {
        let mut hasher = Hasher::new();
        hasher.update(&(i as u64).to_le_bytes());
        hasher.update(&bootstrap[i].to_bytes());
        Block::try_from(&hasher.finalize().as_bytes()[..16])
            .expect("block should be 16 bytes")
    })
}

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

        let starts = compute_chi_starts(chi);
        let segment_size = n / 16;
        let remainder = n % 16;

        let compute_segment = |i: usize| {
            let count = segment_size + if i < remainder { 1 } else { 0 };
            let mut segment = Vec::with_capacity(count);
            let mut val = starts[i];
            for _ in 0..count {
                segment.push(val);
                val = val.gfmul(val);
            }
            segment
        };

        cfg_if! {
            if #[cfg(feature = "rayon")] {
                use rayon::prelude::*;
                (0..16).into_par_iter().flat_map(compute_segment).collect()
            } else {
                (0..16).flat_map(compute_segment).collect()
            }
        }
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
