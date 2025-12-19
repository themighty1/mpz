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

    #[cfg(not(feature = "rayon"))]
    fn compute_chis(&self, mut chi: Block) -> Vec<Block> {
        let mut chis = Vec::with_capacity(self.triples.len());
        chis.push(chi);
        for _ in 1..self.triples.len() {
            chi = chi.gfmul(chi);
            chis.push(chi);
        }
        chis
    }

    #[cfg(feature = "rayon")]
    fn compute_chis(&self, chi: Block) -> Vec<Block> {
        use rayon::prelude::*;

        const NUM_LANES: usize = 8;

        let n = self.triples.len();

        // Compute 8 starting points: chi^1, chi^2, chi^4, ..., chi^128
        let mut starts = [Block::ZERO; NUM_LANES];
        starts[0] = chi;
        for i in 1..NUM_LANES {
            starts[i] = starts[i - 1].gfmul(starts[i - 1]);
        }

        // Compute stride multiplier: chi^256 = (chi^128)^2
        let stride = starts[NUM_LANES - 1].gfmul(starts[NUM_LANES - 1]);

        // Each lane computes ceil(n / NUM_LANES) values
        let per_lane = n.div_ceil(NUM_LANES);

        // Parallel: each lane generates its sequence
        let lane_results: Vec<Vec<Block>> = (0..NUM_LANES)
            .into_par_iter()
            .map(|lane| {
                let mut result = Vec::with_capacity(per_lane);
                let mut current = starts[lane];
                for i in 0..per_lane {
                    let pos = lane + i * NUM_LANES;
                    if pos >= n {
                        break;
                    }
                    result.push(current);
                    current = current.gfmul(stride);
                }
                result
            })
            .collect();

        // Interleave results: position i comes from lane (i % NUM_LANES), index (i / NUM_LANES)
        let mut chis = vec![Block::ZERO; n];
        for (lane, lane_chis) in lane_results.into_iter().enumerate() {
            for (idx, chi_val) in lane_chis.into_iter().enumerate() {
                let pos = lane + idx * NUM_LANES;
                chis[pos] = chi_val;
            }
        }

        chis
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
