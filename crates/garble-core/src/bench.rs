//! Benchmark support utilities.
//!
//! This module provides shared setup code for benchmarks across native and WASM targets.
//! Enable with the `bench-support` feature.

use mpz_circuits::AES128;
use mpz_core::Block;
use mpz_memory_core::correlated::Delta;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{half_gates, three_halves, Key};

/// Pre-computed benchmark state for AES-128 circuit benchmarks.
///
/// This struct contains all the setup data needed to run garble/evaluate benchmarks
/// without re-initializing for each iteration.
pub struct BenchState {
    /// Global correlation for garbling.
    pub delta: Delta,
    /// Input keys for half-gates garbler.
    pub hg_inputs: Vec<Key>,
    /// Input keys for three-halves garbler (LSB = 0).
    pub th_inputs: Vec<Key>,
    /// Input MACs for half-gates evaluator.
    pub hg_eval_inputs: Vec<mpz_memory_core::correlated::Mac>,
    /// Input MACs for three-halves evaluator.
    pub th_eval_inputs: Vec<mpz_memory_core::correlated::Mac>,
    /// Pre-generated half-gates encrypted gates for evaluation benchmarks.
    pub hg_gates: Vec<crate::EncryptedGate>,
    /// Pre-generated three-halves encrypted gates for evaluation benchmarks.
    pub th_gates: Vec<three_halves::EncryptedGate>,
}

impl Default for BenchState {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchState {
    /// Creates a new benchmark state with pre-computed inputs and gates.
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        // Half-gates inputs
        let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();

        // Three-halves inputs (LSB = 0)
        let th_inputs: Vec<Key> = (0..256)
            .map(|_| {
                let mut block: Block = rng.random();
                block.set_lsb(false);
                block.into()
            })
            .collect();

        // Choices for evaluation
        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();

        // Half-gates eval inputs
        let hg_eval_inputs: Vec<_> = hg_inputs
            .iter()
            .zip(&choices)
            .map(|(k, &c)| k.auth(c, &delta))
            .collect();

        // Generate half-gates garbled circuit for evaluation benchmarks
        let mut hg_gb = half_gates::Garbler::default();
        let mut hg_iter = hg_gb.generate(&AES128, delta, &hg_inputs).unwrap();
        let hg_gates: Vec<_> = hg_iter.by_ref().collect();
        let _ = hg_iter.finish().unwrap();

        // Generate three-halves garbled circuit and get input pairs
        let mut th_gb = three_halves::Garbler::default();
        let mut th_rng = StdRng::seed_from_u64(42);
        let mut th_iter = th_gb
            .generate(&AES128, delta, &th_inputs, &mut th_rng)
            .unwrap();
        let th_gates: Vec<_> = th_iter.by_ref().collect();
        let three_halves::GarblerOutput {
            inputs: input_pairs,
            ..
        } = th_iter.finish().unwrap();

        let th_eval_inputs: Vec<_> = input_pairs
            .iter()
            .zip(&choices)
            .map(|((f, t), &c)| if c { *t } else { *f })
            .collect();

        Self {
            delta,
            hg_inputs,
            th_inputs,
            hg_eval_inputs,
            th_eval_inputs,
            hg_gates,
            th_gates,
        }
    }

    /// Returns the number of AND gates in the AES-128 circuit.
    pub fn and_count() -> usize {
        AES128.and_count()
    }
}
