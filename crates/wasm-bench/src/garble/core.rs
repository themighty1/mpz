//! Benchmarks for mpz-garble-core primitives.
//!
//! These benchmarks measure the raw garbling/evaluation performance
//! of half-gates and three-halves schemes without protocol overhead.

use wasm_bindgen::prelude::*;

use mpz_circuits::AES128;
use mpz_core::Block;
use mpz_garble_core::{half_gates, three_halves, Key};
use mpz_memory_core::correlated::Delta;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Shared benchmark state, initialized once.
struct BenchState {
    delta: Delta,
    hg_inputs: Vec<Key>,
    th_inputs: Vec<Key>,
    hg_eval_inputs: Vec<mpz_memory_core::correlated::Mac>,
    th_eval_inputs: Vec<mpz_memory_core::correlated::Mac>,
    hg_gates: Vec<mpz_garble_core::EncryptedGate>,
    th_gates: Vec<three_halves::EncryptedGate>,
}

impl BenchState {
    fn new() -> Self {
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
}

// Thread-local state for benchmarks (WASM is single-threaded)
thread_local! {
    static STATE: BenchState = BenchState::new();
}

/// Returns the number of AND gates in the AES-128 circuit.
#[wasm_bindgen]
pub fn garble_core_aes128_and_count() -> u32 {
    AES128.and_count() as u32
}

/// Benchmark half-gates garbling: garble AES circuit n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub fn garble_core_half_gates_garble(n: u32) -> u32 {
    STATE.with(|state| {
        let mut gb = half_gates::Garbler::default();
        let mut checksum = 0u32;

        for _ in 0..n {
            let mut iter = gb.generate(&AES128, state.delta, &state.hg_inputs).unwrap();
            let gates: Vec<_> = iter.by_ref().collect();
            let _ = iter.finish().unwrap();
            checksum = checksum.wrapping_add(gates.len() as u32);
        }

        checksum
    })
}

/// Benchmark three-halves garbling: garble AES circuit n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub fn garble_core_three_halves_garble(n: u32) -> u32 {
    STATE.with(|state| {
        let mut gb = three_halves::Garbler::default();
        let mut checksum = 0u32;

        for _ in 0..n {
            let mut bench_rng = StdRng::seed_from_u64(42);
            let mut iter = gb
                .generate(&AES128, state.delta, &state.th_inputs, &mut bench_rng)
                .unwrap();
            let gates: Vec<_> = iter.by_ref().collect();
            let _ = iter.finish().unwrap();
            checksum = checksum.wrapping_add(gates.len() as u32);
        }

        checksum
    })
}

/// Benchmark half-gates evaluation: evaluate AES circuit n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub fn garble_core_half_gates_evaluate(n: u32) -> u32 {
    STATE.with(|state| {
        let mut ev = half_gates::Evaluator::default();
        let mut checksum = 0u32;

        for _ in 0..n {
            let mut consumer = ev.evaluate(&AES128, &state.hg_eval_inputs).unwrap();
            for gate in &state.hg_gates {
                consumer.next(*gate);
            }
            let output = consumer.finish().unwrap();
            checksum = checksum.wrapping_add(output.outputs.len() as u32);
        }

        checksum
    })
}

/// Benchmark three-halves evaluation: evaluate AES circuit n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub fn garble_core_three_halves_evaluate(n: u32) -> u32 {
    STATE.with(|state| {
        let mut ev = three_halves::Evaluator::default();
        let mut checksum = 0u32;

        for _ in 0..n {
            let mut consumer = ev.evaluate(&AES128, &state.th_eval_inputs).unwrap();
            for gate in &state.th_gates {
                consumer.next(gate.clone());
            }
            let output = consumer.finish().unwrap();
            checksum = checksum.wrapping_add(output.outputs.len() as u32);
        }

        checksum
    })
}
