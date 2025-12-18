//! Benchmarks for mpz-garble-core evaluator primitives.
//!
//! Measures raw half-gates evaluation performance without protocol overhead.

use wasm_bindgen::prelude::*;

use mpz_circuits::AES128;
use mpz_garble_core::{half_gates, EncryptedGate, Key};
use mpz_memory_core::correlated::{Delta, Mac};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Shared benchmark state, initialized once.
struct BenchState {
    eval_inputs: Vec<Mac>,
    gates: Vec<EncryptedGate>,
}

impl BenchState {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();
        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();

        let eval_inputs: Vec<_> = inputs
            .iter()
            .zip(&choices)
            .map(|(k, &c)| k.auth(c, &delta))
            .collect();

        // Pre-garble circuit for evaluation benchmarks
        let mut gb = half_gates::Garbler::default();
        let mut iter = gb.generate(&AES128, delta, &inputs).unwrap();
        let gates: Vec<_> = iter.by_ref().collect();
        let _ = iter.finish().unwrap();

        Self { eval_inputs, gates }
    }
}

thread_local! {
    static STATE: BenchState = BenchState::new();
}

/// Benchmark half-gates evaluation: evaluate AES circuit n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub fn garble_core_half_gates_evaluate(n: u32) -> u32 {
    STATE.with(|state| {
        let mut ev = half_gates::Evaluator::default();
        let mut checksum = 0u32;

        for _ in 0..n {
            let mut consumer = ev.evaluate(&AES128, &state.eval_inputs).unwrap();
            for gate in &state.gates {
                consumer.next(*gate);
            }
            let output = consumer.finish().unwrap();
            checksum = checksum.wrapping_add(output.outputs.len() as u32);
        }

        checksum
    })
}
