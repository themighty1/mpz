//! Benchmarks for mpz-zk-core (QuickSilver ZK protocol).
//!
//! These benchmarks measure the raw ZK proving/verification performance
//! for the QuickSilver protocol without protocol overhead.

use wasm_bindgen::prelude::*;

use blake3::Hasher;
use std::sync::Arc;
use mpz_circuits::{Circuit, AES128};
use mpz_memory_core::correlated::{Delta, Key, Mac};
use mpz_ot_core::{
    ideal::rcot::IdealRCOT,
    rcot::{RCOTReceiverOutput, RCOTSenderOutput},
};
use mpz_zk_core::{Prover, Verifier};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::BenchResult;

/// Shared benchmark state for zk-core, initialized once per circuit.
struct ZkBenchState {
    circuit: Arc<Circuit>,
    and_count: usize,
    delta: Delta,
    input_keys: Vec<Key>,
    input_macs: Vec<Mac>,
    gate_keys: Vec<Key>,
    gate_macs: Vec<Mac>,
    gate_masks: Vec<bool>,
    svole_keys: Vec<mpz_core::Block>,
    svole_choices: Vec<bool>,
    svole_ev: Vec<mpz_core::Block>,
}

impl ZkBenchState {
    fn new(circuit: Arc<Circuit>) -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        let and_count = circuit.and_count();

        // Allocate and transfer input correlations
        rcot.alloc(circuit.inputs().len());
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { mut keys, .. },
            RCOTReceiverOutput {
                msgs: mut macs,
                choices,
                ..
            },
        ) = rcot.transfer(circuit.inputs().len()).unwrap();

        // Set LSB for keys and macs
        keys.iter_mut().for_each(|key| key.set_lsb(false));
        macs.iter_mut()
            .zip(&choices)
            .for_each(|(mac, &choice)| mac.set_lsb(choice));

        let input_keys = Key::from_blocks(keys);
        let input_macs = Mac::from_blocks(macs);

        // Allocate and transfer gate correlations
        rcot.alloc(and_count);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { keys, .. },
            RCOTReceiverOutput {
                choices: gate_masks,
                msgs: macs,
                ..
            },
        ) = rcot.transfer(and_count).unwrap();

        let gate_keys = Key::from_blocks(keys);
        let gate_macs = Mac::from_blocks(macs);

        // Allocate and transfer SVOLE correlations for check phase
        rcot.alloc(128);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput {
                keys: svole_keys, ..
            },
            RCOTReceiverOutput {
                choices: svole_choices,
                msgs: svole_ev,
                ..
            },
        ) = rcot.transfer(128).unwrap();

        Self {
            circuit,
            and_count,
            delta,
            input_keys,
            input_macs,
            gate_keys,
            gate_macs,
            gate_masks,
            svole_keys,
            svole_choices,
            svole_ev,
        }
    }
}

// Thread-local state for benchmarks (WASM is single-threaded)
thread_local! {
    static ZK_STATE: ZkBenchState = ZkBenchState::new(AES128.clone().into());
}

/// Benchmark full ZK protocol: prove and verify circuit n times.
/// This measures the complete execute + check phases for both prover and verifier.
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub fn zk_core_full_protocol(n: u32) -> BenchResult {
    let performance = web_sys::window().unwrap().performance().unwrap();

    ZK_STATE.with(|state| {
        let start = performance.now();

        for _ in 0..n {
            let mut prover = Prover::default();
            let mut verifier = Verifier::new(state.delta);
            let mut prover_transcript = Hasher::default();
            let mut verifier_transcript = Hasher::default();

            // Execute phase
            let mut prover_exec = prover
                .execute(
                    state.circuit.clone(),
                    &state.input_macs,
                    &state.gate_masks,
                    &state.gate_macs,
                )
                .unwrap();
            let mut verifier_exec = verifier
                .execute(state.circuit.clone(), &state.input_keys, &state.gate_keys)
                .unwrap();

            // Transfer adjustments from prover to verifier
            let mut verifier_consumer = verifier_exec.consumer();
            for adjust in prover_exec.iter() {
                verifier_consumer.next(adjust);
            }

            let _ = prover_exec.finish().unwrap();
            let _ = verifier_exec.finish().unwrap();

            // Check phase
            let uv = prover
                .check(&mut prover_transcript, &state.svole_choices, &state.svole_ev)
                .unwrap();
            verifier
                .check(&mut verifier_transcript, &state.svole_keys, uv)
                .unwrap();
        }

        BenchResult {
            elapsed_ms: performance.now() - start,
            and_gates: n as u64 * state.and_count as u64,
        }
    })
}

/// Benchmark ZK check phase only: run check n times.
/// This measures the SVOLE-based consistency check.
/// Each iteration runs execute (not timed) then check (timed).
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub fn zk_core_check_only(n: u32) -> BenchResult {
    let performance = web_sys::window().unwrap().performance().unwrap();

    ZK_STATE.with(|state| {
        let mut total_check_time = 0.0;

        for _ in 0..n {
            // Execute phase (not timed) - required before check
            let mut prover = Prover::default();
            let mut verifier = Verifier::new(state.delta);

            let mut prover_exec = prover
                .execute(
                    state.circuit.clone(),
                    &state.input_macs,
                    &state.gate_masks,
                    &state.gate_macs,
                )
                .unwrap();
            let mut verifier_exec = verifier
                .execute(state.circuit.clone(), &state.input_keys, &state.gate_keys)
                .unwrap();

            let mut verifier_consumer = verifier_exec.consumer();
            for adjust in prover_exec.iter() {
                verifier_consumer.next(adjust);
            }
            let _ = prover_exec.finish().unwrap();
            let _ = verifier_exec.finish().unwrap();

            // Check phase (timed)
            let mut prover_transcript = Hasher::default();
            let mut verifier_transcript = Hasher::default();

            let check_start = performance.now();
            let uv = prover
                .check(&mut prover_transcript, &state.svole_choices, &state.svole_ev)
                .unwrap();
            verifier
                .check(&mut verifier_transcript, &state.svole_keys, uv)
                .unwrap();
            total_check_time += performance.now() - check_start;
        }

        BenchResult {
            elapsed_ms: total_check_time,
            and_gates: n as u64 * state.and_count as u64,
        }
    })
}
