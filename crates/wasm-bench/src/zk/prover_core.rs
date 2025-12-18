//! Benchmarks for mpz-zk-core prover primitives.
//!
//! Measures raw ZK prover execute and check performance without protocol overhead.

use wasm_bindgen::prelude::*;

use blake3::Hasher;
use std::sync::Arc;
use mpz_circuits::{Circuit, AES128};
use mpz_memory_core::correlated::{Delta, Mac};
use mpz_ot_core::{
    ideal::rcot::IdealRCOT,
    rcot::{RCOTReceiverOutput, RCOTSenderOutput},
};
use mpz_zk_core::Prover;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::BenchResult;

/// Prover benchmark state, initialized once.
struct ProverBenchState {
    circuit: Arc<Circuit>,
    and_count: usize,
    input_macs: Vec<Mac>,
    gate_macs: Vec<Mac>,
    gate_masks: Vec<bool>,
    svole_choices: Vec<bool>,
    svole_ev: Vec<mpz_core::Block>,
}

impl ProverBenchState {
    fn new(circuit: Arc<Circuit>) -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        let and_count = circuit.and_count();

        // Allocate and transfer input correlations
        rcot.alloc(circuit.inputs().len());
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                msgs: mut macs,
                choices,
                ..
            },
        ) = rcot.transfer(circuit.inputs().len()).unwrap();

        // Set LSB for macs
        macs.iter_mut()
            .zip(&choices)
            .for_each(|(mac, &choice)| mac.set_lsb(choice));

        let input_macs = Mac::from_blocks(macs);

        // Allocate and transfer gate correlations
        rcot.alloc(and_count);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                choices: gate_masks,
                msgs: macs,
                ..
            },
        ) = rcot.transfer(and_count).unwrap();

        let gate_macs = Mac::from_blocks(macs);

        // SVOLE correlations for check phase
        rcot.alloc(128);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                choices: svole_choices,
                msgs: svole_ev,
                ..
            },
        ) = rcot.transfer(128).unwrap();

        Self {
            circuit,
            and_count,
            input_macs,
            gate_macs,
            gate_masks,
            svole_choices,
            svole_ev,
        }
    }
}

thread_local! {
    static STATE: ProverBenchState = ProverBenchState::new(AES128.clone().into());
}

/// Benchmark ZK prover execution: prove circuit n times.
/// This measures only the prover's execute phase (generating adjustments).
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub fn zk_core_prover_execute(n: u32) -> BenchResult {
    let performance = web_sys::window().unwrap().performance().unwrap();

    STATE.with(|state| {
        let start = performance.now();

        for _ in 0..n {
            let mut prover = Prover::default();
            let mut prover_exec = prover
                .execute(
                    state.circuit.clone(),
                    &state.input_macs,
                    &state.gate_masks,
                    &state.gate_macs,
                )
                .unwrap();

            // Consume all adjustments
            for _ in prover_exec.iter() {}
            let _ = prover_exec.finish().unwrap();
        }

        BenchResult {
            elapsed_ms: performance.now() - start,
            and_gates: n as u64 * state.and_count as u64,
        }
    })
}

// Gate count thresholds for check (matches native bench)
const CHECK_THRESHOLDS: &[usize] = &[200_000, 400_000, 600_000];

/// Benchmark ZK prover check phase: run check n times for each threshold.
/// Setup (untimed): execute circuits to reach threshold gates.
/// Timed: only the prover check phase.
/// Returns elapsed time and AND gates processed across all thresholds.
#[wasm_bindgen]
pub fn zk_core_prover_check(n: u32) -> BenchResult {
    let performance = web_sys::window().unwrap().performance().unwrap();
    let circuit: Arc<Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    let mut total_check_time = 0.0;
    let mut total_gates = 0u64;

    for &threshold in CHECK_THRESHOLDS {
        let circuit_count = threshold.div_ceil(and_count);
        let actual_gates = circuit_count * and_count;

        // Setup correlations for this threshold
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        // Input correlations
        let total_inputs = inputs_per_circuit * circuit_count;
        rcot.alloc(total_inputs);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                msgs: mut macs,
                choices,
                ..
            },
        ) = rcot.transfer(total_inputs).unwrap();
        macs.iter_mut()
            .zip(&choices)
            .for_each(|(mac, &choice)| mac.set_lsb(choice));
        let input_macs = Mac::from_blocks(macs);

        // Gate correlations
        let total_and_gates = and_count * circuit_count;
        rcot.alloc(total_and_gates);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                choices: gate_masks,
                msgs: macs,
                ..
            },
        ) = rcot.transfer(total_and_gates).unwrap();
        let gate_macs = Mac::from_blocks(macs);

        // SVOLE for check phase
        rcot.alloc(128);
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { .. },
            RCOTReceiverOutput {
                choices: svole_choices,
                msgs: svole_ev,
                ..
            },
        ) = rcot.transfer(128).unwrap();

        for _ in 0..n {
            // Setup (untimed): run execute for all circuits
            let mut prover = Prover::default();

            for i in 0..circuit_count {
                let input_start = i * inputs_per_circuit;
                let input_end = input_start + inputs_per_circuit;
                let gate_start = i * and_count;
                let gate_end = gate_start + and_count;

                let mut prover_exec = prover
                    .execute(
                        circuit.clone(),
                        &input_macs[input_start..input_end],
                        &gate_masks[gate_start..gate_end],
                        &gate_macs[gate_start..gate_end],
                    )
                    .unwrap();

                // Consume adjustments (not timed)
                for _ in prover_exec.iter() {}
                let _ = prover_exec.finish().unwrap();
            }

            // Timed: only check phase
            let mut prover_transcript = Hasher::default();
            let check_start = performance.now();
            let _uv = prover
                .check(&mut prover_transcript, &svole_choices, &svole_ev)
                .unwrap();
            total_check_time += performance.now() - check_start;
        }

        total_gates += n as u64 * actual_gates as u64;
    }

    BenchResult {
        elapsed_ms: total_check_time,
        and_gates: total_gates,
    }
}
