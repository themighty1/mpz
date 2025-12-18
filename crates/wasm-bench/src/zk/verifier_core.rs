//! Benchmarks for mpz-zk-core verifier primitives.
//!
//! Measures raw ZK verifier execute and check performance without protocol overhead.

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

/// Verifier benchmark state, initialized once.
struct VerifierBenchState {
    circuit: Arc<Circuit>,
    and_count: usize,
    delta: Delta,
    input_keys: Vec<Key>,
    gate_keys: Vec<Key>,
    adjustments: Vec<bool>,
}

impl VerifierBenchState {
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

        // Pre-generate adjustments (not timed during benchmark)
        let mut prover = Prover::default();
        let mut prover_exec = prover
            .execute(circuit.clone(), &input_macs, &gate_masks, &gate_macs)
            .unwrap();
        let adjustments: Vec<_> = prover_exec.iter().collect();
        let _ = prover_exec.finish().unwrap();

        Self {
            circuit,
            and_count,
            delta,
            input_keys,
            gate_keys,
            adjustments,
        }
    }
}

thread_local! {
    static STATE: VerifierBenchState = VerifierBenchState::new(AES128.clone().into());
}

/// Benchmark ZK verifier execution: verify circuit n times.
/// This measures only the verifier's execute phase (consuming adjustments).
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub fn zk_core_verifier_execute(n: u32) -> BenchResult {
    let performance = web_sys::window().unwrap().performance().unwrap();

    STATE.with(|state| {
        let start = performance.now();

        for _ in 0..n {
            let mut verifier = Verifier::new(state.delta);
            let mut verifier_exec = verifier
                .execute(state.circuit.clone(), &state.input_keys, &state.gate_keys)
                .unwrap();

            let mut consumer = verifier_exec.consumer();
            for &adjust in &state.adjustments {
                consumer.next(adjust);
            }

            let _ = verifier_exec.finish().unwrap();
        }

        BenchResult {
            elapsed_ms: performance.now() - start,
            and_gates: n as u64 * state.and_count as u64,
        }
    })
}

// Gate count thresholds for check (matches native bench)
#[cfg(target_arch = "wasm32")]
const CHECK_THRESHOLDS: &[usize] = &[200_000, 400_000, 600_000];

/// Benchmark ZK verifier check phase: run check n times for each threshold.
/// Setup (untimed): execute circuits for both prover and verifier, run prover check.
/// Timed: only the verifier check phase (uses rayon internally).
///
/// Runs on a Web Worker because rayon's Atomics.wait is forbidden on main thread.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_core_verifier_check(n: u32, concurrency: u32) -> BenchResult {
    use std::sync::Mutex;
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let _handle = web_spawn::spawn(move || {
        // Initialize web_spawn spawner inside the worker first
        pollster::block_on(wasm_bindgen_futures::JsFuture::from(web_spawn::start_spawner())).ok();

        // Then initialize rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(concurrency as usize)
            .build_global()
            .ok();

        let bench_result = {
            let global = js_sys::global();
            let performance: web_sys::Performance =
                js_sys::Reflect::get(&global, &"performance".into())
                    .expect("performance should exist")
                    .unchecked_into();

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
                    RCOTSenderOutput { mut keys, .. },
                    RCOTReceiverOutput {
                        msgs: mut macs,
                        choices,
                        ..
                    },
                ) = rcot.transfer(total_inputs).unwrap();
                keys.iter_mut().for_each(|key| key.set_lsb(false));
                macs.iter_mut()
                    .zip(&choices)
                    .for_each(|(mac, &choice)| mac.set_lsb(choice));
                let input_keys = Key::from_blocks(keys);
                let input_macs = Mac::from_blocks(macs);

                // Gate correlations
                let total_and_gates = and_count * circuit_count;
                rcot.alloc(total_and_gates);
                rcot.flush().unwrap();
                let (
                    RCOTSenderOutput { keys, .. },
                    RCOTReceiverOutput {
                        choices: gate_masks,
                        msgs: macs,
                        ..
                    },
                ) = rcot.transfer(total_and_gates).unwrap();
                let gate_keys = Key::from_blocks(keys);
                let gate_macs = Mac::from_blocks(macs);

                // SVOLE for check phase
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

                for _ in 0..n {
                    // Setup (untimed): run execute for all circuits (both prover and verifier)
                    let mut prover = Prover::default();
                    let mut verifier = Verifier::new(delta);

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
                        let mut verifier_exec = verifier
                            .execute(
                                circuit.clone(),
                                &input_keys[input_start..input_end],
                                &gate_keys[gate_start..gate_end],
                            )
                            .unwrap();

                        let mut consumer = verifier_exec.consumer();
                        for adjust in prover_exec.iter() {
                            consumer.next(adjust);
                        }

                        let _ = prover_exec.finish().unwrap();
                        let _ = verifier_exec.finish().unwrap();
                    }

                    // Run prover check to get UV (untimed, but also uses rayon)
                    let mut prover_transcript = Hasher::default();
                    let uv = prover
                        .check(&mut prover_transcript, &svole_choices, &svole_ev)
                        .unwrap();

                    // Timed: only verifier check phase (uses rayon internally)
                    let mut verifier_transcript = Hasher::default();
                    let check_start = performance.now();
                    verifier
                        .check(&mut verifier_transcript, &svole_keys, uv)
                        .unwrap();
                    total_check_time += performance.now() - check_start;
                }

                total_gates += n as u64 * actual_gates as u64;
            }

            BenchResult {
                elapsed_ms: total_check_time,
                and_gates: total_gates,
            }
        };
        *result_clone.lock().unwrap() = Some(bench_result);
    });

    // Poll for result from main thread
    loop {
        JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
            .await
            .unwrap();
        if let Some(r) = result.lock().unwrap().take() {
            return r;
        }
        // Small delay before next poll
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }
}
