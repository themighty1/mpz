//! Benchmarks for mpz-garble semihonest protocol.
//!
//! These benchmarks measure the full 2PC protocol performance
//! including garbling, evaluation, and communication.

use wasm_bindgen::prelude::*;

use mpz_circuits::AES128;
use mpz_common::context::test_st_context;
#[cfg(target_arch = "wasm32")]
use mpz_common::context::{
    recording_mt_context_with_spawn_and_limit, replay_mt_context_with_spawn_and_limit,
    test_mt_context_with_concurrency, Multithread, RecordedMtData,
};
use mpz_garble::protocol::semihonest::{Evaluator, Garbler};
use mpz_memory_core::{Array, binary::*, correlated::Delta};
use mpz_ot::ideal::cot::ideal_cot;
#[cfg(target_arch = "wasm32")]
use mpz_ot::ideal::msg_cot::{MsgIdealCOTSender, MsgIdealCOTReceiver, msg_ideal_cot};
use mpz_vm_core::{Call, prelude::*};
use rand::{SeedableRng, rngs::StdRng};

/// Benchmark result containing timing and work done.
#[wasm_bindgen(getter_with_clone)]
pub struct BenchResult {
    pub elapsed_ms: f64,
    pub and_gates: u64,
}

/// Benchmark semihonest 2PC garbling (1000 AES circuits) with single-threaded context.
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub async fn garble_st(n: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 1000;
    let and_gates_per_circuit = AES128.and_count() as u64;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        // Setup (not timed)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let (mut ctx_gb, mut ctx_ev) = test_st_context(8);
        let (cot_send, cot_recv) = ideal_cot(delta.into_inner());

        let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
        let mut ev = Evaluator::new(cot_recv);

        // Timed section: batch 256 AES circuits
        let start = performance.now();

        futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.assign(key, [0u8; 16]).unwrap();
                gb.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = gb.alloc().unwrap();
                    gb.mark_blind(msg).unwrap();
                    gb.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = gb
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(gb.decode(ciphertext).unwrap());
                }

                gb.flush(&mut ctx_gb).await.unwrap();
                gb.execute(&mut ctx_gb).await.unwrap();
                gb.flush(&mut ctx_gb).await.unwrap();
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                ev.mark_blind(key).unwrap();
                ev.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = ev.alloc().unwrap();
                    ev.mark_private(msg).unwrap();
                    ev.assign(msg, [42u8; 16]).unwrap();
                    ev.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = ev
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(ev.decode(ciphertext).unwrap());
                }

                ev.flush(&mut ctx_ev).await.unwrap();
                ev.execute(&mut ctx_ev).await.unwrap();
                ev.flush(&mut ctx_ev).await.unwrap();
            }
        );

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BATCH_SIZE as u64 * and_gates_per_circuit,
    }
}

/// Benchmark semihonest 2PC garbling (1000 AES circuits) with multi-threaded context.
/// Uses web_spawn for WASM threading support.
/// Returns elapsed time and AND gates processed.
///
/// The `concurrency` parameter controls the maximum parallelism level.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn garble_mt(n: u32, concurrency: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 1000;
    let and_gates_per_circuit = AES128.and_count() as u64;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        // Setup (not timed)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let (mut exec_gb, mut exec_ev) =
            test_mt_context_with_concurrency(8, concurrency as usize, |f| {
                let _ = web_spawn::spawn(f);
                Ok(())
            });
        let mut ctx_gb = exec_gb.new_context().await.unwrap();
        let mut ctx_ev = exec_ev.new_context().await.unwrap();

        let (cot_send, cot_recv) = msg_ideal_cot(delta.into_inner());

        let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
        let mut ev = Evaluator::new(cot_recv);

        // Timed section: batch 256 AES circuits
        let start = performance.now();

        futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.assign(key, [0u8; 16]).unwrap();
                gb.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = gb.alloc().unwrap();
                    gb.mark_blind(msg).unwrap();
                    gb.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = gb
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(gb.decode(ciphertext).unwrap());
                }

                gb.flush(&mut ctx_gb).await.unwrap();
                gb.execute(&mut ctx_gb).await.unwrap();
                gb.flush(&mut ctx_gb).await.unwrap();
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                ev.mark_blind(key).unwrap();
                ev.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = ev.alloc().unwrap();
                    ev.mark_private(msg).unwrap();
                    ev.assign(msg, [42u8; 16]).unwrap();
                    ev.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = ev
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(ev.decode(ciphertext).unwrap());
                }

                ev.flush(&mut ctx_ev).await.unwrap();
                ev.execute(&mut ctx_ev).await.unwrap();
                ev.flush(&mut ctx_ev).await.unwrap();
            }
        );

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BATCH_SIZE as u64 * and_gates_per_circuit,
    }
}

// ============================================================================
// Isolated garbler benchmark (MT replay infrastructure)
// ============================================================================

#[cfg(target_arch = "wasm32")]
const BATCH_SIZE_ISOLATED: u32 = 1000;

/// Yields to the browser event loop, allowing console logs to flush and UI to update.
#[cfg(target_arch = "wasm32")]
async fn yield_to_browser() {
    use wasm_bindgen_futures::JsFuture;
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    let _ = JsFuture::from(promise).await;
}

/// Calculate max frame length based on workload size.
#[cfg(target_arch = "wasm32")]
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    // Garble protocol sends encrypted gates (2 blocks per AND gate) plus labels
    let bytes_per_gate = 32; // 2 x 16-byte blocks
    let overhead = 1.5; // serialization overhead
    let and_gates = circuit.and_count() * circuit_count;
    ((and_gates * bytes_per_gate) as f64 * overhead) as usize
}

/// Runs the full garble protocol with MT contexts.
/// Records evaluator->garbler messages.
#[cfg(target_arch = "wasm32")]
async fn run_protocol_record_evaluator_mt(
    exec_gb: &mut Multithread,
    exec_ev: &mut Multithread,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    // Use msg-based COT for proper message recording
    let (cot_send, cot_recv) = msg_ideal_cot(delta.into_inner());

    let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
    let mut ev = Evaluator::new(cot_recv);

    let mut ctx_gb = exec_gb.new_context().await.unwrap();
    let mut ctx_ev = exec_ev.new_context().await.unwrap();

    futures::join!(
        async {
            let key: Array<U8, 16> = gb.alloc().unwrap();

            gb.mark_private(key).unwrap();
            gb.assign(key, [0u8; 16]).unwrap();
            gb.commit(key).unwrap();

            for _ in 0..BATCH_SIZE_ISOLATED {
                let msg: Array<U8, 16> = gb.alloc().unwrap();
                gb.mark_blind(msg).unwrap();
                gb.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(gb.decode(ciphertext).unwrap());
            }

            gb.flush(&mut ctx_gb).await.unwrap();
            gb.execute(&mut ctx_gb).await.unwrap();
            gb.flush(&mut ctx_gb).await.unwrap();
        },
        async {
            let key: Array<U8, 16> = ev.alloc().unwrap();
            ev.mark_blind(key).unwrap();
            ev.commit(key).unwrap();

            for _ in 0..BATCH_SIZE_ISOLATED {
                let msg: Array<U8, 16> = ev.alloc().unwrap();
                ev.mark_private(msg).unwrap();
                ev.assign(msg, [42u8; 16]).unwrap();
                ev.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(ev.decode(ciphertext).unwrap());
            }

            ev.flush(&mut ctx_ev).await.unwrap();
            ev.execute(&mut ctx_ev).await.unwrap();
            ev.flush(&mut ctx_ev).await.unwrap();
        }
    );
}

/// Records evaluator->garbler messages for garbler replay.
#[cfg(target_arch = "wasm32")]
async fn record_for_garbler_mt(seed: u64, concurrency: usize) -> RecordedMtData {
    let (mut exec_gb, mut exec_ev, recorded) = recording_mt_context_with_spawn_and_limit(
        1024 * 1024,
        max_frame_length(&AES128, BATCH_SIZE_ISOLATED as usize),
        concurrency,
        |f| {
            let _ = web_spawn::spawn(f);
            Ok(())
        },
    );
    run_protocol_record_evaluator_mt(&mut exec_gb, &mut exec_ev, seed).await;
    recorded.lock().unwrap().clone()
}

/// Runs garbler only with MT replay context.
#[cfg(target_arch = "wasm32")]
async fn run_garbler_with_replay_mt(exec: &mut Multithread, delta: Delta) {
    // Use msg-based COT sender for replay
    let cot_send = MsgIdealCOTSender::new(delta.into_inner());
    let mut gb = Garbler::new(cot_send, [0u8; 16], delta);

    let mut ctx = exec.new_context().await.unwrap();

    let key: Array<U8, 16> = gb.alloc().unwrap();

    gb.mark_private(key).unwrap();
    gb.assign(key, [0u8; 16]).unwrap();
    gb.commit(key).unwrap();

    for _ in 0..BATCH_SIZE_ISOLATED {
        let msg: Array<U8, 16> = gb.alloc().unwrap();
        gb.mark_blind(msg).unwrap();
        gb.commit(msg).unwrap();

        let ciphertext: Array<U8, 16> = gb
            .call(
                Call::builder(AES128.clone())
                    .arg(key)
                    .arg(msg)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        std::mem::drop(gb.decode(ciphertext).unwrap());
    }

    gb.flush(&mut ctx).await.unwrap();
    gb.execute(&mut ctx).await.unwrap();
    gb.flush(&mut ctx).await.unwrap();
}

/// Benchmark isolated garbler with MT context and message replay.
///
/// Records evaluator->garbler messages once during setup using MT contexts,
/// then benchmarks garbler execution in isolation using MT replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `concurrency` - Maximum parallelism level (max children per parent thread)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn garble_isolated_mt(n: u32, concurrency: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;

    let performance = web_sys::window().unwrap().performance().unwrap();

    // Record messages once (not timed)
    web_sys::console::log_1(
        &format!(
            "[rust] Recording garble messages, concurrency={}...",
            concurrency
        )
        .into(),
    );
    yield_to_browser().await;

    let recorded = record_for_garbler_mt(0, concurrency as usize).await;
    let total_bytes: usize = recorded.channels.values().map(|v| v.len()).sum();
    web_sys::console::log_1(
        &format!(
            "[rust] Recorded {} channels, {} total bytes",
            recorded.channels.len(),
            total_bytes
        )
        .into(),
    );
    yield_to_browser().await;

    // Pre-generate delta for replay runs (consistent with recording)
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        if i % 10 == 0 {
            web_sys::console::log_1(&format!("[rust] Garble MT Iteration {}/{}", i, n).into());
            yield_to_browser().await;
        }

        // Timed section: garbler replay with MT context
        let start = performance.now();

        let mut exec = replay_mt_context_with_spawn_and_limit(
            recorded.clone(),
            max_frame_length(&AES128, BATCH_SIZE_ISOLATED as usize),
            concurrency as usize,
            |f| {
                let _ = web_spawn::spawn(f);
                Ok(())
            },
        );
        run_garbler_with_replay_mt(&mut exec, delta).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BATCH_SIZE_ISOLATED as u64 * and_gates_per_circuit,
    }
}
