//! Benchmarks for mpz-garble semihonest protocol.
//!
//! These benchmarks measure the full 2PC protocol performance
//! including garbling, evaluation, and communication.

use wasm_bindgen::prelude::*;

use mpz_circuits::AES128;
use mpz_common::context::test_st_context;
#[cfg(target_arch = "wasm32")]
use mpz_common::context::test_mt_context_with_concurrency;
use mpz_garble::protocol::semihonest::{Evaluator, Garbler};
use mpz_memory_core::{Array, binary::*, correlated::Delta};
use mpz_ot::ideal::cot::ideal_cot;
use mpz_vm_core::{Call, prelude::*};
use rand::{SeedableRng, rngs::StdRng};

/// Benchmark semihonest AES: run full 2PC protocol n times.
/// Returns a checksum to prevent optimization.
#[wasm_bindgen]
pub async fn garble_semihonest_aes(n: u32) -> u32 {
    let mut checksum = 0u32;

    for _ in 0..n {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let (mut ctx_a, mut ctx_b) = test_st_context(8);
        let (cot_send, cot_recv) = ideal_cot(delta.into_inner());

        let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
        let mut ev = Evaluator::new(cot_recv);

        let (gen_out, ev_out) = futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();
                let msg: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.mark_blind(msg).unwrap();

                let ciphertext: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let ciphertext = gb.decode(ciphertext).unwrap();

                gb.assign(key, [0u8; 16]).unwrap();
                gb.commit(key).unwrap();
                gb.commit(msg).unwrap();

                gb.flush(&mut ctx_a).await.unwrap();
                gb.execute(&mut ctx_a).await.unwrap();
                gb.flush(&mut ctx_a).await.unwrap();

                ciphertext.await.unwrap()
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                let msg: Array<U8, 16> = ev.alloc().unwrap();

                ev.mark_blind(key).unwrap();
                ev.mark_private(msg).unwrap();

                let ciphertext: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let ciphertext = ev.decode(ciphertext).unwrap();

                ev.assign(msg, [42u8; 16]).unwrap();
                ev.commit(key).unwrap();
                ev.commit(msg).unwrap();

                ev.flush(&mut ctx_b).await.unwrap();
                ev.execute(&mut ctx_b).await.unwrap();
                ev.flush(&mut ctx_b).await.unwrap();

                ciphertext.await.unwrap()
            }
        );

        checksum = checksum.wrapping_add(gen_out.len() as u32);
        checksum = checksum.wrapping_add(ev_out.len() as u32);
    }

    checksum
}

/// Benchmark result containing timing and work done.
#[wasm_bindgen(getter_with_clone)]
pub struct BenchResult {
    pub elapsed_ms: f64,
    pub and_gates: u64,
}

/// Benchmark semihonest AES batched (256 circuits) with single-threaded context.
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub async fn garble_semihonest_aes_st_batched(n: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 256;
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

/// Benchmark semihonest AES batched (256 circuits) with multi-threaded context.
/// Uses web_spawn for WASM threading support.
/// Returns elapsed time and AND gates processed.
///
/// The `concurrency` parameter controls the maximum number of worker threads
/// used for parallel garbling.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn garble_semihonest_aes_batched(n: u32, concurrency: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 256;
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

/// Benchmark semihonest AES with multi-threaded context.
/// Uses web_spawn for WASM threading support.
/// Returns elapsed time and AND gates processed.
///
/// The `concurrency` parameter controls the maximum number of worker threads
/// used for parallel garbling.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn garble_semihonest_aes_mt(n: u32, concurrency: u32) -> BenchResult {
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
        let mut ctx_a = exec_gb.new_context().await.unwrap();
        let mut ctx_b = exec_ev.new_context().await.unwrap();

        let (cot_send, cot_recv) = ideal_cot(delta.into_inner());

        let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
        let mut ev = Evaluator::new(cot_recv);

        // Timed section: only garble/evaluate
        let start = performance.now();

        let (_gen_out, _ev_out) = futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();
                let msg: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.mark_blind(msg).unwrap();

                let ciphertext: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let ciphertext = gb.decode(ciphertext).unwrap();

                gb.assign(key, [0u8; 16]).unwrap();
                gb.commit(key).unwrap();
                gb.commit(msg).unwrap();

                gb.flush(&mut ctx_a).await.unwrap();
                gb.execute(&mut ctx_a).await.unwrap();
                gb.flush(&mut ctx_a).await.unwrap();

                ciphertext.await.unwrap()
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                let msg: Array<U8, 16> = ev.alloc().unwrap();

                ev.mark_blind(key).unwrap();
                ev.mark_private(msg).unwrap();

                let ciphertext: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let ciphertext = ev.decode(ciphertext).unwrap();

                ev.assign(msg, [42u8; 16]).unwrap();
                ev.commit(key).unwrap();
                ev.commit(msg).unwrap();

                ev.flush(&mut ctx_b).await.unwrap();
                ev.execute(&mut ctx_b).await.unwrap();
                ev.flush(&mut ctx_b).await.unwrap();

                ciphertext.await.unwrap()
            }
        );

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * and_gates_per_circuit,
    }
}
