//! Benchmarks for mpz-zk (QuickSilver ZK protocol with full VM).
//!
//! These benchmarks measure the full ZK protocol performance
//! including proof generation, verification, and communication.

use wasm_bindgen::prelude::*;

use mpz_circuits::AES128;
use mpz_common::context::test_st_context;
#[cfg(target_arch = "wasm32")]
use mpz_common::context::test_mt_context_with_concurrency;
use mpz_memory_core::{Array, binary::*, correlated::Delta};
use mpz_ot::ideal::rcot::ideal_rcot;
use mpz_vm_core::{Call, prelude::*};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::BenchResult;

/// Benchmark ZK protocol batched (256 circuits) with single-threaded context.
/// Returns elapsed time and AND gates processed.
#[wasm_bindgen]
pub async fn zk_st_batched(n: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 256;
    let and_gates_per_circuit = AES128.and_count() as u64;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        // Setup (not timed)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let (mut ctx_p, mut ctx_v) = test_st_context(8);
        let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

        let mut prover = Prover::new(ProverConfig::default(), ot_recv);
        let mut verifier = Verifier::new(VerifierConfig::default(), delta, ot_send);

        // Timed section: batch 256 AES circuits
        let start = performance.now();

        futures::join!(
            async {
                // Prover: key is private (witness), msg is public
                let key: Array<U8, 16> = prover.alloc().unwrap();

                prover.mark_private(key).unwrap();
                prover.assign(key, [0u8; 16]).unwrap();
                prover.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = prover.alloc().unwrap();
                    prover.mark_public(msg).unwrap();
                    prover.assign(msg, [42u8; 16]).unwrap();
                    prover.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = prover
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(prover.decode(ciphertext).unwrap());
                }

                prover.execute_all(&mut ctx_p).await.unwrap();
            },
            async {
                // Verifier: key is blind (prover's witness), msg is public
                let key: Array<U8, 16> = verifier.alloc().unwrap();

                verifier.mark_blind(key).unwrap();
                verifier.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = verifier.alloc().unwrap();
                    verifier.mark_public(msg).unwrap();
                    verifier.assign(msg, [42u8; 16]).unwrap();
                    verifier.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = verifier
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(verifier.decode(ciphertext).unwrap());
                }

                verifier.execute_all(&mut ctx_v).await.unwrap();
            }
        );

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BATCH_SIZE as u64 * and_gates_per_circuit,
    }
}

/// Benchmark ZK protocol batched (256 circuits) with multi-threaded context.
/// Uses web_spawn for WASM threading support.
/// Returns elapsed time and AND gates processed.
///
/// The `concurrency` parameter controls the maximum number of worker threads
/// used for parallel proof generation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_mt_batched(n: u32, concurrency: u32) -> BenchResult {
    const BATCH_SIZE: u32 = 256;
    let and_gates_per_circuit = AES128.and_count() as u64;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        // Setup (not timed)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);

        let (mut exec_p, mut exec_v) =
            test_mt_context_with_concurrency(8, concurrency as usize, |f| {
                let _ = web_spawn::spawn(f);
                Ok(())
            });
        let mut ctx_p = exec_p.new_context().await.unwrap();
        let mut ctx_v = exec_v.new_context().await.unwrap();

        let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

        let mut prover = Prover::new(ProverConfig::default(), ot_recv);
        let mut verifier = Verifier::new(VerifierConfig::default(), delta, ot_send);

        // Timed section: batch 256 AES circuits
        let start = performance.now();

        futures::join!(
            async {
                // Prover: key is private (witness), msg is public
                let key: Array<U8, 16> = prover.alloc().unwrap();

                prover.mark_private(key).unwrap();
                prover.assign(key, [0u8; 16]).unwrap();
                prover.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = prover.alloc().unwrap();
                    prover.mark_public(msg).unwrap();
                    prover.assign(msg, [42u8; 16]).unwrap();
                    prover.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = prover
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(prover.decode(ciphertext).unwrap());
                }

                prover.execute_all(&mut ctx_p).await.unwrap();
            },
            async {
                // Verifier: key is blind (prover's witness), msg is public
                let key: Array<U8, 16> = verifier.alloc().unwrap();

                verifier.mark_blind(key).unwrap();
                verifier.commit(key).unwrap();

                for _ in 0..BATCH_SIZE {
                    let msg: Array<U8, 16> = verifier.alloc().unwrap();
                    verifier.mark_public(msg).unwrap();
                    verifier.assign(msg, [42u8; 16]).unwrap();
                    verifier.commit(msg).unwrap();

                    let ciphertext: Array<U8, 16> = verifier
                        .call(
                            Call::builder(AES128.clone())
                                .arg(key)
                                .arg(msg)
                                .build()
                                .unwrap(),
                        )
                        .unwrap();

                    std::mem::drop(verifier.decode(ciphertext).unwrap());
                }

                verifier.execute_all(&mut ctx_v).await.unwrap();
            }
        );

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BATCH_SIZE as u64 * and_gates_per_circuit,
    }
}
