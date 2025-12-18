//! Full ZK protocol benchmarks (prover + verifier together).
//!
//! Run with: cargo bench -p mpz-zk --bench zk

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::{test_mt_context, test_st_context};
use mpz_ot::ideal::rcot::ideal_rcot;
use mpz_vm_core::{
    Call,
    memory::{Array, binary::U8, correlated::Delta},
    prelude::*,
};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

// Gate count thresholds
const THRESHOLDS: &[(u64, &str)] = &[(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")];

fn criterion_benchmark(c: &mut Criterion) {
    let circuit = &*AES128;
    let gates_per_circuit = circuit.and_count() as u64;

    // ST full protocol benchmark
    let mut group = c.benchmark_group("full");
    group.sample_size(10);

    for &(threshold, name) in THRESHOLDS {
        let circuit_count = threshold.div_ceil(gates_per_circuit) as usize;
        let actual_gates = circuit_count as u64 * gates_per_circuit;

        group.throughput(Throughput::Elements(actual_gates));

        group.bench_function(BenchmarkId::new("st", name), |b| {
            let mut rng = StdRng::seed_from_u64(0);
            let delta = Delta::random(&mut rng);

            let (mut ctx_p, mut ctx_v) = test_st_context(1024 * 1024);

            b.iter(|| {
                block_on(async {
                    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

                    let mut prover = Prover::new(ProverConfig::default(), ot_recv);
                    let mut verifier = Verifier::new(VerifierConfig::default(), delta, ot_send);

                    futures::join!(
                        async {
                            let key: Array<U8, 16> = prover.alloc().unwrap();
                            prover.mark_private(key).unwrap();
                            prover.assign(key, [0u8; 16]).unwrap();
                            prover.commit(key).unwrap();

                            for _ in 0..circuit_count {
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

                            prover.flush(&mut ctx_p).await.unwrap();
                            prover.execute(&mut ctx_p).await.unwrap();
                            prover.flush(&mut ctx_p).await.unwrap();
                        },
                        async {
                            let key: Array<U8, 16> = verifier.alloc().unwrap();
                            verifier.mark_blind(key).unwrap();
                            verifier.commit(key).unwrap();

                            for _ in 0..circuit_count {
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

                            verifier.flush(&mut ctx_v).await.unwrap();
                            verifier.execute(&mut ctx_v).await.unwrap();
                            verifier.flush(&mut ctx_v).await.unwrap();
                        }
                    );
                })
            });
        });
    }

    group.finish();

    // MT full protocol benchmark
    let mut group_mt = c.benchmark_group("full");
    group_mt.sample_size(10);

    for &(threshold, name) in THRESHOLDS {
        let circuit_count = threshold.div_ceil(gates_per_circuit) as usize;
        let actual_gates = circuit_count as u64 * gates_per_circuit;

        group_mt.throughput(Throughput::Elements(actual_gates));

        group_mt.bench_function(BenchmarkId::new("mt", name), |b| {
            let mut rng = StdRng::seed_from_u64(0);
            let delta = Delta::random(&mut rng);

            let (mut exec_p, mut exec_v) = test_mt_context(8);
            let mut ctx_p = block_on(exec_p.new_context()).unwrap();
            let mut ctx_v = block_on(exec_v.new_context()).unwrap();

            b.iter(|| {
                block_on(async {
                    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

                    let mut prover = Prover::new(ProverConfig::default(), ot_recv);
                    let mut verifier = Verifier::new(VerifierConfig::default(), delta, ot_send);

                    futures::join!(
                        async {
                            let key: Array<U8, 16> = prover.alloc().unwrap();
                            prover.mark_private(key).unwrap();
                            prover.assign(key, [0u8; 16]).unwrap();
                            prover.commit(key).unwrap();

                            for _ in 0..circuit_count {
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

                            prover.flush(&mut ctx_p).await.unwrap();
                            prover.execute(&mut ctx_p).await.unwrap();
                            prover.flush(&mut ctx_p).await.unwrap();
                        },
                        async {
                            let key: Array<U8, 16> = verifier.alloc().unwrap();
                            verifier.mark_blind(key).unwrap();
                            verifier.commit(key).unwrap();

                            for _ in 0..circuit_count {
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

                            verifier.flush(&mut ctx_v).await.unwrap();
                            verifier.execute(&mut ctx_v).await.unwrap();
                            verifier.flush(&mut ctx_v).await.unwrap();
                        }
                    );
                })
            });
        });
    }

    group_mt.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
