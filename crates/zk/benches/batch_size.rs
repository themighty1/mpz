//! Benchmark for different batch sizes in ProverConfig/VerifierConfig.
//!
//! Run with: cargo bench -p mpz-zk --bench batch_size

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::test_mt_context;
use mpz_ot::ideal::rcot::ideal_rcot;
use mpz_vm_core::{
    Call,
    memory::{Array, binary::U8, correlated::Delta},
    prelude::*,
};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size");

    const CIRCUIT_COUNT: usize = 100;
    // Throughput in AND gates (elem/s = AND gates/s)
    let and_gates_per_circuit = AES128.and_count();
    group.throughput(Throughput::Elements((and_gates_per_circuit * CIRCUIT_COUNT) as u64));

    // Test batch sizes: 200K, 400K, 600K, 800K, 1M
    for batch_size in [200_000, 400_000, 600_000, 800_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::new("aes128_x100", format!("batch_{}", batch_size / 1000)),
            &batch_size,
            |b, &batch_size| {
                let mut rng = StdRng::seed_from_u64(0);
                let delta = Delta::random(&mut rng);

                let (mut exec_p, mut exec_v) = test_mt_context(8);
                let mut ctx_p = block_on(exec_p.new_context()).unwrap();
                let mut ctx_v = block_on(exec_v.new_context()).unwrap();

                b.iter(|| {
                    block_on(async {
                        let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

                        let prover_config = ProverConfig::builder()
                            .batch_size(batch_size)
                            .build()
                            .unwrap();
                        let verifier_config = VerifierConfig::builder()
                            .batch_size(batch_size)
                            .build()
                            .unwrap();

                        let mut prover = Prover::new(prover_config, ot_recv);
                        let mut verifier = Verifier::new(verifier_config, delta, ot_send);

                        futures::join!(
                            {
                                let key: Array<U8, 16> = prover.alloc().unwrap();
                                prover.mark_private(key).unwrap();
                                prover.assign(key, [0u8; 16]).unwrap();
                                prover.commit(key).unwrap();

                                for _ in 0..CIRCUIT_COUNT {
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

                                async {
                                    prover.flush(&mut ctx_p).await.unwrap();
                                    prover.execute(&mut ctx_p).await.unwrap();
                                    prover.flush(&mut ctx_p).await.unwrap();
                                }
                            },
                            {
                                let key: Array<U8, 16> = verifier.alloc().unwrap();
                                verifier.mark_blind(key).unwrap();
                                verifier.commit(key).unwrap();

                                for _ in 0..CIRCUIT_COUNT {
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

                                async {
                                    verifier.flush(&mut ctx_v).await.unwrap();
                                    verifier.execute(&mut ctx_v).await.unwrap();
                                    verifier.flush(&mut ctx_v).await.unwrap();
                                }
                            }
                        );
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
