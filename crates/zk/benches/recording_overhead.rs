//! Recording overhead benchmarks.
//!
//! Compares baseline MT context vs recording MT context to measure
//! the overhead of the recording infrastructure.
//!
//! Run with: cargo bench -p mpz-zk --bench recording_overhead

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::{
    Multithread, recording_mt_context_with_limit, test_mt_context,
};
use mpz_ot::ideal::rcot::ideal_rcot;
use mpz_vm_core::{
    Call,
    memory::{Array, binary::U8, correlated::Delta},
    prelude::*,
};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

const BLOCK_COUNT: usize = 1000;
const BATCH_SIZE: usize = 1_000_000;

/// Calculate max frame length based on workload size.
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    let bytes_per_correlation = 1 + 16;
    let overhead = 1.2;
    let correlations = circuit.and_count() * circuit_count;
    ((correlations * bytes_per_correlation) as f64 * overhead) as usize
}

/// Runs the full ZK protocol with given MT contexts.
async fn run_full_protocol(
    exec_p: &mut Multithread,
    exec_v: &mut Multithread,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

    let prover_config = ProverConfig::builder()
        .batch_size(BATCH_SIZE)
        .build()
        .unwrap();
    let verifier_config = VerifierConfig::builder()
        .batch_size(BATCH_SIZE)
        .build()
        .unwrap();

    let mut prover = Prover::new(prover_config, ot_recv);
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    let mut ctx_p = exec_p.new_context().await.unwrap();
    let mut ctx_v = exec_v.new_context().await.unwrap();

    futures::join!(
        {
            let key: Array<U8, 16> = prover.alloc().unwrap();
            prover.mark_private(key).unwrap();
            prover.assign(key, [0u8; 16]).unwrap();
            prover.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
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

            for _ in 0..BLOCK_COUNT {
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
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("recording_overhead");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    let and_gates_per_circuit = AES128.and_count() as u64;
    group.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

    // Baseline: test_mt_context (no recording)
    group.bench_function(BenchmarkId::new("full_protocol", "baseline"), |b| {
        b.iter(|| {
            block_on(async {
                let (mut exec_p, mut exec_v) = test_mt_context(1024 * 1024);
                run_full_protocol(&mut exec_p, &mut exec_v, 0).await;
            })
        });
    });

    // With recording: recording_mt_context_with_limit
    group.bench_function(BenchmarkId::new("full_protocol", "recording"), |b| {
        b.iter(|| {
            block_on(async {
                let (mut exec_p, mut exec_v, _recorded) =
                    recording_mt_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));
                run_full_protocol(&mut exec_p, &mut exec_v, 0).await;
            })
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
