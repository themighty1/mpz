//! Isolated verifier benchmarks.
//!
//! Records protocol messages for replay-based isolated benchmarking of verifier.
//!
//! Run with: cargo bench -p mpz-zk --bench isolated_verifier

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::{
    Multithread, RecordedMtData, recording_mt_context_with_limit, recording_st_context_with_limit,
    replay_mt_context_with_limit, replay_st_context,
};
use mpz_core::Block;
use mpz_ot::ideal::rcot::{IdealRCOTSender, ideal_rcot};
use mpz_vm_core::{
    Call,
    memory::{Array, binary::U8, correlated::Delta},
    prelude::*,
};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

const BLOCK_COUNT: usize = 1000;
const BATCH_SIZES: [usize; 5] = [200_000, 400_000, 600_000, 800_000, 1_000_000];

/// Calculate max frame length based on workload size.
///
/// Ideal OT sends per correlation:
/// - 1 choice bit (serialized as 1 byte)
/// - 1 Block (16 bytes MAC)
/// Plus serialization overhead (~20% buffer).
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    let bytes_per_correlation = 1 + 16; // choice bit + MAC
    let overhead = 1.2; // serialization overhead
    let correlations = circuit.and_count() * circuit_count;
    ((correlations * bytes_per_correlation) as f64 * overhead) as usize
}

/// Runs the full ZK protocol with prover and verifier.
/// Records prover->verifier messages (ctx_p is the recording context).
async fn run_protocol_record_prover(
    ctx_p: &mut mpz_common::Context,
    ctx_v: &mut mpz_common::Context,
    seed: u64,
    batch_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

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
                prover.flush(ctx_p).await.unwrap();
                prover.execute(ctx_p).await.unwrap();
                prover.flush(ctx_p).await.unwrap();
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
                verifier.flush(ctx_v).await.unwrap();
                verifier.execute(ctx_v).await.unwrap();
                verifier.flush(ctx_v).await.unwrap();
            }
        }
    );
}

/// Records prover->verifier messages for verifier replay.
fn record_for_verifier(seed: u64, batch_size: usize) -> (Vec<u8>, Block, Delta) {
    block_on(async {
        // Swap: ctx_1 (prover) is recorded, ctx_0 (verifier) receives
        let (mut ctx_v, mut ctx_p, recorded) =
            recording_st_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));

        // Need to capture delta for verifier replay
        let mut rng = StdRng::seed_from_u64(seed);
        let delta = Delta::random(&mut rng);
        let ot_seed: Block = rng.random();

        run_protocol_record_prover(&mut ctx_p, &mut ctx_v, seed, batch_size).await;
        (recorded.lock().unwrap().clone(), ot_seed, delta)
    })
}

/// Runs verifier only with replay context.
async fn run_verifier_with_replay(
    ctx: &mut mpz_common::Context,
    batch_size: usize,
    delta: Delta,
    ot_seed: Block,
) {
    // OT sender needs seed and delta to generate consistent correlations
    let ot_send = IdealRCOTSender::new(ot_seed, delta.into_inner());
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

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

    verifier.flush(ctx).await.unwrap();
    verifier.execute(ctx).await.unwrap();
    verifier.flush(ctx).await.unwrap();
}

// ============================================================================
// Multi-threaded isolated verifier benchmark
// ============================================================================

/// Runs the full ZK protocol with MT contexts.
/// Records prover->verifier messages.
async fn run_protocol_record_prover_mt(
    exec_p: &mut Multithread,
    exec_v: &mut Multithread,
    seed: u64,
    batch_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

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

/// Records prover->verifier messages for MT verifier replay.
fn record_for_verifier_mt(seed: u64, batch_size: usize) -> (RecordedMtData, Block, Delta) {
    block_on(async {
        // Swap: exec_1 (prover) is recorded, exec_0 (verifier) receives
        let (mut exec_v, mut exec_p, recorded) =
            recording_mt_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));

        // Need to capture delta for verifier replay
        let mut rng = StdRng::seed_from_u64(seed);
        let delta = Delta::random(&mut rng);
        let ot_seed: Block = rng.random();

        run_protocol_record_prover_mt(&mut exec_p, &mut exec_v, seed, batch_size).await;
        let data = recorded.lock().unwrap().clone();
        // Debug: print channel IDs and their sizes
        for (id, bytes) in &data.channels {
            println!("  Channel {:?}: {} bytes", id, bytes.len());
        }
        (data, ot_seed, delta)
    })
}

/// Runs MT verifier only with replay context.
async fn run_verifier_with_replay_mt(
    exec: &mut Multithread,
    batch_size: usize,
    delta: Delta,
    ot_seed: Block,
) {
    let ot_send = IdealRCOTSender::new(ot_seed, delta.into_inner());
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    let mut ctx = exec.new_context().await.unwrap();

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

    verifier.flush(&mut ctx).await.unwrap();
    verifier.execute(&mut ctx).await.unwrap();
    verifier.flush(&mut ctx).await.unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolated_verifier");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    let and_gates_per_circuit = AES128.and_count() as u64;
    group.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

    for &batch_size in &BATCH_SIZES {
        println!("Recording for verifier with batch_size={}...", batch_size);
        let (recorded, ot_seed, delta) = record_for_verifier(0, batch_size);
        println!("Recorded {} bytes", recorded.len());

        // Verify determinism
        let (recorded_2, _, _) = record_for_verifier(0, batch_size);
        assert_eq!(
            recorded, recorded_2,
            "Verifier recordings not deterministic for batch_size={}",
            batch_size
        );

        group.bench_with_input(
            BenchmarkId::new("verifier", format!("batch_{}k", batch_size / 1000)),
            &(recorded, batch_size, delta, ot_seed),
            |b, (recorded, batch_size, delta, ot_seed)| {
                b.iter(|| {
                    block_on(async {
                        let mut ctx =
                            replay_st_context(recorded.clone(), max_frame_length(&AES128, BLOCK_COUNT));
                        run_verifier_with_replay(&mut ctx, *batch_size, *delta, *ot_seed).await;
                    })
                });
            },
        );
    }

    group.finish();

    // MT isolated verifier benchmark
    // Record and bench one batch size at a time to limit memory usage
    for &batch_size in &BATCH_SIZES {
        let mut group_mt = c.benchmark_group("isolated_verifier_mt");
        group_mt.sample_size(10);
        group_mt.measurement_time(std::time::Duration::from_secs(10));
        group_mt.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

        println!("Recording for MT verifier with batch_size={}...", batch_size);
        let (recorded, ot_seed, delta) = record_for_verifier_mt(0, batch_size);
        let total_bytes: usize = recorded.channels.values().map(|v| v.len()).sum();
        println!(
            "Recorded {} channels, {} total bytes",
            recorded.channels.len(),
            total_bytes
        );

        // Verify determinism
        let (recorded_2, _, _) = record_for_verifier_mt(0, batch_size);
        assert_eq!(
            recorded.channels.keys().collect::<std::collections::HashSet<_>>(),
            recorded_2.channels.keys().collect::<std::collections::HashSet<_>>(),
            "MT Verifier recordings have different channels for batch_size={}",
            batch_size
        );
        for (id, data) in &recorded.channels {
            assert_eq!(
                data,
                recorded_2.channels.get(id).unwrap(),
                "MT Verifier recordings not deterministic for channel {:?}, batch_size={}",
                id,
                batch_size
            );
        }

        group_mt.bench_with_input(
            BenchmarkId::new("verifier_mt", format!("batch_{}k", batch_size / 1000)),
            &(recorded, batch_size, delta, ot_seed),
            |b, (recorded, batch_size, delta, ot_seed)| {
                b.iter(|| {
                    block_on(async {
                        let mut exec = replay_mt_context_with_limit(
                            recorded.clone(),
                            max_frame_length(&AES128, BLOCK_COUNT),
                        );
                        run_verifier_with_replay_mt(&mut exec, *batch_size, *delta, *ot_seed).await;
                    })
                });
            },
        );

        group_mt.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
