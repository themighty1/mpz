//! Isolated prover benchmarks.
//!
//! Records protocol messages for replay-based isolated benchmarking of prover.
//!
//! Run with: cargo bench -p mpz-zk --bench isolated_prover

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::{recording_st_context_with_limit, replay_st_context};
use mpz_ot::ideal::msg_rcot::{MsgIdealRCOTReceiver, msg_ideal_rcot};
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
/// Records verifier->prover messages (ctx_v is the recording context).
async fn run_protocol_record_verifier(
    ctx_p: &mut mpz_common::Context,
    ctx_v: &mut mpz_common::Context,
    seed: u64,
    batch_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (ot_send, ot_recv) = msg_ideal_rcot(rng.random(), delta.into_inner());

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

/// Records verifier->prover messages for prover replay.
fn record_for_prover(seed: u64, batch_size: usize) -> Vec<u8> {
    block_on(async {
        // ctx_1 (verifier) is recorded
        let (mut ctx_p, mut ctx_v, recorded) =
            recording_st_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));
        run_protocol_record_verifier(&mut ctx_p, &mut ctx_v, seed, batch_size).await;
        recorded.lock().unwrap().clone()
    })
}

/// Runs prover only with replay context.
async fn run_prover_with_replay(ctx: &mut mpz_common::Context, batch_size: usize) {
    let ot_recv = MsgIdealRCOTReceiver::new();
    let prover_config = ProverConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut prover = Prover::new(prover_config, ot_recv);

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

    prover.flush(ctx).await.unwrap();
    prover.execute(ctx).await.unwrap();
    prover.flush(ctx).await.unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolated_prover");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    let and_gates_per_circuit = AES128.and_count() as u64;
    group.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

    for &batch_size in &BATCH_SIZES {
        println!("Recording for prover with batch_size={}...", batch_size);
        let recorded = record_for_prover(0, batch_size);
        println!("Recorded {} bytes", recorded.len());

        // Verify determinism
        let recorded_2 = record_for_prover(0, batch_size);
        assert_eq!(
            recorded, recorded_2,
            "Prover recordings not deterministic for batch_size={}",
            batch_size
        );

        group.bench_with_input(
            BenchmarkId::new("prover", format!("batch_{}k", batch_size / 1000)),
            &(recorded, batch_size),
            |b, (recorded, batch_size)| {
                b.iter(|| {
                    block_on(async {
                        let mut ctx =
                            replay_st_context(recorded.clone(), max_frame_length(&AES128, BLOCK_COUNT));
                        run_prover_with_replay(&mut ctx, *batch_size).await;
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
