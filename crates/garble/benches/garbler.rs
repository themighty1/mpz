//! Isolated garbler benchmarks.
//!
//! Records protocol messages for replay-based isolated benchmarking of garbler.
//!
//! Run with: cargo bench -p mpz-garble --bench garbler

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_circuits::AES128;
use mpz_common::context::{
    Multithread, RecordedMtData, recording_mt_context_with_limit, recording_st_context_with_limit,
    replay_mt_context_with_limit, replay_st_context,
};
use mpz_garble::protocol::semihonest::{Evaluator, Garbler};
use mpz_memory_core::{Array, binary::U8, correlated::Delta};
use mpz_ot::ideal::msg_cot::{MsgIdealCOTSender, msg_ideal_cot};
use mpz_vm_core::{Call, prelude::*};
use rand::{SeedableRng, rngs::StdRng};

const BLOCK_COUNT: usize = 1000;

/// Calculate max frame length based on workload size.
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    let bytes_per_gate = 32 + 16; // garbled gate + label overhead
    let overhead = 1.5; // serialization overhead
    let gates = circuit.and_count() * circuit_count;
    ((gates * bytes_per_gate) as f64 * overhead) as usize
}

/// Runs the full garble protocol with garbler and evaluator.
/// Records evaluator->garbler messages (ctx_ev is the recording context).
async fn run_protocol_record_evaluator(
    ctx_gb: &mut mpz_common::Context,
    ctx_ev: &mut mpz_common::Context,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (cot_send, cot_recv) = msg_ideal_cot(delta.into_inner());

    let mut gb = Garbler::new(cot_send, [0u8; 16], delta);
    let mut ev = Evaluator::new(cot_recv);

    futures::join!(
        async {
            let key: Array<U8, 16> = gb.alloc().unwrap();
            gb.mark_private(key).unwrap();
            gb.assign(key, [0u8; 16]).unwrap();
            gb.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
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

            gb.flush(ctx_gb).await.unwrap();
            gb.execute(ctx_gb).await.unwrap();
            gb.flush(ctx_gb).await.unwrap();
        },
        async {
            let key: Array<U8, 16> = ev.alloc().unwrap();
            ev.mark_blind(key).unwrap();
            ev.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
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

            ev.flush(ctx_ev).await.unwrap();
            ev.execute(ctx_ev).await.unwrap();
            ev.flush(ctx_ev).await.unwrap();
        }
    );
}

/// Records evaluator->garbler messages for garbler replay.
/// Returns (recorded_bytes, delta) needed for deterministic replay.
fn record_for_garbler(seed: u64) -> (Vec<u8>, Delta) {
    block_on(async {
        // ctx_1 (evaluator) is recorded, ctx_0 (garbler) receives
        let (mut ctx_gb, mut ctx_ev, recorded) =
            recording_st_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));

        // Capture delta for garbler replay
        let mut rng = StdRng::seed_from_u64(seed);
        let delta = Delta::random(&mut rng);

        run_protocol_record_evaluator(&mut ctx_gb, &mut ctx_ev, seed).await;
        (recorded.lock().unwrap().clone(), delta)
    })
}

/// Runs garbler only with replay context.
async fn run_garbler_with_replay(ctx: &mut mpz_common::Context, delta: Delta) {
    let cot_send = MsgIdealCOTSender::new(delta.into_inner());
    let mut gb = Garbler::new(cot_send, [0u8; 16], delta);

    let key: Array<U8, 16> = gb.alloc().unwrap();
    gb.mark_private(key).unwrap();
    gb.assign(key, [0u8; 16]).unwrap();
    gb.commit(key).unwrap();

    for _ in 0..BLOCK_COUNT {
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

    gb.flush(ctx).await.unwrap();
    gb.execute(ctx).await.unwrap();
    gb.flush(ctx).await.unwrap();
}

// ============================================================================
// Multi-threaded isolated garbler benchmark
// ============================================================================

/// Runs the full garble protocol with MT contexts.
/// Records evaluator->garbler messages.
async fn run_protocol_record_evaluator_mt(
    exec_gb: &mut Multithread,
    exec_ev: &mut Multithread,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

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

            for _ in 0..BLOCK_COUNT {
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

            for _ in 0..BLOCK_COUNT {
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

/// Records evaluator->garbler messages for MT garbler replay.
fn record_for_garbler_mt(seed: u64) -> (RecordedMtData, Delta) {
    block_on(async {
        let (mut exec_gb, mut exec_ev, recorded) =
            recording_mt_context_with_limit(1024 * 1024, max_frame_length(&AES128, BLOCK_COUNT));

        let mut rng = StdRng::seed_from_u64(seed);
        let delta = Delta::random(&mut rng);

        run_protocol_record_evaluator_mt(&mut exec_gb, &mut exec_ev, seed).await;
        let data = recorded.lock().unwrap().clone();
        for (id, bytes) in &data.channels {
            println!("  Channel {:?}: {} bytes", id, bytes.len());
        }
        (data, delta)
    })
}

/// Runs MT garbler only with replay context.
async fn run_garbler_with_replay_mt(exec: &mut Multithread, delta: Delta) {
    let cot_send = MsgIdealCOTSender::new(delta.into_inner());
    let mut gb = Garbler::new(cot_send, [0u8; 16], delta);

    let mut ctx = exec.new_context().await.unwrap();

    let key: Array<U8, 16> = gb.alloc().unwrap();
    gb.mark_private(key).unwrap();
    gb.assign(key, [0u8; 16]).unwrap();
    gb.commit(key).unwrap();

    for _ in 0..BLOCK_COUNT {
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("garbler");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    let and_gates_per_circuit = AES128.and_count() as u64;
    group.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

    // ST isolated garbler benchmark
    println!("Recording for garbler...");
    let (recorded, delta) = record_for_garbler(0);
    println!("Recorded {} bytes", recorded.len());

    // Verify determinism
    let (recorded_2, _) = record_for_garbler(0);
    assert_eq!(recorded, recorded_2, "Garbler recordings not deterministic");

    group.bench_function("garbler", |b| {
        b.iter(|| {
            block_on(async {
                let mut ctx = replay_st_context(recorded.clone(), max_frame_length(&AES128, BLOCK_COUNT));
                run_garbler_with_replay(&mut ctx, delta).await;
            })
        });
    });

    group.finish();

    // MT isolated garbler benchmark
    let mut group_mt = c.benchmark_group("garbler_mt");
    group_mt.sample_size(10);
    group_mt.measurement_time(std::time::Duration::from_secs(10));
    group_mt.throughput(Throughput::Elements(and_gates_per_circuit * BLOCK_COUNT as u64));

    println!("Recording for MT garbler...");
    let (recorded_mt, delta_mt) = record_for_garbler_mt(0);
    let total_bytes: usize = recorded_mt.channels.values().map(|v| v.len()).sum();
    println!("Recorded {} channels, {} total bytes", recorded_mt.channels.len(), total_bytes);

    // Verify determinism
    let (recorded_mt_2, _) = record_for_garbler_mt(0);
    assert_eq!(
        recorded_mt.channels.keys().collect::<std::collections::HashSet<_>>(),
        recorded_mt_2.channels.keys().collect::<std::collections::HashSet<_>>(),
        "MT Garbler recordings have different channels"
    );
    for (id, data) in &recorded_mt.channels {
        assert_eq!(
            data,
            recorded_mt_2.channels.get(id).unwrap(),
            "MT Garbler recordings not deterministic for channel {:?}",
            id
        );
    }

    group_mt.bench_function("garbler_mt", |b| {
        b.iter(|| {
            block_on(async {
                let mut exec = replay_mt_context_with_limit(
                    recorded_mt.clone(),
                    max_frame_length(&AES128, BLOCK_COUNT),
                );
                run_garbler_with_replay_mt(&mut exec, delta_mt).await;
            })
        });
    });

    group_mt.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
