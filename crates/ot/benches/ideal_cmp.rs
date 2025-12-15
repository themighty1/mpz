//! Temporary benchmark comparing CallSync-based vs message-based ideal RCOT.
//!
//! Run with: cargo bench -p mpz-ot --features ideal --bench ideal_cmp

use criterion::{Criterion, criterion_group, criterion_main};
use futures::executor::block_on;
use mpz_common::context::test_st_context;
use mpz_common::Flush;
use mpz_core::Block;
use mpz_ot::ideal::msg_rcot::msg_ideal_rcot;
use mpz_ot::ideal::rcot::ideal_rcot;
use mpz_ot::rcot::{RCOTReceiver, RCOTSender};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ideal_rcot_cmp");

    const OT_COUNT: usize = 1_000_000;

    // Benchmark CallSync-based ideal RCOT
    group.bench_function("callsync_ideal_rcot", |b| {
        b.iter(|| {
            block_on(async {
                let mut rng = StdRng::seed_from_u64(0);
                let delta: Block = rng.random();

                let (mut ctx_s, mut ctx_r) = test_st_context(1024 * 1024);
                let (mut sender, mut receiver) = ideal_rcot(rng.random(), delta);

                // Allocate OTs
                sender.alloc(OT_COUNT).unwrap();
                receiver.alloc(OT_COUNT).unwrap();

                // Flush (this is where CallSync synchronizes)
                let (r1, r2) = futures::join!(
                    sender.flush(&mut ctx_s),
                    receiver.flush(&mut ctx_r)
                );
                r1.unwrap();
                r2.unwrap();

                // Transfer
                let _sender_out = sender.try_send_rcot(OT_COUNT).unwrap();
                let _receiver_out = receiver.try_recv_rcot(OT_COUNT).unwrap();
            })
        });
    });

    // Benchmark message-based ideal RCOT
    group.bench_function("msg_ideal_rcot", |b| {
        b.iter(|| {
            block_on(async {
                let mut rng = StdRng::seed_from_u64(0);
                let delta: Block = rng.random();

                let (mut ctx_s, mut ctx_r) = test_st_context(1024 * 1024);
                let (mut sender, mut receiver) = msg_ideal_rcot(rng.random(), delta);

                // Allocate OTs
                sender.alloc(OT_COUNT).unwrap();
                receiver.alloc(OT_COUNT).unwrap();

                // Flush (this is where messages are exchanged)
                let (r1, r2) = futures::join!(
                    sender.flush(&mut ctx_s),
                    receiver.flush(&mut ctx_r)
                );
                r1.unwrap();
                r2.unwrap();

                // Transfer
                let _sender_out = sender.try_send_rcot(OT_COUNT).unwrap();
                let _receiver_out = receiver.try_recv_rcot(OT_COUNT).unwrap();
            })
        });
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
