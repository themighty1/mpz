use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use itybity::ToBits;
use mpz_core::{Block, lpn::LpnType};
use mpz_ot_core::{
    chou_orlandi,
    cot::{COTReceiver, COTSender},
    ferret::{self, FerretConfig},
    ideal::{
        cot::{IdealCOT, ideal_cot},
        ot::{IdealOT, ideal_ot},
        rcot::IdealRCOT,
        rot::{IdealROT, ideal_rot},
    },
    rot::{ROTReceiver, ROTSender},
    kos,
    ot::{OTReceiver, OTSender},
    rcot::{RCOTReceiver, RCOTSender},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

fn chou_orlandi(c: &mut Criterion) {
    let mut group = c.benchmark_group("chou_orlandi");
    for n in [128, 256, 1024] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let msgs = vec![[Block::ONES; 2]; n];
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let choices = (0..n).map(|_| rng.random()).collect::<Vec<bool>>();
            b.iter(|| {
                let sender = chou_orlandi::Sender::default();
                let receiver = chou_orlandi::Receiver::default();

                let (sender_setup, mut sender) = sender.setup();
                let mut receiver = receiver.setup(sender_setup);

                let sender_output = sender.queue_send_ot(&msgs).unwrap();
                let receiver_output = receiver.queue_recv_ot(&choices).unwrap();

                let receiver_payload = receiver.choose();
                let sender_payload = sender.send(receiver_payload).unwrap();
                receiver.receive(sender_payload).unwrap();

                black_box((sender_output, receiver_output))
            })
        });
    }
}

fn kos(c: &mut Criterion) {
    let mut group = c.benchmark_group("kos");
    for n in [1024, 262144] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let delta = Block::random(&mut rng);

            let receiver_seeds: [[Block; 2]; 128] =
                std::array::from_fn(|_| [rng.random(), rng.random()]);
            let sender_seeds: [Block; 128] = delta
                .iter_lsb0()
                .zip(receiver_seeds)
                .map(|(b, seeds)| if b { seeds[1] } else { seeds[0] })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            b.iter(|| {
                let sender = kos::Sender::new(kos::SenderConfig::default(), delta);
                let receiver = kos::Receiver::new(kos::ReceiverConfig::default());

                let mut sender = sender.setup(sender_seeds);
                let mut receiver = receiver.setup(receiver_seeds);

                sender.alloc(n).unwrap();
                receiver.alloc(n).unwrap();

                while receiver.wants_extend() {
                    let extend = receiver.extend().unwrap();
                    sender.extend(extend).unwrap();
                }

                let chi_seed = sender.check_start();
                let check = receiver.check(chi_seed).unwrap();
                sender.check(check).unwrap();

                black_box((sender, receiver));
            })
        });
    }
}

fn ferret(c: &mut Criterion) {
    let mut group = c.benchmark_group("ferret");
    for ty in [LpnType::Uniform, LpnType::Regular] {
        let ty_str = match ty {
            LpnType::Uniform => "uniform",
            LpnType::Regular => "regular",
        };

        for n in [262144, 1_000_000] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(ty_str, n), &n, |b, &n| {
                let mut rng = ChaCha12Rng::seed_from_u64(0);
                let delta = Block::random(&mut rng);

                let mut builder = FerretConfig::builder();
                builder.lpn_type(ty);
                let config = builder.build().unwrap();

                b.iter(|| {
                    let cot = IdealRCOT::new(rng.random(), delta);
                    let mut sender = ferret::Sender::new(rng.random(), config.clone(), cot.clone());
                    let mut receiver = ferret::Receiver::new(rng.random(), config.clone(), cot);

                    let init = receiver.initialize().unwrap();
                    sender.initialize(init).unwrap();
                    sender.alloc_bootstrap().unwrap();
                    receiver.alloc_bootstrap().unwrap();
                    sender.acquire_cot().flush().unwrap();
                    receiver.acquire_cot().flush().unwrap();
                    sender.alloc(n).unwrap();
                    receiver.alloc(n).unwrap();

                    while sender.wants_extend() && receiver.wants_extend() {
                        sender.start_extend().unwrap();
                        let msg = receiver.start_extend().unwrap();
                        let msg = sender.extend(msg).unwrap();
                        let msg = receiver.extend(msg).unwrap();
                        let msg = sender.check(msg).unwrap();
                        receiver.finish_extend(msg).unwrap();
                        sender.finish_extend().unwrap();
                    }

                    black_box((sender, receiver));
                })
            });
        }
    }
}

// TODO: This benchmark is temporary. Can be removed once transition to msg-based COT
// is complete and all dependent crates have been updated.
fn ideal_cot_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ideal_cot_cmp");

    for n in [100, 1000, 10000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));

        // Benchmark IdealCOT wrapper (message-based internally)
        group.bench_with_input(BenchmarkId::new("wrapper", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let delta = Block::random(&mut rng);
            let choices: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let keys: Vec<Block> = (0..n).map(|_| rng.random()).collect();

            b.iter(|| {
                let mut ideal = IdealCOT::new(delta);

                let sender_out = ideal.queue_send_cot(&keys).unwrap();
                let receiver_out = ideal.queue_recv_cot(&choices).unwrap();
                ideal.flush().unwrap();

                black_box((sender_out, receiver_out))
            })
        });

        // Benchmark separate sender/receiver with explicit message passing
        group.bench_with_input(BenchmarkId::new("msg_based", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let delta = Block::random(&mut rng);
            let choices: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let keys: Vec<Block> = (0..n).map(|_| rng.random()).collect();

            b.iter(|| {
                let (mut sender, mut receiver) = ideal_cot(delta);

                let sender_out = sender.queue_send_cot(&keys).unwrap();
                let receiver_out = receiver.queue_recv_cot(&choices).unwrap();

                let flush_msg = sender.flush().unwrap();
                receiver.flush(flush_msg).unwrap();

                black_box((sender_out, receiver_out))
            })
        });
    }

    group.finish();
}

// TODO: This benchmark is temporary. Can be removed once transition to msg-based OT
// is complete and all dependent crates have been updated.
fn ideal_ot_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ideal_ot_cmp");

    for n in [100, 1000, 10000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));

        // Benchmark IdealOT wrapper (message-based internally)
        group.bench_with_input(BenchmarkId::new("wrapper", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let choices: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let msgs: Vec<[Block; 2]> = (0..n).map(|_| [rng.random(), rng.random()]).collect();

            b.iter(|| {
                let mut ideal = IdealOT::new();

                let sender_out = ideal.queue_send_ot(&msgs).unwrap();
                let receiver_out = ideal.queue_recv_ot(&choices).unwrap();
                ideal.flush().unwrap();

                black_box((sender_out, receiver_out))
            })
        });

        // Benchmark separate sender/receiver with explicit message passing
        group.bench_with_input(BenchmarkId::new("msg_based", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let choices: Vec<bool> = (0..n).map(|_| rng.random()).collect();
            let msgs: Vec<[Block; 2]> = (0..n).map(|_| [rng.random(), rng.random()]).collect();

            b.iter(|| {
                let (mut sender, mut receiver) = ideal_ot();

                let sender_out = sender.queue_send_ot(&msgs).unwrap();
                let receiver_out = receiver.queue_recv_ot(&choices).unwrap();

                let flush_msg = sender.flush().unwrap();
                receiver.flush(flush_msg).unwrap();

                black_box((sender_out, receiver_out))
            })
        });
    }

    group.finish();
}

// TODO: This benchmark is temporary. Can be removed once transition to msg-based ROT
// is complete and all dependent crates have been updated.
fn ideal_rot_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ideal_rot_cmp");

    for n in [100, 1000, 10000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));

        // Benchmark IdealROT wrapper (message-based internally)
        group.bench_with_input(BenchmarkId::new("wrapper", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let seed = Block::random(&mut rng);

            b.iter(|| {
                let mut ideal = IdealROT::new(seed);

                ROTSender::alloc(&mut ideal, n).unwrap();
                ROTReceiver::alloc(&mut ideal, n).unwrap();
                ideal.flush().unwrap();

                let sender_out = ideal.try_send_rot(n).unwrap();
                let receiver_out = ideal.try_recv_rot(n).unwrap();

                black_box((sender_out, receiver_out))
            })
        });

        // Benchmark separate sender/receiver with explicit message passing
        group.bench_with_input(BenchmarkId::new("msg_based", n), &n, |b, &n| {
            let mut rng = ChaCha12Rng::seed_from_u64(0);
            let seed = Block::random(&mut rng);

            b.iter(|| {
                let (mut sender, mut receiver) = ideal_rot(seed);

                ROTSender::alloc(&mut sender, n).unwrap();
                ROTReceiver::alloc(&mut receiver, n).unwrap();

                let flush_msg = sender.flush().unwrap();
                receiver.flush(flush_msg).unwrap();

                let sender_out = sender.try_send_rot(n).unwrap();
                let receiver_out = receiver.try_recv_rot(n).unwrap();

                black_box((sender_out, receiver_out))
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = chou_orlandi_benches;
    config = Criterion::default().sample_size(50);
    targets = chou_orlandi
}

criterion_group! {
    name = kos_benches;
    config = Criterion::default().sample_size(50);
    targets = kos
}

criterion_group! {
    name = ferret_benches;
    config = Criterion::default().sample_size(10);
    targets = ferret
}

criterion_group! {
    name = ideal_cot_benches;
    config = Criterion::default().sample_size(50);
    targets = ideal_cot_cmp
}

criterion_group! {
    name = ideal_ot_benches;
    config = Criterion::default().sample_size(50);
    targets = ideal_ot_cmp
}

criterion_group! {
    name = ideal_rot_benches;
    config = Criterion::default().sample_size(50);
    targets = ideal_rot_cmp
}

criterion_main!(chou_orlandi_benches, kos_benches, ferret_benches, ideal_cot_benches, ideal_ot_benches, ideal_rot_benches);
