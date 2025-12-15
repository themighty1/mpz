//! Benchmarks comparing half-gates vs three-halves garbling schemes.
//!
//! Run with: `cargo bench -p mpz-garble-core --bench garble`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mpz_circuits::AES128;
use mpz_core::Block;
use mpz_garble_core::{half_gates, three_halves, Key};
use mpz_memory_core::correlated::Delta;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Benchmark single AES circuit garbling
fn bench_garble_aes(c: &mut Criterion) {
    let mut group = c.benchmark_group("garble_aes128");
    group.throughput(Throughput::Elements(1));

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    // Half-gates inputs
    let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();

    // Three-halves inputs (LSB = 0)
    let th_inputs: Vec<Key> = (0..256)
        .map(|_| {
            let mut block: Block = rng.random();
            block.set_lsb(false);
            block.into()
        })
        .collect();

    group.bench_function("half_gates", |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            let mut iter = gb.generate(&AES128, delta, &hg_inputs).unwrap();
            let _: Vec<_> = iter.by_ref().collect();
            black_box(iter.finish().unwrap())
        })
    });

    group.bench_function("three_halves", |b| {
        let mut gb = three_halves::Garbler::default();
        b.iter(|| {
            let mut bench_rng = StdRng::seed_from_u64(42);
            let mut iter = gb
                .generate(&AES128, delta, &th_inputs, &mut bench_rng)
                .unwrap();
            let _: Vec<_> = iter.by_ref().collect();
            black_box(iter.finish().unwrap())
        })
    });

    group.finish();
}

/// Benchmark single AES circuit evaluation
fn bench_evaluate_aes(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_aes128");
    group.throughput(Throughput::Elements(1));

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    // Half-gates setup
    let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();
    let mut hg_gb = half_gates::Garbler::default();
    let mut hg_iter = hg_gb.generate(&AES128, delta, &hg_inputs).unwrap();
    let hg_gates: Vec<_> = hg_iter.by_ref().collect();
    let _ = hg_iter.finish().unwrap();

    let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
    let hg_eval_inputs: Vec<_> = hg_inputs
        .iter()
        .zip(&choices)
        .map(|(k, &c)| k.auth(c, &delta))
        .collect();

    // Three-halves setup
    let th_inputs: Vec<Key> = (0..256)
        .map(|_| {
            let mut block: Block = rng.random();
            block.set_lsb(false);
            block.into()
        })
        .collect();

    let mut th_gb = three_halves::Garbler::default();
    let mut th_rng = StdRng::seed_from_u64(42);
    let mut th_iter = th_gb
        .generate(&AES128, delta, &th_inputs, &mut th_rng)
        .unwrap();
    let th_gates: Vec<_> = th_iter.by_ref().collect();
    let three_halves::GarblerOutput {
        inputs: input_pairs,
        ..
    } = th_iter.finish().unwrap();

    let th_eval_inputs: Vec<_> = input_pairs
        .iter()
        .zip(&choices)
        .map(|((f, t), &c)| if c { *t } else { *f })
        .collect();

    group.bench_function("half_gates", |b| {
        let mut ev = half_gates::Evaluator::default();
        b.iter(|| {
            let mut consumer = ev.evaluate(&AES128, &hg_eval_inputs).unwrap();
            for gate in &hg_gates {
                consumer.next(*gate);
            }
            black_box(consumer.finish().unwrap())
        })
    });

    group.bench_function("three_halves", |b| {
        let mut ev = three_halves::Evaluator::default();
        b.iter(|| {
            let mut consumer = ev.evaluate(&AES128, &th_eval_inputs).unwrap();
            for gate in &th_gates {
                consumer.next(gate.clone());
            }
            black_box(consumer.finish().unwrap())
        })
    });

    group.finish();
}

/// Benchmark 100 AES circuits (throughput test)
fn bench_100_aes(c: &mut Criterion) {
    const N: usize = 100;

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    // Half-gates inputs
    let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();
    let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
    let hg_eval_inputs: Vec<_> = hg_inputs
        .iter()
        .zip(&choices)
        .map(|(k, &c)| k.auth(c, &delta))
        .collect();

    // Three-halves inputs
    let th_inputs: Vec<Key> = (0..256)
        .map(|_| {
            let mut block: Block = rng.random();
            block.set_lsb(false);
            block.into()
        })
        .collect();

    // Get input pairs for three-halves evaluation
    let mut setup_gb = three_halves::Garbler::default();
    let mut setup_rng = StdRng::seed_from_u64(42);
    let mut setup_iter = setup_gb
        .generate(&AES128, delta, &th_inputs, &mut setup_rng)
        .unwrap();
    let _: Vec<_> = setup_iter.by_ref().collect();
    let three_halves::GarblerOutput {
        inputs: input_pairs,
        ..
    } = setup_iter.finish().unwrap();

    let th_eval_inputs: Vec<_> = input_pairs
        .iter()
        .zip(&choices)
        .map(|((f, t), &c)| if c { *t } else { *f })
        .collect();

    // Pre-generate gates for evaluation benchmarks
    let mut hg_gb = half_gates::Garbler::default();
    let hg_all_gates: Vec<Vec<_>> = (0..N)
        .map(|_| {
            let mut iter = hg_gb.generate(&AES128, delta, &hg_inputs).unwrap();
            let gates: Vec<_> = iter.by_ref().collect();
            let _ = iter.finish().unwrap();
            gates
        })
        .collect();

    let mut th_gb = three_halves::Garbler::default();
    let th_all_gates: Vec<Vec<_>> = (0..N)
        .map(|_| {
            let mut bench_rng = StdRng::seed_from_u64(42);
            let mut iter = th_gb
                .generate(&AES128, delta, &th_inputs, &mut bench_rng)
                .unwrap();
            let gates: Vec<_> = iter.by_ref().collect();
            let _ = iter.finish().unwrap();
            gates
        })
        .collect();

    // === Garble 100x ===
    {
        let mut group = c.benchmark_group("garble_100x_aes128");
        group.throughput(Throughput::Elements(N as u64));

        group.bench_function("half_gates", |b| {
            let mut gb = half_gates::Garbler::default();
            b.iter(|| {
                for _ in 0..N {
                    let mut iter = gb.generate(&AES128, delta, &hg_inputs).unwrap();
                    let _: Vec<_> = iter.by_ref().collect();
                    black_box(iter.finish().unwrap());
                }
            })
        });

        group.bench_function("three_halves", |b| {
            let mut gb = three_halves::Garbler::default();
            b.iter(|| {
                for _ in 0..N {
                    let mut bench_rng = StdRng::seed_from_u64(42);
                    let mut iter = gb
                        .generate(&AES128, delta, &th_inputs, &mut bench_rng)
                        .unwrap();
                    let _: Vec<_> = iter.by_ref().collect();
                    black_box(iter.finish().unwrap());
                }
            })
        });

        group.finish();
    }

    // === Evaluate 100x ===
    {
        let mut group = c.benchmark_group("evaluate_100x_aes128");
        group.throughput(Throughput::Elements(N as u64));

        group.bench_function("half_gates", |b| {
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for gates in &hg_all_gates {
                    let mut consumer = ev.evaluate(&AES128, &hg_eval_inputs).unwrap();
                    for gate in gates {
                        consumer.next(*gate);
                    }
                    black_box(consumer.finish().unwrap());
                }
            })
        });

        group.bench_function("three_halves", |b| {
            let mut ev = three_halves::Evaluator::default();
            b.iter(|| {
                for gates in &th_all_gates {
                    let mut consumer = ev.evaluate(&AES128, &th_eval_inputs).unwrap();
                    for gate in gates {
                        consumer.next(gate.clone());
                    }
                    black_box(consumer.finish().unwrap());
                }
            })
        });

        group.finish();
    }

    // === Garble+Evaluate 100x ===
    {
        let mut group = c.benchmark_group("garble_and_evaluate_100x_aes128");
        group.throughput(Throughput::Elements(N as u64));

        group.bench_function("half_gates", |b| {
            let mut gb = half_gates::Garbler::default();
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for _ in 0..N {
                    let mut gb_iter = gb.generate(&AES128, delta, &hg_inputs).unwrap();
                    let mut ev_consumer = ev.evaluate(&AES128, &hg_eval_inputs).unwrap();
                    for gate in gb_iter.by_ref() {
                        ev_consumer.next(gate);
                    }
                    black_box(gb_iter.finish().unwrap());
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });

        group.bench_function("three_halves", |b| {
            let mut gb = three_halves::Garbler::default();
            let mut ev = three_halves::Evaluator::default();
            b.iter(|| {
                for _ in 0..N {
                    let mut bench_rng = StdRng::seed_from_u64(42);
                    let mut gb_iter = gb
                        .generate(&AES128, delta, &th_inputs, &mut bench_rng)
                        .unwrap();
                    let mut ev_consumer = ev.evaluate(&AES128, &th_eval_inputs).unwrap();
                    for gate in gb_iter.by_ref() {
                        ev_consumer.next(gate);
                    }
                    black_box(gb_iter.finish().unwrap());
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_garble_aes, bench_evaluate_aes, bench_100_aes);
criterion_main!(benches);
