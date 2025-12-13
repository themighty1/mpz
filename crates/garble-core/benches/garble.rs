use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mpz_circuits::AES128;
use mpz_garble_core::{half_gates, three_halves, Key};
use mpz_memory_core::correlated::Delta;
use rand::{Rng, SeedableRng, rngs::StdRng};
use mpz_core::Block;

fn bench_garble(c: &mut Criterion) {
    let mut group = c.benchmark_group("garble");

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let inputs: Vec<_> = (0..256).map(|_| rng.random()).collect();

    // Half-gates garbling
    group.bench_function(BenchmarkId::new("aes128", "half_gates"), |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            let mut gb_iter = gb.generate(&AES128, delta, &inputs).unwrap();
            let _: Vec<_> = gb_iter.by_ref().collect();
            black_box(gb_iter.finish().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("aes128_batched", "half_gates"), |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            let mut gb_iter = gb.generate_batched(&AES128, delta, &inputs).unwrap();
            let _: Vec<_> = gb_iter.by_ref().collect();
            black_box(gb_iter.finish().unwrap())
        })
    });

    // Three-halves garbling (requires input keys with LSB = 0)
    let three_halves_inputs: Vec<_> = (0..256)
        .map(|_| {
            let mut block: Block = rng.random();
            block.set_lsb(false);
            block.into()
        })
        .collect();

    group.bench_function(BenchmarkId::new("aes128", "three_halves"), |b| {
        let mut gb = three_halves::Garbler::default();
        b.iter(|| {
            let mut bench_rng = StdRng::seed_from_u64(42);
            let mut gb_iter = gb
                .generate(&AES128, delta, &three_halves_inputs, &mut bench_rng)
                .unwrap();
            let _: Vec<_> = gb_iter.by_ref().collect();
            black_box(gb_iter.finish().unwrap())
        })
    });

    group.bench_function(BenchmarkId::new("aes128_batched", "three_halves"), |b| {
        let mut gb = three_halves::Garbler::default();
        b.iter(|| {
            let mut bench_rng = StdRng::seed_from_u64(42);
            let mut gb_iter = gb
                .generate_batched(&AES128, delta, &three_halves_inputs, &mut bench_rng)
                .unwrap();
            let _: Vec<_> = gb_iter.by_ref().collect();
            black_box(gb_iter.finish().unwrap())
        })
    });

    group.finish();
}

fn bench_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate");

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let inputs: Vec<_> = (0..256).map(|_| rng.random()).collect();

    // Half-gates evaluation
    {
        let mut gb = half_gates::Garbler::default();
        let mut gb_iter = gb.generate(&AES128, delta, &inputs).unwrap();
        let gates: Vec<_> = gb_iter.by_ref().collect();
        let _ = gb_iter.finish().unwrap();

        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
        let eval_inputs: Vec<_> = inputs
            .iter()
            .zip(&choices)
            .map(|(input, &choice)| input.auth(choice, &delta))
            .collect();

        group.bench_function(BenchmarkId::new("aes128", "half_gates"), |b| {
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                for gate in &gates {
                    ev_consumer.next(*gate);
                }
                black_box(ev_consumer.finish().unwrap());
            })
        });
    }

    // Three-halves evaluation
    {
        let three_halves_inputs: Vec<_> = (0..256)
            .map(|_| {
                let mut block: Block = rng.random();
                block.set_lsb(false);
                block.into()
            })
            .collect();

        let mut gb = three_halves::Garbler::default();
        let mut bench_rng = StdRng::seed_from_u64(42);
        let mut gb_iter = gb
            .generate(&AES128, delta, &three_halves_inputs, &mut bench_rng)
            .unwrap();
        let gates: Vec<_> = gb_iter.by_ref().collect();
        let three_halves::GarblerOutput {
            inputs: input_pairs,
            ..
        } = gb_iter.finish().unwrap();

        // Select input MACs from input pairs based on random choices
        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
        let eval_inputs: Vec<_> = input_pairs
            .iter()
            .zip(&choices)
            .map(|((false_label, true_label), &choice)| {
                if choice {
                    *true_label
                } else {
                    *false_label
                }
            })
            .collect();

        group.bench_function(BenchmarkId::new("aes128", "three_halves"), |b| {
            let mut ev = three_halves::Evaluator::default();
            b.iter(|| {
                let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                for gate in &gates {
                    ev_consumer.next(gate.clone());
                }
                black_box(ev_consumer.finish().unwrap());
            })
        });
    }

    group.finish();
}

fn bench_chained_aes(c: &mut Criterion) {
    const NUM_CIRCUITS: usize = 100;

    let mut group = c.benchmark_group("chained_100_aes");

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    // Half-gates: garble + evaluate 100 AES circuits
    {
        let inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();
        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
        let eval_inputs: Vec<_> = inputs
            .iter()
            .zip(&choices)
            .map(|(input, &choice)| input.auth(choice, &delta))
            .collect();

        group.bench_function(BenchmarkId::new("garble", "half_gates"), |b| {
            let mut gb = half_gates::Garbler::default();
            b.iter(|| {
                for _ in 0..NUM_CIRCUITS {
                    let mut gb_iter = gb.generate(&AES128, delta, &inputs).unwrap();
                    let _: Vec<_> = gb_iter.by_ref().collect();
                    black_box(gb_iter.finish().unwrap());
                }
            })
        });

        // Pre-garble for evaluation benchmark
        let mut gb = half_gates::Garbler::default();
        let all_gates: Vec<Vec<_>> = (0..NUM_CIRCUITS)
            .map(|_| {
                let mut gb_iter = gb.generate(&AES128, delta, &inputs).unwrap();
                let gates: Vec<_> = gb_iter.by_ref().collect();
                let _ = gb_iter.finish().unwrap();
                gates
            })
            .collect();

        group.bench_function(BenchmarkId::new("evaluate", "half_gates"), |b| {
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for gates in &all_gates {
                    let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                    for gate in gates {
                        ev_consumer.next(*gate);
                    }
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("garble+evaluate", "half_gates"), |b| {
            let mut gb = half_gates::Garbler::default();
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for _ in 0..NUM_CIRCUITS {
                    let mut gb_iter = gb.generate(&AES128, delta, &inputs).unwrap();
                    let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                    for gate in gb_iter.by_ref() {
                        ev_consumer.next(gate);
                    }
                    black_box(gb_iter.finish().unwrap());
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });
    }

    // Three-halves: garble + evaluate 100 AES circuits
    {
        let three_halves_inputs: Vec<_> = (0..256)
            .map(|_| {
                let mut block: Block = rng.random();
                block.set_lsb(false);
                block.into()
            })
            .collect();

        // Pre-garble once to get input pairs for evaluation
        let mut setup_gb = three_halves::Garbler::default();
        let mut setup_rng = StdRng::seed_from_u64(42);
        let mut setup_iter = setup_gb
            .generate(&AES128, delta, &three_halves_inputs, &mut setup_rng)
            .unwrap();
        let _: Vec<_> = setup_iter.by_ref().collect();
        let three_halves::GarblerOutput {
            inputs: input_pairs,
            ..
        } = setup_iter.finish().unwrap();

        let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
        let eval_inputs: Vec<_> = input_pairs
            .iter()
            .zip(&choices)
            .map(|((false_label, true_label), &choice)| {
                if choice {
                    *true_label
                } else {
                    *false_label
                }
            })
            .collect();

        group.bench_function(BenchmarkId::new("garble", "three_halves"), |b| {
            let mut gb = three_halves::Garbler::default();
            b.iter(|| {
                for _ in 0..NUM_CIRCUITS {
                    let mut bench_rng = StdRng::seed_from_u64(42);
                    let mut gb_iter = gb
                        .generate(&AES128, delta, &three_halves_inputs, &mut bench_rng)
                        .unwrap();
                    let _: Vec<_> = gb_iter.by_ref().collect();
                    black_box(gb_iter.finish().unwrap());
                }
            })
        });

        // Pre-garble for evaluation benchmark
        let mut gb = three_halves::Garbler::default();
        let all_gates: Vec<Vec<_>> = (0..NUM_CIRCUITS)
            .map(|_| {
                let mut bench_rng = StdRng::seed_from_u64(42);
                let mut gb_iter = gb
                    .generate(&AES128, delta, &three_halves_inputs, &mut bench_rng)
                    .unwrap();
                let gates: Vec<_> = gb_iter.by_ref().collect();
                let _ = gb_iter.finish().unwrap();
                gates
            })
            .collect();

        group.bench_function(BenchmarkId::new("evaluate", "three_halves"), |b| {
            let mut ev = three_halves::Evaluator::default();
            b.iter(|| {
                for gates in &all_gates {
                    let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                    for gate in gates {
                        ev_consumer.next(gate.clone());
                    }
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });

        group.bench_function(BenchmarkId::new("garble+evaluate", "three_halves"), |b| {
            let mut gb = three_halves::Garbler::default();
            let mut ev = three_halves::Evaluator::default();
            b.iter(|| {
                for _ in 0..NUM_CIRCUITS {
                    let mut bench_rng = StdRng::seed_from_u64(42);
                    let mut gb_iter = gb
                        .generate(&AES128, delta, &three_halves_inputs, &mut bench_rng)
                        .unwrap();
                    let mut ev_consumer = ev.evaluate(&AES128, &eval_inputs).unwrap();
                    for gate in gb_iter.by_ref() {
                        ev_consumer.next(gate);
                    }
                    black_box(gb_iter.finish().unwrap());
                    black_box(ev_consumer.finish().unwrap());
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_garble, bench_evaluate, bench_chained_aes);
criterion_main!(benches);
