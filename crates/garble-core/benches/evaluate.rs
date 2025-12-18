//! Benchmarks for half-gates evaluation.
//!
//! Run with: `cargo bench -p mpz-garble-core --bench evaluate`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mpz_circuits::AES128;
use mpz_garble_core::{Key, half_gates};
use mpz_memory_core::correlated::Delta;
use rand::{Rng, SeedableRng, rngs::StdRng};

// Gate count thresholds
const THRESHOLDS: &[(u64, &str)] = &[(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")];

fn bench_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate");
    group.sample_size(10);
    let circuit = &*AES128;

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);

    // Prepare inputs
    let inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();
    let choices: Vec<bool> = (0..256).map(|_| rng.random()).collect();
    let eval_inputs: Vec<_> = inputs
        .iter()
        .zip(&choices)
        .map(|(k, &c)| k.auth(c, &delta))
        .collect();

    let gates_per_circuit = circuit.and_count() as u64;

    for &(threshold, name) in THRESHOLDS {
        let iterations = threshold.div_ceil(gates_per_circuit) as usize;
        let actual_gates = iterations as u64 * gates_per_circuit;

        // Pre-generate garbled circuits (single gates)
        let mut gb = half_gates::Garbler::default();
        let all_gates: Vec<Vec<_>> = (0..iterations)
            .map(|_| {
                let mut iter = gb.generate(circuit, delta, &inputs).unwrap();
                let gates: Vec<_> = iter.by_ref().collect();
                let _ = iter.finish().unwrap();
                gates
            })
            .collect();

        // Pre-generate garbled circuits (batched)
        let mut gb = half_gates::Garbler::default();
        let all_batches: Vec<Vec<_>> = (0..iterations)
            .map(|_| {
                let mut iter = gb.generate_batched(circuit, delta, &inputs).unwrap();
                let batches: Vec<_> = iter.by_ref().collect();
                let _ = iter.finish().unwrap();
                batches
            })
            .collect();

        group.throughput(Throughput::Elements(actual_gates));

        // Iterator-based (one gate at a time)
        group.bench_function(BenchmarkId::new("iter", name), |b| {
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for gates in &all_gates {
                    let mut consumer = ev.evaluate(circuit, &eval_inputs).unwrap();
                    for gate in gates {
                        consumer.next(*gate);
                    }
                    black_box(consumer.finish().unwrap());
                }
            })
        });

        // Batched (multiple gates at a time)
        group.bench_function(BenchmarkId::new("batched", name), |b| {
            let mut ev = half_gates::Evaluator::default();
            b.iter(|| {
                for batches in &all_batches {
                    let mut consumer = ev.evaluate_batched(circuit, &eval_inputs).unwrap();
                    for batch in batches {
                        consumer.next(batch.clone());
                    }
                    black_box(consumer.finish().unwrap());
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_evaluate);
criterion_main!(benches);
