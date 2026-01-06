//! Benchmarks for WRK17 authenticated garbling with Fcp preprocessing.
//!
//! Measures garbler throughput (AND gates/sec) for the authenticated garbling
//! protocol. Preprocessing is done beforehand and not included in timing.
//!
//! Run with: `cargo bench -p mpz-garble-core --bench wrk17_garble`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mpz_circuits::AES128;
use mpz_garble_core::{AuthGen, Key, SSP, bit_shares_from_cot};
use mpz_memory_core::correlated::Delta;
use rand::{Rng, SeedableRng, rngs::StdRng};

// Gate count thresholds
const THRESHOLDS: &[(u64, &str)] = &[(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")];

fn bench_wrk17_garble(c: &mut Criterion) {
    let mut group = c.benchmark_group("wrk17_garble");
    group.sample_size(10);
    let circuit = &*AES128;

    let mut rng = StdRng::seed_from_u64(0);
    let delta_a = Delta::random(&mut rng).set_lsb(true);
    let delta_b = Delta::random(&mut rng).set_lsb(false);

    let input_keys: Vec<Key> = (0..256).map(|_| rng.random()).collect();

    let gates_per_circuit = circuit.and_count() as u64;

    for &(threshold, name) in THRESHOLDS {
        let iterations = threshold.div_ceil(gates_per_circuit) as usize;
        let actual_gates = iterations as u64 * gates_per_circuit;
        let total_and_gates = (circuit.and_count() * iterations) as usize;

        group.throughput(Throughput::Elements(actual_gates));

        // Calculate preprocessing size
        let bucket_size = (SSP as f64 / (circuit.and_count() as f64).log2()).ceil() as usize;
        let num_input_shares = circuit.inputs().len();
        let num_and_shares = total_and_gates * (3 * bucket_size + 1);
        let total_shares = num_input_shares + num_and_shares;

        // Pre-generate all auth bit shares (outside benchmark timing)
        let (gen_all_shares, eval_all_shares) =
            bit_shares_from_cot(total_shares, delta_a, delta_b).unwrap();

        let (gen_input_shares, gen_and_shares) = {
            let (input, and) = gen_all_shares.split_at(num_input_shares);
            (input.to_vec(), and.to_vec())
        };
        let (eval_input_shares, eval_and_shares) = {
            let (input, and) = eval_all_shares.split_at(num_input_shares);
            (input.to_vec(), and.to_vec())
        };

        // Iterator-based (one gate at a time)
        group.bench_function(BenchmarkId::new("iter", name), |b| {
            // Do preprocessing once, outside the benchmark timing
            let mut gb = AuthGen::new(0, bucket_size);
            let mut ev = AuthGen::new(0, bucket_size);

            let (c_gen, mut g_gen) = gb
                .generate_pre_1(circuit, delta_a, &gen_input_shares, &gen_and_shares)
                .unwrap();
            let (c_eval, mut g_eval) = ev
                .generate_pre_1(circuit, delta_b, &eval_input_shares, &eval_and_shares)
                .unwrap();

            let gr_gen = g_eval.clone();
            let gr_eval = g_gen.clone();

            let d_gen = gb
                .generate_pre_2(delta_a, c_gen, &mut g_gen, gr_gen)
                .unwrap();
            let d_eval = ev
                .generate_pre_2(delta_b, c_eval, &mut g_eval, gr_eval)
                .unwrap();

            let dr_gen = d_eval.clone();
            let dr_eval = d_gen.clone();

            let data_gen = gb
                .generate_pre_3(delta_a, &mut g_gen, d_gen, dr_gen)
                .unwrap();
            let data_eval = ev
                .generate_pre_3(delta_b, &mut g_eval, d_eval, dr_eval)
                .unwrap();

            let data_recv_gen = data_eval.clone();
            let data_recv_eval = data_gen.clone();

            gb.generate_pre_4(data_gen, data_recv_gen).unwrap();
            ev.generate_pre_4(data_eval, data_recv_eval).unwrap();

            gb.generate_free(circuit).unwrap();
            ev.generate_free(circuit).unwrap();

            let (_px_gen, _py_gen) = gb.generate_de(circuit).unwrap();
            let (px_eval, py_eval) = ev.generate_de(circuit).unwrap();

            // Now measure only the garbling phase
            b.iter(|| {
                for _ in 0..iterations {
                    let mut iter = gb
                        .generate(
                            circuit,
                            delta_a,
                            &input_keys,
                            px_eval.clone(),
                            py_eval.clone(),
                        )
                        .unwrap();
                    let gates: Vec<_> = iter.by_ref().collect();
                    black_box(gates);
                }
            })
        });

        // Batched (multiple gates at a time)
        group.bench_function(BenchmarkId::new("batched", name), |b| {
            // Do preprocessing once, outside the benchmark timing
            let mut gb = AuthGen::new(0, bucket_size);
            let mut ev = AuthGen::new(0, bucket_size);

            let (c_gen, mut g_gen) = gb
                .generate_pre_1(circuit, delta_a, &gen_input_shares, &gen_and_shares)
                .unwrap();
            let (c_eval, mut g_eval) = ev
                .generate_pre_1(circuit, delta_b, &eval_input_shares, &eval_and_shares)
                .unwrap();

            let gr_gen = g_eval.clone();
            let gr_eval = g_gen.clone();

            let d_gen = gb
                .generate_pre_2(delta_a, c_gen, &mut g_gen, gr_gen)
                .unwrap();
            let d_eval = ev
                .generate_pre_2(delta_b, c_eval, &mut g_eval, gr_eval)
                .unwrap();

            let dr_gen = d_eval.clone();
            let dr_eval = d_gen.clone();

            let data_gen = gb
                .generate_pre_3(delta_a, &mut g_gen, d_gen, dr_gen)
                .unwrap();
            let data_eval = ev
                .generate_pre_3(delta_b, &mut g_eval, d_eval, dr_eval)
                .unwrap();

            let data_recv_gen = data_eval.clone();
            let data_recv_eval = data_gen.clone();

            gb.generate_pre_4(data_gen, data_recv_gen).unwrap();
            ev.generate_pre_4(data_eval, data_recv_eval).unwrap();

            gb.generate_free(circuit).unwrap();
            ev.generate_free(circuit).unwrap();

            let (_px_gen, _py_gen) = gb.generate_de(circuit).unwrap();
            let (px_eval, py_eval) = ev.generate_de(circuit).unwrap();

            // Now measure only the garbling phase
            b.iter(|| {
                for _ in 0..iterations {
                    let mut iter = gb
                        .generate_batched(
                            circuit,
                            delta_a,
                            &input_keys,
                            px_eval.clone(),
                            py_eval.clone(),
                        )
                        .unwrap();
                    let gates: Vec<_> = iter.by_ref().collect();
                    black_box(gates);
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_wrk17_garble);
criterion_main!(benches);
