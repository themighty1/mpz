//! End-to-end benchmark for packed evaluation protocol.
//!
//! This benchmarks the rotation-free polynomial evaluation approach:
//! - V: Encrypt powers [Λ^0, ..., Λ^{n-1}] in slots
//! - P: Slot-wise multiply + blind (no Galois keys needed)
//! - P: 2-way pack pairs of rows
//! - V: Decrypt and sum slots in the clear
//!
//! Run with: cargo bench --bench packed_eval_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use mpz_justvengers_core::{
    RnsBgvParams, RnsKeyPair, GOLDILOCKS,
    PackedEncryptedPowers, PackedProverEvaluator,
};
use rand::{rng, Rng};

/// Configure criterion for benchmarks.
fn bench_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(500))
}

/// Benchmark the full packed evaluation protocol for B+C rows.
///
/// Simulates the JV protocol flow:
/// 1. V generates encrypted powers
/// 2. P evaluates all rows with blinding
/// 3. P packs pairs of rows (2-way)
/// 4. V decrypts and sums
fn bench_packed_eval_e2e(c: &mut Criterion) {
    let mut rng = rng();

    // Production params: 8K slots, 4 moduli (~240 bits)
    let params = RnsBgvParams::new(8192, GOLDILOCKS, 4, 3.2);
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n; // 8192 slots
    let t = params.t;

    let mut group = c.benchmark_group("packed_eval_e2e");

    // Test different numbers of rows (B+C)
    for num_rows in [64, 128, 256, 512] {
        // Pre-generate random polynomial coefficients for each row
        let rows: Vec<Vec<u64>> = (0..num_rows)
            .map(|_| (0..n).map(|_| rng.random::<u64>() % t).collect())
            .collect();

        // Pre-generate VOLE blinders (one per row)
        let vole_blinders: Vec<u64> = (0..num_rows)
            .map(|_| rng.random::<u64>() % t)
            .collect();

        // Random evaluation point
        let lambda: u64 = rng.random::<u64>() % t;

        group.bench_function(BenchmarkId::new("full_protocol", num_rows), |b| {
            b.iter(|| {
                // === V's Setup ===
                // Generate encrypted powers [Λ^0, ..., Λ^{n-1}]
                let enc_powers = PackedEncryptedPowers::generate(
                    black_box(&keypair.pk),
                    black_box(lambda),
                    &mut rng,
                );

                // === P's Evaluation ===
                let evaluator = PackedProverEvaluator::new(&enc_powers);

                // Evaluate all rows with blinding and 2-way packing
                let packed_cts = evaluator.evaluate_all_rows(
                    black_box(&rows),
                    black_box(&vole_blinders),
                    &mut rng,
                );

                // === V's Verification ===
                // Decrypt and sum each packed ciphertext
                let mut results = Vec::with_capacity(packed_cts.len() * 2);
                for packed_ct in &packed_cts {
                    let slots = packed_ct.decrypt_slots(black_box(&keypair.sk));
                    // Sum all slots to get f(Λ) - u
                    let sum: u64 = slots.iter()
                        .fold(0u128, |acc, &s| (acc + s as u128) % t as u128) as u64;
                    results.push(sum);
                }

                black_box(results)
            });
        });

        // Benchmark just P's evaluation (without V's setup/verification)
        let enc_powers = PackedEncryptedPowers::generate(&keypair.pk, lambda, &mut rng);

        group.bench_function(BenchmarkId::new("prover_only", num_rows), |b| {
            b.iter(|| {
                let evaluator = PackedProverEvaluator::new(black_box(&enc_powers));
                let packed_cts = evaluator.evaluate_all_rows(
                    black_box(&rows),
                    black_box(&vole_blinders),
                    &mut rng,
                );
                black_box(packed_cts)
            });
        });

        // Benchmark just V's decryption + summing
        let evaluator = PackedProverEvaluator::new(&enc_powers);
        let packed_cts = evaluator.evaluate_all_rows(&rows, &vole_blinders, &mut rng);

        group.bench_function(BenchmarkId::new("verifier_only", num_rows), |b| {
            b.iter(|| {
                let mut results = Vec::with_capacity(packed_cts.len() * 2);
                for packed_ct in black_box(&packed_cts) {
                    let slots = packed_ct.decrypt_slots(black_box(&keypair.sk));
                    let sum: u64 = slots.iter()
                        .fold(0u128, |acc, &s| (acc + s as u128) % t as u128) as u64;
                    results.push(sum);
                }
                black_box(results)
            });
        });
    }

    group.finish();
}

/// Benchmark encrypted powers generation (V's setup cost).
fn bench_encrypted_powers_gen(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::new(8192, GOLDILOCKS, 4, 3.2);
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let t = params.t;
    let lambda: u64 = rng.random::<u64>() % t;

    let mut group = c.benchmark_group("encrypted_powers");

    group.bench_function("generate_8k", |b| {
        b.iter(|| {
            PackedEncryptedPowers::generate(
                black_box(&keypair.pk),
                black_box(lambda),
                &mut rng,
            )
        });
    });

    group.finish();
}

/// Compare old rotation-based approach vs new packed approach.
///
/// The old approach uses Galois keys and sum_slots_to_slot.
/// The new approach uses slot-wise blinding and verifier summing.
fn bench_comparison(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::new(8192, GOLDILOCKS, 4, 3.2);
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n;
    let t = params.t;

    let mut group = c.benchmark_group("approach_comparison");
    group.sample_size(10);

    let num_rows = 128;

    // Pre-generate data
    let rows: Vec<Vec<u64>> = (0..num_rows)
        .map(|_| (0..n).map(|_| rng.random::<u64>() % t).collect())
        .collect();
    let vole_blinders: Vec<u64> = (0..num_rows)
        .map(|_| rng.random::<u64>() % t)
        .collect();
    let lambda: u64 = rng.random::<u64>() % t;

    // New approach: no Galois keys, verifier sums in clear
    group.bench_function("new_packed_eval", |b| {
        b.iter(|| {
            let enc_powers = PackedEncryptedPowers::generate(&keypair.pk, lambda, &mut rng);
            let evaluator = PackedProverEvaluator::new(&enc_powers);
            let packed_cts = evaluator.evaluate_all_rows(
                black_box(&rows),
                black_box(&vole_blinders),
                &mut rng,
            );

            // V decrypts and sums
            let mut results = Vec::with_capacity(num_rows);
            for packed_ct in &packed_cts {
                let slots = packed_ct.decrypt_slots(&keypair.sk);
                let sum: u64 = slots.iter()
                    .fold(0u128, |acc, &s| (acc + s as u128) % t as u128) as u64;
                results.push(sum);
            }
            black_box(results)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = bench_config();
    targets = bench_packed_eval_e2e, bench_encrypted_powers_gen, bench_comparison
}
criterion_main!(benches);
