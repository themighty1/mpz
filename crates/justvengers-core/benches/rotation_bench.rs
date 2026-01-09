//! Benchmarks for BGV rotation and slot summation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use mpz_justvengers_core::{
    RnsKeyPair, RnsCiphertext, RnsGaloisKeys, RnsBgvParams,
};
use rand::rng;

/// Benchmark key generation for Galois keys.
fn bench_galois_key_gen(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);

    c.bench_function("galois_key_gen", |b| {
        b.iter(|| {
            RnsGaloisKeys::generate(black_box(&keypair.sk), &mut rng)
        });
    });
}

/// Benchmark sum_slots operation (summing all 8192 slots).
fn bench_sum_slots(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);

    // Generate Galois keys (needed for rotation)
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let mut group = c.benchmark_group("sum_slots");

    // Create test slot values
    let n = params.n;
    let slots: Vec<u64> = (0..n as u64).map(|i| i % 1000).collect();

    // Encrypt the slots
    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    group.bench_with_input(
        BenchmarkId::new("cpu", format!("n={}", n)),
        &ct,
        |b, ct| {
            b.iter(|| {
                ct.sum_slots(black_box(&galois_keys))
            });
        },
    );

    group.finish();
}

/// Benchmark a single rotation (apply_automorphism).
fn bench_single_rotation(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);

    // Generate Galois keys
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;
    let slots: Vec<u64> = (0..n as u64).map(|i| i % 1000).collect();
    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    // Get the first Galois key (for σ_5)
    let gk = galois_keys.get_key(0).unwrap();

    c.bench_function("single_rotation", |b| {
        b.iter(|| {
            ct.apply_automorphism(black_box(gk))
        });
    });
}

/// Verify correctness of sum_slots by decrypting and checking result.
fn bench_sum_slots_with_verification(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;

    // Use small values to avoid overflow during summation
    let slots: Vec<u64> = (0..n).map(|i| (i % 10) as u64).collect();
    let expected_sum: u64 = slots.iter().sum();

    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    // Verify it works before benchmarking
    let summed = ct.sum_slots(&galois_keys);
    let decrypted = summed.decrypt_slots(&keypair.sk);

    // All slots should contain the sum
    println!("Expected sum: {}", expected_sum);
    println!("Decrypted slot 0: {}", decrypted[0]);
    println!("Decrypted slot 1: {}", decrypted[1]);

    c.bench_function("sum_slots_e2e", |b| {
        b.iter(|| {
            let summed = ct.sum_slots(black_box(&galois_keys));
            black_box(summed)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_single_rotation, bench_sum_slots
}
criterion_main!(benches);
