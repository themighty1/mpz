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

/// Benchmark sum_slots + masking + 80 CT additions.
///
/// Measures the full operation:
/// 1. sum_slots: sum all 8192 slots (13 rotations)
/// 2. mask: zero out all slots except slot 1
/// 3. add: add 80 pre-prepared ciphertexts with values in slots 2-81
///
/// All 80 ciphertexts are generated once outside the benchmark and reused.
fn bench_sum_slots_masked(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;
    let slots: Vec<u64> = (0..n as u64).map(|i| i % 1000).collect();
    let expected_sum: u64 = slots.iter().sum();
    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    // Pre-create mask: 1 in slot 1, 0 elsewhere
    let mut mask = vec![0u64; n];
    mask[1] = 1;

    // Pre-create 80 ciphertexts with values in slots 2-81 (not timed)
    let num_cts = 80;
    println!("Generating {} ciphertexts for benchmark (one-time cost)...", num_cts);
    let mut additional_cts = Vec::with_capacity(num_cts);
    let mut expected_slot_values = vec![0u64; 2 + num_cts];
    expected_slot_values[1] = expected_sum;

    for slot_idx in 2..=(1 + num_cts) {
        let value = (slot_idx * 1000 + 123) as u64;
        expected_slot_values[slot_idx] = value;

        let mut slot_vals = vec![0u64; n];
        slot_vals[slot_idx] = value;
        let ct_additional = RnsCiphertext::encrypt_slots(&keypair.pk, &slot_vals, &mut rng);
        additional_cts.push(ct_additional);
    }
    println!("Done generating ciphertexts.");

    let mut group = c.benchmark_group("sum_slots_masked_80cts");
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("cpu", format!("n={}", n)),
        &(&ct, &galois_keys, &mask, &additional_cts),
        |b, (ct, gks, mask, additional_cts)| {
            b.iter(|| {
                // Sum all slots
                let summed = ct.sum_slots(black_box(gks));
                // Mask to keep only slot 1
                let masked = summed.mul_plaintext_slots(black_box(mask));
                // Add all 80 pre-prepared ciphertexts
                let mut result = masked;
                for ct_add in additional_cts.iter() {
                    result = result.add(black_box(ct_add));
                }
                black_box(result)
            });
        },
    );

    group.finish();

    // Verify correctness (not timed): run one more iteration and check result
    let summed = ct.sum_slots(&galois_keys);
    let masked = summed.mul_plaintext_slots(&mask);
    let mut result = masked;
    for ct_add in additional_cts.iter() {
        result = result.add(ct_add);
    }
    let decrypted = result.decrypt_slots(&keypair.sk);

    assert_eq!(
        decrypted[0], 0,
        "CORRECTNESS CHECK FAILED: slot 0 should be 0, got {}",
        decrypted[0]
    );
    assert_eq!(
        decrypted[1], expected_sum,
        "CORRECTNESS CHECK FAILED: slot 1 should be {}, got {}",
        expected_sum, decrypted[1]
    );
    for slot_idx in 2..=(1 + num_cts) {
        assert_eq!(
            decrypted[slot_idx], expected_slot_values[slot_idx],
            "CORRECTNESS CHECK FAILED: slot {} should be {}, got {}",
            slot_idx, expected_slot_values[slot_idx], decrypted[slot_idx]
        );
    }
    for i in (2 + num_cts)..n {
        assert_eq!(
            decrypted[i], 0,
            "CORRECTNESS CHECK FAILED: slot {} should be 0, got {}",
            i, decrypted[i]
        );
    }
    println!("Correctness verified: slot[0]=0, slot[1]={} (sum), slots[2-81] have values, others=0",
             decrypted[1]);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_single_rotation, bench_sum_slots, bench_sum_slots_masked
}
criterion_main!(benches);
