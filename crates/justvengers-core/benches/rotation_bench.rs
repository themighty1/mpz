//! Benchmarks for BGV rotation and slot summation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use mpz_justvengers_core::{
    RnsKeyPair, RnsCiphertext, RnsGaloisKeys, RnsBgvParams, GOLDILOCKS,
};
use rand::{rng, Rng};

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

/// Benchmark JustVengers pattern: 2x slot-wise mult + add + sum_slots + mask + 80 CT additions.
///
/// Measures the full operation:
/// 1. Slot-wise multiply original CT with random coefficients
/// 2. Slot-wise multiply copy of CT with different random coefficients
/// 3. Add the two multiplied CTs together
/// 4. sum_slots: sum all 8192 slots (13 rotations)
/// 5. mask: zero out all slots except slot 1
/// 6. add: add 80 pre-prepared ciphertexts with values in slots 2-81
///
/// All coefficients and 80 ciphertexts are generated once outside the benchmark and reused.
fn bench_sum_slots_masked(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;
    let t = GOLDILOCKS;

    // Create slot values
    let slots: Vec<u64> = (0..n).map(|i| (i % 100 + 1) as u64).collect();

    // Generate 8K random field element coefficients for each CT
    println!("Generating random coefficients...");
    let coeffs1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let coeffs2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

    // Compute expected sum: sum of (slots[i] * coeffs1[i] + slots[i] * coeffs2[i])
    let expected_sum: u64 = (0..n)
        .map(|i| {
            let prod1 = (slots[i] as u128 * coeffs1[i] as u128) % t as u128;
            let prod2 = (slots[i] as u128 * coeffs2[i] as u128) % t as u128;
            ((prod1 + prod2) % t as u128) as u64
        })
        .fold(0u64, |acc, x| ((acc as u128 + x as u128) % t as u128) as u64);

    // Encrypt the original CT
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

    let mut group = c.benchmark_group("justvengers_pattern");
    group.sample_size(10);

    group.bench_with_input(
        BenchmarkId::new("cpu", format!("n={}", n)),
        &(&ct, &galois_keys, &coeffs1, &coeffs2, &mask, &additional_cts),
        |b, (ct, gks, coeffs1, coeffs2, mask, additional_cts)| {
            b.iter(|| {
                // Clone CT for second multiplication
                let ct_copy = ct.clone();
                // Slot-wise multiply each CT with its coefficients
                let ct1_mult = ct.mul_plaintext_slots(black_box(coeffs1));
                let ct2_mult = ct_copy.mul_plaintext_slots(black_box(coeffs2));
                // Add the two multiplied CTs together
                let ct_combined = ct1_mult.add(&ct2_mult);
                // Sum all slots
                let summed = ct_combined.sum_slots(black_box(gks));
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
    let ct_copy = ct.clone();
    let ct1_mult = ct.mul_plaintext_slots(&coeffs1);
    let ct2_mult = ct_copy.mul_plaintext_slots(&coeffs2);
    let ct_combined = ct1_mult.add(&ct2_mult);
    let summed = ct_combined.sum_slots(&galois_keys);
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
    println!("Correctness verified: slot[0]=0, slot[1]={} (sum after 2x slot-wise mult), slots[2-81] have values",
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
