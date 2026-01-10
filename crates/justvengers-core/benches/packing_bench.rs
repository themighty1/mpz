//! Benchmarks for ciphertext-space packing operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use mpz_justvengers_core::{
    RnsKeyPair, RnsCiphertext, RnsBgvParams, GOLDILOCKS,
    CiphertextPacking, SlotPacker,
};
use rand::{rng, Rng};

/// Configure criterion for fast benchmarks (total ~10s).
fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(1))
        .warm_up_time(Duration::from_millis(500))
}

/// Benchmark 4-way packing with 16K slots: encrypt + pack.
fn bench_pack_4way_16k(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks_16k_packed_4();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n;
    let t = params.t;

    let mut group = c.benchmark_group("pack_4way_16k");

    // Pre-generate random slot values
    let v0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

    // Benchmark encryption only (4 ciphertexts)
    group.bench_function(BenchmarkId::new("encrypt_4", n), |b| {
        b.iter(|| {
            let ct0 = RnsCiphertext::encrypt_slots(black_box(&keypair.pk), black_box(&v0), &mut rng);
            let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
            let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
            let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);
            (ct0, ct1, ct2, ct3)
        });
    });

    // Pre-encrypt for packing benchmark
    let ct0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
    let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
    let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
    let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);

    // Benchmark packing only (shift + add)
    group.bench_function(BenchmarkId::new("pack_only", n), |b| {
        b.iter(|| {
            RnsCiphertext::pack_4way(
                black_box(&ct0),
                black_box(&ct1),
                black_box(&ct2),
                black_box(&ct3),
            )
        });
    });

    // Benchmark full pipeline: encrypt + pack
    group.bench_function(BenchmarkId::new("encrypt_and_pack", n), |b| {
        b.iter(|| {
            let ct0 = RnsCiphertext::encrypt_slots(&keypair.pk, black_box(&v0), &mut rng);
            let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
            let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
            let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);
            RnsCiphertext::pack_4way(&ct0, &ct1, &ct2, &ct3)
        });
    });

    group.finish();
}

/// Benchmark 4-way packing with slot-wise multiplication (16K slots).
fn bench_pack_4way_with_mul_16k(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks_16k_packed_4();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n;
    let t = params.t;

    let mut group = c.benchmark_group("pack_4way_mul_16k");

    // Pre-generate random slot values and scalars
    let v0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let s0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let s1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let s2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let s3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

    // Pre-encrypt
    let ct0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
    let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
    let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
    let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);

    // Benchmark slot-wise multiplication only
    group.bench_function(BenchmarkId::new("mul_plaintext_slots_4x", n), |b| {
        b.iter(|| {
            let ct0s = ct0.mul_plaintext_slots(black_box(&s0));
            let ct1s = ct1.mul_plaintext_slots(&s1);
            let ct2s = ct2.mul_plaintext_slots(&s2);
            let ct3s = ct3.mul_plaintext_slots(&s3);
            (ct0s, ct1s, ct2s, ct3s)
        });
    });

    // Pre-compute scaled ciphertexts
    let ct0s = ct0.mul_plaintext_slots(&s0);
    let ct1s = ct1.mul_plaintext_slots(&s1);
    let ct2s = ct2.mul_plaintext_slots(&s2);
    let ct3s = ct3.mul_plaintext_slots(&s3);

    // Benchmark pack after multiplication
    group.bench_function(BenchmarkId::new("pack_after_mul", n), |b| {
        b.iter(|| {
            RnsCiphertext::pack_4way(
                black_box(&ct0s),
                black_box(&ct1s),
                black_box(&ct2s),
                black_box(&ct3s),
            )
        });
    });

    // Benchmark full: mul + pack
    group.bench_function(BenchmarkId::new("mul_and_pack", n), |b| {
        b.iter(|| {
            let ct0s = ct0.mul_plaintext_slots(black_box(&s0));
            let ct1s = ct1.mul_plaintext_slots(&s1);
            let ct2s = ct2.mul_plaintext_slots(&s2);
            let ct3s = ct3.mul_plaintext_slots(&s3);
            RnsCiphertext::pack_4way(&ct0s, &ct1s, &ct2s, &ct3s)
        });
    });

    // Benchmark using SlotPacker helper
    let packer = SlotPacker::new(&keypair.pk);
    group.bench_function(BenchmarkId::new("slot_packer_full", n), |b| {
        b.iter(|| {
            packer.encrypt_pack_4way(
                black_box(&v0), black_box(&s0),
                &v1, &s1,
                &v2, &s2,
                &v3, &s3,
                &mut rng,
            )
        });
    });

    group.finish();
}

/// Benchmark decryption of packed ciphertext (16K slots).
fn bench_decrypt_packed_16k(c: &mut Criterion) {
    let mut rng = rng();
    let params = RnsBgvParams::goldilocks_16k_packed_4();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n;
    let t = params.t;

    let mut group = c.benchmark_group("decrypt_packed_16k");

    // Create packed ciphertext
    let v0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

    let ct0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
    let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
    let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
    let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);
    let packed = RnsCiphertext::pack_4way(&ct0, &ct1, &ct2, &ct3);

    // Standard decryption (mod t)
    group.bench_function(BenchmarkId::new("decrypt_slots", n), |b| {
        b.iter(|| {
            packed.decrypt_slots(black_box(&keypair.sk))
        });
    });

    // Packed decryption (raw values)
    group.bench_function(BenchmarkId::new("decrypt_packed", n), |b| {
        b.iter(|| {
            packed.decrypt_packed(black_box(&keypair.sk))
        });
    });

    group.finish();
}

/// Benchmark 2-way packing with 8K slots (4 ciphertexts, 2-way each sequentially).
///
/// This simulates packing 4 pairs of values using 2-way packing on 8K slots.
/// Uses production params: 8K slots, 4 moduli (~240 bits).
fn bench_pack_2way_8k(c: &mut Criterion) {
    let mut rng = rng();
    // Production params: 8K slots, 4 moduli (~240 bits)
    let params = RnsBgvParams::new(8192, GOLDILOCKS, 4, 3.2);
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let n = params.n;
    let t = params.t;

    let mut group = c.benchmark_group("pack_2way_8k");

    // Pre-generate 8 slot vectors (4 pairs for 2-way packing)
    let v0a: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v0b: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v1a: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v1b: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v2a: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v2b: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v3a: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
    let v3b: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

    // Pre-encrypt all 8 ciphertexts
    let ct0a = RnsCiphertext::encrypt_slots(&keypair.pk, &v0a, &mut rng);
    let ct0b = RnsCiphertext::encrypt_slots(&keypair.pk, &v0b, &mut rng);
    let ct1a = RnsCiphertext::encrypt_slots(&keypair.pk, &v1a, &mut rng);
    let ct1b = RnsCiphertext::encrypt_slots(&keypair.pk, &v1b, &mut rng);
    let ct2a = RnsCiphertext::encrypt_slots(&keypair.pk, &v2a, &mut rng);
    let ct2b = RnsCiphertext::encrypt_slots(&keypair.pk, &v2b, &mut rng);
    let ct3a = RnsCiphertext::encrypt_slots(&keypair.pk, &v3a, &mut rng);
    let ct3b = RnsCiphertext::encrypt_slots(&keypair.pk, &v3b, &mut rng);

    // Benchmark: 4x 2-way packing sequentially
    group.bench_function(BenchmarkId::new("pack_4x_2way_seq", n), |b| {
        b.iter(|| {
            let p0 = RnsCiphertext::pack_2way(black_box(&ct0a), black_box(&ct0b));
            let p1 = RnsCiphertext::pack_2way(&ct1a, &ct1b);
            let p2 = RnsCiphertext::pack_2way(&ct2a, &ct2b);
            let p3 = RnsCiphertext::pack_2way(&ct3a, &ct3b);
            (p0, p1, p2, p3)
        });
    });

    // Benchmark: encrypt + 4x 2-way pack
    group.bench_function(BenchmarkId::new("encrypt_and_pack_4x_2way", n), |b| {
        b.iter(|| {
            let ct0a = RnsCiphertext::encrypt_slots(&keypair.pk, black_box(&v0a), &mut rng);
            let ct0b = RnsCiphertext::encrypt_slots(&keypair.pk, &v0b, &mut rng);
            let ct1a = RnsCiphertext::encrypt_slots(&keypair.pk, &v1a, &mut rng);
            let ct1b = RnsCiphertext::encrypt_slots(&keypair.pk, &v1b, &mut rng);
            let ct2a = RnsCiphertext::encrypt_slots(&keypair.pk, &v2a, &mut rng);
            let ct2b = RnsCiphertext::encrypt_slots(&keypair.pk, &v2b, &mut rng);
            let ct3a = RnsCiphertext::encrypt_slots(&keypair.pk, &v3a, &mut rng);
            let ct3b = RnsCiphertext::encrypt_slots(&keypair.pk, &v3b, &mut rng);

            let p0 = RnsCiphertext::pack_2way(&ct0a, &ct0b);
            let p1 = RnsCiphertext::pack_2way(&ct1a, &ct1b);
            let p2 = RnsCiphertext::pack_2way(&ct2a, &ct2b);
            let p3 = RnsCiphertext::pack_2way(&ct3a, &ct3b);
            (p0, p1, p2, p3)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets = bench_pack_4way_16k, bench_pack_4way_with_mul_16k, bench_decrypt_packed_16k, bench_pack_2way_8k
}
criterion_main!(benches);
