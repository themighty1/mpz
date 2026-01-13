//! Benchmark for sum_slots_to_slot operation using pre-generated keys from disk.
//!
//! Run with: cargo bench -p mpz-justvengers-core --bench sum_slots_to_slot_bench
//!
//! First generate fixtures: cargo run -p mpz-justvengers-core --release --example generate_bgv_fixture_binary

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use mpz_justvengers_core::{RnsKeyPair, RnsCiphertext, RnsGaloisKeys, RnsSecretKey, RnsPublicKey};

/// Load keys from fixture directory.
fn load_keys_from_fixture() -> (RnsKeyPair, RnsGaloisKeys) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = PathBuf::from(manifest_dir).parent().unwrap().parent().unwrap().to_path_buf();
    let fixture_dir = workspace_root.join("bgv_fixtures");

    let sk_bytes = fs::read(fixture_dir.join("secret_key.bin"))
        .expect("Failed to read secret_key.bin - run generate_bgv_fixture_binary first");
    let pk_bytes = fs::read(fixture_dir.join("public_key.bin"))
        .expect("Failed to read public_key.bin");
    let gks_bytes = fs::read(fixture_dir.join("galois_keys.bin"))
        .expect("Failed to read galois_keys.bin");

    let sk: RnsSecretKey = bincode::deserialize(&sk_bytes)
        .expect("Failed to deserialize secret key");
    let pk: RnsPublicKey = bincode::deserialize(&pk_bytes)
        .expect("Failed to deserialize public key");
    let galois_keys: RnsGaloisKeys = bincode::deserialize(&gks_bytes)
        .expect("Failed to deserialize Galois keys");

    println!("Loaded keys from disk:");
    println!("  Secret key: {} bytes", sk_bytes.len());
    println!("  Public key: {} bytes", pk_bytes.len());
    println!("  Galois keys: {} bytes ({} keys)", gks_bytes.len(), galois_keys.num_keys());

    (RnsKeyPair { sk, pk }, galois_keys)
}

/// Benchmark sum_slots_to_slot for a single ciphertext.
fn bench_sum_slots_to_slot(c: &mut Criterion) {
    let (keypair, galois_keys) = load_keys_from_fixture();

    const N: usize = 8192;

    // Create slot values (small values to avoid overflow)
    let slots: Vec<u64> = (0..N).map(|i| (i % 100 + 1) as u64).collect();

    // Encrypt once in offline phase
    let mut rng = rand::rng();
    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    println!("Encrypted ciphertext with {} slots", N);

    let mut group = c.benchmark_group("sum_slots_to_slot");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Benchmark sum_slots_to_slot (parallel first for faster feedback)
    #[cfg(feature = "rayon")]
    group.bench_with_input(
        BenchmarkId::new("parallel", format!("n={}", N)),
        &(&ct, &galois_keys),
        |b, (ct, gks)| {
            b.iter(|| {
                ct.sum_slots_to_slot_parallel(black_box(gks), black_box(0), black_box(N))
            });
        },
    );

    // Benchmark sum_slots_to_slot (sequential)
    group.bench_with_input(
        BenchmarkId::new("sequential", format!("n={}", N)),
        &(&ct, &galois_keys),
        |b, (ct, gks)| {
            b.iter(|| {
                ct.sum_slots_to_slot(black_box(gks), black_box(0), black_box(N))
            });
        },
    );

    group.finish();

    // Verify correctness
    let expected_sum: u64 = slots.iter().sum();
    let result = ct.sum_slots_to_slot(&galois_keys, 0, N);
    let decrypted = result.decrypt_slots(&keypair.sk);

    println!("Correctness check:");
    println!("  Expected sum: {}", expected_sum);
    println!("  Decrypted slot 0: {}", decrypted[0]);
    println!("  Slot 1 (should be 0): {}", decrypted[1]);

    assert_eq!(decrypted[0], expected_sum, "sum_slots_to_slot produced wrong result");
    assert_eq!(decrypted[1], 0, "slot 1 should be masked to 0");
    println!("Correctness verified!");
}

/// Benchmark multiple sum_slots_to_slot calls (simulating JV commit pattern).
fn bench_sum_slots_to_slot_multiple(c: &mut Criterion) {
    let (keypair, galois_keys) = load_keys_from_fixture();

    const N: usize = 8192;
    const NUM_POLYS: usize = 10; // Simulate 10 wire polynomials

    // Create multiple ciphertexts (one per polynomial)
    let mut rng = rand::rng();
    let cts: Vec<RnsCiphertext> = (0..NUM_POLYS)
        .map(|poly_idx| {
            let slots: Vec<u64> = (0..N).map(|i| ((i + poly_idx * 100) % 100 + 1) as u64).collect();
            RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng)
        })
        .collect();

    println!("Encrypted {} ciphertexts with {} slots each", NUM_POLYS, N);

    let mut group = c.benchmark_group("sum_slots_to_slot_multiple");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    // Benchmark: for each polynomial, sum_slots_to_slot to different target slots
    // This is the pattern used in JV commit
    group.bench_with_input(
        BenchmarkId::new("sequential", format!("n={}_polys={}", N, NUM_POLYS)),
        &(&cts, &galois_keys),
        |b, (cts, gks)| {
            b.iter(|| {
                let mut accumulated: Option<RnsCiphertext> = None;
                for (i, ct) in cts.iter().enumerate() {
                    let ct_summed = ct.sum_slots_to_slot(black_box(gks), black_box(i), black_box(N));
                    accumulated = Some(match accumulated {
                        None => ct_summed,
                        Some(acc) => acc.add(&ct_summed),
                    });
                }
                black_box(accumulated)
            });
        },
    );

    #[cfg(feature = "rayon")]
    group.bench_with_input(
        BenchmarkId::new("parallel", format!("n={}_polys={}", N, NUM_POLYS)),
        &(&cts, &galois_keys),
        |b, (cts, gks)| {
            b.iter(|| {
                let mut accumulated: Option<RnsCiphertext> = None;
                for (i, ct) in cts.iter().enumerate() {
                    let ct_summed = ct.sum_slots_to_slot_parallel(black_box(gks), black_box(i), black_box(N));
                    accumulated = Some(match accumulated {
                        None => ct_summed,
                        Some(acc) => acc.add(&ct_summed),
                    });
                }
                black_box(accumulated)
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_sum_slots_to_slot, bench_sum_slots_to_slot_multiple);
criterion_main!(benches);
