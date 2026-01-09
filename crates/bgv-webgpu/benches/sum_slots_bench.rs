//! Benchmarks for GPU-accelerated sum_slots operation.
//!
//! This benchmark measures the performance of summing 8192 slots into a single result
//! using log2(8192) = 13 rotations on the GPU with production Goldilocks parameters.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

use bgv_webgpu::{
    GpuRotationContext, GpuRnsParams, GpuRnsPoly, GpuRnsCiphertext,
    GpuGaloisKey, GpuGaloisKeys, gpu_sum_slots, rotation_exponent,
};

/// Creates mock Galois keys for benchmarking.
/// These are random keys that won't produce correct cryptographic results,
/// but will exercise the same GPU code paths for timing purposes.
fn create_mock_galois_keys(
    ctx: &GpuRotationContext,
    n: usize,
    num_moduli: usize,
    num_limbs: usize,
    digits_per_limb: usize,
) -> GpuGaloisKeys {
    let log_n = (n as f64).log2() as usize;
    let mut keys = Vec::with_capacity(log_n);

    for i in 0..log_n {
        let step = 1 << i;
        let k = rotation_exponent(step, n);

        // Create mock key data (random-ish values for benchmarking)
        let mut keys_b_data = Vec::with_capacity(num_limbs);
        let mut keys_a_data = Vec::with_capacity(num_limbs);

        for limb_idx in 0..num_limbs {
            let mut limb_keys_b = Vec::with_capacity(digits_per_limb);
            let mut limb_keys_a = Vec::with_capacity(digits_per_limb);

            for digit_idx in 0..digits_per_limb {
                // Create mock residues (just use index-based values)
                let residues_b: Vec<Vec<u64>> = (0..num_moduli)
                    .map(|mod_idx| {
                        (0..n)
                            .map(|coeff_idx| {
                                ((limb_idx + digit_idx + mod_idx + coeff_idx + i) % 1000) as u64
                            })
                            .collect()
                    })
                    .collect();

                let residues_a: Vec<Vec<u64>> = (0..num_moduli)
                    .map(|mod_idx| {
                        (0..n)
                            .map(|coeff_idx| {
                                ((limb_idx + digit_idx + mod_idx + coeff_idx + i + 500) % 1000) as u64
                            })
                            .collect()
                    })
                    .collect();

                limb_keys_b.push(residues_b);
                limb_keys_a.push(residues_a);
            }

            keys_b_data.push(limb_keys_b);
            keys_a_data.push(limb_keys_a);
        }

        let galois_key = GpuGaloisKey::from_coefficients(ctx, k, &keys_b_data, &keys_a_data)
            .expect("Failed to create mock Galois key");

        keys.push(galois_key);
    }

    GpuGaloisKeys { keys }
}

/// Creates a mock ciphertext for benchmarking.
fn create_mock_ciphertext(
    ctx: &GpuRotationContext,
    n: usize,
    num_moduli: usize,
) -> GpuRnsCiphertext {
    // Create mock residues with small values
    let c0_residues: Vec<Vec<u64>> = (0..num_moduli)
        .map(|mod_idx| {
            (0..n)
                .map(|coeff_idx| ((mod_idx * n + coeff_idx) % 1000) as u64)
                .collect()
        })
        .collect();

    let c1_residues: Vec<Vec<u64>> = (0..num_moduli)
        .map(|mod_idx| {
            (0..n)
                .map(|coeff_idx| ((mod_idx * n + coeff_idx + 500) % 1000) as u64)
                .collect()
        })
        .collect();

    GpuRnsCiphertext::from_residues(ctx, &c0_residues, &c1_residues)
        .expect("Failed to create mock ciphertext")
}

fn bench_sum_slots(c: &mut Criterion) {
    // Use production Goldilocks parameters: 3 RNS moduli (~180 bit q)
    // This matches CPU benchmark for fair comparison
    let params = GpuRnsParams::goldilocks();
    let n = params.n;
    let num_moduli = params.moduli.len();
    let num_limbs = num_moduli;  // One limb per RNS modulus
    let digits_per_limb = params.digits_per_limb;

    println!("Using Goldilocks params: n={}, num_moduli={}, digits_per_limb={}",
             n, num_moduli, digits_per_limb);

    let ctx = match GpuRotationContext::new(params) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("GPU context creation failed: {:?}", e);
            eprintln!("This may be due to missing NTT support for the chosen primes.");
            eprintln!("Skipping sum_slots benchmarks.");
            return;
        }
    };

    println!("GPU context created successfully");

    // Create mock data
    let galois_keys = create_mock_galois_keys(&ctx, n, num_moduli, num_limbs, digits_per_limb);
    let ciphertext = create_mock_ciphertext(&ctx, n, num_moduli);

    println!("Mock Galois keys created: {} keys for log2({}) = {} rotations",
             galois_keys.keys.len(), n, (n as f64).log2() as usize);

    let mut group = c.benchmark_group("gpu_sum_slots");
    group.sample_size(10);  // Fewer samples since GPU operations are expensive
    group.measurement_time(Duration::from_secs(30));

    group.bench_with_input(
        BenchmarkId::new("sum_slots", format!("n={}_moduli={}", n, num_moduli)),
        &(&ctx, &ciphertext, &galois_keys),
        |b, (ctx, ct, keys)| {
            b.iter(|| {
                gpu_sum_slots(black_box(ctx), black_box(ct), black_box(keys)).unwrap()
            });
        },
    );

    group.finish();
}

/// Benchmark individual operations for comparison
fn bench_individual_ops(c: &mut Criterion) {
    // Use production Goldilocks parameters
    let params = GpuRnsParams::goldilocks();
    let n = params.n;
    let num_moduli = params.moduli.len();

    let ctx = match GpuRotationContext::new(params) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping individual ops benchmarks: {:?}", e);
            return;
        }
    };

    // Create test polynomials with residues for each modulus
    let residues_a: Vec<Vec<u64>> = (0..num_moduli)
        .map(|_| (0..n).map(|i| (i % 1000) as u64).collect())
        .collect();
    let residues_b: Vec<Vec<u64>> = (0..num_moduli)
        .map(|_| (0..n).map(|i| ((i + 500) % 1000) as u64).collect())
        .collect();

    let poly_a = GpuRnsPoly::from_residues(&ctx, &residues_a).unwrap();
    let poly_b = GpuRnsPoly::from_residues(&ctx, &residues_b).unwrap();

    let mut group = c.benchmark_group("gpu_individual_ops");
    group.sample_size(20);

    // Benchmark NTT multiplication
    group.bench_function(
        BenchmarkId::new("ntt_mul", format!("n={}_moduli={}", n, num_moduli)),
        |b| {
            b.iter(|| {
                bgv_webgpu::gpu_ntt_mul(black_box(&ctx), black_box(&poly_a), black_box(&poly_b)).unwrap()
            });
        },
    );

    // Benchmark addition
    group.bench_function(
        BenchmarkId::new("add", format!("n={}_moduli={}", n, num_moduli)),
        |b| {
            b.iter(|| {
                bgv_webgpu::gpu_add(black_box(&ctx), black_box(&poly_a), black_box(&poly_b)).unwrap()
            });
        },
    );

    // Benchmark automorphism (without key-switching)
    group.bench_function(
        BenchmarkId::new("automorphism", format!("n={}_moduli={}", n, num_moduli)),
        |b| {
            b.iter(|| {
                bgv_webgpu::gpu_automorphism(black_box(&ctx), black_box(&poly_a), 5).unwrap()
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_sum_slots, bench_individual_ops);
criterion_main!(benches);
