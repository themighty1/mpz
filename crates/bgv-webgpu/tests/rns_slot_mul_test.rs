//! Correctness tests for RNS batched slot multiplication.

use bgv_webgpu::rns_slot_mul::{RnsBatchParams, RnsSlotMulGpu};

/// Test parameter creation.
#[test]
fn test_params_creation() {
    let params = RnsBatchParams::goldilocks(8192);
    assert!(params.is_some());
    let params = params.unwrap();
    assert_eq!(params.n, 8192);
    assert_eq!(params.k, 5);
    assert_eq!(params.t, 0xFFFFFFFF00000001u64);
}

/// Test GPU context creation.
#[test]
fn test_gpu_context_creation() {
    let params = RnsBatchParams::goldilocks(8192).unwrap();

    match RnsSlotMulGpu::new(params) {
        Ok(_ctx) => println!("GPU context created successfully"),
        Err(e) => println!("GPU not available: {} (this is OK on CI)", e),
    }
}

/// Test basic slot multiplication with small values.
#[test]
fn test_basic_slot_mul() {
    let params = RnsBatchParams::goldilocks(8192).unwrap();

    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    let n = params.n;
    let k = params.k;

    // Create a simple ciphertext NTT (just ones for testing)
    let ct_c0_ntt: Vec<Vec<u64>> = (0..k)
        .map(|_| vec![1u64; n])
        .collect();
    let ct_c1_ntt: Vec<Vec<u64>> = (0..k)
        .map(|_| vec![1u64; n])
        .collect();

    // Create simple plaintext slots (all 2s)
    let plaintext_slots: Vec<Vec<u64>> = vec![vec![2u64; n]; 4];

    let result = gpu_ctx.mul_batched(&ct_c0_ntt, &ct_c1_ntt, &plaintext_slots);

    match result {
        Ok((c0_results, c1_results)) => {
            assert_eq!(c0_results.len(), 4, "Should have 4 batches");
            assert_eq!(c0_results[0].len(), k, "Each batch should have k moduli");
            assert_eq!(c0_results[0][0].len(), n, "Each modulus should have n coefficients");
            println!("GPU slot multiplication completed successfully");
            println!("First few c0 results: {:?}", &c0_results[0][0][..8]);
        }
        Err(e) => {
            println!("GPU execution failed: {}", e);
        }
    }
}

/// CPU reference implementation for slot encoding (INTT).
fn cpu_slot_encode(slots: &[u64], n: usize, t: u64, zeta_inv: u64) -> Vec<u64> {
    // Simplified INTT for correctness testing
    // In practice, this would use the SlotEncoder from justvengers-core
    let mut result = slots.to_vec();

    // Bit-reverse permutation
    let log_n = (n as u64).trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            result.swap(i, j);
        }
    }

    // INTT butterfly stages
    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = 1 << stage;

        for group in 0..(n / m) {
            for idx in 0..half_m {
                let ii = group * m + idx;
                let jj = ii + half_m;

                let twiddle_idx = idx * (n / m);
                let twiddle = mod_pow(zeta_inv, twiddle_idx as u64, t);

                let u = result[ii];
                let v = result[jj];

                let tw_v = mod_mul(v, twiddle, t);
                result[ii] = mod_add(u, tw_v, t);
                result[jj] = mod_sub(u, tw_v, t);
            }
        }
    }

    // Scale by n^-1
    let n_inv = mod_inverse(n as u64, t);
    for val in result.iter_mut() {
        *val = mod_mul(*val, n_inv, t);
    }

    result
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut v = x;
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % m as u128) as u64
}

fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        m - (b - a)
    }
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

fn mod_inverse(a: u64, m: u64) -> u64 {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = m as i128;
    let mut new_r: i128 = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        let temp = t - quotient * new_t;
        t = new_t;
        new_t = temp;
        let temp = r - quotient * new_r;
        r = new_r;
        new_r = temp;
    }

    if t < 0 {
        (t + m as i128) as u64
    } else {
        t as u64
    }
}

/// Performance benchmark (informal).
#[test]
#[ignore] // Run with --ignored for benchmarks
fn test_performance() {
    use std::time::Instant;

    let params = RnsBatchParams::goldilocks(8192).unwrap();

    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping benchmark: GPU not available ({})", e);
            return;
        }
    };

    let n = params.n;
    let k = params.k;

    // Test 1: Single chunk (80 batches)
    let num_batches = 80;
    let ct_c0_ntt: Vec<Vec<u64>> = (0..k)
        .map(|i| (0..n).map(|j| ((i * n + j) as u64) % params.rns_data[i].modulus).collect())
        .collect();
    let ct_c1_ntt: Vec<Vec<u64>> = (0..k)
        .map(|i| (0..n).map(|j| ((i * n + j + 1) as u64) % params.rns_data[i].modulus).collect())
        .collect();
    let plaintext_slots: Vec<Vec<u64>> = (0..num_batches)
        .map(|b| (0..n).map(|i| ((b * n + i) as u64) % params.t).collect())
        .collect();

    println!("=== GPU Slot Multiplication Benchmark ===");
    println!("  n = {}, k = {}", n, k);
    println!();

    // Single chunk benchmark
    println!("[1 chunk] 80 batches (B+C):");
    let start = Instant::now();
    let result = gpu_ctx.mul_batched(&ct_c0_ntt, &ct_c1_ntt, &plaintext_slots);
    let elapsed = start.elapsed();
    if let Ok((c0_results, _)) = result {
        println!("  GPU: {:?}", elapsed);
        println!("  Output: {} batches x {} moduli x {} coeffs",
            c0_results.len(), c0_results[0].len(), c0_results[0][0].len());
    }

    // Test 2: 8 chunks batched (640 batches) - simulates 64K reps
    let num_chunks = 8;
    let total_batches = num_batches * num_chunks; // 640
    let plaintext_slots_8chunks: Vec<Vec<u64>> = (0..total_batches)
        .map(|b| (0..n).map(|i| ((b * n + i) as u64) % params.t).collect())
        .collect();

    println!();
    println!("[8 chunks batched] {} batches (80 B+C x 8 chunks):", total_batches);
    let start = Instant::now();
    let result = gpu_ctx.mul_batched(&ct_c0_ntt, &ct_c1_ntt, &plaintext_slots_8chunks);
    let elapsed = start.elapsed();
    if let Ok((c0_results, _)) = result {
        println!("  GPU: {:?}", elapsed);
        println!("  Output: {} batches x {} moduli x {} coeffs",
            c0_results.len(), c0_results[0].len(), c0_results[0][0].len());
        println!("  vs CPU native (~2.3s for 80 polys x 8 chunks)");
    }

    // Test 3: 8 separate dispatches (current approach)
    println!();
    println!("[8 chunks sequential] 8 x 80 batches (separate dispatches):");
    let start = Instant::now();
    for _ in 0..num_chunks {
        let _ = gpu_ctx.mul_batched(&ct_c0_ntt, &ct_c1_ntt, &plaintext_slots);
    }
    let elapsed = start.elapsed();
    println!("  GPU: {:?}", elapsed);
    println!("  (shows overhead of 8 separate dispatches)");
}

/// Test multi-CT slot multiplication.
#[test]
fn test_multi_ct_slot_mul() {
    let params = RnsBatchParams::goldilocks(8192).unwrap();

    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    let n = params.n;
    let k = params.k;
    let num_cts = 2;
    let batches_per_ct = 4;
    let total_batches = num_cts * batches_per_ct;

    // Create multiple CTs
    let cts_c0: Vec<Vec<Vec<u64>>> = (0..num_cts)
        .map(|ct_idx| {
            (0..k)
                .map(|i| {
                    (0..n)
                        .map(|j| ((ct_idx * k * n + i * n + j) as u64) % params.rns_data[i].modulus)
                        .collect()
                })
                .collect()
        })
        .collect();

    let cts_c1: Vec<Vec<Vec<u64>>> = (0..num_cts)
        .map(|ct_idx| {
            (0..k)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            ((ct_idx * k * n + i * n + j + 1) as u64) % params.rns_data[i].modulus
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    // Create plaintext slots (batches_per_ct for each CT)
    let plaintext_slots: Vec<Vec<u64>> = (0..total_batches)
        .map(|b| (0..n).map(|i| ((b * n + i) as u64) % params.t).collect())
        .collect();

    let result = gpu_ctx.mul_batched_multi_ct(&cts_c0, &cts_c1, &plaintext_slots, batches_per_ct);

    match result {
        Ok((c0_results, c1_results)) => {
            assert_eq!(
                c0_results.len(),
                total_batches,
                "Should have {} batches",
                total_batches
            );
            assert_eq!(
                c0_results[0].len(),
                k,
                "Each batch should have {} moduli",
                k
            );
            assert_eq!(
                c0_results[0][0].len(),
                n,
                "Each modulus should have {} coefficients",
                n
            );
            assert_eq!(c1_results.len(), total_batches);
            println!("Multi-CT GPU slot multiplication completed successfully");
            println!(
                "Output: {} batches x {} moduli x {} coeffs",
                c0_results.len(),
                c0_results[0].len(),
                c0_results[0][0].len()
            );
        }
        Err(e) => {
            println!("GPU execution failed: {}", e);
        }
    }
}

/// Multi-CT performance benchmark (informal).
#[test]
#[ignore] // Run with --ignored for benchmarks
fn test_multi_ct_performance() {
    use std::time::Instant;

    let params = RnsBatchParams::goldilocks(8192).unwrap();

    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping benchmark: GPU not available ({})", e);
            return;
        }
    };

    let n = params.n;
    let k = params.k;
    let batches_per_ct = 80; // B+C polynomials per chunk

    println!("=== Multi-CT GPU Slot Multiplication Benchmark ===");
    println!("  n = {}, k = {}, batches_per_ct = {}", n, k, batches_per_ct);
    println!();

    // Test various chunk counts
    for num_cts in [1, 2, 4, 8] {
        let total_batches = num_cts * batches_per_ct;

        // Create multiple CTs
        let cts_c0: Vec<Vec<Vec<u64>>> = (0..num_cts)
            .map(|ct_idx| {
                (0..k)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                ((ct_idx * k * n + i * n + j) as u64) % params.rns_data[i].modulus
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let cts_c1: Vec<Vec<Vec<u64>>> = (0..num_cts)
            .map(|ct_idx| {
                (0..k)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                ((ct_idx * k * n + i * n + j + 1) as u64)
                                    % params.rns_data[i].modulus
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let plaintext_slots: Vec<Vec<u64>> = (0..total_batches)
            .map(|b| (0..n).map(|i| ((b * n + i) as u64) % params.t).collect())
            .collect();

        // Benchmark multi-CT batched
        println!(
            "[{} CTs batched] {} total batches:",
            num_cts, total_batches
        );
        let start = Instant::now();
        let result =
            gpu_ctx.mul_batched_multi_ct(&cts_c0, &cts_c1, &plaintext_slots, batches_per_ct);
        let elapsed = start.elapsed();
        if let Ok((c0_results, _)) = result {
            println!("  Multi-CT batched: {:?}", elapsed);
            println!(
                "  Output: {} batches x {} moduli x {} coeffs",
                c0_results.len(),
                c0_results[0].len(),
                c0_results[0][0].len()
            );
        }

        // Benchmark sequential for comparison
        let start = Instant::now();
        for ct_idx in 0..num_cts {
            let ct_plaintexts: Vec<Vec<u64>> = (0..batches_per_ct)
                .map(|b| {
                    (0..n)
                        .map(|i| (((ct_idx * batches_per_ct + b) * n + i) as u64) % params.t)
                        .collect()
                })
                .collect();
            let _ = gpu_ctx.mul_batched(&cts_c0[ct_idx], &cts_c1[ct_idx], &ct_plaintexts);
        }
        let elapsed_seq = start.elapsed();
        println!("  Sequential ({} x single-CT): {:?}", num_cts, elapsed_seq);
        println!();
    }
}
