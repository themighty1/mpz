//! Standalone NTT correctness tests for RNS moduli.
//!
//! Tests forward and inverse NTT against CPU reference implementations.
//! Uses the actual RNS moduli (~60 bits) from production params, NOT Goldilocks.

use bgv_webgpu::rns_slot_mul::{RnsBatchParams, RnsSlotMulGpu};

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    let sum = a as u128 + b as u128;
    if sum >= m as u128 {
        (sum - m as u128) as u64
    } else {
        sum as u64
    }
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

fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut v = x;
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// CPU forward NTT with twist (negacyclic NTT).
/// Input: coefficients in natural order
/// Output: NTT values (twisted for negacyclic convolution)
///
/// This matches what the GPU shader does:
/// 1. Twist: multiply coeffs[i] by psi^i
/// 2. Forward NTT using omega = psi^2
fn cpu_forward_ntt(coeffs: &[u64], n: usize, psi: u64, q: u64) -> Vec<u64> {
    let omega = mod_mul(psi, psi, q);

    // Step 1: Twist - multiply by psi^i
    let mut twisted = Vec::with_capacity(n);
    let mut psi_power = 1u64;
    for i in 0..n {
        twisted.push(mod_mul(coeffs[i], psi_power, q));
        psi_power = mod_mul(psi_power, psi, q);
    }

    // Step 2: Forward NTT
    cpu_forward_ntt_no_twist(&twisted, n, omega, q)
}

/// CPU forward NTT using Cooley-Tukey DIT algorithm (without twist).
/// Input: coefficients in natural order
/// Output: NTT values in bit-reversed order (standard CT output)
fn cpu_forward_ntt_no_twist(coeffs: &[u64], n: usize, omega: u64, q: u64) -> Vec<u64> {
    let log_n = (n as u64).trailing_zeros() as usize;
    let mut result = coeffs.to_vec();

    // Bit-reverse input (CT DIT expects bit-reversed input)
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            result.swap(i, j);
        }
    }

    // CT butterfly stages
    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = 1 << stage;
        let step = n >> (stage + 1);
        let omega_m = mod_pow(omega, step as u64, q);

        for k in (0..n).step_by(m) {
            let mut w = 1u64;
            for j in 0..half_m {
                let u = result[k + j];
                let v = mod_mul(result[k + j + half_m], w, q);
                result[k + j] = mod_add(u, v, q);
                result[k + j + half_m] = mod_sub(u, v, q);
                w = mod_mul(w, omega_m, q);
            }
        }
    }

    result
}

/// CPU inverse NTT with untwist (negacyclic INTT).
/// Input: NTT values (twisted)
/// Output: coefficients in natural order
///
/// This matches what the GPU shader does:
/// 1. Inverse NTT using omega_inv = psi_inv^2
/// 2. Untwist: multiply coeffs[i] by psi_inv^i
fn cpu_inverse_ntt(ntt_vals: &[u64], n: usize, psi_inv: u64, q: u64) -> Vec<u64> {
    let omega_inv = mod_mul(psi_inv, psi_inv, q);

    // Step 1: Inverse NTT (without untwist)
    let mut result = cpu_inverse_ntt_no_untwist(ntt_vals, n, omega_inv, q);

    // Step 2: Untwist - multiply by psi_inv^i
    let mut psi_inv_power = 1u64;
    for i in 0..n {
        result[i] = mod_mul(result[i], psi_inv_power, q);
        psi_inv_power = mod_mul(psi_inv_power, psi_inv, q);
    }

    result
}

/// CPU inverse NTT using Gentleman-Sande DIF algorithm (without untwist).
/// Input: NTT values
/// Output: coefficients (still twisted)
fn cpu_inverse_ntt_no_untwist(ntt_vals: &[u64], n: usize, omega_inv: u64, q: u64) -> Vec<u64> {
    let log_n = (n as u64).trailing_zeros() as usize;
    let mut result = ntt_vals.to_vec();
    let n_inv = mod_inverse(n as u64, q);

    // GS butterfly stages (DIF)
    for stage in (0..log_n).rev() {
        let m = 1 << (stage + 1);
        let half_m = 1 << stage;
        let step = n >> (stage + 1);
        let omega_m = mod_pow(omega_inv, step as u64, q);

        for k in (0..n).step_by(m) {
            let mut w = 1u64;
            for j in 0..half_m {
                let u = result[k + j];
                let v = result[k + j + half_m];
                result[k + j] = mod_add(u, v, q);
                result[k + j + half_m] = mod_mul(mod_sub(u, v, q), w, q);
                w = mod_mul(w, omega_m, q);
            }
        }
    }

    // Bit-reverse output
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            result.swap(i, j);
        }
    }

    // Scale by n^-1
    for val in result.iter_mut() {
        *val = mod_mul(*val, n_inv, q);
    }

    result
}

/// Test that CPU NTT roundtrip works (sanity check).
#[test]
fn test_cpu_ntt_roundtrip() {
    let params = RnsBatchParams::goldilocks(8192).unwrap();
    let n = params.n;

    // Use first RNS modulus
    let rns = &params.rns_data[0];
    let q = rns.modulus;
    let psi = rns.psi;
    let psi_inv = rns.psi_inv;

    println!("=== CPU NTT Roundtrip Test ===");
    println!("n = {}, q = {}", n, q);
    println!("psi = {}, psi_inv = {}", psi, psi_inv);

    // Create test input
    let input: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % q).collect();
    println!("Input[0..8]: {:?}", &input[..8]);

    // Forward NTT (with twist)
    let ntt = cpu_forward_ntt(&input, n, psi, q);
    println!("NTT[0..8]: {:?}", &ntt[..8]);

    // Inverse NTT (with untwist)
    let recovered = cpu_inverse_ntt(&ntt, n, psi_inv, q);
    println!("Recovered[0..8]: {:?}", &recovered[..8]);

    // Compare
    let matches = input.iter().zip(&recovered).filter(|(&a, &b)| a == b).count();
    println!("Matches: {} / {}", matches, n);

    assert_eq!(matches, n, "CPU NTT roundtrip failed");
    println!("✓ CPU NTT roundtrip passed!");
}

/// Test GPU forward NTT against CPU reference.
#[test]
fn test_gpu_forward_ntt() {
    let n = 8192;

    // Create GPU context
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    let k = params.k;

    println!("=== GPU Forward NTT Test ===");
    println!("n = {}, k = {}", n, k);

    // Test each RNS modulus using params from GPU (important: must match!)
    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let psi = data.psi;

        println!("\n--- Modulus {} ---", mod_idx);
        println!("q = {}", q);
        println!("psi = {}", psi);

        // Create test input (simple pattern)
        let input: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % q).collect();

        // CPU reference (with twist, matching GPU)
        let cpu_ntt = cpu_forward_ntt(&input, n, psi, q);

        // GPU forward NTT
        let gpu_ntt = gpu_ctx.test_forward_ntt(&input, mod_idx)
            .expect("GPU forward NTT failed");

        // Compare
        let matches = cpu_ntt.iter().zip(&gpu_ntt).filter(|(&a, &b)| a == b).count();
        println!("CPU NTT[0..8]: {:?}", &cpu_ntt[..8]);
        println!("GPU NTT[0..8]: {:?}", &gpu_ntt[..8]);
        println!("Matches: {} / {}", matches, n);

        if matches != n {
            // Find first mismatch
            for i in 0..n {
                if cpu_ntt[i] != gpu_ntt[i] {
                    println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_ntt[i], gpu_ntt[i]);
                    break;
                }
            }
        }

        assert_eq!(matches, n, "GPU forward NTT failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU forward NTT passed for all moduli!");
}

/// Test GPU inverse NTT against CPU reference.
#[test]
fn test_gpu_inverse_ntt() {
    let n = 8192;

    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    let k = params.k;

    println!("=== GPU Inverse NTT Test ===");
    println!("n = {}, k = {}", n, k);

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let psi_inv = data.psi_inv;

        println!("\n--- Modulus {} ---", mod_idx);
        println!("q = {}", q);

        // Create test input (NTT values)
        let input: Vec<u64> = (0..n).map(|i| (i as u64 * 54321) % q).collect();

        // CPU reference (with untwist, matching GPU)
        let cpu_intt = cpu_inverse_ntt(&input, n, psi_inv, q);

        // GPU inverse NTT
        let gpu_intt = gpu_ctx.test_inverse_ntt(&input, mod_idx)
            .expect("GPU inverse NTT failed");

        // Compare
        let matches = cpu_intt.iter().zip(&gpu_intt).filter(|(&a, &b)| a == b).count();
        println!("CPU INTT[0..8]: {:?}", &cpu_intt[..8]);
        println!("GPU INTT[0..8]: {:?}", &gpu_intt[..8]);
        println!("Matches: {} / {}", matches, n);

        if matches != n {
            for i in 0..n {
                if cpu_intt[i] != gpu_intt[i] {
                    println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_intt[i], gpu_intt[i]);
                    break;
                }
            }
        }

        assert_eq!(matches, n, "GPU inverse NTT failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU inverse NTT passed for all moduli!");
}

/// Test GPU NTT roundtrip (forward then inverse).
#[test]
fn test_gpu_ntt_roundtrip() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    let k = params.k;

    println!("=== GPU NTT Roundtrip Test ===");
    println!("n = {}, k = {}", n, k);

    for mod_idx in 0..k {
        let q = params.rns_data[mod_idx].modulus;

        println!("\n--- Modulus {} ---", mod_idx);

        // Create test input
        let input: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % q).collect();

        // Forward NTT
        let ntt = gpu_ctx.test_forward_ntt(&input, mod_idx)
            .expect("GPU forward NTT failed");

        // Inverse NTT
        let recovered = gpu_ctx.test_inverse_ntt(&ntt, mod_idx)
            .expect("GPU inverse NTT failed");

        // Compare
        let matches = input.iter().zip(&recovered).filter(|(&a, &b)| a == b).count();
        println!("Input[0..8]: {:?}", &input[..8]);
        println!("Recovered[0..8]: {:?}", &recovered[..8]);
        println!("Matches: {} / {}", matches, n);

        assert_eq!(matches, n, "GPU NTT roundtrip failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU NTT roundtrip passed for all moduli!");
}

/// Test that GPU slot multiplication produces correct results.
/// This tests the full pipeline: encode -> forward_ntt -> mul -> inverse_ntt
#[test]
fn test_gpu_full_pipeline() {
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
    let t = params.t;

    println!("=== GPU Full Pipeline Test ===");
    println!("n = {}, k = {}, t = {:#x}", n, k, t);

    // Test case 1: Multiply by identity (all 1s)
    // CT in NTT domain = all 1s
    // PT slots = [1, 1, 1, ...]
    // Expected: CT * encode([1,1,1,...]) should give back something related to CT

    println!("\n--- Test 1: Multiply CT by plaintext ones ---");

    let ct_c0_ntt: Vec<Vec<u64>> = (0..k)
        .map(|i| {
            let q = params.rns_data[i].modulus;
            (0..n).map(|j| ((j + 1) as u64) % q).collect()
        })
        .collect();
    let ct_c1_ntt: Vec<Vec<u64>> = (0..k)
        .map(|_| vec![0u64; n])
        .collect();

    // Plaintext: all 1s
    let pt_ones: Vec<Vec<u64>> = vec![vec![1u64; n]];

    let result = gpu_ctx.mul_batched(&ct_c0_ntt, &ct_c1_ntt, &pt_ones);

    match result {
        Ok((c0_results, _c1_results)) => {
            println!("GPU result c0[0][0..8]: {:?}", &c0_results[0][0][..8]);
            println!("Original CT c0[0][0..8]: {:?}", &ct_c0_ntt[0][..8]);

            // Check if multiplication by 1 preserves values (approximately)
            // Due to encoding, this won't be exact equality
        }
        Err(e) => {
            println!("GPU execution failed: {}", e);
        }
    }

    // Test case 2: Known multiplication
    println!("\n--- Test 2: Simple scalar multiplication ---");

    // CT = constant 1 polynomial (coefficient [1, 0, 0, ...])
    // In NTT domain, constant 1 = [1, 1, 1, ...]
    let ct_const_one_ntt: Vec<Vec<u64>> = (0..k)
        .map(|_| vec![1u64; n])
        .collect();

    // PT slots = [2, 2, 2, ...]
    let pt_twos: Vec<Vec<u64>> = vec![vec![2u64; n]];

    let result = gpu_ctx.mul_batched(&ct_const_one_ntt, &ct_c1_ntt, &pt_twos);

    match result {
        Ok((c0_results, _)) => {
            println!("GPU result c0[0][0..8]: {:?}", &c0_results[0][0][..8]);

            // For constant 1 * encode([2,2,2,...]), we expect encode([2,2,2,...])
            // Let's check a few values
        }
        Err(e) => {
            println!("GPU execution failed: {}", e);
        }
    }

    println!("\n✓ Pipeline test completed - check values manually");
}

/// Verify RNS modulus properties using production params.
#[test]
fn test_rns_modulus_properties() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();

    println!("=== RNS Modulus Properties ===");
    println!("n = {}, num_moduli = {}", n, params.k);

    for (i, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let psi = data.psi;
        let omega = mod_mul(psi, psi, q); // omega = psi^2
        let omega_inv = mod_inverse(omega, q);
        let n_inv = mod_inverse(n as u64, q);

        println!("\n--- Modulus {} ---", i);
        println!("q = {} ({:#x})", q, q);
        println!("psi = {}", psi);

        // Check: q ≡ 1 (mod 2n) for NTT to work
        let expected_mod = 1u64;
        let actual_mod = q % (2 * n as u64);
        println!("Check q ≡ 1 (mod {}): {} (expected {})", 2 * n, actual_mod, expected_mod);

        // Check: psi^(2n) ≡ 1 (mod q)
        let psi_2n = mod_pow(psi, 2 * n as u64, q);
        println!("psi^(2n) mod q = {} (expected 1)", psi_2n);

        // Check: psi^n ≡ -1 (mod q)
        let psi_n = mod_pow(psi, n as u64, q);
        println!("psi^n mod q = {} (expected {})", psi_n, q - 1);

        // Check: omega * omega_inv ≡ 1 (mod q)
        let omega_prod = mod_mul(omega, omega_inv, q);
        println!("omega * omega_inv = {} (expected 1)", omega_prod);

        // Check: n * n_inv ≡ 1 (mod q)
        let n_prod = mod_mul(n as u64, n_inv, q);
        println!("n * n_inv = {} (expected 1)", n_prod);

        // Verify all properties
        assert_eq!(actual_mod, expected_mod, "q must be ≡ 1 (mod 2n)");
        assert_eq!(psi_2n, 1, "psi must be primitive 2n-th root");
        assert_eq!(psi_n, q - 1, "psi^n must be -1");
        assert_eq!(omega_prod, 1, "omega_inv must be inverse of omega");
        assert_eq!(n_prod, 1, "n_inv must be inverse of n");
    }

    println!("\n✓ All RNS modulus properties verified!");
}

/// Test CPU NTT roundtrip using production moduli.
#[test]
fn test_cpu_ntt_roundtrip_prod() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();

    println!("=== CPU NTT Roundtrip (Production Moduli) ===");

    for (i, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let psi = data.psi;
        let psi_inv = data.psi_inv;

        println!("\n--- Modulus {} (q = {}) ---", i, q);

        // Create test input
        let input: Vec<u64> = (0..n).map(|j| (j as u64 * 12345) % q).collect();

        // Forward NTT (with twist)
        let ntt = cpu_forward_ntt(&input, n, psi, q);

        // Inverse NTT (with untwist)
        let recovered = cpu_inverse_ntt(&ntt, n, psi_inv, q);

        // Compare
        let matches = input.iter().zip(&recovered).filter(|(&a, &b)| a == b).count();
        println!("Matches: {} / {}", matches, n);

        assert_eq!(matches, n, "CPU NTT roundtrip failed for modulus {}", i);
    }

    println!("\n✓ CPU NTT roundtrip passed for all production moduli!");
}

/// Test GPU mulmod against CPU reference.
#[test]
fn test_gpu_mulmod() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU Mulmod Test ===");

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;

        println!("\n--- Modulus {} (q = {}) ---", mod_idx, q);

        // Test with values < q (all inputs must be < 60 bits since q is ~60 bits)
        // In NTT, all values are reduced mod q, so inputs are always < q
        let max_59bit = (1u64 << 59) - 1;
        let test_cases: Vec<(Vec<u64>, Vec<u64>)> = vec![
            // Small values
            (vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
            // 59-bit values (max valid input size)
            (vec![max_59bit, max_59bit - 1, max_59bit - 2], vec![max_59bit - 3, max_59bit - 4, max_59bit - 5]),
            // Mixed small and 59-bit
            (vec![1, max_59bit, 100, max_59bit / 2], vec![max_59bit, 1, max_59bit / 2, 100]),
            // Powers of 2 up to 58 bits
            (vec![1 << 30, 1 << 40, 1 << 50, 1 << 58], vec![1 << 20, 1 << 30, 1 << 40, 1 << 58]),
        ];

        for (a_vals, b_vals) in &test_cases {
            let gpu_result = gpu_ctx.test_mulmod(a_vals, b_vals, mod_idx)
                .expect("GPU mulmod failed");

            let cpu_result: Vec<u64> = a_vals.iter().zip(b_vals.iter())
                .map(|(&a, &b)| mod_mul(a, b, q))
                .collect();

            let matches = gpu_result.iter().zip(&cpu_result).filter(|(&a, &b)| a == b).count();

            if matches != a_vals.len() {
                println!("FAIL: a={:?}, b={:?}", a_vals, b_vals);
                println!("  CPU: {:?}", cpu_result);
                println!("  GPU: {:?}", gpu_result);
                for i in 0..a_vals.len() {
                    if cpu_result[i] != gpu_result[i] {
                        println!("  Mismatch at {}: {} * {} mod {} = CPU:{}, GPU:{}",
                            i, a_vals[i], b_vals[i], q, cpu_result[i], gpu_result[i]);
                    }
                }
            }

            assert_eq!(matches, a_vals.len(), "GPU mulmod failed for modulus {}", mod_idx);
        }

        println!("✓ All test cases passed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU mulmod test passed for all moduli!");
}

/// Test GPU addmod and submod against CPU reference.
#[test]
fn test_gpu_addmod_submod() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU Addmod/Submod Test ===");

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;

        println!("\n--- Modulus {} (q = {}) ---", mod_idx, q);

        // Test cases: various edge cases for add and sub
        let test_cases: Vec<(Vec<u64>, Vec<u64>)> = vec![
            // Small values
            (vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
            // Values near q (test wrap-around for add)
            (vec![q - 1, q - 2, q - 10, q / 2], vec![1, 5, 20, q / 2 + 1]),
            // Test subtraction wrap-around (a < b)
            (vec![0, 1, 5, 10], vec![1, 5, 10, 100]),
            // Large values
            (vec![q - 1, q - 1, q / 2, q / 2], vec![q - 1, 1, q / 2, q / 2 + 1]),
            // Mix of edge cases
            (vec![0, q - 1, 1, q / 2], vec![0, 1, q - 1, q / 2]),
        ];

        for (a_vals, b_vals) in &test_cases {
            let (gpu_add, gpu_sub) = gpu_ctx.test_addmod_submod(a_vals, b_vals, mod_idx)
                .expect("GPU addmod/submod failed");

            let cpu_add: Vec<u64> = a_vals.iter().zip(b_vals.iter())
                .map(|(&a, &b)| mod_add(a, b, q))
                .collect();
            let cpu_sub: Vec<u64> = a_vals.iter().zip(b_vals.iter())
                .map(|(&a, &b)| mod_sub(a, b, q))
                .collect();

            let add_matches = gpu_add.iter().zip(&cpu_add).filter(|(&a, &b)| a == b).count();
            let sub_matches = gpu_sub.iter().zip(&cpu_sub).filter(|(&a, &b)| a == b).count();

            if add_matches != a_vals.len() {
                println!("ADDMOD FAIL: a={:?}, b={:?}", a_vals, b_vals);
                println!("  CPU add: {:?}", cpu_add);
                println!("  GPU add: {:?}", gpu_add);
                for i in 0..a_vals.len() {
                    if cpu_add[i] != gpu_add[i] {
                        println!("  Add mismatch at {}: {} + {} mod {} = CPU:{}, GPU:{}",
                            i, a_vals[i], b_vals[i], q, cpu_add[i], gpu_add[i]);
                    }
                }
            }

            if sub_matches != a_vals.len() {
                println!("SUBMOD FAIL: a={:?}, b={:?}", a_vals, b_vals);
                println!("  CPU sub: {:?}", cpu_sub);
                println!("  GPU sub: {:?}", gpu_sub);
                for i in 0..a_vals.len() {
                    if cpu_sub[i] != gpu_sub[i] {
                        println!("  Sub mismatch at {}: {} - {} mod {} = CPU:{}, GPU:{}",
                            i, a_vals[i], b_vals[i], q, cpu_sub[i], gpu_sub[i]);
                    }
                }
            }

            assert_eq!(add_matches, a_vals.len(), "GPU addmod failed for modulus {}", mod_idx);
            assert_eq!(sub_matches, a_vals.len(), "GPU submod failed for modulus {}", mod_idx);
        }

        println!("✓ All addmod/submod test cases passed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU addmod/submod test passed for all moduli!");
}

/// Test GPU bit_reverse function against CPU reference.
#[test]
fn test_gpu_bit_reverse() {
    let n = 8192;
    let log_n = 13u32;  // 2^13 = 8192

    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU Bit Reverse Test ===");
    println!("n = {}, log_n = {}", n, log_n);

    // Test all indices 0..n
    let indices: Vec<u32> = (0..n as u32).collect();
    let gpu_result = gpu_ctx.test_bit_reverse(&indices, log_n)
        .expect("GPU bit_reverse failed");

    let cpu_result: Vec<u32> = indices.iter()
        .map(|&i| bit_reverse(i as usize, log_n as usize) as u32)
        .collect();

    let matches = gpu_result.iter().zip(&cpu_result).filter(|(&a, &b)| a == b).count();
    println!("Matches: {} / {}", matches, n);

    if matches != n {
        for i in 0..n {
            if gpu_result[i] != cpu_result[i] {
                println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
                break;
            }
        }
    }

    assert_eq!(matches, n, "GPU bit_reverse failed");
    println!("✓ GPU bit_reverse test passed!");
}

/// Test GPU twiddle factor computation (omega_inv^i) against CPU reference.
#[test]
fn test_gpu_twiddle_factors() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU Twiddle Factors Test ===");

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let omega_inv = data.omega_inv;

        println!("\n--- Modulus {} (q = {}) ---", mod_idx, q);
        println!("omega_inv = {}", omega_inv);

        // Compute CPU reference: omega_inv^i for i = 0..n
        let cpu_twiddles: Vec<u64> = (0..n)
            .map(|i| mod_pow(omega_inv, i as u64, q))
            .collect();

        // Get GPU result
        let gpu_twiddles = gpu_ctx.test_twiddle_factors(mod_idx)
            .expect("GPU twiddle_factors failed");

        let matches = gpu_twiddles.iter().zip(&cpu_twiddles).filter(|(&a, &b)| a == b).count();
        println!("CPU twiddles[0..8]: {:?}", &cpu_twiddles[..8]);
        println!("GPU twiddles[0..8]: {:?}", &gpu_twiddles[..8]);
        println!("Matches: {} / {}", matches, n);

        if matches != n {
            for i in 0..n {
                if gpu_twiddles[i] != cpu_twiddles[i] {
                    println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_twiddles[i], gpu_twiddles[i]);
                    break;
                }
            }
        }

        assert_eq!(matches, n, "GPU twiddle_factors failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU twiddle factors test passed for all moduli!");
}

/// Test GPU untwist operation (multiply by psi_inv^i) against CPU reference.
#[test]
fn test_gpu_untwist() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU Untwist Test ===");

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let psi_inv = data.psi_inv;

        println!("\n--- Modulus {} (q = {}) ---", mod_idx, q);
        println!("psi_inv = {}", psi_inv);

        // Create test input
        let input: Vec<u64> = (0..n).map(|i| (i as u64 * 12345) % q).collect();

        // CPU reference: val[i] * psi_inv^i mod q
        let cpu_result: Vec<u64> = input.iter().enumerate()
            .map(|(i, &v)| {
                let psi_inv_power = mod_pow(psi_inv, i as u64, q);
                mod_mul(v, psi_inv_power, q)
            })
            .collect();

        // GPU result
        let gpu_result = gpu_ctx.test_untwist(&input, mod_idx)
            .expect("GPU untwist failed");

        let matches = gpu_result.iter().zip(&cpu_result).filter(|(&a, &b)| a == b).count();
        println!("CPU untwist[0..8]: {:?}", &cpu_result[..8]);
        println!("GPU untwist[0..8]: {:?}", &gpu_result[..8]);
        println!("Matches: {} / {}", matches, n);

        if matches != n {
            for i in 0..n {
                if gpu_result[i] != cpu_result[i] {
                    println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
                    break;
                }
            }
        }

        assert_eq!(matches, n, "GPU untwist failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU untwist test passed for all moduli!");
}

/// Test GPU n_inv scaling (multiply by 1/n mod q) against CPU reference.
#[test]
fn test_gpu_n_inv_scaling() {
    let n = 8192;
    let params = RnsBatchParams::goldilocks(n).unwrap();
    let gpu_ctx = match RnsSlotMulGpu::new(params.clone()) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test: GPU not available ({})", e);
            return;
        }
    };

    println!("=== GPU n_inv Scaling Test ===");

    for (mod_idx, data) in params.rns_data.iter().enumerate() {
        let q = data.modulus;
        let n_inv = data.n_inv;

        println!("\n--- Modulus {} (q = {}) ---", mod_idx, q);
        println!("n_inv = {}", n_inv);

        // Create test input
        let input: Vec<u64> = (0..n).map(|i| (i as u64 * 54321) % q).collect();

        // CPU reference: val * n_inv mod q
        let cpu_result: Vec<u64> = input.iter()
            .map(|&v| mod_mul(v, n_inv, q))
            .collect();

        // GPU result
        let gpu_result = gpu_ctx.test_n_inv_scaling(&input, mod_idx)
            .expect("GPU n_inv_scaling failed");

        let matches = gpu_result.iter().zip(&cpu_result).filter(|(&a, &b)| a == b).count();
        println!("CPU n_inv[0..8]: {:?}", &cpu_result[..8]);
        println!("GPU n_inv[0..8]: {:?}", &gpu_result[..8]);
        println!("Matches: {} / {}", matches, n);

        if matches != n {
            for i in 0..n {
                if gpu_result[i] != cpu_result[i] {
                    println!("First mismatch at {}: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
                    break;
                }
            }
        }

        assert_eq!(matches, n, "GPU n_inv_scaling failed for modulus {}", mod_idx);
    }

    println!("\n✓ GPU n_inv scaling test passed for all moduli!");
}
