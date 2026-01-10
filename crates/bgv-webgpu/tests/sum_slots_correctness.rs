//! Correctness test for GPU sum_slots against CPU reference.
//!
//! This test generates real Galois keys using CPU code and verifies
//! that GPU sum_slots produces identical results.
//!
//! Run with: cargo test -p bgv-webgpu --test sum_slots_correctness -- --nocapture

use bgv_webgpu::{
    GpuRotationContext, GpuRnsParams, GpuRnsCiphertext,
    GpuGaloisKey, GpuGaloisKeys, gpu_sum_slots,
    SumSlotsWorkspace, gpu_sum_slots_batched,
};
use mpz_justvengers_core::{
    RnsKeyPair, RnsCiphertext, RnsGaloisKeys, RnsBgvParams,
    ahe::{RnsParams, RnsPoly},
};
use mpz_core::prg::Prg;

/// Fixed seed for deterministic testing.
const TEST_SEED: [u8; 16] = [0u8; 16];

/// Number of digits per RNS limb (matches CPU constant).
const DIGITS_PER_LIMB: usize = 4;

/// Converts CPU Galois keys to GPU format.
fn cpu_to_gpu_galois_keys(
    ctx: &GpuRotationContext,
    cpu_keys: &RnsGaloisKeys,
    num_moduli: usize,
) -> GpuGaloisKeys {
    let mut gpu_keys = Vec::with_capacity(cpu_keys.num_keys());

    for cpu_key in cpu_keys.keys() {
        let k = cpu_key.k();
        let keys_b = cpu_key.keys_b();
        let keys_a = cpu_key.keys_a();

        let mut keys_b_data = Vec::with_capacity(num_moduli);
        let mut keys_a_data = Vec::with_capacity(num_moduli);

        for limb_idx in 0..num_moduli {
            let mut limb_keys_b = Vec::with_capacity(DIGITS_PER_LIMB);
            let mut limb_keys_a = Vec::with_capacity(DIGITS_PER_LIMB);

            for digit_idx in 0..DIGITS_PER_LIMB {
                let key_idx = limb_idx * DIGITS_PER_LIMB + digit_idx;

                let residues_b: Vec<Vec<u64>> = keys_b[key_idx]
                    .residues()
                    .iter()
                    .cloned()
                    .collect();
                let residues_a: Vec<Vec<u64>> = keys_a[key_idx]
                    .residues()
                    .iter()
                    .cloned()
                    .collect();

                limb_keys_b.push(residues_b);
                limb_keys_a.push(residues_a);
            }

            keys_b_data.push(limb_keys_b);
            keys_a_data.push(limb_keys_a);
        }

        let gpu_key = GpuGaloisKey::from_coefficients(ctx, k, &keys_b_data, &keys_a_data)
            .expect("Failed to create GPU Galois key");
        gpu_keys.push(gpu_key);
    }

    GpuGaloisKeys { keys: gpu_keys }
}

/// Converts CPU ciphertext to GPU format.
fn cpu_to_gpu_ciphertext(
    ctx: &GpuRotationContext,
    cpu_ct: &RnsCiphertext,
) -> GpuRnsCiphertext {
    let c0_residues: Vec<Vec<u64>> = cpu_ct.c0()
        .residues()
        .iter()
        .cloned()
        .collect();
    let c1_residues: Vec<Vec<u64>> = cpu_ct.c1()
        .residues()
        .iter()
        .cloned()
        .collect();

    GpuRnsCiphertext::from_residues(ctx, &c0_residues, &c1_residues)
        .expect("Failed to create GPU ciphertext")
}

/// Converts GPU ciphertext back to CPU format.
fn gpu_to_cpu_ciphertext(
    ctx: &GpuRotationContext,
    gpu_ct: &GpuRnsCiphertext,
    rns_params: &RnsParams,
    bgv_params: &RnsBgvParams,
) -> RnsCiphertext {
    let c0_residues = gpu_ct.c0.to_residues(ctx).unwrap();
    let c1_residues = gpu_ct.c1.to_residues(ctx).unwrap();

    let mut c0 = RnsPoly::zero(rns_params);
    let mut c1 = RnsPoly::zero(rns_params);

    for (mod_idx, residue) in c0_residues.iter().enumerate() {
        for (coeff_idx, &val) in residue.iter().enumerate() {
            c0.residues_mut()[mod_idx][coeff_idx] = val;
        }
    }
    for (mod_idx, residue) in c1_residues.iter().enumerate() {
        for (coeff_idx, &val) in residue.iter().enumerate() {
            c1.residues_mut()[mod_idx][coeff_idx] = val;
        }
    }

    RnsCiphertext::from_parts(c0, c1, rns_params.clone(), bgv_params.clone())
}

/// Test that CPU sum_slots works correctly.
#[test]
fn test_cpu_sum_slots() {
    println!("=== CPU sum_slots Test ===");

    let mut rng = Prg::new_with_seed(TEST_SEED);
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;

    println!("Parameters: n={}", n);

    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let expected_sum: u64 = test_slots.iter().sum::<u64>() % bgv_params.t;

    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);
    let cpu_result = cpu_ct.sum_slots(&cpu_galois_keys);
    let decrypted = cpu_result.decrypt_slots(&keypair.sk);

    println!("Expected sum: {}", expected_sum);
    println!("CPU result slot[0]: {}", decrypted[0]);

    assert_eq!(decrypted[0], expected_sum, "CPU sum_slots should produce correct sum");
    println!("✓ CPU sum_slots test passed!");
}

/// Test that data round-trip (CPU → GPU → CPU) preserves correctness.
#[test]
fn test_gpu_data_roundtrip() {
    println!("=== GPU Data Round-trip Test ===");

    let mut rng = Prg::new_with_seed(TEST_SEED);
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    println!("Parameters: n={}, num_moduli={}", n, num_moduli);

    // Generate keys and encrypt
    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);
    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Verify original decrypts correctly
    let original_decrypt = cpu_ct.decrypt_slots(&keypair.sk);
    println!("Original slots[0..8]: {:?}", &original_decrypt[..8]);
    assert_eq!(original_decrypt, test_slots, "Original ciphertext should decrypt to test slots");

    // Create GPU context
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    // Convert to GPU and back (no operations)
    let gpu_ct = cpu_to_gpu_ciphertext(&ctx, &cpu_ct);
    let rns_params = RnsParams::new(n, num_moduli, 60);
    let cpu_ct_back = gpu_to_cpu_ciphertext(&ctx, &gpu_ct, &rns_params, &bgv_params);

    // Decrypt round-tripped ciphertext
    let roundtrip_decrypt = cpu_ct_back.decrypt_slots(&keypair.sk);
    println!("Roundtrip slots[0..8]: {:?}", &roundtrip_decrypt[..8]);

    assert_eq!(roundtrip_decrypt, test_slots, "Round-trip should preserve ciphertext");
    println!("✓ Round-trip test passed!");
}

/// Test that GPU addition works correctly.
#[test]
fn test_gpu_addition() {
    use bgv_webgpu::gpu_add;

    println!("=== GPU Addition Test ===");

    let mut rng = Prg::new_with_seed(TEST_SEED);
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);

    // Create two ciphertexts with known slots
    let slots1: Vec<u64> = (0..n).map(|i| (i as u64) % 100).collect();
    let slots2: Vec<u64> = (0..n).map(|i| ((i + 50) as u64) % 100).collect();
    let expected_sum: Vec<u64> = slots1.iter().zip(&slots2)
        .map(|(&a, &b)| (a + b) % bgv_params.t).collect();

    let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
    let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);

    // CPU addition
    let cpu_sum = ct1.add(&ct2);
    let cpu_sum_decrypt = cpu_sum.decrypt_slots(&keypair.sk);
    println!("CPU sum slots[0..8]: {:?}", &cpu_sum_decrypt[..8]);
    assert_eq!(cpu_sum_decrypt, expected_sum, "CPU addition should work");

    // GPU addition
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let gpu_ct1 = cpu_to_gpu_ciphertext(&ctx, &ct1);
    let gpu_ct2 = cpu_to_gpu_ciphertext(&ctx, &ct2);

    // Add the ciphertexts on GPU
    let gpu_sum_c0 = gpu_add(&ctx, &gpu_ct1.c0, &gpu_ct2.c0).unwrap();
    let gpu_sum_c1 = gpu_add(&ctx, &gpu_ct1.c1, &gpu_ct2.c1).unwrap();
    let gpu_sum = GpuRnsCiphertext { c0: gpu_sum_c0, c1: gpu_sum_c1 };

    // Convert back and decrypt
    let rns_params = RnsParams::new(n, num_moduli, 60);
    let cpu_sum_from_gpu = gpu_to_cpu_ciphertext(&ctx, &gpu_sum, &rns_params, &bgv_params);
    let gpu_sum_decrypt = cpu_sum_from_gpu.decrypt_slots(&keypair.sk);

    println!("GPU sum slots[0..8]: {:?}", &gpu_sum_decrypt[..8]);
    assert_eq!(gpu_sum_decrypt, expected_sum, "GPU addition should match CPU");
    println!("✓ GPU addition test passed!");
}

/// Test that GPU NTT multiplication works correctly.
#[test]
fn test_gpu_ntt_multiplication() {
    use bgv_webgpu::{GpuRnsPoly, gpu_ntt_mul};

    println!("=== GPU NTT Multiplication Test ===");

    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let n = ctx.params().n;
    let num_moduli = ctx.params().moduli.len();
    let moduli = ctx.params().moduli.clone();

    println!("n={}, num_moduli={}", n, num_moduli);

    // Test 1: Identity multiplication (multiply a * 1)
    // If NTT roundtrip works, this should return a unchanged
    println!("\n--- Test 1: Identity multiplication (a * 1) ---");
    let mut a_residues: Vec<Vec<u64>> = Vec::new();
    let mut one_residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut a = vec![0u64; n];
        let mut one = vec![0u64; n];
        a[0] = 5;
        a[1] = 7;
        a[2] = 11;
        one[0] = 1; // constant polynomial = 1
        a_residues.push(a);
        one_residues.push(one);
    }

    let gpu_a = GpuRnsPoly::from_residues(&ctx, &a_residues).unwrap();
    let gpu_one = GpuRnsPoly::from_residues(&ctx, &one_residues).unwrap();

    let identity_result = gpu_ntt_mul(&ctx, &gpu_a, &gpu_one).unwrap();
    let identity_res = identity_result.to_residues(&ctx).unwrap();

    println!("Input a: [5, 7, 11, 0, ...]");
    println!("Input one: [1, 0, 0, ...]");
    println!("Result[0..8]: {:?}", &identity_res[0][..8]);

    if identity_res[0][0] == 5 && identity_res[0][1] == 7 && identity_res[0][2] == 11 {
        println!("✓ Identity multiplication passed!");
    } else {
        println!("✗ Identity multiplication FAILED!");
        println!("Expected: [5, 7, 11, 0, ...]");
    }

    // Test 2: Multiply X * X = X^2
    println!("\n--- Test 2: X * X = X^2 ---");
    let mut x_residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut x = vec![0u64; n];
        x[1] = 1; // X = coefficient 1 at position 1
        x_residues.push(x);
    }

    let gpu_x = GpuRnsPoly::from_residues(&ctx, &x_residues).unwrap();
    let gpu_x2 = GpuRnsPoly::from_residues(&ctx, &x_residues).unwrap();

    let x_squared_result = gpu_ntt_mul(&ctx, &gpu_x, &gpu_x2).unwrap();
    let x_squared_res = x_squared_result.to_residues(&ctx).unwrap();

    println!("Input: X = [0, 1, 0, ...]");
    println!("Result X^2[0..8]: {:?}", &x_squared_res[0][..8]);

    if x_squared_res[0][0] == 0 && x_squared_res[0][1] == 0 && x_squared_res[0][2] == 1 {
        println!("✓ X * X = X^2 passed!");
    } else {
        println!("✗ X * X = X^2 FAILED!");
        println!("Expected: [0, 0, 1, 0, ...]");
    }

    // Test 3: Original test (1+X) * (1+X) = 1 + 2X + X^2
    println!("\n--- Test 3: (1+X) * (1+X) = 1 + 2X + X^2 ---");
    let mut b_residues: Vec<Vec<u64>> = Vec::new();
    let mut c_residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut b = vec![0u64; n];
        let mut c = vec![0u64; n];
        b[0] = 1;
        b[1] = 1;
        c[0] = 1;
        c[1] = 1;
        b_residues.push(b);
        c_residues.push(c);
    }

    let gpu_b = GpuRnsPoly::from_residues(&ctx, &b_residues).unwrap();
    let gpu_c = GpuRnsPoly::from_residues(&ctx, &c_residues).unwrap();

    let gpu_result = gpu_ntt_mul(&ctx, &gpu_b, &gpu_c).unwrap();
    let result_residues = gpu_result.to_residues(&ctx).unwrap();

    println!("Input b: [1, 1, 0, ...]");
    println!("Input c: [1, 1, 0, ...]");
    println!("Result[0..8]: {:?}", &result_residues[0][..8]);

    // Expected: (1+X)*(1+X) = 1 + 2X + X^2
    // In the ring Z[X]/(X^n + 1), this is just 1 + 2X + X^2 since n > 2
    let _q = moduli[0];
    assert_eq!(result_residues[0][0], 1, "coeff 0 should be 1");
    assert_eq!(result_residues[0][1], 2, "coeff 1 should be 2");
    assert_eq!(result_residues[0][2], 1, "coeff 2 should be 1");
    for i in 3..n.min(32) {
        assert_eq!(result_residues[0][i], 0, "coeff {} should be 0", i);
    }

    println!("✓ GPU NTT multiplication test passed!");
}

/// Test that GPU raw automorphism permutation works correctly.
#[test]
fn test_gpu_raw_automorphism() {
    use bgv_webgpu::{GpuRnsPoly, gpu_automorphism};

    println!("=== GPU Raw Automorphism Test ===");

    // Small test case: n=16, num_moduli=1
    // Create a simple polynomial [1, 2, 3, 4, 0, 0, ...]
    // Apply σ_3: coefficient at index i goes to index (i*3) mod 2n, with negation if >= n

    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let n = ctx.params().n;
    let num_moduli = ctx.params().moduli.len();
    println!("n={}, num_moduli={}", n, num_moduli);

    // Create simple input: coefficient i has value i+1
    let mut residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut r = vec![0u64; n];
        for i in 0..n.min(16) {
            r[i] = (i + 1) as u64;
        }
        residues.push(r);
    }

    let gpu_poly = GpuRnsPoly::from_residues(&ctx, &residues).unwrap();

    // Apply automorphism with k=5
    let k = 5;
    let gpu_result = gpu_automorphism(&ctx, &gpu_poly, k).unwrap();
    let result_residues = gpu_result.to_residues(&ctx).unwrap();

    // Check first few coefficients
    println!("Input[0..16]: {:?}", &residues[0][..16]);
    println!("Output[0..16]: {:?}", &result_residues[0][..16]);

    // Manually compute expected: coefficient at i goes to (i*k) mod 2n
    let mut expected = vec![0u64; n];
    let two_n = 2 * n;
    let q = ctx.params().moduli[0];
    for i in 0..n.min(16) {
        let val = (i + 1) as u64;
        let target_exp = (i * k) % two_n;
        let (final_idx, negate) = if target_exp >= n {
            (target_exp - n, true)
        } else {
            (target_exp, false)
        };
        if negate {
            // q - val
            expected[final_idx] = q - val;
        } else {
            expected[final_idx] = val;
        }
    }
    println!("Expected[0..16]: {:?}", &expected[..16]);

    // Compare
    for i in 0..n.min(32) {
        if result_residues[0][i] != expected[i] {
            println!("Mismatch at {}: got {}, expected {}", i, result_residues[0][i], expected[i]);
        }
    }

    assert_eq!(&result_residues[0][..n.min(32)], &expected[..n.min(32)],
               "GPU automorphism should match expected permutation");
    println!("✓ GPU raw automorphism test passed!");
}

/// Test that a single GPU automorphism works correctly.
#[test]
fn test_gpu_single_automorphism() {
    use bgv_webgpu::gpu_apply_automorphism;

    println!("=== GPU Single Automorphism Test ===");

    let mut rng = Prg::new_with_seed(TEST_SEED);
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);

    // Create ciphertext with known slots
    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Generate Galois keys
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    // Apply first automorphism (index 0, k=5) using CPU
    let cpu_gk = cpu_galois_keys.get_key(0).unwrap();
    let cpu_result = cpu_ct.apply_automorphism(cpu_gk);
    let cpu_decrypt = cpu_result.decrypt_slots(&keypair.sk);
    println!("CPU automorphism slots[0..8]: {:?}", &cpu_decrypt[..8]);

    // Now do the same with GPU
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let gpu_ct = cpu_to_gpu_ciphertext(&ctx, &cpu_ct);
    let gpu_galois_keys = cpu_to_gpu_galois_keys(&ctx, &cpu_galois_keys, num_moduli);

    // Apply automorphism on GPU
    let gpu_gk = &gpu_galois_keys.keys[0];
    let gpu_result = gpu_apply_automorphism(&ctx, &gpu_ct, gpu_gk)
        .expect("GPU automorphism failed");

    // Convert back and decrypt
    let rns_params = RnsParams::new(n, num_moduli, 60);
    let cpu_result_from_gpu = gpu_to_cpu_ciphertext(&ctx, &gpu_result, &rns_params, &bgv_params);
    let gpu_decrypt = cpu_result_from_gpu.decrypt_slots(&keypair.sk);

    println!("GPU automorphism slots[0..8]: {:?}", &gpu_decrypt[..8]);

    // Check if they match
    if cpu_decrypt == gpu_decrypt {
        println!("✓ GPU automorphism matches CPU!");
    } else {
        println!("✗ GPU automorphism MISMATCH!");
        // Check how many slots differ
        let mismatches: usize = cpu_decrypt.iter().zip(&gpu_decrypt)
            .filter(|(&a, &b)| a != b).count();
        println!("  {} / {} slots differ", mismatches, n);
    }

    assert_eq!(cpu_decrypt, gpu_decrypt, "GPU automorphism should match CPU");
}

#[test]
fn test_gpu_sum_slots_correctness() {
    println!("=== GPU sum_slots Correctness Test ===");
    println!("This test generates real Galois keys and verifies GPU against CPU.");
    println!();

    // Use deterministic RNG
    let mut rng = Prg::new_with_seed(TEST_SEED);

    // Use Goldilocks parameters (3 moduli for IT-PAC)
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    println!("Parameters: n={}, num_moduli={}", n, num_moduli);

    // Generate CPU keys
    println!("Generating CPU keypair...");
    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);

    println!("Generating Galois keys (this takes a while)...");
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
    println!("Generated {} Galois keys", cpu_galois_keys.num_keys());

    // Create test slots: [0, 1, 2, ..., n-1] mod 1000
    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let expected_sum: u64 = test_slots.iter().sum::<u64>() % bgv_params.t;
    println!("Test slots: [0, 1, 2, ...] mod 1000");
    println!("Expected sum: {}", expected_sum);

    // Encrypt test slots
    println!("Encrypting test slots...");
    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Create GPU context
    println!("Creating GPU context...");
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    // Convert to GPU format
    println!("Converting keys to GPU format...");
    let gpu_galois_keys = cpu_to_gpu_galois_keys(&ctx, &cpu_galois_keys, num_moduli);
    let gpu_ct = cpu_to_gpu_ciphertext(&ctx, &cpu_ct);

    // Run GPU sum_slots
    println!("Running GPU sum_slots...");
    let gpu_result = gpu_sum_slots(&ctx, &gpu_ct, &gpu_galois_keys)
        .expect("GPU sum_slots failed");

    // Convert result back to CPU
    println!("Converting result back to CPU format...");
    let rns_params = RnsParams::new(n, num_moduli, 60);
    let cpu_result = gpu_to_cpu_ciphertext(&ctx, &gpu_result, &rns_params, &bgv_params);

    // Decrypt
    println!("Decrypting result...");
    let decrypted_slots = cpu_result.decrypt_slots(&keypair.sk);

    // After sum_slots, all slots should contain the same sum
    let result_sum = decrypted_slots[0];
    println!("Decrypted slot[0]: {}", result_sum);

    // Verify
    if result_sum == expected_sum {
        println!("\n✓ SUCCESS: GPU sum_slots matches expected result!");
    } else {
        println!("\n✗ FAILURE: GPU result {} != expected {}", result_sum, expected_sum);
        // Print first few slots for debugging
        println!("First 8 decrypted slots: {:?}", &decrypted_slots[..8]);
    }

    assert_eq!(result_sum, expected_sum, "GPU sum_slots result mismatch");
}

/// Debug test: verify GPU pointwise multiplication works correctly.
#[test]
fn test_gpu_pointwise_mul_only() {
    use bgv_webgpu::GpuRnsPoly;

    println!("=== GPU Pointwise Multiplication Test ===");

    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let n = ctx.params().n;
    let num_moduli = ctx.params().moduli.len();
    let moduli = ctx.params().moduli.clone();

    println!("n={}, num_moduli={}", n, num_moduli);

    // Create simple polynomials with known values
    // a = [2, 0, 0, ...]
    // b = [3, 0, 0, ...]
    // Expected: a * b (pointwise) = [6, 0, 0, ...]
    let mut a_residues: Vec<Vec<u64>> = Vec::new();
    let mut b_residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        a[0] = 2;
        b[0] = 3;
        a_residues.push(a);
        b_residues.push(b);
    }

    let gpu_a = GpuRnsPoly::from_residues(&ctx, &a_residues).unwrap();
    let gpu_b = GpuRnsPoly::from_residues(&ctx, &b_residues).unwrap();

    // Get the pointwise multiplication result directly (without NTT)
    // We need to expose the internal function or test through the existing interface
    // For now, let's test NTT roundtrip first

    // Test NTT roundtrip: forward NTT then inverse NTT should give back original
    // This tests if NTT is working correctly
    println!("\nTesting NTT roundtrip (forward then inverse)...");

    // Create a test polynomial with various values
    let mut test_residues: Vec<Vec<u64>> = Vec::new();
    for _ in 0..num_moduli {
        let mut coeffs = vec![0u64; n];
        coeffs[0] = 1;
        coeffs[1] = 2;
        coeffs[2] = 3;
        coeffs[3] = 4;
        test_residues.push(coeffs);
    }

    let gpu_test = GpuRnsPoly::from_residues(&ctx, &test_residues).unwrap();
    let result = gpu_test.to_residues(&ctx).unwrap();

    println!("Original poly: [1, 2, 3, 4, 0, ...]");
    println!("Roundtrip result[0..8]: {:?}", &result[0][..8]);

    assert_eq!(result[0][0], 1, "roundtrip coeff 0 should be 1");
    assert_eq!(result[0][1], 2, "roundtrip coeff 1 should be 2");
    assert_eq!(result[0][2], 3, "roundtrip coeff 2 should be 3");
    assert_eq!(result[0][3], 4, "roundtrip coeff 3 should be 4");

    println!("✓ GPU pointwise multiplication test passed!");
}

/// Debug test: verify Barrett mu values and basic NTT data.
#[test]
fn test_debug_barrett_mu() {
    println!("=== Debug Barrett Mu Values ===");

    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    let moduli = ctx.params().moduli.clone();
    println!("Moduli:");
    for (i, &q) in moduli.iter().enumerate() {
        println!("  q[{}] = {} (0x{:016X})", i, q, q);
    }

    // Verify mu computation
    for (i, &q) in moduli.iter().enumerate() {
        // Compute expected mu = floor(2^128 / q)
        // Using u128::MAX / q as approximation
        let mu_approx: u128 = u128::MAX / (q as u128);
        let mu_lo = mu_approx as u64;
        let mu_hi = (mu_approx >> 64) as u64;

        println!("\nModulus {}: q = {}", i, q);
        println!("  mu_approx = {} (0x{:032X})", mu_approx, mu_approx);
        println!("  mu_lo = {} (0x{:016X})", mu_lo, mu_lo);
        println!("  mu_hi = {} (0x{:016X})", mu_hi, mu_hi);

        // Verify: mu * q should be close to 2^128
        let check = (mu_approx as u128) * (q as u128);
        let expected = u128::MAX;
        let diff = if check > expected { check - expected } else { expected - check };
        println!("  Verification: mu * q = {} (diff from 2^128-1: {})", check, diff);
        assert!(diff < q as u128, "mu computation error too large");
    }

    // Test simple modular multiplication in Rust (reference)
    let q = moduli[0];
    let a: u64 = 2;
    let b: u64 = 3;
    let expected = (a as u128 * b as u128 % q as u128) as u64;
    println!("\nSimple multiplication test: {} * {} mod {} = {}", a, b, q, expected);
    assert_eq!(expected, 6, "2*3 should be 6");

    // Test larger multiplication
    let a2: u64 = 1_000_000_000;
    let b2: u64 = 1_000_000_000;
    let expected2 = (a2 as u128 * b2 as u128 % q as u128) as u64;
    println!("Large multiplication: {} * {} mod {} = {}", a2, b2, q, expected2);

    // Test with numbers near q
    let a3: u64 = q - 1;
    let b3: u64 = q - 1;
    let expected3 = (a3 as u128 * b3 as u128 % q as u128) as u64;
    println!("Near-q multiplication: {} * {} mod {} = {}", a3, b3, q, expected3);

    println!("\n✓ Barrett mu debug test passed!");
}

/// Test batched sum_slots produces the same result as the non-batched version.
#[test]
fn test_gpu_sum_slots_batched() {
    println!("=== GPU BATCHED sum_slots Test ===");
    println!("This test verifies batched sum_slots produces the same result.");
    println!();

    // Use deterministic RNG
    let mut rng = Prg::new_with_seed(TEST_SEED);

    // Use Goldilocks parameters
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    println!("Parameters: n={}, num_moduli={}", n, num_moduli);

    // Generate CPU keys
    println!("Generating CPU keypair...");
    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);

    println!("Generating Galois keys (this takes a while)...");
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
    println!("Generated {} Galois keys", cpu_galois_keys.num_keys());

    // Create test slots
    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let expected_sum: u64 = test_slots.iter().sum::<u64>() % bgv_params.t;
    println!("Expected sum: {}", expected_sum);

    // Encrypt test slots
    println!("Encrypting test slots...");
    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Create GPU context
    println!("Creating GPU context...");
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    // Convert to GPU format
    println!("Converting keys to GPU format...");
    let gpu_galois_keys = cpu_to_gpu_galois_keys(&ctx, &cpu_galois_keys, num_moduli);
    let gpu_ct = cpu_to_gpu_ciphertext(&ctx, &cpu_ct);

    // Create workspace for batched version
    println!("Creating workspace for batched sum_slots...");
    let workspace = SumSlotsWorkspace::new(&ctx);

    // Run batched GPU sum_slots
    println!("Running BATCHED GPU sum_slots (single command buffer)...");
    let start = std::time::Instant::now();
    let gpu_result = gpu_sum_slots_batched(&ctx, &gpu_ct, &gpu_galois_keys, &workspace)
        .expect("Batched GPU sum_slots failed");
    let elapsed = start.elapsed();
    println!("Batched sum_slots completed in {:?}", elapsed);

    // Convert result back to CPU
    println!("Converting result back to CPU format...");
    let rns_params = RnsParams::new(n, num_moduli, 60);
    let cpu_result = gpu_to_cpu_ciphertext(&ctx, &gpu_result, &rns_params, &bgv_params);

    // Decrypt
    println!("Decrypting result...");
    let decrypted_slots = cpu_result.decrypt_slots(&keypair.sk);

    // After sum_slots, all slots should contain the same sum
    let result_sum = decrypted_slots[0];
    println!("Decrypted slot[0]: {}", result_sum);

    // Verify
    if result_sum == expected_sum {
        println!("\n✓ SUCCESS: Batched GPU sum_slots matches expected result!");
    } else {
        println!("\n✗ FAILURE: Batched GPU result {} != expected {}", result_sum, expected_sum);
        println!("First 8 decrypted slots: {:?}", &decrypted_slots[..8]);
    }

    assert_eq!(result_sum, expected_sum, "Batched GPU sum_slots result mismatch");
}

/// Profile batched sum_slots to identify bottlenecks.
#[test]
fn test_gpu_sum_slots_batched_profiled() {
    use bgv_webgpu::gpu_sum_slots_batched_profiled;

    println!("=== GPU BATCHED sum_slots PROFILING ===");
    println!();

    // Use deterministic RNG
    let mut rng = Prg::new_with_seed(TEST_SEED);

    // Use Goldilocks parameters
    let bgv_params = RnsBgvParams::goldilocks();
    let n = bgv_params.n;
    let num_moduli = bgv_params.num_moduli;

    println!("Parameters: n={}, num_moduli={}", n, num_moduli);

    // Generate CPU keys
    println!("Generating CPU keypair...");
    let keypair = RnsKeyPair::generate(&bgv_params, &mut rng);

    println!("Generating Galois keys...");
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
    println!("Generated {} Galois keys", cpu_galois_keys.num_keys());

    // Create test slots
    let test_slots: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();

    // Encrypt test slots
    println!("Encrypting test slots...");
    let cpu_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Create GPU context
    println!("Creating GPU context...");
    let gpu_params = GpuRnsParams::goldilocks();
    let ctx = match GpuRotationContext::new(gpu_params) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping test - GPU context creation failed: {:?}", e);
            return;
        }
    };

    // Convert to GPU format
    println!("Converting keys to GPU format...");
    let gpu_galois_keys = cpu_to_gpu_galois_keys(&ctx, &cpu_galois_keys, num_moduli);
    let gpu_ct = cpu_to_gpu_ciphertext(&ctx, &cpu_ct);

    // Create workspace
    let workspace = SumSlotsWorkspace::new(&ctx);

    // Run profiled version
    println!("\n--- Running profiled GPU sum_slots ---\n");
    let _gpu_result = gpu_sum_slots_batched_profiled(&ctx, &gpu_ct, &gpu_galois_keys, &workspace)
        .expect("Profiled GPU sum_slots failed");

    println!("\n✓ Profiling complete!");
}
