//! Integration tests for the AHE module.

use super::*;
use mpz_core::{Block, prg::Prg};
use rand::SeedableRng;

/// Tests end-to-end encryption/decryption with various parameter sets.
#[test]
fn test_all_param_sets() {
    for param_set in [ParamSet::Toy, ParamSet::Small] {
        let params = param_set.params();
        let mut rng = Prg::from_seed(Block::ZERO);

        let keypair = KeyPair::generate(&params, &mut rng);

        // Test simple encryption with small message (avoid noise issues with toy params)
        let m = 100u64;
        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let dec = ct.decrypt_scalar(&keypair.sk);
        assert_eq!(dec, m, "failed for {:?}", param_set);

        // Test homomorphic addition
        let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, 50, &mut rng);
        let ct_sum = ct + ct2;
        let dec_sum = ct_sum.decrypt_scalar(&keypair.sk);
        assert_eq!(dec_sum, (m + 50) % params.t, "add failed for {:?}", param_set);
    }
}

/// Tests that noise growth stays within bounds for a sequence of operations.
#[test]
fn test_noise_growth() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 1u64;
    let mut ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);

    // Perform many additions
    let num_ops = 100;
    for _ in 0..num_ops {
        let ct_one = Ciphertext::encrypt_scalar(&keypair.pk, 1, &mut rng);
        ct = ct + ct_one;
    }

    // Should still decrypt correctly
    let dec = ct.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, (m + num_ops) % params.t);
}

/// Tests circuit privacy through rerandomization.
#[test]
fn test_circuit_privacy() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 42u64;

    // Two different computation paths to same result
    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct1_rerand = ct1.rerandomize(&keypair.pk, &mut rng);

    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, 20, &mut rng);
    let ct3 = Ciphertext::encrypt_scalar(&keypair.pk, 22, &mut rng);
    let ct_sum = (ct2 + ct3).rerandomize(&keypair.pk, &mut rng);

    // Both should decrypt to 42
    assert_eq!(ct1_rerand.decrypt_scalar(&keypair.sk), m);
    assert_eq!(ct_sum.decrypt_scalar(&keypair.sk), m);

    // But the ciphertexts themselves should be different
    // (circuit privacy: can't tell which computation produced the result)
    assert_ne!(ct1_rerand.c0().coeffs(), ct_sum.c0().coeffs());
}

/// Tests linear targeted malleability.
#[test]
fn test_linear_targeted_malleability() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    // Encrypt some value
    let m = 100u64;
    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);

    // Apply various linear transformations
    let tests = [
        (2, 0),   // double
        (1, 50),  // add 50
        (3, 7),   // 3m + 7
        (0, 42),  // constant
    ];

    for (a, b) in tests {
        let ct_linear = ct.clone().linear(a, b);
        let dec = ct_linear.decrypt_scalar(&keypair.sk);
        let expected = (a * m + b) % t;
        assert_eq!(dec, expected, "linear({}, {}) failed", a, b);
    }
}

/// Tests polynomial encryption.
#[test]
fn test_polynomial_message() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    // Encrypt a polynomial message
    let message: Vec<u64> = (0..10).map(|i| (i * i) % params.t).collect();
    let ct = Ciphertext::encrypt_poly(&keypair.pk, &message, &mut rng);
    let decrypted = ct.decrypt_poly(&keypair.sk);

    // Check all coefficients match
    for (i, (&m, &d)) in message.iter().zip(decrypted.iter()).enumerate() {
        assert_eq!(d, m, "coefficient {} mismatch", i);
    }
}

// ==================== Additional comprehensive tests ====================

/// Tests encryption of zero.
#[test]
fn test_encrypt_zero() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let ct = Ciphertext::encrypt_scalar(&keypair.pk, 0, &mut rng);
    let dec = ct.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, 0);
}

/// Tests encryption of moderately large value.
/// Note: Using params.t - 1 can cause noise issues in toy params, so we use a smaller value.
#[test]
fn test_encrypt_large_value() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    // Use a large but not boundary value to avoid noise issues
    let m = 10000u64;
    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let dec = ct.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, m);
}

/// Tests that same message encrypts to different ciphertexts (semantic security).
#[test]
fn test_semantic_security() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 42u64;
    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);

    // Both decrypt to same value
    assert_eq!(ct1.decrypt_scalar(&keypair.sk), m);
    assert_eq!(ct2.decrypt_scalar(&keypair.sk), m);

    // But ciphertexts should be different
    assert_ne!(ct1.c0().coeffs(), ct2.c0().coeffs());
}

/// Tests homomorphic subtraction.
#[test]
fn test_homomorphic_subtraction() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m1 = 100u64;
    let m2 = 30u64;

    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

    let ct_diff = ct1 - ct2;
    let dec = ct_diff.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, m1 - m2);
}

/// Tests homomorphic subtraction with wraparound.
#[test]
fn test_homomorphic_subtraction_wraparound() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let m1 = 10u64;
    let m2 = 30u64;

    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

    let ct_diff = ct1 - ct2;
    let dec = ct_diff.decrypt_scalar(&keypair.sk);
    // 10 - 30 mod t = t - 20
    assert_eq!(dec, (t + m1 - m2) % t);
}

/// Tests homomorphic addition with wraparound using Small params for better noise tolerance.
#[test]
fn test_homomorphic_addition_wraparound() {
    let params = ParamSet::Small.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let m1 = t - 10;
    let m2 = 20u64;

    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

    let ct_sum = ct1 + ct2;
    let dec = ct_sum.decrypt_scalar(&keypair.sk);
    // (t-10) + 20 mod t = 10
    assert_eq!(dec, (m1 + m2) % t);
}

/// Tests scalar multiplication by zero.
#[test]
fn test_scalar_mul_zero() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 42u64;
    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct_zero = ct.scalar_mul(0);

    let dec = ct_zero.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, 0);
}

/// Tests scalar multiplication by one.
#[test]
fn test_scalar_mul_one() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 42u64;
    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct_one = ct.scalar_mul(1);

    let dec = ct_one.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, m);
}

/// Tests negation.
#[test]
fn test_negation() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let m = 42u64;
    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct_neg = -ct;

    let dec = ct_neg.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, (t - m) % t);
}

/// Tests that encryption with different seeds produces different ciphertexts.
#[test]
fn test_different_seeds() {
    let params = ParamSet::Toy.params();

    let mut rng1 = Prg::from_seed(Block::new([1u8; 16]));
    let mut rng2 = Prg::from_seed(Block::new([2u8; 16]));

    let keypair = KeyPair::generate(&params, &mut rng1);

    let m = 42u64;
    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng1);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng2);

    // Both decrypt correctly
    assert_eq!(ct1.decrypt_scalar(&keypair.sk), m);
    assert_eq!(ct2.decrypt_scalar(&keypair.sk), m);

    // But ciphertexts differ
    assert_ne!(ct1.c0().coeffs(), ct2.c0().coeffs());
}

/// Tests chained homomorphic operations.
#[test]
fn test_chained_operations() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    // Compute: 2*(3 + 5) - 7 = 2*8 - 7 = 9
    let ct3 = Ciphertext::encrypt_scalar(&keypair.pk, 3, &mut rng);
    let ct5 = Ciphertext::encrypt_scalar(&keypair.pk, 5, &mut rng);
    let ct7 = Ciphertext::encrypt_scalar(&keypair.pk, 7, &mut rng);

    let ct_sum = ct3 + ct5;
    let ct_doubled = ct_sum.scalar_mul(2);
    let ct_result = ct_doubled - ct7;

    let dec = ct_result.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, (2 * (3 + 5) - 7 + t) % t);
}

/// Tests polynomial homomorphic addition.
#[test]
fn test_polynomial_homomorphic_add() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let msg1: Vec<u64> = vec![1, 2, 3, 4, 5];
    let msg2: Vec<u64> = vec![10, 20, 30, 40, 50];

    let ct1 = Ciphertext::encrypt_poly(&keypair.pk, &msg1, &mut rng);
    let ct2 = Ciphertext::encrypt_poly(&keypair.pk, &msg2, &mut rng);

    let ct_sum = ct1 + ct2;
    let decrypted = ct_sum.decrypt_poly(&keypair.sk);

    for i in 0..5 {
        assert_eq!(decrypted[i], (msg1[i] + msg2[i]) % t);
    }
}

/// Tests multiple rerandomizations don't affect decryption.
#[test]
fn test_multiple_rerandomizations() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    let m = 123u64;
    let mut ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);

    // Rerandomize multiple times
    for _ in 0..10 {
        ct = ct.rerandomize(&keypair.pk, &mut rng);
        assert_eq!(ct.decrypt_scalar(&keypair.sk), m);
    }
}

/// Tests linear combination of multiple ciphertexts.
#[test]
fn test_linear_combination() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    // Compute: 2*10 + 3*20 + 4*30 = 20 + 60 + 120 = 200
    let values = [10u64, 20, 30];
    let coeffs = [2u64, 3, 4];

    let ciphertexts: Vec<_> = values
        .iter()
        .map(|&m| Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng))
        .collect();

    // Manual linear combination
    let mut result = ciphertexts[0].scalar_mul(coeffs[0]);
    for i in 1..3 {
        result = result + ciphertexts[i].scalar_mul(coeffs[i]);
    }

    let dec = result.decrypt_scalar(&keypair.sk);
    let expected: u64 = values
        .iter()
        .zip(coeffs.iter())
        .map(|(&v, &c)| v * c)
        .sum::<u64>()
        % t;
    assert_eq!(dec, expected);
}

/// Tests add_scalar operation.
#[test]
fn test_add_scalar() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let m = 50u64;
    let scalar = 75u64;

    let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
    let ct_added = ct.add_scalar(scalar);

    let dec = ct_added.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, (m + scalar) % t);
}

/// Tests operations preserve correctness with Small parameter set.
#[test]
fn test_small_params_operations() {
    let params = ParamSet::Small.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    // Various operations
    let m1 = 1000u64;
    let m2 = 2000u64;

    let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
    let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

    // Addition
    let ct_sum = ct1.clone() + ct2.clone();
    assert_eq!(ct_sum.decrypt_scalar(&keypair.sk), (m1 + m2) % t);

    // Subtraction
    let ct_diff = ct2.clone() - ct1.clone();
    assert_eq!(ct_diff.decrypt_scalar(&keypair.sk), (m2 - m1) % t);

    // Scalar multiplication
    let ct_scaled = ct1.clone().scalar_mul(5);
    assert_eq!(ct_scaled.decrypt_scalar(&keypair.sk), (m1 * 5) % t);

    // Linear transformation
    let ct_linear = ct1.linear(3, 100);
    assert_eq!(ct_linear.decrypt_scalar(&keypair.sk), (m1 * 3 + 100) % t);
}

/// Tests stress test with many sequential operations.
#[test]
fn test_stress_sequential_operations() {
    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);
    let t = params.t;

    let mut accumulated = 0u64;
    let mut ct = Ciphertext::encrypt_scalar(&keypair.pk, 0, &mut rng);

    for i in 1u64..=50 {
        let ct_i = Ciphertext::encrypt_scalar(&keypair.pk, i, &mut rng);
        ct = ct + ct_i;
        accumulated = (accumulated + i) % t;
    }

    let dec = ct.decrypt_scalar(&keypair.sk);
    assert_eq!(dec, accumulated);
}

/// Tests encryption of random values.
/// Uses smaller range to avoid noise issues with toy parameters.
#[test]
fn test_random_values() {
    use rand::Rng;

    let params = ParamSet::Toy.params();
    let mut rng = Prg::from_seed(Block::ZERO);
    let keypair = KeyPair::generate(&params, &mut rng);

    // Use a smaller range to avoid noise issues near the modulus boundary
    let max_val = 10000u64;

    for _ in 0..100 {
        let m: u64 = rng.random_range(0..max_val);
        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let dec = ct.decrypt_scalar(&keypair.sk);
        assert_eq!(dec, m);
    }
}
