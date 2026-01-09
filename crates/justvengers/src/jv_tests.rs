//! Tests for the JustVengers protocol implementation.

use crate::jv::*;
use crate::topology::Circuit;
use crate::{CircuitBatch, SolderingConstraint, VolePool};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::SeedableRng;

const TEST_MODULUS: u64 = 65537;

#[test]
fn test_jv_prover_setup() {
    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    circuit.add_mul(x, y);

    let batch = CircuitBatch::new(vec![circuit]);

    let mut prover = JVProver::<2>::new(vec![0, 0], TEST_MODULUS);
    let inputs = vec![vec![3, 4], vec![5, 6]];

    prover.setup(&batch, &inputs).unwrap();
    assert_eq!(prover.phase(), &JVProverPhase::Setup);
}

#[test]
fn test_jv_protocol_simple() {
    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    circuit.add_mul(x, y);

    let batch = CircuitBatch::new(vec![circuit]);
    let active_branches = vec![0, 0];
    let inputs = vec![vec![3, 4], vec![5, 6]];

    let result = run_jv_protocol::<2>(&batch, &active_branches, &inputs, &[], TEST_MODULUS);

    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_communication_estimate() {
    // R=1000, C=100, B=4, M=50
    let est = estimate_communication::<1000>(100, 4, 50);

    // Batchman: R*C = 1000*100 = 100K elements in disclosure
    // JustVengers: R = 1000 elements in disclosure
    // Should see significant savings
    assert!(est.savings_percent > 50.0, "Expected >50% savings, got {}%", est.savings_percent);

    println!("Batchman: {} bytes", est.batchman_bytes);
    println!("JustVengers: {} bytes", est.justvengers_bytes);
    println!("Savings: {} bytes ({:.1}%)", est.savings_bytes, est.savings_percent);
}

#[test]
fn test_disclosure_size_comparison() {
    // The key optimization: disclosure message size
    // Batchman: O(RC) - sends all masked witness values
    // JustVengers: O(R) - sends only topology products + aggregated eval

    let r = 10000;
    let c = 100;

    let batchman_disclosure_elements = r * c; // O(RC)
    let jv_disclosure_elements = r + 1;       // O(R)

    let ratio = batchman_disclosure_elements as f64 / jv_disclosure_elements as f64;
    assert!(ratio > 90.0, "Expected >90x reduction, got {:.1}x", ratio);

    println!("R={}, C={}", r, c);
    println!("Batchman disclosure: {} elements", batchman_disclosure_elements);
    println!("JustVengers disclosure: {} elements", jv_disclosure_elements);
    println!("Reduction: {:.1}x", ratio);
}

#[test]
fn test_jv_protocol_with_soldering() {
    const R: usize = 10;

    // Circuit: x*y, y*1 (identity for soldering)
    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    let one = circuit.add_const(1);
    circuit.add_mul(x, y);
    circuit.add_mul(y, one); // mult_output[1] = y

    let batch = CircuitBatch::new(vec![circuit]);

    // Chained inputs: input[0] at rep j = mult_output[1] at rep j-1 = y at rep j-1
    let mut inputs = Vec::with_capacity(R);
    let mut prev_y = 2u64;
    for _ in 0..R {
        let y_val = 3u64;
        inputs.push(vec![prev_y, y_val]);
        prev_y = y_val; // Next rep's input[0] = this rep's y
    }
    let branches = vec![0; R];

    // Constraint: input[0] = mult_output[1]
    let constraint = SolderingConstraint::new(0, 1);

    // Run with Goldilocks (to test NTT code path for main protocol)
    let result = run_jv_protocol::<R>(&batch, &branches, &inputs, &[constraint], GOLDILOCKS);
    assert!(result.is_ok(), "JV protocol with soldering failed: {:?}", result.err());
    assert!(result.unwrap(), "JV protocol verification failed");
}

#[test]
fn test_itpac_commitment_decryption() {
    // Test that IT-PAC ciphertexts are properly decrypted
    // With F_Com pattern, flow is:
    // 1. P sends commitment (hashes only)
    // 2. V sends chi
    // 3. P opens ciphertexts
    // 4. V verifies and decrypts
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    circuit.add_mul(x, y);

    let batch = CircuitBatch::new(vec![circuit]);

    // Setup
    let mut prover = JVProver::<2>::new(vec![0, 0], GOLDILOCKS);
    prover.setup(&batch, &[vec![3, 4], vec![5, 6]]).unwrap();
    prover.setup_soldering(vec![], &mut rng).unwrap();

    let mut verifier = JVVerifier::<2>::new(GOLDILOCKS, &mut rng);
    let setup_msg = verifier.setup(&batch, &mut rng).unwrap();

    // Create VOLE pool and commit
    let circuit_size = batch.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();

    // Receive commitment (only hashes with F_Com)
    let _chi = verifier.receive_commitment(commitment).unwrap();

    // Before ciphertext opening, decrypted should be None
    assert!(verifier.decrypted_commitments().is_none(), "No decryption before ciphertext opening");

    // Open ciphertexts - this triggers decryption
    let ciphertext_opening = prover.open_ciphertexts();
    verifier.receive_ciphertext_opening(ciphertext_opening).unwrap();

    // Now check that decrypted values exist
    let decrypted = verifier.decrypted_commitments();
    assert!(decrypted.is_some(), "Decrypted commitments should exist after opening");
    assert!(!decrypted.unwrap().is_empty(), "Should have decrypted values");

    println!("IT-PAC decryption test passed!");
    println!("Decrypted {} commitment values", decrypted.unwrap().len());
}

// Note: test_itpac_verify_poly_at_lambda was removed because it tests private
// implementation details (verifier.lambda, verifier.ahe_keypair, evaluate_poly_at_lambda).
// The polynomial evaluation is indirectly tested through the full protocol tests.

#[test]
fn test_itmac_opening_verification() {
    // Test the full IT-MAC opening and verification flow
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    circuit.add_mul(x, y);

    let batch = CircuitBatch::new(vec![circuit]);

    // Setup prover and verifier
    let mut prover = JVProver::<2>::new(vec![0, 0], GOLDILOCKS);
    prover.setup(&batch, &[vec![3, 4], vec![5, 6]]).unwrap();
    prover.setup_soldering(vec![], &mut rng).unwrap();

    let mut verifier = JVVerifier::<2>::new(GOLDILOCKS, &mut rng);
    let setup_msg = verifier.setup(&batch, &mut rng).unwrap();

    // Create VOLE pool
    let circuit_size = batch.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    // Extract verifier shares BEFORE passing pool to prover
    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // Prover commits
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
    let _chi = verifier.receive_commitment(commitment).unwrap();

    // Prover opens IT-PAC commitments
    let open_msg = prover.open_itpac().unwrap();

    // Verify the opening message structure
    assert!(!open_msg.polynomials.is_empty(), "Should have polynomials");
    assert!(!open_msg.mac_tags.is_empty(), "Should have MAC tags");
    assert_eq!(open_msg.polynomials.len(), open_msg.mac_tags.len(),
        "Polynomials and MAC tags should match in count");

    // Verify IT-MAC opening
    let verification_result = verifier.verify_itpac_opening(&open_msg);
    assert!(verification_result, "IT-MAC verification should pass");

    println!("IT-MAC opening verification test passed!");
    println!("Verified {} polynomial commitments", open_msg.polynomials.len());
}

#[test]
fn test_itmac_verification_fails_on_tampered_tag() {
    // Test that verification fails when MAC tag is tampered
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    circuit.add_mul(x, y);

    let batch = CircuitBatch::new(vec![circuit]);

    // Setup
    let mut prover = JVProver::<2>::new(vec![0, 0], GOLDILOCKS);
    prover.setup(&batch, &[vec![3, 4], vec![5, 6]]).unwrap();
    prover.setup_soldering(vec![], &mut rng).unwrap();

    let mut verifier = JVVerifier::<2>::new(GOLDILOCKS, &mut rng);
    let setup_msg = verifier.setup(&batch, &mut rng).unwrap();

    // Create VOLE pool and extract shares
    let circuit_size = batch.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);
    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // Commit and open
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
    let _chi = verifier.receive_commitment(commitment).unwrap();
    let mut open_msg = prover.open_itpac().unwrap();

    // Tamper with the first MAC tag
    if !open_msg.mac_tags.is_empty() {
        // Add 1 to the first MAC tag to corrupt it
        open_msg.mac_tags[0] = ItMacFieldType::new(open_msg.mac_tags[0].inner() + 1);
    }

    // Verification should now fail
    let verification_result = verifier.verify_itpac_opening(&open_msg);
    assert!(!verification_result, "IT-MAC verification should fail with tampered tag");

    println!("Tamper detection test passed!");
}

#[test]
fn test_polynomial_operations() {
    // Test basic polynomial operations
    let modulus = 65537u64;

    // Test poly_mul: (1 + 2X) * (3 + 4X) = 3 + 4X + 6X + 8X² = 3 + 10X + 8X²
    let a = vec![1, 2];
    let b = vec![3, 4];
    let product = poly_mul(&a, &b, modulus);
    assert_eq!(product, vec![3, 10, 8]);

    // Test poly_add: (1 + 2X) + (3 + 4X) = 4 + 6X
    let sum = poly_add(&a, &b, modulus);
    assert_eq!(sum, vec![4, 6]);

    // Test poly_sub: (3 + 4X) - (1 + 2X) = 2 + 2X
    let diff = poly_sub(&b, &a, modulus);
    assert_eq!(diff, vec![2, 2]);

    // Test poly_scale: 3 * (1 + 2X) = 3 + 6X
    let scaled = poly_scale(&a, 3, modulus);
    assert_eq!(scaled, vec![3, 6]);

    println!("Polynomial operations test passed!");
}

#[test]
fn test_vanishing_polynomial() {
    let modulus = 65537u64;

    // Evaluation points: {1, 2, 3}
    let eval_points = vec![1u64, 2, 3];

    // Vanishing polynomial Z(X) = (X-1)(X-2)(X-3)
    let z = compute_vanishing_poly(&eval_points, modulus);

    // Z should have degree 3 (4 coefficients)
    assert_eq!(z.len(), 4);

    // Verify Z vanishes at each evaluation point
    for &alpha in &eval_points {
        let z_alpha = evaluate_poly(&z, alpha, modulus);
        assert_eq!(z_alpha, 0, "Z({}) should be 0", alpha);
    }

    // Verify Z doesn't vanish elsewhere
    let z_4 = evaluate_poly(&z, 4, modulus);
    assert_ne!(z_4, 0, "Z(4) should not be 0");

    println!("Vanishing polynomial test passed!");
}

#[test]
fn test_polynomial_division() {
    let modulus = 65537u64;

    // Test: (X² - 1) / (X - 1) = (X + 1) with remainder 0
    // X² - 1 = [65536, 0, 1] in coefficient form (constant, X, X²)
    // Note: -1 mod 65537 = 65536
    let dividend = vec![modulus - 1, 0, 1]; // -1 + 0X + X²
    let divisor = vec![modulus - 1, 1];      // -1 + X = (X - 1)

    let (quotient, remainder) = poly_div(&dividend, &divisor, modulus);

    // Quotient should be X + 1 = [1, 1]
    assert_eq!(quotient, vec![1, 1], "Quotient should be X + 1");

    // Remainder should be 0 (or empty/all zeros)
    let rem_is_zero = remainder.is_empty() || remainder.iter().all(|&c| c == 0);
    assert!(rem_is_zero, "Remainder should be 0");

    println!("Polynomial division test passed!");
}

#[test]
fn test_vanishing_poly_division() {
    let modulus = 65537u64;

    // Create a polynomial H(X) that vanishes at points {1, 2}
    // H(X) = (X-1)(X-2) = X² - 3X + 2
    let eval_points = vec![1u64, 2];
    let z = compute_vanishing_poly(&eval_points, modulus);

    // H(X) = 2*(X-1)(X-2) (scaled to make it non-trivial)
    let h = poly_scale(&z, 2, modulus);

    // Divide H by Z, should get quotient 2 with zero remainder
    let (quotient, remainder) = poly_div(&h, &z, modulus);

    // Quotient should be just [2]
    assert_eq!(quotient, vec![2], "Quotient should be constant 2");

    // Remainder should be zero
    let rem_is_zero = remainder.is_empty() || remainder.iter().all(|&c| c == 0);
    assert!(rem_is_zero, "Remainder should be 0");

    println!("Vanishing polynomial division test passed!");
}
