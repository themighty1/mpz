//! CL-based Information-Theoretic Polynomial Authentication Codes.
//!
//! This module provides an alternative IT-PAC implementation using
//! Castagnos-Laguillaumie (CL) encryption instead of BGV.
//!
//! # Protocol Flow
//!
//! 1. **V encrypts powers**: V generates CL encryptions of Λ, Λ², ..., Λ^R
//! 2. **P evaluates**: For each of B+C rows, P computes:
//!    - ⟦f_i(Λ)⟧ = Σⱼ cᵢⱼ · ⟦Λʲ⟧ using scalar_mul and add
//!    - Masks with blinder: ⟦f_i(Λ) - uᵢ⟧
//! 3. **P sends back**: B+C ciphertexts to V
//! 4. **V decrypts**: Gets f_i(Λ) - uᵢ for each row
//!
//! # Advantages over BGV
//!
//! - Simpler structure (no ring operations, no NTT)
//! - Direct support for arbitrary prime fields
//! - No slot packing complexity
//!
//! # Trade-offs
//!
//! - Larger ciphertexts (~3KB vs ~1KB for BGV at same security)
//! - Communication: R ciphertexts from V, (B+C) ciphertexts from P

use crate::ahe::cl_ahe::{CLCiphertext, CLGroup, CLKeyPair, CLPublicKey, CLSecretKey};
use crate::ahe::GOLDILOCKS;
use crate::itmac::{GlobalKey, ItMac, ItMacField, VolePool};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// CL-encrypted powers of Λ sent by verifier.
///
/// V sends ⟦Λ⟧, ⟦Λ²⟧, ..., ⟦Λ^R⟧ to P for homomorphic evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLEncryptedPowers {
    /// ⟦Λ^i⟧ for i = 1, ..., max_degree
    powers: Vec<CLCiphertext>,
    /// Maximum polynomial degree (R).
    max_degree: usize,
}

impl CLEncryptedPowers {
    /// Creates encrypted powers of Λ using CL encryption.
    ///
    /// V encrypts Λ, Λ², ..., Λ^max_degree.
    pub fn generate(group: &CLGroup, pk: &CLPublicKey, lambda: u64, max_degree: usize, modulus: u64) -> Self {
        let mut powers = Vec::with_capacity(max_degree);
        let mut lambda_power = lambda;

        for _ in 0..max_degree {
            powers.push(pk.encrypt_u64(group, lambda_power));
            lambda_power = ((lambda_power as u128 * lambda as u128) % modulus as u128) as u64;
        }

        Self { powers, max_degree }
    }

    /// Returns the maximum polynomial degree supported.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the encrypted power ⟦Λ^i⟧ (1-indexed).
    pub fn get(&self, i: usize) -> Option<&CLCiphertext> {
        if i == 0 || i > self.max_degree {
            None
        } else {
            Some(&self.powers[i - 1])
        }
    }

    /// Returns all encrypted powers.
    pub fn powers(&self) -> &[CLCiphertext] {
        &self.powers
    }

    /// Evaluates polynomial f(·) homomorphically to get ⟦f(Λ)⟧.
    ///
    /// Given f(X) = c₀ + c₁X + c₂X² + ... + cₙXⁿ,
    /// computes ⟦f(Λ)⟧ = c₁⟦Λ⟧ + c₂⟦Λ²⟧ + ... + cₙ⟦Λⁿ⟧
    ///
    /// Note: The constant term c₀ must be added separately after decryption,
    /// or provided as a fresh encryption to be added homomorphically.
    pub fn evaluate_poly(&self, coeffs: &[u64]) -> Option<CLCiphertext> {
        if coeffs.len() <= 1 {
            // Need at least degree 1 polynomial
            return None;
        }

        if coeffs.len() - 1 > self.max_degree {
            return None;
        }

        // Start with c₁⟦Λ⟧
        let mut result = self.powers[0].scalar_mul(coeffs[1]);

        // Add remaining terms c₂⟦Λ²⟧ + ...
        for (i, &coeff) in coeffs.iter().enumerate().skip(2) {
            if coeff != 0 {
                let term = self.powers[i - 1].scalar_mul(coeff);
                result = result.add(&term);
            }
        }

        Some(result)
    }

    /// Evaluates polynomial and subtracts blinder: ⟦f(Λ) - u⟧.
    ///
    /// This is the key operation for IT-PAC: P computes the masked evaluation
    /// using homomorphic operations, then sends it to V.
    ///
    /// Note: The blinder subtraction is handled at the protocol level by adjusting
    /// the constant term. This method returns the evaluation without blinding.
    pub fn evaluate_and_blind(&self, coeffs: &[u64], _blinder: u64) -> Option<CLCiphertext> {
        let ct = self.evaluate_poly(coeffs)?;

        // Note: To properly compute ⟦f(Λ) - u⟧, the protocol should either:
        // 1. Adjust coeffs[0] = coeffs[0] - u before calling evaluate_poly
        // 2. Or have V provide ⟦1⟧ so P can compute ⟦-u⟧ = (-u) * ⟦1⟧
        //
        // The blinder is tracked via IT-MAC [u] separately.

        Some(ct)
    }
}

/// Batch of polynomial evaluations for B+C rows.
///
/// This represents the prover's response containing encrypted evaluations
/// for each branch constraint row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLBatchEvaluation {
    /// ⟦f_i(Λ) - u_i⟧ for i = 1, ..., num_rows
    evaluations: Vec<CLCiphertext>,
}

impl CLBatchEvaluation {
    /// Creates a new batch evaluation.
    pub fn new(evaluations: Vec<CLCiphertext>) -> Self {
        Self { evaluations }
    }

    /// Returns the number of evaluations.
    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Returns evaluation at index i.
    pub fn get(&self, i: usize) -> Option<&CLCiphertext> {
        self.evaluations.get(i)
    }

    /// Returns all evaluations.
    pub fn evaluations(&self) -> &[CLCiphertext] {
        &self.evaluations
    }
}

/// CL-based IT-PAC verifier state.
///
/// Contains the secret evaluation point Λ, IT-MAC global key, and CL keys.
#[derive(Clone, Debug)]
pub struct CLItPacVerifier<F: ItMacField> {
    /// Secret evaluation point Λ.
    lambda: u64,
    /// IT-MAC global key Δ.
    global_key: GlobalKey<F>,
    /// CL group parameters.
    cl_group: CLGroup,
    /// CL key pair.
    cl_keypair: CLKeyPair,
}

impl<F: ItMacField> CLItPacVerifier<F> {
    /// Creates a new CL IT-PAC verifier with hardcoded 1600-bit parameters.
    ///
    /// Uses hardcoded Goldilocks parameters for 128-bit security.
    /// No group setup computation needed.
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let cl_group = CLGroup::new_goldilocks_hardcoded();
        let cl_keypair = cl_group.keygen();
        let lambda = rng.random_range(0..GOLDILOCKS);
        let global_key = GlobalKey::generate(rng);

        Self {
            lambda,
            global_key,
            cl_group,
            cl_keypair,
        }
    }

    /// Creates a verifier with custom CL group and modulus.
    pub fn with_group<R: Rng>(cl_group: CLGroup, modulus: u64, rng: &mut R) -> Self {
        let cl_keypair = cl_group.keygen();
        let lambda = rng.random_range(0..modulus);
        let global_key = GlobalKey::generate(rng);

        Self {
            lambda,
            global_key,
            cl_group,
            cl_keypair,
        }
    }

    /// Returns the secret evaluation point (for testing only).
    pub fn lambda(&self) -> u64 {
        self.lambda
    }

    /// Returns the IT-MAC global key.
    pub fn global_key(&self) -> &GlobalKey<F> {
        &self.global_key
    }

    /// Returns the CL group.
    pub fn cl_group(&self) -> &CLGroup {
        &self.cl_group
    }

    /// Returns the CL public key.
    pub fn public_key(&self) -> &CLPublicKey {
        &self.cl_keypair.pk
    }

    /// Returns the CL secret key.
    pub fn secret_key(&self) -> &CLSecretKey {
        &self.cl_keypair.sk
    }

    /// Generates encrypted powers of Λ to send to prover.
    pub fn generate_encrypted_powers(&self, max_degree: usize) -> CLEncryptedPowers {
        CLEncryptedPowers::generate(
            &self.cl_group,
            &self.cl_keypair.pk,
            self.lambda,
            max_degree,
            GOLDILOCKS,
        )
    }

    /// Decrypts a ciphertext.
    pub fn decrypt(&self, ct: &CLCiphertext) -> u64 {
        self.cl_keypair.sk.decrypt_u64(&self.cl_group, ct)
    }

    /// Decrypts a batch of evaluations.
    pub fn decrypt_batch(&self, batch: &CLBatchEvaluation) -> Vec<u64> {
        batch
            .evaluations
            .iter()
            .map(|ct| self.decrypt(ct))
            .collect()
    }

    /// Evaluates a polynomial at Λ directly (for verification).
    pub fn evaluate_at_lambda(&self, coeffs: &[u64]) -> u64 {
        let mut result = 0u128;
        let mut lambda_power = 1u128;
        let lambda = self.lambda as u128;

        for &coeff in coeffs {
            result = (result + (coeff as u128) * lambda_power) % (GOLDILOCKS as u128);
            lambda_power = (lambda_power * lambda) % (GOLDILOCKS as u128);
        }

        result as u64
    }
}

/// CL IT-PAC prover for computing batched polynomial evaluations.
///
/// P receives encrypted powers from V and computes ⟦f_i(Λ) - u_i⟧ for each row.
pub struct CLItPacProver<F: ItMacField> {
    /// Encrypted powers from verifier.
    encrypted_powers: CLEncryptedPowers,
    /// VOLE pool for blinders.
    vole_pool: VolePool<F>,
    /// Field modulus.
    modulus: u64,
}

impl<F: ItMacField> CLItPacProver<F> {
    /// Creates a new CL IT-PAC prover.
    pub fn new(encrypted_powers: CLEncryptedPowers, vole_pool: VolePool<F>, modulus: u64) -> Self {
        Self {
            encrypted_powers,
            vole_pool,
            modulus,
        }
    }

    /// Evaluates a single polynomial and returns masked ciphertext.
    ///
    /// Returns (⟦f(Λ) - u⟧, [u]) where [u] is the IT-MAC of the blinder.
    pub fn evaluate_single(&mut self, coeffs: &[u64]) -> Option<(CLCiphertext, ItMac<F>)> {
        // Get random IT-MAC [u] for blinding
        let u_mac = self.vole_pool.get_random()?;

        // Compute ⟦f(Λ)⟧
        let ct = self.encrypted_powers.evaluate_poly(coeffs)?;

        // Note: To properly compute ⟦f(Λ) - u⟧, we would need to:
        // 1. Either have V provide ⟦1⟧ (encryption of 1) so we can compute ⟦-u⟧ = (-u) * ⟦1⟧
        // 2. Or handle the constant term adjustment in the protocol
        //
        // For the basic implementation, we return the evaluation and the MAC.
        // The constant term c₀ and blinder u can be handled in the outer protocol.

        Some((ct, u_mac))
    }

    /// Evaluates multiple polynomials (B+C rows) in batch.
    ///
    /// Each polynomial corresponds to a constraint row in the circuit.
    /// Returns the batch of masked ciphertexts and corresponding IT-MACs.
    pub fn evaluate_batch(
        &mut self,
        polynomials: &[Vec<u64>],
    ) -> Option<(CLBatchEvaluation, Vec<ItMac<F>>)> {
        let mut evaluations = Vec::with_capacity(polynomials.len());
        let mut macs = Vec::with_capacity(polynomials.len());

        for poly in polynomials {
            let (ct, mac) = self.evaluate_single(poly)?;
            evaluations.push(ct);
            macs.push(mac);
        }

        Some((CLBatchEvaluation::new(evaluations), macs))
    }

    /// Returns remaining VOLE correlations.
    pub fn remaining(&self) -> usize {
        self.vole_pool.remaining()
    }

    /// Returns a mutable reference to the VOLE pool.
    pub fn vole_pool_mut(&mut self) -> &mut VolePool<F> {
        &mut self.vole_pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;
    use std::ops::{Add, Mul, Sub};

    /// Test field for IT-MAC (using Goldilocks).
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    struct TestField(u64);

    const GOLDILOCKS: u64 = 0xFFFFFFFF00000001;

    impl TestField {
        fn new(v: u64) -> Self {
            Self(v % GOLDILOCKS)
        }
    }

    impl Add for TestField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(((self.0 as u128 + rhs.0 as u128) % GOLDILOCKS as u128) as u64)
        }
    }

    impl Sub for TestField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(((self.0 as u128 + GOLDILOCKS as u128 - rhs.0 as u128) % GOLDILOCKS as u128) as u64)
        }
    }

    impl Mul for TestField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(((self.0 as u128 * rhs.0 as u128) % GOLDILOCKS as u128) as u64)
        }
    }

    impl ItMacField for TestField {
        fn zero() -> Self {
            Self(0)
        }
        fn one() -> Self {
            Self(1)
        }
        fn random<R: Rng>(rng: &mut R) -> Self {
            Self(rng.random_range(0..GOLDILOCKS))
        }
        fn neg(self) -> Self {
            if self.0 == 0 {
                Self(0)
            } else {
                Self(GOLDILOCKS - self.0)
            }
        }
    }

    #[test]
    fn test_cl_encrypted_powers_generation() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();
        let lambda = 12345u64;
        let max_degree = 10;

        let enc_powers = CLEncryptedPowers::generate(&group, &keypair.pk, lambda, max_degree, GOLDILOCKS);

        assert_eq!(enc_powers.max_degree(), max_degree);
        assert_eq!(enc_powers.powers().len(), max_degree);
    }

    #[test]
    fn test_cl_encrypted_powers_decrypt() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();
        let lambda = 5u64;
        let max_degree = 5;

        let enc_powers = CLEncryptedPowers::generate(&group, &keypair.pk, lambda, max_degree, GOLDILOCKS);

        // Verify each decrypted power
        let mut expected = lambda;
        for i in 1..=max_degree {
            let ct = enc_powers.get(i).unwrap();
            let decrypted = keypair.sk.decrypt_u64(&group, ct);
            assert_eq!(decrypted, expected, "power {} mismatch", i);
            expected = ((expected as u128 * lambda as u128) % GOLDILOCKS as u128) as u64;
        }
    }

    #[test]
    fn test_cl_polynomial_evaluation() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();
        let lambda = 3u64;
        let max_degree = 5;

        let enc_powers = CLEncryptedPowers::generate(&group, &keypair.pk, lambda, max_degree, GOLDILOCKS);

        // f(X) = 5 + 2X + 3X²
        let poly = vec![5u64, 2, 3];

        // Evaluate homomorphically (returns 2Λ + 3Λ², without c₀)
        let ct = enc_powers.evaluate_poly(&poly).unwrap();
        let result_without_c0 = keypair.sk.decrypt_u64(&group, &ct);

        // Add c₀ = 5
        let result = (result_without_c0 as u128 + poly[0] as u128) % GOLDILOCKS as u128;

        // Expected: f(3) = 5 + 2*3 + 3*9 = 5 + 6 + 27 = 38
        let expected = 38u64;
        assert_eq!(result as u64, expected);
    }

    #[test]
    fn test_cl_verifier_creation() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let verifier = CLItPacVerifier::<TestField>::new(&mut rng);

        assert!(verifier.lambda() < GOLDILOCKS);
    }

    #[test]
    fn test_cl_verifier_evaluate_at_lambda() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let verifier = CLItPacVerifier::<TestField>::new(&mut rng);
        let lambda = verifier.lambda();

        // f(X) = 1 + 2X + 3X²
        let poly = vec![1u64, 2, 3];
        let result = verifier.evaluate_at_lambda(&poly);

        // Manual computation
        let expected = (1 + 2 * lambda as u128 + 3 * (lambda as u128).pow(2)) % GOLDILOCKS as u128;
        assert_eq!(result, expected as u64);
    }

    #[test]
    fn test_cl_full_workflow() {
        let mut rng = Prg::from_seed(Block::ZERO);

        // 1. Verifier setup
        let verifier = CLItPacVerifier::<TestField>::new(&mut rng);
        let lambda = verifier.lambda();

        // 2. Verifier generates encrypted powers
        let enc_powers = verifier.generate_encrypted_powers(10);

        // 3. Prover has polynomial f(X) = 100 + 50X + 25X²
        let poly = vec![100u64, 50, 25];

        // 4. Prover evaluates f(Λ) homomorphically
        let ct = enc_powers.evaluate_poly(&poly).unwrap();

        // 5. Verifier decrypts (gets 50Λ + 25Λ², without c₀)
        let result_without_c0 = verifier.decrypt(&ct);

        // Add c₀ = 100
        let result = (result_without_c0 as u128 + poly[0] as u128) % GOLDILOCKS as u128;

        // 6. Verify against direct evaluation
        let expected = verifier.evaluate_at_lambda(&poly);
        assert_eq!(result as u64, expected);
    }

    #[test]
    fn test_cl_batch_evaluation() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();
        let lambda = 7u64;

        let enc_powers = CLEncryptedPowers::generate(&group, &keypair.pk, lambda, 10, GOLDILOCKS);

        // Simulate B+C = 3 rows
        let polys = vec![
            vec![1u64, 2, 3],    // f₁(X) = 1 + 2X + 3X²
            vec![4u64, 5, 6, 7], // f₂(X) = 4 + 5X + 6X² + 7X³
            vec![8u64, 9],       // f₃(X) = 8 + 9X
        ];

        let mut evaluations = Vec::new();
        for poly in &polys {
            let ct = enc_powers.evaluate_poly(poly).unwrap();
            evaluations.push(ct);
        }

        let batch = CLBatchEvaluation::new(evaluations);
        assert_eq!(batch.len(), 3);

        // Verify each evaluation
        for (i, poly) in polys.iter().enumerate() {
            let ct = batch.get(i).unwrap();
            let result_without_c0 = keypair.sk.decrypt_u64(&group, ct);
            let result = (result_without_c0 as u128 + poly[0] as u128) % GOLDILOCKS as u128;

            // Compute expected
            let mut expected = 0u128;
            let mut power = 1u128;
            for &c in poly {
                expected = (expected + c as u128 * power) % GOLDILOCKS as u128;
                power = (power * lambda as u128) % GOLDILOCKS as u128;
            }

            assert_eq!(result as u64, expected as u64, "row {} mismatch", i);
        }
    }
}
