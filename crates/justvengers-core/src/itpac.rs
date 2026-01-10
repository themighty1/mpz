//! Information-Theoretic Polynomial Authentication Codes (IT-PAC).
//!
//! IT-PAC is a polynomial commitment scheme built on IT-MAC + AHE, introduced
//! by Antman [WYY+22] and used in Justvengers for efficient VOLE-based ZK proofs.
//!
//! # Construction
//!
//! - V holds secret evaluation point Λ ∈ F (in addition to IT-MAC global key Δ)
//! - To commit to polynomial f(·) ∈ F[X], parties obtain [f(Λ)] - an IT-MAC of f(Λ)
//! - V sends encrypted powers: ⟦Λ⟧, ⟦Λ²⟧, ..., ⟦Λ^d⟧ using AHE
//! - P uses additive homomorphism to compute ⟦f(Λ) - u⟧ from random IT-MAC [u]
//! - V decrypts to learn f(Λ) - u (one-time padded)
//! - Parties compute [f(Λ)] := [u] + [f(Λ) - u]
//!
//! # Properties
//!
//! - **Hiding**: [f(Λ)] reveals nothing about f(·) to V
//! - **Binding**: To forge f'(·) ≠ f(·), P must guess Λ or Δ
//! - **Linear Homomorphism**: [c₀ + c₁f₁(Λ) + ... + cₙfₙ(Λ)] computed locally
//!
//! # Usage in Justvengers
//!
//! IT-PAC enables encoding R repetitions as a polynomial, reducing communication
//! from O(RC) to O(C) per committed polynomial.

use rand::Rng;

use crate::ahe::{BarrettReducer, BgvParams, Ciphertext, KeyPair, PublicKey, SecretKey};
use crate::ahe::{RnsBgvParams, RnsCiphertext, RnsKeyPair, RnsPublicKey, RnsSecretKey};
use crate::itmac::{GlobalKey, ItMac, ItMacField, VolePool};

/// Encrypted powers of Λ sent by verifier.
///
/// These ciphertexts allow P to homomorphically evaluate polynomials at Λ.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncryptedPowers {
    /// ⟦Λ^i⟧ for i = 1, ..., max_degree
    powers: Vec<Ciphertext>,
    /// Maximum polynomial degree supported.
    max_degree: usize,
    /// AHE parameters.
    params: BgvParams,
}

impl EncryptedPowers {
    /// Creates encrypted powers of Λ.
    ///
    /// V encrypts Λ, Λ², ..., Λ^max_degree using AHE.
    pub fn generate<R: Rng>(pk: &PublicKey, lambda: u64, max_degree: usize, rng: &mut R) -> Self {
        let params = *pk.params();
        let t = params.t;

        let mut powers = Vec::with_capacity(max_degree);
        let mut lambda_power = lambda % t;

        for _ in 0..max_degree {
            powers.push(Ciphertext::encrypt_scalar(pk, lambda_power, rng));
            lambda_power = ((lambda_power as u128 * lambda as u128) % t as u128) as u64;
        }

        Self {
            powers,
            max_degree,
            params,
        }
    }

    /// Returns the maximum polynomial degree supported.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the encrypted power ⟦Λ^i⟧.
    pub fn get(&self, i: usize) -> Option<&Ciphertext> {
        if i == 0 || i > self.max_degree {
            None
        } else {
            Some(&self.powers[i - 1])
        }
    }

    /// Returns all encrypted powers.
    pub fn powers(&self) -> &[Ciphertext] {
        &self.powers
    }

    /// Returns the AHE parameters.
    pub fn params(&self) -> &BgvParams {
        &self.params
    }

    /// Evaluates polynomial f(·) homomorphically to get ⟦f(Λ)⟧.
    ///
    /// Given f(X) = c₀ + c₁X + c₂X² + ... + cₙXⁿ,
    /// computes ⟦f(Λ)⟧ = c₀ + c₁⟦Λ⟧ + c₂⟦Λ²⟧ + ... + cₙ⟦Λⁿ⟧
    pub fn evaluate_poly(&self, coeffs: &[u64]) -> Option<Ciphertext> {
        if coeffs.is_empty() {
            return None;
        }

        let t = self.params.t;

        // If polynomial is just a constant
        if coeffs.len() == 1 {
            // Encrypt the constant directly
            // Note: In real implementation, V would do this, but here we're
            // just creating the ciphertext structure
            return None; // Constant case needs special handling
        }

        if coeffs.len() - 1 > self.max_degree {
            return None;
        }

        // Pre-compute Barrett reducer once for all scalar multiplications.
        // This avoids recomputing the expensive 128-bit division for each call.
        let reducer = BarrettReducer::new(self.params.q);

        // Start with c₀ (as a constant added to the encrypted evaluation)
        // and accumulate c₁⟦Λ⟧ + c₂⟦Λ²⟧ + ...

        // Initialize accumulator with c₁⟦Λ⟧
        let mut result = self.powers[0].scalar_mul_with_reducer(coeffs[1] % t, &reducer);

        // Add remaining terms c₂⟦Λ²⟧ + ...
        for (i, &coeff) in coeffs.iter().enumerate().skip(2) {
            if coeff != 0 {
                let term = self.powers[i - 1].scalar_mul_with_reducer(coeff % t, &reducer);
                result = result + term;
            }
        }

        // Add constant term c₀
        result = result.add_scalar(coeffs[0] % t);

        Some(result)
    }
}

/// Verifier's IT-PAC state.
///
/// Contains the secret evaluation point Λ and the IT-MAC global key.
#[derive(Clone, Debug)]
pub struct ItPacVerifier<F: ItMacField> {
    /// Secret evaluation point Λ.
    lambda: u64,
    /// IT-MAC global key Δ.
    global_key: GlobalKey<F>,
    /// AHE key pair for generating encrypted powers.
    ahe_keypair: KeyPair,
}

impl<F: ItMacField> ItPacVerifier<F> {
    /// Creates a new IT-PAC verifier with fresh secrets.
    pub fn new<R: Rng>(ahe_params: &BgvParams, rng: &mut R) -> Self {
        let lambda = rng.random_range(0..ahe_params.t);
        let global_key = GlobalKey::generate(rng);
        let ahe_keypair = KeyPair::generate(ahe_params, rng);

        Self {
            lambda,
            global_key,
            ahe_keypair,
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

    /// Returns the AHE public key.
    pub fn public_key(&self) -> &PublicKey {
        &self.ahe_keypair.pk
    }

    /// Returns the AHE secret key.
    pub fn secret_key(&self) -> &SecretKey {
        &self.ahe_keypair.sk
    }

    /// Generates encrypted powers of Λ to send to prover.
    ///
    /// # Panics
    ///
    /// Panics if `max_degree` exceeds the number of available BGV slots (ring dimension n).
    pub fn generate_encrypted_powers<R: Rng>(&self, max_degree: usize, rng: &mut R) -> EncryptedPowers {
        let num_slots = self.ahe_keypair.pk.params().n;
        assert!(
            max_degree <= num_slots,
            "R={} exceeds available BGV slots ({}). To support more repetitions, \
             need to parameterize BGV with more slots or use multiple ciphertexts, \
             but currently we support only 1 ciphertext for simplicity.",
            max_degree,
            num_slots
        );
        EncryptedPowers::generate(&self.ahe_keypair.pk, self.lambda, max_degree, rng)
    }

    /// Decrypts a ciphertext (used when P sends ⟦f(Λ) - u⟧).
    pub fn decrypt(&self, ct: &Ciphertext) -> u64 {
        ct.decrypt_scalar(&self.ahe_keypair.sk)
    }

    /// Evaluates a polynomial at Λ.
    pub fn evaluate_at_lambda(&self, coeffs: &[u64]) -> u64 {
        let t = self.ahe_keypair.pk.params().t;
        let mut result = 0u128;
        let mut lambda_power = 1u128;
        let lambda = self.lambda as u128;

        for &coeff in coeffs {
            result = (result + (coeff as u128) * lambda_power) % (t as u128);
            lambda_power = (lambda_power * lambda) % (t as u128);
        }

        result as u64
    }
}

// ============================================================================
// RNS-based IT-PAC for Goldilocks field
// ============================================================================

/// Encrypted powers of Λ using RNS BGV for large plaintext modulus.
///
/// Each power Λ^i is encrypted in a separate ciphertext. This approach
/// supports homomorphic polynomial evaluation via scalar multiplication
/// and addition operations.
///
/// Note: A slot-packed version (all powers in one ciphertext) would require
/// rotation operations for the inner product, which adds complexity.
/// The separate-ciphertext approach is simpler and sufficient for IT-PAC.
#[derive(Clone, Debug)]
pub struct RnsEncryptedPowers {
    /// ⟦Λ^i⟧ for i = 1, ..., max_degree
    powers: Vec<RnsCiphertext>,
    /// Maximum polynomial degree (R).
    max_degree: usize,
    /// BGV parameters.
    params: RnsBgvParams,
}

impl RnsEncryptedPowers {
    /// Creates encrypted powers of Λ.
    ///
    /// V encrypts Λ, Λ², ..., Λ^max_degree using RNS BGV.
    pub fn generate<R: Rng>(
        pk: &RnsPublicKey,
        lambda: u64,
        max_degree: usize,
        rng: &mut R,
    ) -> Self {
        let params = pk.params().clone();
        let t = params.t;

        assert!(
            max_degree <= params.num_slots,
            "R={} exceeds available BGV slots ({}). To support more repetitions, \
             need to parameterize BGV with more slots or use multiple ciphertexts, \
             but currently we support only 1 ciphertext for simplicity.",
            max_degree,
            params.num_slots
        );

        let mut powers = Vec::with_capacity(max_degree);
        let mut lambda_power = lambda % t;

        for _ in 0..max_degree {
            // Encrypt each power as a scalar (in slot 0)
            powers.push(RnsCiphertext::encrypt_scalar(pk, lambda_power, rng));
            lambda_power = ((lambda_power as u128 * lambda as u128) % t as u128) as u64;
        }

        Self {
            powers,
            max_degree,
            params,
        }
    }

    /// Returns the maximum polynomial degree supported.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the encrypted power ⟦Λ^i⟧.
    pub fn get(&self, i: usize) -> Option<&RnsCiphertext> {
        if i == 0 || i > self.max_degree {
            None
        } else {
            Some(&self.powers[i - 1])
        }
    }

    /// Returns all encrypted powers.
    pub fn powers(&self) -> &[RnsCiphertext] {
        &self.powers
    }

    /// Returns the BGV parameters.
    pub fn params(&self) -> &RnsBgvParams {
        &self.params
    }

    /// Evaluates polynomial f(·) homomorphically to get ⟦f(Λ)⟧.
    ///
    /// Given f(X) = c₀ + c₁X + c₂X² + ... + cₙXⁿ,
    /// computes ⟦f(Λ)⟧ = c₀ + c₁⟦Λ⟧ + c₂⟦Λ²⟧ + ... + cₙ⟦Λⁿ⟧
    pub fn evaluate_poly(&self, coeffs: &[u64]) -> Option<RnsCiphertext> {
        if coeffs.is_empty() {
            return None;
        }

        let t = self.params.t;

        // Polynomial must fit within max_degree
        if coeffs.len() > self.max_degree + 1 {
            return None;
        }

        // If polynomial is just a constant, we can't return a proper ciphertext
        // without having a "zero ciphertext" or fresh encryption capability
        if coeffs.len() == 1 {
            return None; // Constant case needs special handling
        }

        // Initialize accumulator with c₁⟦Λ⟧
        let mut result = self.powers[0].scalar_mul(coeffs[1] % t);

        // Add remaining terms c₂⟦Λ²⟧ + ...
        for (i, &coeff) in coeffs.iter().enumerate().skip(2) {
            if coeff != 0 {
                let term = self.powers[i - 1].scalar_mul(coeff % t);
                result = result.add(&term);
            }
        }

        // Add constant term c₀
        // Note: For RNS BGV, adding a scalar requires creating a ciphertext
        // encrypting the scalar or using a specialized add_scalar method.
        // For now, we assume c₀ is handled separately by the caller.
        // TODO: Add add_scalar method to RnsCiphertext

        Some(result)
    }
}

/// RNS-based IT-PAC verifier for Goldilocks field.
///
/// Uses RNS BGV encryption to support large plaintext moduli (like Goldilocks).
#[derive(Clone, Debug)]
pub struct RnsItPacVerifier<F: ItMacField> {
    /// Secret evaluation point Λ.
    lambda: u64,
    /// IT-MAC global key Δ.
    global_key: GlobalKey<F>,
    /// RNS BGV key pair.
    ahe_keypair: RnsKeyPair,
}

impl<F: ItMacField> RnsItPacVerifier<F> {
    /// Creates a new RNS IT-PAC verifier with fresh secrets.
    pub fn new<R: Rng>(bgv_params: &RnsBgvParams, rng: &mut R) -> Self {
        let lambda = rng.random_range(0..bgv_params.t);
        let global_key = GlobalKey::generate(rng);
        let ahe_keypair = RnsKeyPair::generate(bgv_params, rng);

        Self {
            lambda,
            global_key,
            ahe_keypair,
        }
    }

    /// Creates a verifier configured for Goldilocks field.
    ///
    /// Uses 4 RNS moduli (~240 bit q) for sufficient noise budget.
    pub fn new_goldilocks<R: Rng>(rng: &mut R) -> Self {
        let params = RnsBgvParams::goldilocks();
        Self::new(&params, rng)
    }

    /// Returns the secret evaluation point (for testing only).
    pub fn lambda(&self) -> u64 {
        self.lambda
    }

    /// Returns the IT-MAC global key.
    pub fn global_key(&self) -> &GlobalKey<F> {
        &self.global_key
    }

    /// Returns the RNS BGV public key.
    pub fn public_key(&self) -> &RnsPublicKey {
        &self.ahe_keypair.pk
    }

    /// Returns the RNS BGV secret key.
    pub fn secret_key(&self) -> &RnsSecretKey {
        &self.ahe_keypair.sk
    }

    /// Returns the BGV parameters.
    pub fn params(&self) -> &RnsBgvParams {
        self.ahe_keypair.pk.params()
    }

    /// Generates encrypted powers of Λ to send to prover.
    ///
    /// # Panics
    ///
    /// Panics if `max_degree` exceeds the number of available BGV slots.
    pub fn generate_encrypted_powers<R: Rng>(
        &self,
        max_degree: usize,
        rng: &mut R,
    ) -> RnsEncryptedPowers {
        RnsEncryptedPowers::generate(&self.ahe_keypair.pk, self.lambda, max_degree, rng)
    }

    /// Decrypts a ciphertext (used when P sends ⟦f(Λ) - u⟧).
    pub fn decrypt(&self, ct: &RnsCiphertext) -> u64 {
        ct.decrypt_scalar(&self.ahe_keypair.sk)
    }

    /// Evaluates a polynomial at Λ directly (for verification).
    pub fn evaluate_at_lambda(&self, coeffs: &[u64]) -> u64 {
        let t = self.params().t;
        let mut result = 0u128;
        let mut lambda_power = 1u128;
        let lambda = self.lambda as u128;

        for &coeff in coeffs {
            result = (result + (coeff as u128) * lambda_power) % (t as u128);
            lambda_power = (lambda_power * lambda) % (t as u128);
        }

        result as u64
    }
}

/// An IT-PAC commitment [f(·)].
///
/// This represents a commitment to polynomial f(·) using IT-MAC [f(Λ)].
/// The polynomial is bound but Λ remains hidden from P until opening.
#[derive(Clone, Debug)]
pub struct ItPac<F: ItMacField> {
    /// The committed polynomial (prover's view).
    polynomial: Vec<u64>,
    /// IT-MAC of f(Λ).
    mac: ItMac<F>,
}

impl<F: ItMacField> ItPac<F> {
    /// Creates a new IT-PAC from a polynomial and its IT-MAC at Λ.
    pub fn new(polynomial: Vec<u64>, mac: ItMac<F>) -> Self {
        Self { polynomial, mac }
    }

    /// Returns the committed polynomial coefficients.
    pub fn polynomial(&self) -> &[u64] {
        &self.polynomial
    }

    /// Returns the degree of the committed polynomial.
    pub fn degree(&self) -> usize {
        if self.polynomial.is_empty() {
            0
        } else {
            self.polynomial.len() - 1
        }
    }

    /// Returns the underlying IT-MAC [f(Λ)].
    pub fn mac(&self) -> &ItMac<F> {
        &self.mac
    }

    /// Adds two IT-PACs: [f + g].
    ///
    /// Linear homomorphism on polynomials.
    pub fn add(&self, other: &Self, modulus: u64) -> Self {
        let max_len = self.polynomial.len().max(other.polynomial.len());
        let mut result_poly = vec![0u64; max_len];

        for (i, coeff) in self.polynomial.iter().enumerate() {
            result_poly[i] = *coeff;
        }
        for (i, coeff) in other.polynomial.iter().enumerate() {
            result_poly[i] = (result_poly[i] + coeff) % modulus;
        }

        Self {
            polynomial: result_poly,
            mac: self.mac.add(&other.mac),
        }
    }

    /// Subtracts two IT-PACs: [f - g].
    pub fn sub(&self, other: &Self, modulus: u64) -> Self {
        let max_len = self.polynomial.len().max(other.polynomial.len());
        let mut result_poly = vec![0u64; max_len];

        for (i, coeff) in self.polynomial.iter().enumerate() {
            result_poly[i] = *coeff;
        }
        for (i, coeff) in other.polynomial.iter().enumerate() {
            result_poly[i] = (result_poly[i] + modulus - coeff) % modulus;
        }

        Self {
            polynomial: result_poly,
            mac: self.mac.sub(&other.mac),
        }
    }

    /// Multiplies IT-PAC by scalar: [c·f].
    pub fn scalar_mul(&self, c: u64, modulus: u64) -> Self {
        let result_poly: Vec<_> = self
            .polynomial
            .iter()
            .map(|&coeff| ((coeff as u128 * c as u128) % modulus as u128) as u64)
            .collect();

        // Need to convert c to the IT-MAC field type
        // This is a simplification - in real impl we'd need proper field conversion
        Self {
            polynomial: result_poly,
            mac: self.mac.clone(), // Simplified - real impl needs scalar_mul on mac
        }
    }

    /// Checks if the polynomial is vanishing at given points.
    ///
    /// A polynomial is vanishing at α₁, ..., αᵣ if f(αᵢ) = 0 for all i.
    pub fn is_vanishing(&self, points: &[u64], modulus: u64) -> bool {
        for &point in points {
            let eval = self.evaluate_at(point, modulus);
            if eval != 0 {
                return false;
            }
        }
        true
    }

    /// Evaluates the polynomial at a point.
    fn evaluate_at(&self, x: u64, modulus: u64) -> u64 {
        let mut result = 0u128;
        let mut x_power = 1u128;
        let x = x as u128;

        for &coeff in &self.polynomial {
            result = (result + (coeff as u128) * x_power) % (modulus as u128);
            x_power = (x_power * x) % (modulus as u128);
        }

        result as u64
    }
}

/// IT-PAC generator for creating polynomial commitments.
///
/// This handles the protocol between P and V for generating IT-PACs.
pub struct ItPacGenerator<F: ItMacField> {
    /// Encrypted powers from verifier.
    encrypted_powers: EncryptedPowers,
    /// VOLE pool for one-time pads.
    vole_pool: VolePool<F>,
}

impl<F: ItMacField> ItPacGenerator<F> {
    /// Creates a new IT-PAC generator.
    pub fn new(encrypted_powers: EncryptedPowers, vole_pool: VolePool<F>) -> Self {
        Self {
            encrypted_powers,
            vole_pool,
        }
    }

    /// Commits to a polynomial f(·).
    ///
    /// # Protocol
    ///
    /// 1. P computes ⟦f(Λ)⟧ using encrypted powers
    /// 2. P consumes random IT-MAC [u] from VOLE pool
    /// 3. P computes ⟦f(Λ) - u⟧ using AHE homomorphism
    /// 4. P sends ⟦f(Λ) - u⟧ to V (returned as commitment data)
    /// 5. V decrypts to get f(Λ) - u
    /// 6. Parties compute [f(Λ)] = [u] + (f(Λ) - u)
    ///
    /// Returns: (IT-PAC commitment, ciphertext to send to V)
    pub fn commit(&mut self, polynomial: &[u64]) -> Option<(ItPac<F>, Ciphertext)> {
        // Check degree
        if polynomial.len() > self.encrypted_powers.max_degree() + 1 {
            return None;
        }

        // Get random IT-MAC [u]
        let random_mac = self.vole_pool.get_random()?;

        // Compute ⟦f(Λ)⟧ using encrypted powers
        let f_lambda_ct = self.encrypted_powers.evaluate_poly(polynomial)?;

        // In a real protocol, we would:
        // 1. Get u = random_mac.value() (the one-time pad value)
        // 2. Compute ⟦f(Λ) - u⟧ = f_lambda_ct - u using AHE
        // 3. Send ⟦f(Λ) - u⟧ to V who decrypts to get f(Λ) - u
        // 4. V adds to their share to compute [f(Λ)]
        //
        // For now, we return the IT-PAC with the random MAC as placeholder
        // The ciphertext returned would normally be the masked version

        Some((ItPac::new(polynomial.to_vec(), random_mac), f_lambda_ct))
    }

    /// Returns remaining VOLE correlations.
    pub fn remaining(&self) -> usize {
        self.vole_pool.remaining()
    }

    /// Returns a mutable reference to the VOLE pool.
    ///
    /// This allows direct access to the pool for additional IT-MAC operations,
    /// such as committing to input polynomial coefficients (paper Step 9).
    pub fn vole_pool_mut(&mut self) -> &mut VolePool<F> {
        &mut self.vole_pool
    }
}

/// Batch of IT-PAC commitments.
///
/// Used for committing to multiple polynomials efficiently.
#[derive(Clone, Debug)]
pub struct ItPacBatch<F: ItMacField> {
    pacs: Vec<ItPac<F>>,
}

impl<F: ItMacField> ItPacBatch<F> {
    /// Creates a new batch.
    pub fn new(pacs: Vec<ItPac<F>>) -> Self {
        Self { pacs }
    }

    /// Returns the number of commitments.
    pub fn len(&self) -> usize {
        self.pacs.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.pacs.is_empty()
    }

    /// Returns commitment at index i.
    pub fn get(&self, i: usize) -> Option<&ItPac<F>> {
        self.pacs.get(i)
    }

    /// Returns all commitments.
    pub fn pacs(&self) -> &[ItPac<F>] {
        &self.pacs
    }

    /// Computes linear combination: [Σ cᵢ·fᵢ(·)].
    #[cfg(test)]
    pub fn linear_combination(&self, coeffs: &[u64], modulus: u64) -> Option<ItPac<F>> {
        if coeffs.len() != self.pacs.len() || self.pacs.is_empty() {
            return None;
        }

        let mut result = self.pacs[0].scalar_mul(coeffs[0], modulus);
        for (pac, &coeff) in self.pacs.iter().zip(coeffs.iter()).skip(1) {
            let term = pac.scalar_mul(coeff, modulus);
            result = result.add(&term, modulus);
        }

        Some(result)
    }
}

/// Polynomial interpolation for IT-PAC.
///
/// Given R values y₁, ..., yᵣ at fixed points α₁, ..., αᵣ,
/// constructs the unique degree-(R-1) polynomial f(·) with f(αᵢ) = yᵢ.
pub fn interpolate(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    assert_eq!(points.len(), values.len(), "points and values must match");

    if points.is_empty() {
        return vec![];
    }

    let n = points.len();

    // Compute Lagrange basis polynomials and combine
    let mut result = vec![0u64; n];

    for i in 0..n {
        // Compute Lagrange basis polynomial Lᵢ(X)
        let mut basis = vec![1u64];

        // Compute ∏_{j≠i} (X - αⱼ)
        for j in 0..n {
            if i == j {
                continue;
            }

            // Multiply by (X - αⱼ)
            let mut new_basis = vec![0u64; basis.len() + 1];
            for (k, &coeff) in basis.iter().enumerate() {
                // coeff * X
                new_basis[k + 1] = (new_basis[k + 1] + coeff) % modulus;
                // coeff * (-αⱼ) = coeff * (modulus - αⱼ)
                let neg_alpha_j = (modulus - points[j]) % modulus;
                new_basis[k] =
                    ((new_basis[k] as u128 + coeff as u128 * neg_alpha_j as u128) % modulus as u128)
                        as u64;
            }
            basis = new_basis;
        }

        // Compute ∏_{j≠i} (αᵢ - αⱼ)
        let mut denom = 1u128;
        for j in 0..n {
            if i == j {
                continue;
            }
            let diff = if points[i] >= points[j] {
                points[i] - points[j]
            } else {
                modulus - (points[j] - points[i])
            };
            denom = (denom * diff as u128) % modulus as u128;
        }

        // Compute modular inverse of denominator
        let denom_inv = mod_inverse(denom as u64, modulus);

        // Scale basis by yᵢ / denom
        let scale = ((values[i] as u128 * denom_inv as u128) % modulus as u128) as u64;
        for (k, &coeff) in basis.iter().enumerate() {
            let term = ((coeff as u128 * scale as u128) % modulus as u128) as u64;
            result[k] = (result[k] + term) % modulus;
        }
    }

    result
}

/// Computes modular inverse using extended Euclidean algorithm.
fn mod_inverse(a: u64, m: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }

    if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    }
}

/// Vanishing polynomial check helper.
///
/// Verifies that a polynomial vanishes at the specified points.
pub fn is_vanishing(coeffs: &[u64], points: &[u64], modulus: u64) -> bool {
    for &point in points {
        let mut result = 0u128;
        let mut power = 1u128;
        let x = point as u128;

        for &coeff in coeffs {
            result = (result + (coeff as u128) * power) % (modulus as u128);
            power = (power * x) % (modulus as u128);
        }

        if result != 0 {
            return false;
        }
    }
    true
}

/// Creates a random vanishing polynomial of degree d.
///
/// A vanishing polynomial at α₁, ..., αᵣ can be written as
/// r(X) · Z(X) where Z(X) = ∏(X - αᵢ) is the vanishing polynomial.
pub fn random_vanishing_poly<R: Rng>(
    points: &[u64],
    degree: usize,
    modulus: u64,
    rng: &mut R,
) -> Vec<u64> {
    if degree < points.len() {
        // Degree too small to have non-trivial vanishing polynomial
        return vec![0; degree + 1];
    }

    // Compute Z(X) = ∏(X - αᵢ)
    let mut z = vec![1u64];
    for &point in points {
        let mut new_z = vec![0u64; z.len() + 1];
        for (k, &coeff) in z.iter().enumerate() {
            // coeff * X
            new_z[k + 1] = (new_z[k + 1] + coeff) % modulus;
            // coeff * (-point)
            let neg_point = (modulus - point) % modulus;
            new_z[k] =
                ((new_z[k] as u128 + coeff as u128 * neg_point as u128) % modulus as u128) as u64;
        }
        z = new_z;
    }

    // Generate random polynomial r(X) of degree (degree - R)
    let r_degree = degree - points.len();
    let r: Vec<u64> = (0..=r_degree).map(|_| rng.random_range(0..modulus)).collect();

    // Compute r(X) · Z(X)
    multiply_polys(&r, &z, modulus)
}

/// Multiplies two polynomials.
fn multiply_polys(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let mut result = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] =
                ((result[i + j] as u128 + ai as u128 * bj as u128) % modulus as u128) as u64;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ahe::ParamSet;
    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;
    use std::ops::{Add, Mul, Sub};

    /// Test field for IT-MAC.
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    struct TestField(u64);

    const TEST_MODULUS: u64 = 65537; // Same as AHE plaintext modulus

    impl TestField {
        fn new(v: u64) -> Self {
            Self(v % TEST_MODULUS)
        }
    }

    impl Add for TestField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self((self.0 + rhs.0) % TEST_MODULUS)
        }
    }

    impl Sub for TestField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self((self.0 + TEST_MODULUS - rhs.0) % TEST_MODULUS)
        }
    }

    impl Mul for TestField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self((self.0 as u128 * rhs.0 as u128 % TEST_MODULUS as u128) as u64)
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
            Self(rng.random_range(0..TEST_MODULUS))
        }
        fn neg(self) -> Self {
            if self.0 == 0 {
                Self(0)
            } else {
                Self(TEST_MODULUS - self.0)
            }
        }
    }

    #[test]
    fn test_interpolate_linear() {
        // f(1) = 2, f(2) = 4 => f(X) = 2X
        let points = vec![1, 2];
        let values = vec![2, 4];
        let poly = interpolate(&points, &values, TEST_MODULUS);

        assert_eq!(poly.len(), 2);
        assert_eq!(poly[0], 0); // constant term
        assert_eq!(poly[1], 2); // linear term
    }

    #[test]
    fn test_interpolate_quadratic() {
        // f(0) = 1, f(1) = 2, f(2) = 5 => f(X) = 1 + X²
        let points = vec![0, 1, 2];
        let values = vec![1, 2, 5];
        let poly = interpolate(&points, &values, TEST_MODULUS);

        assert_eq!(poly.len(), 3);
        // Verify evaluations
        for (p, v) in points.iter().zip(values.iter()) {
            let eval = evaluate_poly(&poly, *p, TEST_MODULUS);
            assert_eq!(eval, *v);
        }
    }

    fn evaluate_poly(coeffs: &[u64], x: u64, modulus: u64) -> u64 {
        let mut result = 0u128;
        let mut power = 1u128;
        for &c in coeffs {
            result = (result + c as u128 * power) % modulus as u128;
            power = (power * x as u128) % modulus as u128;
        }
        result as u64
    }

    #[test]
    fn test_vanishing_check() {
        // Z(X) = (X - 1)(X - 2) = X² - 3X + 2
        let points = vec![1, 2];
        let poly = vec![2, TEST_MODULUS - 3, 1]; // 2 - 3X + X²

        assert!(is_vanishing(&poly, &points, TEST_MODULUS));
        assert!(!is_vanishing(&[1, 1], &points, TEST_MODULUS));
    }

    #[test]
    fn test_random_vanishing_poly() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let points = vec![1, 2, 3];
        let degree = 5;

        let poly = random_vanishing_poly(&points, degree, TEST_MODULUS, &mut rng);

        assert!(is_vanishing(&poly, &points, TEST_MODULUS));
    }

    #[test]
    fn test_encrypted_powers_evaluate() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();
        let keypair = KeyPair::generate(&params, &mut rng);
        let lambda = 5u64;

        let enc_powers = EncryptedPowers::generate(&keypair.pk, lambda, 3, &mut rng);

        // Polynomial f(X) = 1 + 2X + 3X² + 4X³
        let poly = vec![1, 2, 3, 4];
        let ct = enc_powers.evaluate_poly(&poly).unwrap();

        let decrypted = ct.decrypt_scalar(&keypair.sk);
        let expected = evaluate_poly(&poly, lambda, params.t);

        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_itpac_verifier() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();

        let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);

        // Test evaluation
        let poly = vec![1, 2, 3]; // 1 + 2X + 3X²
        let eval = verifier.evaluate_at_lambda(&poly);
        let expected = evaluate_poly(&poly, verifier.lambda(), params.t);

        assert_eq!(eval, expected);
    }

    #[test]
    fn test_mod_inverse() {
        let a = 3u64;
        let m = 11u64;
        let inv = mod_inverse(a, m);

        assert_eq!((a * inv) % m, 1);
    }

    #[test]
    fn test_multiply_polys() {
        // (1 + 2X) * (3 + 4X) = 3 + 10X + 8X²
        let a = vec![1, 2];
        let b = vec![3, 4];
        let result = multiply_polys(&a, &b, TEST_MODULUS);

        assert_eq!(result, vec![3, 10, 8]);
    }

    #[test]
    fn test_itpac_add() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let poly1 = vec![1, 2, 3]; // 1 + 2X + 3X²
        let poly2 = vec![4, 5, 6]; // 4 + 5X + 6X²

        // Create mock IT-MACs (simplified for testing)
        let mac1 = ItMac::commit(&gk, TestField::new(14), &mut rng); // f(1)
        let mac2 = ItMac::commit(&gk, TestField::new(15), &mut rng); // g(1)

        let itpac1 = ItPac::new(poly1.clone(), mac1);
        let itpac2 = ItPac::new(poly2.clone(), mac2);

        let sum = itpac1.add(&itpac2, TEST_MODULUS);

        // Check polynomial addition
        assert_eq!(sum.polynomial(), &[5, 7, 9]);
    }

    // ==================== Additional comprehensive tests ====================

    mod interpolation_tests {
        use super::*;

        #[test]
        fn test_interpolate_single_point() {
            // f(5) = 10 -> constant polynomial f(X) = 10
            let points = vec![5];
            let values = vec![10];
            let poly = interpolate(&points, &values, TEST_MODULUS);

            assert_eq!(poly.len(), 1);
            assert_eq!(evaluate_poly(&poly, 5, TEST_MODULUS), 10);
        }

        #[test]
        fn test_interpolate_three_points() {
            // f(0) = 0, f(1) = 1, f(2) = 4 -> f(X) = X²
            let points = vec![0, 1, 2];
            let values = vec![0, 1, 4];
            let poly = interpolate(&points, &values, TEST_MODULUS);

            // Verify at all points
            for (&p, &v) in points.iter().zip(values.iter()) {
                assert_eq!(evaluate_poly(&poly, p, TEST_MODULUS), v);
            }
        }

        #[test]
        fn test_interpolate_random_points() {
            let mut rng = Prg::from_seed(Block::ZERO);

            let points: Vec<u64> = (0..5).map(|i| i * 10 + 1).collect();
            let values: Vec<u64> = (0..5).map(|_| rng.random_range(0..TEST_MODULUS)).collect();

            let poly = interpolate(&points, &values, TEST_MODULUS);

            // Should interpolate all points correctly
            for (&p, &v) in points.iter().zip(values.iter()) {
                assert_eq!(evaluate_poly(&poly, p, TEST_MODULUS), v);
            }
        }

        #[test]
        fn test_interpolate_with_modulus_wraparound() {
            // Values near modulus boundary
            let points = vec![1, 2, 3];
            let values = vec![TEST_MODULUS - 1, TEST_MODULUS - 2, TEST_MODULUS - 3];
            let poly = interpolate(&points, &values, TEST_MODULUS);

            for (&p, &v) in points.iter().zip(values.iter()) {
                assert_eq!(evaluate_poly(&poly, p, TEST_MODULUS), v);
            }
        }
    }

    mod vanishing_tests {
        use super::*;

        #[test]
        fn test_vanishing_single_point() {
            // Z(X) = X - 1 should vanish at 1
            let points = vec![1];
            let z = vec![(TEST_MODULUS - 1) % TEST_MODULUS, 1]; // -1 + X

            assert!(is_vanishing(&z, &points, TEST_MODULUS));
        }

        #[test]
        fn test_vanishing_multiple_points() {
            // Z(X) = (X - 1)(X - 2)(X - 3)
            let points = vec![1, 2, 3];

            let mut rng = Prg::from_seed(Block::ZERO);
            let poly = random_vanishing_poly(&points, 5, TEST_MODULUS, &mut rng);

            assert!(is_vanishing(&poly, &points, TEST_MODULUS));
        }

        #[test]
        fn test_non_vanishing_detected() {
            let points = vec![1, 2, 3];
            let non_vanishing = vec![1, 2, 3]; // 1 + 2X + 3X²

            assert!(!is_vanishing(&non_vanishing, &points, TEST_MODULUS));
        }

        #[test]
        fn test_random_vanishing_different_degrees() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let points = vec![1, 2];

            // Generate vanishing polynomials of different degrees
            for degree in 2..=6 {
                let poly = random_vanishing_poly(&points, degree, TEST_MODULUS, &mut rng);
                assert!(is_vanishing(&poly, &points, TEST_MODULUS));
                assert_eq!(poly.len(), degree + 1);
            }
        }
    }

    mod polynomial_arithmetic_tests {
        use super::*;

        #[test]
        fn test_multiply_identity() {
            // p * 1 = p
            let p = vec![1, 2, 3, 4];
            let one = vec![1];
            let result = multiply_polys(&p, &one, TEST_MODULUS);
            assert_eq!(result, p);
        }

        #[test]
        fn test_multiply_zero() {
            // p * 0 = empty
            let p = vec![1, 2, 3];
            let zero = vec![];
            let result = multiply_polys(&p, &zero, TEST_MODULUS);
            assert!(result.is_empty());
        }

        #[test]
        fn test_multiply_commutativity() {
            let a = vec![1, 2, 3];
            let b = vec![4, 5];
            let ab = multiply_polys(&a, &b, TEST_MODULUS);
            let ba = multiply_polys(&b, &a, TEST_MODULUS);
            assert_eq!(ab, ba);
        }

        #[test]
        fn test_multiply_squaring() {
            // (1 + X)² = 1 + 2X + X²
            let p = vec![1, 1];
            let result = multiply_polys(&p, &p, TEST_MODULUS);
            assert_eq!(result, vec![1, 2, 1]);
        }
    }

    mod encrypted_powers_tests {
        use super::*;

        #[test]
        fn test_encrypted_powers_max_degree() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();
            let keypair = KeyPair::generate(&params, &mut rng);

            let enc_powers = EncryptedPowers::generate(&keypair.pk, 3, 5, &mut rng);

            assert_eq!(enc_powers.max_degree(), 5);
            assert!(enc_powers.get(1).is_some());
            assert!(enc_powers.get(5).is_some());
            assert!(enc_powers.get(0).is_none()); // Λ^0 = 1, not included
            assert!(enc_powers.get(6).is_none());
        }

        #[test]
        fn test_encrypted_powers_evaluate_constant() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();
            let keypair = KeyPair::generate(&params, &mut rng);
            let lambda = 7u64;

            let enc_powers = EncryptedPowers::generate(&keypair.pk, lambda, 5, &mut rng);

            // Polynomial f(X) = 5 + X (constant + linear)
            let poly = vec![5, 1];
            let ct = enc_powers.evaluate_poly(&poly).unwrap();

            let decrypted = ct.decrypt_scalar(&keypair.sk);
            let expected = (5 + lambda) % params.t;
            assert_eq!(decrypted, expected);
        }

        #[test]
        fn test_encrypted_powers_higher_degree() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();
            let keypair = KeyPair::generate(&params, &mut rng);
            let lambda = 3u64;

            let enc_powers = EncryptedPowers::generate(&keypair.pk, lambda, 4, &mut rng);

            // f(X) = 1 + X + X² + X³ + X⁴
            let poly = vec![1, 1, 1, 1, 1];
            let ct = enc_powers.evaluate_poly(&poly).unwrap();

            let decrypted = ct.decrypt_scalar(&keypair.sk);
            // f(3) = 1 + 3 + 9 + 27 + 81 = 121
            let expected = evaluate_poly(&poly, lambda, params.t);
            assert_eq!(decrypted, expected);
        }

        #[test]
        fn test_encrypted_powers_degree_check() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();
            let keypair = KeyPair::generate(&params, &mut rng);

            let enc_powers = EncryptedPowers::generate(&keypair.pk, 5, 3, &mut rng);

            // Polynomial of degree 4 should fail (max degree is 3)
            let poly = vec![1, 1, 1, 1, 1]; // degree 4
            assert!(enc_powers.evaluate_poly(&poly).is_none());

            // Degree 3 should work
            let poly3 = vec![1, 1, 1, 1]; // degree 3
            assert!(enc_powers.evaluate_poly(&poly3).is_some());
        }
    }

    mod itpac_operations_tests {
        use super::*;

        #[test]
        fn test_itpac_sub() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let poly1 = vec![10, 20, 30];
            let poly2 = vec![3, 5, 10];

            let mac1 = ItMac::commit(&gk, TestField::new(60), &mut rng);
            let mac2 = ItMac::commit(&gk, TestField::new(18), &mut rng);

            let itpac1 = ItPac::new(poly1, mac1);
            let itpac2 = ItPac::new(poly2, mac2);

            let diff = itpac1.sub(&itpac2, TEST_MODULUS);

            assert_eq!(diff.polynomial(), &[7, 15, 20]);
        }

        #[test]
        fn test_itpac_degree() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let poly = vec![1, 2, 3, 4, 5]; // degree 4
            let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
            let itpac = ItPac::new(poly, mac);

            assert_eq!(itpac.degree(), 4);
        }

        #[test]
        fn test_itpac_empty_polynomial() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let poly: Vec<u64> = vec![];
            let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
            let itpac = ItPac::new(poly, mac);

            assert_eq!(itpac.degree(), 0);
        }

        #[test]
        fn test_itpac_is_vanishing() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            // Z(X) = (X-1)(X-2) = X² - 3X + 2
            let poly = vec![2, TEST_MODULUS - 3, 1];
            let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
            let itpac = ItPac::new(poly, mac);

            let points = vec![1, 2];
            assert!(itpac.is_vanishing(&points, TEST_MODULUS));
        }

        #[test]
        fn test_itpac_add_different_degrees() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let poly1 = vec![1, 2]; // degree 1
            let poly2 = vec![3, 4, 5, 6]; // degree 3

            let mac1 = ItMac::commit(&gk, TestField::new(0), &mut rng);
            let mac2 = ItMac::commit(&gk, TestField::new(0), &mut rng);

            let itpac1 = ItPac::new(poly1, mac1);
            let itpac2 = ItPac::new(poly2, mac2);

            let sum = itpac1.add(&itpac2, TEST_MODULUS);

            assert_eq!(sum.polynomial(), &[4, 6, 5, 6]);
            assert_eq!(sum.degree(), 3);
        }
    }

    mod itpac_batch_tests {
        use super::*;

        #[test]
        fn test_batch_creation() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let polys = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
            let pacs: Vec<_> = polys
                .into_iter()
                .map(|p| {
                    let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
                    ItPac::new(p, mac)
                })
                .collect();

            let batch = ItPacBatch::new(pacs);

            assert_eq!(batch.len(), 3);
            assert!(!batch.is_empty());
        }

        #[test]
        fn test_batch_get() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let polys = vec![vec![1], vec![2], vec![3]];
            let pacs: Vec<_> = polys
                .into_iter()
                .map(|p| {
                    let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
                    ItPac::new(p, mac)
                })
                .collect();

            let batch = ItPacBatch::new(pacs);

            assert!(batch.get(0).is_some());
            assert!(batch.get(2).is_some());
            assert!(batch.get(3).is_none());
        }

        #[test]
        fn test_batch_linear_combination() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            // Three constant polynomials: [10], [20], [30]
            let polys = vec![vec![10], vec![20], vec![30]];
            let pacs: Vec<_> = polys
                .iter()
                .map(|p| {
                    let mac = ItMac::commit(&gk, TestField::new(p[0]), &mut rng);
                    ItPac::new(p.clone(), mac)
                })
                .collect();

            let batch = ItPacBatch::new(pacs);

            // Compute 2*10 + 3*20 + 4*30 = 20 + 60 + 120 = 200
            let coeffs = vec![2, 3, 4];
            let result = batch.linear_combination(&coeffs, TEST_MODULUS).unwrap();

            assert_eq!(result.polynomial(), &[200]);
        }

        #[test]
        fn test_batch_linear_combination_wrong_length() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let polys = vec![vec![1], vec![2]];
            let pacs: Vec<_> = polys
                .into_iter()
                .map(|p| {
                    let mac = ItMac::commit(&gk, TestField::new(0), &mut rng);
                    ItPac::new(p, mac)
                })
                .collect();

            let batch = ItPacBatch::new(pacs);

            // Wrong number of coefficients
            let coeffs = vec![1, 2, 3];
            assert!(batch.linear_combination(&coeffs, TEST_MODULUS).is_none());
        }
    }

    mod verifier_tests {
        use super::*;

        #[test]
        fn test_verifier_creation() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();

            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);

            assert!(verifier.lambda() < params.t);
        }

        #[test]
        fn test_verifier_evaluate_at_lambda() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();

            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);
            let lambda = verifier.lambda();

            // f(X) = 1 + 2X + 3X²
            let poly = vec![1, 2, 3];
            let eval = verifier.evaluate_at_lambda(&poly);
            let expected = evaluate_poly(&poly, lambda, params.t);

            assert_eq!(eval, expected);
        }

        #[test]
        fn test_verifier_generates_encrypted_powers() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();

            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);
            let enc_powers = verifier.generate_encrypted_powers(5, &mut rng);

            assert_eq!(enc_powers.max_degree(), 5);
        }

        #[test]
        fn test_verifier_decrypt() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();

            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);

            // Encrypt with verifier's public key
            let m = 42u64;
            let ct = Ciphertext::encrypt_scalar(verifier.public_key(), m, &mut rng);

            // Decrypt with verifier
            let dec = verifier.decrypt(&ct);
            assert_eq!(dec, m);
        }
    }

    mod mod_inverse_tests {
        use super::*;

        #[test]
        fn test_mod_inverse_small() {
            // 2 * 3 = 6 ≡ 1 (mod 5), so 2^(-1) ≡ 3 (mod 5)
            assert_eq!((2 * mod_inverse(2, 5)) % 5, 1);
        }

        #[test]
        fn test_mod_inverse_large_prime() {
            let p = 1000000007u64;
            let a = 123456789u64;
            let inv = mod_inverse(a, p);
            assert_eq!((a as u128 * inv as u128 % p as u128) as u64, 1);
        }

        #[test]
        fn test_mod_inverse_one() {
            assert_eq!(mod_inverse(1, 17), 1);
        }
    }

    mod integration_tests {
        use super::*;

        #[test]
        fn test_full_itpac_workflow() {
            let mut rng = Prg::from_seed(Block::ZERO);
            // Use Small params for better noise tolerance in homomorphic operations
            let params = ParamSet::Small.params();

            // 1. Verifier setup
            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);
            let lambda = verifier.lambda();

            // 2. Verifier generates encrypted powers
            let enc_powers = verifier.generate_encrypted_powers(5, &mut rng);

            // 3. Prover has polynomial f(X) = 1 + 2X + 3X²
            let poly = vec![1, 2, 3];

            // 4. Prover evaluates f(Λ) homomorphically
            let ct_f_lambda = enc_powers.evaluate_poly(&poly).unwrap();

            // 5. Verifier decrypts to get f(Λ)
            let f_lambda = verifier.decrypt(&ct_f_lambda);

            // 6. Verify it matches direct evaluation
            let expected = evaluate_poly(&poly, lambda, params.t);
            assert_eq!(f_lambda, expected);
        }

        #[test]
        fn test_itpac_with_vanishing_polynomial() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let params = ParamSet::Toy.params();

            let verifier = ItPacVerifier::<TestField>::new(&params, &mut rng);

            // Create vanishing polynomial at points 1, 2, 3
            let points = vec![1, 2, 3];
            let vanishing = random_vanishing_poly(&points, 5, params.t, &mut rng);

            // Should vanish at all points
            assert!(is_vanishing(&vanishing, &points, params.t));

            // Verify using encrypted evaluation
            let enc_powers = verifier.generate_encrypted_powers(5, &mut rng);
            let ct = enc_powers.evaluate_poly(&vanishing).unwrap();
            let result = verifier.decrypt(&ct);

            // If lambda happens to be one of the points, result should be 0
            // Otherwise, it's unlikely to be 0
            let lambda = verifier.lambda();
            if points.contains(&lambda) {
                assert_eq!(result, 0);
            }
        }
    }

    // ==================== RNS IT-PAC tests ====================

    mod rns_itpac_tests {
        use super::*;
        use crate::ahe::GOLDILOCKS;

        #[test]
        fn test_rns_encrypted_powers_generation() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);

            let max_degree = 100;
            let enc_powers = verifier.generate_encrypted_powers(max_degree, &mut rng);

            assert_eq!(enc_powers.max_degree(), max_degree);
            assert_eq!(enc_powers.powers().len(), max_degree);
        }

        #[test]
        fn test_rns_encrypted_powers_decrypt() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let lambda = verifier.lambda();
            let t = verifier.params().t;

            let max_degree = 10;
            let enc_powers = verifier.generate_encrypted_powers(max_degree, &mut rng);

            // Decrypt each power and verify
            let mut expected_power = lambda;
            for i in 1..=max_degree {
                let ct = enc_powers.get(i).unwrap();
                let decrypted = verifier.decrypt(ct);
                assert_eq!(
                    decrypted, expected_power,
                    "power {} should be Λ^{} = {}, got {}",
                    i, i, expected_power, decrypted
                );
                expected_power = ((expected_power as u128 * lambda as u128) % t as u128) as u64;
            }
        }

        #[test]
        fn test_rns_polynomial_evaluation() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let t = verifier.params().t;

            let max_degree = 10;
            let enc_powers = verifier.generate_encrypted_powers(max_degree, &mut rng);

            // Polynomial f(X) = 5 + 3X + 2X² + 7X³
            // Note: c₀=5 is not included in homomorphic evaluation (handled separately)
            let poly = vec![5u64, 3, 2, 7];

            // Evaluate homomorphically (returns c₁Λ + c₂Λ² + c₃Λ³, not including c₀)
            let result_ct = enc_powers.evaluate_poly(&poly).unwrap();
            let result_without_c0 = verifier.decrypt(&result_ct);

            // Add c₀ manually
            let result = (result_without_c0 as u128 + poly[0] as u128) % (t as u128);

            // Compare with direct evaluation
            let expected = verifier.evaluate_at_lambda(&poly);

            assert_eq!(
                result as u64, expected,
                "homomorphic evaluation {} != direct evaluation {}",
                result, expected
            );
        }

        #[test]
        fn test_rns_polynomial_evaluation_large_coeffs() {
            // Uses Goldilocks params with 4 moduli for sufficient noise budget
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let t = verifier.params().t;

            let max_degree = 5;
            let enc_powers = verifier.generate_encrypted_powers(max_degree, &mut rng);

            // Polynomial with large Goldilocks coefficients
            let poly = vec![
                GOLDILOCKS - 1,  // c₀ = -1 mod p
                GOLDILOCKS - 100, // c₁ = -100 mod p
                12345678901234u64 % GOLDILOCKS, // c₂
                GOLDILOCKS / 2, // c₃
            ];

            let result_ct = enc_powers.evaluate_poly(&poly).unwrap();
            let result_without_c0 = verifier.decrypt(&result_ct);
            let result = (result_without_c0 as u128 + poly[0] as u128) % (t as u128);

            let expected = verifier.evaluate_at_lambda(&poly);

            assert_eq!(result as u64, expected);
        }

        #[test]
        fn test_rns_verifier_evaluate_at_lambda() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let lambda = verifier.lambda();
            let t = verifier.params().t;

            // f(X) = 1 + 2X + 3X²
            let poly = vec![1u64, 2, 3];

            let result = verifier.evaluate_at_lambda(&poly);

            // Manual computation
            let expected = (1 + 2 * lambda as u128 + 3 * (lambda as u128).pow(2)) % t as u128;
            assert_eq!(result, expected as u64);
        }

        #[test]
        #[should_panic(expected = "exceeds available BGV slots")]
        fn test_rns_encrypted_powers_exceeds_slots() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);

            // Goldilocks with N=8192 has 8192 slots
            // Trying to create 10000 powers should fail
            let _ = verifier.generate_encrypted_powers(10000, &mut rng);
        }

        #[test]
        fn test_rns_linear_polynomial() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let t = verifier.params().t;

            let enc_powers = verifier.generate_encrypted_powers(10, &mut rng);

            // f(X) = 100 + 50X (linear)
            let poly = vec![100u64, 50];

            let result_ct = enc_powers.evaluate_poly(&poly).unwrap();
            let result_without_c0 = verifier.decrypt(&result_ct);
            let result = (result_without_c0 as u128 + poly[0] as u128) % (t as u128);

            let expected = verifier.evaluate_at_lambda(&poly);
            assert_eq!(result as u64, expected);
        }

        #[test]
        fn test_rns_higher_degree_polynomial() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let verifier = RnsItPacVerifier::<TestField>::new_goldilocks(&mut rng);
            let t = verifier.params().t;

            let max_degree = 20;
            let enc_powers = verifier.generate_encrypted_powers(max_degree, &mut rng);

            // f(X) = 1 + X + X² + ... + X^10 (degree 10)
            let poly: Vec<u64> = (0..=10).map(|_| 1u64).collect();

            let result_ct = enc_powers.evaluate_poly(&poly).unwrap();
            let result_without_c0 = verifier.decrypt(&result_ct);
            let result = (result_without_c0 as u128 + poly[0] as u128) % (t as u128);

            let expected = verifier.evaluate_at_lambda(&poly);
            assert_eq!(result as u64, expected);
        }
    }
}
