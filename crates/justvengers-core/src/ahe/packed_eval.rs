//! Packed polynomial evaluation for JustVengers protocol.
//!
//! This module provides rotation-free polynomial evaluation using slot-wise
//! multiplication and 2-way packing. Instead of using expensive Galois key
//! rotations to sum slots, the verifier sums in the clear after decryption.
//!
//! # Protocol Overview
//!
//! ```text
//! V → P: pk, Enc([Λ^0, Λ^1, ..., Λ^{n-1}]) in slots
//!
//! P (for each pair of rows i, i+1):
//!   1. ct_i = Enc(powers) ⊙ coeffs_i        // slot-wise mul
//!   2. Generate blinders r_0..r_{n-2} random, r_{n-1} = u - Σr_i
//!   3. ct_i = ct_i - [r_0, r_1, ..., r_{n-1}]  // blind each slot
//!   4. packed = pack_2way(ct_i, ct_{i+1})
//!   → send packed to V
//!
//! V (on receive):
//!   1. Decrypt with packed decryption → get blinded products per slot
//!   2. Sum all n values per row → f_i(Λ) - u_i
//!   3. Use VOLE correlation [u_i] for IT-MAC verification
//! ```
//!
//! # Advantages
//!
//! - **No Galois keys**: Saves ~10MB communication
//! - **No rotations**: Saves expensive key-switching operations on P side
//! - **2-way packing**: Halves the number of ciphertexts sent
//! - **Same security**: Each individual product is hidden by random blinder

use super::packing::CiphertextPacking;
use super::rns_bgv::{RnsCiphertext, RnsKeyPair, RnsPublicKey, RnsSecretKey};
use super::params::RnsBgvParams;
use rand::Rng;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Encrypted powers for packed evaluation.
///
/// Contains a single ciphertext with Λ^i in slot i.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PackedEncryptedPowers {
    /// Ciphertext with [Λ^0, Λ^1, ..., Λ^{n-1}] in slots.
    pub powers_ct: RnsCiphertext,
    /// Number of powers (= number of slots).
    pub num_powers: usize,
    /// Plaintext modulus t.
    pub t: u64,
}

impl PackedEncryptedPowers {
    /// Generates encrypted powers [Λ^0, Λ^1, ..., Λ^{n-1}] in slots.
    ///
    /// # Arguments
    /// * `pk` - Public key for encryption
    /// * `lambda` - Secret evaluation point Λ
    /// * `rng` - Random number generator
    pub fn generate<R: Rng>(pk: &RnsPublicKey, lambda: u64, rng: &mut R) -> Self {
        let n = pk.params().n;
        let t = pk.params().t;

        // Compute powers: [Λ^0, Λ^1, ..., Λ^{n-1}]
        let mut powers = Vec::with_capacity(n);
        let mut power = 1u64;
        for _ in 0..n {
            powers.push(power);
            power = ((power as u128 * lambda as u128) % t as u128) as u64;
        }

        // Encrypt powers in slots
        let powers_ct = RnsCiphertext::encrypt_slots(pk, &powers, rng);

        Self {
            powers_ct,
            num_powers: n,
            t,
        }
    }
}

/// Result of evaluating a row with blinding.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct BlindedRowResult {
    /// Ciphertext with blinded products [c_i * Λ^i - r_i] in slots.
    pub blinded_ct: RnsCiphertext,
    /// The VOLE blinder u = Σr_i (prover knows this).
    pub vole_blinder: u64,
}

/// Prover-side packed evaluator.
///
/// Evaluates polynomials using slot-wise multiplication and blinding,
/// without requiring Galois key rotations.
pub struct PackedProverEvaluator<'a> {
    /// Encrypted powers received from verifier.
    encrypted_powers: &'a PackedEncryptedPowers,
}

impl<'a> PackedProverEvaluator<'a> {
    /// Creates a new evaluator with encrypted powers.
    pub fn new(encrypted_powers: &'a PackedEncryptedPowers) -> Self {
        Self { encrypted_powers }
    }

    /// Evaluates a single row (polynomial coefficients) with blinding.
    ///
    /// # Arguments
    /// * `coeffs` - Polynomial coefficients [c_0, c_1, ..., c_{n-1}]
    /// * `vole_u` - VOLE blinder for this row (from VOLE pool)
    /// * `rng` - Random number generator for local blinders
    ///
    /// # Returns
    /// Ciphertext with blinded products: Enc([c_i * Λ^i - r_i])
    /// where Σr_i = vole_u
    pub fn evaluate_row_blinded<R: Rng>(
        &self,
        coeffs: &[u64],
        vole_u: u64,
        rng: &mut R,
    ) -> RnsCiphertext {
        let n = self.encrypted_powers.num_powers;
        let t = self.encrypted_powers.t;
        assert_eq!(coeffs.len(), n, "coeffs must match number of slots");

        // Step 1: Slot-wise multiplication: Enc(c_i * Λ^i)
        let ct_products = self.encrypted_powers.powers_ct.mul_plaintext_slots(coeffs);

        // Step 2: Generate blinders r_0, ..., r_{n-2} randomly
        // Set r_{n-1} = vole_u - Σr_i so that Σr_i = vole_u
        let mut blinders = Vec::with_capacity(n);
        let mut sum: u128 = 0;
        for _ in 0..n - 1 {
            let r: u64 = rng.random::<u64>() % t;
            blinders.push(r);
            sum = (sum + r as u128) % t as u128;
        }
        // r_{n-1} = vole_u - sum (mod t)
        let r_last = ((vole_u as u128 + t as u128 - sum) % t as u128) as u64;
        blinders.push(r_last);

        // Step 3: Subtract blinders from ciphertext
        ct_products.sub_plaintext_slots(&blinders)
    }

    /// Evaluates two rows and packs them into a single ciphertext.
    ///
    /// # Arguments
    /// * `coeffs0` - First row coefficients
    /// * `coeffs1` - Second row coefficients
    /// * `vole_u0` - VOLE blinder for first row
    /// * `vole_u1` - VOLE blinder for second row
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Packed ciphertext containing both rows (2-way packed)
    pub fn evaluate_and_pack_2way<R: Rng>(
        &self,
        coeffs0: &[u64],
        coeffs1: &[u64],
        vole_u0: u64,
        vole_u1: u64,
        rng: &mut R,
    ) -> RnsCiphertext {
        let ct0 = self.evaluate_row_blinded(coeffs0, vole_u0, rng);
        let ct1 = self.evaluate_row_blinded(coeffs1, vole_u1, rng);
        RnsCiphertext::pack_2way(&ct0, &ct1)
    }

    /// Evaluates multiple row pairs and returns packed ciphertexts.
    ///
    /// For B+C rows, returns (B+C)/2 packed ciphertexts.
    ///
    /// # Arguments
    /// * `rows` - All row coefficients (must have even length)
    /// * `vole_blinders` - VOLE blinders for each row
    /// * `rng` - Random number generator
    pub fn evaluate_all_rows<R: Rng>(
        &self,
        rows: &[Vec<u64>],
        vole_blinders: &[u64],
        rng: &mut R,
    ) -> Vec<RnsCiphertext> {
        assert_eq!(rows.len(), vole_blinders.len());
        assert!(rows.len() % 2 == 0, "number of rows must be even");

        let mut packed_cts = Vec::with_capacity(rows.len() / 2);

        for i in (0..rows.len()).step_by(2) {
            let packed = self.evaluate_and_pack_2way(
                &rows[i],
                &rows[i + 1],
                vole_blinders[i],
                vole_blinders[i + 1],
                rng,
            );
            packed_cts.push(packed);
        }

        packed_cts
    }

    /// Evaluates multiple row pairs in parallel using rayon.
    ///
    /// For B+C rows, returns (B+C)/2 packed ciphertexts.
    /// Each row pair is processed independently in parallel.
    ///
    /// # Arguments
    /// * `rows` - All row coefficients (must have even length)
    /// * `vole_blinders` - VOLE blinders for each row
    #[cfg(feature = "rayon")]
    pub fn evaluate_all_rows_parallel(
        &self,
        rows: &[Vec<u64>],
        vole_blinders: &[u64],
    ) -> Vec<RnsCiphertext> {
        assert_eq!(rows.len(), vole_blinders.len());
        assert!(rows.len() % 2 == 0, "number of rows must be even");

        // Create pairs of (row0, row1, blinder0, blinder1) for parallel processing
        let pairs: Vec<_> = (0..rows.len())
            .step_by(2)
            .map(|i| (&rows[i], &rows[i + 1], vole_blinders[i], vole_blinders[i + 1]))
            .collect();

        pairs
            .par_iter()
            .map(|(row0, row1, blinder0, blinder1)| {
                // Each thread gets its own RNG
                let mut rng = rand::rng();
                self.evaluate_and_pack_2way(row0, row1, *blinder0, *blinder1, &mut rng)
            })
            .collect()
    }
}

/// Verifier-side result of unpacking.
pub struct UnpackedRowValues {
    /// Slot values for first row: [c_i * Λ^i - r_i]
    pub row0_slots: Vec<u64>,
    /// Slot values for second row: [c_j * Λ^j - r_j]
    pub row1_slots: Vec<u64>,
    /// Sum of row0 slots: f_0(Λ) - u_0
    pub row0_sum: u64,
    /// Sum of row1 slots: f_1(Λ) - u_1
    pub row1_sum: u64,
}

/// Verifier-side packed evaluator.
///
/// Decrypts packed ciphertexts and sums slot values to get
/// blinded polynomial evaluations.
pub struct PackedVerifierEvaluator<'a> {
    /// Secret key for decryption.
    sk: &'a RnsSecretKey,
    /// Plaintext modulus.
    t: u64,
}

impl<'a> PackedVerifierEvaluator<'a> {
    /// Creates a new verifier evaluator.
    pub fn new(sk: &'a RnsSecretKey, t: u64) -> Self {
        Self { sk, t }
    }

    /// Unpacks and processes a single 2-way packed ciphertext.
    ///
    /// # Returns
    /// - Slot values for both rows
    /// - Sum of slots for each row (= f(Λ) - u)
    pub fn unpack_and_sum(&self, packed_ct: &RnsCiphertext) -> UnpackedRowValues {
        // Decrypt all slots
        let all_slots = packed_ct.decrypt_slots(self.sk);
        let _n = all_slots.len();

        // For 2-way packing: each slot contains row0_val * 2^64 + row1_val (mod t)
        // We need to extract both values from each slot.
        //
        // Since t = Goldilocks ≈ 2^64, the packed value is:
        // slot = row0_val * 2^64 + row1_val (mod t)
        //
        // To extract: we need the shift constant
        let _shift64_mod_t = ((1u128 << 64) % self.t as u128) as u64;

        // For now, use standard decryption which gives us the combined value.
        // The proper unpacking requires the extended decryption.
        // TODO: Use proper packed decryption from packing module

        // Simplified: assume we have separate ciphertexts or use different approach
        // For the actual protocol, we'd use decrypt_packed from packing.rs

        let mut row0_sum: u128 = 0;
        let row1_sum: u128 = 0;

        // Placeholder: in real implementation, extract packed values properly
        // For now, just demonstrate the summing
        for &slot in &all_slots {
            row0_sum = (row0_sum + slot as u128) % self.t as u128;
        }

        UnpackedRowValues {
            row0_slots: all_slots.clone(),
            row1_slots: all_slots,
            row0_sum: row0_sum as u64,
            row1_sum: row1_sum as u64,
        }
    }

    /// Processes a batch of packed ciphertexts.
    ///
    /// # Returns
    /// Vector of (sum0, sum1) for each packed ciphertext,
    /// giving f_i(Λ) - u_i for all rows.
    pub fn process_batch(&self, packed_cts: &[RnsCiphertext]) -> Vec<(u64, u64)> {
        packed_cts
            .iter()
            .map(|ct| {
                let result = self.unpack_and_sum(ct);
                (result.row0_sum, result.row1_sum)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_packed_eval_single_row() {
        let mut rng = rand::rng();
        // Use 8K with 4 moduli (production params)
        let params = RnsBgvParams::new(8192, super::super::params::GOLDILOCKS, 4, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let n = params.n;
        let t = params.t;

        // Verifier generates encrypted powers
        let lambda: u64 = rng.random::<u64>() % t;
        let enc_powers = PackedEncryptedPowers::generate(&keypair.pk, lambda, &mut rng);

        // Prover has coefficients for one row
        let coeffs: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let vole_u: u64 = rng.random::<u64>() % t;

        // Prover evaluates with blinding
        let evaluator = PackedProverEvaluator::new(&enc_powers);
        let blinded_ct = evaluator.evaluate_row_blinded(&coeffs, vole_u, &mut rng);

        // Verifier decrypts and sums
        let slots = blinded_ct.decrypt_slots(&keypair.sk);
        let mut sum: u128 = 0;
        for &s in &slots {
            sum = (sum + s as u128) % t as u128;
        }
        let decrypted_sum = sum as u64;

        // Compute expected: f(Λ) - u
        let mut expected_f_lambda: u128 = 0;
        let mut power: u128 = 1;
        for &c in &coeffs {
            expected_f_lambda = (expected_f_lambda + (c as u128 * power) % t as u128) % t as u128;
            power = (power * lambda as u128) % t as u128;
        }
        let expected = ((expected_f_lambda + t as u128 - vole_u as u128) % t as u128) as u64;

        assert_eq!(decrypted_sum, expected, "f(Λ) - u mismatch");
        println!("✓ Single row packed evaluation works");
        println!("  f(Λ) - u = {} (expected {})", decrypted_sum, expected);
    }

    #[test]
    fn test_packed_eval_blinding_hides_individual() {
        let mut rng = rand::rng();
        let params = RnsBgvParams::new(8192, super::super::params::GOLDILOCKS, 4, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let n = params.n;
        let t = params.t;

        let lambda: u64 = rng.random::<u64>() % t;
        let enc_powers = PackedEncryptedPowers::generate(&keypair.pk, lambda, &mut rng);

        // Same coefficients, different blinding
        let coeffs: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let vole_u: u64 = rng.random::<u64>() % t;

        let evaluator = PackedProverEvaluator::new(&enc_powers);

        // Evaluate twice with same coeffs but different random blinding
        let ct1 = evaluator.evaluate_row_blinded(&coeffs, vole_u, &mut rng);
        let ct2 = evaluator.evaluate_row_blinded(&coeffs, vole_u, &mut rng);

        let slots1 = ct1.decrypt_slots(&keypair.sk);
        let slots2 = ct2.decrypt_slots(&keypair.sk);

        // Individual slots should differ (different random blinders)
        let mut differ_count = 0;
        for i in 0..n {
            if slots1[i] != slots2[i] {
                differ_count += 1;
            }
        }

        // Most slots should differ (only last one is constrained)
        assert!(differ_count > n / 2, "Blinding should randomize individual slots");

        // But sums should be the same (f(Λ) - u)
        let sum1: u64 = slots1.iter().fold(0u128, |a, &b| (a + b as u128) % t as u128) as u64;
        let sum2: u64 = slots2.iter().fold(0u128, |a, &b| (a + b as u128) % t as u128) as u64;
        assert_eq!(sum1, sum2, "Sums should match despite different blinding");

        println!("✓ Blinding hides individual products");
        println!("  {}/{} slots differ between evaluations", differ_count, n);
        println!("  Both sums = {} (same)", sum1);
    }
}
