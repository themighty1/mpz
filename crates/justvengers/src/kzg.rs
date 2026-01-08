//! KZG Polynomial Commitment Scheme over BLS12-381.
//!
//! This module provides KZG commitments for polynomials with Goldilocks coefficients.
//! Goldilocks elements (64-bit) are embedded into the BLS12-381 scalar field (~253-bit).
//!
//! # Communication Savings
//!
//! - Full polynomial (n coefficients): n × 8 bytes
//! - KZG commitment: 48 bytes (compressed G1 point)
//! - KZG opening proof: 48 bytes + 32 bytes (G1 point + scalar)
//!
//! For n=65536: 512 KB → 80 bytes (99.98% reduction)
//!
//! # Usage
//!
//! KZG is optional and can be enabled with the `kzg` feature. The default JV protocol
//! sends full polynomial coefficients for information-theoretic security. With KZG,
//! you trade computational security for smaller messages.

use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, SeedableRng};

/// KZG trusted setup parameters (Structured Reference String).
///
/// For production, this should be generated via a trusted setup ceremony.
/// For testing/development, can use a deterministic "toxic waste" setup.
#[derive(Clone)]
pub struct KzgSrs {
    /// Powers of tau in G1: [τ⁰]₁, [τ¹]₁, ..., [τⁿ⁻¹]₁
    pub powers_g1: Vec<G1Affine>,
    /// [τ]₂ for pairing verification
    pub tau_g2: G2Affine,
    /// [1]₂ generator
    pub g2_gen: G2Affine,
}

/// A KZG commitment (compressed G1 point, 48 bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgCommitment(pub G1Affine);

/// A KZG opening proof.
#[derive(Clone, Debug)]
pub struct KzgProof {
    /// The quotient commitment [q(τ)]₁
    pub quotient: G1Affine,
    /// The claimed evaluation p(z)
    pub evaluation: Fr,
}

// ============================================================================
// Helper functions for polynomial arithmetic over Fr
// ============================================================================

/// Evaluates polynomial at point z using Horner's method.
fn evaluate_poly_fr(coeffs: &[Fr], z: Fr) -> Fr {
    if coeffs.is_empty() {
        return Fr::zero();
    }
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * z + coeffs[i];
    }
    result
}

/// Divides polynomial p(X) by (X - z) using synthetic division.
/// Returns quotient q(X) such that p(X) = (X - z) * q(X) + p(z).
fn divide_by_linear(coeffs: &[Fr], z: Fr) -> Vec<Fr> {
    if coeffs.len() <= 1 {
        return vec![];
    }

    let n = coeffs.len();
    let mut quotient = vec![Fr::zero(); n - 1];

    // Synthetic division: start from highest degree
    quotient[n - 2] = coeffs[n - 1];
    for i in (0..n - 2).rev() {
        quotient[i] = coeffs[i + 1] + z * quotient[i + 1];
    }

    quotient
}

// ============================================================================
// KzgSrs Implementation
// ============================================================================

impl KzgSrs {
    /// Generates a trusted setup for polynomials up to degree `max_degree`.
    ///
    /// WARNING: This uses a deterministic "toxic waste" for testing only.
    /// In production, use a proper trusted setup ceremony.
    pub fn setup(max_degree: usize) -> Self {
        // Use deterministic RNG for reproducible setup (testing only!)
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xDEADBEEF);

        // Generate random tau (toxic waste - in production this would be MPC)
        let tau = Fr::from(rng.random::<u64>());

        let g1_gen = G1Projective::generator();
        let g2_gen = G2Projective::generator();

        // Compute powers of tau in G1
        let mut powers_g1 = Vec::with_capacity(max_degree + 1);
        let mut tau_power = Fr::one();
        for _ in 0..=max_degree {
            powers_g1.push((g1_gen * tau_power).into_affine());
            tau_power *= tau;
        }

        // Compute [τ]₂
        let tau_g2 = (g2_gen * tau).into_affine();

        Self {
            powers_g1,
            tau_g2,
            g2_gen: g2_gen.into_affine(),
        }
    }

    /// Returns the maximum polynomial degree this SRS supports.
    pub fn max_degree(&self) -> usize {
        self.powers_g1.len().saturating_sub(1)
    }

    /// Serializes the SRS to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.powers_g1
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        self.tau_g2
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        self.g2_gen
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        bytes
    }

    /// Deserializes the SRS from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        // Use a mutable slice reference - arkworks advances it as it reads
        let mut reader = bytes;
        let powers_g1 =
            Vec::<G1Affine>::deserialize_compressed(&mut reader).map_err(|_| "invalid G1 powers")?;
        let tau_g2 =
            G2Affine::deserialize_compressed(&mut reader).map_err(|_| "invalid tau_g2")?;
        let g2_gen =
            G2Affine::deserialize_compressed(&mut reader).map_err(|_| "invalid g2_gen")?;
        Ok(Self {
            powers_g1,
            tau_g2,
            g2_gen,
        })
    }
}

// ============================================================================
// KzgCommitment Implementation
// ============================================================================

impl KzgCommitment {
    /// Commits to a polynomial with Goldilocks (u64) coefficients.
    ///
    /// The coefficients are lifted into the BLS12-381 scalar field.
    pub fn commit(srs: &KzgSrs, coeffs: &[u64]) -> Self {
        assert!(
            coeffs.len() <= srs.powers_g1.len(),
            "polynomial degree exceeds SRS size"
        );

        if coeffs.is_empty() {
            return Self(G1Affine::identity());
        }

        // Convert u64 coefficients to Fr
        let scalars: Vec<Fr> = coeffs.iter().map(|&c| Fr::from(c)).collect();

        // Multi-scalar multiplication: Σ coeffᵢ × [τⁱ]₁
        let commitment = G1Projective::msm(&srs.powers_g1[..scalars.len()], &scalars)
            .expect("MSM failed")
            .into_affine();

        Self(commitment)
    }

    /// Serializes the commitment to 48 bytes (compressed G1).
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        self.0
            .serialize_compressed(&mut bytes[..])
            .expect("serialization failed");
        bytes
    }

    /// Deserializes a commitment from 48 bytes.
    pub fn from_bytes(bytes: &[u8; 48]) -> Result<Self, &'static str> {
        let point =
            G1Affine::deserialize_compressed(&bytes[..]).map_err(|_| "invalid commitment")?;
        Ok(Self(point))
    }
}

// ============================================================================
// KzgProof Implementation
// ============================================================================

impl KzgProof {
    /// Creates an opening proof for polynomial p(X) at point z.
    ///
    /// Proves that the committed polynomial evaluates to `evaluation` at `z`.
    pub fn open(srs: &KzgSrs, coeffs: &[u64], z: u64) -> Self {
        let z_fr = Fr::from(z);

        // Convert coefficients to Fr
        let fr_coeffs: Vec<Fr> = coeffs.iter().map(|&c| Fr::from(c)).collect();

        // Evaluate p(z)
        let evaluation = evaluate_poly_fr(&fr_coeffs, z_fr);

        // Compute quotient q(X) = (p(X) - p(z)) / (X - z) using synthetic division
        let q_coeffs = divide_by_linear(&fr_coeffs, z_fr);

        // Commit to quotient
        let quotient_commitment = if q_coeffs.is_empty() {
            G1Affine::identity()
        } else {
            G1Projective::msm(&srs.powers_g1[..q_coeffs.len()], &q_coeffs)
                .expect("MSM failed")
                .into_affine()
        };

        Self {
            quotient: quotient_commitment,
            evaluation,
        }
    }

    /// Verifies the opening proof.
    ///
    /// Checks: e([p(τ)]₁ - [y]₁, [1]₂) = e([q(τ)]₁, [τ - z]₂)
    pub fn verify(&self, srs: &KzgSrs, commitment: &KzgCommitment, z: u64) -> bool {
        let z_fr = Fr::from(z);
        let g1_gen = G1Projective::generator();

        // [y]₁ = evaluation × G1
        let y_g1 = (g1_gen * self.evaluation).into_affine();

        // C - [y]₁
        let lhs_g1 = (G1Projective::from(commitment.0) - G1Projective::from(y_g1)).into_affine();

        // [τ]₂ - [z]₂ = [τ - z]₂
        let z_g2 = (G2Projective::generator() * z_fr).into_affine();
        let rhs_g2 = (G2Projective::from(srs.tau_g2) - G2Projective::from(z_g2)).into_affine();

        // Pairing check: e(C - [y]₁, [1]₂) = e([q(τ)]₁, [τ - z]₂)
        let lhs = Bls12_381::pairing(lhs_g1, srs.g2_gen);
        let rhs = Bls12_381::pairing(self.quotient, rhs_g2);

        lhs == rhs
    }

    /// Returns the evaluation as u64 (only valid if result fits in 64 bits).
    pub fn evaluation_u64(&self) -> u64 {
        // Extract the u64 from Fr
        let bytes = self.evaluation.into_bigint().to_bytes_le();
        u64::from_le_bytes(bytes[0..8].try_into().unwrap())
    }

    /// Serializes the proof to bytes (~80 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.quotient
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        self.evaluation
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        bytes
    }

    /// Deserializes a proof from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        // Use a mutable slice reference - arkworks advances it as it reads
        let mut reader = bytes;
        let quotient =
            G1Affine::deserialize_compressed(&mut reader).map_err(|_| "invalid quotient")?;
        let evaluation =
            Fr::deserialize_compressed(&mut reader).map_err(|_| "invalid evaluation")?;
        Ok(Self {
            quotient,
            evaluation,
        })
    }
}

// ============================================================================
// Batch KZG Proof (for multiple polynomials at same point)
// ============================================================================

/// Batch KZG opening - open multiple polynomials at same point.
///
/// More efficient than individual openings when opening at the same z.
pub struct BatchKzgProof {
    /// Combined quotient commitment
    pub quotient: G1Affine,
    /// Individual evaluations
    pub evaluations: Vec<Fr>,
}

impl BatchKzgProof {
    /// Creates batch opening proofs for multiple polynomials at the same point z.
    ///
    /// Uses random linear combination to batch into a single pairing check.
    pub fn open_batch(srs: &KzgSrs, polys: &[&[u64]], z: u64, gamma: u64) -> Self {
        let z_fr = Fr::from(z);
        let gamma_fr = Fr::from(gamma);

        let mut evaluations = Vec::with_capacity(polys.len());
        let mut combined_quotient = G1Projective::zero();
        let mut gamma_power = Fr::one();

        for coeffs in polys {
            // Convert to Fr and evaluate
            let fr_coeffs: Vec<Fr> = coeffs.iter().map(|&c| Fr::from(c)).collect();
            let eval = evaluate_poly_fr(&fr_coeffs, z_fr);
            evaluations.push(eval);

            // Compute quotient
            let q_coeffs = divide_by_linear(&fr_coeffs, z_fr);

            // Add γⁱ × [q_i(τ)]₁ to combined quotient
            if !q_coeffs.is_empty() {
                let q_commitment = G1Projective::msm(&srs.powers_g1[..q_coeffs.len()], &q_coeffs)
                    .expect("MSM failed");
                combined_quotient += q_commitment * gamma_power;
            }

            gamma_power *= gamma_fr;
        }

        Self {
            quotient: combined_quotient.into_affine(),
            evaluations,
        }
    }

    /// Verifies the batch opening proof.
    pub fn verify_batch(
        &self,
        srs: &KzgSrs,
        commitments: &[KzgCommitment],
        z: u64,
        gamma: u64,
    ) -> bool {
        assert_eq!(commitments.len(), self.evaluations.len());

        let z_fr = Fr::from(z);
        let gamma_fr = Fr::from(gamma);
        let g1_gen = G1Projective::generator();

        // Combine commitments and evaluations with powers of gamma
        let mut combined_commitment = G1Projective::zero();
        let mut combined_eval = Fr::zero();
        let mut gamma_power = Fr::one();

        for (comm, &eval) in commitments.iter().zip(&self.evaluations) {
            combined_commitment += G1Projective::from(comm.0) * gamma_power;
            combined_eval += eval * gamma_power;
            gamma_power *= gamma_fr;
        }

        // [y]₁ = combined_eval × G1
        let y_g1 = g1_gen * combined_eval;

        // C - [y]₁
        let lhs_g1 = (combined_commitment - y_g1).into_affine();

        // [τ - z]₂
        let z_g2 = (G2Projective::generator() * z_fr).into_affine();
        let rhs_g2 = (G2Projective::from(srs.tau_g2) - G2Projective::from(z_g2)).into_affine();

        // Pairing check
        let lhs = Bls12_381::pairing(lhs_g1, srs.g2_gen);
        let rhs = Bls12_381::pairing(self.quotient, rhs_g2);

        lhs == rhs
    }

    /// Returns evaluations as u64 (only valid if results fit in 64 bits).
    pub fn evaluations_u64(&self) -> Vec<u64> {
        self.evaluations
            .iter()
            .map(|e| {
                let bytes = e.into_bigint().to_bytes_le();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap())
            })
            .collect()
    }

    /// Serializes the batch proof to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.quotient
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        self.evaluations
            .serialize_compressed(&mut bytes)
            .expect("serialization failed");
        bytes
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_eval() {
        // p(X) = 1 + 2X + 3X²
        let coeffs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        // p(2) = 1 + 4 + 12 = 17
        let result = evaluate_poly_fr(&coeffs, Fr::from(2u64));
        assert_eq!(result, Fr::from(17u64));
    }

    #[test]
    fn test_synthetic_division() {
        // p(X) = 6 + 5X + X² = (X - 2)(X + 3) + 0 when evaluated correctly
        // Actually: p(X) - p(2) = (6 + 10 + 4) = 20, so p(2) = 20
        // (p(X) - 20) / (X - 2) should give quotient
        let coeffs = vec![Fr::from(6u64), Fr::from(5u64), Fr::from(1u64)];
        let z = Fr::from(2u64);
        let eval = evaluate_poly_fr(&coeffs, z);
        assert_eq!(eval, Fr::from(20u64)); // 6 + 10 + 4 = 20

        let quotient = divide_by_linear(&coeffs, z);
        // q(X) should be X + 7 (coeffs [7, 1])
        assert_eq!(quotient.len(), 2);
    }

    #[test]
    fn test_kzg_commit_and_open() {
        // Setup for degree 16
        let srs = KzgSrs::setup(16);

        // Polynomial: p(X) = 1 + 2X + 3X² + 4X³
        let coeffs = vec![1u64, 2, 3, 4];

        // Commit
        let commitment = KzgCommitment::commit(&srs, &coeffs);

        // Open at z = 5
        let z = 5u64;
        let proof = KzgProof::open(&srs, &coeffs, z);

        // Expected: p(5) = 1 + 10 + 75 + 500 = 586
        assert_eq!(proof.evaluation_u64(), 586);

        // Verify
        assert!(proof.verify(&srs, &commitment, z));
    }

    #[test]
    fn test_kzg_wrong_eval_fails() {
        let srs = KzgSrs::setup(16);
        let coeffs = vec![1u64, 2, 3, 4];
        let commitment = KzgCommitment::commit(&srs, &coeffs);
        let z = 5u64;

        let mut proof = KzgProof::open(&srs, &coeffs, z);
        // Tamper with evaluation
        proof.evaluation = Fr::from(999u64);

        // Should fail verification
        assert!(!proof.verify(&srs, &commitment, z));
    }

    #[test]
    fn test_kzg_batch_opening() {
        let srs = KzgSrs::setup(16);

        // Two polynomials
        let p1 = vec![1u64, 2, 3];
        let p2 = vec![4u64, 5, 6, 7];

        let c1 = KzgCommitment::commit(&srs, &p1);
        let c2 = KzgCommitment::commit(&srs, &p2);

        let z = 3u64;
        let gamma = 7u64;

        let batch_proof = BatchKzgProof::open_batch(&srs, &[&p1, &p2], z, gamma);

        // p1(3) = 1 + 6 + 27 = 34
        // p2(3) = 4 + 15 + 54 + 189 = 262
        let evals = batch_proof.evaluations_u64();
        assert_eq!(evals[0], 34);
        assert_eq!(evals[1], 262);

        // Verify
        assert!(batch_proof.verify_batch(&srs, &[c1, c2], z, gamma));
    }

    #[test]
    fn test_commitment_serialization() {
        let srs = KzgSrs::setup(8);
        let coeffs = vec![1u64, 2, 3, 4];
        let commitment = KzgCommitment::commit(&srs, &coeffs);

        let bytes = commitment.to_bytes();
        assert_eq!(bytes.len(), 48);

        let recovered = KzgCommitment::from_bytes(&bytes).unwrap();
        assert_eq!(commitment, recovered);
    }

    #[test]
    fn test_proof_size() {
        let srs = KzgSrs::setup(1024);
        let coeffs: Vec<u64> = (0..1024).collect();

        let commitment = KzgCommitment::commit(&srs, &coeffs);
        let proof = KzgProof::open(&srs, &coeffs, 2);

        assert!(proof.verify(&srs, &commitment, 2));

        // Commitment is only 48 bytes regardless of polynomial size!
        assert_eq!(commitment.to_bytes().len(), 48);

        // Proof is ~80 bytes (48 for G1 + 32 for scalar)
        let proof_bytes = proof.to_bytes();
        assert!(proof_bytes.len() < 100);

        // Compare to sending full polynomial: 1024 * 8 = 8192 bytes
        // Savings: 8192 -> ~130 bytes = 98% reduction
    }
}
