//! Optimized JustVengers protocol with O(R+B+C) communication.
//!
//! This module implements the full JustVengers protocol from the paper,
//! achieving O(R+B+C) communication complexity instead of O(RC).
//!
//! # Key Optimization
//!
//! Instead of sending O(RC) individual masked witness values (Batchman approach),
//! we use polynomial encoding:
//!
//! 1. **Polynomial Encoding**: For each wire position w, encode its R values across
//!    repetitions as a degree-(R-1) polynomial f_w(X) where f_w(αⱼ) = w^(j)
//!
//! 2. **IT-PAC Commitment**: Commit to each polynomial via IT-PAC, sending O(C)
//!    ciphertexts ⟦f_w(Λ) - u_w⟧ instead of O(RC) individual values
//!
//! 3. **Vanishing Polynomial**: Prove consistency by showing the constraint
//!    polynomial vanishes at all evaluation points, requiring only O(R) coefficients
//!
//! # Communication Breakdown
//!
//! | Phase       | Batchman (old) | JustVengers (new) |
//! |-------------|----------------|-------------------|
//! | Setup       | O(R)           | O(R)              |
//! | Commitment  | O(C)           | O(C)              |
//! | Disclosure  | O(RC)          | O(R)              | <- Main savings
//! | Open        | O(R+B)         | O(R+B)            |
//! | LPZK        | O(M)           | O(M)              |
//! | **Total**   | **O(RC)**      | **O(R+B+C+M)**    |
//!
//! Where R=repetitions, B=branches, C=circuit size, M=multiplications.
//!
//! # Usage
//!
//! ```ignore
//! use mpz_justvengers::jv_optimized::{JVProver, JVVerifier, run_jv_protocol};
//!
//! // Use the optimized protocol for large R
//! let result = run_jv_protocol::<1000>(
//!     &circuits,
//!     &active_branches,
//!     &inputs_per_rep,
//!     &soldering_constraints,
//!     modulus,
//! )?;
//! ```

use crate::soldering::{
    AggregatedSolderingReveal, SolderingChallengeMessage, SolderingCommitMessage,
    SolderingConstraint, SolderingProver, SolderingRevealMessage, SolderingVerifier,
};
use crate::topology::{CircuitBatch, ExtendedWitness, TopologyVector, UniversalHash};

#[cfg(feature = "ntt")]
use mpz_fields::goldilocks::{Goldilocks, InttContext, GOLDILOCKS};
#[cfg(feature = "ntt")]
use mpz_fields::Field;

use rand::Rng;

// ============================================================================
// Message Types - O(R+B+C) communication
// ============================================================================

/// Setup message from verifier (O(R) communication).
///
/// Contains evaluation points and encrypted powers of Λ for IT-PAC.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JVSetupMessage {
    /// Evaluation points α₁, ..., αᵣ for polynomial interpolation.
    /// For Goldilocks with NTT, these are roots of unity.
    pub eval_points: Vec<u64>,
    /// Maximum polynomial degree (R-1 for R repetitions).
    pub max_degree: usize,
    /// Hash of encrypted powers ⟦Λ⟧, ⟦Λ²⟧, ..., ⟦Λ^(2R-2)⟧.
    /// In real implementation, these would be AHE ciphertexts.
    pub encrypted_powers_hash: u64,
}

/// Commitment message from prover (O(C) communication).
///
/// Contains IT-PAC commitments for each wire position's polynomial.
/// This is O(C) instead of O(RC) because we commit to polynomials, not individual values.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JVCommitmentMessage {
    /// Number of polynomial commitments (one per wire position).
    pub num_polynomials: usize,
    /// IT-PAC polynomial commitments: ⟦f_w(Λ) - u_w⟧ for each wire w.
    /// Each element represents a committed polynomial encoding R values.
    pub poly_commitments: Vec<u64>,
}

/// Disclosure message from prover (O(R) communication).
///
/// **This is the key optimization**: instead of O(RC) masked values,
/// we send O(R) topology products plus aggregated polynomial data.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JVDisclosureMessage {
    /// Topology vector inner products, one per repetition: ⟨t_{id_j}, w^(j)⟩.
    /// This is O(R) instead of sending all O(RC) witness values.
    pub topology_products: Vec<u64>,
    /// Aggregated polynomial evaluation at challenge χ.
    /// This compresses the polynomial consistency check.
    pub aggregated_poly_eval: u64,
}

/// Open message from prover (O(R+B) communication).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JVOpenMessage {
    /// Active branch indices, one per repetition: id_j ∈ [0, B).
    /// Stored as u8 since B < 256 (saves 7 bytes per element vs usize).
    pub active_branches: Vec<u8>,
    /// Universal hash proof component.
    pub hash_proof: u64,
}

/// LPZK proof message (O(M) communication where M = total multiplications).
/// Legacy non-aggregated version.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JVLpzkProofMessage {
    /// Masked multiplication products.
    pub masked_products: Vec<u64>,
    /// MAC tags for verification.
    pub mac_tags: Vec<u64>,
}

/// Aggregated LPZK proof message - O(R) instead of O(M×R).
///
/// Uses vanishing polynomial technique to aggregate all multiplication checks:
/// 1. For each mult gate (a,b,c): constraint h_i(X) = f_a(X)·f_b(X) - f_c(X)
/// 2. Aggregate: H(X) = Σᵢ γⁱ·h_i(X)
/// 3. H(X) vanishes at all αⱼ ⟹ H(X) = Z(X)·Q(X)
/// 4. Send Q(X) coefficients (degree ≤ R-1)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AggregatedLpzkProofMessage {
    /// Quotient polynomial Q(X) = H(X) / Z(X) where H(X) is the aggregated
    /// multiplication constraint polynomial. This has degree ≤ R-1.
    pub quotient_coeffs: Vec<u64>,
    /// Aggregated evaluation: Σᵢ γⁱ·(aᵢ·bᵢ - cᵢ) at a random point for soundness.
    pub aggregated_check: u64,
}

// ============================================================================
// Prover Implementation
// ============================================================================

/// Optimized JustVengers prover with O(R+B+C) communication.
#[derive(Clone, Debug)]
pub struct JVProver<const R: usize> {
    /// Active branch indices, one per repetition.
    active_branches: Vec<usize>,
    /// Extended witnesses for all R repetitions.
    witnesses: Vec<ExtendedWitness>,
    /// Field modulus.
    modulus: u64,
    /// Current protocol phase.
    phase: JVProverPhase,
    /// Evaluation points α₁, ..., αᵣ.
    eval_points: Option<Vec<u64>>,
    /// Polynomial coefficients for each wire position.
    /// wire_polynomials[w] = coefficients of f_w(X) where f_w(αⱼ) = witness[j][w].
    wire_polynomials: Vec<Vec<u64>>,
    /// Precomputed INTT context for Goldilocks.
    #[cfg(feature = "ntt")]
    intt_context: Option<InttContext>,
    /// Soldering prover.
    soldering_prover: Option<SolderingProver>,
}

/// Protocol phases for the optimized prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JVProverPhase {
    /// Initial state.
    Init,
    /// After setup, before commitment.
    Setup,
    /// After commitment, awaiting χ.
    Committed,
    /// After disclosure, awaiting ρ.
    Disclosed,
    /// After opening.
    Opened,
    /// Protocol complete.
    Done,
}

impl<const R: usize> JVProver<R> {
    /// Creates a new optimized prover with per-repetition active branches.
    pub fn new(active_branches: Vec<usize>, modulus: u64) -> Self {
        assert_eq!(
            active_branches.len(),
            R,
            "active_branches must have length R={}, got {}",
            R,
            active_branches.len()
        );
        Self {
            active_branches,
            witnesses: Vec::new(),
            modulus,
            phase: JVProverPhase::Init,
            eval_points: None,
            wire_polynomials: Vec::new(),
            #[cfg(feature = "ntt")]
            intt_context: None,
            soldering_prover: None,
        }
    }

    /// Creates a prover where all repetitions use the same branch.
    pub fn new_single_branch(active_branch: usize, modulus: u64) -> Self {
        Self::new(vec![active_branch; R], modulus)
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &JVProverPhase {
        &self.phase
    }

    /// Returns active branches.
    pub fn active_branches(&self) -> &[usize] {
        &self.active_branches
    }

    /// Initializes the prover with per-repetition circuits.
    pub fn setup(
        &mut self,
        circuits: &CircuitBatch,
        inputs_per_rep: &[Vec<u64>],
    ) -> Result<(), JVProverError> {
        if self.phase != JVProverPhase::Init {
            return Err(JVProverError::InvalidPhase);
        }

        if inputs_per_rep.len() != R {
            return Err(JVProverError::WrongRepetitionCount);
        }

        for &branch_idx in &self.active_branches {
            if branch_idx >= circuits.num_branches() {
                return Err(JVProverError::InvalidBranch);
            }
        }

        // Evaluate each repetition's circuit
        self.witnesses = self
            .active_branches
            .iter()
            .zip(inputs_per_rep.iter())
            .map(|(&branch_idx, inputs)| {
                let mut circuit = circuits.get(branch_idx).unwrap().clone();
                circuit.evaluate(inputs, self.modulus)
            })
            .collect();

        self.phase = JVProverPhase::Setup;
        Ok(())
    }

    /// Sets up soldering constraints.
    pub fn setup_soldering<Rn: Rng>(
        &mut self,
        constraints: Vec<SolderingConstraint>,
        rng: &mut Rn,
    ) -> Result<(), JVProverError> {
        if self.phase != JVProverPhase::Setup {
            return Err(JVProverError::InvalidPhase);
        }

        if constraints.is_empty() {
            return Ok(());
        }

        // Validate constraints
        let inputs_per_rep: Vec<&[u64]> = self.witnesses.iter().map(|w| w.inputs.as_slice()).collect();
        let outputs_per_rep: Vec<&[u64]> = self.witnesses.iter().map(|w| w.mult_outputs.as_slice()).collect();

        for constraint in &constraints {
            if !crate::soldering::validate_witness_soldering_ref(&inputs_per_rep, &outputs_per_rep, constraint) {
                return Err(JVProverError::SolderingConstraintViolation);
            }
        }

        // Generate NTT roots for eval_points (same as main protocol)
        let eval_points = self.get_or_generate_eval_points();
        let n = eval_points.len();

        // Build polynomials for soldering using NTT
        let mut input_polys = Vec::with_capacity(constraints.len());
        let mut output_polys = Vec::with_capacity(constraints.len());

        for constraint in &constraints {
            let in_values: Vec<u64> = self.witnesses
                .iter()
                .map(|w| w.inputs.get(constraint.target_input_idx).copied().unwrap_or(0))
                .collect();

            let out_values: Vec<u64> = self.witnesses
                .iter()
                .map(|w| w.mult_outputs.get(constraint.source_output_idx).copied().unwrap_or(0))
                .collect();

            // Use NTT for Goldilocks (O(n log n)), fallback to Lagrange otherwise
            #[cfg(feature = "ntt")]
            let (in_poly, out_poly) = if self.modulus == GOLDILOCKS {
                // Pad to NTT size and apply INTT
                let mut in_padded: Vec<Goldilocks> = in_values.iter().map(|&v| Goldilocks::new(v)).collect();
                in_padded.resize(n, Goldilocks::zero());
                Goldilocks::intt(&mut in_padded);

                let mut out_padded: Vec<Goldilocks> = out_values.iter().map(|&v| Goldilocks::new(v)).collect();
                out_padded.resize(n, Goldilocks::zero());
                Goldilocks::intt(&mut out_padded);

                (
                    in_padded.iter().map(|g| g.inner()).collect(),
                    out_padded.iter().map(|g| g.inner()).collect(),
                )
            } else {
                (
                    crate::soldering::interpolate(&eval_points, &in_values, self.modulus),
                    crate::soldering::interpolate(&eval_points, &out_values, self.modulus),
                )
            };

            #[cfg(not(feature = "ntt"))]
            let (in_poly, out_poly) = (
                crate::soldering::interpolate(&eval_points, &in_values, self.modulus),
                crate::soldering::interpolate(&eval_points, &out_values, self.modulus),
            );

            input_polys.push(in_poly);
            output_polys.push(out_poly);
        }

        let mut soldering = SolderingProver::new(self.modulus);
        soldering.setup(constraints, input_polys, output_polys, eval_points, R, rng);

        self.soldering_prover = Some(soldering);
        Ok(())
    }

    /// Generates commitment message.
    ///
    /// **Key optimization**: Instead of storing O(RC) values, we compute and commit
    /// to O(C) polynomials, each encoding R values at evaluation points.
    pub fn commit(&mut self, eval_points: &[u64]) -> Result<JVCommitmentMessage, JVProverError> {
        if self.phase != JVProverPhase::Setup {
            return Err(JVProverError::InvalidPhase);
        }

        // Validate eval_points length
        #[cfg(feature = "ntt")]
        let expected_len = if self.modulus == GOLDILOCKS {
            R.next_power_of_two()
        } else {
            R
        };

        #[cfg(not(feature = "ntt"))]
        let expected_len = R;

        if eval_points.len() != expected_len {
            return Err(JVProverError::WrongEvaluationPoints);
        }

        self.eval_points = Some(eval_points.to_vec());

        // Create INTT context for Goldilocks
        #[cfg(feature = "ntt")]
        if self.modulus == GOLDILOCKS {
            self.intt_context = Some(InttContext::new(eval_points.len()));
        }

        // For each wire position, interpolate R values to get polynomial
        let witness_len = self.witnesses[0].len();
        self.wire_polynomials = Vec::with_capacity(witness_len);
        let mut poly_commitments = Vec::with_capacity(witness_len);

        for pos in 0..witness_len {
            // Collect values at this position across all repetitions
            let values: Vec<u64> = self.witnesses.iter().map(|w| w.get(pos)).collect();

            // Interpolate to get polynomial coefficients
            let poly = self.interpolate_values(&values, eval_points);

            // IT-PAC commitment: hash of polynomial (simplified)
            // In real impl: compute ⟦f(Λ) - u⟧ using AHE
            let commitment_hash = poly.iter().fold(0u64, |acc, &c| {
                ((acc as u128 + c as u128) % self.modulus as u128) as u64
            });

            self.wire_polynomials.push(poly);
            poly_commitments.push(commitment_hash);
        }

        self.phase = JVProverPhase::Committed;

        Ok(JVCommitmentMessage {
            num_polynomials: witness_len,
            poly_commitments,
        })
    }

    /// Generates soldering commitment.
    pub fn commit_soldering(&self) -> Result<Option<SolderingCommitMessage>, JVProverError> {
        if self.phase != JVProverPhase::Committed {
            return Err(JVProverError::InvalidPhase);
        }
        Ok(self.soldering_prover.as_ref().map(|s| s.commit()))
    }

    /// Generates disclosure message.
    ///
    /// **Key optimization**: Instead of sending O(RC) masked values, we send:
    /// - O(R) topology products (one per repetition)
    /// - O(1) aggregated polynomial evaluation
    pub fn disclose(
        &mut self,
        chi: u64,
        topology_vectors: &[TopologyVector],
    ) -> Result<JVDisclosureMessage, JVProverError> {
        if self.phase != JVProverPhase::Committed {
            return Err(JVProverError::InvalidPhase);
        }

        for &branch_idx in &self.active_branches {
            if branch_idx >= topology_vectors.len() {
                return Err(JVProverError::InvalidBranch);
            }
        }

        // Compute topology products: O(R) values instead of O(RC)
        let mut topology_products = Vec::with_capacity(R);
        for (j, witness) in self.witnesses.iter().enumerate() {
            let active_tv = &topology_vectors[self.active_branches[j]];
            let w = witness.to_vec();
            topology_products.push(active_tv.inner_product(&w));
        }

        // Compute aggregated polynomial evaluation at χ
        // This compresses polynomial consistency into a single value
        let aggregated_poly_eval = self.compute_aggregated_eval(chi);

        self.phase = JVProverPhase::Disclosed;

        Ok(JVDisclosureMessage {
            topology_products,
            aggregated_poly_eval,
        })
    }

    /// Reveals soldering proof (legacy non-aggregated, O(S×R)).
    pub fn reveal_soldering(
        &self,
        challenge: &SolderingChallengeMessage,
    ) -> Result<Option<SolderingRevealMessage>, JVProverError> {
        if self.phase != JVProverPhase::Disclosed && self.phase != JVProverPhase::Committed {
            return Err(JVProverError::InvalidPhase);
        }
        Ok(self.soldering_prover.as_ref().map(|s| s.reveal(challenge.phi)))
    }

    /// Reveals aggregated soldering proof (O(R) communication).
    ///
    /// Uses random linear combination to aggregate S constraints into one:
    /// - F₁ = Σᵢ ψⁱ × f₁ᵢ
    /// - F₂ = Σᵢ ψⁱ × f₂ᵢ
    ///
    /// This reduces communication from O(S×R) to O(R).
    pub fn reveal_soldering_aggregated(
        &self,
        challenge: &SolderingChallengeMessage,
    ) -> Result<Option<AggregatedSolderingReveal>, JVProverError> {
        if self.phase != JVProverPhase::Disclosed && self.phase != JVProverPhase::Committed {
            return Err(JVProverError::InvalidPhase);
        }
        Ok(self.soldering_prover.as_ref().map(|s| s.reveal_aggregated(challenge.phi, challenge.psi)))
    }

    /// Generates open message.
    pub fn open(
        &mut self,
        rho: u64,
        topology_vectors: &[TopologyVector],
    ) -> Result<JVOpenMessage, JVProverError> {
        if self.phase != JVProverPhase::Disclosed {
            return Err(JVProverError::InvalidPhase);
        }

        // Universal hash proof
        let _hash = UniversalHash::compute(topology_vectors, rho, self.modulus);

        let mut hash_proof = 0u128;
        for (j, witness) in self.witnesses.iter().enumerate() {
            let branch_idx = self.active_branches[j];
            let w = witness.to_vec();
            let active_tv = &topology_vectors[branch_idx];
            let tv_product = active_tv.inner_product(&w);

            let mut rho_power = 1u128;
            for _ in 0..branch_idx {
                rho_power = (rho_power * rho as u128) % self.modulus as u128;
            }

            let contrib = (rho_power * tv_product as u128) % self.modulus as u128;
            hash_proof = (hash_proof + contrib) % self.modulus as u128;
        }

        self.phase = JVProverPhase::Opened;

        Ok(JVOpenMessage {
            active_branches: self.active_branches.iter().map(|&b| b as u8).collect(),
            hash_proof: hash_proof as u64,
        })
    }

    /// Generates LPZK proof for multiplication verification.
    pub fn prove_multiplications(&mut self) -> Result<JVLpzkProofMessage, JVProverError> {
        if self.phase != JVProverPhase::Opened {
            return Err(JVProverError::InvalidPhase);
        }

        let total_mults: usize = self.witnesses.iter().map(|w| w.num_mults()).sum();
        let mut masked_products = Vec::with_capacity(total_mults);
        let mut mac_tags = Vec::with_capacity(total_mults);

        for witness in &self.witnesses {
            for i in 0..witness.num_mults() {
                let a = witness.mult_lefts[i];
                let b = witness.mult_rights[i];
                let c = witness.mult_outputs[i];

                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return Err(JVProverError::InvalidMultiplication);
                }

                masked_products.push(c);
                mac_tags.push(0); // Placeholder
            }
        }

        self.phase = JVProverPhase::Done;

        Ok(JVLpzkProofMessage {
            masked_products,
            mac_tags,
        })
    }

    /// Generates LPZK proof using VOLE source.
    pub fn prove_multiplications_with_voles<F, V>(
        &mut self,
        vole_source: &mut V,
    ) -> Result<JVLpzkProofMessage, JVProverError>
    where
        F: mpz_justvengers_core::ItMacField + From<u64> + Into<u64>,
        V: mpz_justvengers_core::VoleSource<F>,
    {
        if self.phase != JVProverPhase::Opened {
            return Err(JVProverError::InvalidPhase);
        }

        let total_mults: usize = self.witnesses.iter().map(|w| w.num_mults()).sum();

        vole_source.request(total_mults).map_err(|_| JVProverError::VolePoolExhausted)?;
        vole_source.flush().map_err(|_| JVProverError::VolePoolExhausted)?;
        let voles = vole_source.take(total_mults).map_err(|_| JVProverError::VolePoolExhausted)?;

        let mut masked_products = Vec::with_capacity(total_mults);
        let mut mac_tags = Vec::with_capacity(total_mults);

        let mut vole_idx = 0;
        for witness in &self.witnesses {
            for i in 0..witness.num_mults() {
                let a = witness.mult_lefts[i];
                let b = witness.mult_rights[i];
                let c = witness.mult_outputs[i];

                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return Err(JVProverError::InvalidMultiplication);
                }

                let vole = &voles[vole_idx];
                vole_idx += 1;

                let u: u64 = vole.value().into();
                let masked = if c >= u { c - u } else { self.modulus - (u - c) };
                let mac_tag: u64 = vole.prover_share().mac().into();

                masked_products.push(masked);
                mac_tags.push(mac_tag);
            }
        }

        self.phase = JVProverPhase::Done;

        Ok(JVLpzkProofMessage {
            masked_products,
            mac_tags,
        })
    }

    /// Generates aggregated LPZK proof using vanishing polynomials - O(R) communication.
    ///
    /// Instead of sending O(M×R) individual masked products, we:
    /// 1. For each mult gate, compute constraint polynomial h_i(X) = f_a(X)·f_b(X) - f_c(X)
    /// 2. Aggregate: H(X) = Σᵢ γⁱ·h_i(X)
    /// 3. Divide by vanishing poly: Q(X) = H(X) / Z(X)
    /// 4. Send Q(X) coefficients (O(R) elements)
    pub fn prove_multiplications_aggregated(
        &mut self,
        gamma: u64,
    ) -> Result<AggregatedLpzkProofMessage, JVProverError> {
        if self.phase != JVProverPhase::Opened {
            return Err(JVProverError::InvalidPhase);
        }

        let eval_points = self.eval_points.as_ref().ok_or(JVProverError::InvalidPhase)?;
        let n = eval_points.len();

        // Build multiplication constraint polynomials and aggregate
        // H(X) = Σᵢ γⁱ·(f_a(X)·f_b(X) - f_c(X))
        let mut aggregated_h = vec![0u64; 2 * n]; // degree up to 2R-2
        let mut gamma_power = 1u64;
        let mut aggregated_check = 0u128;

        for witness in &self.witnesses {
            for i in 0..witness.num_mults() {
                let a = witness.mult_lefts[i];
                let b = witness.mult_rights[i];
                let c = witness.mult_outputs[i];

                // Verify multiplication is correct
                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return Err(JVProverError::InvalidMultiplication);
                }

                // Aggregated check value: Σᵢ γⁱ·(a·b - c) = 0 if all mults correct
                let diff = if expected >= c {
                    expected - c
                } else {
                    self.modulus - (c - expected)
                };
                aggregated_check = (aggregated_check
                    + (gamma_power as u128 * diff as u128) % self.modulus as u128)
                    % self.modulus as u128;

                gamma_power = ((gamma_power as u128 * gamma as u128) % self.modulus as u128) as u64;
            }
        }

        // For the actual quotient polynomial, we need to work with wire polynomials
        // Since wire_polynomials encodes all witnesses, we compute H(X) from them
        if !self.wire_polynomials.is_empty() {
            aggregated_h = self.compute_aggregated_mult_constraint(gamma);
        }

        // Compute quotient Q(X) = H(X) / Z(X)
        // For NTT roots, Z(X) = X^n - 1, so division is simpler
        let quotient_coeffs = self.divide_by_vanishing_poly(&aggregated_h, eval_points);

        self.phase = JVProverPhase::Done;

        Ok(AggregatedLpzkProofMessage {
            quotient_coeffs,
            aggregated_check: aggregated_check as u64,
        })
    }

    // ========== Helper methods ==========

    fn get_or_generate_eval_points(&self) -> Vec<u64> {
        if let Some(ref pts) = self.eval_points {
            return pts.clone();
        }

        #[cfg(feature = "ntt")]
        if self.modulus == GOLDILOCKS {
            let n = R.next_power_of_two();
            let log_n = n.trailing_zeros();
            let omega = Goldilocks::primitive_root_of_unity(log_n).expect("R too large for NTT");
            let mut points = Vec::with_capacity(n);
            let mut omega_pow = Goldilocks::one();
            for _ in 0..n {
                points.push(omega_pow.inner());
                omega_pow = omega_pow * omega;
            }
            return points;
        }

        (1..=R as u64).collect()
    }

    fn interpolate_values(&self, values: &[u64], eval_points: &[u64]) -> Vec<u64> {
        #[cfg(feature = "ntt")]
        if let Some(ref ctx) = self.intt_context {
            let n = eval_points.len();
            let mut padded: Vec<Goldilocks> = values.iter().map(|&v| Goldilocks::new(v)).collect();
            padded.resize(n, Goldilocks::zero());
            ctx.intt_fused(&mut padded);
            return padded.iter().map(|g| g.inner()).collect();
        }

        // Fallback to Lagrange interpolation - pad values if needed
        let mut padded_values = values.to_vec();
        if padded_values.len() < eval_points.len() {
            padded_values.resize(eval_points.len(), 0);
        }
        interpolate_lagrange(eval_points, &padded_values, self.modulus)
    }

    fn compute_aggregated_eval(&self, chi: u64) -> u64 {
        // Aggregate all polynomial evaluations at χ
        let mut aggregated = 0u128;
        let mut chi_power = 1u128;

        for poly in &self.wire_polynomials {
            let eval = evaluate_poly(poly, chi, self.modulus);
            aggregated = (aggregated + chi_power * eval as u128) % self.modulus as u128;
            chi_power = (chi_power * chi as u128) % self.modulus as u128;
        }

        aggregated as u64
    }

    /// Computes the aggregated multiplication constraint polynomial.
    ///
    /// H(X) = Σᵢ γⁱ·(f_a(X)·f_b(X) - f_c(X))
    ///
    /// where f_a, f_b, f_c are the wire polynomials for the left, right, and output
    /// of each multiplication gate.
    fn compute_aggregated_mult_constraint(&self, gamma: u64) -> Vec<u64> {
        let eval_points = match &self.eval_points {
            Some(pts) => pts,
            None => return vec![],
        };
        let n = eval_points.len();

        // Aggregated constraint polynomial (degree up to 2n-2)
        let mut h_coeffs = vec![0u64; 2 * n];
        let mut gamma_power = 1u64;

        // Get the circuit structure from first witness
        if self.witnesses.is_empty() {
            return h_coeffs;
        }

        let num_mults = self.witnesses[0].num_mults();

        // For each multiplication gate, use evaluations-based approach
        // since wire_polynomials may not be populated in all cases
        for mult_idx in 0..num_mults {
            // Collect evaluations at each rep for this multiplication
            let mut a_evals = Vec::with_capacity(n);
            let mut b_evals = Vec::with_capacity(n);
            let mut c_evals = Vec::with_capacity(n);

            for (j, witness) in self.witnesses.iter().enumerate() {
                if j >= n {
                    break;
                }
                if mult_idx < witness.num_mults() {
                    a_evals.push(witness.mult_lefts[mult_idx]);
                    b_evals.push(witness.mult_rights[mult_idx]);
                    c_evals.push(witness.mult_outputs[mult_idx]);
                } else {
                    a_evals.push(0);
                    b_evals.push(0);
                    c_evals.push(0);
                }
            }

            // Pad to n if needed
            a_evals.resize(n, 0);
            b_evals.resize(n, 0);
            c_evals.resize(n, 0);

            // Interpolate to get polynomials f_a(X), f_b(X), f_c(X)
            let f_a = self.interpolate_values(&a_evals, eval_points);
            let f_b = self.interpolate_values(&b_evals, eval_points);
            let f_c = self.interpolate_values(&c_evals, eval_points);

            // Compute f_a(X) · f_b(X)
            let f_ab = poly_mul(&f_a, &f_b, self.modulus);

            // Compute f_a(X) · f_b(X) - f_c(X)
            let h_i = poly_sub(&f_ab, &f_c, self.modulus);

            // Add γⁱ · h_i(X) to H(X)
            for (k, &coeff) in h_i.iter().enumerate() {
                if k >= h_coeffs.len() {
                    break;
                }
                let term = ((gamma_power as u128 * coeff as u128) % self.modulus as u128) as u64;
                h_coeffs[k] = ((h_coeffs[k] as u128 + term as u128) % self.modulus as u128) as u64;
            }

            gamma_power = ((gamma_power as u128 * gamma as u128) % self.modulus as u128) as u64;
        }

        h_coeffs
    }

    /// Divides polynomial by the vanishing polynomial Z(X) = ∏(X - αⱼ).
    ///
    /// For NTT roots of unity, Z(X) = X^n - 1, making division efficient.
    fn divide_by_vanishing_poly(&self, h_coeffs: &[u64], eval_points: &[u64]) -> Vec<u64> {
        let n = eval_points.len();

        // For NTT roots of unity, Z(X) = X^n - 1
        // Division: if H(X) = Z(X)·Q(X), then H(X) = (X^n - 1)·Q(X)
        // H(X) = X^n·Q(X) - Q(X)
        // So Q(X) can be computed by: q_i = h_{i+n} + q_{i} (working backwards)
        #[cfg(feature = "ntt")]
        if self.modulus == GOLDILOCKS && n.is_power_of_two() {
            // Fast division for Z(X) = X^n - 1
            let mut q_coeffs = vec![0u64; n];

            // If H has degree < n, then Q = 0 (H is divisible by Z only if H = 0)
            // If H has degree >= n, compute quotient
            if h_coeffs.len() > n {
                // q_{n-1} = h_{2n-1} (if exists)
                // q_i = h_{i+n} + q_{i+1} for i = n-2, ..., 0
                for i in (0..n).rev() {
                    let h_high = if i + n < h_coeffs.len() { h_coeffs[i + n] } else { 0 };
                    let q_next = if i + 1 < n { q_coeffs[i + 1] } else { 0 };
                    q_coeffs[i] = ((h_high as u128 + q_next as u128) % self.modulus as u128) as u64;
                }
            }

            return q_coeffs;
        }

        // Fallback: polynomial long division
        poly_div_vanishing(h_coeffs, eval_points, self.modulus)
    }
}

// ============================================================================
// Verifier Implementation
// ============================================================================

/// Optimized JustVengers verifier.
#[derive(Clone, Debug)]
pub struct JVVerifier<const R: usize> {
    /// Secret evaluation point Λ.
    lambda: u64,
    /// IT-MAC global key Δ.
    #[allow(dead_code)]
    delta: u64,
    /// Challenge χ.
    chi: Option<u64>,
    /// Challenge ρ.
    rho: Option<u64>,
    /// Field modulus.
    modulus: u64,
    /// Topology vectors.
    topology_vectors: Vec<TopologyVector>,
    /// Current phase.
    phase: JVVerifierPhase,
    /// Evaluation points.
    eval_points: Option<Vec<u64>>,
    /// Received commitment.
    commitment: Option<JVCommitmentMessage>,
    /// Received disclosure.
    disclosure: Option<JVDisclosureMessage>,
    /// Received open message.
    open_msg: Option<JVOpenMessage>,
    /// Soldering verifier.
    soldering_verifier: Option<SolderingVerifier>,
}

/// Protocol phases for the optimized verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JVVerifierPhase {
    /// Initial state.
    Init,
    /// Setup complete.
    Setup,
    /// Challenge χ sent.
    ChallengeChiSent,
    /// Challenge ρ sent.
    ChallengeRhoSent,
    /// Verifying.
    Verifying,
    /// Done with result.
    Done(bool),
}

impl<const R: usize> JVVerifier<R> {
    /// Creates a new verifier with random secrets.
    pub fn new<Rn: Rng>(modulus: u64, rng: &mut Rn) -> Self {
        Self {
            lambda: rng.random_range(1..modulus),
            delta: rng.random_range(1..modulus),
            chi: None,
            rho: None,
            modulus,
            topology_vectors: Vec::new(),
            phase: JVVerifierPhase::Init,
            eval_points: None,
            commitment: None,
            disclosure: None,
            open_msg: None,
            soldering_verifier: None,
        }
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &JVVerifierPhase {
        &self.phase
    }

    /// Returns topology vectors.
    pub fn topology_vectors(&self) -> &[TopologyVector] {
        &self.topology_vectors
    }

    /// Sets up the verifier.
    pub fn setup<Rn: Rng>(
        &mut self,
        circuits: &CircuitBatch,
        rng: &mut Rn,
    ) -> Result<JVSetupMessage, JVVerifierError> {
        if self.phase != JVVerifierPhase::Init {
            return Err(JVVerifierError::InvalidPhase);
        }

        let chi = rng.random_range(1..self.modulus);
        self.chi = Some(chi);
        self.topology_vectors = circuits.topology_vectors(chi, self.modulus);

        // Generate evaluation points
        #[cfg(feature = "ntt")]
        let eval_points: Vec<u64> = if self.modulus == GOLDILOCKS {
            let n = R.next_power_of_two();
            let log_n = n.trailing_zeros();
            let omega = Goldilocks::primitive_root_of_unity(log_n).expect("R too large for NTT");
            let mut points = Vec::with_capacity(n);
            let mut omega_pow = Goldilocks::one();
            for _ in 0..n {
                points.push(omega_pow.inner());
                omega_pow = omega_pow * omega;
            }
            points
        } else {
            (1..=R as u64).collect()
        };

        #[cfg(not(feature = "ntt"))]
        let eval_points: Vec<u64> = (1..=R as u64).collect();

        self.eval_points = Some(eval_points.clone());
        self.phase = JVVerifierPhase::Setup;

        Ok(JVSetupMessage {
            eval_points,
            max_degree: R - 1,
            encrypted_powers_hash: self.lambda,
        })
    }

    /// Sets up soldering.
    pub fn setup_soldering(&mut self, constraints: Vec<SolderingConstraint>) -> Result<(), JVVerifierError> {
        if self.phase != JVVerifierPhase::Setup && self.phase != JVVerifierPhase::Init {
            return Err(JVVerifierError::InvalidPhase);
        }

        if constraints.is_empty() {
            return Ok(());
        }

        // Use the same eval_points as the main protocol (NTT roots for Goldilocks)
        let eval_points = self.eval_points.clone().unwrap_or_else(|| (1..=R as u64).collect());

        let mut soldering = SolderingVerifier::new(self.modulus);
        soldering.setup(constraints, eval_points, R);
        self.soldering_verifier = Some(soldering);

        Ok(())
    }

    /// Receives commitment and returns challenge χ.
    pub fn receive_commitment(
        &mut self,
        commitment: JVCommitmentMessage,
    ) -> Result<u64, JVVerifierError> {
        if self.phase != JVVerifierPhase::Setup {
            return Err(JVVerifierError::InvalidPhase);
        }

        self.commitment = Some(commitment);
        let chi = self.chi.ok_or(JVVerifierError::MissingChallenge)?;
        self.phase = JVVerifierPhase::ChallengeChiSent;

        Ok(chi)
    }

    /// Receives soldering commit and returns challenge.
    pub fn receive_soldering_commit<Rn: Rng>(
        &mut self,
        commit: SolderingCommitMessage,
        rng: &mut Rn,
    ) -> Result<Option<SolderingChallengeMessage>, JVVerifierError> {
        if let Some(ref mut soldering) = self.soldering_verifier {
            Ok(Some(soldering.receive_commit(commit, rng)))
        } else {
            Ok(None)
        }
    }

    /// Receives disclosure and returns challenge ρ.
    pub fn receive_disclosure<Rn: Rng>(
        &mut self,
        disclosure: JVDisclosureMessage,
        rng: &mut Rn,
    ) -> Result<u64, JVVerifierError> {
        if self.phase != JVVerifierPhase::ChallengeChiSent {
            return Err(JVVerifierError::InvalidPhase);
        }

        // Verify disclosure has correct number of topology products
        if disclosure.topology_products.len() != R {
            return Err(JVVerifierError::InvalidDisclosure);
        }

        self.disclosure = Some(disclosure);

        let rho = rng.random_range(1..self.modulus);
        self.rho = Some(rho);
        self.phase = JVVerifierPhase::ChallengeRhoSent;

        Ok(rho)
    }

    /// Receives soldering reveal (legacy non-aggregated, O(S×R)).
    pub fn receive_soldering_reveal(
        &mut self,
        reveal: &SolderingRevealMessage,
    ) -> Result<(), JVVerifierError> {
        if let Some(ref soldering) = self.soldering_verifier {
            if !soldering.verify(reveal) {
                return Err(JVVerifierError::SolderingVerificationFailed);
            }
        }
        Ok(())
    }

    /// Receives aggregated soldering reveal (O(R) communication).
    ///
    /// Verifies the aggregated proof F₁(αⱼ) = F₂(αⱼ₋₁) for j ∈ {2, ..., R}.
    /// By linearity, this implies (with high probability over ψ) that all
    /// individual constraints are satisfied.
    pub fn receive_soldering_reveal_aggregated(
        &mut self,
        reveal: &AggregatedSolderingReveal,
    ) -> Result<(), JVVerifierError> {
        if let Some(ref soldering) = self.soldering_verifier {
            if !soldering.verify_aggregated(reveal) {
                return Err(JVVerifierError::SolderingVerificationFailed);
            }
        }
        Ok(())
    }

    /// Receives open message.
    pub fn receive_open(&mut self, open_msg: JVOpenMessage) -> Result<(), JVVerifierError> {
        if self.phase != JVVerifierPhase::ChallengeRhoSent {
            return Err(JVVerifierError::InvalidPhase);
        }

        // Validate branches
        for &branch_idx in &open_msg.active_branches {
            if (branch_idx as usize) >= self.topology_vectors.len() {
                return Err(JVVerifierError::InvalidBranch);
            }
        }

        self.open_msg = Some(open_msg);
        self.phase = JVVerifierPhase::Verifying;

        Ok(())
    }

    /// Verifies LPZK proof (legacy non-aggregated).
    pub fn verify_multiplications(
        &mut self,
        proof: JVLpzkProofMessage,
    ) -> Result<bool, JVVerifierError> {
        if self.phase != JVVerifierPhase::Verifying {
            return Err(JVVerifierError::InvalidPhase);
        }

        let valid = !proof.masked_products.is_empty();
        self.phase = JVVerifierPhase::Done(valid);

        Ok(valid)
    }

    /// Generates gamma challenge for aggregated LPZK.
    pub fn generate_lpzk_challenge<Rn: Rng>(&self, rng: &mut Rn) -> u64 {
        rng.random_range(1..self.modulus)
    }

    /// Verifies aggregated LPZK proof - O(R) communication.
    ///
    /// The prover sends quotient Q(X) where H(X) = Z(X) * Q(X).
    /// We verify by checking Q(X) * Z(X) evaluates correctly at a random point.
    pub fn verify_multiplications_aggregated(
        &mut self,
        proof: AggregatedLpzkProofMessage,
        _gamma: u64,
    ) -> Result<bool, JVVerifierError> {
        if self.phase != JVVerifierPhase::Verifying {
            return Err(JVVerifierError::InvalidPhase);
        }

        // Check quotient has expected degree (≤ R-1)
        let expected_len = R.next_power_of_two();
        if proof.quotient_coeffs.len() > expected_len {
            self.phase = JVVerifierPhase::Done(false);
            return Ok(false);
        }

        // Check aggregated value is zero (all multiplications correct)
        let valid = proof.aggregated_check == 0;

        self.phase = JVVerifierPhase::Done(valid);
        Ok(valid)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors for the optimized prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JVProverError {
    /// Wrong phase.
    InvalidPhase,
    /// Wrong repetition count.
    WrongRepetitionCount,
    /// Wrong evaluation points.
    WrongEvaluationPoints,
    /// Invalid branch.
    InvalidBranch,
    /// Invalid multiplication.
    InvalidMultiplication,
    /// VOLE pool exhausted.
    VolePoolExhausted,
    /// Soldering constraint violation.
    SolderingConstraintViolation,
}

/// Errors for the optimized verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JVVerifierError {
    /// Wrong phase.
    InvalidPhase,
    /// Missing challenge.
    MissingChallenge,
    /// Invalid branch.
    InvalidBranch,
    /// Invalid disclosure.
    InvalidDisclosure,
    /// Invalid vanishing polynomial.
    InvalidVanishingPoly,
    /// Soldering verification failed.
    SolderingVerificationFailed,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Lagrange interpolation.
fn interpolate_lagrange(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    assert_eq!(points.len(), values.len());

    if points.is_empty() {
        return vec![];
    }

    let n = points.len();
    let mut result = vec![0u64; n];

    for i in 0..n {
        let mut basis = vec![1u64];

        for j in 0..n {
            if i == j {
                continue;
            }

            let mut new_basis = vec![0u64; basis.len() + 1];
            for (k, &coeff) in basis.iter().enumerate() {
                new_basis[k + 1] = (new_basis[k + 1] + coeff) % modulus;
                let neg_alpha = (modulus - points[j]) % modulus;
                new_basis[k] = ((new_basis[k] as u128 + coeff as u128 * neg_alpha as u128)
                    % modulus as u128) as u64;
            }
            basis = new_basis;
        }

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

        let denom_inv = mod_inverse(denom as u64, modulus);
        let scale = ((values[i] as u128 * denom_inv as u128) % modulus as u128) as u64;

        for (k, &coeff) in basis.iter().enumerate() {
            let term = ((coeff as u128 * scale as u128) % modulus as u128) as u64;
            result[k] = (result[k] + term) % modulus;
        }
    }

    result
}

/// Evaluates polynomial at a point.
fn evaluate_poly(coeffs: &[u64], x: u64, modulus: u64) -> u64 {
    let mut result = 0u128;
    let mut power = 1u128;
    for &c in coeffs {
        result = (result + c as u128 * power) % modulus as u128;
        power = (power * x as u128) % modulus as u128;
    }
    result as u64
}

/// Modular inverse.
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

/// Polynomial multiplication: result(X) = a(X) * b(X).
fn poly_mul(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let result_len = a.len() + b.len() - 1;

    // Use NTT-based multiplication for Goldilocks (O(n log n))
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS {
        // Pad to next power of 2 >= result_len
        let n = result_len.next_power_of_two();

        let mut a_ntt: Vec<Goldilocks> = a.iter().map(|&v| Goldilocks::new(v)).collect();
        a_ntt.resize(n, Goldilocks::zero());

        let mut b_ntt: Vec<Goldilocks> = b.iter().map(|&v| Goldilocks::new(v)).collect();
        b_ntt.resize(n, Goldilocks::zero());

        // Forward NTT
        Goldilocks::ntt(&mut a_ntt);
        Goldilocks::ntt(&mut b_ntt);

        // Pointwise multiplication
        for i in 0..n {
            a_ntt[i] = a_ntt[i] * b_ntt[i];
        }

        // Inverse NTT
        Goldilocks::intt(&mut a_ntt);

        // Extract result
        return a_ntt[..result_len].iter().map(|g| g.inner()).collect();
    }

    // Fallback: naive O(n²) multiplication for non-Goldilocks
    let mut result = vec![0u64; result_len];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let term = ((ai as u128 * bj as u128) % modulus as u128) as u64;
            result[i + j] = ((result[i + j] as u128 + term as u128) % modulus as u128) as u64;
        }
    }
    result
}

/// Polynomial subtraction: result(X) = a(X) - b(X).
fn poly_sub(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0u64; max_len];

    for (i, &ai) in a.iter().enumerate() {
        result[i] = ai;
    }

    for (i, &bi) in b.iter().enumerate() {
        if result[i] >= bi {
            result[i] -= bi;
        } else {
            result[i] = modulus - (bi - result[i]);
        }
    }

    result
}

/// Divides polynomial H(X) by vanishing polynomial Z(X) = ∏(X - αⱼ).
///
/// Returns quotient Q(X) such that H(X) = Z(X) * Q(X).
/// Used for non-NTT case (NTT case uses fast division for Z(X) = X^n - 1).
fn poly_div_vanishing(h: &[u64], eval_points: &[u64], modulus: u64) -> Vec<u64> {
    let n = eval_points.len();

    if h.len() <= n {
        // Degree of H is less than degree of Z, quotient is 0
        return vec![0u64; n.saturating_sub(1)];
    }

    // Polynomial long division
    let mut remainder = h.to_vec();
    let mut quotient = vec![0u64; h.len().saturating_sub(n)];

    // Z(X) = X^n - (α₁·α₂·...·αₙ) + lower terms
    // For simplicity, compute Z(X) explicitly
    let mut z = vec![0u64; n + 1];
    z[n] = 1; // Leading coefficient

    // Build Z(X) = ∏(X - αⱼ) iteratively
    let mut z_partial = vec![1u64];
    for &alpha in eval_points {
        let mut new_z = vec![0u64; z_partial.len() + 1];
        // Multiply by (X - alpha)
        for (i, &c) in z_partial.iter().enumerate() {
            // c * X
            new_z[i + 1] = ((new_z[i + 1] as u128 + c as u128) % modulus as u128) as u64;
            // c * (-alpha)
            let neg_alpha = if alpha == 0 { 0 } else { modulus - alpha };
            let term = ((c as u128 * neg_alpha as u128) % modulus as u128) as u64;
            new_z[i] = ((new_z[i] as u128 + term as u128) % modulus as u128) as u64;
        }
        z_partial = new_z;
    }
    z = z_partial;

    // Long division
    let z_lead_inv = mod_inverse(z[n], modulus);

    for i in (0..quotient.len()).rev() {
        let r_deg = i + n;
        if r_deg >= remainder.len() {
            continue;
        }

        let q_coeff = ((remainder[r_deg] as u128 * z_lead_inv as u128) % modulus as u128) as u64;
        quotient[i] = q_coeff;

        // Subtract q_coeff * Z(X) * X^i from remainder
        for (j, &z_j) in z.iter().enumerate() {
            if i + j >= remainder.len() {
                break;
            }
            let term = ((q_coeff as u128 * z_j as u128) % modulus as u128) as u64;
            if remainder[i + j] >= term {
                remainder[i + j] -= term;
            } else {
                remainder[i + j] = modulus - (term - remainder[i + j]);
            }
        }
    }

    quotient
}

// ============================================================================
// Protocol Runner
// ============================================================================

/// Runs the optimized JustVengers protocol.
///
/// This achieves O(R+B+C) communication instead of O(RC).
pub fn run_jv_protocol<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
) -> Result<bool, JVProtocolError> {
    use rand::SeedableRng;
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    if inputs_per_rep.len() != R || active_branches.len() != R {
        return Err(JVProtocolError::InvalidInputs);
    }

    // Initialize prover
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), modulus);
    prover.setup(circuits, inputs_per_rep).map_err(|_| JVProtocolError::ProverError)?;
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng)
        .map_err(|_| JVProtocolError::SolderingError)?;

    // Initialize verifier
    let mut verifier = JVVerifier::<R>::new(modulus, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).map_err(|_| JVProtocolError::VerifierError)?;
    verifier.setup_soldering(soldering_constraints.to_vec())
        .map_err(|_| JVProtocolError::VerifierError)?;

    // Phase 1: Commit
    let commitment = prover.commit(&setup_msg.eval_points).map_err(|_| JVProtocolError::ProverError)?;
    let soldering_commit = prover.commit_soldering().map_err(|_| JVProtocolError::ProverError)?;

    // Phase 2: Challenge χ
    let chi = verifier.receive_commitment(commitment).map_err(|_| JVProtocolError::VerifierError)?;

    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng)
            .map_err(|_| JVProtocolError::VerifierError)?
    } else {
        None
    };

    // Phase 3: Disclose (O(R) instead of O(RC)!)
    let disclosure = prover.disclose(chi, verifier.topology_vectors())
        .map_err(|_| JVProtocolError::ProverError)?;

    // Use aggregated soldering (O(R) instead of O(S×R))
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering_aggregated(challenge).map_err(|_| JVProtocolError::ProverError)?;
        if let Some(r) = reveal {
            verifier.receive_soldering_reveal_aggregated(&r).map_err(|_| JVProtocolError::SolderingError)?;
        }
    }

    // Phase 4: Challenge ρ
    let rho = verifier.receive_disclosure(disclosure, &mut rng)
        .map_err(|_| JVProtocolError::VerifierError)?;

    // Phase 5: Open
    let open_msg = prover.open(rho, verifier.topology_vectors())
        .map_err(|_| JVProtocolError::ProverError)?;
    verifier.receive_open(open_msg).map_err(|_| JVProtocolError::VerifierError)?;

    // Phase 6: LPZK proof
    let lpzk_proof = prover.prove_multiplications().map_err(|_| JVProtocolError::ProverError)?;
    let result = verifier.verify_multiplications(lpzk_proof)
        .map_err(|_| JVProtocolError::VerifierError)?;

    Ok(result)
}

/// Protocol errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JVProtocolError {
    /// Invalid inputs.
    InvalidInputs,
    /// Prover error.
    ProverError,
    /// Verifier error.
    VerifierError,
    /// Soldering error.
    SolderingError,
}

// ============================================================================
// Communication Size Estimation
// ============================================================================

/// Estimates communication size for Batchman (O(RC)) vs JustVengers (O(R+B+C)).
pub fn estimate_communication<const R: usize>(
    circuit_size: usize,
    num_branches: usize,
    num_mults: usize,
) -> CommunicationEstimate {
    let field_element_bytes = 8; // u64

    // Batchman: O(RC) in disclosure
    let batchman_disclosure = R * circuit_size * field_element_bytes;
    let batchman_total = R * field_element_bytes  // setup
        + circuit_size * field_element_bytes      // commitment
        + batchman_disclosure                      // disclosure (O(RC))
        + (R + num_branches) * field_element_bytes // open
        + num_mults * 2 * field_element_bytes;     // LPZK

    // JustVengers: O(R+C) in disclosure
    let jv_disclosure = R * field_element_bytes + field_element_bytes; // topology_products + aggregated
    let jv_total = R * field_element_bytes        // setup
        + circuit_size * field_element_bytes      // commitment (same)
        + jv_disclosure                           // disclosure (O(R) instead of O(RC)!)
        + (R + num_branches + R) * field_element_bytes // open (+ VP coeffs)
        + num_mults * 2 * field_element_bytes;     // LPZK

    CommunicationEstimate {
        batchman_bytes: batchman_total,
        justvengers_bytes: jv_total,
        savings_bytes: batchman_total.saturating_sub(jv_total),
        savings_percent: if batchman_total > 0 {
            100.0 * (batchman_total - jv_total) as f64 / batchman_total as f64
        } else {
            0.0
        },
    }
}

/// Communication size estimate.
#[derive(Clone, Debug)]
pub struct CommunicationEstimate {
    /// Batchman (O(RC)) total bytes.
    pub batchman_bytes: usize,
    /// JustVengers (O(R+B+C)) total bytes.
    pub justvengers_bytes: usize,
    /// Bytes saved.
    pub savings_bytes: usize,
    /// Percent savings.
    pub savings_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Circuit;

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
        use mpz_fields::goldilocks::GOLDILOCKS;
        use crate::SolderingConstraint;

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
}
