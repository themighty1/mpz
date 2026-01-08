//! Prover implementation for Justvengers ZK protocol.
//!
//! # Communication Complexity Note
//!
//! **This module implements the "Batchman-style" O(RC) communication approach.**
//!
//! The DisclosureMessage sends `masked_values` which contains all R×C witness values.
//! This results in O(RC) communication where R is repetitions and C is circuit size.
//!
//! For the optimized O(R+B+C) JustVengers protocol that uses polynomial encoding
//! via IT-PAC, see the `jv_optimized` module. The key difference:
//! - Batchman (this module): sends R×C individual masked values
//! - JustVengers (jv_optimized): encodes R values per wire as polynomial, sends O(C)
//!   polynomial commitments + O(R) vanishing polynomial coefficients
//!
//! We keep both implementations as they have different trade-offs:
//! - Batchman: simpler, no AHE overhead, good for small R
//! - JustVengers: better asymptotic complexity, requires AHE setup
//!
//! Implements the 5-phase prover from the Justvengers paper:
//!
//! 1. **Initialization**: Receive encrypted powers and setup VOLE pool
//! 2. **Commitment**: Commit to extended witness polynomials using IT-PAC
//! 3. **Disclosure**: Receive challenge χ, compute topology vectors, send masked values
//! 4. **Open**: Receive challenge ρ, compute universal hash response
//! 5. **Verification**: Generate LPZK proof for multiplication triples
//!
//! # Protocol Overview
//!
//! The prover holds:
//! - Private input x for one of B circuits (branch b*)
//! - Extended witness w = (inputs, mult_lefts, mult_rights, mult_outputs)
//!
//! The prover generates IT-PAC commitments [f_i(·)] for each witness component,
//! where f_i encodes R repetitions of the value using polynomial interpolation.
//!
//! After the verifier sends challenges, the prover proves:
//! 1. The witness satisfies topology constraints (via universal hash)
//! 2. Multiplication gates are correct (via LPZK)
//! 3. The vanishing polynomial property holds for non-executed branches

use crate::soldering::{
    SolderingChallengeMessage, SolderingCommitMessage, SolderingConstraint, SolderingProver,
    SolderingRevealMessage,
};
use crate::topology::{Circuit, ExtendedWitness, TopologyVector, UniversalHash};

#[cfg(feature = "ntt")]
use mpz_fields::goldilocks::{Goldilocks, InttContext, GOLDILOCKS};
#[cfg(feature = "ntt")]
use mpz_fields::Field;

/// Prover state during protocol execution.
#[derive(Clone, Debug)]
pub struct ProverState<const R: usize> {
    /// The active branch indices, one per repetition: id_j ∈ [0, B) for j ∈ [R].
    /// Each repetition can execute a different branch (as per JV paper Section 4.3).
    active_branches: Vec<usize>,
    /// Extended witnesses for all R repetitions.
    witnesses: Vec<ExtendedWitness>,
    /// Field modulus.
    modulus: u64,
    /// Current protocol phase.
    phase: ProverPhase,
    /// Evaluation points α₁, ..., αᵣ used for polynomial interpolation.
    /// When using Goldilocks field with NTT, these are roots of unity.
    eval_points: Option<Vec<u64>>,
    /// Precomputed INTT context for fast inverse NTT (Goldilocks only).
    #[cfg(feature = "ntt")]
    intt_context: Option<InttContext>,
    /// Soldering prover for cross-repetition constraints.
    soldering_prover: Option<SolderingProver>,
}

/// Protocol phases for the prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProverPhase {
    /// Initial state before receiving setup.
    Init,
    /// After initialization, before commitment.
    Setup,
    /// After commitment, awaiting challenge χ.
    Committed,
    /// After disclosure, awaiting challenge ρ.
    Disclosed,
    /// After opening, verification in progress.
    Opened,
    /// Protocol complete.
    Done,
}

/// Messages sent by the prover.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProverMessage {
    /// Commitment message: encrypted polynomial evaluations.
    Commitment(CommitmentMessage),
    /// Disclosure message: masked witness polynomials at Λ.
    Disclosure(DisclosureMessage),
    /// Open message: universal hash response.
    Open(OpenMessage),
    /// LPZK proof for multiplication verification.
    LpzkProof(LpzkProofMessage),
}

/// Commitment phase message.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommitmentMessage {
    /// Number of IT-PAC commitments (one per witness component).
    pub num_commitments: usize,
    /// Encrypted masked polynomial evaluations ⟦f_i(Λ) - u_i⟧.
    pub masked_evaluations: Vec<u64>,
}

/// Disclosure phase message.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DisclosureMessage {
    /// Evaluation points α₁, ..., αᵣ used for interpolation.
    pub eval_points: Vec<u64>,
    /// Masked witness values at challenge χ.
    pub masked_values: Vec<u64>,
    /// Claimed topology vector inner products.
    pub topology_products: Vec<u64>,
}

/// Open phase message.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpenMessage {
    /// Active branch indices, one per repetition: id_j ∈ [0, B) for j ∈ [R].
    pub active_branches: Vec<usize>,
    /// Universal hash proof component.
    pub hash_proof: u64,
    /// Vanishing polynomial coefficients for non-active branches.
    pub vanishing_coeffs: Vec<Vec<u64>>,
}

/// LPZK proof message for multiplication verification.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LpzkProofMessage {
    /// Masked multiplication products.
    pub masked_products: Vec<u64>,
    /// MAC tags for verification.
    pub mac_tags: Vec<u64>,
}

impl<const R: usize> ProverState<R> {
    /// Creates a new prover with a single active branch for all repetitions.
    ///
    /// This is a convenience constructor for the common case where all repetitions
    /// execute the same branch.
    ///
    /// # Arguments
    /// * `active_branch` - The branch index b* ∈ [0, B) to use for all repetitions
    /// * `modulus` - The field modulus p
    pub fn new(active_branch: usize, modulus: u64) -> Self {
        Self {
            active_branches: vec![active_branch; R],
            witnesses: Vec::new(),
            modulus,
            phase: ProverPhase::Init,
            eval_points: None,
            #[cfg(feature = "ntt")]
            intt_context: None,
            soldering_prover: None,
        }
    }

    /// Creates a new prover with per-repetition active branches.
    ///
    /// As per JV paper Section 4.3, each repetition j can execute a different
    /// branch id_j ∈ [0, B).
    ///
    /// # Arguments
    /// * `active_branches` - Branch indices, one per repetition (must have length R)
    /// * `modulus` - The field modulus p
    ///
    /// # Panics
    /// Panics if `active_branches.len() != R`
    pub fn new_per_rep(active_branches: Vec<usize>, modulus: u64) -> Self {
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
            phase: ProverPhase::Init,
            eval_points: None,
            #[cfg(feature = "ntt")]
            intt_context: None,
            soldering_prover: None,
        }
    }

    /// Returns the active branch index for the first repetition.
    ///
    /// For backwards compatibility. Use `active_branches()` to get all per-rep branches.
    pub fn active_branch(&self) -> usize {
        self.active_branches[0]
    }

    /// Returns all active branch indices, one per repetition.
    pub fn active_branches(&self) -> &[usize] {
        &self.active_branches
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &ProverPhase {
        &self.phase
    }

    /// Returns the field modulus.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Initializes the prover with the circuit and inputs.
    ///
    /// Evaluates the same circuit R times with the given inputs to generate
    /// the extended witnesses. Use `setup_per_rep()` for per-repetition circuits.
    pub fn setup(&mut self, circuit: &mut Circuit, inputs_per_rep: &[Vec<u64>]) -> Result<(), ProverError> {
        if self.phase != ProverPhase::Init {
            return Err(ProverError::InvalidPhase);
        }

        if inputs_per_rep.len() != R {
            return Err(ProverError::WrongRepetitionCount);
        }

        // Evaluate circuit for each repetition
        self.witnesses = inputs_per_rep
            .iter()
            .map(|inputs| circuit.evaluate(inputs, self.modulus))
            .collect();

        self.phase = ProverPhase::Setup;
        Ok(())
    }

    /// Initializes the prover with per-repetition circuits from a batch.
    ///
    /// For each repetition j, evaluates `circuits[active_branches[j]]` with
    /// `inputs_per_rep[j]`. This supports the JV paper's per-repetition
    /// branch selection (id_j ∈ [B] for each j ∈ [R]).
    ///
    /// # Arguments
    /// * `circuits` - The circuit batch containing all B branches
    /// * `inputs_per_rep` - Input vectors, one per repetition
    ///
    /// # Errors
    /// Returns error if:
    /// - Not in Init phase
    /// - `inputs_per_rep.len() != R`
    /// - Any `active_branches[j] >= circuits.num_circuits()`
    pub fn setup_per_rep(
        &mut self,
        circuits: &crate::CircuitBatch,
        inputs_per_rep: &[Vec<u64>],
    ) -> Result<(), ProverError> {
        if self.phase != ProverPhase::Init {
            return Err(ProverError::InvalidPhase);
        }

        if inputs_per_rep.len() != R {
            return Err(ProverError::WrongRepetitionCount);
        }

        // Validate all active branches are valid
        for &branch_idx in &self.active_branches {
            if branch_idx >= circuits.num_branches() {
                return Err(ProverError::InvalidBranch);
            }
        }

        // Evaluate each repetition's circuit with its inputs
        self.witnesses = self
            .active_branches
            .iter()
            .zip(inputs_per_rep.iter())
            .map(|(&branch_idx, inputs)| {
                let mut circuit = circuits.get(branch_idx).unwrap().clone();
                circuit.evaluate(inputs, self.modulus)
            })
            .collect();

        self.phase = ProverPhase::Setup;
        Ok(())
    }

    /// Generates commitment message.
    ///
    /// Creates IT-PAC commitments for the extended witness polynomials:
    /// - For each witness component (inputs, mult_lefts, mult_rights, mult_outputs)
    /// - Interpolate R values into degree-(R-1) polynomial
    /// - Commit using IT-PAC
    ///
    /// When using Goldilocks field with NTT feature, uses O(n log n) inverse NTT
    /// instead of O(n²) Lagrange interpolation.
    pub fn commit(&mut self, eval_points: &[u64]) -> Result<CommitmentMessage, ProverError> {
        if self.phase != ProverPhase::Setup {
            return Err(ProverError::InvalidPhase);
        }

        // For Goldilocks with NTT, eval_points may be padded to next_power_of_two(R)
        #[cfg(feature = "ntt")]
        let expected_len = if self.modulus == GOLDILOCKS {
            R.next_power_of_two()
        } else {
            R
        };

        #[cfg(not(feature = "ntt"))]
        let expected_len = R;

        if eval_points.len() != expected_len {
            return Err(ProverError::WrongEvaluationPoints);
        }

        // Store eval_points for later use (e.g., soldering)
        self.eval_points = Some(eval_points.to_vec());

        // Create precomputed INTT context for Goldilocks (reused for all interpolations)
        #[cfg(feature = "ntt")]
        if self.modulus == GOLDILOCKS {
            self.intt_context = Some(InttContext::new(eval_points.len()));
        }

        // For each witness position, interpolate across R repetitions
        let witness_len = self.witnesses[0].len();
        let mut masked_evaluations = Vec::with_capacity(witness_len);

        for pos in 0..witness_len {
            // Collect values at this position across all repetitions
            // Use get() instead of to_vec()[pos] to avoid O(witness_len) allocation per iteration
            let values: Vec<u64> = self.witnesses
                .iter()
                .map(|w| w.get(pos))
                .collect();

            // Interpolate to get polynomial f(·) where f(αᵢ) = values[i]
            // For Goldilocks with NTT: use O(n log n) inverse NTT with precomputed context
            // Otherwise: use O(n²) Lagrange interpolation
            #[cfg(feature = "ntt")]
            let poly = if let Some(ref ctx) = self.intt_context {
                // Use precomputed INTT context (faster)
                let n = eval_points.len();
                let mut padded: Vec<Goldilocks> = values
                    .iter()
                    .map(|&v| Goldilocks::new(v))
                    .collect();
                padded.resize(n, Goldilocks::zero());

                // Fast inverse NTT with precomputed twiddles and fused scaling
                ctx.intt_fused(&mut padded);

                padded.iter().map(|g| g.inner()).collect()
            } else {
                interpolate(eval_points, &values, self.modulus)
            };

            #[cfg(not(feature = "ntt"))]
            let poly = interpolate(eval_points, &values, self.modulus);

            // In real implementation:
            // 1. Use IT-PAC to commit to poly
            // 2. Return encrypted ⟦f(Λ) - u⟧
            //
            // For now, we just record a hash of the polynomial
            let hash = poly.iter().fold(0u64, |acc, &c| {
                ((acc as u128 + c as u128) % self.modulus as u128) as u64
            });
            masked_evaluations.push(hash);
        }

        self.phase = ProverPhase::Committed;

        Ok(CommitmentMessage {
            num_commitments: witness_len,
            masked_evaluations,
        })
    }

    /// Generates disclosure message after receiving challenge χ.
    ///
    /// Computes topology vector inner products and masked values.
    /// Uses per-repetition active branches: for rep j, uses topology_vectors[active_branches[j]].
    pub fn disclose(
        &mut self,
        chi: u64,
        topology_vectors: &[TopologyVector],
    ) -> Result<DisclosureMessage, ProverError> {
        if self.phase != ProverPhase::Committed {
            return Err(ProverError::InvalidPhase);
        }

        // Validate all active branches are within bounds
        for &branch_idx in &self.active_branches {
            if branch_idx >= topology_vectors.len() {
                return Err(ProverError::InvalidBranch);
            }
        }

        // Pre-allocate with capacity to avoid reallocations
        let total_witness_values: usize = self.witnesses.iter().map(|w| w.len()).sum();
        let mut masked_values = Vec::with_capacity(total_witness_values);
        let mut topology_products = Vec::with_capacity(self.witnesses.len());

        // Combined loop: compute topology products and masked values in one pass
        // Uses iter() instead of to_vec() to avoid allocation per witness
        // Each repetition j uses its own active branch: topology_vectors[active_branches[j]]
        for (j, witness) in self.witnesses.iter().enumerate() {
            // Get topology vector for this repetition's active branch
            let active_tv = &topology_vectors[self.active_branches[j]];

            // Compute ⟨t_{id_j}, w^(j)⟩ using to_vec() only once
            let w = witness.to_vec();
            let product = active_tv.inner_product(&w);
            topology_products.push(product);

            // Add masked values from the same vector (no second allocation)
            for val in w {
                masked_values.push((val + chi) % self.modulus);
            }
        }

        // Use the stored evaluation points (could be roots of unity for Goldilocks)
        let eval_points = self.eval_points.clone().unwrap_or_else(|| {
            (1..=R as u64).collect()
        });

        self.phase = ProverPhase::Disclosed;

        Ok(DisclosureMessage {
            eval_points,
            masked_values,
            topology_products,
        })
    }

    /// Generates open message after receiving challenge ρ.
    ///
    /// Proves membership in the set of valid topology vectors using universal hash.
    /// For per-rep active branches, aggregates hash proofs across all repetitions.
    pub fn open(
        &mut self,
        rho: u64,
        topology_vectors: &[TopologyVector],
    ) -> Result<OpenMessage, ProverError> {
        if self.phase != ProverPhase::Disclosed {
            return Err(ProverError::InvalidPhase);
        }

        // Compute universal hash of topology vectors
        let _hash = UniversalHash::compute(topology_vectors, rho, self.modulus);

        // For per-rep active branches, compute aggregate hash proof
        // Sum of ρ^{id_j} · ⟨t_{id_j}, w^(j)⟩ for each repetition j
        let mut hash_proof = 0u128;
        for (j, witness) in self.witnesses.iter().enumerate() {
            let branch_idx = self.active_branches[j];
            let w = witness.to_vec();
            let active_tv = &topology_vectors[branch_idx];
            let tv_product = active_tv.inner_product(&w);

            // Compute ρ^{id_j}
            let mut rho_power = 1u128;
            for _ in 0..branch_idx {
                rho_power = (rho_power * rho as u128) % self.modulus as u128;
            }

            // Add contribution: ρ^{id_j} · ⟨t_{id_j}, w^(j)⟩
            let contrib = (rho_power * tv_product as u128) % self.modulus as u128;
            hash_proof = (hash_proof + contrib) % self.modulus as u128;
        }

        // Generate vanishing polynomial coefficients for non-active branches
        // These prove that f(αᵢ) = 0 for other branches
        let vanishing_coeffs = Vec::new(); // Simplified

        self.phase = ProverPhase::Opened;

        Ok(OpenMessage {
            active_branches: self.active_branches.clone(),
            hash_proof: hash_proof as u64,
            vanishing_coeffs,
        })
    }

    /// Generates LPZK proof for multiplication verification.
    ///
    /// Proves that for each multiplication gate: c = a * b
    pub fn prove_multiplications(&mut self) -> Result<LpzkProofMessage, ProverError> {
        if self.phase != ProverPhase::Opened {
            return Err(ProverError::InvalidPhase);
        }

        // Pre-allocate with total multiplication count
        let total_mults: usize = self.witnesses.iter().map(|w| w.num_mults()).sum();
        let mut masked_products = Vec::with_capacity(total_mults);
        let mut mac_tags = Vec::with_capacity(total_mults);

        for witness in &self.witnesses {
            for i in 0..witness.num_mults() {
                let a = witness.mult_lefts[i];
                let b = witness.mult_rights[i];
                let c = witness.mult_outputs[i];

                // Verify locally: c = a * b
                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return Err(ProverError::InvalidMultiplication);
                }

                // In real LPZK:
                // 1. P sends masked product a*b - u for random u
                // 2. P uses VOLE correlation to prove consistency
                masked_products.push(c);
                mac_tags.push(0); // Placeholder
            }
        }

        self.phase = ProverPhase::Done;

        Ok(LpzkProofMessage {
            masked_products,
            mac_tags,
        })
    }

    /// Generates LPZK proof for multiplication verification using a VOLE source.
    ///
    /// This is the full implementation that uses real VOLE correlations from
    /// the injected VoleSource. The source handles OT→VOLE conversion internally.
    ///
    /// # Arguments
    /// * `vole_source` - Source of VOLE correlations (e.g., `VoleProvider` backed by OT)
    ///
    /// # Type Parameters
    /// * `F` - IT-MAC field type (must be compatible with u64 modulus)
    /// * `V` - VoleSource implementation
    ///
    /// # Returns
    /// LPZK proof message with masked products and MAC tags
    pub fn prove_multiplications_with_voles<F, V>(
        &mut self,
        vole_source: &mut V,
    ) -> Result<LpzkProofMessage, ProverError>
    where
        F: mpz_justvengers_core::ItMacField + From<u64> + Into<u64>,
        V: mpz_justvengers_core::VoleSource<F>,
    {
        if self.phase != ProverPhase::Opened {
            return Err(ProverError::InvalidPhase);
        }

        // Count total multiplications needed
        let total_mults: usize = self.witnesses.iter().map(|w| w.num_mults()).sum();

        // Request and generate VOLE correlations
        // Each multiplication needs one VOLE for the masking
        vole_source
            .request(total_mults)
            .map_err(|_| ProverError::VolePoolExhausted)?;
        vole_source
            .flush()
            .map_err(|_| ProverError::VolePoolExhausted)?;
        let voles = vole_source
            .take(total_mults)
            .map_err(|_| ProverError::VolePoolExhausted)?;

        let mut masked_products = Vec::with_capacity(total_mults);
        let mut mac_tags = Vec::with_capacity(total_mults);

        let mut vole_idx = 0;
        for witness in &self.witnesses {
            for i in 0..witness.num_mults() {
                let a = witness.mult_lefts[i];
                let b = witness.mult_rights[i];
                let c = witness.mult_outputs[i];

                // Verify locally: c = a * b
                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return Err(ProverError::InvalidMultiplication);
                }

                // LPZK protocol:
                // 1. Get random VOLE correlation [u] where:
                //    - Prover has (u, m_u) where m_u = k_u + u·Δ
                //    - Verifier has k_u
                let vole = &voles[vole_idx];
                vole_idx += 1;

                // 2. Prover computes masked product: c - u
                let u: u64 = vole.value().into();
                let masked = if c >= u {
                    c - u
                } else {
                    self.modulus - (u - c)
                };

                // 3. Prover reveals MAC tag m_u (for verification)
                let mac_tag: u64 = vole.prover_share().mac().into();

                masked_products.push(masked);
                mac_tags.push(mac_tag);
            }
        }

        self.phase = ProverPhase::Done;

        Ok(LpzkProofMessage {
            masked_products,
            mac_tags,
        })
    }

    // ==================== Soldering Support ====================

    /// Sets up soldering constraints for cross-repetition data flow.
    ///
    /// Must be called after `setup()` and before `commit()`. This validates
    /// that the witness satisfies the soldering constraints.
    ///
    /// # Arguments
    /// * `constraints` - Soldering constraints to enforce
    /// * `rng` - Random number generator for masking polynomials
    ///
    /// # Example
    /// ```ignore
    /// // Chain output 0 of each rep to input 0 of next rep
    /// let constraint = SolderingConstraint::new(0, 0);
    /// prover.setup_soldering(vec![constraint], &mut rng)?;
    /// ```
    pub fn setup_soldering<Rng: rand::Rng>(
        &mut self,
        constraints: Vec<SolderingConstraint>,
        rng: &mut Rng,
    ) -> Result<(), ProverError> {
        if self.phase != ProverPhase::Setup {
            return Err(ProverError::InvalidPhase);
        }

        if constraints.is_empty() {
            return Ok(());
        }

        // Need eval_points - generate roots of unity for Goldilocks, otherwise 1..=R
        let eval_points: Vec<u64> = self.eval_points.clone().unwrap_or_else(|| {
            #[cfg(feature = "ntt")]
            if self.modulus == GOLDILOCKS {
                let n = R.next_power_of_two();
                let log_n = n.trailing_zeros();
                let omega = Goldilocks::primitive_root_of_unity(log_n)
                    .expect("R too large for NTT");
                let mut points = Vec::with_capacity(n);
                let mut omega_pow = Goldilocks::one();
                for _ in 0..n {
                    points.push(omega_pow.inner());
                    omega_pow = omega_pow * omega;
                }
                return points;
            }

            (1..=R as u64).collect()
        });

        // Validate that witness satisfies all constraints
        // Collect inputs and outputs once (not per constraint)
        let inputs_per_rep: Vec<&[u64]> = self.witnesses
            .iter()
            .map(|w| w.inputs.as_slice())
            .collect();
        let outputs_per_rep: Vec<&[u64]> = self.witnesses
            .iter()
            .map(|w| w.mult_outputs.as_slice())
            .collect();

        for constraint in &constraints {
            if !crate::soldering::validate_witness_soldering_ref(
                &inputs_per_rep,
                &outputs_per_rep,
                constraint,
            ) {
                return Err(ProverError::SolderingConstraintViolation);
            }
        }

        // Build input and output polynomials for each constraint
        let mut input_polys = Vec::with_capacity(constraints.len());
        let mut output_polys = Vec::with_capacity(constraints.len());

        for constraint in &constraints {
            // Input values at target_input_idx across all reps
            let in_values: Vec<u64> = self.witnesses
                .iter()
                .map(|w| w.inputs.get(constraint.target_input_idx).copied().unwrap_or(0))
                .collect();

            // Output values at source_output_idx across all reps
            let out_values: Vec<u64> = self.witnesses
                .iter()
                .map(|w| w.mult_outputs.get(constraint.source_output_idx).copied().unwrap_or(0))
                .collect();

            // Use inverse NTT for Goldilocks, otherwise Lagrange interpolation
            #[cfg(feature = "ntt")]
            let (in_poly, out_poly) = if self.modulus == GOLDILOCKS {
                let n = eval_points.len();

                // Input polynomial via inverse NTT
                let mut in_padded: Vec<Goldilocks> = in_values
                    .iter()
                    .map(|&v| Goldilocks::new(v))
                    .collect();
                in_padded.resize(n, Goldilocks::zero());

                // Output polynomial via inverse NTT
                let mut out_padded: Vec<Goldilocks> = out_values
                    .iter()
                    .map(|&v| Goldilocks::new(v))
                    .collect();
                out_padded.resize(n, Goldilocks::zero());

                // Use precomputed INTT context if available
                if let Some(ref ctx) = self.intt_context {
                    ctx.intt_fused(&mut in_padded);
                    ctx.intt_fused(&mut out_padded);
                } else {
                    Goldilocks::intt(&mut in_padded);
                    Goldilocks::intt(&mut out_padded);
                }

                (
                    in_padded.iter().map(|g| g.inner()).collect(),
                    out_padded.iter().map(|g| g.inner()).collect(),
                )
            } else {
                (
                    interpolate(&eval_points, &in_values, self.modulus),
                    interpolate(&eval_points, &out_values, self.modulus),
                )
            };

            #[cfg(not(feature = "ntt"))]
            let (in_poly, out_poly) = (
                interpolate(&eval_points, &in_values, self.modulus),
                interpolate(&eval_points, &out_values, self.modulus),
            );

            input_polys.push(in_poly);
            output_polys.push(out_poly);
        }

        // Create and setup soldering prover
        let mut soldering = SolderingProver::new(self.modulus);
        soldering.setup(constraints, input_polys, output_polys, eval_points, R, rng);

        self.soldering_prover = Some(soldering);
        Ok(())
    }

    /// Generates soldering commitment message.
    ///
    /// Must be called after `commit()` to commit the masking polynomials.
    pub fn commit_soldering(&self) -> Result<Option<SolderingCommitMessage>, ProverError> {
        if self.phase != ProverPhase::Committed {
            return Err(ProverError::InvalidPhase);
        }

        Ok(self.soldering_prover.as_ref().map(|s| s.commit()))
    }

    /// Reveals masked soldering polynomials after receiving challenge.
    ///
    /// # Arguments
    /// * `challenge` - Challenge message from verifier containing φ
    pub fn reveal_soldering(
        &self,
        challenge: &SolderingChallengeMessage,
    ) -> Result<Option<SolderingRevealMessage>, ProverError> {
        if self.phase != ProverPhase::Disclosed && self.phase != ProverPhase::Committed {
            return Err(ProverError::InvalidPhase);
        }

        Ok(self.soldering_prover.as_ref().map(|s| s.reveal(challenge.phi)))
    }

    /// Returns the witnesses (for testing/debugging).
    pub fn witnesses(&self) -> &[ExtendedWitness] {
        &self.witnesses
    }
}

/// Errors that can occur during proving.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProverError {
    /// Operation called in wrong phase.
    InvalidPhase,
    /// Wrong number of repetitions provided.
    WrongRepetitionCount,
    /// Wrong number of evaluation points.
    WrongEvaluationPoints,
    /// Invalid branch index.
    InvalidBranch,
    /// Multiplication constraint not satisfied.
    InvalidMultiplication,
    /// VOLE pool exhausted.
    VolePoolExhausted,
    /// Soldering constraint not satisfied by witness.
    SolderingConstraintViolation,
}

/// Lagrange interpolation.
///
/// Given points (α₁, y₁), ..., (αᵣ, yᵣ), finds polynomial f(·) with f(αᵢ) = yᵢ.
///
/// When the `ntt` feature is enabled and the modulus is Goldilocks, uses the
/// optimized NTT-based interpolation which is O(n²) for arbitrary points.
/// Otherwise falls back to naive O(n³) Lagrange interpolation.
fn interpolate(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    assert_eq!(points.len(), values.len());

    if points.is_empty() {
        return vec![];
    }

    // Use Goldilocks NTT-optimized interpolation when available
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS {
        return Goldilocks::interpolate_u64(points, values);
    }

    // Fall back to naive Lagrange interpolation for other moduli
    interpolate_naive(points, values, modulus)
}

/// Naive Lagrange interpolation (O(n³) complexity).
///
/// Used as fallback when NTT is not available or modulus is not Goldilocks.
fn interpolate_naive(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    let n = points.len();
    let mut result = vec![0u64; n];

    for i in 0..n {
        // Compute Lagrange basis polynomial Lᵢ(X)
        let mut basis = vec![1u64];

        // ∏_{j≠i} (X - αⱼ)
        for j in 0..n {
            if i == j {
                continue;
            }

            let mut new_basis = vec![0u64; basis.len() + 1];
            for (k, &coeff) in basis.iter().enumerate() {
                // coeff * X
                new_basis[k + 1] = (new_basis[k + 1] + coeff) % modulus;
                // coeff * (-αⱼ)
                let neg_alpha = (modulus - points[j]) % modulus;
                new_basis[k] = ((new_basis[k] as u128 + coeff as u128 * neg_alpha as u128)
                    % modulus as u128) as u64;
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

        // Modular inverse
        let denom_inv = mod_inverse(denom as u64, modulus);

        // Scale by yᵢ / denom
        let scale = ((values[i] as u128 * denom_inv as u128) % modulus as u128) as u64;

        for (k, &coeff) in basis.iter().enumerate() {
            let term = ((coeff as u128 * scale as u128) % modulus as u128) as u64;
            result[k] = (result[k] + term) % modulus;
        }
    }

    result
}

/// Modular inverse using extended Euclidean algorithm.
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

/// Prover for a single repetition (simplified interface).
pub struct SingleRepProver {
    /// Extended witness.
    witness: Option<ExtendedWitness>,
    /// Active branch (used for disjunctive statements).
    #[allow(dead_code)]
    active_branch: usize,
    /// Field modulus.
    modulus: u64,
}

impl SingleRepProver {
    /// Creates a new single-repetition prover.
    pub fn new(active_branch: usize, modulus: u64) -> Self {
        Self {
            witness: None,
            active_branch,
            modulus,
        }
    }

    /// Sets up with circuit and inputs.
    pub fn setup(&mut self, circuit: &mut Circuit, inputs: &[u64]) {
        self.witness = Some(circuit.evaluate(inputs, self.modulus));
    }

    /// Returns the extended witness.
    pub fn witness(&self) -> Option<&ExtendedWitness> {
        self.witness.as_ref()
    }

    /// Verifies multiplication constraints locally.
    pub fn verify_multiplications(&self) -> bool {
        if let Some(w) = &self.witness {
            for i in 0..w.num_mults() {
                let a = w.mult_lefts[i];
                let b = w.mult_rights[i];
                let c = w.mult_outputs[i];
                let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
                if c != expected {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 65537;

    #[test]
    fn test_interpolation_basic() {
        // f(1) = 2, f(2) = 4 => f(X) = 2X
        let points = vec![1, 2];
        let values = vec![2, 4];
        let poly = interpolate(&points, &values, TEST_MODULUS);

        // Verify f(1) = 2 and f(2) = 4
        let f1 = evaluate_poly(&poly, 1, TEST_MODULUS);
        let f2 = evaluate_poly(&poly, 2, TEST_MODULUS);
        assert_eq!(f1, 2);
        assert_eq!(f2, 4);
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
    fn test_prover_state_new() {
        let prover: ProverState<4> = ProverState::new(0, TEST_MODULUS);
        assert_eq!(prover.active_branch(), 0);
        assert_eq!(prover.phase(), &ProverPhase::Init);
    }

    #[test]
    fn test_prover_setup() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let mut prover: ProverState<2> = ProverState::new(0, TEST_MODULUS);

        let inputs = vec![
            vec![3, 4], // Rep 1: x=3, y=4
            vec![5, 6], // Rep 2: x=5, y=6
        ];

        prover.setup(&mut circuit, &inputs).unwrap();
        assert_eq!(prover.phase(), &ProverPhase::Setup);
    }

    #[test]
    fn test_prover_wrong_rep_count() {
        let mut circuit = Circuit::new();
        circuit.add_input();

        let mut prover: ProverState<3> = ProverState::new(0, TEST_MODULUS);

        // Provide 2 repetitions but prover expects 3
        let inputs = vec![vec![1], vec![2]];
        let result = prover.setup(&mut circuit, &inputs);

        assert_eq!(result, Err(ProverError::WrongRepetitionCount));
    }

    #[test]
    fn test_single_rep_prover() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let mut prover = SingleRepProver::new(0, TEST_MODULUS);
        prover.setup(&mut circuit, &[7, 8]);

        assert!(prover.verify_multiplications());

        let w = prover.witness().unwrap();
        assert_eq!(w.mult_outputs[0], 56); // 7 * 8 = 56
    }

    #[test]
    fn test_prover_commit() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let mut prover: ProverState<2> = ProverState::new(0, TEST_MODULUS);

        let inputs = vec![vec![3, 4], vec![5, 6]];
        prover.setup(&mut circuit, &inputs).unwrap();

        let eval_points = vec![1, 2];
        let commit_msg = prover.commit(&eval_points).unwrap();

        assert!(commit_msg.num_commitments > 0);
        assert_eq!(prover.phase(), &ProverPhase::Committed);
    }

    #[test]
    fn test_mod_inverse() {
        let a = 3u64;
        let m = 11u64;
        let inv = mod_inverse(a, m);
        assert_eq!((a * inv) % m, 1);
    }

    #[test]
    fn test_prover_full_flow() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let mut prover: ProverState<2> = ProverState::new(0, TEST_MODULUS);

        // Setup
        let inputs = vec![vec![3, 4], vec![5, 6]];
        prover.setup(&mut circuit, &inputs).unwrap();

        // Commit
        let eval_points = vec![1, 2];
        let _commit_msg = prover.commit(&eval_points).unwrap();

        // Disclose (need topology vectors)
        let tv = crate::topology::TopologyVector::new(vec![1, 2, 3, 4, 5], 7, TEST_MODULUS);
        let topology_vectors = vec![tv];

        let chi = 42;
        let _disclose_msg = prover.disclose(chi, &topology_vectors).unwrap();

        // Open
        let rho = 13;
        let _open_msg = prover.open(rho, &topology_vectors).unwrap();

        // LPZK proof
        let _proof_msg = prover.prove_multiplications().unwrap();

        assert_eq!(prover.phase(), &ProverPhase::Done);
    }

    /// Test field compatible with IT-MAC and u64 conversion.
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    struct VoleTestField(u64);

    impl std::ops::Add for VoleTestField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self((self.0 + rhs.0) % TEST_MODULUS)
        }
    }

    impl std::ops::Sub for VoleTestField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self((self.0 + TEST_MODULUS - rhs.0) % TEST_MODULUS)
        }
    }

    impl std::ops::Mul for VoleTestField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self((self.0 as u128 * rhs.0 as u128 % TEST_MODULUS as u128) as u64)
        }
    }

    impl mpz_justvengers_core::ItMacField for VoleTestField {
        fn zero() -> Self {
            Self(0)
        }
        fn one() -> Self {
            Self(1)
        }
        fn random<Rng: rand::Rng>(rng: &mut Rng) -> Self {
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

    impl From<u64> for VoleTestField {
        fn from(v: u64) -> Self {
            Self(v % TEST_MODULUS)
        }
    }

    impl From<VoleTestField> for u64 {
        fn from(v: VoleTestField) -> u64 {
            v.0
        }
    }

    #[test]
    fn test_prover_with_vole_source() {
        use crate::vole::VoleProvider;
        use mpz_ot_core::ideal::rcot::IdealRCOT;

        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let mut prover: ProverState<2> = ProverState::new(0, TEST_MODULUS);

        // Setup
        let inputs = vec![vec![3, 4], vec![5, 6]];
        prover.setup(&mut circuit, &inputs).unwrap();

        // Commit
        let eval_points = vec![1, 2];
        let _commit_msg = prover.commit(&eval_points).unwrap();

        // Disclose
        let tv = crate::topology::TopologyVector::new(vec![1, 2, 3, 4, 5], 7, TEST_MODULUS);
        let topology_vectors = vec![tv];

        let chi = 42;
        let _disclose_msg = prover.disclose(chi, &topology_vectors).unwrap();

        // Open
        let rho = 13;
        let _open_msg = prover.open(rho, &topology_vectors).unwrap();

        // LPZK proof with VOLE source backed by IdealRCOT
        let rcot = IdealRCOT::default();
        let mut vole_source = VoleProvider::<VoleTestField, _>::new(rcot);
        let proof_msg = prover.prove_multiplications_with_voles(&mut vole_source).unwrap();

        // Should have 2 multiplications (one per repetition)
        assert_eq!(proof_msg.masked_products.len(), 2);
        assert_eq!(proof_msg.mac_tags.len(), 2);

        // MAC tags should be non-zero (real VOLE values)
        // Note: there's a small probability they could be zero, but extremely unlikely
        assert!(proof_msg.mac_tags.iter().any(|&t| t != 0), "MAC tags should have non-zero values");

        assert_eq!(prover.phase(), &ProverPhase::Done);
    }
}
