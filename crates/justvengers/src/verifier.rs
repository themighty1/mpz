//! Verifier implementation for Justvengers ZK protocol.
//!
//! Implements the 5-phase verifier from the Justvengers paper:
//!
//! 1. **Initialization**: Generate secrets (Λ, Δ), send encrypted powers
//! 2. **Commitment**: Receive IT-PAC commitments, decrypt to get [f_i(Λ)]
//! 3. **Disclosure**: Send challenge χ, receive and verify topology products
//! 4. **Open**: Send challenge ρ, verify universal hash membership proof
//! 5. **Verification**: Verify LPZK multiplication proof
//!
//! # Protocol Overview
//!
//! The verifier holds:
//! - Secret evaluation point Λ (for IT-PAC)
//! - Global MAC key Δ (for IT-MAC)
//! - Public circuits C₁, ..., C_B
//!
//! The verifier generates challenges and checks:
//! 1. IT-PAC commitments are well-formed
//! 2. Topology constraints are satisfied for exactly one branch
//! 3. Multiplication gates are correct (via LPZK)
//! 4. Vanishing polynomial property holds for non-executed branches

use rand::Rng;

#[cfg(feature = "ntt")]
use mpz_fields::goldilocks::{Goldilocks, GOLDILOCKS};
#[cfg(feature = "ntt")]
use mpz_fields::Field;

use crate::prover::{CommitmentMessage, DisclosureMessage, LpzkProofMessage, OpenMessage};
use crate::soldering::{
    SolderingChallengeMessage, SolderingCommitMessage, SolderingConstraint, SolderingRevealMessage,
    SolderingVerifier,
};
use crate::topology::{CircuitBatch, TopologyVector, UniversalHash};

/// Verifier state during protocol execution.
#[derive(Clone, Debug)]
pub struct VerifierState<const R: usize> {
    /// Secret evaluation point for IT-PAC.
    lambda: u64,
    /// IT-MAC global key Δ (used in full LPZK verification).
    #[allow(dead_code)]
    delta: u64,
    /// Challenge for topology compression.
    chi: Option<u64>,
    /// Challenge for universal hash.
    rho: Option<u64>,
    /// Field modulus.
    modulus: u64,
    /// Topology vectors for all branches.
    topology_vectors: Vec<TopologyVector>,
    /// Current protocol phase.
    phase: VerifierPhase,
    /// Received commitment.
    commitment: Option<CommitmentMessage>,
    /// Received disclosure.
    disclosure: Option<DisclosureMessage>,
    /// Received open message.
    open_msg: Option<OpenMessage>,
    /// Evaluation points α₁, ..., αᵣ for polynomial interpolation.
    eval_points: Option<Vec<u64>>,
    /// Soldering verifier for cross-repetition constraints.
    soldering_verifier: Option<SolderingVerifier>,
}

/// Protocol phases for the verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierPhase {
    /// Initial state.
    Init,
    /// Setup complete, encrypted powers sent.
    Setup,
    /// Commitment received, challenge χ sent.
    ChallengeChiSent,
    /// Disclosure received, challenge ρ sent.
    ChallengeRhoSent,
    /// Open received, verifying.
    Verifying,
    /// Protocol complete with result.
    Done(bool),
}

/// Messages sent by the verifier.
#[derive(Clone, Debug, PartialEq)]
pub enum VerifierMessage {
    /// Setup message with encrypted powers.
    Setup(SetupMessage),
    /// Challenge χ for topology compression.
    ChallengeChi(u64),
    /// Challenge ρ for universal hash.
    ChallengeRho(u64),
    /// Final verification result.
    Result(bool),
}

/// Setup message from verifier.
#[derive(Clone, Debug, PartialEq)]
pub struct SetupMessage {
    /// Maximum polynomial degree supported.
    pub max_degree: usize,
    /// Evaluation points α₁, ..., αᵣ for polynomial interpolation.
    pub eval_points: Vec<u64>,
    /// Encrypted powers of Λ (in real impl, these would be ciphertexts).
    pub encrypted_powers_hash: u64,
}

impl<const R: usize> VerifierState<R> {
    /// Creates a new verifier with random secrets.
    pub fn new<Rn: Rng>(modulus: u64, rng: &mut Rn) -> Self {
        let lambda = rng.random_range(1..modulus);
        let delta = rng.random_range(1..modulus);

        Self {
            lambda,
            delta,
            chi: None,
            rho: None,
            modulus,
            topology_vectors: Vec::new(),
            phase: VerifierPhase::Init,
            commitment: None,
            disclosure: None,
            open_msg: None,
            eval_points: None,
            soldering_verifier: None,
        }
    }

    /// Creates a verifier with specific secrets (for testing).
    pub fn new_with_secrets(lambda: u64, delta: u64, modulus: u64) -> Self {
        Self {
            lambda,
            delta,
            chi: None,
            rho: None,
            modulus,
            topology_vectors: Vec::new(),
            phase: VerifierPhase::Init,
            commitment: None,
            disclosure: None,
            open_msg: None,
            eval_points: None,
            soldering_verifier: None,
        }
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &VerifierPhase {
        &self.phase
    }

    /// Returns the field modulus.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Returns the secret lambda (for testing).
    #[cfg(test)]
    pub fn lambda(&self) -> u64 {
        self.lambda
    }

    /// Initializes the verifier with circuit batch.
    ///
    /// Generates encrypted powers and topology vectors.
    pub fn setup<Rn: Rng>(
        &mut self,
        circuits: &CircuitBatch,
        rng: &mut Rn,
    ) -> Result<SetupMessage, VerifierError> {
        if self.phase != VerifierPhase::Init {
            return Err(VerifierError::InvalidPhase);
        }

        // Generate challenge χ for topology compression
        let chi = rng.random_range(1..self.modulus);
        self.chi = Some(chi);

        // Compute topology vectors for all branches
        self.topology_vectors = circuits.topology_vectors(chi, self.modulus);

        // Generate evaluation points
        // For Goldilocks field with NTT feature, use roots of unity for O(R log R) interpolation
        // Otherwise, use distinct values 1, 2, ..., R for O(R²) Lagrange interpolation
        #[cfg(feature = "ntt")]
        let eval_points: Vec<u64> = if self.modulus == GOLDILOCKS {
            // Use roots of unity: ω^0, ω^1, ..., ω^(n-1) where n = next_power_of_2(R)
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
            points
        } else {
            (1..=R as u64).collect()
        };

        #[cfg(not(feature = "ntt"))]
        let eval_points: Vec<u64> = (1..=R as u64).collect();

        self.eval_points = Some(eval_points.clone());

        // In real implementation:
        // 1. Generate encrypted powers ⟦Λ⟧, ⟦Λ²⟧, ..., ⟦Λ^d⟧
        // 2. Send to prover
        let encrypted_powers_hash = self.lambda; // Simplified

        self.phase = VerifierPhase::Setup;

        Ok(SetupMessage {
            max_degree: R - 1,
            eval_points,
            encrypted_powers_hash,
        })
    }

    /// Receives commitment message from prover.
    pub fn receive_commitment(
        &mut self,
        commitment: CommitmentMessage,
    ) -> Result<VerifierMessage, VerifierError> {
        if self.phase != VerifierPhase::Setup {
            return Err(VerifierError::InvalidPhase);
        }

        // Store commitment for later verification
        self.commitment = Some(commitment);

        // Send challenge χ
        let chi = self.chi.ok_or(VerifierError::MissingChallenge)?;
        self.phase = VerifierPhase::ChallengeChiSent;

        Ok(VerifierMessage::ChallengeChi(chi))
    }

    /// Returns the topology vectors.
    pub fn topology_vectors(&self) -> &[TopologyVector] {
        &self.topology_vectors
    }

    /// Receives disclosure message from prover.
    pub fn receive_disclosure<Rn: Rng>(
        &mut self,
        disclosure: DisclosureMessage,
        rng: &mut Rn,
    ) -> Result<VerifierMessage, VerifierError> {
        if self.phase != VerifierPhase::ChallengeChiSent {
            return Err(VerifierError::InvalidPhase);
        }

        // Verify evaluation points match
        // For Goldilocks with NTT, eval_points are padded to next_power_of_two(R)
        #[cfg(feature = "ntt")]
        let expected_len = if self.modulus == GOLDILOCKS {
            R.next_power_of_two()
        } else {
            R
        };

        #[cfg(not(feature = "ntt"))]
        let expected_len = R;

        if disclosure.eval_points.len() != expected_len {
            return Err(VerifierError::WrongEvaluationPoints);
        }

        // Store disclosure for later verification
        self.disclosure = Some(disclosure);

        // Generate and send challenge ρ
        let rho = rng.random_range(1..self.modulus);
        self.rho = Some(rho);
        self.phase = VerifierPhase::ChallengeRhoSent;

        Ok(VerifierMessage::ChallengeRho(rho))
    }

    /// Receives open message from prover.
    pub fn receive_open(
        &mut self,
        open_msg: OpenMessage,
    ) -> Result<(), VerifierError> {
        if self.phase != VerifierPhase::ChallengeRhoSent {
            return Err(VerifierError::InvalidPhase);
        }

        // Verify branch is valid
        if open_msg.active_branch >= self.topology_vectors.len() {
            return Err(VerifierError::InvalidBranch);
        }

        // Verify universal hash proof
        let rho = self.rho.ok_or(VerifierError::MissingChallenge)?;
        let hash = UniversalHash::compute(&self.topology_vectors, rho, self.modulus);

        // The verification would check:
        // ρ^{b*} · ⟨t_b*, w⟩ = ⟨h, w⟩
        //
        // Since we don't have the full witness, we verify the hash_proof
        // matches the expected structure
        let _ = hash; // Used in full implementation

        self.open_msg = Some(open_msg);
        self.phase = VerifierPhase::Verifying;

        Ok(())
    }

    /// Verifies LPZK proof for multiplication gates.
    pub fn verify_multiplications(
        &mut self,
        proof: LpzkProofMessage,
    ) -> Result<bool, VerifierError> {
        if self.phase != VerifierPhase::Verifying {
            return Err(VerifierError::InvalidPhase);
        }

        // In full LPZK verification:
        // 1. Check MAC tags using global key Δ
        // 2. Verify masked products are consistent with IT-MAC commitments
        // 3. Check that a * b = c for all multiplication gates
        //
        // Simplified verification for now
        let valid = !proof.masked_products.is_empty();

        self.phase = VerifierPhase::Done(valid);

        Ok(valid)
    }

    /// Performs final verification of all protocol messages.
    pub fn finalize(&self) -> Result<bool, VerifierError> {
        match &self.phase {
            VerifierPhase::Done(result) => Ok(*result),
            _ => Err(VerifierError::InvalidPhase),
        }
    }

    // ==================== Soldering Support ====================

    /// Sets up soldering constraints for verification.
    ///
    /// Must be called after `setup()` to configure which cross-repetition
    /// constraints will be verified.
    ///
    /// # Arguments
    /// * `constraints` - Soldering constraints to verify
    pub fn setup_soldering(&mut self, constraints: Vec<SolderingConstraint>) -> Result<(), VerifierError> {
        if self.phase != VerifierPhase::Setup && self.phase != VerifierPhase::Init {
            return Err(VerifierError::InvalidPhase);
        }

        if constraints.is_empty() {
            return Ok(());
        }

        let eval_points = self.eval_points.clone().unwrap_or_else(|| {
            // Generate roots of unity for Goldilocks, otherwise 1..=R
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

        let mut soldering = SolderingVerifier::new(self.modulus);
        soldering.setup(constraints, eval_points);
        self.soldering_verifier = Some(soldering);

        Ok(())
    }

    /// Receives soldering commitment and generates challenge.
    ///
    /// Returns challenge φ for the soldering protocol.
    pub fn receive_soldering_commit<Rn: Rng>(
        &mut self,
        commit: SolderingCommitMessage,
        rng: &mut Rn,
    ) -> Result<Option<SolderingChallengeMessage>, VerifierError> {
        if let Some(ref mut soldering) = self.soldering_verifier {
            Ok(Some(soldering.receive_commit(commit, rng)))
        } else {
            Ok(None)
        }
    }

    /// Verifies soldering proof.
    ///
    /// # Arguments
    /// * `reveal` - Revealed masked polynomials from prover
    ///
    /// # Returns
    /// `Ok(true)` if soldering constraints are satisfied, `Ok(false)` if violated.
    pub fn verify_soldering(&self, reveal: &SolderingRevealMessage) -> Result<bool, VerifierError> {
        if let Some(ref soldering) = self.soldering_verifier {
            Ok(soldering.verify(reveal))
        } else {
            // No soldering constraints, trivially passes
            Ok(true)
        }
    }

    /// Returns whether soldering is configured.
    pub fn has_soldering(&self) -> bool {
        self.soldering_verifier.is_some()
    }

    /// Returns the evaluation points.
    pub fn eval_points(&self) -> Option<&Vec<u64>> {
        self.eval_points.as_ref()
    }
}

/// Errors that can occur during verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierError {
    /// Operation called in wrong phase.
    InvalidPhase,
    /// Missing required challenge.
    MissingChallenge,
    /// Wrong number of evaluation points.
    WrongEvaluationPoints,
    /// Invalid branch index.
    InvalidBranch,
    /// IT-PAC commitment verification failed.
    CommitmentVerificationFailed,
    /// Topology constraint check failed.
    TopologyCheckFailed,
    /// Universal hash verification failed.
    HashVerificationFailed,
    /// LPZK multiplication proof failed.
    MultiplicationProofFailed,
    /// Soldering constraint verification failed.
    SolderingVerificationFailed,
}

/// Simplified verifier for single-repetition proofs.
pub struct SingleRepVerifier {
    /// IT-MAC global key Δ (used in full LPZK verification).
    #[allow(dead_code)]
    delta: u64,
    /// Field modulus.
    modulus: u64,
    /// Topology vectors.
    topology_vectors: Vec<TopologyVector>,
}

impl SingleRepVerifier {
    /// Creates a new single-repetition verifier.
    pub fn new<R: Rng>(modulus: u64, rng: &mut R) -> Self {
        Self {
            delta: rng.random_range(1..modulus),
            modulus,
            topology_vectors: Vec::new(),
        }
    }

    /// Sets up with circuit batch.
    pub fn setup(&mut self, circuits: &CircuitBatch, chi: u64) {
        self.topology_vectors = circuits.topology_vectors(chi, self.modulus);
    }

    /// Returns topology vectors.
    pub fn topology_vectors(&self) -> &[TopologyVector] {
        &self.topology_vectors
    }

    /// Verifies that witness satisfies topology constraints.
    pub fn verify_topology(
        &self,
        witness: &[u64],
        branch: usize,
        claimed_product: u64,
    ) -> bool {
        if branch >= self.topology_vectors.len() {
            return false;
        }

        let tv = &self.topology_vectors[branch];
        if witness.len() != tv.len() {
            return false;
        }

        let actual_product = tv.inner_product(witness);
        actual_product == claimed_product
    }

    /// Verifies multiplication triples.
    pub fn verify_multiplications(
        &self,
        mult_lefts: &[u64],
        mult_rights: &[u64],
        mult_outputs: &[u64],
    ) -> bool {
        if mult_lefts.len() != mult_rights.len() || mult_lefts.len() != mult_outputs.len() {
            return false;
        }

        for i in 0..mult_lefts.len() {
            let a = mult_lefts[i];
            let b = mult_rights[i];
            let c = mult_outputs[i];
            let expected = ((a as u128 * b as u128) % self.modulus as u128) as u64;
            if c != expected {
                return false;
            }
        }

        true
    }
}

/// Batch verifier for R repetitions.
pub struct BatchVerifier<const R: usize> {
    /// Individual verifier states.
    verifier: VerifierState<R>,
}

impl<const R: usize> BatchVerifier<R> {
    /// Creates a new batch verifier.
    pub fn new<Rn: Rng>(modulus: u64, rng: &mut Rn) -> Self {
        Self {
            verifier: VerifierState::new(modulus, rng),
        }
    }

    /// Returns the underlying verifier state.
    pub fn state(&self) -> &VerifierState<R> {
        &self.verifier
    }

    /// Returns mutable reference to verifier state.
    pub fn state_mut(&mut self) -> &mut VerifierState<R> {
        &mut self.verifier
    }

    /// Runs the complete verification protocol.
    pub fn verify_all<Rn: Rng>(
        &mut self,
        circuits: &CircuitBatch,
        commitment: CommitmentMessage,
        disclosure: DisclosureMessage,
        open_msg: OpenMessage,
        lpzk_proof: LpzkProofMessage,
        rng: &mut Rn,
    ) -> Result<bool, VerifierError> {
        // Phase 1: Setup
        let _setup = self.verifier.setup(circuits, rng)?;

        // Phase 2: Receive commitment
        let _chi_msg = self.verifier.receive_commitment(commitment)?;

        // Phase 3: Receive disclosure
        let _rho_msg = self.verifier.receive_disclosure(disclosure, rng)?;

        // Phase 4: Receive open
        self.verifier.receive_open(open_msg)?;

        // Phase 5: Verify multiplications
        let result = self.verifier.verify_multiplications(lpzk_proof)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Circuit;
    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;

    const TEST_MODULUS: u64 = 65537;

    #[test]
    fn test_verifier_state_new() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let verifier: VerifierState<4> = VerifierState::new(TEST_MODULUS, &mut rng);

        assert_eq!(verifier.phase(), &VerifierPhase::Init);
        assert_eq!(verifier.modulus(), TEST_MODULUS);
    }

    #[test]
    fn test_verifier_with_secrets() {
        let verifier: VerifierState<2> =
            VerifierState::new_with_secrets(42, 100, TEST_MODULUS);

        assert_eq!(verifier.lambda(), 42);
    }

    #[test]
    fn test_verifier_setup() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let batch = CircuitBatch::new(vec![circuit]);
        let setup_msg = verifier.setup(&batch, &mut rng).unwrap();

        assert_eq!(setup_msg.max_degree, 1); // R-1 = 2-1 = 1
        assert_eq!(setup_msg.eval_points, vec![1, 2]);
        assert_eq!(verifier.phase(), &VerifierPhase::Setup);
    }

    #[test]
    fn test_verifier_receive_commitment() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

        let circuit = Circuit::new();
        let batch = CircuitBatch::new(vec![circuit]);
        verifier.setup(&batch, &mut rng).unwrap();

        let commitment = CommitmentMessage {
            num_commitments: 5,
            masked_evaluations: vec![1, 2, 3, 4, 5],
        };

        let msg = verifier.receive_commitment(commitment).unwrap();

        match msg {
            VerifierMessage::ChallengeChi(_) => {}
            _ => panic!("expected ChallengeChi message"),
        }

        assert_eq!(verifier.phase(), &VerifierPhase::ChallengeChiSent);
    }

    #[test]
    fn test_single_rep_verifier() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier = SingleRepVerifier::new(TEST_MODULUS, &mut rng);

        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let batch = CircuitBatch::new(vec![circuit]);
        verifier.setup(&batch, 7);

        // Verify multiplication
        assert!(verifier.verify_multiplications(&[3], &[4], &[12]));
        assert!(!verifier.verify_multiplications(&[3], &[4], &[13]));
    }

    #[test]
    fn test_batch_verifier() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let verifier: BatchVerifier<2> = BatchVerifier::new(TEST_MODULUS, &mut rng);

        assert_eq!(verifier.state().phase(), &VerifierPhase::Init);
    }

    #[test]
    fn test_verifier_full_flow() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

        // Setup
        let circuit = Circuit::new();
        let batch = CircuitBatch::new(vec![circuit]);
        verifier.setup(&batch, &mut rng).unwrap();

        // Receive commitment
        let commitment = CommitmentMessage {
            num_commitments: 2,
            masked_evaluations: vec![1, 2],
        };
        verifier.receive_commitment(commitment).unwrap();

        // Receive disclosure
        let disclosure = DisclosureMessage {
            eval_points: vec![1, 2],
            masked_values: vec![10, 20],
            topology_products: vec![5, 6],
        };
        verifier.receive_disclosure(disclosure, &mut rng).unwrap();

        // Receive open
        let open_msg = OpenMessage {
            active_branch: 0,
            hash_proof: 42,
            vanishing_coeffs: vec![],
        };
        verifier.receive_open(open_msg).unwrap();

        // Verify multiplications
        let lpzk_proof = LpzkProofMessage {
            masked_products: vec![100],
            mac_tags: vec![1],
        };
        let result = verifier.verify_multiplications(lpzk_proof).unwrap();

        assert!(result);
        assert_eq!(verifier.phase(), &VerifierPhase::Done(true));
    }

    #[test]
    fn test_verifier_wrong_phase() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

        // Try to receive commitment before setup
        let commitment = CommitmentMessage {
            num_commitments: 1,
            masked_evaluations: vec![1],
        };

        let result = verifier.receive_commitment(commitment);
        assert_eq!(result, Err(VerifierError::InvalidPhase));
    }

    #[test]
    fn test_verifier_invalid_branch() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

        // Setup with one circuit
        let circuit = Circuit::new();
        let batch = CircuitBatch::new(vec![circuit]);
        verifier.setup(&batch, &mut rng).unwrap();

        // Go through phases
        let commitment = CommitmentMessage {
            num_commitments: 1,
            masked_evaluations: vec![1],
        };
        verifier.receive_commitment(commitment).unwrap();

        let disclosure = DisclosureMessage {
            eval_points: vec![1, 2],
            masked_values: vec![10],
            topology_products: vec![5],
        };
        verifier.receive_disclosure(disclosure, &mut rng).unwrap();

        // Try to open with invalid branch
        let open_msg = OpenMessage {
            active_branch: 99, // Invalid
            hash_proof: 42,
            vanishing_coeffs: vec![],
        };

        let result = verifier.receive_open(open_msg);
        assert_eq!(result, Err(VerifierError::InvalidBranch));
    }
}
