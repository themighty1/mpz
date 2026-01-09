//! Justvengers: VOLE-based ZK for batched disjunctive statements.
//!
//! This crate implements the Justvengers protocol for proving knowledge of
//! a valid witness for ONE circuit among B possible branches, achieving
//! O(R + B + C) communication complexity where:
//!
//! - R is the number of repetitions (for soundness amplification)
//! - B is the number of branches (disjunctive statements)
//! - C is the circuit size per branch
//!
//! # Protocol Overview
//!
//! Justvengers uses three main building blocks:
//!
//! 1. **IT-MAC** (Information-Theoretic MACs): Commitments using VOLE correlations
//!    - Verifier holds global key Δ
//!    - For value x: V has local key k, P has mac m = k + x·Δ
//!
//! 2. **IT-PAC** (Polynomial Authentication Codes): Polynomial commitments
//!    - Commit to f(·) by obtaining IT-MAC [f(Λ)] at secret point Λ
//!    - Uses AHE (BGV) for homomorphic polynomial evaluation
//!
//! 3. **LPZK** (Line-Point Zero Knowledge): Multiplication verification
//!    - Proves c = a·b using VOLE correlations
//!
//! # Example
//!
//! ```ignore
//! use mpz_justvengers::{Circuit, ProverState, VerifierState, CircuitBatch};
//!
//! // Create a simple multiplication circuit
//! let mut circuit = Circuit::new();
//! let x = circuit.add_input();
//! let y = circuit.add_input();
//! let z = circuit.add_mul(x, y);
//!
//! // Setup prover (branch 0, 2 repetitions)
//! let mut prover: ProverState<2> = ProverState::new(0, 65537);
//! prover.setup(&mut circuit, &[vec![3, 4], vec![5, 6]]).unwrap();
//!
//! // Run protocol phases...
//! ```
//!
//! # Modules
//!
//! - [`topology`]: Circuit representation and topology vectors
//! - [`prover`]: 5-phase prover implementation
//! - [`verifier`]: 5-phase verifier implementation
//!
//! # References
//!
//! Based on the Justvengers paper:
//! "Efficient Zero-Knowledge Proofs for Set Membership in Blockchain-Based Systems"

#![deny(missing_docs, unreachable_pub, unused_must_use)]
#![deny(clippy::all)]

pub mod prover;
pub mod protocol;
pub mod soldering;
pub mod topology;
pub mod verifier;
pub mod vole;
pub mod jv;

#[cfg(test)]
mod jv_tests;

/// KZG polynomial commitment module (requires `kzg` feature).
#[cfg(feature = "kzg")]
pub mod kzg;

// Re-exports for convenience
pub use prover::{
    CommitmentMessage, DisclosureMessage, LpzkProofMessage, OpenMessage, ProverError,
    ProverMessage, ProverPhase, ProverState, SingleRepProver,
};
pub use topology::{
    Circuit, CircuitBatch, ExtendedWitness, Gate, GateType, TopologyMatrix, TopologyVector,
    UniversalHash, WireId,
};
pub use soldering::{
    AggregatedSolderingReveal, SolderingChallengeMessage, SolderingCommitMessage,
    SolderingConstraint, SolderingProver, SolderingRevealMessage, SolderingVerifier,
};
pub use verifier::{
    BatchVerifier, SetupMessage, SingleRepVerifier, VerifierError, VerifierMessage, VerifierPhase,
    VerifierState,
};
pub use vole::{VoleProvider, VoleProviderError, VoleStats};
pub use mpz_justvengers_core::{VolePool, GlobalKey};
pub use protocol::{run_prover, run_prover_with_vole, run_verifier, run_protocol as run_protocol_async, ProtocolError as AsyncProtocolError};
pub use jv::{
    JVProver, JVVerifier, JVProverPhase, JVVerifierPhase,
    JVSetupMessage, JVCommitmentMessage, JVDisclosureMessage, JVOpenMessage,
    JVLpzkProofMessage, AggregatedLpzkProofMessage, ItPacOpenMessage,
    JVProverError, JVVerifierError, JVProtocolError,
    run_jv_protocol, estimate_communication, CommunicationEstimate,
    extract_verifier_shares_from_pool,
    // IT-MAC field types
    GoldilocksItMac, ItMacFieldType, ITMAC_MODULUS,
    // MK polynomial types for zero-knowledge branch hiding
    MKCommitmentMessage, MKCiphertextOpenMessage, MKBinaryProofMessage,
    MKSumProofMessage, MKHashProofMessage,
};
#[cfg(feature = "mersenne")]
pub use jv::MersenneItMac;

/// Protocol parameters for Justvengers.
#[derive(Clone, Debug)]
pub struct ProtocolParams {
    /// Number of repetitions R.
    pub repetitions: usize,
    /// Number of branches B.
    pub branches: usize,
    /// Field modulus p.
    pub modulus: u64,
    /// Soundness parameter (bits).
    pub soundness_bits: usize,
}

impl ProtocolParams {
    /// Creates new protocol parameters.
    pub fn new(repetitions: usize, branches: usize, modulus: u64) -> Self {
        // Soundness is approximately R * log(|F|) bits
        let soundness_bits = repetitions * (64 - modulus.leading_zeros() as usize);
        Self {
            repetitions,
            branches,
            modulus,
            soundness_bits,
        }
    }

    /// Creates parameters with target soundness.
    ///
    /// Computes the required number of repetitions for target soundness.
    pub fn with_soundness(target_bits: usize, branches: usize, modulus: u64) -> Self {
        let bits_per_rep = 64 - modulus.leading_zeros() as usize;
        let repetitions = (target_bits + bits_per_rep - 1) / bits_per_rep;
        Self {
            repetitions,
            branches,
            modulus,
            soundness_bits: repetitions * bits_per_rep,
        }
    }

    /// Returns estimated communication complexity in field elements.
    ///
    /// O(R + B + C) where C is the circuit size.
    pub fn communication_complexity(&self, circuit_size: usize) -> usize {
        // Commitment: O(C) IT-PAC commitments
        // Disclosure: O(1) topology products
        // Open: O(B) vanishing polynomial coefficients
        // LPZK: O(M) multiplication proofs where M is number of mult gates
        self.repetitions + self.branches + circuit_size
    }
}

impl Default for ProtocolParams {
    fn default() -> Self {
        Self::new(40, 2, 65537)
    }
}

/// Runs the complete Justvengers protocol.
///
/// This is a simplified synchronous execution for testing.
/// In practice, the protocol would be run over a network channel.
pub fn run_protocol<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    modulus: u64,
) -> Result<bool, ProtocolError> {
    use rand::SeedableRng;
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    // Verify inputs
    if inputs_per_rep.len() != R {
        return Err(ProtocolError::InvalidInputs);
    }
    if active_branch >= circuits.num_branches() {
        return Err(ProtocolError::InvalidBranch);
    }

    // Get the circuit for the active branch
    let circuit = circuits.get(active_branch)
        .ok_or(ProtocolError::InvalidBranch)?;

    // Initialize prover
    let mut prover: ProverState<R> = ProverState::new(active_branch, modulus);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep)
        .map_err(|_| ProtocolError::ProverError)?;

    // Initialize verifier
    let mut verifier: VerifierState<R> = VerifierState::new(modulus, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 1: Prover commits
    let commitment = prover.commit(&setup_msg.eval_points)
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 2: Verifier sends challenge χ
    let chi_msg = verifier.receive_commitment(commitment)
        .map_err(|_| ProtocolError::VerifierError)?;
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 3: Prover discloses
    let disclosure = prover.disclose(chi, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 4: Verifier sends challenge ρ
    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 5: Prover opens
    let open_msg = prover.open(rho, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;
    verifier.receive_open(open_msg)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 6: Prover sends LPZK proof
    let lpzk_proof = prover.prove_multiplications()
        .map_err(|_| ProtocolError::ProverError)?;

    // Final verification
    let result = verifier.verify_multiplications(lpzk_proof)
        .map_err(|_| ProtocolError::VerifierError)?;

    Ok(result)
}

/// Runs the complete Justvengers protocol with soldering constraints.
///
/// This version supports cross-repetition data flow where output values from
/// one repetition can be used as inputs in the next repetition.
///
/// # Arguments
/// * `circuits` - Batch of circuits (disjunctive branches)
/// * `active_branch` - Which branch is being executed
/// * `inputs_per_rep` - Inputs for each repetition (must satisfy soldering constraints)
/// * `soldering_constraints` - Constraints linking outputs to inputs across reps
/// * `modulus` - Field modulus
///
/// # Example
/// ```ignore
/// // Chain output 0 to input 0 across repetitions
/// let constraint = SolderingConstraint::new(0, 0);
/// let result = run_protocol_with_soldering::<4>(
///     &circuits,
///     0,
///     &inputs,
///     &[constraint],
///     65537,
/// )?;
/// ```
pub fn run_protocol_with_soldering<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
) -> Result<bool, ProtocolError> {
    use rand::SeedableRng;
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    // Verify inputs
    if inputs_per_rep.len() != R {
        return Err(ProtocolError::InvalidInputs);
    }
    if active_branch >= circuits.num_branches() {
        return Err(ProtocolError::InvalidBranch);
    }

    // Get the circuit for the active branch
    let circuit = circuits.get(active_branch)
        .ok_or(ProtocolError::InvalidBranch)?;

    // Initialize prover
    let mut prover: ProverState<R> = ProverState::new(active_branch, modulus);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep)
        .map_err(|_| ProtocolError::ProverError)?;

    // Setup soldering on prover (validates constraints)
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng)
        .map_err(|_| ProtocolError::SolderingError)?;

    // Initialize verifier
    let mut verifier: VerifierState<R> = VerifierState::new(modulus, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Setup soldering on verifier
    verifier.setup_soldering(soldering_constraints.to_vec())
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 1: Prover commits
    let commitment = prover.commit(&setup_msg.eval_points)
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 1.5: Prover commits soldering
    let soldering_commit = prover.commit_soldering()
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 2: Verifier sends challenge χ
    let chi_msg = verifier.receive_commitment(commitment)
        .map_err(|_| ProtocolError::VerifierError)?;
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 2.5: Verifier sends soldering challenge
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng)
            .map_err(|_| ProtocolError::VerifierError)?
    } else {
        None
    };

    // Phase 3: Prover discloses
    let disclosure = prover.disclose(chi, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 3.5: Prover reveals soldering
    let soldering_reveal = if let Some(ref challenge) = soldering_challenge {
        prover.reveal_soldering(challenge)
            .map_err(|_| ProtocolError::ProverError)?
    } else {
        None
    };

    // Verify soldering if present
    if let Some(ref reveal) = soldering_reveal {
        let soldering_ok = verifier.verify_soldering(reveal)
            .map_err(|_| ProtocolError::VerifierError)?;
        if !soldering_ok {
            return Err(ProtocolError::SolderingError);
        }
    }

    // Phase 4: Verifier sends challenge ρ
    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 5: Prover opens
    let open_msg = prover.open(rho, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;
    verifier.receive_open(open_msg)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 6: Prover sends LPZK proof
    let lpzk_proof = prover.prove_multiplications()
        .map_err(|_| ProtocolError::ProverError)?;

    // Final verification
    let result = verifier.verify_multiplications(lpzk_proof)
        .map_err(|_| ProtocolError::VerifierError)?;

    Ok(result)
}

/// Runs the full JV protocol with soldering using per-repetition active branches.
///
/// This variant allows each repetition j to use a different branch id_j,
/// as described in the JV paper Section 4.3.
pub fn run_protocol_with_soldering_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
) -> Result<bool, ProtocolError> {
    use rand::SeedableRng;
    let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

    // Verify inputs
    if inputs_per_rep.len() != R {
        return Err(ProtocolError::InvalidInputs);
    }
    if active_branches.len() != R {
        return Err(ProtocolError::InvalidInputs);
    }
    for &branch in active_branches {
        if branch >= circuits.num_branches() {
            return Err(ProtocolError::InvalidBranch);
        }
    }

    // Initialize prover with per-rep branches
    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), modulus);
    prover.setup_per_rep(circuits, inputs_per_rep)
        .map_err(|_| ProtocolError::ProverError)?;

    // Setup soldering on prover (validates constraints)
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng)
        .map_err(|_| ProtocolError::SolderingError)?;

    // Initialize verifier
    let mut verifier: VerifierState<R> = VerifierState::new(modulus, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Setup soldering on verifier
    verifier.setup_soldering(soldering_constraints.to_vec())
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 1: Prover commits
    let commitment = prover.commit(&setup_msg.eval_points)
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 1.5: Prover commits soldering
    let soldering_commit = prover.commit_soldering()
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 2: Verifier sends challenge χ
    let chi_msg = verifier.receive_commitment(commitment)
        .map_err(|_| ProtocolError::VerifierError)?;
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 2.5: Verifier sends soldering challenge
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng)
            .map_err(|_| ProtocolError::VerifierError)?
    } else {
        None
    };

    // Phase 3: Prover discloses
    let disclosure = prover.disclose(chi, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;

    // Phase 3.5: Prover reveals soldering
    let soldering_reveal = if let Some(ref challenge) = soldering_challenge {
        prover.reveal_soldering(challenge)
            .map_err(|_| ProtocolError::ProverError)?
    } else {
        None
    };

    // Verify soldering if present
    if let Some(ref reveal) = soldering_reveal {
        let soldering_ok = verifier.verify_soldering(reveal)
            .map_err(|_| ProtocolError::VerifierError)?;
        if !soldering_ok {
            return Err(ProtocolError::SolderingError);
        }
    }

    // Phase 4: Verifier sends challenge ρ
    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng)
        .map_err(|_| ProtocolError::VerifierError)?;
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => return Err(ProtocolError::ProtocolViolation),
    };

    // Phase 5: Prover opens
    let open_msg = prover.open(rho, verifier.topology_vectors())
        .map_err(|_| ProtocolError::ProverError)?;
    verifier.receive_open(open_msg)
        .map_err(|_| ProtocolError::VerifierError)?;

    // Phase 6: Prover sends LPZK proof
    let lpzk_proof = prover.prove_multiplications()
        .map_err(|_| ProtocolError::ProverError)?;

    // Final verification
    let result = verifier.verify_multiplications(lpzk_proof)
        .map_err(|_| ProtocolError::VerifierError)?;

    Ok(result)
}

/// Errors that can occur during protocol execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// Invalid inputs provided.
    InvalidInputs,
    /// Invalid branch index.
    InvalidBranch,
    /// Error in prover.
    ProverError,
    /// Error in verifier.
    VerifierError,
    /// Protocol violation (unexpected message).
    ProtocolViolation,
    /// Soldering constraint violation or verification failure.
    SolderingError,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 65537;

    #[test]
    fn test_protocol_params() {
        let params = ProtocolParams::new(40, 2, TEST_MODULUS);

        assert_eq!(params.repetitions, 40);
        assert_eq!(params.branches, 2);
        assert_eq!(params.modulus, TEST_MODULUS);
    }

    #[test]
    fn test_protocol_params_with_soundness() {
        let params = ProtocolParams::with_soundness(128, 4, TEST_MODULUS);

        assert!(params.soundness_bits >= 128);
    }

    #[test]
    fn test_communication_complexity() {
        let params = ProtocolParams::new(10, 4, TEST_MODULUS);
        let complexity = params.communication_complexity(100);

        // O(R + B + C) = 10 + 4 + 100 = 114
        assert_eq!(complexity, 114);
    }

    #[test]
    fn test_run_protocol_simple() {
        // Create a simple multiplication circuit
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        circuit.add_mul(x, y);

        let batch = CircuitBatch::new(vec![circuit]);

        // Run with 2 repetitions
        let inputs = vec![
            vec![3, 4], // Rep 1: 3 * 4 = 12
            vec![5, 6], // Rep 2: 5 * 6 = 30
        ];

        let result = run_protocol::<2>(&batch, 0, &inputs, TEST_MODULUS);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_run_protocol_invalid_branch() {
        let circuit = Circuit::new();
        let batch = CircuitBatch::new(vec![circuit]);

        let inputs = vec![vec![], vec![]];
        let result = run_protocol::<2>(&batch, 99, &inputs, TEST_MODULUS);

        assert_eq!(result, Err(ProtocolError::InvalidBranch));
    }

    #[test]
    fn test_run_protocol_wrong_rep_count() {
        let circuit = Circuit::new();
        let batch = CircuitBatch::new(vec![circuit]);

        // Provide 1 repetition but protocol expects 2
        let inputs = vec![vec![]];
        let result = run_protocol::<2>(&batch, 0, &inputs, TEST_MODULUS);

        assert_eq!(result, Err(ProtocolError::InvalidInputs));
    }

    #[test]
    fn test_circuit_with_multiple_operations() {
        // z = (x + y) * x
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let sum = circuit.add_add(x, y);
        let _z = circuit.add_mul(sum, x);

        let batch = CircuitBatch::new(vec![circuit]);

        let inputs = vec![
            vec![3, 4], // (3 + 4) * 3 = 21
            vec![2, 5], // (2 + 5) * 2 = 14
        ];

        let result = run_protocol::<2>(&batch, 0, &inputs, TEST_MODULUS);

        assert!(result.is_ok());
    }

    #[test]
    fn test_disjunctive_statement() {
        // Two different circuits
        let mut c1 = Circuit::new();
        let x1 = c1.add_input();
        let y1 = c1.add_input();
        c1.add_mul(x1, y1);

        let mut c2 = Circuit::new();
        let x2 = c2.add_input();
        let y2 = c2.add_input();
        let sum = c2.add_add(x2, y2);
        c2.add_mul(sum, x2);

        let batch = CircuitBatch::new(vec![c1, c2]);

        // Execute branch 0 (simple multiplication)
        let inputs = vec![
            vec![7, 8], // 7 * 8 = 56
            vec![3, 5], // 3 * 5 = 15
        ];

        let result = run_protocol::<2>(&batch, 0, &inputs, TEST_MODULUS);
        assert!(result.is_ok());

        // Execute branch 1 (add then multiply)
        let inputs2 = vec![
            vec![2, 3], // (2 + 3) * 2 = 10
            vec![4, 1], // (4 + 1) * 4 = 20
        ];

        let result2 = run_protocol::<2>(&batch, 1, &inputs2, TEST_MODULUS);
        assert!(result2.is_ok());
    }

    mod integration_tests {
        use super::*;

        #[test]
        fn test_full_protocol_flow() {
            use rand::SeedableRng;
            let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

            // Create circuit: z = x * y
            let mut circuit = Circuit::new();
            let x = circuit.add_input();
            let y = circuit.add_input();
            circuit.add_mul(x, y);

            let batch = CircuitBatch::new(vec![circuit.clone()]);

            // Initialize prover and verifier separately
            let mut prover: ProverState<2> = ProverState::new(0, TEST_MODULUS);
            let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

            // Prover setup
            let inputs = vec![vec![10, 20], vec![5, 7]];
            prover.setup(&mut circuit.clone(), &inputs).unwrap();

            // Verifier setup
            let setup_msg = verifier.setup(&batch, &mut rng).unwrap();

            // Prover commits
            let commit_msg = prover.commit(&setup_msg.eval_points).unwrap();
            assert!(commit_msg.num_commitments > 0);

            // Verifier receives commitment
            let chi_msg = verifier.receive_commitment(commit_msg).unwrap();
            let chi = match chi_msg {
                VerifierMessage::ChallengeChi(c) => c,
                _ => panic!("expected chi"),
            };

            // Prover discloses
            let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();
            assert_eq!(disclosure.eval_points.len(), 2);

            // Verifier receives disclosure
            let rho_msg = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
            let rho = match rho_msg {
                VerifierMessage::ChallengeRho(r) => r,
                _ => panic!("expected rho"),
            };

            // Prover opens
            let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
            assert!(open_msg.active_branches.iter().all(|&b| b == 0));

            // Verifier receives open
            verifier.receive_open(open_msg).unwrap();

            // Prover generates LPZK proof
            let lpzk_proof = prover.prove_multiplications().unwrap();
            assert!(!lpzk_proof.masked_products.is_empty());

            // Final verification
            let result = verifier.verify_multiplications(lpzk_proof).unwrap();
            assert!(result);
        }

        #[test]
        fn test_prover_verifier_state_tracking() {
            use rand::SeedableRng;
            let mut rng = mpz_core::prg::Prg::from_seed(mpz_core::Block::ZERO);

            let circuit = Circuit::new();
            let batch = CircuitBatch::new(vec![circuit]);

            let mut verifier: VerifierState<2> = VerifierState::new(TEST_MODULUS, &mut rng);

            // Track phase transitions
            assert_eq!(verifier.phase(), &VerifierPhase::Init);

            verifier.setup(&batch, &mut rng).unwrap();
            assert_eq!(verifier.phase(), &VerifierPhase::Setup);

            let commit = CommitmentMessage {
                num_commitments: 1,
                masked_evaluations: vec![1],
            };
            verifier.receive_commitment(commit).unwrap();
            assert_eq!(verifier.phase(), &VerifierPhase::ChallengeChiSent);

            let disclosure = DisclosureMessage {
                eval_points: vec![1, 2],
                masked_values: vec![10],
                topology_products: vec![5],
            };
            verifier.receive_disclosure(disclosure, &mut rng).unwrap();
            assert_eq!(verifier.phase(), &VerifierPhase::ChallengeRhoSent);
        }
    }
}
