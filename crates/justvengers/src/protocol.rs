//! Async protocol layer for Justvengers.
//!
//! This module provides async wrappers around ProverState and VerifierState
//! that communicate via mpz-common Context, enabling:
//! - Network communication between prover and verifier
//! - Recording of protocol messages for benchmarks
//! - Replay-based isolated testing

use crate::prover::{
    CommitmentMessage, DisclosureMessage, LpzkProofMessage, OpenMessage, ProverState,
};
use crate::soldering::{SolderingChallengeMessage, SolderingCommitMessage, SolderingRevealMessage};
use crate::topology::{CircuitBatch, TopologyVector};
use crate::verifier::{SetupMessage, VerifierState};
use crate::SolderingConstraint;

use mpz_common::context::Context;
use rand::Rng;
use serio::{SinkExt, stream::IoStreamExt};

/// Error type for async protocol operations.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// Prover error.
    #[error("prover error: {0:?}")]
    Prover(crate::prover::ProverError),
    /// Verifier error.
    #[error("verifier error: {0:?}")]
    Verifier(crate::verifier::VerifierError),
    /// IO error during communication.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<crate::prover::ProverError> for ProtocolError {
    fn from(e: crate::prover::ProverError) -> Self {
        ProtocolError::Prover(e)
    }
}

impl From<crate::verifier::VerifierError> for ProtocolError {
    fn from(e: crate::verifier::VerifierError) -> Self {
        ProtocolError::Verifier(e)
    }
}

/// Runs the prover side of the protocol over a Context.
///
/// This is the async version that sends/receives messages via Context,
/// allowing network communication and message recording.
pub async fn run_prover<const R: usize>(
    ctx: &mut Context,
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
) -> Result<(), ProtocolError> {
    let mut rng = mpz_core::prg::Prg::new();

    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), modulus);
    prover.setup_per_rep(circuits, inputs_per_rep)?;
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng)?;

    // Receive: SetupMessage from verifier
    let setup_msg: SetupMessage = ctx.io_mut().expect_next().await?;

    // Send: CommitmentMessage
    let commitment = prover.commit(&setup_msg.eval_points)?;
    ctx.io_mut().send(commitment).await?;

    // Send: SolderingCommitMessage (optional)
    let soldering_commit = prover.commit_soldering()?;
    ctx.io_mut().send(soldering_commit.clone()).await?;

    // Receive: ChallengeChi
    let chi: u64 = ctx.io_mut().expect_next().await?;

    // Receive: TopologyVectors
    let topology_vectors: Vec<TopologyVector> = ctx.io_mut().expect_next().await?;

    // Receive: SolderingChallengeMessage (optional)
    let soldering_challenge: Option<SolderingChallengeMessage> = ctx.io_mut().expect_next().await?;

    // Send: DisclosureMessage
    let disclosure = prover.disclose(chi, &topology_vectors)?;
    ctx.io_mut().send(disclosure).await?;

    // Send: SolderingRevealMessage (optional)
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering(challenge)?;
        ctx.io_mut().send(reveal).await?;
    }

    // Receive: ChallengeRho
    let rho: u64 = ctx.io_mut().expect_next().await?;

    // Send: OpenMessage
    let open_msg = prover.open(rho, &topology_vectors)?;
    ctx.io_mut().send(open_msg).await?;

    // Send: LpzkProofMessage
    let lpzk_proof = prover.prove_multiplications()?;
    ctx.io_mut().send(lpzk_proof).await?;

    // Receive: verification result
    let _result: bool = ctx.io_mut().expect_next().await?;

    Ok(())
}

/// Runs the prover side with VOLE-based multiplication proofs.
///
/// Same as `run_prover` but uses a VoleSource for multiplication verification.
/// The VoleSource handles OT→VOLE conversion internally (not counted in protocol bytes).
///
/// # Type Parameters
/// - `R`: Number of repetitions
/// - `F`: Field type for IT-MACs
/// - `V`: VoleSource implementation
pub async fn run_prover_with_vole<const R: usize, F, V>(
    ctx: &mut Context,
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
    vole_source: &mut V,
) -> Result<(), ProtocolError>
where
    F: mpz_justvengers_core::ItMacField + From<u64> + Into<u64>,
    V: mpz_justvengers_core::VoleSource<F>,
{
    let mut rng = mpz_core::prg::Prg::new();

    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), modulus);
    prover.setup_per_rep(circuits, inputs_per_rep)?;
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng)?;

    // Receive: SetupMessage from verifier
    let setup_msg: SetupMessage = ctx.io_mut().expect_next().await?;

    // Send: CommitmentMessage
    let commitment = prover.commit(&setup_msg.eval_points)?;
    ctx.io_mut().send(commitment).await?;

    // Send: SolderingCommitMessage (optional)
    let soldering_commit = prover.commit_soldering()?;
    ctx.io_mut().send(soldering_commit.clone()).await?;

    // Receive: ChallengeChi
    let chi: u64 = ctx.io_mut().expect_next().await?;

    // Receive: TopologyVectors
    let topology_vectors: Vec<TopologyVector> = ctx.io_mut().expect_next().await?;

    // Receive: SolderingChallengeMessage (optional)
    let soldering_challenge: Option<SolderingChallengeMessage> = ctx.io_mut().expect_next().await?;

    // Send: DisclosureMessage
    let disclosure = prover.disclose(chi, &topology_vectors)?;
    ctx.io_mut().send(disclosure).await?;

    // Send: SolderingRevealMessage (optional)
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering(challenge)?;
        ctx.io_mut().send(reveal).await?;
    }

    // Receive: ChallengeRho
    let rho: u64 = ctx.io_mut().expect_next().await?;

    // Send: OpenMessage
    let open_msg = prover.open(rho, &topology_vectors)?;
    ctx.io_mut().send(open_msg).await?;

    // Send: LpzkProofMessage (using VOLE - OT consumption happens here, but NOT over ctx)
    let lpzk_proof = prover.prove_multiplications_with_voles(vole_source)?;
    ctx.io_mut().send(lpzk_proof).await?;

    // Receive: verification result
    let _result: bool = ctx.io_mut().expect_next().await?;

    Ok(())
}

/// Runs the verifier side of the protocol over a Context.
///
/// This is the async version that sends/receives messages via Context,
/// allowing network communication and message recording.
pub async fn run_verifier<const R: usize, Rn: Rng>(
    ctx: &mut Context,
    circuits: &CircuitBatch,
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
    rng: &mut Rn,
) -> Result<bool, ProtocolError> {
    let mut verifier: VerifierState<R> = VerifierState::new(modulus, rng);
    let setup_msg = verifier.setup(circuits, rng)?;
    verifier.setup_soldering(soldering_constraints.to_vec())?;

    // Send: SetupMessage
    ctx.io_mut().send(setup_msg).await?;

    // Receive: CommitmentMessage
    let commitment: CommitmentMessage = ctx.io_mut().expect_next().await?;

    // Receive: SolderingCommitMessage (optional)
    let soldering_commit: Option<SolderingCommitMessage> = ctx.io_mut().expect_next().await?;

    // Generate and send: ChallengeChi
    let chi_msg = verifier.receive_commitment(commitment)?;
    let chi = match chi_msg {
        crate::verifier::VerifierMessage::ChallengeChi(c) => c,
        _ => {
            return Err(ProtocolError::Verifier(
                crate::verifier::VerifierError::Phase("expected chi".to_string()),
            ))
        }
    };
    ctx.io_mut().send(chi).await?;

    // Send: TopologyVectors
    ctx.io_mut().send(verifier.topology_vectors().to_vec()).await?;

    // Generate and send: SolderingChallengeMessage (optional)
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, rng)?
    } else {
        None
    };
    ctx.io_mut().send(soldering_challenge.clone()).await?;

    // Receive: DisclosureMessage
    let disclosure: DisclosureMessage = ctx.io_mut().expect_next().await?;

    // Receive: SolderingRevealMessage (optional)
    if soldering_challenge.is_some() {
        let reveal: Option<SolderingRevealMessage> = ctx.io_mut().expect_next().await?;
        if let Some(r) = reveal {
            verifier.receive_soldering_reveal(&r)?;
        }
    }

    // Generate and send: ChallengeRho
    let rho_msg = verifier.receive_disclosure(disclosure, rng)?;
    let rho = match rho_msg {
        crate::verifier::VerifierMessage::ChallengeRho(r) => r,
        _ => {
            return Err(ProtocolError::Verifier(
                crate::verifier::VerifierError::Phase("expected rho".to_string()),
            ))
        }
    };
    ctx.io_mut().send(rho).await?;

    // Receive: OpenMessage
    let open_msg: OpenMessage = ctx.io_mut().expect_next().await?;
    verifier.receive_open(open_msg)?;

    // Receive: LpzkProofMessage
    let lpzk_proof: LpzkProofMessage = ctx.io_mut().expect_next().await?;
    let result = verifier.verify_multiplications(lpzk_proof)?;

    // Send: result
    ctx.io_mut().send(result).await?;

    Ok(result)
}

/// Runs the full protocol with prover and verifier communicating via Context.
///
/// Returns the verification result.
pub async fn run_protocol<const R: usize, Rn: Rng + Clone>(
    ctx_prover: &mut Context,
    ctx_verifier: &mut Context,
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    modulus: u64,
    rng: &mut Rn,
) -> Result<bool, ProtocolError> {
    let circuits_clone = circuits.clone();
    let active_branches = active_branches.to_vec();
    let inputs_per_rep = inputs_per_rep.to_vec();
    let soldering_constraints_clone = soldering_constraints.to_vec();

    let (prover_result, verifier_result) = futures::join!(
        run_prover::<R>(
            ctx_prover,
            &circuits_clone,
            &active_branches,
            &inputs_per_rep,
            &soldering_constraints_clone,
            modulus,
        ),
        run_verifier::<R, _>(ctx_verifier, circuits, soldering_constraints, modulus, rng,)
    );

    prover_result?;
    verifier_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use mpz_common::context::test_st_context;
    use mpz_core::prg::Prg;

    use crate::topology::Circuit;

    const TEST_MODULUS: u64 = 65537;

    fn create_simple_circuit() -> Circuit {
        let mut circuit = Circuit::new();
        let a = circuit.add_input();
        let b = circuit.add_input();
        circuit.add_mul(a, b);
        circuit
    }

    #[test]
    fn test_protocol_communication() {
        block_on(async {
            let (mut ctx_p, mut ctx_v) = test_st_context(1024 * 1024);

            let circuit = create_simple_circuit();
            let circuits = CircuitBatch::new(vec![circuit]);
            let active_branches = vec![0];
            let inputs_per_rep = vec![vec![3, 5]]; // a=3, b=5
            let soldering = vec![];

            let mut rng = Prg::new();

            let result = run_protocol::<1, _>(
                &mut ctx_p,
                &mut ctx_v,
                &circuits,
                &active_branches,
                &inputs_per_rep,
                &soldering,
                TEST_MODULUS,
                &mut rng,
            )
            .await
            .unwrap();

            assert!(result, "Protocol should verify");
        });
    }
}
