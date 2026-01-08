//! VM-style benchmark for Justvengers prover with per-rep active branches.
//!
//! Simulates a simple VM with:
//! - 30 opcodes (branches)
//! - 32-element state vectors
//! - 97 multiplications per circuit
//! - State soldering across repetitions
//!
//! Run with: cargo bench -p mpz-justvengers --bench vm_bench
//!
//! Configure via environment variables:
//!   REPS=1000 cargo bench -p mpz-justvengers --bench vm_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mpz_justvengers::{
    Circuit, CircuitBatch, ProverState, SolderingConstraint, VerifierState, VerifierMessage,
    VoleProvider,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    // JustVengers O(R+B+C) optimized prover
    JVProver, JVVerifier,
};
use mpz_justvengers_core::ItMacField;
use mpz_ot_core::ideal::rcot::IdealRCOT;

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::{Rng, SeedableRng};

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 30;
const STATE_SIZE: usize = 32;
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1; // old_state + new_state + op

/// Goldilocks field for VOLE operations.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct GoldilocksField(u64);

impl std::ops::Add for GoldilocksField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self((self.0 + rhs.0) % MODULUS)
    }
}

impl std::ops::Sub for GoldilocksField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self((self.0 + MODULUS - rhs.0) % MODULUS)
    }
}

impl std::ops::Mul for GoldilocksField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self((self.0 as u128 * rhs.0 as u128 % MODULUS as u128) as u64)
    }
}

impl ItMacField for GoldilocksField {
    fn zero() -> Self { Self(0) }
    fn one() -> Self { Self(1) }
    fn random<R: Rng>(rng: &mut R) -> Self {
        Self(rng.random_range(0..MODULUS))
    }
    fn neg(self) -> Self {
        if self.0 == 0 { Self(0) } else { Self(MODULUS - self.0) }
    }
}

impl From<u64> for GoldilocksField {
    fn from(v: u64) -> Self { Self(v % MODULUS) }
}

impl From<GoldilocksField> for u64 {
    fn from(v: GoldilocksField) -> u64 { v.0 }
}

#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
    /// Protocol communication stats computed during recording.
    stats: ProtocolStats,
}

/// Protocol communication statistics.
#[derive(Clone, Debug, Default)]
struct ProtocolStats {
    /// Bytes sent by prover (P → V).
    prover_sent: usize,
    /// Bytes received by prover (V → P).
    prover_received: usize,
    /// Breakdown by message type.
    commitment: usize,
    soldering_commit: usize,
    disclosure: usize,
    soldering_reveal: usize,
    open: usize,
    lpzk: usize,
}

impl ProtocolStats {
    fn total(&self) -> usize {
        self.prover_sent + self.prover_received
    }

    fn format_kb(&self) -> String {
        format!(
            "P→V: {:.1} KB, V→P: {:.1} KB, total: {:.1} KB",
            self.prover_sent as f64 / 1024.0,
            self.prover_received as f64 / 1024.0,
            self.total() as f64 / 1024.0
        )
    }

    fn format_breakdown(&self) -> String {
        format!(
            "  commit: {:.1} KB, solder_commit: {:.1} KB, disclosure: {:.1} KB, solder_reveal: {:.1} KB, open: {:.1} KB, lpzk: {:.1} KB",
            self.commitment as f64 / 1024.0,
            self.soldering_commit as f64 / 1024.0,
            self.disclosure as f64 / 1024.0,
            self.soldering_reveal as f64 / 1024.0,
            self.open as f64 / 1024.0,
            self.lpzk as f64 / 1024.0,
        )
    }
}

// Message size computation helpers
mod msg_size {
    use mpz_justvengers::{
        prover::{CommitmentMessage, DisclosureMessage, OpenMessage, LpzkProofMessage},
        verifier::SetupMessage,
        soldering::{SolderingCommitMessage, SolderingChallengeMessage, SolderingRevealMessage},
        AggregatedSolderingReveal,
        // JV optimized message types
        JVSetupMessage, JVCommitmentMessage, JVDisclosureMessage, JVOpenMessage, JVLpzkProofMessage,
    };

    pub fn setup_message(msg: &SetupMessage) -> usize {
        8 // max_degree: usize
        + msg.eval_points.len() * 8 // Vec<u64>
        + 8 // encrypted_powers_hash: u64
    }

    pub fn commitment_message(msg: &CommitmentMessage) -> usize {
        8 // num_commitments: usize
        + msg.masked_evaluations.len() * 8 // Vec<u64>
    }

    pub fn soldering_commit_message(msg: &SolderingCommitMessage) -> usize {
        8 // num_constraints: usize
        + msg.commitment_hashes.len() * 16 // Vec<(u64, u64)>
    }

    pub fn disclosure_message(msg: &DisclosureMessage) -> usize {
        msg.eval_points.len() * 8
        + msg.masked_values.len() * 8
        + msg.topology_products.len() * 8
    }

    pub fn soldering_challenge_message(_msg: &SolderingChallengeMessage) -> usize {
        16 // phi: u64 + psi: u64
    }

    pub fn soldering_reveal_message(msg: &SolderingRevealMessage) -> usize {
        msg.masked_polys.iter()
            .map(|(f1, f2)| (f1.len() + f2.len()) * 8)
            .sum()
    }

    pub fn aggregated_soldering_reveal(msg: &AggregatedSolderingReveal) -> usize {
        (msg.aggregated_f1.len() + msg.aggregated_f2.len()) * 8
    }

    pub fn open_message(msg: &OpenMessage) -> usize {
        msg.active_branches.len() * 8 // Vec<usize>
        + 8 // hash_proof: u64
        + msg.vanishing_coeffs.iter().map(|v| v.len() * 8).sum::<usize>()
    }

    pub fn lpzk_proof_message(msg: &LpzkProofMessage) -> usize {
        msg.masked_products.len() * 8
        + msg.mac_tags.len() * 8
    }

    pub fn challenge_u64() -> usize {
        8
    }

    // ========== JV Optimized Message Sizes ==========

    pub fn jv_setup_message(msg: &JVSetupMessage) -> usize {
        msg.eval_points.len() * 8 // Vec<u64>
        + 8 // max_degree: usize
        + 8 // encrypted_powers_hash: u64
    }

    pub fn jv_commitment_message(msg: &JVCommitmentMessage) -> usize {
        8 // num_polynomials: usize
        + msg.poly_commitments.len() * 8 // Vec<u64>
    }

    /// JV Disclosure: O(R) instead of O(RC)!
    pub fn jv_disclosure_message(msg: &JVDisclosureMessage) -> usize {
        msg.topology_products.len() * 8 // O(R) - topology products
        + 8 // aggregated_poly_eval: u64
    }

    pub fn jv_open_message(msg: &JVOpenMessage) -> usize {
        msg.active_branches.len() * 8 // Vec<usize>
        + 8 // hash_proof: u64
        + msg.vp_coefficients.len() * 8 // Vec<u64> - O(R)
    }

    pub fn jv_lpzk_proof_message(msg: &JVLpzkProofMessage) -> usize {
        msg.masked_products.len() * 8
        + msg.mac_tags.len() * 8
    }
}

/// Creates a VM circuit for a specific opcode.
///
/// Inputs: old_state[32] + new_state[32] + op = 65 inputs
/// Mults: ~65 constraint checks + 32 identity mults = 97 total
fn create_vm_circuit(op_value: u64) -> Circuit {
    let mut circuit = Circuit::new();

    let mut old_state = Vec::with_capacity(STATE_SIZE);
    let mut new_state = Vec::with_capacity(STATE_SIZE);

    for _ in 0..STATE_SIZE {
        old_state.push(circuit.add_input());
    }
    for _ in 0..STATE_SIZE {
        new_state.push(circuit.add_input());
    }
    let op = circuit.add_input();

    let op_const = circuit.add_const(op_value);
    let neg_one = circuit.add_const(MODULUS - 1);
    let one = circuit.add_const(1);

    // Constraint: new_state[0] - op = 0
    let neg_op = circuit.add_mul(neg_one, op);
    let check0 = circuit.add_add(new_state[0], neg_op);

    let mut product = check0;

    // Constraints: old_state[j] - new_state[j] = 0 for j = 1..30
    for j in 1..STATE_SIZE - 1 {
        let neg_new = circuit.add_mul(neg_one, new_state[j]);
        let diff = circuit.add_add(old_state[j], neg_new);
        product = circuit.add_mul(product, diff);
    }

    // Constraint: new_state[31] - old_state[31] - OP_i = 0
    let neg_old_31 = circuit.add_mul(neg_one, old_state[STATE_SIZE - 1]);
    let neg_op_const = circuit.add_mul(neg_one, op_const);
    let acc_check = circuit.add_add(new_state[STATE_SIZE - 1], neg_old_31);
    let acc_check = circuit.add_add(acc_check, neg_op_const);
    product = circuit.add_mul(product, acc_check);

    // Constraint: op - OP_i = 0
    let op_check = circuit.add_add(op, neg_op_const);
    let _final = circuit.add_mul(product, op_check);

    // Identity mults to expose new_state as mult_outputs for soldering
    for j in 0..STATE_SIZE {
        circuit.add_mul(new_state[j], one);
    }

    circuit
}

fn create_vm_circuit_batch() -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..NUM_BRANCHES as u64)
        .map(|op| create_vm_circuit(op))
        .collect();
    CircuitBatch::new(circuits)
}

fn state_output_offset(circuit: &Circuit) -> usize {
    circuit.num_mults() - STATE_SIZE
}

fn create_soldering_constraints(state_offset: usize) -> Vec<SolderingConstraint> {
    (0..STATE_SIZE)
        .map(|j| SolderingConstraint::new(j, state_offset + j))
        .collect()
}

/// Generates VM inputs with random ops per repetition.
/// Returns (inputs_per_rep, active_branches, final_accumulator).
fn generate_vm_inputs_per_rep(num_repetitions: usize) -> (Vec<Vec<u64>>, Vec<usize>, u64) {
    let mut rng = Prg::from_seed(Block::ZERO);
    let mut inputs = Vec::with_capacity(num_repetitions);
    let mut active_branches = Vec::with_capacity(num_repetitions);
    let mut old_state = vec![0u64; STATE_SIZE];

    for _ in 0..num_repetitions {
        let active_op = rng.random_range(0..NUM_BRANCHES);
        active_branches.push(active_op);

        let mut new_state = old_state.clone();
        new_state[0] = active_op as u64;
        new_state[STATE_SIZE - 1] = (old_state[STATE_SIZE - 1] + active_op as u64) % MODULUS;

        let mut rep_inputs = Vec::with_capacity(NUM_INPUTS);
        rep_inputs.extend_from_slice(&old_state);
        rep_inputs.extend_from_slice(&new_state);
        rep_inputs.push(active_op as u64);

        inputs.push(rep_inputs);
        old_state = new_state;
    }

    let final_acc = old_state[STATE_SIZE - 1];
    (inputs, active_branches, final_acc)
}

fn record_verifier_messages_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::from_seed(Block::ZERO);
    let mut stats = ProtocolStats::default();

    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), MODULUS);
    prover.setup_per_rep(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier: VerifierState<R> = VerifierState::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    // V → P: SetupMessage
    stats.prover_received += msg_size::setup_message(&setup_msg);

    let eval_points = setup_msg.eval_points.clone();
    let topology_vectors = verifier.topology_vectors().to_vec();

    let commitment = prover.commit(&setup_msg.eval_points).unwrap();
    // P → V: CommitmentMessage
    let commit_size = msg_size::commitment_message(&commitment);
    stats.prover_sent += commit_size;
    stats.commitment = commit_size;

    let soldering_commit = prover.commit_soldering().unwrap();
    // P → V: SolderingCommitMessage (optional)
    if let Some(ref commit) = soldering_commit {
        let size = msg_size::soldering_commit_message(commit);
        stats.prover_sent += size;
        stats.soldering_commit = size;
    }

    let chi_msg = verifier.receive_commitment(commitment).unwrap();
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => panic!("expected chi"),
    };
    // V → P: ChallengeChi
    stats.prover_received += msg_size::challenge_u64();

    let soldering_challenge = if let Some(commit) = soldering_commit {
        let challenge = verifier.receive_soldering_commit(commit, &mut rng).unwrap();
        // V → P: SolderingChallengeMessage (optional)
        if let Some(ref ch) = challenge {
            stats.prover_received += msg_size::soldering_challenge_message(ch);
        }
        challenge
    } else {
        None
    };

    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();
    // P → V: DisclosureMessage
    let disclosure_size = msg_size::disclosure_message(&disclosure);
    stats.prover_sent += disclosure_size;
    stats.disclosure = disclosure_size;

    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering(challenge).unwrap();
        // P → V: SolderingRevealMessage (optional)
        if let Some(ref rev) = reveal {
            let size = msg_size::soldering_reveal_message(rev);
            stats.prover_sent += size;
            stats.soldering_reveal = size;
        }
    }

    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => panic!("expected rho"),
    };
    // V → P: ChallengeRho
    stats.prover_received += msg_size::challenge_u64();

    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    // P → V: OpenMessage
    let open_size = msg_size::open_message(&open_msg);
    stats.prover_sent += open_size;
    stats.open = open_size;

    verifier.receive_open(open_msg).unwrap();

    let lpzk_proof = prover.prove_multiplications().unwrap();
    // P → V: LpzkProofMessage
    let lpzk_size = msg_size::lpzk_proof_message(&lpzk_proof);
    stats.prover_sent += lpzk_size;
    stats.lpzk = lpzk_size;

    let result = verifier.verify_multiplications(lpzk_proof).unwrap();
    assert!(result, "Protocol verification failed during recording");

    RecordedVerifierMessages {
        eval_points,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        stats,
    }
}

fn run_prover_with_replay_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
) {
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), MODULUS);
    prover.setup_per_rep(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let _lpzk_proof = prover.prove_multiplications().unwrap();
}

use mpz_justvengers_core::VoleSource;

// ============================================================================
// JV Optimized Protocol - O(R+B+C) Communication
// ============================================================================

/// Recorded verifier messages for JV protocol replay.
#[derive(Clone)]
struct JVRecordedMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
    /// Protocol communication stats.
    stats: JVProtocolStats,
}

/// JV Protocol communication statistics.
#[derive(Clone, Debug, Default)]
struct JVProtocolStats {
    prover_sent: usize,
    prover_received: usize,
    commitment: usize,
    soldering_commit: usize,
    disclosure: usize,  // This is O(R) instead of O(RC)!
    soldering_reveal: usize,
    open: usize,
    lpzk: usize,
}

impl JVProtocolStats {
    fn total(&self) -> usize {
        self.prover_sent + self.prover_received
    }

    fn format_kb(&self) -> String {
        format!(
            "P→V: {:.1} KB, V→P: {:.1} KB, total: {:.1} KB",
            self.prover_sent as f64 / 1024.0,
            self.prover_received as f64 / 1024.0,
            self.total() as f64 / 1024.0
        )
    }

    fn format_breakdown(&self) -> String {
        format!(
            "  commit: {:.1} KB, solder_commit: {:.1} KB, disclosure: {:.1} KB, solder_reveal: {:.1} KB, open: {:.1} KB, lpzk: {:.1} KB",
            self.commitment as f64 / 1024.0,
            self.soldering_commit as f64 / 1024.0,
            self.disclosure as f64 / 1024.0,
            self.soldering_reveal as f64 / 1024.0,
            self.open as f64 / 1024.0,
            self.lpzk as f64 / 1024.0,
        )
    }
}

/// Records JV verifier messages for replay benchmarking.
fn jv_record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> JVRecordedMessages {
    let mut rng = Prg::from_seed(Block::ZERO);
    let mut stats = JVProtocolStats::default();

    // Use JVProver instead of ProverState
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    // V → P: SetupMessage
    stats.prover_received += msg_size::jv_setup_message(&setup_msg);

    let eval_points = setup_msg.eval_points.clone();
    let topology_vectors = verifier.topology_vectors().to_vec();

    // P → V: CommitmentMessage
    let commitment = prover.commit(&setup_msg.eval_points).unwrap();
    let commit_size = msg_size::jv_commitment_message(&commitment);
    stats.prover_sent += commit_size;
    stats.commitment = commit_size;

    // P → V: SolderingCommitMessage (optional)
    let soldering_commit = prover.commit_soldering().unwrap();
    if let Some(ref commit) = soldering_commit {
        let size = msg_size::soldering_commit_message(commit);
        stats.prover_sent += size;
        stats.soldering_commit = size;
    }

    // V → P: ChallengeChi
    let chi = verifier.receive_commitment(commitment).unwrap();
    stats.prover_received += msg_size::challenge_u64();

    // V → P: SolderingChallengeMessage (optional)
    let soldering_challenge = if let Some(commit) = soldering_commit {
        let challenge = verifier.receive_soldering_commit(commit, &mut rng).unwrap();
        if let Some(ref ch) = challenge {
            stats.prover_received += msg_size::soldering_challenge_message(ch);
        }
        challenge
    } else {
        None
    };

    // P → V: DisclosureMessage - THIS IS O(R) instead of O(RC)!
    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();
    let disclosure_size = msg_size::jv_disclosure_message(&disclosure);
    stats.prover_sent += disclosure_size;
    stats.disclosure = disclosure_size;

    // P → V: AggregatedSolderingReveal (optional) - O(R) instead of O(S×R)!
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        if let Some(ref rev) = reveal {
            let size = msg_size::aggregated_soldering_reveal(rev);
            stats.prover_sent += size;
            stats.soldering_reveal = size;
            verifier.receive_soldering_reveal_aggregated(rev).unwrap();
        }
    }

    // V → P: ChallengeRho
    let rho = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    stats.prover_received += msg_size::challenge_u64();

    // P → V: OpenMessage
    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    let open_size = msg_size::jv_open_message(&open_msg);
    stats.prover_sent += open_size;
    stats.open = open_size;

    verifier.receive_open(open_msg).unwrap();

    // P → V: LpzkProofMessage
    let lpzk_proof = prover.prove_multiplications().unwrap();
    let lpzk_size = msg_size::jv_lpzk_proof_message(&lpzk_proof);
    stats.prover_sent += lpzk_size;
    stats.lpzk = lpzk_size;

    let result = verifier.verify_multiplications(lpzk_proof).unwrap();
    assert!(result, "JV Protocol verification failed during recording");

    JVRecordedMessages {
        eval_points,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        stats,
    }
}

/// Runs JV prover with replay (no VOLE).
fn jv_run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &JVRecordedMessages,
) {
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let _lpzk_proof = prover.prove_multiplications().unwrap();
}

/// Runs JV prover with replay and VOLE source.
fn jv_run_prover_with_replay_vole<const R: usize, V>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &JVRecordedMessages,
    vole_source: &mut V,
) where
    V: VoleSource<GoldilocksField>,
{
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let _lpzk_proof = prover.prove_multiplications_with_voles(vole_source).unwrap();
}

/// Runs prover with injected VOLE source.
///
/// The VoleProvider is created externally and passed in, allowing:
/// - Pre-generation of VOLEs
/// - Reuse across protocol runs
/// - External control of OT backend
fn run_prover_with_replay_per_rep_vole<const R: usize, V>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
    vole_source: &mut V,
) where
    V: VoleSource<GoldilocksField>,
{
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), MODULUS);
    prover.setup_per_rep(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();

    let _lpzk_proof = prover.prove_multiplications_with_voles(vole_source).unwrap();
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_vm_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_prover");
    group.sample_size(10);

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);

    let num_mults = sample_circuit.num_mults();

    // 10 reps
    {
        const R: usize = 10;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "10_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    // 100 reps
    {
        const R: usize = 100;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "100_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    // 1000 reps
    {
        const R: usize = 1000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        println!("\n[1K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "1K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    // 10000 reps
    {
        const R: usize = 10000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "10K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    // 25000 reps
    {
        const R: usize = 25000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "25K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    // 100000 reps
    {
        const R: usize = 100000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = record_verifier_messages_per_rep::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("per_rep_branches", "100K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );
                black_box(())
            });
        });
    }

    group.finish();
}

fn bench_vm_e2e(c: &mut Criterion) {
    use mpz_justvengers::run_protocol_with_soldering_per_rep;

    let mut group = c.benchmark_group("vm_e2e");
    group.sample_size(10);

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);
    let num_mults = sample_circuit.num_mults();

    // 10 reps - full e2e
    {
        const R: usize = 10;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("full_protocol", "10_reps"), |b| {
            b.iter(|| {
                black_box(run_protocol_with_soldering_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    MODULUS,
                ))
            });
        });
    }

    // 100 reps - full e2e
    {
        const R: usize = 100;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("full_protocol", "100_reps"), |b| {
            b.iter(|| {
                black_box(run_protocol_with_soldering_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    MODULUS,
                ))
            });
        });
    }

    // 1000 reps - full e2e
    {
        const R: usize = 1000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("full_protocol", "1K_reps"), |b| {
            b.iter(|| {
                black_box(run_protocol_with_soldering_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    MODULUS,
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark VM prover with VOLE generation cost - JustVengers O(R+B+C) protocol.
///
/// Uses VoleProvider backed by IdealRCOT:
/// - OT correlations are free (ideal)
/// - VOLE conversion cost is included
fn bench_vm_prover_with_vole(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_prover_with_vole");
    group.sample_size(10);

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);

    let num_mults = sample_circuit.num_mults();

    // 100 reps with VOLE - JV protocol
    {
        const R: usize = 100;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Run once to get stats
        let rcot = IdealRCOT::default();
        let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
        jv_run_prover_with_replay_vole::<R, _>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
            &recorded,
            &mut vole_source,
        );
        let vole_stats = vole_source.stats();
        println!("\n[JV 100 reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 100 reps] VOLEs consumed: {}, OTs consumed: {}", vole_stats.voles_consumed, vole_stats.ots_consumed);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "100_reps"), |b| {
            b.iter(|| {
                let rcot = IdealRCOT::default();
                let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
                jv_run_prover_with_replay_vole::<R, _>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &mut vole_source,
                );
                black_box(())
            });
        });
    }

    // 1000 reps with VOLE - JV protocol
    {
        const R: usize = 1000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Run once to get stats
        let rcot = IdealRCOT::default();
        let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
        jv_run_prover_with_replay_vole::<R, _>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
            &recorded,
            &mut vole_source,
        );
        let vole_stats = vole_source.stats();
        println!("\n[JV 1K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 1K reps] VOLEs consumed: {}, OTs consumed: {}", vole_stats.voles_consumed, vole_stats.ots_consumed);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "1K_reps"), |b| {
            b.iter(|| {
                let rcot = IdealRCOT::default();
                let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
                jv_run_prover_with_replay_vole::<R, _>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &mut vole_source,
                );
                black_box(())
            });
        });
    }

    // 10000 reps with VOLE - JV protocol
    {
        const R: usize = 10000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Run once to get stats
        let rcot = IdealRCOT::default();
        let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
        jv_run_prover_with_replay_vole::<R, _>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
            &recorded,
            &mut vole_source,
        );
        let vole_stats = vole_source.stats();
        println!("\n[JV 10K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 10K reps] VOLEs consumed: {}, OTs consumed: {}", vole_stats.voles_consumed, vole_stats.ots_consumed);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "10K_reps"), |b| {
            b.iter(|| {
                let rcot = IdealRCOT::default();
                let mut vole_source = VoleProvider::<GoldilocksField, _>::new(rcot);
                jv_run_prover_with_replay_vole::<R, _>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &mut vole_source,
                );
                black_box(())
            });
        });
    }

    group.finish();
}

/// Benchmark with actual recorded communication bytes.
///
/// Uses mpz-common recording infrastructure to measure actual serialized bytes.
fn bench_vm_recorded_communication(c: &mut Criterion) {
    use futures::executor::block_on;
    use mpz_common::context::{recording_st_context_with_limit, replay_st_context};
    use mpz_justvengers::protocol::{run_prover, run_verifier};

    let mut group = c.benchmark_group("vm_recorded");
    group.sample_size(10);

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);

    let num_mults = sample_circuit.num_mults();

    // Helper to run protocol with recording
    fn record_and_run<const R: usize>(
        circuits: &CircuitBatch,
        branches: &[usize],
        inputs: &[Vec<u64>],
        soldering: &[SolderingConstraint],
    ) -> (Vec<u8>, usize) {
        use futures::executor::block_on;
        use mpz_common::context::recording_st_context_with_limit;
        use mpz_justvengers::protocol::{run_prover, run_verifier};

        // Large buffer for recording
        const IO_BUFFER: usize = 64 * 1024 * 1024; // 64 MB
        const MAX_FRAME: usize = 16 * 1024 * 1024; // 16 MB frames

        block_on(async {
            let (mut ctx_p, mut ctx_v, recorded) =
                recording_st_context_with_limit(IO_BUFFER, MAX_FRAME);

            let circuits_v = circuits.clone();
            let soldering_v = soldering.to_vec();

            // Run protocol
            let mut rng = Prg::from_seed(Block::ZERO);

            let result = futures::join!(
                run_prover::<R>(
                    &mut ctx_p,
                    circuits,
                    branches,
                    inputs,
                    soldering,
                    MODULUS,
                ),
                run_verifier::<R, _>(
                    &mut ctx_v,
                    &circuits_v,
                    &soldering_v,
                    MODULUS,
                    &mut rng,
                )
            );

            // Check both sides succeeded
            result.0.unwrap();
            result.1.unwrap();

            let recorded_bytes = recorded.lock().unwrap().clone();
            let bytes_v_to_p = recorded_bytes.len();
            (recorded_bytes, bytes_v_to_p)
        })
    }

    // 100 reps
    {
        const R: usize = 100;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        // Record once to get bytes
        let (recorded_bytes, bytes_v_to_p) = record_and_run::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        println!("\n[100 reps] Recorded V→P: {:.1} KB", bytes_v_to_p as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("recorded", "100_reps"), |b| {
            b.iter(|| {
                block_on(async {
                    const MAX_FRAME: usize = 16 * 1024 * 1024;
                    let mut ctx_p = replay_st_context(recorded_bytes.clone(), MAX_FRAME);

                    run_prover::<R>(
                        &mut ctx_p,
                        &circuits,
                        &branches,
                        &inputs,
                        &soldering,
                        MODULUS,
                    ).await.unwrap();
                });
                black_box(())
            });
        });
    }

    // 1000 reps
    {
        const R: usize = 1000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let (recorded_bytes, bytes_v_to_p) = record_and_run::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        println!("\n[1K reps] Recorded V→P: {:.1} KB", bytes_v_to_p as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("recorded", "1K_reps"), |b| {
            b.iter(|| {
                block_on(async {
                    const MAX_FRAME: usize = 16 * 1024 * 1024;
                    let mut ctx_p = replay_st_context(recorded_bytes.clone(), MAX_FRAME);

                    run_prover::<R>(
                        &mut ctx_p,
                        &circuits,
                        &branches,
                        &inputs,
                        &soldering,
                        MODULUS,
                    ).await.unwrap();
                });
                black_box(())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_vm_prover, bench_vm_e2e, bench_vm_prover_with_vole, bench_vm_recorded_communication);
criterion_main!(benches);
