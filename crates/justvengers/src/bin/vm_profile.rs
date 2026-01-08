//! VM-like profiling binary for the Justvengers prover.
//!
//! Simulates a simple VM with 30 opcodes and 32-element state vectors.
//! Each branch represents one opcode, with constraints:
//!   - new_state[0] == op (first element holds executed op)
//!   - new_state[1..30] == old_state[1..30] (unchanged)
//!   - new_state[31] == old_state[31] + OP_i (accumulator)
//!
//! Usage:
//!   REPS=100 ITERS=1000 cargo run --release --bin vm_profile
//!
//! Then profile with:
//!   samply record target/release/vm_profile
//!   perf record -g target/release/vm_profile && perf report

use std::env;

use mpz_justvengers::{
    Circuit, CircuitBatch, ProverState, SolderingConstraint, VerifierState, VerifierMessage,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::SeedableRng;

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 30;
const STATE_SIZE: usize = 32;

// Inputs: old_state[32] + new_state[32] + op = 65 inputs
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
}

/// Creates a VM circuit for a specific opcode.
///
/// Inputs layout:
///   0..31: old_state[0..31]
///   32..63: new_state[0..31]
///   64: op
///
/// Constraints (all should be 0 for valid execution):
///   - new_state[0] - op = 0
///   - old_state[j] - new_state[j] = 0  for j = 1..30
///   - new_state[31] - old_state[31] - OP_i = 0
///   - op - OP_i = 0 (branch selection)
///
/// Mult outputs layout (for soldering support):
///   - First ~65 outputs: internal constraint check multiplications
///   - Last 32 outputs: new_state[0..31] (identity mults for soldering)
///
/// The identity multiplications `new_state[j] * 1` expose state values as
/// mult_outputs, enabling soldering constraints to chain state across reps.
fn create_vm_circuit(op_value: u64) -> Circuit {
    let mut circuit = Circuit::new();

    // Add all inputs
    let mut old_state = Vec::with_capacity(STATE_SIZE);
    let mut new_state = Vec::with_capacity(STATE_SIZE);

    for _ in 0..STATE_SIZE {
        old_state.push(circuit.add_input());
    }
    for _ in 0..STATE_SIZE {
        new_state.push(circuit.add_input());
    }
    let op = circuit.add_input();

    // Constants
    let op_const = circuit.add_const(op_value);
    let neg_one = circuit.add_const(MODULUS - 1); // -1 in the field
    let one = circuit.add_const(1);

    // Constraint 1: new_state[0] - op = 0
    // Compute: new_state[0] + (-1) * op
    let neg_op = circuit.add_mul(neg_one, op);
    let check0 = circuit.add_add(new_state[0], neg_op);

    // Start building the product with check0
    let mut product = check0;

    // Constraints 2: old_state[j] - new_state[j] = 0 for j = 1..30
    for j in 1..STATE_SIZE - 1 {
        // Compute: old_state[j] + (-1) * new_state[j]
        let neg_new = circuit.add_mul(neg_one, new_state[j]);
        let diff = circuit.add_add(old_state[j], neg_new);
        product = circuit.add_mul(product, diff);
    }

    // Constraint 3: new_state[31] - old_state[31] - OP_i = 0
    // Compute: new_state[31] + (-1) * old_state[31] + (-1) * OP_i
    let neg_old_31 = circuit.add_mul(neg_one, old_state[STATE_SIZE - 1]);
    let neg_op_const = circuit.add_mul(neg_one, op_const);
    let acc_check = circuit.add_add(new_state[STATE_SIZE - 1], neg_old_31);
    let acc_check = circuit.add_add(acc_check, neg_op_const);
    product = circuit.add_mul(product, acc_check);

    // Constraint 4: op - OP_i = 0 (branch selection)
    // Compute: op + (-1) * OP_i
    let op_check = circuit.add_add(op, neg_op_const);
    let _final = circuit.add_mul(product, op_check);

    // Identity multiplications to expose new_state as mult_outputs for soldering.
    // These appear at the END of mult_outputs, after the constraint check mults.
    // mult_output[num_constraint_mults + j] = new_state[j] * 1 = new_state[j]
    for j in 0..STATE_SIZE {
        circuit.add_mul(new_state[j], one);
    }

    circuit
}

/// Returns the mult_output index where new_state values start.
/// Used for constructing soldering constraints.
fn state_output_offset(circuit: &Circuit) -> usize {
    // The identity mults for new_state are added at the end
    // Total mults = constraint_mults + STATE_SIZE
    // So state outputs start at: total_mults - STATE_SIZE
    circuit.num_mults() - STATE_SIZE
}

/// Creates a batch of 30 VM circuits, one per opcode.
fn create_vm_circuit_batch() -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..NUM_BRANCHES as u64)
        .map(|op| create_vm_circuit(op))
        .collect();
    CircuitBatch::new(circuits)
}

/// Generates valid state transitions for the VM with per-rep active ops.
///
/// Per JV paper, each repetition j can execute a different branch id_j ∈ [B].
/// Uses random ops seeded deterministically for reproducibility.
///
/// Each iteration:
///   - new_state[0] = op
///   - new_state[1..30] = old_state[1..30]
///   - new_state[31] = old_state[31] + op
///
/// Returns (inputs_per_rep, active_branches, final_accumulator, expected_sum).
fn generate_vm_inputs_per_rep(num_repetitions: usize) -> (Vec<Vec<u64>>, Vec<usize>, u64, u64) {
    use rand::Rng;

    let mut rng = Prg::from_seed(Block::ZERO);
    let mut inputs = Vec::with_capacity(num_repetitions);
    let mut active_branches = Vec::with_capacity(num_repetitions);
    let mut old_state = vec![0u64; STATE_SIZE];
    let mut expected_sum: u64 = 0;

    for _ in 0..num_repetitions {
        // Random op for this rep
        let active_op = rng.random_range(0..NUM_BRANCHES);
        active_branches.push(active_op);
        expected_sum = (expected_sum + active_op as u64) % MODULUS;

        // Compute new state
        let mut new_state = old_state.clone();
        new_state[0] = active_op as u64;
        new_state[STATE_SIZE - 1] = (old_state[STATE_SIZE - 1] + active_op as u64) % MODULUS;

        // Build input vector: old_state + new_state + op
        let mut rep_inputs = Vec::with_capacity(NUM_INPUTS);
        rep_inputs.extend_from_slice(&old_state);
        rep_inputs.extend_from_slice(&new_state);
        rep_inputs.push(active_op as u64);

        inputs.push(rep_inputs);

        // new_state becomes old_state for next iteration
        old_state = new_state;
    }

    let final_acc = old_state[STATE_SIZE - 1];
    assert_eq!(final_acc, expected_sum, "Accumulator mismatch - state chain broken!");
    (inputs, active_branches, final_acc, expected_sum)
}

/// Generates valid state transitions for the VM with a single op (legacy).
fn generate_vm_inputs(
    num_repetitions: usize,
    active_op: usize,
) -> Vec<Vec<u64>> {
    let mut inputs = Vec::with_capacity(num_repetitions);
    let mut old_state = vec![0u64; STATE_SIZE];

    for _ in 0..num_repetitions {
        // Compute new state
        let mut new_state = old_state.clone();
        new_state[0] = active_op as u64;
        new_state[STATE_SIZE - 1] = (old_state[STATE_SIZE - 1] + active_op as u64) % MODULUS;

        // Build input vector: old_state + new_state + op
        let mut rep_inputs = Vec::with_capacity(NUM_INPUTS);
        rep_inputs.extend_from_slice(&old_state);
        rep_inputs.extend_from_slice(&new_state);
        rep_inputs.push(active_op as u64);

        inputs.push(rep_inputs);

        // new_state becomes old_state for next iteration
        old_state = new_state;
    }

    inputs
}

/// Creates soldering constraints to chain new_state → old_state across iterations.
///
/// For each state element j:
///   mult_output[state_offset + j] at rep i == input[j] at rep i+1
///
/// Where:
///   - state_offset = num_constraint_mults (identity mults are at end)
///   - mult_output[state_offset + j] = new_state[j] (from identity mult)
///   - input[j] = old_state[j]
///
/// This chains the state: new_state of rep i becomes old_state of rep i+1.
fn create_soldering_constraints(state_offset: usize) -> Vec<SolderingConstraint> {
    let mut constraints = Vec::with_capacity(STATE_SIZE);
    for j in 0..STATE_SIZE {
        // SolderingConstraint::new(target_input_idx, source_output_idx)
        // Target: old_state[j] at input index j
        // Source: new_state[j] exposed as mult_output[state_offset + j]
        constraints.push(SolderingConstraint::new(j, state_offset + j));
    }
    constraints
}

macro_rules! impl_profile {
    ($r:expr, $circuits:expr, $active_branch:expr, $inputs:expr, $soldering:expr, $iters:expr) => {{
        const R: usize = $r;

        eprintln!("Recording verifier messages for R={}...", R);
        let recorded = record_verifier_messages::<R>(
            &$circuits,
            $active_branch,
            &$inputs,
            &$soldering,
        );
        eprintln!("Recording done. Starting {} iterations of prover replay...", $iters);

        for i in 0..$iters {
            run_prover_with_replay::<R>(
                &$circuits,
                $active_branch,
                &$inputs,
                &$soldering,
                &recorded,
            );
            if (i + 1) % 100 == 0 {
                eprintln!("  {} iterations done", i + 1);
            }
        }
    }};
}

fn record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::from_seed(Block::ZERO);

    let circuit = circuits.get(active_branch).unwrap();

    let mut prover: ProverState<R> = ProverState::new(active_branch, MODULUS);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier: VerifierState<R> = VerifierState::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let eval_points = setup_msg.eval_points.clone();
    let topology_vectors = verifier.topology_vectors().to_vec();

    let commitment = prover.commit(&setup_msg.eval_points).unwrap();
    let soldering_commit = prover.commit_soldering().unwrap();

    let chi_msg = verifier.receive_commitment(commitment).unwrap();
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => panic!("expected chi"),
    };

    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    let _soldering_reveal = if let Some(ref challenge) = soldering_challenge {
        prover.reveal_soldering(challenge).unwrap()
    } else {
        None
    };

    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => panic!("expected rho"),
    };

    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    verifier.receive_open(open_msg).unwrap();

    let lpzk_proof = prover.prove_multiplications().unwrap();
    let result = verifier.verify_multiplications(lpzk_proof).unwrap();
    assert!(result, "Protocol verification failed during recording");

    RecordedVerifierMessages {
        eval_points,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
    }
}

fn run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
) {
    let mut rng = Prg::from_seed(Block::ZERO);

    let circuit = circuits.get(active_branch).unwrap();

    let mut prover: ProverState<R> = ProverState::new(active_branch, MODULUS);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _soldering_reveal = prover.reveal_soldering(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let _lpzk_proof = prover.prove_multiplications().unwrap();
}

// ==================== Per-Rep Active Branch Support ====================

/// Records verifier messages using per-repetition active branches.
fn record_verifier_messages_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::from_seed(Block::ZERO);

    // Use per-rep active branches
    let mut prover: ProverState<R> = ProverState::new_per_rep(active_branches.to_vec(), MODULUS);
    prover.setup_per_rep(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier: VerifierState<R> = VerifierState::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let eval_points = setup_msg.eval_points.clone();
    let topology_vectors = verifier.topology_vectors().to_vec();

    let commitment = prover.commit(&setup_msg.eval_points).unwrap();
    let soldering_commit = prover.commit_soldering().unwrap();

    let chi_msg = verifier.receive_commitment(commitment).unwrap();
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => panic!("expected chi"),
    };

    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    let _soldering_reveal = if let Some(ref challenge) = soldering_challenge {
        prover.reveal_soldering(challenge).unwrap()
    } else {
        None
    };

    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => panic!("expected rho"),
    };

    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    verifier.receive_open(open_msg).unwrap();

    let lpzk_proof = prover.prove_multiplications().unwrap();
    let result = verifier.verify_multiplications(lpzk_proof).unwrap();
    assert!(result, "Protocol verification failed during recording (per-rep)");

    RecordedVerifierMessages {
        eval_points,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
    }
}

/// Runs prover with replay using per-repetition active branches.
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
        let _soldering_reveal = prover.reveal_soldering(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let _lpzk_proof = prover.prove_multiplications().unwrap();
}

macro_rules! impl_profile_per_rep {
    ($r:expr, $circuits:expr, $active_branches:expr, $inputs:expr, $soldering:expr, $iters:expr, $final_acc:expr) => {{
        const R: usize = $r;

        eprintln!("Recording verifier messages for R={} (per-rep branches)...", R);
        let recorded = record_verifier_messages_per_rep::<R>(
            &$circuits,
            &$active_branches,
            &$inputs,
            &$soldering,
        );
        eprintln!("Recording done. Starting {} iterations of prover replay...", $iters);

        for i in 0..$iters {
            run_prover_with_replay_per_rep::<R>(
                &$circuits,
                &$active_branches,
                &$inputs,
                &$soldering,
                &recorded,
            );
            if (i + 1) % 100 == 0 {
                eprintln!("  {} iterations done", i + 1);
            }
        }

        eprintln!("E2E proof verified. Final accumulator (sum of {} random ops): {}", R, $final_acc);
    }};
}

fn main() {
    let reps: usize = env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let iters: usize = env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    // MODE: "per_rep" (default) uses cycling ops 0,1,2,...,29,0,1,...
    //       "single" uses a fixed op for all reps
    let mode = env::var("MODE").unwrap_or_else(|_| "per_rep".to_string());
    let active_op: usize = env::var("OP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15); // Only used in single mode

    eprintln!("VM Profile Configuration:");
    eprintln!("  Branches: {}", NUM_BRANCHES);
    eprintln!("  State size: {}", STATE_SIZE);
    eprintln!("  Inputs per rep: {}", NUM_INPUTS);
    eprintln!("  Mode: {}", mode);
    if mode == "single" {
        eprintln!("  Active op: {}", active_op);
    } else {
        eprintln!("  Per-rep ops: random from 0..{}", NUM_BRANCHES);
    }
    eprintln!("  REPS={} ITERS={}", reps, iters);

    let circuits = create_vm_circuit_batch();

    // Verify circuit structure
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    eprintln!("  Mults per circuit: {}", sample_circuit.num_mults());
    eprintln!("  Circuit inputs: {}", sample_circuit.num_inputs());
    eprintln!("  State output offset: {}", state_offset);

    // Create soldering constraints that chain new_state → old_state across reps.
    // Works for both single and per-rep modes because the identity multiplications
    // produce the same new_state values regardless of which branch circuit is used.
    let soldering = create_soldering_constraints(state_offset);

    if mode == "per_rep" {
        // Per-rep mode: each rep uses a different branch (cycling through all ops)
        match reps {
            10 => {
                let (inputs, branches, final_acc, _) = generate_vm_inputs_per_rep(10);
                impl_profile_per_rep!(10, circuits, branches, inputs, soldering, iters, final_acc);
            }
            100 => {
                let (inputs, branches, final_acc, _) = generate_vm_inputs_per_rep(100);
                impl_profile_per_rep!(100, circuits, branches, inputs, soldering, iters, final_acc);
            }
            1000 => {
                let (inputs, branches, final_acc, _) = generate_vm_inputs_per_rep(1000);
                impl_profile_per_rep!(1000, circuits, branches, inputs, soldering, iters, final_acc);
            }
            10000 => {
                let (inputs, branches, final_acc, _) = generate_vm_inputs_per_rep(10000);
                impl_profile_per_rep!(10000, circuits, branches, inputs, soldering, iters, final_acc);
            }
            _ => {
                eprintln!("Unsupported REPS value: {}. Supported: 10, 100, 1000, 10000", reps);
                std::process::exit(1);
            }
        }
    } else {
        // Single op mode: all reps use the same branch
        match reps {
            10 => {
                let inputs = generate_vm_inputs(10, active_op);
                impl_profile!(10, circuits, active_op, inputs, soldering, iters);
            }
            100 => {
                let inputs = generate_vm_inputs(100, active_op);
                impl_profile!(100, circuits, active_op, inputs, soldering, iters);
            }
            1000 => {
                let inputs = generate_vm_inputs(1000, active_op);
                impl_profile!(1000, circuits, active_op, inputs, soldering, iters);
            }
            10000 => {
                let inputs = generate_vm_inputs(10000, active_op);
                impl_profile!(10000, circuits, active_op, inputs, soldering, iters);
            }
            _ => {
                eprintln!("Unsupported REPS value: {}. Supported: 10, 100, 1000, 10000", reps);
                std::process::exit(1);
            }
        }
    }

    eprintln!("Done!");
}
