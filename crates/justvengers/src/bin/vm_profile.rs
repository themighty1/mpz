//! VM-like profiling binary for the JustVengers prover.
//!
//! Simulates a simple VM with 30 opcodes and 32-element state vectors.
//! Uses JVProver (the O(R+B+C) optimized protocol) to match the benchmark.
//!
//! Usage:
//!   REPS=50000 ITERS=10 cargo run --release --bin vm_profile
//!
//! Then profile with:
//!   samply record target/release/vm_profile
//!   perf record -g target/release/vm_profile && perf report

use std::env;
use std::time::Instant;

use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    JVProver, JVVerifier,
};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::{Rng, SeedableRng};

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 30;
const STATE_SIZE: usize = 32;
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
    gamma: u64,
}

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

    let neg_op = circuit.add_mul(neg_one, op);
    let check0 = circuit.add_add(new_state[0], neg_op);

    let mut product = check0;

    for j in 1..STATE_SIZE - 1 {
        let neg_new = circuit.add_mul(neg_one, new_state[j]);
        let diff = circuit.add_add(old_state[j], neg_new);
        product = circuit.add_mul(product, diff);
    }

    let neg_old_31 = circuit.add_mul(neg_one, old_state[STATE_SIZE - 1]);
    let neg_op_const = circuit.add_mul(neg_one, op_const);
    let acc_check = circuit.add_add(new_state[STATE_SIZE - 1], neg_old_31);
    let acc_check = circuit.add_add(acc_check, neg_op_const);
    product = circuit.add_mul(product, acc_check);

    let op_check = circuit.add_add(op, neg_op_const);
    let _final = circuit.add_mul(product, op_check);

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

fn record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let eval_points = setup_msg.eval_points.clone();
    let topology_vectors = verifier.topology_vectors().to_vec();

    let commitment = prover.commit(&setup_msg.eval_points).unwrap();
    let soldering_commit = prover.commit_soldering().unwrap();

    let chi = verifier.receive_commitment(commitment).unwrap();

    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        if let Some(ref rev) = reveal {
            verifier.receive_soldering_reveal_aggregated(rev).unwrap();
        }
    }

    let rho = verifier.receive_disclosure(disclosure, &mut rng).unwrap();

    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    verifier.receive_open(open_msg).unwrap();

    let gamma = verifier.generate_lpzk_challenge(&mut rng);
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
    let result = verifier.verify_multiplications_aggregated(lpzk_proof, gamma).unwrap();
    assert!(result, "Protocol verification failed during recording");

    RecordedVerifierMessages {
        eval_points,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        gamma,
    }
}

fn run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
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
    let _lpzk_proof = prover.prove_multiplications_aggregated(recorded.gamma).unwrap();
}

/// Runs prover with detailed timing for each phase.
fn run_prover_with_timing<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
) {
    let mut rng = Prg::from_seed(Block::ZERO);

    let t0 = Instant::now();
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    let t_new = t0.elapsed();

    let t1 = Instant::now();
    prover.setup(circuits, inputs_per_rep).unwrap();
    let t_setup = t1.elapsed();

    let t2 = Instant::now();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();
    let t_setup_solder = t2.elapsed();

    let t3 = Instant::now();
    let _commitment = prover.commit(&recorded.eval_points).unwrap();
    let t_commit = t3.elapsed();

    let t4 = Instant::now();
    let _soldering_commit = prover.commit_soldering().unwrap();
    let t_solder_commit = t4.elapsed();

    let t5 = Instant::now();
    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();
    let t_disclose = t5.elapsed();

    let t6 = Instant::now();
    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
    }
    let t_solder_reveal = t6.elapsed();

    let t7 = Instant::now();
    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();
    let t_open = t7.elapsed();

    let t8 = Instant::now();
    let _lpzk_proof = prover.prove_multiplications_aggregated(recorded.gamma).unwrap();
    let t_lpzk = t8.elapsed();

    let total = t0.elapsed();

    eprintln!("\nPhase timings:");
    eprintln!("  new():                {:>8.2} ms ({:>5.1}%)", t_new.as_secs_f64() * 1000.0, t_new.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  setup():              {:>8.2} ms ({:>5.1}%)", t_setup.as_secs_f64() * 1000.0, t_setup.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  setup_soldering():    {:>8.2} ms ({:>5.1}%)", t_setup_solder.as_secs_f64() * 1000.0, t_setup_solder.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  commit():             {:>8.2} ms ({:>5.1}%)", t_commit.as_secs_f64() * 1000.0, t_commit.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  commit_soldering():   {:>8.2} ms ({:>5.1}%)", t_solder_commit.as_secs_f64() * 1000.0, t_solder_commit.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  disclose():           {:>8.2} ms ({:>5.1}%)", t_disclose.as_secs_f64() * 1000.0, t_disclose.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  reveal_soldering():   {:>8.2} ms ({:>5.1}%)", t_solder_reveal.as_secs_f64() * 1000.0, t_solder_reveal.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  open():               {:>8.2} ms ({:>5.1}%)", t_open.as_secs_f64() * 1000.0, t_open.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  prove_mults():        {:>8.2} ms ({:>5.1}%)", t_lpzk.as_secs_f64() * 1000.0, t_lpzk.as_secs_f64() / total.as_secs_f64() * 100.0);
    eprintln!("  TOTAL:                {:>8.2} ms", total.as_secs_f64() * 1000.0);
}

macro_rules! impl_profile {
    ($r:expr, $circuits:expr, $active_branches:expr, $inputs:expr, $soldering:expr, $iters:expr, $final_acc:expr) => {{
        const R: usize = $r;

        eprintln!("Recording verifier messages for R={}...", R);
        let recorded = record_verifier_messages::<R>(
            &$circuits,
            &$active_branches,
            &$inputs,
            &$soldering,
        );
        eprintln!("Recording done.");

        // Run once with detailed timing
        run_prover_with_timing::<R>(
            &$circuits,
            &$active_branches,
            &$inputs,
            &$soldering,
            &recorded,
        );

        eprintln!("\nStarting {} iterations of prover replay...", $iters);
        let start = Instant::now();

        for i in 0..$iters {
            run_prover_with_replay::<R>(
                &$circuits,
                &$active_branches,
                &$inputs,
                &$soldering,
                &recorded,
            );
            if (i + 1) % 10 == 0 {
                eprintln!("  {} iterations done", i + 1);
            }
        }

        let elapsed = start.elapsed();
        eprintln!("\n{} iterations completed in {:.2} s ({:.2} ms/iter)",
                  $iters, elapsed.as_secs_f64(), elapsed.as_secs_f64() * 1000.0 / $iters as f64);
        eprintln!("Final accumulator: {}", $final_acc);
    }};
}

fn main() {
    let reps: usize = env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let iters: usize = env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    eprintln!("VM Profile Configuration:");
    eprintln!("  Branches: {}", NUM_BRANCHES);
    eprintln!("  State size: {}", STATE_SIZE);
    eprintln!("  Inputs per rep: {}", NUM_INPUTS);
    eprintln!("  REPS={} ITERS={}", reps, iters);

    let circuits = create_vm_circuit_batch();

    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    eprintln!("  Mults per circuit: {}", sample_circuit.num_mults());
    eprintln!("  Total mults: {} × {} × {} = {}",
              reps, NUM_BRANCHES, sample_circuit.num_mults(),
              reps * NUM_BRANCHES * sample_circuit.num_mults());

    let soldering = create_soldering_constraints(state_offset);

    match reps {
        100 => {
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(100);
            impl_profile!(100, circuits, branches, inputs, soldering, iters, final_acc);
        }
        1000 => {
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(1000);
            impl_profile!(1000, circuits, branches, inputs, soldering, iters, final_acc);
        }
        10000 => {
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(10000);
            impl_profile!(10000, circuits, branches, inputs, soldering, iters, final_acc);
        }
        25000 => {
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(25000);
            impl_profile!(25000, circuits, branches, inputs, soldering, iters, final_acc);
        }
        50000 => {
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(50000);
            impl_profile!(50000, circuits, branches, inputs, soldering, iters, final_acc);
        }
        _ => {
            eprintln!("Unsupported REPS value: {}. Supported: 100, 1000, 10000, 25000, 50000", reps);
            std::process::exit(1);
        }
    }

    eprintln!("Done!");
}
