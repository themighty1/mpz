//! Profiling binary for the Justvengers prover.
//!
//! Runs only the prover hot path (replay) without recording overhead.
//!
//! Usage:
//!   BRANCHES=5 MULTS=5 REPS=100 ITERS=1000 cargo run --release --bin prover_profile
//!
//! Then profile with:
//!   samply record target/release/prover_profile
//!   # or
//!   perf record -g target/release/prover_profile && perf report

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

#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
}

fn create_circuit_with_mults(num_mults: usize) -> Circuit {
    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    let mut prev = circuit.add_mul(x, y);
    for _ in 1..num_mults {
        prev = circuit.add_mul(prev, x);
    }
    circuit
}

fn create_circuit_batch(num_branches: usize, num_mults: usize) -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..num_branches)
        .map(|_| create_circuit_with_mults(num_mults))
        .collect();
    CircuitBatch::new(circuits)
}

fn generate_chained_inputs(
    circuit: &Circuit,
    num_repetitions: usize,
    initial_value: u64,
    multiplier: u64,
    modulus: u64,
) -> Vec<Vec<u64>> {
    let mut inputs = Vec::with_capacity(num_repetitions);
    let mut prev_output = initial_value;

    for _ in 0..num_repetitions {
        let num_inputs = circuit.num_inputs();
        let mut rep_inputs = vec![prev_output];
        for _ in 1..num_inputs {
            rep_inputs.push(multiplier);
        }
        inputs.push(rep_inputs.clone());

        let mut circuit_clone = circuit.clone();
        let witness = circuit_clone.evaluate(&rep_inputs, modulus);
        if !witness.mult_outputs.is_empty() {
            prev_output = witness.mult_outputs[0];
        }
    }

    inputs
}

macro_rules! impl_profile {
    ($r:expr, $circuits:expr, $active_branch:expr, $chained_inputs:expr, $soldering_constraint:expr, $iters:expr) => {{
        const R: usize = $r;

        // Record phase (not profiled - happens once)
        eprintln!("Recording verifier messages for R={}...", R);
        let recorded = record_verifier_messages::<R>(
            &$circuits,
            $active_branch,
            &$chained_inputs,
            &[$soldering_constraint.clone()],
        );
        eprintln!("Recording done. Starting {} iterations of prover replay...", $iters);

        // Hot path - this is what we profile
        for i in 0..$iters {
            run_prover_with_replay::<R>(
                &$circuits,
                $active_branch,
                &$chained_inputs,
                &[$soldering_constraint.clone()],
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

fn main() {
    let num_branches: usize = env::var("BRANCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let num_mults: usize = env::var("MULTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let reps: usize = env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let iters: usize = env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    eprintln!("Configuration: BRANCHES={} MULTS={} REPS={} ITERS={}",
              num_branches, num_mults, reps, iters);

    let circuits = create_circuit_batch(num_branches, num_mults);
    let active_branch = num_branches / 2;
    let circuit = circuits.get(active_branch).unwrap();
    let soldering_constraint = SolderingConstraint::new(0, 0);

    // Use macro to handle const generic R at compile time
    // Support common rep counts
    match reps {
        10 => {
            let chained_inputs = generate_chained_inputs(circuit, 10, 3, 2, MODULUS);
            impl_profile!(10, circuits, active_branch, chained_inputs, soldering_constraint, iters);
        }
        100 => {
            let chained_inputs = generate_chained_inputs(circuit, 100, 3, 2, MODULUS);
            impl_profile!(100, circuits, active_branch, chained_inputs, soldering_constraint, iters);
        }
        1000 => {
            let chained_inputs = generate_chained_inputs(circuit, 1000, 3, 2, MODULUS);
            impl_profile!(1000, circuits, active_branch, chained_inputs, soldering_constraint, iters);
        }
        10000 => {
            let chained_inputs = generate_chained_inputs(circuit, 10000, 3, 2, MODULUS);
            impl_profile!(10000, circuits, active_branch, chained_inputs, soldering_constraint, iters);
        }
        100000 => {
            let chained_inputs = generate_chained_inputs(circuit, 100000, 3, 2, MODULUS);
            impl_profile!(100000, circuits, active_branch, chained_inputs, soldering_constraint, iters);
        }
        _ => {
            eprintln!("Unsupported REPS value: {}. Supported: 10, 100, 1000, 10000, 100000", reps);
            std::process::exit(1);
        }
    }

    eprintln!("Done!");
}
