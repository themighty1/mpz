//! Isolated prover benchmarks for Justvengers.
//!
//! Uses record/replay pattern to benchmark the prover in isolation:
//! 1. Record phase: Run full protocol, capture V→P messages
//! 2. Replay phase: Run just the prover with recorded V messages
//!
//! Run with: cargo bench -p mpz-justvengers --bench prover
//!
//! Configure via environment variables:
//!   BRANCHES=10 MULTS=100 cargo bench -p mpz-justvengers --bench prover

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::env;

use mpz_justvengers::{
    Circuit, CircuitBatch, ProverState, SolderingConstraint, VerifierState, VerifierMessage,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::SeedableRng;

/// Goldilocks prime: 2^64 - 2^32 + 1 (NTT-friendly)
const MODULUS: u64 = GOLDILOCKS;

/// Recorded verifier messages for prover replay.
#[derive(Clone)]
struct RecordedVerifierMessages {
    /// Evaluation points α₁, ..., αᵣ
    eval_points: Vec<u64>,
    /// Challenge χ for topology compression
    chi: u64,
    /// Topology vectors for all branches
    topology_vectors: Vec<TopologyVector>,
    /// Soldering challenge φ (if any)
    soldering_challenge: Option<SolderingChallengeMessage>,
    /// Challenge ρ for universal hash
    rho: u64,
}

/// Creates a circuit with the specified number of multiplication gates.
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

/// Creates a batch of circuits.
fn create_circuit_batch(num_branches: usize, num_mults: usize) -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..num_branches)
        .map(|_| create_circuit_with_mults(num_mults))
        .collect();
    CircuitBatch::new(circuits)
}

/// Generates chained inputs for soldering where each rep uses previous rep's output.
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

/// Records verifier messages by running the full protocol.
/// Returns the recorded messages needed for prover replay.
fn record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    seed: u64,
) -> RecordedVerifierMessages {
    let _ = seed; // Seed parameter for future use
    let mut rng = Prg::from_seed(Block::ZERO);

    // Get the circuit for the active branch
    let circuit = circuits.get(active_branch).unwrap();

    // Initialize prover
    let mut prover: ProverState<R> = ProverState::new(active_branch, MODULUS);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep).unwrap();

    // Setup soldering on prover
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    // Initialize verifier
    let mut verifier: VerifierState<R> = VerifierState::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();

    // Setup soldering on verifier
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    // Record eval_points
    let eval_points = setup_msg.eval_points.clone();

    // Record topology_vectors
    let topology_vectors = verifier.topology_vectors().to_vec();

    // Phase 1: Prover commits
    let commitment = prover.commit(&setup_msg.eval_points).unwrap();

    // Phase 1.5: Prover commits soldering
    let soldering_commit = prover.commit_soldering().unwrap();

    // Phase 2: Verifier sends challenge χ
    let chi_msg = verifier.receive_commitment(commitment).unwrap();
    let chi = match chi_msg {
        VerifierMessage::ChallengeChi(c) => c,
        _ => panic!("expected chi"),
    };

    // Phase 2.5: Verifier sends soldering challenge
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    // Phase 3: Prover discloses
    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    // Phase 3.5: Prover reveals soldering
    let _soldering_reveal = if let Some(ref challenge) = soldering_challenge {
        prover.reveal_soldering(challenge).unwrap()
    } else {
        None
    };

    // Phase 4: Verifier sends challenge ρ
    let rho_msg = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    let rho = match rho_msg {
        VerifierMessage::ChallengeRho(r) => r,
        _ => panic!("expected rho"),
    };

    // Phase 5: Prover opens
    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    verifier.receive_open(open_msg).unwrap();

    // Phase 6: Prover sends LPZK proof
    let lpzk_proof = prover.prove_multiplications().unwrap();

    // Final verification
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

/// Runs the prover only using recorded verifier messages.
fn run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
    seed: u64,
) {
    let _ = seed; // Seed parameter for future use
    let mut rng = Prg::from_seed(Block::ZERO);

    // Get the circuit for the active branch
    let circuit = circuits.get(active_branch).unwrap();

    // Initialize prover
    let mut prover: ProverState<R> = ProverState::new(active_branch, MODULUS);
    let mut circuit_clone = circuit.clone();
    prover.setup(&mut circuit_clone, inputs_per_rep).unwrap();

    // Setup soldering on prover
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    // Phase 1: Prover commits (uses recorded eval_points)
    let _commitment = prover.commit(&recorded.eval_points).unwrap();

    // Phase 1.5: Prover commits soldering
    let _soldering_commit = prover.commit_soldering().unwrap();

    // Phase 3: Prover discloses (uses recorded chi and topology_vectors)
    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    // Phase 3.5: Prover reveals soldering (uses recorded soldering_challenge)
    if let Some(ref challenge) = recorded.soldering_challenge {
        let _soldering_reveal = prover.reveal_soldering(challenge).unwrap();
    }

    // Phase 5: Prover opens (uses recorded rho and topology_vectors)
    let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();

    // Phase 6: Prover sends LPZK proof
    let _lpzk_proof = prover.prove_multiplications().unwrap();
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_isolated_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolated_prover");
    group.sample_size(10);

    // Configuration via env vars, defaults: 100 branches, 1000 mults per branch
    let num_branches: usize = env::var("BRANCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let num_mults: usize = env::var("MULTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let active_branch = num_branches / 2;

    let circuits = create_circuit_batch(num_branches, num_mults);
    let circuit = circuits.get(active_branch).unwrap();
    let soldering_constraint = SolderingConstraint::new(0, 0);

    // Test different repetition counts
    // 10 reps
    {
        const R: usize = 10;
        let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

        // Record verifier messages
        let recorded = record_verifier_messages::<R>(
            &circuits,
            active_branch,
            &chained_inputs,
            &[soldering_constraint.clone()],
            0,
        );

        let total_and_gates = (R * num_branches * num_mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        group.bench_function(BenchmarkId::new("with_soldering", "10_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                    0,
                );
                black_box(())
            });
        });
    }

    // 100 reps
    {
        const R: usize = 100;
        let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

        let recorded = record_verifier_messages::<R>(
            &circuits,
            active_branch,
            &chained_inputs,
            &[soldering_constraint.clone()],
            0,
        );

        let total_and_gates = (R * num_branches * num_mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        group.bench_function(BenchmarkId::new("with_soldering", "100_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                    0,
                );
                black_box(())
            });
        });
    }

    // 1000 reps
    {
        const R: usize = 1000;
        let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

        let recorded = record_verifier_messages::<R>(
            &circuits,
            active_branch,
            &chained_inputs,
            &[soldering_constraint.clone()],
            0,
        );

        let total_and_gates = (R * num_branches * num_mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        group.bench_function(BenchmarkId::new("with_soldering", "1000_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                    0,
                );
                black_box(())
            });
        });
    }

    // 10000 reps
    {
        const R: usize = 10000;
        let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

        let recorded = record_verifier_messages::<R>(
            &circuits,
            active_branch,
            &chained_inputs,
            &[soldering_constraint.clone()],
            0,
        );

        let total_and_gates = (R * num_branches * num_mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        group.bench_function(BenchmarkId::new("with_soldering", "10K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                    0,
                );
                black_box(())
            });
        });
    }

    // 100000 reps
    {
        const R: usize = 100000;
        let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

        let recorded = record_verifier_messages::<R>(
            &circuits,
            active_branch,
            &chained_inputs,
            &[soldering_constraint.clone()],
            0,
        );

        let total_and_gates = (R * num_branches * num_mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        group.bench_function(BenchmarkId::new("with_soldering", "100K_reps"), |b| {
            b.iter(|| {
                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                    0,
                );
                black_box(())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_isolated_prover);
criterion_main!(benches);
