//! VM-style benchmark for Justvengers prover with per-rep active branches.
//!
//! Simulates a simple VM with:
//! - 30 opcodes (branches)
//! - 32-element state vectors
//! - 97 multiplications per circuit
//! - State soldering across repetitions
//!
//! Run with: cargo bench -p mpz-justvengers --bench vm_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    // JustVengers O(R+B+C) optimized prover with IT-PAC
    JVProver, JVVerifier, JVSetupMessage, GoldilocksItMac,
    extract_verifier_shares_from_pool,
};
use mpz_justvengers_core::{VolePool, GlobalKey};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::{Rng, SeedableRng};

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 30;
const STATE_SIZE: usize = 32;
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1; // old_state + new_state + op

// Message size computation helpers
mod msg_size {
    use mpz_justvengers::{
        soldering::SolderingCommitMessage,
        AggregatedSolderingReveal, AggregatedLpzkProofMessage,
        // JV optimized message types
        JVSetupMessage, JVCommitmentMessage, JVDisclosureMessage, JVOpenMessage,
    };

    pub fn soldering_commit_message(msg: &SolderingCommitMessage) -> usize {
        8 // num_constraints: usize
        + msg.commitment_hashes.len() * 16 // Vec<(u64, u64)>
    }

    pub fn soldering_challenge_message() -> usize {
        16 // phi: u64 + psi: u64
    }

    pub fn aggregated_soldering_reveal(msg: &AggregatedSolderingReveal) -> usize {
        (msg.aggregated_f1.len() + msg.aggregated_f2.len()) * 8
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
        + msg.ciphertext_commitments.len() * 64 // Vec<Ciphertext> - estimate
    }

    /// JV Disclosure: O(R) instead of O(RC)!
    pub fn jv_disclosure_message(msg: &JVDisclosureMessage) -> usize {
        msg.topology_products.len() * 8 // O(R) - topology products
        + 8 // aggregated_poly_eval: u64
    }

    pub fn jv_open_message(msg: &JVOpenMessage) -> usize {
        msg.active_branches.len() // Vec<u8> - 1 byte each
        + 8 // hash_proof: u64
    }

    pub fn aggregated_lpzk_proof_message(msg: &AggregatedLpzkProofMessage) -> usize {
        msg.quotient_coeffs.len() * 8  // O(R) coefficients
        + 8  // aggregated_check: u64
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

// ============================================================================
// JV Optimized Protocol - O(R+B+C) Communication
// ============================================================================

/// Recorded verifier messages for JV protocol replay.
#[derive(Clone)]
struct JVRecordedMessages {
    /// Full setup message including encrypted powers for IT-PAC
    setup_msg: JVSetupMessage,
    /// Global key for VOLE generation
    global_key: GlobalKey<GoldilocksItMac>,
    /// Circuit size for VOLE pool generation
    circuit_size: usize,
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

    // Use JVProver with IT-PAC
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    // V → P: SetupMessage (includes encrypted powers for IT-PAC)
    stats.prover_received += msg_size::jv_setup_message(&setup_msg);

    let topology_vectors = verifier.topology_vectors().to_vec();

    // Create VOLE pool for IT-PAC commitments
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    // Extract verifier shares before passing pool to prover
    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
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
        if challenge.is_some() {
            stats.prover_received += msg_size::soldering_challenge_message();
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

    // P → V: IT-PAC Opening (polynomials + MAC tags)
    let itpac_open_msg = prover.open_itpac().unwrap();
    assert!(verifier.verify_itpac_opening(&itpac_open_msg), "IT-PAC verification failed");

    // P → V: AggregatedLpzkProofMessage - O(R) instead of O(M×R)!
    let gamma = verifier.generate_lpzk_challenge(&mut rng);
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
    let lpzk_size = msg_size::aggregated_lpzk_proof_message(&lpzk_proof);
    stats.prover_sent += lpzk_size;
    stats.lpzk = lpzk_size;

    let result = verifier.verify_multiplications_aggregated(lpzk_proof, gamma).unwrap();
    assert!(result, "JV Protocol verification failed during recording");

    JVRecordedMessages {
        setup_msg,
        global_key: verifier.global_key().clone(),
        circuit_size,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        stats,
    }
}

/// Pre-setup prover for efficient cloning during benchmark.
struct PreSetupProver<const R: usize> {
    prover: JVProver<R>,
    gamma: u64,
}

impl<const R: usize> PreSetupProver<R> {
    fn new(
        circuits: &CircuitBatch,
        active_branches: &[usize],
        inputs_per_rep: &[Vec<u64>],
        soldering_constraints: &[SolderingConstraint],
    ) -> Self {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
        prover.setup(circuits, inputs_per_rep).unwrap();
        prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();
        // Use deterministic gamma
        let gamma = rng.random_range(1..MODULUS);
        Self { prover, gamma }
    }

    fn run_iteration(&self, recorded: &JVRecordedMessages) {
        let mut rng = Prg::from_seed(Block::ZERO);
        let mut prover = self.prover.clone();

        // Create VOLE pool for IT-PAC commitments
        let vole_pool = VolePool::generate(&recorded.global_key, recorded.circuit_size * 2, &mut rng);

        // P → V: CommitmentMessage (IT-PAC ciphertexts)
        let _commitment = prover.commit(&recorded.setup_msg, vole_pool).unwrap();
        let _soldering_commit = prover.commit_soldering().unwrap();

        let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

        if let Some(ref challenge) = recorded.soldering_challenge {
            let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
        }

        let _open_msg = prover.open(recorded.rho, &recorded.topology_vectors).unwrap();

        // IT-PAC opening
        let _itpac_open_msg = prover.open_itpac().unwrap();

        let _lpzk_proof = prover.prove_multiplications_aggregated(self.gamma).unwrap();
    }
}

/// Runs JV prover with replay (legacy function for compatibility).
fn jv_run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &JVRecordedMessages,
) {
    let pre_setup = PreSetupProver::<R>::new(circuits, active_branches, inputs_per_rep, soldering_constraints);
    pre_setup.run_iteration(recorded);
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark VM prover - JustVengers O(R+B+C) protocol.
///
/// Uses aggregated LPZK proof which doesn't require VOLE correlations.
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

        // Pre-setup prover once (setup is not part of benchmark)
        let pre_setup = PreSetupProver::<R>::new(&circuits, &branches, &inputs, &soldering);

        // Run once to get stats
        pre_setup.run_iteration(&recorded);
        println!("\n[JV 100 reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 100 reps] VOLEs consumed: 0, OTs consumed: 0");

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "100_reps"), |b| {
            b.iter(|| {
                pre_setup.run_iteration(&recorded);
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

        // Pre-setup prover once (setup is not part of benchmark)
        let pre_setup = PreSetupProver::<R>::new(&circuits, &branches, &inputs, &soldering);

        // Run once to get stats
        pre_setup.run_iteration(&recorded);
        println!("\n[JV 1K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 1K reps] VOLEs consumed: 0, OTs consumed: 0");

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "1K_reps"), |b| {
            b.iter(|| {
                pre_setup.run_iteration(&recorded);
                black_box(())
            });
        });
    }

    // 10000 reps - JV protocol
    {
        const R: usize = 10000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Pre-setup prover once (setup is not part of benchmark)
        let pre_setup = PreSetupProver::<R>::new(&circuits, &branches, &inputs, &soldering);

        pre_setup.run_iteration(&recorded);
        println!("\n[JV 10K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 10K reps] VOLEs consumed: 0, OTs consumed: 0");

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "10K_reps"), |b| {
            b.iter(|| {
                pre_setup.run_iteration(&recorded);
                black_box(())
            });
        });
    }

    // 25000 reps - JV protocol
    {
        const R: usize = 25000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Pre-setup prover once (setup is not part of benchmark)
        let pre_setup = PreSetupProver::<R>::new(&circuits, &branches, &inputs, &soldering);

        pre_setup.run_iteration(&recorded);
        println!("\n[JV 25K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 25K reps] VOLEs consumed: 0, OTs consumed: 0");

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "25K_reps"), |b| {
            b.iter(|| {
                pre_setup.run_iteration(&recorded);
                black_box(())
            });
        });
    }

    // 50000 reps - JV protocol
    {
        const R: usize = 50000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);

        let recorded = jv_record_verifier_messages::<R>(
            &circuits,
            &branches,
            &inputs,
            &soldering,
        );

        // Pre-setup prover once (setup is not part of benchmark)
        let pre_setup = PreSetupProver::<R>::new(&circuits, &branches, &inputs, &soldering);

        pre_setup.run_iteration(&recorded);
        println!("\n[JV 50K reps] Communication: {}", recorded.stats.format_kb());
        println!("{}", recorded.stats.format_breakdown());
        println!("[JV 50K reps] VOLEs consumed: 0, OTs consumed: 0");

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function(BenchmarkId::new("jv_with_vole", "50K_reps"), |b| {
            b.iter(|| {
                pre_setup.run_iteration(&recorded);
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
    use mpz_common::context::replay_st_context;
    use mpz_justvengers::protocol::run_prover;

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

criterion_group!(benches, bench_vm_prover_with_vole, bench_vm_recorded_communication);
criterion_main!(benches);
