//! JV VM benchmark.
//!
//! Run with: cargo bench -p mpz-justvengers --bench vm_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::executor::block_on;
use serio::{SinkExt, stream::IoStreamExt};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    JVProver, JVVerifier, JVSetupMessage, ItMacFieldType,
    MKCommitmentMessage,
    extract_verifier_shares_from_pool,
};
use mpz_justvengers_core::{VolePool, GlobalKey, RnsKeyPair, RnsPublicKey, RnsSecretKey};
use mpz_common::context::{Context, recording_st_context_with_limit, replay_st_context};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::{Rng, SeedableRng};

/// Path to BGV fixture directory.
const FIXTURE_DIR: &str = "bgv_fixtures";

/// Lazily-loaded RNS keypair from fixture files.
static PRELOADED_RNS_KEYPAIR: LazyLock<Option<RnsKeyPair>> = LazyLock::new(|| {
    load_rns_keypair_from_fixture().ok()
});

/// Loads RNS keypair from fixture files.
fn load_rns_keypair_from_fixture() -> Result<RnsKeyPair, Box<dyn std::error::Error>> {
    let fixture_path = Path::new(FIXTURE_DIR);

    // Load secret key
    let mut sk_file = File::open(fixture_path.join("secret_key.bin"))?;
    let mut sk_bytes = Vec::new();
    sk_file.read_to_end(&mut sk_bytes)?;
    let sk: RnsSecretKey = bincode::deserialize(&sk_bytes)?;

    // Load public key
    let mut pk_file = File::open(fixture_path.join("public_key.bin"))?;
    let mut pk_bytes = Vec::new();
    pk_file.read_to_end(&mut pk_bytes)?;
    let pk: RnsPublicKey = bincode::deserialize(&pk_bytes)?;

    Ok(RnsKeyPair { sk, pk })
}

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 60;
const STATE_SIZE: usize = 16;
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

/// Max frame length for protocol messages.
fn max_frame_length(num_reps: usize) -> usize {
    // With n=4096 AHE ring dimension, each ciphertext is ~64KB
    // Encrypted powers has R ciphertexts → ~64KB per rep
    // Plus topology vectors and overhead
    num_reps * 80 * 1024 + 8 * 1024 * 1024
}

/// Creates a VM circuit for a specific opcode.
///
/// Inputs: old_state[16] + new_state[16] + op = 33 inputs
/// Mults: ~33 constraint checks + 16 identity mults = ~49 total
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
// JV Optimized Protocol - O(R+B+C) Communication with Context IO
// ============================================================================

/// Recorded verifier messages for JV protocol replay.
#[derive(Clone)]
struct JVRecordedMessages {
    global_key: GlobalKey<ItMacFieldType>,
    circuit_size: usize,
}

/// Runs the full JV protocol with prover and verifier using context IO.
/// Records verifier->prover messages (V→P communication).
async fn run_protocol_record_verifier<const R: usize>(
    ctx_p: &mut Context,
    ctx_v: &mut Context,
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> JVRecordedMessages {
    let mut rng = Prg::from_seed(Block::ZERO);

    // Setup prover
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    // Setup verifier with preloaded keypair if available
    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    if let Some(ref keypair) = *PRELOADED_RNS_KEYPAIR {
        verifier.set_preloaded_rns_keypair(keypair.clone());
    }
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let topology_vectors = verifier.topology_vectors().to_vec();
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // V → P: Setup message
    ctx_v.io_mut().send(setup_msg.clone()).await.unwrap();
    let setup_msg_recv: JVSetupMessage = ctx_p.io_mut().expect_next().await.unwrap();

    // P → V: Commitment
    let commitment = prover.commit(&setup_msg_recv, vole_pool).unwrap();
    ctx_p.io_mut().send(commitment.clone()).await.unwrap();
    let commitment_recv = ctx_v.io_mut().expect_next().await.unwrap();

    // P → V: MK polynomial commitment
    let mk_commitment = prover.commit_mk_polynomials().unwrap();
    ctx_p.io_mut().send(mk_commitment.clone()).await.unwrap();
    let _mk_commitment_recv: MKCommitmentMessage = ctx_v.io_mut().expect_next().await.unwrap();

    // P → V: Soldering commitment (optional)
    let soldering_commit = prover.commit_soldering().unwrap();
    ctx_p.io_mut().send(soldering_commit.clone()).await.unwrap();
    let soldering_commit_recv = ctx_v.io_mut().expect_next().await.unwrap();

    // V → P: Chi challenge
    let chi = verifier.receive_commitment(commitment_recv).unwrap();
    ctx_v.io_mut().send(chi).await.unwrap();
    let chi_recv: u64 = ctx_p.io_mut().expect_next().await.unwrap();

    // V → P: Topology vectors
    ctx_v.io_mut().send(topology_vectors.clone()).await.unwrap();
    let topology_vectors_recv: Vec<TopologyVector> = ctx_p.io_mut().expect_next().await.unwrap();

    // V → P: Soldering challenge (optional)
    let soldering_challenge = if let Some(commit) = soldering_commit_recv {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };
    ctx_v.io_mut().send(soldering_challenge.clone()).await.unwrap();
    let soldering_challenge_recv: Option<SolderingChallengeMessage> = ctx_p.io_mut().expect_next().await.unwrap();

    // P → V: Disclosure
    let disclosure = prover.disclose(chi_recv, &topology_vectors_recv).unwrap();
    ctx_p.io_mut().send(disclosure.clone()).await.unwrap();
    let disclosure_recv = ctx_v.io_mut().expect_next().await.unwrap();

    // P → V: Soldering reveal (optional)
    if let Some(ref challenge) = soldering_challenge_recv {
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        ctx_p.io_mut().send(reveal.clone()).await.unwrap();
        let reveal_recv = ctx_v.io_mut().expect_next().await.unwrap();
        if let Some(ref rev) = reveal_recv {
            verifier.receive_soldering_reveal_aggregated(rev).unwrap();
        }
    }

    // V → P: Rho challenge
    let rho = verifier.receive_disclosure(disclosure_recv, &mut rng).unwrap();
    ctx_v.io_mut().send(rho).await.unwrap();
    let rho_recv: u64 = ctx_p.io_mut().expect_next().await.unwrap();

    // V → P: Gamma challenge
    let gamma = verifier.generate_lpzk_challenge(&mut rng);
    ctx_v.io_mut().send(gamma).await.unwrap();
    let gamma_recv: u64 = ctx_p.io_mut().expect_next().await.unwrap();

    // P → V: Open message
    let open_msg = prover.open(rho_recv, gamma_recv, &topology_vectors_recv).unwrap();
    ctx_p.io_mut().send(open_msg.clone()).await.unwrap();
    let open_msg_recv = ctx_v.io_mut().expect_next().await.unwrap();
    verifier.receive_open(open_msg_recv, gamma).unwrap();

    // P → V: IT-PAC opening
    let itpac_open_msg = prover.open_itpac().unwrap();
    ctx_p.io_mut().send(itpac_open_msg.clone()).await.unwrap();
    let itpac_open_msg_recv = ctx_v.io_mut().expect_next().await.unwrap();
    assert!(verifier.verify_itpac_opening(&itpac_open_msg_recv), "IT-PAC verification failed");

    // P → V: LPZK proof
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma_recv).unwrap();
    ctx_p.io_mut().send(lpzk_proof.clone()).await.unwrap();
    let lpzk_proof_recv = ctx_v.io_mut().expect_next().await.unwrap();
    let result = verifier.verify_multiplications_aggregated(lpzk_proof_recv, gamma).unwrap();
    assert!(result, "JV Protocol verification failed during recording");

    JVRecordedMessages {
        global_key: verifier.global_key().clone(),
        circuit_size,
    }
}

/// Records verifier->prover messages for prover replay.
/// Returns (recorded_bytes, recorded_messages).
fn record_for_prover<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> (Vec<u8>, JVRecordedMessages) {
    block_on(async {
        // Recording buffer: ~150KB per rep + 16MB base (for n=4096 AHE)
        let buffer_size = R * 150 * 1024 + 16 * 1024 * 1024;
        let (mut ctx_p, mut ctx_v, recorded) =
            recording_st_context_with_limit(buffer_size, max_frame_length(R));
        let messages = run_protocol_record_verifier::<R>(
            &mut ctx_p, &mut ctx_v, circuits, active_branches, inputs_per_rep, soldering_constraints
        ).await;
        (recorded.lock().unwrap().clone(), messages)
    })
}

/// Runs prover only with replay context.
async fn run_prover_with_replay<const R: usize>(
    ctx: &mut Context,
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded_messages: &JVRecordedMessages,
) {
    let total_start = std::time::Instant::now();
    let mut rng = Prg::from_seed(Block::ZERO);

    // Fresh prover with fresh witness setup each iteration
    let new_start = std::time::Instant::now();
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    eprintln!("[bench] new: {:?}", new_start.elapsed());

    let setup_start = std::time::Instant::now();
    prover.setup(circuits, inputs_per_rep).unwrap();
    eprintln!("[bench] setup: {:?}", setup_start.elapsed());

    let solder_setup_start = std::time::Instant::now();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();
    eprintln!("[bench] setup_soldering: {:?}", solder_setup_start.elapsed());

    // Create VOLE pool for IT-PAC commitments
    let vole_start = std::time::Instant::now();
    let vole_pool = VolePool::generate(&recorded_messages.global_key, recorded_messages.circuit_size * 2, &mut rng);
    eprintln!("[bench] vole_pool: {:?}", vole_start.elapsed());

    // V → P: Setup message
    let setup_msg: JVSetupMessage = ctx.io_mut().expect_next().await.unwrap();

    // P → V: Commitment
    let commit_start = std::time::Instant::now();
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
    eprintln!("[bench] commit: {:?}", commit_start.elapsed());
    ctx.io_mut().send(commitment).await.unwrap();

    // P → V: MK polynomial commitment
    let mk_start = std::time::Instant::now();
    let mk_commitment = prover.commit_mk_polynomials().unwrap();
    eprintln!("[bench] commit_mk: {:?}", mk_start.elapsed());
    ctx.io_mut().send(mk_commitment).await.unwrap();

    // P → V: Soldering commitment
    let solder_start = std::time::Instant::now();
    let soldering_commit = prover.commit_soldering().unwrap();
    eprintln!("[bench] commit_soldering: {:?}", solder_start.elapsed());
    ctx.io_mut().send(soldering_commit).await.unwrap();

    // V → P: Chi challenge
    let chi: u64 = ctx.io_mut().expect_next().await.unwrap();

    // V → P: Topology vectors
    let topology_vectors: Vec<TopologyVector> = ctx.io_mut().expect_next().await.unwrap();

    // V → P: Soldering challenge
    let soldering_challenge: Option<SolderingChallengeMessage> = ctx.io_mut().expect_next().await.unwrap();

    // P → V: Disclosure
    let disclose_start = std::time::Instant::now();
    let disclosure = prover.disclose(chi, &topology_vectors).unwrap();
    eprintln!("[bench] disclose: {:?}", disclose_start.elapsed());
    ctx.io_mut().send(disclosure).await.unwrap();

    // P → V: Soldering reveal
    if let Some(ref challenge) = soldering_challenge {
        let reveal_start = std::time::Instant::now();
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        eprintln!("[bench] reveal_soldering: {:?}", reveal_start.elapsed());
        ctx.io_mut().send(reveal).await.unwrap();
    }

    // V → P: Rho challenge
    let rho: u64 = ctx.io_mut().expect_next().await.unwrap();

    // V → P: Gamma challenge
    let gamma: u64 = ctx.io_mut().expect_next().await.unwrap();

    // P → V: Open message
    let open_start = std::time::Instant::now();
    let open_msg = prover.open(rho, gamma, &topology_vectors).unwrap();
    eprintln!("[bench] open: {:?}", open_start.elapsed());
    ctx.io_mut().send(open_msg).await.unwrap();

    // P → V: IT-PAC opening
    let itpac_start = std::time::Instant::now();
    let itpac_open_msg = prover.open_itpac().unwrap();
    eprintln!("[bench] open_itpac: {:?}", itpac_start.elapsed());
    ctx.io_mut().send(itpac_open_msg).await.unwrap();

    // P → V: LPZK proof
    let lpzk_start = std::time::Instant::now();
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
    eprintln!("[bench] lpzk: {:?}", lpzk_start.elapsed());
    ctx.io_mut().send(lpzk_proof).await.unwrap();

    eprintln!("[bench] TOTAL: {:?}", total_start.elapsed());
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark JV VM prover.
fn bench_jv_vm(c: &mut Criterion) {
    let mut group = c.benchmark_group("jv_vm");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));
    group.warm_up_time(std::time::Duration::from_secs(2));

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);
    let num_mults = sample_circuit.num_mults();

    // 128 reps
    {
        const R: usize = 128;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("128_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 256 reps
    {
        const R: usize = 256;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("256_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 512 reps
    {
        const R: usize = 512;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("512_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 1K reps
    {
        const R: usize = 1000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("1K_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 2K reps
    {
        const R: usize = 2000;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("2K_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 4K reps
    {
        const R: usize = 4096;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("4K_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    // 8K reps (max for 8192 slots)
    {
        const R: usize = 8192;
        let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
        let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

        println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

        let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
        group.throughput(Throughput::Elements(total_mults));

        group.bench_function("8K_reps", |b| {
            b.iter(|| {
                block_on(async {
                    let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                    run_prover_with_replay::<R>(
                        &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                    ).await;
                });
                black_box(())
            });
        });
    }

    group.finish();
}

/// Benchmark 8K reps only (for quick testing).
fn bench_jv_vm_8k(c: &mut Criterion) {
    let mut group = c.benchmark_group("jv_vm");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));
    group.warm_up_time(std::time::Duration::from_secs(2));

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);
    let num_mults = sample_circuit.num_mults();

    const R: usize = 8192;
    let (inputs, branches, _acc) = generate_vm_inputs_per_rep(R);
    let (recorded_bytes, recorded_messages) = record_for_prover::<R>(&circuits, &branches, &inputs, &soldering);

    println!("[JV {} reps] Communication: total {:.1} KB", R, recorded_bytes.len() as f64 / 1024.0);

    let total_mults = (R * NUM_BRANCHES * num_mults) as u64;
    group.throughput(Throughput::Elements(total_mults));

    group.bench_function("8K_reps", |b| {
        b.iter(|| {
            block_on(async {
                let mut ctx = replay_st_context(recorded_bytes.clone(), max_frame_length(R));
                run_prover_with_replay::<R>(
                    &mut ctx, &circuits, &branches, &inputs, &soldering, &recorded_messages
                ).await;
            });
            black_box(())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_jv_vm);
criterion_group!(benches_8k, bench_jv_vm_8k);
criterion_main!(benches, benches_8k);
