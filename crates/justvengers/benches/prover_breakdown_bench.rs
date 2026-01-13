//! Breakdown benchmark to isolate where time is spent in prover flow.
//!
//! Run with: cargo bench -p mpz-justvengers --bench prover_breakdown_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use futures::executor::block_on;
use serio::stream::IoStreamExt;

use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    JVProver, JVVerifier, JVSetupMessage, ItMacFieldType,
};
use mpz_justvengers_core::{VolePool, GlobalKey, RnsKeyPair, RnsPublicKey, RnsSecretKey};
use mpz_common::context::replay_st_context;

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 60;
const STATE_SIZE: usize = 16;
const R: usize = 128;

/// Path to BGV fixture directory.
const FIXTURE_DIR: &str = "bgv_fixtures";

/// Lazily-loaded RNS keypair from fixture files.
static PRELOADED_RNS_KEYPAIR: LazyLock<Option<RnsKeyPair>> = LazyLock::new(|| {
    load_rns_keypair_from_fixture().ok()
});

fn load_rns_keypair_from_fixture() -> Result<RnsKeyPair, Box<dyn std::error::Error>> {
    let fixture_path = Path::new(FIXTURE_DIR);
    let mut sk_file = File::open(fixture_path.join("secret_key.bin"))?;
    let mut sk_bytes = Vec::new();
    sk_file.read_to_end(&mut sk_bytes)?;
    let sk: RnsSecretKey = bincode::deserialize(&sk_bytes)?;

    let mut pk_file = File::open(fixture_path.join("public_key.bin"))?;
    let mut pk_bytes = Vec::new();
    pk_file.read_to_end(&mut pk_bytes)?;
    let pk: RnsPublicKey = bincode::deserialize(&pk_bytes)?;

    Ok(RnsKeyPair { sk, pk })
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

fn generate_vm_inputs_per_rep(num_repetitions: usize) -> (Vec<Vec<u64>>, Vec<usize>) {
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

        let mut rep_inputs = Vec::with_capacity(STATE_SIZE * 2 + 1);
        rep_inputs.extend_from_slice(&old_state);
        rep_inputs.extend_from_slice(&new_state);
        rep_inputs.push(active_op as u64);

        inputs.push(rep_inputs);
        old_state = new_state;
    }

    (inputs, active_branches)
}

/// Recorded verifier messages for replay.
struct RecordedData {
    global_key: GlobalKey<ItMacFieldType>,
    circuit_size: usize,
    setup_msg: JVSetupMessage,
    topology_vectors: Vec<TopologyVector>,
}

fn max_frame_length() -> usize {
    R * 80 * 1024 + 8 * 1024 * 1024
}

/// Records protocol once and extracts needed data.
fn record_protocol_data(
    circuits: &CircuitBatch,
    branches: &[usize],
    inputs: &[Vec<u64>],
    soldering: &[SolderingConstraint],
) -> RecordedData {
    use futures::executor::block_on;
    use serio::SinkExt;
    use mpz_common::context::recording_st_context_with_limit;

    let mut rng = Prg::from_seed(Block::ZERO);

    // Setup prover
    let mut prover = JVProver::<R>::new(branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs).unwrap();
    prover.setup_soldering(soldering.to_vec(), &mut rng).unwrap();

    // Setup verifier with preloaded keypair
    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    if let Some(ref keypair) = *PRELOADED_RNS_KEYPAIR {
        verifier.set_preloaded_rns_keypair(keypair.clone());
    }
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering.to_vec()).unwrap();

    let topology_vectors = verifier.topology_vectors().to_vec();
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);

    RecordedData {
        global_key: verifier.global_key().clone(),
        circuit_size,
        setup_msg,
        topology_vectors,
    }
}

fn bench_prover_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("prover_breakdown");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(8));
    group.warm_up_time(std::time::Duration::from_millis(500));

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);

    let (inputs, branches) = generate_vm_inputs_per_rep(R);

    println!("Recording protocol data for {} reps...", R);
    let data = record_protocol_data(&circuits, &branches, &inputs, &soldering);
    println!("Recording done. Starting benchmarks.");

    // 1. Bench prover setup only
    group.bench_function("1_prover_setup", |b| {
        b.iter(|| {
            let mut rng = Prg::from_seed(Block::ZERO);
            let mut prover = JVProver::<R>::new(branches.clone(), MODULUS);
            prover.setup(black_box(&circuits), black_box(&inputs)).unwrap();
            prover.setup_soldering(soldering.clone(), &mut rng).unwrap();
            black_box(prover)
        });
    });

    // 2. Bench commit only (setup is overhead, but we measure the delta)
    group.bench_function("2_commit", |b| {
        b.iter(|| {
            let mut rng = Prg::from_seed(Block::ZERO);
            let mut prover = JVProver::<R>::new(branches.clone(), MODULUS);
            prover.setup(&circuits, &inputs).unwrap();
            prover.setup_soldering(soldering.clone(), &mut rng).unwrap();

            let vole_pool = VolePool::generate(&data.global_key, data.circuit_size * 2, &mut rng);
            let commitment = prover.commit(black_box(&data.setup_msg), vole_pool).unwrap();
            black_box(commitment)
        });
    });

    // 3. Bench through MK commit
    group.bench_function("3_through_mk_commit", |b| {
        b.iter(|| {
            let mut rng = Prg::from_seed(Block::ZERO);
            let mut prover = JVProver::<R>::new(branches.clone(), MODULUS);
            prover.setup(&circuits, &inputs).unwrap();
            prover.setup_soldering(soldering.clone(), &mut rng).unwrap();

            let vole_pool = VolePool::generate(&data.global_key, data.circuit_size * 2, &mut rng);
            let _ = prover.commit(&data.setup_msg, vole_pool).unwrap();
            let mk_commit = prover.commit_mk_polynomials().unwrap();
            black_box(mk_commit)
        });
    });

    // 4. Bench through disclose
    let chi: u64 = 12345;
    group.bench_function("4_through_disclose", |b| {
        b.iter(|| {
            let mut rng = Prg::from_seed(Block::ZERO);
            let mut prover = JVProver::<R>::new(branches.clone(), MODULUS);
            prover.setup(&circuits, &inputs).unwrap();
            prover.setup_soldering(soldering.clone(), &mut rng).unwrap();

            let vole_pool = VolePool::generate(&data.global_key, data.circuit_size * 2, &mut rng);
            let _ = prover.commit(&data.setup_msg, vole_pool).unwrap();
            let _ = prover.commit_mk_polynomials().unwrap();
            let _ = prover.commit_soldering().unwrap();
            let disclosure = prover.disclose(black_box(chi), black_box(&data.topology_vectors)).unwrap();
            black_box(disclosure)
        });
    });

    // 5. Bench full prover flow
    let rho: u64 = 67890;
    let gamma: u64 = 11111;
    group.bench_function("5_full_prover", |b| {
        b.iter(|| {
            let mut rng = Prg::from_seed(Block::ZERO);
            let mut prover = JVProver::<R>::new(branches.clone(), MODULUS);
            prover.setup(&circuits, &inputs).unwrap();
            prover.setup_soldering(soldering.clone(), &mut rng).unwrap();

            let vole_pool = VolePool::generate(&data.global_key, data.circuit_size * 2, &mut rng);
            let _ = prover.commit(&data.setup_msg, vole_pool).unwrap();
            let _ = prover.commit_mk_polynomials().unwrap();
            let _ = prover.commit_soldering().unwrap();
            let _ = prover.disclose(chi, &data.topology_vectors).unwrap();
            let _ = prover.open(rho, gamma, &data.topology_vectors).unwrap();
            let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
            black_box(lpzk_proof)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_prover_breakdown);
criterion_main!(benches);
