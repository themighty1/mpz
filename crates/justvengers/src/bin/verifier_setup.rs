//! Verifier setup binary - generates and saves messages for prover benchmarking.
//!
//! Usage:
//!   REPS=3000 cargo run --release --bin verifier_setup
//!
//! Outputs: verifier_msgs_REPS.bin

use std::env;
use std::fs::File;
use std::io::BufWriter;

use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    JVProver, JVVerifier, JVSetupMessage, GoldilocksItMac,
    extract_verifier_shares_from_pool,
};

use mpz_core::{prg::Prg, Block};
use mpz_fields::goldilocks::GOLDILOCKS;
use mpz_justvengers_core::{GlobalKey, VolePool};
use rand::{Rng, SeedableRng};
use serde::{Serialize, Deserialize};

const MODULUS: u64 = GOLDILOCKS;
const NUM_BRANCHES: usize = 30;
const STATE_SIZE: usize = 32;
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

#[derive(Serialize, Deserialize)]
struct VerifierMessages {
    setup_msg: JVSetupMessage,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
    gamma: u64,
    global_key: GlobalKey<GoldilocksItMac>,
    circuit_size: usize,
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

        let mut rep_inputs = Vec::with_capacity(NUM_INPUTS);
        rep_inputs.extend_from_slice(&old_state);
        rep_inputs.extend_from_slice(&new_state);
        rep_inputs.push(active_op as u64);

        inputs.push(rep_inputs);
        old_state = new_state;
    }

    (inputs, active_branches)
}

fn generate_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> VerifierMessages {
    let mut rng = Prg::from_seed(Block::ZERO);

    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let topology_vectors = verifier.topology_vectors().to_vec();
    let global_key = verifier.global_key().clone();
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);

    let vole_pool = VolePool::generate(&global_key, circuit_size * 2, &mut rng);

    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
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
    assert!(result, "Protocol verification failed");

    VerifierMessages {
        setup_msg,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        gamma,
        global_key,
        circuit_size,
    }
}

macro_rules! impl_generate {
    ($r:expr, $circuits:expr, $branches:expr, $inputs:expr, $soldering:expr) => {{
        generate_verifier_messages::<$r>($circuits, $branches, $inputs, $soldering)
    }};
}

fn main() {
    let reps: usize = env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    eprintln!("Generating verifier messages for REPS={}...", reps);

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);

    let (inputs, branches) = generate_vm_inputs_per_rep(reps);

    let msgs = match reps {
        100 => impl_generate!(100, &circuits, &branches, &inputs, &soldering),
        1000 => impl_generate!(1000, &circuits, &branches, &inputs, &soldering),
        3000 => impl_generate!(3000, &circuits, &branches, &inputs, &soldering),
        10000 => impl_generate!(10000, &circuits, &branches, &inputs, &soldering),
        _ => {
            eprintln!("Unsupported REPS: {}. Use 100, 1000, 3000, or 10000", reps);
            std::process::exit(1);
        }
    };

    let filename = format!("verifier_msgs_{}.bin", reps);
    let file = File::create(&filename).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &msgs).expect("Failed to serialize");

    eprintln!("Saved to {}", filename);
}
