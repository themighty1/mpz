//! VM-style JV prover benchmark for WASM with per-rep active branches.
//!
//! Simulates a VM with:
//! - 30 opcodes (branches)
//! - 32-element state vectors
//! - 97 multiplications per circuit
//! - State soldering across repetitions
//! - Random active branch per repetition

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers::{
    Circuit, CircuitBatch, ProverState, SolderingConstraint, VerifierState, VerifierMessage,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
};

#[cfg(target_arch = "wasm32")]
use mpz_core::{prg::Prg, Block};
#[cfg(target_arch = "wasm32")]
use mpz_fields::goldilocks::GOLDILOCKS;
#[cfg(target_arch = "wasm32")]
use rand::{Rng, SeedableRng};

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
const MODULUS: u64 = GOLDILOCKS;
#[cfg(target_arch = "wasm32")]
const NUM_BRANCHES: usize = 30;
#[cfg(target_arch = "wasm32")]
const STATE_SIZE: usize = 32;
#[cfg(target_arch = "wasm32")]
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
}

/// Creates a VM circuit for a specific opcode.
#[cfg(target_arch = "wasm32")]
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

    // Identity mults to expose new_state for soldering
    for j in 0..STATE_SIZE {
        circuit.add_mul(new_state[j], one);
    }

    circuit
}

#[cfg(target_arch = "wasm32")]
fn create_vm_circuit_batch() -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..NUM_BRANCHES as u64)
        .map(|op| create_vm_circuit(op))
        .collect();
    CircuitBatch::new(circuits)
}

#[cfg(target_arch = "wasm32")]
fn state_output_offset(circuit: &Circuit) -> usize {
    circuit.num_mults() - STATE_SIZE
}

#[cfg(target_arch = "wasm32")]
fn create_soldering_constraints(state_offset: usize) -> Vec<SolderingConstraint> {
    (0..STATE_SIZE)
        .map(|j| SolderingConstraint::new(j, state_offset + j))
        .collect()
}

/// Generates VM inputs with random ops per repetition.
#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
fn record_verifier_messages_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::new();

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

    if let Some(ref challenge) = soldering_challenge {
        let _ = prover.reveal_soldering(challenge).unwrap();
    }

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

#[cfg(target_arch = "wasm32")]
fn run_prover_with_replay_per_rep<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
) {
    let mut rng = Prg::new();

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

/// Benchmark VM-style JV prover with per-rep active branches.
///
/// Simulates a simple VM with 30 opcodes, 32-element state, and state soldering.
/// Each repetition uses a random active branch.
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `reps` - Number of repetitions (10, 100, 1000, 10000, or 100000)
///
/// # Returns
/// BenchResult with elapsed_ms and total multiplications
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn jv_vm_prover(n: u32, reps: u32) -> Result<BenchResult, JsValue> {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<Result<BenchResult, String>>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let _handle = web_spawn::spawn(move || {
        let bench_result = run_vm_bench(n, reps as usize);
        *result_clone.lock().unwrap() = Some(bench_result);
    });

    loop {
        JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
            .await
            .unwrap();
        if let Some(r) = result.lock().unwrap().take() {
            return r.map_err(|e| JsValue::from_str(&e));
        }
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
fn run_vm_bench(n: u32, reps: usize) -> Result<BenchResult, String> {
    let global = js_sys::global();
    let performance: web_sys::Performance =
        js_sys::Reflect::get(&global, &"performance".into())
            .map_err(|_| "performance not available")?
            .unchecked_into();

    let circuits = create_vm_circuit_batch();
    let sample_circuit = circuits.get(0).unwrap();
    let state_offset = state_output_offset(sample_circuit);
    let soldering = create_soldering_constraints(state_offset);
    let num_mults = sample_circuit.num_mults();

    macro_rules! run_bench {
        ($r:expr) => {{
            const R: usize = $r;
            let (inputs, branches, final_acc) = generate_vm_inputs_per_rep(R);

            // Record phase (not timed)
            let recorded = record_verifier_messages_per_rep::<R>(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            );

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, final_acc={}, starting {} iterations",
                    R, num_mults, final_acc, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                let start = performance.now();

                run_prover_with_replay_per_rep::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                );

                total_elapsed_ms += performance.now() - start;

                if (i + 1) % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[jv_vm] {} iterations done", i + 1).into(),
                    );
                }
            }

            // Total mults = iterations * reps * branches * mults_per_circuit
            let total_mults = n as u64 * R as u64 * NUM_BRANCHES as u64 * num_mults as u64;

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Done: {:.2}ms total, {:.2}ms/iter, {} total mults",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64,
                    total_mults
                )
                .into(),
            );

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults,
            })
        }};
    }

    match reps {
        10 => run_bench!(10),
        100 => run_bench!(100),
        1000 => run_bench!(1000),
        10000 => run_bench!(10000),
        25000 => run_bench!(25000),
        100000 => run_bench!(100000),
        _ => Err(format!(
            "Unsupported reps value: {}. Supported: 10, 100, 1000, 10000, 25000, 100000",
            reps
        )),
    }
}
