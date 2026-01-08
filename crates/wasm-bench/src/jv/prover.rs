//! Isolated JV (Justvengers) prover benchmark for WASM.
//!
//! Records verifier messages once, then benchmarks prover execution
//! in isolation using replay. This allows measuring pure prover
//! computation without protocol overhead.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers::{
    Circuit, CircuitBatch, ProverState, SolderingConstraint, VerifierState, VerifierMessage,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
};

#[cfg(target_arch = "wasm32")]
use mpz_core::prg::Prg;
#[cfg(target_arch = "wasm32")]
use mpz_fields::goldilocks::GOLDILOCKS;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
const MODULUS: u64 = GOLDILOCKS;

/// Recorded verifier messages for prover replay.
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct RecordedVerifierMessages {
    eval_points: Vec<u64>,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
}

/// Creates a circuit with a chain of multiplications.
#[cfg(target_arch = "wasm32")]
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

/// Creates a batch of identical circuits.
#[cfg(target_arch = "wasm32")]
fn create_circuit_batch(num_branches: usize, num_mults: usize) -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..num_branches)
        .map(|_| create_circuit_with_mults(num_mults))
        .collect();
    CircuitBatch::new(circuits)
}

/// Generates chained inputs where each rep uses previous output.
#[cfg(target_arch = "wasm32")]
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

/// Records verifier messages by running full protocol once.
#[cfg(target_arch = "wasm32")]
fn record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::new();

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

/// Runs prover with pre-recorded verifier messages.
#[cfg(target_arch = "wasm32")]
fn run_prover_with_replay<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &RecordedVerifierMessages,
) {
    let mut rng = Prg::new();

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

/// Benchmark JV prover with message replay.
///
/// Records verifier messages once during setup, then benchmarks
/// prover execution in isolation using replay.
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `branches` - Number of circuit branches (disjunctive statements)
/// * `mults` - Number of multiplication gates per circuit
/// * `reps` - Number of repetitions (must be 10, 100, or 1000)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn jv_prover(n: u32, branches: u32, mults: u32, reps: u32) -> Result<BenchResult, JsValue> {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<Result<BenchResult, String>>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    // Run benchmark on web worker thread (where blocking is allowed)
    let _handle = web_spawn::spawn(move || {
        let bench_result = run_jv_bench(n, branches as usize, mults as usize, reps as usize);
        *result_clone.lock().unwrap() = Some(bench_result);
    });

    // Poll for result on main thread
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

/// Inner benchmark function that runs on worker thread.
#[cfg(target_arch = "wasm32")]
fn run_jv_bench(n: u32, branches: usize, mults: usize, reps: usize) -> Result<BenchResult, String> {
    let global = js_sys::global();
    let performance: web_sys::Performance =
        js_sys::Reflect::get(&global, &"performance".into())
            .map_err(|_| "performance not available")?
            .unchecked_into();

    let circuits = create_circuit_batch(branches, mults);
    let active_branch = branches / 2;
    let circuit = circuits.get(active_branch).unwrap();
    let soldering_constraint = SolderingConstraint::new(0, 0);

    // Handle const generic R at compile time via macro
    macro_rules! run_bench {
        ($r:expr) => {{
            const R: usize = $r;
            let chained_inputs = generate_chained_inputs(circuit, R, 3, 2, MODULUS);

            // Record phase (not timed)
            let recorded = record_verifier_messages::<R>(
                &circuits,
                active_branch,
                &chained_inputs,
                &[soldering_constraint.clone()],
            );

            let mut total_elapsed_ms = 0.0;

            // Benchmark iterations
            for _ in 0..n {
                let start = performance.now();

                run_prover_with_replay::<R>(
                    &circuits,
                    active_branch,
                    &chained_inputs,
                    &[soldering_constraint.clone()],
                    &recorded,
                );

                total_elapsed_ms += performance.now() - start;
            }

            // Total multiplications = iterations * reps * mults_per_circuit
            let total_mults = n as u64 * R as u64 * mults as u64;

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults, // Reusing and_gates field for mult count
            })
        }};
    }

    match reps {
        10 => run_bench!(10),
        100 => run_bench!(100),
        1000 => run_bench!(1000),
        10000 => run_bench!(10000),
        100000 => run_bench!(100000),
        _ => Err(format!(
            "Unsupported reps value: {}. Supported: 10, 100, 1000, 10000, 100000",
            reps
        )),
    }
}
