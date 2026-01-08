//! Isolated JV (Justvengers) prover benchmark for WASM.
//!
//! Records verifier messages once, then benchmarks prover execution
//! in isolation using replay. This allows measuring pure prover
//! computation without protocol overhead.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    // JustVengers O(R+B+C) optimized prover with IT-PAC
    JVProver, JVVerifier, JVSetupMessage, GoldilocksItMac,
    extract_verifier_shares_from_pool,
    // Re-exported from justvengers-core
    VolePool, GlobalKey,
};

#[cfg(target_arch = "wasm32")]
use mpz_core::prg::Prg;
#[cfg(target_arch = "wasm32")]
use mpz_fields::goldilocks::GOLDILOCKS;
#[cfg(target_arch = "wasm32")]
use rand::Rng;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
const MODULUS: u64 = GOLDILOCKS;

/// Recorded verifier messages for prover replay with IT-PAC.
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct RecordedVerifierMessages {
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

/// Records verifier messages by running full protocol once with IT-PAC.
#[cfg(target_arch = "wasm32")]
fn record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> RecordedVerifierMessages {
    let mut rng = Prg::new();

    // Use JVProver with IT-PAC
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let topology_vectors = verifier.topology_vectors().to_vec();

    // Create VOLE pool for IT-PAC commitments
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    // Extract verifier shares before passing pool to prover
    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
    let soldering_commit = prover.commit_soldering().unwrap();

    // V → P: ChallengeChi
    let chi = verifier.receive_commitment(commitment).unwrap();

    // V → P: SolderingChallengeMessage (optional)
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    // P → V: DisclosureMessage
    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    // P → V: AggregatedSolderingReveal (optional)
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        if let Some(ref rev) = reveal {
            verifier.receive_soldering_reveal_aggregated(rev).unwrap();
        }
    }

    // V → P: ChallengeRho
    let rho = verifier.receive_disclosure(disclosure, &mut rng).unwrap();

    // P → V: OpenMessage
    let open_msg = prover.open(rho, verifier.topology_vectors()).unwrap();
    verifier.receive_open(open_msg).unwrap();

    // P → V: IT-PAC Opening
    let itpac_open_msg = prover.open_itpac().unwrap();
    assert!(verifier.verify_itpac_opening(&itpac_open_msg), "IT-PAC verification failed");

    // P → V: AggregatedLpzkProofMessage
    let gamma = verifier.generate_lpzk_challenge(&mut rng);
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
    let result = verifier.verify_multiplications_aggregated(lpzk_proof, gamma).unwrap();
    assert!(result, "JV Protocol verification failed during recording");

    RecordedVerifierMessages {
        setup_msg,
        global_key: verifier.global_key().clone(),
        circuit_size,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
    }
}

/// Pre-setup prover for efficient cloning during benchmark.
#[cfg(target_arch = "wasm32")]
struct PreSetupProver<const R: usize> {
    prover: JVProver<R>,
    gamma: u64,
}

#[cfg(target_arch = "wasm32")]
impl<const R: usize> PreSetupProver<R> {
    fn new(
        circuits: &CircuitBatch,
        active_branches: &[usize],
        inputs_per_rep: &[Vec<u64>],
        soldering_constraints: &[SolderingConstraint],
    ) -> Self {
        let mut rng = Prg::new();
        let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
        prover.setup(circuits, inputs_per_rep).unwrap();
        prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();
        // Use deterministic gamma
        let gamma = rng.random_range(1..MODULUS);
        Self { prover, gamma }
    }

    fn run_iteration(&self, recorded: &RecordedVerifierMessages) {
        let mut rng = Prg::new();
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
            // JVProver needs active_branches for each repetition
            let active_branches: Vec<usize> = vec![active_branch; R];

            // Record verifier messages (not timed)
            let recorded = record_verifier_messages::<R>(
                &circuits,
                &active_branches,
                &chained_inputs,
                &[soldering_constraint.clone()],
            );

            // Pre-setup prover once (setup is not part of benchmark)
            let pre_setup = PreSetupProver::<R>::new(
                &circuits,
                &active_branches,
                &chained_inputs,
                &[soldering_constraint.clone()],
            );

            let mut total_elapsed_ms = 0.0;

            // Benchmark iterations
            for _ in 0..n {
                let start = performance.now();

                // Clone pre-setup prover and run iteration
                pre_setup.run_iteration(&recorded);

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
