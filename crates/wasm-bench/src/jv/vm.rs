//! VM-style JV prover benchmark for WASM with per-rep active branches.
//!
//! Simulates a VM with:
//! - 60 opcodes (branches)
//! - 16-element state vectors
//! - ~49 multiplications per circuit
//! - State soldering across repetitions
//! - Random active branch per repetition
//!
//! Uses IT-PAC polynomial commitment and pre-setup prover pattern
//! for efficient benchmarking.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers::{
    Circuit, CircuitBatch, SolderingConstraint,
    topology::TopologyVector,
    soldering::SolderingChallengeMessage,
    // JustVengers O(R+B+C) optimized prover with IT-PAC
    JVProver, JVVerifier, JVSetupMessage, ItMacFieldType,
    extract_verifier_shares_from_pool,
    // Re-exported from justvengers-core
    VolePool, GlobalKey,
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
const NUM_BRANCHES: usize = 60;
#[cfg(target_arch = "wasm32")]
const STATE_SIZE: usize = 16;
#[cfg(target_arch = "wasm32")]
const NUM_INPUTS: usize = STATE_SIZE * 2 + 1;

/// Recorded verifier messages for JV protocol replay with IT-PAC.
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct JVRecordedMessages {
    /// Full setup message including encrypted powers for IT-PAC
    setup_msg: JVSetupMessage,
    /// Global key for VOLE generation
    global_key: GlobalKey<ItMacFieldType>,
    /// Circuit size for VOLE pool generation
    circuit_size: usize,
    chi: u64,
    topology_vectors: Vec<TopologyVector>,
    soldering_challenge: Option<SolderingChallengeMessage>,
    rho: u64,
    gamma: u64,
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

/// Records JV verifier messages for replay benchmarking with IT-PAC.
#[cfg(target_arch = "wasm32")]
async fn jv_record_verifier_messages<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> JVRecordedMessages {
    web_sys::console::log_1(&"[jv_record] Starting verifier message recording...".into());
    let mut rng = Prg::from_seed(Block::ZERO);

    // Use JVProver with IT-PAC
    web_sys::console::log_1(&"[jv_record] Creating prover...".into());
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    web_sys::console::log_1(&"[jv_record] Setting up prover...".into());
    prover.setup(circuits, inputs_per_rep).unwrap();
    web_sys::console::log_1(&"[jv_record] Setting up soldering...".into());
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    web_sys::console::log_1(&"[jv_record] Creating verifier...".into());
    let mut verifier = JVVerifier::<R>::new(MODULUS, &mut rng);
    web_sys::console::log_1(&"[jv_record] Verifier setup...".into());
    let setup_msg = verifier.setup(circuits, &mut rng).unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier setup soldering...".into());
    verifier.setup_soldering(soldering_constraints.to_vec()).unwrap();

    let topology_vectors = verifier.topology_vectors().to_vec();

    // Create VOLE pool for IT-PAC commitments
    web_sys::console::log_1(&"[jv_record] Generating VOLE pool...".into());
    let circuit_size = circuits.get(0).map(|c| c.num_wires()).unwrap_or(10);
    let vole_pool = VolePool::generate(verifier.global_key(), circuit_size * 2, &mut rng);

    // Extract verifier shares before passing pool to prover
    web_sys::console::log_1(&"[jv_record] Extracting verifier shares...".into());
    let verifier_shares = extract_verifier_shares_from_pool(&vole_pool, circuit_size * 2);
    verifier.set_verifier_local_keys(verifier_shares);

    // Pre-initialize GPU context (async in WASM) - MUST succeed (GPU is only path)
    web_sys::console::log_1(&"[jv_record] Preparing GPU context...".into());
    prover.prepare_gpu_async(&setup_msg).await.expect("[jv_vm] GPU init failed - GPU is the only supported path");
    web_sys::console::log_1(&"[jv_record] GPU context ready".into());

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    web_sys::console::log_1(&"[jv_record] Prover commit (IT-PAC)...".into());
    let commitment = prover.commit(&setup_msg, vole_pool).unwrap();
    web_sys::console::log_1(&"[jv_record] Prover commit done".into());

    // P → V: MK polynomial commitment
    web_sys::console::log_1(&"[jv_record] Prover commit MK polynomials...".into());
    let _mk_commitment = prover.commit_mk_polynomials().unwrap();
    web_sys::console::log_1(&"[jv_record] MK commit done".into());

    // P → V: Soldering commitment (optional)
    web_sys::console::log_1(&"[jv_record] Prover commit soldering...".into());
    let soldering_commit = prover.commit_soldering().unwrap();

    // V → P: ChallengeChi
    web_sys::console::log_1(&"[jv_record] Verifier receive commitment...".into());
    let chi = verifier.receive_commitment(commitment).unwrap();

    // V → P: SolderingChallengeMessage (optional)
    web_sys::console::log_1(&"[jv_record] Verifier receive soldering commit...".into());
    let soldering_challenge = if let Some(commit) = soldering_commit {
        verifier.receive_soldering_commit(commit, &mut rng).unwrap()
    } else {
        None
    };

    // P → V: DisclosureMessage
    web_sys::console::log_1(&"[jv_record] Prover disclose...".into());
    let disclosure = prover.disclose(chi, verifier.topology_vectors()).unwrap();

    // P → V: AggregatedSolderingReveal (optional)
    web_sys::console::log_1(&"[jv_record] Prover reveal soldering...".into());
    if let Some(ref challenge) = soldering_challenge {
        let reveal = prover.reveal_soldering_aggregated(challenge).unwrap();
        if let Some(ref rev) = reveal {
            verifier.receive_soldering_reveal_aggregated(rev).unwrap();
        }
    }

    // V → P: ChallengeRho
    web_sys::console::log_1(&"[jv_record] Verifier receive disclosure...".into());
    let rho = verifier.receive_disclosure(disclosure, &mut rng).unwrap();
    let gamma = verifier.generate_lpzk_challenge(&mut rng);

    // P → V: OpenMessage
    web_sys::console::log_1(&"[jv_record] Prover open...".into());
    let open_msg = prover.open(rho, gamma, verifier.topology_vectors()).unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier receive open...".into());
    verifier.receive_open(open_msg, gamma).unwrap();

    // P → V: IT-PAC Opening
    web_sys::console::log_1(&"[jv_record] Prover open IT-PAC...".into());
    let itpac_open_msg = prover.open_itpac().unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier verify IT-PAC...".into());
    assert!(verifier.verify_itpac_opening(&itpac_open_msg), "IT-PAC verification failed");

    // P → V: AggregatedLpzkProofMessage
    web_sys::console::log_1(&"[jv_record] Prover prove multiplications...".into());
    let lpzk_proof = prover.prove_multiplications_aggregated(gamma).unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier verify multiplications...".into());
    let result = verifier.verify_multiplications_aggregated(lpzk_proof, gamma).unwrap();
    assert!(result, "JV Protocol verification failed during recording");

    web_sys::console::log_1(&"[jv_record] Recording complete!".into());

    JVRecordedMessages {
        setup_msg,
        global_key: verifier.global_key().clone(),
        circuit_size,
        chi,
        topology_vectors,
        soldering_challenge,
        rho,
        gamma,
    }
}

/// Runs a single prover iteration with fresh witness setup.
#[cfg(target_arch = "wasm32")]
async fn run_prover_iteration<const R: usize>(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &JVRecordedMessages,
    gpu_ctx: &std::sync::Arc<bgv_webgpu::RnsSlotMulGpu>,
) {
    let mut rng = Prg::from_seed(Block::ZERO);

    // Fresh prover with fresh witness setup each iteration
    let mut prover = JVProver::<R>::new(active_branches.to_vec(), MODULUS);
    prover.setup(circuits, inputs_per_rep).unwrap();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    // Create VOLE pool for IT-PAC commitments
    let vole_pool = VolePool::generate(&recorded.global_key, recorded.circuit_size * 2, &mut rng);

    // Set pre-initialized GPU context (always present - GPU is only path)
    prover.set_gpu_context(gpu_ctx.clone());

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    let _commitment = prover.commit(&recorded.setup_msg, vole_pool).unwrap();

    // P → V: MK polynomial commitment
    let _mk_commitment = prover.commit_mk_polynomials().unwrap();

    // P → V: Soldering commitment (optional)
    let _soldering_commit = prover.commit_soldering().unwrap();

    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();

    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
    }

    let _open_msg = prover.open(recorded.rho, recorded.gamma, &recorded.topology_vectors).unwrap();

    // IT-PAC opening
    let _itpac_open_msg = prover.open_itpac().unwrap();

    let _lpzk_proof = prover.prove_multiplications_aggregated(recorded.gamma).unwrap();
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
/// NOTE: Disabled - uses web_spawn which we're avoiding
#[cfg(all(target_arch = "wasm32", feature = "web-spawn-mt"))]
#[wasm_bindgen]
pub async fn jv_vm_prover(n: u32, reps: u32) -> Result<BenchResult, JsValue> {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<Result<BenchResult, String>>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    // Run benchmark on web worker thread (where Atomics.wait is allowed)
    let _handle = web_spawn::spawn(move || {
        // Use spawn_local inside worker to run async GPU init
        wasm_bindgen_futures::spawn_local(async move {
            let bench_result = run_vm_bench_async(n, reps as usize).await;
            *result_clone.lock().unwrap() = Some(bench_result);
        });
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

#[cfg(all(target_arch = "wasm32", feature = "web-spawn-mt"))]
async fn run_vm_bench_async(n: u32, reps: usize) -> Result<BenchResult, String> {
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
            let (inputs, branches, _final_acc) = generate_vm_inputs_per_rep(R);

            // Record verifier messages (not timed) - async for GPU init
            let recorded = jv_record_verifier_messages::<R>(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::<1>::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations",
                    R, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                let start = performance.now();

                // Fresh prover with fresh witness each iteration
                run_prover_iteration::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                ).await;

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
        1000 => run_bench!(1000),
        2000 => run_bench!(2000),
        3000 => run_bench!(3000),
        8192 => run_bench!(8192),
        16384 => run_bench!(16384),
        32768 => run_bench!(32768),
        65536 => run_bench!(65536),
        131072 => run_bench!(131072),
        _ => Err(format!(
            "Unsupported reps value: {}. Supported: 1000, 2000, 3000, 8192, 16384, 32768, 65536, 131072",
            reps
        )),
    }
}

// =============================================================================
// WORKAROUND FOR web_spawn BUG - MAIN THREAD VERSION WITH GPU
// =============================================================================
//
// The functions below are duplicates of the versions above, but:
// 1. Remove web_spawn::spawn wrapper (run directly on main thread)
// 2. KEEP GPU support (GPU is async and doesn't block main thread!)
//
// This is a temporary workaround for the web_spawn crash issue documented in:
// commit 240d7d58 on branch debug/webspawn-bug-investigation
//
// Issue: web_spawn::spawn() crashes with "RuntimeError: unreachable"
// Root cause: Unknown, but happens inside web_spawn library during worker creation
//
// Note: GPU operations are async and run on GPU hardware independently.
// The main thread just submits work to GPU without blocking.
//
// Limitation: CPU computation runs on main thread and will block UI.
// Proper fix: Either debug web_spawn or implement manual Web Worker via wasm_bindgen.
// =============================================================================

/// Benchmark runner WITH GPU on main thread (no web_spawn).
/// Uses the original GPU-enabled functions but runs directly on main thread.
#[cfg(target_arch = "wasm32")]
async fn run_vm_bench_async_main_thread(n: u32, reps: usize) -> Result<BenchResult, String> {
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
            let (inputs, branches, _final_acc) = generate_vm_inputs_per_rep(R);

            // Record verifier messages (not timed) - WITH GPU async init
            let recorded = jv_record_verifier_messages::<R>(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::<1>::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations (main thread with GPU)",
                    R, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                let start = performance.now();

                // Fresh prover with fresh witness each iteration - WITH GPU
                run_prover_iteration::<R>(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                ).await;

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
        1000 => run_bench!(1000),
        2000 => run_bench!(2000),
        3000 => run_bench!(3000),
        8192 => run_bench!(8192),
        16384 => run_bench!(16384),
        32768 => run_bench!(32768),
        65536 => run_bench!(65536),
        131072 => run_bench!(131072),
        _ => Err(format!(
            "Unsupported reps value: {}. Supported: 1000, 2000, 3000, 8192, 16384, 32768, 65536, 131072",
            reps
        )),
    }
}

/// Worker entry point - runs the benchmark in a Web Worker context.
/// This is called FROM the worker thread, not from main thread.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn jv_vm_prover_worker(n: u32, reps: u32) -> Result<BenchResult, JsValue> {
    web_sys::console::log_1(&"[worker] jv_vm_prover_worker started".into());
    run_vm_bench_async_main_thread(n, reps as usize)
        .await
        .map_err(|e| JsValue::from_str(&e))
}

/// WASM entry point for JV VM benchmark WITH manual Web Worker.
///
/// **WORKAROUND**: Creates a manual Web Worker to bypass the web_spawn bug.
/// See commit 240d7d58 on debug/webspawn-bug-investigation for details.
///
/// **GPU SUPPORT**: Includes GPU acceleration running in worker context!
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `reps` - Number of repetitions (1000, 2000, 3000, 8192, 16384, 32768, 65536, 131072)
///
/// # Returns
/// BenchResult with elapsed_ms and total multiplications
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn jv_vm_prover_main_thread(n: u32, reps: u32) -> Result<BenchResult, JsValue> {
    web_sys::console::log_1(&"[main] Running JV benchmark on main thread (no atomics, no worker needed)".into());

    // Without atomics, we can run directly on main thread
    jv_vm_prover_worker(n, reps).await
}

// Keep worker code in case we want to re-enable atomics later
#[cfg(all(target_arch = "wasm32", feature = "use-worker"))]
#[wasm_bindgen]
pub async fn jv_vm_prover_main_thread_with_worker(n: u32, reps: u32) -> Result<BenchResult, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    web_sys::console::log_1(&"[main] Creating JV worker".into());

    // Create worker with module type (required for dynamic import())
    let mut options = web_sys::WorkerOptions::new();
    options.type_(web_sys::WorkerType::Module);

    let worker = web_sys::Worker::new_with_options("./js/jv_worker.js", &options)
        .map_err(|e| {
            web_sys::console::log_1(&format!("[main] Failed to create JV worker: {:?}", e).into());
            JsValue::from_str("Failed to create JV worker")
        })?;

    web_sys::console::log_1(&"[main] Worker created successfully, setting up message handler".into());

    // Set up message handler to receive result from worker
    let (sender, receiver) = futures::channel::oneshot::channel::<Result<BenchResult, String>>();
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(sender)));

    let onmessage_callback = {
        let sender = sender.clone();
        wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            web_sys::console::log_1(&"[main] Received message from worker".into());

            if let Some(data) = e.data().as_string() {
                web_sys::console::log_1(&format!("[main] Worker message: {}", data).into());

                // Parse JSON response from worker
                match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(json) => {
                        if let Some(error) = json.get("error") {
                            // Worker reported an error
                            let error_msg = error.as_str().unwrap_or("Unknown error");
                            web_sys::console::log_1(&format!("[main] Worker error: {}", error_msg).into());
                            if let Some(tx) = sender.borrow_mut().take() {
                                let _ = tx.send(Err(error_msg.to_string()));
                            }
                        } else if let (Some(elapsed), Some(gates)) = (json.get("elapsed_ms"), json.get("and_gates")) {
                            // Success case
                            let elapsed_ms = elapsed.as_f64().unwrap_or(0.0);
                            let and_gates = gates.as_u64().unwrap_or(0);
                            web_sys::console::log_1(&format!("[main] Benchmark complete: {} ms, {} AND gates", elapsed_ms, and_gates).into());
                            if let Some(tx) = sender.borrow_mut().take() {
                                let _ = tx.send(Ok(BenchResult { elapsed_ms, and_gates }));
                            }
                        } else {
                            web_sys::console::log_1(&"[main] Unexpected JSON format".into());
                            if let Some(tx) = sender.borrow_mut().take() {
                                let _ = tx.send(Err("Unexpected JSON format".to_string()));
                            }
                        }
                    }
                    Err(e) => {
                        web_sys::console::log_1(&format!("[main] Failed to parse JSON: {}", e).into());
                        if let Some(tx) = sender.borrow_mut().take() {
                            let _ = tx.send(Err(format!("JSON parse error: {}", e)));
                        }
                    }
                }
            }
        }) as Box<dyn FnMut(_)>)
    };

    worker.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
    onmessage_callback.forget(); // Keep callback alive

    // Set up error handler
    let onerror_callback = {
        let sender = sender.clone();
        wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::Event| {
            web_sys::console::log_1(&"[main] Worker error event received".into());

            // Try to extract error message
            let msg = if let Some(error_event) = e.dyn_ref::<web_sys::ErrorEvent>() {
                format!("Worker error: {}", error_event.message())
            } else {
                "Worker error (no details)".to_string()
            };

            web_sys::console::log_1(&format!("[main] {}", msg).into());

            if let Some(tx) = sender.borrow_mut().take() {
                let _ = tx.send(Err(msg));
            }
        }) as Box<dyn FnMut(_)>)
    };

    worker.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
    onerror_callback.forget();

    // Send benchmark parameters to worker
    let msg = js_sys::Object::new();
    js_sys::Reflect::set(&msg, &"cmd".into(), &"jv_vm_prover".into())?;
    js_sys::Reflect::set(&msg, &"n".into(), &JsValue::from(n))?;
    js_sys::Reflect::set(&msg, &"reps".into(), &JsValue::from(reps))?;

    web_sys::console::log_1(&format!("[main] Posting message to worker: n={}, reps={}", n, reps).into());
    worker.post_message(&msg)
        .map_err(|_| JsValue::from_str("Failed to post message to worker"))?;

    // Wait for result with timeout
    web_sys::console::log_1(&"[main] Waiting for worker result (10s timeout)...".into());

    let timeout_future = async {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10000)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
        Err(JsValue::from_str("Timeout after 10s"))
    };

    let result_future = async {
        receiver.await
            .map_err(|_| JsValue::from_str("Worker channel closed"))?
            .map_err(|e| JsValue::from_str(&e))
    };

    // Race between result and timeout
    let result = futures::future::select(
        Box::pin(result_future),
        Box::pin(timeout_future),
    ).await;

    worker.terminate();

    match result {
        futures::future::Either::Left((res, _)) => res,
        futures::future::Either::Right((res, _)) => res,
    }
}
