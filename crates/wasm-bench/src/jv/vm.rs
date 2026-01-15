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
async fn jv_record_verifier_messages(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
) -> JVRecordedMessages {
    let r = active_branches.len();
    web_sys::console::log_1(&format!(
        "[jv_record] Starting verifier message recording... r={}, inputs_per_rep.len()={}",
        r, inputs_per_rep.len()
    ).into());
    let mut rng = Prg::from_seed(Block::ZERO);

    // Use JVProver with IT-PAC
    web_sys::console::log_1(&"[jv_record] Creating prover...".into());
    let mut prover = JVProver::new(active_branches.to_vec(), MODULUS);
    web_sys::console::log_1(&"[jv_record] Setting up prover...".into());
    prover.setup(circuits, inputs_per_rep).unwrap();
    web_sys::console::log_1(&"[jv_record] Setting up soldering...".into());
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();

    web_sys::console::log_1(&"[jv_record] Creating verifier...".into());
    let mut verifier = JVVerifier::new(r, MODULUS, &mut rng);
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

    // Pre-initialize GPU context for slot operations (async in WASM)
    web_sys::console::log_1(&"[jv_record] Preparing GPU context...".into());
    prover.prepare_gpu_async(&setup_msg).await.expect("[jv_vm] GPU init failed - GPU is the only supported path");
    web_sys::console::log_1(&"[jv_record] GPU context ready".into());

    // Pre-initialize NTT GPU context for polynomial multiplication
    web_sys::console::log_1(&"[jv_record] Preparing NTT GPU context...".into());
    let ntt_size = 1024; // Max single-pass NTT; batched_poly_mul uses multipass for larger sizes
    let ntt_gpu = bgv_webgpu::GoldilocksNttGpu::new_async(ntt_size).await
        .expect("[jv_record] NTT GPU init failed");
    prover.set_ntt_gpu_context(std::sync::Arc::new(ntt_gpu));
    web_sys::console::log_1(&"[jv_record] NTT GPU context ready".into());

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    web_sys::console::log_1(&"[jv_record] Prover commit (IT-PAC)...".into());
    let commitment = prover.commit(&setup_msg, vole_pool).await.unwrap();
    web_sys::console::log_1(&"[jv_record] Prover commit done".into());

    // P → V: MK polynomial commitment
    web_sys::console::log_1(&"[jv_record] Prover commit MK polynomials...".into());
    let _mk_commitment = prover.commit_mk_polynomials().await.unwrap();
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

    // P → V: OpenMessage (GPU async path)
    web_sys::console::log_1(&"[jv_record] Prover open (GPU)...".into());
    let open_msg = prover.open_async(rho, gamma, verifier.topology_vectors()).await.unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier receive open...".into());
    let receive_open_result = verifier.receive_open(open_msg, gamma).unwrap();
    web_sys::console::log_1(&format!("[jv_record] receive_open result: {}", receive_open_result).into());
    assert!(receive_open_result, "receive_open failed - MK proof verification failed (GPU poly_mul may be incorrect)");

    // P → V: IT-PAC Opening
    web_sys::console::log_1(&"[jv_record] Prover open IT-PAC...".into());
    let itpac_open_msg = prover.open_itpac().unwrap();
    web_sys::console::log_1(&"[jv_record] Verifier verify IT-PAC...".into());
    assert!(verifier.verify_itpac_opening(&itpac_open_msg), "IT-PAC verification failed");

    // P → V: AggregatedLpzkProofMessage
    web_sys::console::log_1(&"[jv_record] Prover prove multiplications starting...".into());
    let lpzk_proof = match prover.prove_multiplications_aggregated(gamma) {
        Ok(p) => {
            web_sys::console::log_1(&format!(
                "[jv_record] LPZK proof success: quotient_len={}, aggregated_check={}",
                p.quotient_coeffs.len(), p.aggregated_check
            ).into());
            p
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[jv_record] LPZK proof FAILED: {:?}", e).into());
            panic!("prove_multiplications_aggregated failed: {:?}", e);
        }
    };
    web_sys::console::log_1(&"[jv_record] Verifier verify multiplications starting...".into());
    let result = match verifier.verify_multiplications_aggregated(lpzk_proof, gamma) {
        Ok(r) => {
            web_sys::console::log_1(&format!("[jv_record] verify result: {}", r).into());
            r
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[jv_record] verify FAILED: {:?}", e).into());
            panic!("verify_multiplications_aggregated failed: {:?}", e);
        }
    };
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

/// Timing breakdown struct for all measured operations.
/// This matches the 17 fields from JVProver::timing_breakdown() plus benchmark-level timing.
#[cfg(target_arch = "wasm32")]
#[derive(Default, Clone, Copy)]
pub struct TimingBreakdown {
    // From JVProver::timing_breakdown()
    pub intt_ms: f64,
    pub collapse_ms: f64,
    pub poly_div_ms: f64,
    pub open_ms: f64,
    pub mk_poly_ms: f64,
    pub itpac_ms: f64,
    pub packing_ms: f64,
    pub setup_ms: f64,
    pub disclose_ms: f64,
    pub commit_soldering_ms: f64,
    pub lpzk_accumulation_ms: f64,
    pub reveal_soldering_ms: f64,
    pub mk_binary_ms: f64,
    pub mk_sum_ms: f64,
    pub open_polynomial_ms: f64,
    pub mk_commit_vole_ms: f64,
    pub mk_commit_packing_ms: f64,
    // Benchmark-level timing (total call time, includes internal timing)
    pub vole_pool_ms: f64,
    pub prover_new_ms: f64,
    pub prover_setup_ms: f64,
    pub prover_setup_soldering_ms: f64,
    pub commit_ms: f64,
    pub commit_mk_poly_ms: f64,
    pub commit_soldering_call_ms: f64,
    pub disclose_call_ms: f64,
    pub reveal_soldering_call_ms: f64,
    pub open_call_ms: f64,
    pub open_itpac_ms: f64,
    pub prove_mults_ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl TimingBreakdown {
    fn from_prover_timing(t: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)) -> Self {
        Self {
            intt_ms: t.0,
            collapse_ms: t.1,
            poly_div_ms: t.2,
            open_ms: t.3,
            mk_poly_ms: t.4,
            itpac_ms: t.5,
            packing_ms: t.6,
            setup_ms: t.7,
            disclose_ms: t.8,
            commit_soldering_ms: t.9,
            lpzk_accumulation_ms: t.10,
            reveal_soldering_ms: t.11,
            mk_binary_ms: t.12,
            mk_sum_ms: t.13,
            open_polynomial_ms: t.14,
            mk_commit_vole_ms: t.15,
            mk_commit_packing_ms: t.16,
            ..Default::default()
        }
    }
}

/// Runs a single prover iteration with fresh witness setup.
/// Returns (gpu_time_ms, TimingBreakdown with all timing fields)
#[cfg(target_arch = "wasm32")]
async fn run_prover_iteration(
    circuits: &CircuitBatch,
    active_branches: &[usize],
    inputs_per_rep: &[Vec<u64>],
    soldering_constraints: &[SolderingConstraint],
    recorded: &JVRecordedMessages,
    gpu_ctx: &std::sync::Arc<bgv_webgpu::RnsSlotMulGpu>,
    ntt_gpu_ctx: &std::sync::Arc<bgv_webgpu::GoldilocksNttGpu>,
) -> (f64, TimingBreakdown) {
    let performance = web_sys::window().unwrap().performance().unwrap();
    let mut rng = Prg::from_seed(Block::ZERO);

    // Fresh prover with fresh witness setup each iteration
    let prover_new_start = performance.now();
    let mut prover = JVProver::new(active_branches.to_vec(), MODULUS);
    let prover_new_ms = performance.now() - prover_new_start;

    let prover_setup_start = performance.now();
    prover.setup(circuits, inputs_per_rep).unwrap();
    let prover_setup_ms = performance.now() - prover_setup_start;

    let prover_setup_soldering_start = performance.now();
    prover.setup_soldering(soldering_constraints.to_vec(), &mut rng).unwrap();
    let prover_setup_soldering_ms = performance.now() - prover_setup_soldering_start;

    // Create VOLE pool for IT-PAC commitments
    let vole_start = performance.now();
    let vole_pool = VolePool::generate(&recorded.global_key, recorded.circuit_size * 2, &mut rng);
    let vole_pool_ms = performance.now() - vole_start;

    // Set pre-initialized GPU contexts (always present - GPU is only path)
    prover.set_gpu_context(gpu_ctx.clone());
    prover.set_ntt_gpu_context(ntt_gpu_ctx.clone());

    // P → V: CommitmentMessage (IT-PAC ciphertexts)
    let commit_start = performance.now();
    let _commitment = prover.commit(&recorded.setup_msg, vole_pool).await.unwrap();
    let commit_ms = performance.now() - commit_start;

    // P → V: MK polynomial commitment
    let commit_mk_start = performance.now();
    let _mk_commitment = prover.commit_mk_polynomials().await.unwrap();
    let commit_mk_poly_ms = performance.now() - commit_mk_start;

    // P → V: Soldering commitment (optional)
    let commit_soldering_start = performance.now();
    let _soldering_commit = prover.commit_soldering().unwrap();
    let commit_soldering_call_ms = performance.now() - commit_soldering_start;

    let disclose_start = performance.now();
    let _disclosure = prover.disclose(recorded.chi, &recorded.topology_vectors).unwrap();
    let disclose_call_ms = performance.now() - disclose_start;

    let reveal_soldering_start = performance.now();
    if let Some(ref challenge) = recorded.soldering_challenge {
        let _ = prover.reveal_soldering_aggregated(challenge).unwrap();
    }
    let reveal_soldering_call_ms = performance.now() - reveal_soldering_start;

    let open_start = performance.now();
    let _open_msg = prover.open_async(recorded.rho, recorded.gamma, &recorded.topology_vectors).await.unwrap();
    let open_call_ms = performance.now() - open_start;

    // IT-PAC opening
    let open_itpac_start = performance.now();
    let _itpac_open_msg = prover.open_itpac().unwrap();
    let open_itpac_ms = performance.now() - open_itpac_start;

    let prove_mults_start = performance.now();
    let _lpzk_proof = prover.prove_multiplications_aggregated(recorded.gamma).unwrap();
    let prove_mults_ms = performance.now() - prove_mults_start;

    // Build timing breakdown from prover + benchmark-level timing
    let mut timing = TimingBreakdown::from_prover_timing(prover.timing_breakdown());
    timing.vole_pool_ms = vole_pool_ms;
    timing.prover_new_ms = prover_new_ms;
    timing.prover_setup_ms = prover_setup_ms;
    timing.prover_setup_soldering_ms = prover_setup_soldering_ms;
    timing.commit_ms = commit_ms;
    timing.commit_mk_poly_ms = commit_mk_poly_ms;
    timing.commit_soldering_call_ms = commit_soldering_call_ms;
    timing.disclose_call_ms = disclose_call_ms;
    timing.reveal_soldering_call_ms = reveal_soldering_call_ms;
    timing.open_call_ms = open_call_ms;
    timing.open_itpac_ms = open_itpac_ms;
    timing.prove_mults_ms = prove_mults_ms;

    (prover.total_gpu_time_ms(), timing)
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
            let recorded = jv_record_verifier_messages(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            // Test: create a dummy second GPU device to verify multiple devices work
            {
                web_sys::console::log_1(&"[jv_vm] Testing second GPU device creation...".into());
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..Default::default()
                });
                let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }).await.expect("Failed to get adapter");
                let (_device2, _queue2) = adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("test-device-2"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                ).await.expect("Failed to create second device");
                web_sys::console::log_1(&"[jv_vm] Second GPU device created successfully!".into());
            }

            // Pre-initialize NTT GPU context for polynomial multiplication
            // NTT is capped at 2048 due to GPU shared memory limits (16KB safe for browser WebGPU)
            let ntt_gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing NTT GPU context...".into());
                let ntt_start = performance.now();
                let ntt_size = 1024; // Max single-pass NTT; batched_poly_mul uses multipass for larger sizes
                let ctx = bgv_webgpu::GoldilocksNttGpu::new_async(ntt_size).await
                    .expect("[jv_vm] NTT GPU context creation failed");
                let elapsed = performance.now() - ntt_start;
                let note = if (2 * R).next_power_of_two() > 2048 { " (capped)" } else { "" };
                web_sys::console::log_1(&format!("[jv_vm] NTT GPU context (n={}) created in {:.2}ms{}", ntt_size, elapsed, note).into());
                std::sync::Arc::new(ctx)
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations",
                    R, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;
            let mut total_gpu_time_ms = 0.0;
            let mut total_timing = TimingBreakdown::default();

            for i in 0..n {
                let start = performance.now();

                // Fresh prover with fresh witness each iteration
                let (gpu_time, timing) = run_prover_iteration(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                    &ntt_gpu_ctx,
                ).await;

                total_elapsed_ms += performance.now() - start;
                total_gpu_time_ms += gpu_time;
                // Accumulate all timing fields
                total_timing.intt_ms += timing.intt_ms;
                total_timing.collapse_ms += timing.collapse_ms;
                total_timing.poly_div_ms += timing.poly_div_ms;
                total_timing.open_ms += timing.open_ms;
                total_timing.mk_poly_ms += timing.mk_poly_ms;
                total_timing.itpac_ms += timing.itpac_ms;
                total_timing.packing_ms += timing.packing_ms;
                total_timing.setup_ms += timing.setup_ms;
                total_timing.disclose_ms += timing.disclose_ms;
                total_timing.commit_soldering_ms += timing.commit_soldering_ms;
                total_timing.lpzk_accumulation_ms += timing.lpzk_accumulation_ms;
                total_timing.reveal_soldering_ms += timing.reveal_soldering_ms;
                total_timing.mk_binary_ms += timing.mk_binary_ms;
                total_timing.mk_sum_ms += timing.mk_sum_ms;
                total_timing.open_polynomial_ms += timing.open_polynomial_ms;
                total_timing.mk_commit_vole_ms += timing.mk_commit_vole_ms;
                total_timing.mk_commit_packing_ms += timing.mk_commit_packing_ms;
                // Benchmark-level timing (total call times)
                total_timing.vole_pool_ms += timing.vole_pool_ms;
                total_timing.prover_new_ms += timing.prover_new_ms;
                total_timing.prover_setup_ms += timing.prover_setup_ms;
                total_timing.prover_setup_soldering_ms += timing.prover_setup_soldering_ms;
                total_timing.commit_ms += timing.commit_ms;
                total_timing.commit_mk_poly_ms += timing.commit_mk_poly_ms;
                total_timing.commit_soldering_call_ms += timing.commit_soldering_call_ms;
                total_timing.disclose_call_ms += timing.disclose_call_ms;
                total_timing.reveal_soldering_call_ms += timing.reveal_soldering_call_ms;
                total_timing.open_call_ms += timing.open_call_ms;
                total_timing.open_itpac_ms += timing.open_itpac_ms;
                total_timing.prove_mults_ms += timing.prove_mults_ms;

                if (i + 1) % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[jv_vm] {} iterations done", i + 1).into(),
                    );
                }
            }

            // Total mults = iterations * reps * branches * mults_per_circuit
            let total_mults = n as u64 * R as u64 * NUM_BRANCHES as u64 * num_mults as u64;

            // Helper macro for printing timing
            macro_rules! print_timing {
                ($name:expr, $value:expr) => {
                    web_sys::console::log_1(
                        &format!(
                            "[jv_vm]   {}: {:.2}ms ({:.1}%)",
                            $name,
                            $value,
                            ($value / total_elapsed_ms) * 100.0
                        )
                        .into(),
                    );
                };
            }

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Done: {:.2}ms total, {:.2}ms/iter, {} total mults",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64,
                    total_mults
                )
                .into(),
            );
            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Total GPU time: {:.2}ms ({:.1}% of total time)",
                    total_gpu_time_ms,
                    (total_gpu_time_ms / total_elapsed_ms) * 100.0
                )
                .into(),
            );

            // Print timing breakdown - Prover internal timing
            web_sys::console::log_1(&"[jv_vm] === Prover Internal Timing ===".into());
            print_timing!("INTT (interpolation)", total_timing.intt_ms);
            print_timing!("Collapse (CT addition)", total_timing.collapse_ms);
            print_timing!("Poly division (LPZK)", total_timing.poly_div_ms);
            print_timing!("Open (IT-PAC eval)", total_timing.open_ms);
            print_timing!("MK poly (interpolation)", total_timing.mk_poly_ms);
            print_timing!("IT-PAC creation", total_timing.itpac_ms);
            print_timing!("Packing (2-way)", total_timing.packing_ms);
            print_timing!("Setup (circuit eval)", total_timing.setup_ms);
            print_timing!("Disclose (topology)", total_timing.disclose_ms);
            print_timing!("Commit soldering", total_timing.commit_soldering_ms);
            print_timing!("LPZK poly accumulation", total_timing.lpzk_accumulation_ms);
            print_timing!("Reveal soldering", total_timing.reveal_soldering_ms);
            print_timing!("MK binary eval", total_timing.mk_binary_ms);
            print_timing!("MK sum", total_timing.mk_sum_ms);
            print_timing!("Open polynomial", total_timing.open_polynomial_ms);
            print_timing!("MK commit VOLE", total_timing.mk_commit_vole_ms);
            print_timing!("MK commit packing", total_timing.mk_commit_packing_ms);

            // Print benchmark-level timing (total time for each call)
            web_sys::console::log_1(&"[jv_vm] === Call-Level Timing (should sum to ~100%) ===".into());
            print_timing!("VOLE pool generation", total_timing.vole_pool_ms);
            print_timing!("Prover::new", total_timing.prover_new_ms);
            print_timing!("Prover::setup", total_timing.prover_setup_ms);
            print_timing!("Prover::setup_soldering", total_timing.prover_setup_soldering_ms);
            print_timing!("Prover::commit", total_timing.commit_ms);
            print_timing!("Prover::commit_mk_polynomials", total_timing.commit_mk_poly_ms);
            print_timing!("Prover::commit_soldering", total_timing.commit_soldering_call_ms);
            print_timing!("Prover::disclose", total_timing.disclose_call_ms);
            print_timing!("Prover::reveal_soldering", total_timing.reveal_soldering_call_ms);
            print_timing!("Prover::open", total_timing.open_call_ms);
            print_timing!("Prover::open_itpac", total_timing.open_itpac_ms);
            print_timing!("Prover::prove_multiplications", total_timing.prove_mults_ms);

            // Calculate total call time (should be close to elapsed)
            let total_call_time = total_timing.vole_pool_ms + total_timing.prover_new_ms +
                                  total_timing.prover_setup_ms + total_timing.prover_setup_soldering_ms +
                                  total_timing.commit_ms + total_timing.commit_mk_poly_ms +
                                  total_timing.commit_soldering_call_ms + total_timing.disclose_call_ms +
                                  total_timing.reveal_soldering_call_ms + total_timing.open_call_ms +
                                  total_timing.open_itpac_ms + total_timing.prove_mults_ms;
            let loop_overhead = total_elapsed_ms - total_call_time;

            web_sys::console::log_1(&"[jv_vm] === Summary ===".into());
            print_timing!("Total call time", total_call_time);
            print_timing!("Loop/other overhead", loop_overhead);

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults,
            })
        }};
    }

    match reps {
        100 => run_bench!(100),
        1000 => run_bench!(1000),
        2000 => run_bench!(2000),
        2048 => run_bench!(2048),
        3000 => run_bench!(3000),
        4096 => run_bench!(4096),
        8192 => run_bench!(8192),
        16384 => run_bench!(16384),
        32768 => run_bench!(32768),
        65536 => run_bench!(65536),
        131072 => run_bench!(131072),
        _ => {
            // Validate reps <= 32768 (GPU multipass NTT supports up to 1M elements)
            // Polynomial multiplication needs 2*R coefficients, so max R = 32768 for NTT size 65536
            if reps > 32768 {
                return Err(format!(
                    "reps={} exceeds max 32768 (GPU multipass NTT limited to ~1M elements)",
                    reps
                ));
            }

            // Runtime reps value (no const generic needed anymore)
            let (inputs, branches, _final_acc) = generate_vm_inputs_per_rep(reps);

            // Record verifier messages (not timed) - async for GPU init
            let recorded = jv_record_verifier_messages(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            // Test: create a dummy second GPU device to verify multiple devices work
            {
                web_sys::console::log_1(&"[jv_vm] Testing second GPU device creation...".into());
                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..Default::default()
                });
                let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }).await.expect("Failed to get adapter");
                let (_device2, _queue2) = adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("test-device-2"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                ).await.expect("Failed to create second device");
                web_sys::console::log_1(&"[jv_vm] Second GPU device created successfully!".into());
            }

            // Pre-initialize NTT GPU context for polynomial multiplication
            let ntt_gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing NTT GPU context...".into());
                let ntt_start = performance.now();
                let ntt_size = 1024; // Max single-pass NTT; batched_poly_mul uses multipass for larger sizes
                let ctx = bgv_webgpu::GoldilocksNttGpu::new_async(ntt_size).await
                    .expect("[jv_vm] NTT GPU context creation failed");
                let elapsed = performance.now() - ntt_start;
                let note = if (2 * reps).next_power_of_two() > 2048 { " (capped)" } else { "" };
                web_sys::console::log_1(&format!("[jv_vm] NTT GPU context (n={}) created in {:.2}ms{}", ntt_size, elapsed, note).into());
                std::sync::Arc::new(ctx)
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations",
                    reps, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;
            let mut total_gpu_time_ms = 0.0;
            let mut total_timing = TimingBreakdown::default();

            for i in 0..n {
                let start = performance.now();

                let (gpu_time, timing) = run_prover_iteration(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                    &ntt_gpu_ctx,
                ).await;

                total_elapsed_ms += performance.now() - start;
                total_gpu_time_ms += gpu_time;
                total_timing.intt_ms += timing.intt_ms;
                total_timing.collapse_ms += timing.collapse_ms;
                total_timing.poly_div_ms += timing.poly_div_ms;
                total_timing.open_ms += timing.open_ms;
                total_timing.mk_poly_ms += timing.mk_poly_ms;
                total_timing.itpac_ms += timing.itpac_ms;
                total_timing.packing_ms += timing.packing_ms;
                total_timing.setup_ms += timing.setup_ms;
                total_timing.disclose_ms += timing.disclose_ms;
                total_timing.commit_soldering_ms += timing.commit_soldering_ms;
                total_timing.lpzk_accumulation_ms += timing.lpzk_accumulation_ms;
                total_timing.reveal_soldering_ms += timing.reveal_soldering_ms;
                total_timing.mk_binary_ms += timing.mk_binary_ms;
                total_timing.mk_sum_ms += timing.mk_sum_ms;
                total_timing.open_polynomial_ms += timing.open_polynomial_ms;
                total_timing.mk_commit_vole_ms += timing.mk_commit_vole_ms;
                total_timing.mk_commit_packing_ms += timing.mk_commit_packing_ms;
                total_timing.vole_pool_ms += timing.vole_pool_ms;
                total_timing.prover_new_ms += timing.prover_new_ms;
                total_timing.prover_setup_ms += timing.prover_setup_ms;
                total_timing.prover_setup_soldering_ms += timing.prover_setup_soldering_ms;
                total_timing.commit_ms += timing.commit_ms;
                total_timing.commit_mk_poly_ms += timing.commit_mk_poly_ms;
                total_timing.commit_soldering_call_ms += timing.commit_soldering_call_ms;
                total_timing.disclose_call_ms += timing.disclose_call_ms;
                total_timing.reveal_soldering_call_ms += timing.reveal_soldering_call_ms;
                total_timing.open_call_ms += timing.open_call_ms;
                total_timing.open_itpac_ms += timing.open_itpac_ms;
                total_timing.prove_mults_ms += timing.prove_mults_ms;

                if (i + 1) % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[jv_vm] {} iterations done", i + 1).into(),
                    );
                }
            }

            let total_mults = n as u64 * reps as u64 * NUM_BRANCHES as u64 * num_mults as u64;

            macro_rules! print_timing {
                ($name:expr, $value:expr) => {
                    web_sys::console::log_1(
                        &format!(
                            "[jv_vm]   {}: {:.2}ms ({:.1}%)",
                            $name,
                            $value,
                            ($value / total_elapsed_ms) * 100.0
                        )
                        .into(),
                    );
                };
            }

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Done: {:.2}ms total, {:.2}ms/iter, {} total mults",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64,
                    total_mults
                )
                .into(),
            );
            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Total GPU time: {:.2}ms ({:.1}% of total time)",
                    total_gpu_time_ms,
                    (total_gpu_time_ms / total_elapsed_ms) * 100.0
                )
                .into(),
            );

            web_sys::console::log_1(&"[jv_vm] === Prover Internal Timing ===".into());
            print_timing!("INTT (interpolation)", total_timing.intt_ms);
            print_timing!("Collapse (CT addition)", total_timing.collapse_ms);
            print_timing!("Poly division (LPZK)", total_timing.poly_div_ms);
            print_timing!("Open (IT-PAC eval)", total_timing.open_ms);
            print_timing!("MK poly (interpolation)", total_timing.mk_poly_ms);
            print_timing!("IT-PAC creation", total_timing.itpac_ms);
            print_timing!("Packing (2-way)", total_timing.packing_ms);
            print_timing!("Setup (circuit eval)", total_timing.setup_ms);
            print_timing!("Disclose (topology)", total_timing.disclose_ms);
            print_timing!("Commit soldering", total_timing.commit_soldering_ms);
            print_timing!("LPZK poly accumulation", total_timing.lpzk_accumulation_ms);
            print_timing!("Reveal soldering", total_timing.reveal_soldering_ms);
            print_timing!("MK binary eval", total_timing.mk_binary_ms);
            print_timing!("MK sum", total_timing.mk_sum_ms);
            print_timing!("Open polynomial", total_timing.open_polynomial_ms);
            print_timing!("MK commit VOLE", total_timing.mk_commit_vole_ms);
            print_timing!("MK commit packing", total_timing.mk_commit_packing_ms);

            web_sys::console::log_1(&"[jv_vm] === Call-Level Timing (should sum to ~100%) ===".into());
            print_timing!("VOLE pool generation", total_timing.vole_pool_ms);
            print_timing!("Prover::new", total_timing.prover_new_ms);
            print_timing!("Prover::setup", total_timing.prover_setup_ms);
            print_timing!("Prover::setup_soldering", total_timing.prover_setup_soldering_ms);
            print_timing!("Prover::commit", total_timing.commit_ms);
            print_timing!("Prover::commit_mk_polynomials", total_timing.commit_mk_poly_ms);
            print_timing!("Prover::commit_soldering", total_timing.commit_soldering_call_ms);
            print_timing!("Prover::disclose", total_timing.disclose_call_ms);
            print_timing!("Prover::reveal_soldering", total_timing.reveal_soldering_call_ms);
            print_timing!("Prover::open", total_timing.open_call_ms);
            print_timing!("Prover::open_itpac", total_timing.open_itpac_ms);
            print_timing!("Prover::prove_multiplications", total_timing.prove_mults_ms);

            let total_call_time = total_timing.vole_pool_ms + total_timing.prover_new_ms +
                                  total_timing.prover_setup_ms + total_timing.prover_setup_soldering_ms +
                                  total_timing.commit_ms + total_timing.commit_mk_poly_ms +
                                  total_timing.commit_soldering_call_ms + total_timing.disclose_call_ms +
                                  total_timing.reveal_soldering_call_ms + total_timing.open_call_ms +
                                  total_timing.open_itpac_ms + total_timing.prove_mults_ms;
            let loop_overhead = total_elapsed_ms - total_call_time;

            web_sys::console::log_1(&"[jv_vm] === Summary ===".into());
            print_timing!("Total call time", total_call_time);
            print_timing!("Loop/other overhead", loop_overhead);

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults,
            })
        },
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
            let recorded = jv_record_verifier_messages(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            // Pre-initialize NTT GPU context for polynomial multiplication
            // Uses a separate device (wgpu Device/Queue don't impl Clone, so can't share)
            // NTT is capped at 2048 due to GPU shared memory limits (16KB safe for browser WebGPU)
            let ntt_gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing NTT GPU context (new device)...".into());
                let ntt_start = performance.now();
                let ntt_size = 1024; // Max single-pass NTT; batched_poly_mul uses multipass for larger sizes
                let ctx = bgv_webgpu::GoldilocksNttGpu::new_async(ntt_size).await
                    .expect("[jv_vm] NTT GPU context creation failed");
                let elapsed = performance.now() - ntt_start;
                let note = if (2 * R).next_power_of_two() > 2048 { " (capped)" } else { "" };
                web_sys::console::log_1(&format!("[jv_vm] NTT GPU context (n={}) created in {:.2}ms{}", ntt_size, elapsed, note).into());
                std::sync::Arc::new(ctx)
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations (main thread with GPU)",
                    R, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;
            let mut total_gpu_time_ms = 0.0;
            let mut total_timing = TimingBreakdown::default();

            for i in 0..n {
                let start = performance.now();

                // Fresh prover with fresh witness each iteration - WITH GPU
                let (gpu_time, timing) = run_prover_iteration(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                    &ntt_gpu_ctx,
                ).await;

                total_elapsed_ms += performance.now() - start;
                total_gpu_time_ms += gpu_time;
                // Accumulate all timing fields
                total_timing.intt_ms += timing.intt_ms;
                total_timing.collapse_ms += timing.collapse_ms;
                total_timing.poly_div_ms += timing.poly_div_ms;
                total_timing.open_ms += timing.open_ms;
                total_timing.mk_poly_ms += timing.mk_poly_ms;
                total_timing.itpac_ms += timing.itpac_ms;
                total_timing.packing_ms += timing.packing_ms;
                total_timing.setup_ms += timing.setup_ms;
                total_timing.disclose_ms += timing.disclose_ms;
                total_timing.commit_soldering_ms += timing.commit_soldering_ms;
                total_timing.lpzk_accumulation_ms += timing.lpzk_accumulation_ms;
                total_timing.reveal_soldering_ms += timing.reveal_soldering_ms;
                total_timing.mk_binary_ms += timing.mk_binary_ms;
                total_timing.mk_sum_ms += timing.mk_sum_ms;
                total_timing.open_polynomial_ms += timing.open_polynomial_ms;
                total_timing.mk_commit_vole_ms += timing.mk_commit_vole_ms;
                total_timing.mk_commit_packing_ms += timing.mk_commit_packing_ms;
                // Benchmark-level timing (total call times)
                total_timing.vole_pool_ms += timing.vole_pool_ms;
                total_timing.prover_new_ms += timing.prover_new_ms;
                total_timing.prover_setup_ms += timing.prover_setup_ms;
                total_timing.prover_setup_soldering_ms += timing.prover_setup_soldering_ms;
                total_timing.commit_ms += timing.commit_ms;
                total_timing.commit_mk_poly_ms += timing.commit_mk_poly_ms;
                total_timing.commit_soldering_call_ms += timing.commit_soldering_call_ms;
                total_timing.disclose_call_ms += timing.disclose_call_ms;
                total_timing.reveal_soldering_call_ms += timing.reveal_soldering_call_ms;
                total_timing.open_call_ms += timing.open_call_ms;
                total_timing.open_itpac_ms += timing.open_itpac_ms;
                total_timing.prove_mults_ms += timing.prove_mults_ms;

                if (i + 1) % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[jv_vm] {} iterations done", i + 1).into(),
                    );
                }
            }

            // Total mults = iterations * reps * branches * mults_per_circuit
            let total_mults = n as u64 * R as u64 * NUM_BRANCHES as u64 * num_mults as u64;

            // Helper macro for printing timing
            macro_rules! print_timing {
                ($name:expr, $value:expr) => {
                    web_sys::console::log_1(
                        &format!(
                            "[jv_vm]   {}: {:.2}ms ({:.1}%)",
                            $name,
                            $value,
                            ($value / total_elapsed_ms) * 100.0
                        )
                        .into(),
                    );
                };
            }

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Done: {:.2}ms total, {:.2}ms/iter, {} total mults",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64,
                    total_mults
                )
                .into(),
            );
            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Total GPU time: {:.2}ms ({:.1}% of total time)",
                    total_gpu_time_ms,
                    (total_gpu_time_ms / total_elapsed_ms) * 100.0
                )
                .into(),
            );

            // Print timing breakdown - Prover internal timing
            web_sys::console::log_1(&"[jv_vm] === Prover Internal Timing ===".into());
            print_timing!("INTT (interpolation)", total_timing.intt_ms);
            print_timing!("Collapse (CT addition)", total_timing.collapse_ms);
            print_timing!("Poly division (LPZK)", total_timing.poly_div_ms);
            print_timing!("Open (IT-PAC eval)", total_timing.open_ms);
            print_timing!("MK poly (interpolation)", total_timing.mk_poly_ms);
            print_timing!("IT-PAC creation", total_timing.itpac_ms);
            print_timing!("Packing (2-way)", total_timing.packing_ms);
            print_timing!("Setup (circuit eval)", total_timing.setup_ms);
            print_timing!("Disclose (topology)", total_timing.disclose_ms);
            print_timing!("Commit soldering", total_timing.commit_soldering_ms);
            print_timing!("LPZK poly accumulation", total_timing.lpzk_accumulation_ms);
            print_timing!("Reveal soldering", total_timing.reveal_soldering_ms);
            print_timing!("MK binary eval", total_timing.mk_binary_ms);
            print_timing!("MK sum", total_timing.mk_sum_ms);
            print_timing!("Open polynomial", total_timing.open_polynomial_ms);
            print_timing!("MK commit VOLE", total_timing.mk_commit_vole_ms);
            print_timing!("MK commit packing", total_timing.mk_commit_packing_ms);

            // Print benchmark-level timing (total time for each call)
            web_sys::console::log_1(&"[jv_vm] === Call-Level Timing (should sum to ~100%) ===".into());
            print_timing!("VOLE pool generation", total_timing.vole_pool_ms);
            print_timing!("Prover::new", total_timing.prover_new_ms);
            print_timing!("Prover::setup", total_timing.prover_setup_ms);
            print_timing!("Prover::setup_soldering", total_timing.prover_setup_soldering_ms);
            print_timing!("Prover::commit", total_timing.commit_ms);
            print_timing!("Prover::commit_mk_polynomials", total_timing.commit_mk_poly_ms);
            print_timing!("Prover::commit_soldering", total_timing.commit_soldering_call_ms);
            print_timing!("Prover::disclose", total_timing.disclose_call_ms);
            print_timing!("Prover::reveal_soldering", total_timing.reveal_soldering_call_ms);
            print_timing!("Prover::open", total_timing.open_call_ms);
            print_timing!("Prover::open_itpac", total_timing.open_itpac_ms);
            print_timing!("Prover::prove_multiplications", total_timing.prove_mults_ms);

            // Calculate total call time (should be close to elapsed)
            let total_call_time = total_timing.vole_pool_ms + total_timing.prover_new_ms +
                                  total_timing.prover_setup_ms + total_timing.prover_setup_soldering_ms +
                                  total_timing.commit_ms + total_timing.commit_mk_poly_ms +
                                  total_timing.commit_soldering_call_ms + total_timing.disclose_call_ms +
                                  total_timing.reveal_soldering_call_ms + total_timing.open_call_ms +
                                  total_timing.open_itpac_ms + total_timing.prove_mults_ms;
            let loop_overhead = total_elapsed_ms - total_call_time;

            web_sys::console::log_1(&"[jv_vm] === Summary ===".into());
            print_timing!("Total call time", total_call_time);
            print_timing!("Loop/other overhead", loop_overhead);

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults,
            })
        }};
    }

    match reps {
        100 => run_bench!(100),
        1000 => run_bench!(1000),
        2000 => run_bench!(2000),
        2048 => run_bench!(2048),
        3000 => run_bench!(3000),
        4096 => run_bench!(4096),
        8192 => run_bench!(8192),
        16384 => run_bench!(16384),
        32768 => run_bench!(32768),
        65536 => run_bench!(65536),
        131072 => run_bench!(131072),
        _ => {
            // Validate reps <= 32768 (GPU multipass NTT supports up to 1M elements)
            // Polynomial multiplication needs 2*R coefficients, so max R = 32768 for NTT size 65536
            if reps > 32768 {
                return Err(format!(
                    "reps={} exceeds max 32768 (GPU multipass NTT limited to ~1M elements)",
                    reps
                ));
            }

            // Runtime reps value (no const generic needed anymore)
            let (inputs, branches, _final_acc) = generate_vm_inputs_per_rep(reps);

            // Record verifier messages (not timed) - WITH GPU async init
            let recorded = jv_record_verifier_messages(
                &circuits,
                &branches,
                &inputs,
                &soldering,
            ).await;

            // Pre-initialize GPU context once (not timed) - MUST succeed (GPU is only path)
            let gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing GPU context...".into());
                let gpu_start = performance.now();
                let ctx = mpz_justvengers::JVProver::create_gpu_context_goldilocks(8192).await
                    .expect("[jv_vm] GPU context creation failed - GPU is the only supported path");
                let elapsed = performance.now() - gpu_start;
                web_sys::console::log_1(&format!("[jv_vm] GPU context created in {:.2}ms", elapsed).into());
                ctx
            };

            // Pre-initialize NTT GPU context for polynomial multiplication
            let ntt_gpu_ctx = {
                web_sys::console::log_1(&"[jv_vm] Pre-initializing NTT GPU context (new device)...".into());
                let ntt_start = performance.now();
                let ntt_size = 1024; // Max single-pass NTT; batched_poly_mul uses multipass for larger sizes
                let ctx = bgv_webgpu::GoldilocksNttGpu::new_async(ntt_size).await
                    .expect("[jv_vm] NTT GPU context creation failed");
                let elapsed = performance.now() - ntt_start;
                let note = if (2 * reps).next_power_of_two() > 2048 { " (capped)" } else { "" };
                web_sys::console::log_1(&format!("[jv_vm] NTT GPU context (n={}) created in {:.2}ms{}", ntt_size, elapsed, note).into());
                std::sync::Arc::new(ctx)
            };

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] R={}, mults/circuit={}, starting {} iterations (main thread with GPU)",
                    reps, num_mults, n
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;
            let mut total_gpu_time_ms = 0.0;
            let mut total_timing = TimingBreakdown::default();

            for i in 0..n {
                let start = performance.now();

                let (gpu_time, timing) = run_prover_iteration(
                    &circuits,
                    &branches,
                    &inputs,
                    &soldering,
                    &recorded,
                    &gpu_ctx,
                    &ntt_gpu_ctx,
                ).await;

                total_elapsed_ms += performance.now() - start;
                total_gpu_time_ms += gpu_time;
                total_timing.intt_ms += timing.intt_ms;
                total_timing.collapse_ms += timing.collapse_ms;
                total_timing.poly_div_ms += timing.poly_div_ms;
                total_timing.open_ms += timing.open_ms;
                total_timing.mk_poly_ms += timing.mk_poly_ms;
                total_timing.itpac_ms += timing.itpac_ms;
                total_timing.packing_ms += timing.packing_ms;
                total_timing.setup_ms += timing.setup_ms;
                total_timing.disclose_ms += timing.disclose_ms;
                total_timing.commit_soldering_ms += timing.commit_soldering_ms;
                total_timing.lpzk_accumulation_ms += timing.lpzk_accumulation_ms;
                total_timing.reveal_soldering_ms += timing.reveal_soldering_ms;
                total_timing.mk_binary_ms += timing.mk_binary_ms;
                total_timing.mk_sum_ms += timing.mk_sum_ms;
                total_timing.open_polynomial_ms += timing.open_polynomial_ms;
                total_timing.mk_commit_vole_ms += timing.mk_commit_vole_ms;
                total_timing.mk_commit_packing_ms += timing.mk_commit_packing_ms;
                total_timing.vole_pool_ms += timing.vole_pool_ms;
                total_timing.prover_new_ms += timing.prover_new_ms;
                total_timing.prover_setup_ms += timing.prover_setup_ms;
                total_timing.prover_setup_soldering_ms += timing.prover_setup_soldering_ms;
                total_timing.commit_ms += timing.commit_ms;
                total_timing.commit_mk_poly_ms += timing.commit_mk_poly_ms;
                total_timing.commit_soldering_call_ms += timing.commit_soldering_call_ms;
                total_timing.disclose_call_ms += timing.disclose_call_ms;
                total_timing.reveal_soldering_call_ms += timing.reveal_soldering_call_ms;
                total_timing.open_call_ms += timing.open_call_ms;
                total_timing.open_itpac_ms += timing.open_itpac_ms;
                total_timing.prove_mults_ms += timing.prove_mults_ms;

                if (i + 1) % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[jv_vm] {} iterations done", i + 1).into(),
                    );
                }
            }

            let total_mults = n as u64 * reps as u64 * NUM_BRANCHES as u64 * num_mults as u64;

            macro_rules! print_timing {
                ($name:expr, $value:expr) => {
                    web_sys::console::log_1(
                        &format!(
                            "[jv_vm]   {}: {:.2}ms ({:.1}%)",
                            $name,
                            $value,
                            ($value / total_elapsed_ms) * 100.0
                        )
                        .into(),
                    );
                };
            }

            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Done: {:.2}ms total, {:.2}ms/iter, {} total mults",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64,
                    total_mults
                )
                .into(),
            );
            web_sys::console::log_1(
                &format!(
                    "[jv_vm] Total GPU time: {:.2}ms ({:.1}% of total time)",
                    total_gpu_time_ms,
                    (total_gpu_time_ms / total_elapsed_ms) * 100.0
                )
                .into(),
            );

            web_sys::console::log_1(&"[jv_vm] === Prover Internal Timing ===".into());
            print_timing!("INTT (interpolation)", total_timing.intt_ms);
            print_timing!("Collapse (CT addition)", total_timing.collapse_ms);
            print_timing!("Poly division (LPZK)", total_timing.poly_div_ms);
            print_timing!("Open (IT-PAC eval)", total_timing.open_ms);
            print_timing!("MK poly (interpolation)", total_timing.mk_poly_ms);
            print_timing!("IT-PAC creation", total_timing.itpac_ms);
            print_timing!("Packing (2-way)", total_timing.packing_ms);
            print_timing!("Setup (circuit eval)", total_timing.setup_ms);
            print_timing!("Disclose (topology)", total_timing.disclose_ms);
            print_timing!("Commit soldering", total_timing.commit_soldering_ms);
            print_timing!("LPZK poly accumulation", total_timing.lpzk_accumulation_ms);
            print_timing!("Reveal soldering", total_timing.reveal_soldering_ms);
            print_timing!("MK binary eval", total_timing.mk_binary_ms);
            print_timing!("MK sum", total_timing.mk_sum_ms);
            print_timing!("Open polynomial", total_timing.open_polynomial_ms);
            print_timing!("MK commit VOLE", total_timing.mk_commit_vole_ms);
            print_timing!("MK commit packing", total_timing.mk_commit_packing_ms);

            web_sys::console::log_1(&"[jv_vm] === Call-Level Timing (should sum to ~100%) ===".into());
            print_timing!("VOLE pool generation", total_timing.vole_pool_ms);
            print_timing!("Prover::new", total_timing.prover_new_ms);
            print_timing!("Prover::setup", total_timing.prover_setup_ms);
            print_timing!("Prover::setup_soldering", total_timing.prover_setup_soldering_ms);
            print_timing!("Prover::commit", total_timing.commit_ms);
            print_timing!("Prover::commit_mk_polynomials", total_timing.commit_mk_poly_ms);
            print_timing!("Prover::commit_soldering", total_timing.commit_soldering_call_ms);
            print_timing!("Prover::disclose", total_timing.disclose_call_ms);
            print_timing!("Prover::reveal_soldering", total_timing.reveal_soldering_call_ms);
            print_timing!("Prover::open", total_timing.open_call_ms);
            print_timing!("Prover::open_itpac", total_timing.open_itpac_ms);
            print_timing!("Prover::prove_multiplications", total_timing.prove_mults_ms);

            let total_call_time = total_timing.vole_pool_ms + total_timing.prover_new_ms +
                                  total_timing.prover_setup_ms + total_timing.prover_setup_soldering_ms +
                                  total_timing.commit_ms + total_timing.commit_mk_poly_ms +
                                  total_timing.commit_soldering_call_ms + total_timing.disclose_call_ms +
                                  total_timing.reveal_soldering_call_ms + total_timing.open_call_ms +
                                  total_timing.open_itpac_ms + total_timing.prove_mults_ms;
            let loop_overhead = total_elapsed_ms - total_call_time;

            web_sys::console::log_1(&"[jv_vm] === Summary ===".into());
            print_timing!("Total call time", total_call_time);
            print_timing!("Loop/other overhead", loop_overhead);

            Ok(BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: total_mults,
            })
        },
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
