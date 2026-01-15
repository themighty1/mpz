//! BGV homomorphic encryption benchmark for WASM with WebGPU-accelerated sum_slots.
//!
//! JustVengers pattern benchmark with GPU acceleration:
//! 1. CPU: Copy main CT NUM_COPIES times
//! 2. CPU: Slot-wise multiply each copy with random field coefficients
//! 3. CPU: Add all multiplied copies together
//! 4. **GPU: sum_slots via WebGPU** (13 rotations with key-switching)
//! 5. CPU: mask: zero out all slots except slot 1
//! 6. CPU: add: add 80 pre-prepared ciphertexts

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers_core::{
    RnsBgvParams, RnsCiphertext, RnsKeyPair, RnsGaloisKeys, GOLDILOCKS,
};

#[cfg(target_arch = "wasm32")]
use bgv_webgpu::{
    GpuRotationContext, GpuRnsParams, GpuRnsCiphertext, GpuGaloisKeys, GpuGaloisKey,
    rotation_exponent,
    old_code::{gpu_sum_slots_batched, SumSlotsWorkspace},
};

#[cfg(target_arch = "wasm32")]
use mpz_core::{prg::Prg, Block};

#[cfg(target_arch = "wasm32")]
use rand::{Rng, SeedableRng};

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
const NUM_COPIES: usize = 10;

#[cfg(target_arch = "wasm32")]
const NUM_ADDITIONAL_CTS: usize = 80;

/// Pre-computed data for BGV benchmark with WebGPU.
#[cfg(target_arch = "wasm32")]
struct BgvWebGpuBenchData {
    // CPU-side data
    ct_main: RnsCiphertext,
    all_coeffs: Vec<Vec<u64>>,
    mask: Vec<u64>,
    additional_cts: Vec<RnsCiphertext>,
    // GPU-side data
    gpu_ctx: GpuRotationContext,
    gpu_galois_keys: GpuGaloisKeys,
    gpu_workspace: SumSlotsWorkspace,
}

/// Converts CPU RnsCiphertext to GPU format.
#[cfg(target_arch = "wasm32")]
fn cpu_ct_to_gpu(ctx: &GpuRotationContext, ct: &RnsCiphertext) -> Result<GpuRnsCiphertext, JsValue> {
    // Extract residues from CPU ciphertext
    let c0_residues: Vec<Vec<u64>> = ct.c0().residues().iter().cloned().collect();
    let c1_residues: Vec<Vec<u64>> = ct.c1().residues().iter().cloned().collect();

    GpuRnsCiphertext::from_residues(ctx, &c0_residues, &c1_residues)
        .map_err(|e| JsValue::from_str(&format!("GPU ciphertext creation failed: {:?}", e)))
}

/// Converts GPU RnsCiphertext back to CPU format (async for WASM).
#[cfg(target_arch = "wasm32")]
async fn gpu_ct_to_cpu(
    ctx: &GpuRotationContext,
    gpu_ct: &GpuRnsCiphertext,
    template: &RnsCiphertext,
) -> Result<RnsCiphertext, JsValue> {
    // Read back residues from GPU (async)
    let c0_residues = gpu_ct.c0.to_residues_async(ctx).await
        .map_err(|e| JsValue::from_str(&format!("GPU read failed: {:?}", e)))?;
    let c1_residues = gpu_ct.c1.to_residues_async(ctx).await
        .map_err(|e| JsValue::from_str(&format!("GPU read failed: {:?}", e)))?;

    // Create new CPU ciphertext with GPU results
    Ok(RnsCiphertext::from_residues(
        c0_residues,
        c1_residues,
        template.rns_params().clone(),
        template.bgv_params().clone(),
    ))
}

/// Converts CPU Galois keys to GPU format.
#[cfg(target_arch = "wasm32")]
fn cpu_galois_keys_to_gpu(
    ctx: &GpuRotationContext,
    cpu_keys: &RnsGaloisKeys,
    n: usize,
) -> Result<GpuGaloisKeys, JsValue> {
    let log_n = (n as f64).log2() as usize;
    let mut gpu_keys = Vec::with_capacity(log_n);
    let digits_per_limb = ctx.params().digits_per_limb;
    let num_moduli = ctx.params().moduli.len();

    // Convert each Galois key
    for i in 0..log_n {
        let step = 1 << i;
        let k = rotation_exponent(step, n);

        if let Some(cpu_key) = cpu_keys.get_key(i) {
            // Extract key data from CPU key
            let keys_b = cpu_key.keys_b();
            let keys_a = cpu_key.keys_a();

            // Reshape for GPU: [limb][digit][modulus][coeff]
            let num_limbs = keys_b.len() / digits_per_limb;

            let mut keys_b_data = Vec::with_capacity(num_limbs);
            let mut keys_a_data = Vec::with_capacity(num_limbs);

            for limb_idx in 0..num_limbs {
                let mut limb_keys_b = Vec::with_capacity(digits_per_limb);
                let mut limb_keys_a = Vec::with_capacity(digits_per_limb);

                for digit_idx in 0..digits_per_limb {
                    let key_idx = limb_idx * digits_per_limb + digit_idx;

                    // Get residues for this key component
                    let residues_b: Vec<Vec<u64>> = keys_b[key_idx].residues()
                        .iter()
                        .cloned()
                        .collect();
                    let residues_a: Vec<Vec<u64>> = keys_a[key_idx].residues()
                        .iter()
                        .cloned()
                        .collect();

                    limb_keys_b.push(residues_b);
                    limb_keys_a.push(residues_a);
                }

                keys_b_data.push(limb_keys_b);
                keys_a_data.push(limb_keys_a);
            }

            let gpu_key = GpuGaloisKey::from_coefficients(ctx, k, &keys_b_data, &keys_a_data)
                .map_err(|e| JsValue::from_str(&format!("GPU key creation failed: {:?}", e)))?;

            gpu_keys.push(gpu_key);
        }
    }

    // Add conjugation key if present
    if let Some(conj_key) = cpu_keys.get_conjugation_key() {
        let k = 2 * n - 1; // Conjugation exponent

        let keys_b = conj_key.keys_b();
        let keys_a = conj_key.keys_a();

        let num_limbs = keys_b.len() / digits_per_limb;

        let mut keys_b_data = Vec::with_capacity(num_limbs);
        let mut keys_a_data = Vec::with_capacity(num_limbs);

        for limb_idx in 0..num_limbs {
            let mut limb_keys_b = Vec::with_capacity(digits_per_limb);
            let mut limb_keys_a = Vec::with_capacity(digits_per_limb);

            for digit_idx in 0..digits_per_limb {
                let key_idx = limb_idx * digits_per_limb + digit_idx;

                let residues_b: Vec<Vec<u64>> = keys_b[key_idx].residues()
                    .iter()
                    .cloned()
                    .collect();
                let residues_a: Vec<Vec<u64>> = keys_a[key_idx].residues()
                    .iter()
                    .cloned()
                    .collect();

                limb_keys_b.push(residues_b);
                limb_keys_a.push(residues_a);
            }

            keys_b_data.push(limb_keys_b);
            keys_a_data.push(limb_keys_a);
        }

        let gpu_key = GpuGaloisKey::from_coefficients(ctx, k, &keys_b_data, &keys_a_data)
            .map_err(|e| JsValue::from_str(&format!("GPU conjugation key failed: {:?}", e)))?;

        gpu_keys.push(gpu_key);
    }

    Ok(GpuGaloisKeys { keys: gpu_keys })
}

/// Check if WebGPU is available in the browser.
#[cfg(target_arch = "wasm32")]
fn check_webgpu_available() -> Result<(), JsValue> {
    let global = js_sys::global();
    let navigator = js_sys::Reflect::get(&global, &"navigator".into())
        .map_err(|_| JsValue::from_str("navigator not available"))?;

    let gpu = js_sys::Reflect::get(&navigator, &"gpu".into())
        .map_err(|_| JsValue::from_str("WebGPU not available: navigator.gpu is undefined"))?;

    if gpu.is_undefined() || gpu.is_null() {
        return Err(JsValue::from_str(
            "WebGPU is NOT available in this browser. \
             Please use a browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly with flags). \
             For headless Chrome, use --enable-unsafe-webgpu and --headless=new flags."
        ));
    }

    web_sys::console::log_1(&"[bgv-webgpu] WebGPU is available!".into());
    Ok(())
}

/// Generates all benchmark data (not timed).
#[cfg(target_arch = "wasm32")]
async fn setup_bgv_webgpu_bench() -> Result<BgvWebGpuBenchData, JsValue> {
    // First check if WebGPU is available
    check_webgpu_available()?;

    let mut rng = Prg::from_seed(Block::ZERO);
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let cpu_galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;
    let t = GOLDILOCKS;

    web_sys::console::log_1(&"[bgv-webgpu] Creating GPU context...".into());

    // Create GPU context (async for WASM)
    let gpu_params = GpuRnsParams::goldilocks();
    let gpu_ctx = GpuRotationContext::new_async(gpu_params).await
        .map_err(|e| {
            let msg = format!(
                "GPU context creation failed: {:?}. \
                 This may indicate WebGPU adapter not found or GPU not available in headless mode.",
                e
            );
            web_sys::console::error_1(&msg.clone().into());
            JsValue::from_str(&msg)
        })?;

    web_sys::console::log_1(&"[bgv-webgpu] Converting Galois keys to GPU...".into());

    // Convert Galois keys to GPU format
    let gpu_galois_keys = cpu_galois_keys_to_gpu(&gpu_ctx, &cpu_galois_keys, n)?;

    // Create GPU workspace
    let gpu_workspace = SumSlotsWorkspace::new(&gpu_ctx);

    web_sys::console::log_1(&"[bgv-webgpu] Generating CPU data...".into());

    // Create slot values
    let slots: Vec<u64> = (0..n).map(|i| (i % 100 + 1) as u64).collect();

    // Generate coefficients for each copy
    let all_coeffs: Vec<Vec<u64>> = (0..NUM_COPIES)
        .map(|_| (0..n).map(|_| rng.random::<u64>() % t).collect())
        .collect();

    // Main CT
    let ct_main = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

    // Mask: 1 in slot 1, 0 elsewhere
    let mut mask = vec![0u64; n];
    mask[1] = 1;

    // Pre-create additional ciphertexts
    let mut additional_cts = Vec::with_capacity(NUM_ADDITIONAL_CTS);
    for slot_idx in 2..=(1 + NUM_ADDITIONAL_CTS) {
        let value = (slot_idx * 1000 + 123) as u64;
        let mut slot_vals = vec![0u64; n];
        slot_vals[slot_idx] = value;
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slot_vals, &mut rng);
        additional_cts.push(ct);
    }

    Ok(BgvWebGpuBenchData {
        ct_main,
        all_coeffs,
        mask,
        additional_cts,
        gpu_ctx,
        gpu_galois_keys,
        gpu_workspace,
    })
}

/// Runs a single iteration with WebGPU-accelerated sum_slots.
#[cfg(target_arch = "wasm32")]
async fn run_bgv_webgpu_iteration(data: &BgvWebGpuBenchData) -> Result<(), JsValue> {
    // CPU: Copy and slot-wise multiply each copy with its coefficients
    let multiplied: Vec<RnsCiphertext> = data.all_coeffs.iter()
        .map(|coeffs| data.ct_main.clone().mul_plaintext_slots(coeffs))
        .collect();

    // CPU: Add all multiplied copies together
    let ct_combined = multiplied.iter().skip(1)
        .fold(multiplied[0].clone(), |acc, ct| acc.add(ct));

    // GPU: Convert to GPU format
    let gpu_ct = cpu_ct_to_gpu(&data.gpu_ctx, &ct_combined)?;

    // GPU: sum_slots using WebGPU
    let gpu_summed = gpu_sum_slots_batched(
        &data.gpu_ctx,
        &gpu_ct,
        &data.gpu_galois_keys,
        &data.gpu_workspace,
    ).map_err(|e| JsValue::from_str(&format!("GPU sum_slots failed: {:?}", e)))?;

    // GPU -> CPU: Convert result back (async)
    let summed = gpu_ct_to_cpu(&data.gpu_ctx, &gpu_summed, &ct_combined).await?;

    // CPU: Mask to keep only slot 1
    let masked = summed.mul_plaintext_slots(&data.mask);

    // CPU: Add all additional ciphertexts
    let mut result = masked;
    for ct_add in data.additional_cts.iter() {
        result = result.add(ct_add);
    }

    // Prevent optimization
    std::hint::black_box(result);

    Ok(())
}

/// Benchmark JustVengers BGV pattern with WebGPU-accelerated sum_slots.
///
/// Pattern: CPU (copies + mults + adds) + GPU sum_slots + CPU (mask + additions)
///
/// # Arguments
/// * `n` - Number of benchmark iterations
///
/// # Returns
/// BenchResult with elapsed_ms
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn bgv_justvengers_pattern_webgpu(n: u32) -> Result<BenchResult, JsValue> {
    let global = js_sys::global();
    let performance: web_sys::Performance =
        js_sys::Reflect::get(&global, &"performance".into())
            .map_err(|_| JsValue::from_str("performance not available"))?
            .unchecked_into();

    web_sys::console::log_1(
        &format!(
            "[bgv-webgpu] Setting up: {} copies, {} additional CTs, 8192 slots, WebGPU sum_slots",
            NUM_COPIES, NUM_ADDITIONAL_CTS
        ).into(),
    );

    let setup_start = performance.now();
    let data = setup_bgv_webgpu_bench().await?;
    let setup_time = performance.now() - setup_start;

    web_sys::console::log_1(
        &format!("[bgv-webgpu] Setup complete in {:.2}ms, starting {} iterations", setup_time, n).into(),
    );

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        let start = performance.now();
        run_bgv_webgpu_iteration(&data).await?;
        total_elapsed_ms += performance.now() - start;

        if (i + 1) % 5 == 0 || i == 0 {
            web_sys::console::log_1(
                &format!(
                    "[bgv-webgpu] Iteration {}/{} done, avg {:.2}ms/iter",
                    i + 1, n, total_elapsed_ms / (i + 1) as f64
                ).into(),
            );
        }
    }

    web_sys::console::log_1(
        &format!(
            "[bgv-webgpu] Done: {:.2}ms total, {:.2}ms/iter",
            total_elapsed_ms,
            total_elapsed_ms / n as f64
        ).into(),
    );

    Ok(BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: 0,
    })
}

/// Test version: runs bgv_webgpu benchmark directly (no worker).
/// This tests if async GPU code works at all before trying workers.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn bgv_webgpu_worker_test(n: u32) -> Result<BenchResult, JsValue> {
    web_sys::console::log_1(&"[bgv-webgpu-worker-test] Running directly (no worker)...".into());

    // Just call the same benchmark code directly
    run_bgv_worker_bench(n).await
        .map_err(|e| JsValue::from_str(&e))
}

/// Inner async benchmark for worker test.
#[cfg(target_arch = "wasm32")]
async fn run_bgv_worker_bench(n: u32) -> Result<BenchResult, String> {
    let global = js_sys::global();
    let performance: web_sys::Performance =
        js_sys::Reflect::get(&global, &"performance".into())
            .map_err(|_| "performance not available".to_string())?
            .unchecked_into();

    web_sys::console::log_1(&"[bgv-worker-bench] Setting up GPU context...".into());

    let setup_start = performance.now();
    let data = setup_bgv_webgpu_bench().await
        .map_err(|e| format!("Setup failed: {:?}", e))?;
    let setup_time = performance.now() - setup_start;

    web_sys::console::log_1(
        &format!("[bgv-worker-bench] Setup done in {:.2}ms, running {} iterations", setup_time, n).into(),
    );

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        let start = performance.now();
        run_bgv_webgpu_iteration(&data).await
            .map_err(|e| format!("Iteration failed: {:?}", e))?;
        total_elapsed_ms += performance.now() - start;

        if (i + 1) % 5 == 0 || i == 0 {
            web_sys::console::log_1(
                &format!("[bgv-worker-bench] Iteration {}/{} done", i + 1, n).into(),
            );
        }
    }

    web_sys::console::log_1(
        &format!("[bgv-worker-bench] Done: {:.2}ms total", total_elapsed_ms).into(),
    );

    Ok(BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: 0,
    })
}
