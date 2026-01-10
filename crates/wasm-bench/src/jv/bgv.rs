//! BGV homomorphic encryption benchmark for WASM.
//!
//! JustVengers pattern benchmark:
//! 1. Copy main CT NUM_COPIES times
//! 2. Slot-wise multiply each copy with random field coefficients
//! 3. Add all multiplied copies together
//! 4. sum_slots: sum all 8192 slots (13 rotations)
//! 5. mask: zero out all slots except slot 1
//! 6. add: add 80 pre-prepared ciphertexts

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_justvengers_core::{
    RnsBgvParams, RnsCiphertext, RnsKeyPair, RnsGaloisKeys, GOLDILOCKS,
};

#[cfg(target_arch = "wasm32")]
use mpz_core::{prg::Prg, Block};

#[cfg(target_arch = "wasm32")]
use rand::{Rng, SeedableRng};

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
const NUM_COPIES: usize = 10;

#[cfg(target_arch = "wasm32")]
const NUM_ADDITIONAL_CTS: usize = 80;

/// Pre-computed data for BGV benchmark (generated once, reused across iterations).
#[cfg(target_arch = "wasm32")]
struct BgvBenchData {
    ct_main: RnsCiphertext,
    galois_keys: RnsGaloisKeys,
    all_coeffs: Vec<Vec<u64>>,
    mask: Vec<u64>,
    additional_cts: Vec<RnsCiphertext>,
}

/// Generates all benchmark data (not timed).
#[cfg(target_arch = "wasm32")]
fn setup_bgv_bench() -> BgvBenchData {
    let mut rng = Prg::from_seed(Block::ZERO);
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    let n = params.n;
    let t = GOLDILOCKS;

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

    BgvBenchData {
        ct_main,
        galois_keys,
        all_coeffs,
        mask,
        additional_cts,
    }
}

/// Runs a single iteration of the JustVengers BGV pattern (sequential).
#[cfg(target_arch = "wasm32")]
fn run_bgv_iteration(data: &BgvBenchData) {
    // Copy and slot-wise multiply each copy with its coefficients
    let multiplied: Vec<RnsCiphertext> = data.all_coeffs.iter()
        .map(|coeffs| data.ct_main.clone().mul_plaintext_slots(coeffs))
        .collect();

    // Add all multiplied copies together
    let ct_combined = multiplied.iter().skip(1)
        .fold(multiplied[0].clone(), |acc, ct| acc.add(ct));

    // Sum all slots (sequential)
    let summed = ct_combined.sum_slots(&data.galois_keys);

    // Mask to keep only slot 1
    let masked = summed.mul_plaintext_slots(&data.mask);

    // Add all additional ciphertexts
    let mut result = masked;
    for ct_add in data.additional_cts.iter() {
        result = result.add(ct_add);
    }

    // Prevent optimization
    std::hint::black_box(result);
}

/// Runs a single iteration of the JustVengers BGV pattern (parallel sum_slots).
#[cfg(target_arch = "wasm32")]
fn run_bgv_iteration_parallel(data: &BgvBenchData) {
    // Copy and slot-wise multiply each copy with its coefficients
    let multiplied: Vec<RnsCiphertext> = data.all_coeffs.iter()
        .map(|coeffs| data.ct_main.clone().mul_plaintext_slots(coeffs))
        .collect();

    // Add all multiplied copies together
    let ct_combined = multiplied.iter().skip(1)
        .fold(multiplied[0].clone(), |acc, ct| acc.add(ct));

    // Sum all slots (parallel key-switching)
    let summed = ct_combined.sum_slots_parallel(&data.galois_keys);

    // Mask to keep only slot 1
    let masked = summed.mul_plaintext_slots(&data.mask);

    // Add all additional ciphertexts
    let mut result = masked;
    for ct_add in data.additional_cts.iter() {
        result = result.add(ct_add);
    }

    // Prevent optimization
    std::hint::black_box(result);
}

/// Benchmark JustVengers BGV pattern in WASM (sequential).
///
/// Pattern: NUM_COPIES x (copy + slot-wise mult) + sum_slots + mask + 80 CT additions
///
/// # Arguments
/// * `n` - Number of benchmark iterations
///
/// # Returns
/// BenchResult with elapsed_ms
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn bgv_justvengers_pattern(n: u32) -> Result<BenchResult, JsValue> {
    let global = js_sys::global();
    let performance: web_sys::Performance =
        js_sys::Reflect::get(&global, &"performance".into())
            .map_err(|_| JsValue::from_str("performance not available"))?
            .unchecked_into();

    web_sys::console::log_1(
        &format!(
            "[bgv] Setting up: {} copies, {} additional CTs, 8192 slots, 5 RNS moduli",
            NUM_COPIES, NUM_ADDITIONAL_CTS
        ).into(),
    );

    let setup_start = performance.now();
    let data = setup_bgv_bench();
    let setup_time = performance.now() - setup_start;

    web_sys::console::log_1(
        &format!("[bgv] Setup complete in {:.2}ms, starting {} iterations", setup_time, n).into(),
    );

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        let start = performance.now();
        run_bgv_iteration(&data);
        total_elapsed_ms += performance.now() - start;

        if (i + 1) % 5 == 0 || i == 0 {
            web_sys::console::log_1(
                &format!(
                    "[bgv] Iteration {}/{} done, avg {:.2}ms/iter",
                    i + 1, n, total_elapsed_ms / (i + 1) as f64
                ).into(),
            );
        }
    }

    web_sys::console::log_1(
        &format!(
            "[bgv] Done: {:.2}ms total, {:.2}ms/iter",
            total_elapsed_ms,
            total_elapsed_ms / n as f64
        ).into(),
    );

    Ok(BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: 0, // Not applicable for BGV
    })
}

/// Benchmark JustVengers BGV pattern in WASM with parallel key-switching.
///
/// Uses rayon thread pool in a web worker for parallel sum_slots.
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `concurrency` - Number of threads for rayon pool
///
/// # Returns
/// BenchResult with elapsed_ms
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn bgv_justvengers_pattern_parallel(n: u32, concurrency: u32) -> BenchResult {
    use std::sync::Mutex;
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let _handle = web_spawn::spawn(move || {
        // Create a local thread pool for rayon
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrency as usize)
            .spawn_handler(|thread| {
                let _ = web_spawn::spawn(move || thread.run());
                Ok(())
            })
            .build()
            .expect("failed to build rayon pool");

        let bench_result = pool.install(|| {
            let global = js_sys::global();
            let performance: web_sys::Performance =
                js_sys::Reflect::get(&global, &"performance".into())
                    .expect("performance should exist")
                    .unchecked_into();

            web_sys::console::log_1(
                &format!(
                    "[bgv-parallel] Setting up: {} copies, {} additional CTs, 8192 slots, 5 RNS moduli, {} threads",
                    NUM_COPIES, NUM_ADDITIONAL_CTS, concurrency
                ).into(),
            );

            let setup_start = performance.now();
            let data = setup_bgv_bench();
            let setup_time = performance.now() - setup_start;

            web_sys::console::log_1(
                &format!("[bgv-parallel] Setup complete in {:.2}ms, starting {} iterations", setup_time, n).into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                let start = performance.now();
                run_bgv_iteration_parallel(&data);
                total_elapsed_ms += performance.now() - start;

                if (i + 1) % 5 == 0 || i == 0 {
                    web_sys::console::log_1(
                        &format!(
                            "[bgv-parallel] Iteration {}/{} done, avg {:.2}ms/iter",
                            i + 1, n, total_elapsed_ms / (i + 1) as f64
                        ).into(),
                    );
                }
            }

            web_sys::console::log_1(
                &format!(
                    "[bgv-parallel] Done: {:.2}ms total, {:.2}ms/iter",
                    total_elapsed_ms,
                    total_elapsed_ms / n as f64
                ).into(),
            );

            BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: 0,
            }
        });

        *result_clone.lock().unwrap() = Some(bench_result);
    });

    // Poll for result from main thread
    loop {
        JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
            .await
            .unwrap();
        if let Some(r) = result.lock().unwrap().take() {
            return r;
        }
        // Small delay before next poll
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }
}
