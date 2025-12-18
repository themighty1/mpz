//! Miscellaneous WASM benchmarks.
//!
//! This crate is for quick iteration on new benchmarks without the
//! compilation overhead of wasm-bench.

mod aes_compare;
mod gf128_compare;
mod gf128_polyval;
mod gf128_zig;

pub use aes_compare::*;
pub use gf128_compare::*;
pub use gf128_polyval::*;
pub use gf128_zig::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Benchmark result returned to JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct BenchResult {
    pub elapsed_ms: f64,
    pub and_gates: u64,
}

#[cfg(target_arch = "wasm32")]
impl BenchResult {
    pub fn new(elapsed_ms: f64, and_gates: u64) -> Self {
        Self { elapsed_ms, and_gates }
    }
}

/// Initialize the web_spawn spawner and rayon thread pool for MT benchmarks.
/// Must be called before running any MT benchmarks.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init_thread_pool(thread_count: usize) -> Result<(), JsValue> {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Arc;
    use wasm_bindgen_futures::JsFuture;

    const INIT_PENDING: u8 = 0;
    const INIT_SUCCESS: u8 = 1;
    const INIT_FAILED: u8 = 2;

    web_sys::console::log_1(&"[rust] init_thread_pool: starting web_spawn spawner...".into());

    // Check if SharedArrayBuffer is available (requires COOP/COEP headers)
    let sab_available = ::js_sys::Reflect::has(&::js_sys::global(), &"SharedArrayBuffer".into())
        .unwrap_or(false);
    web_sys::console::log_1(&format!("[rust] SharedArrayBuffer available: {}", sab_available).into());

    if !sab_available {
        return Err(JsValue::from_str("SharedArrayBuffer not available - check COOP/COEP headers"));
    }

    // Initialize web_spawn spawner
    web_sys::console::log_1(&"[rust] Calling web_spawn::start_spawner()...".into());
    JsFuture::from(web_spawn::start_spawner()).await?;

    web_sys::console::log_1(&"[rust] init_thread_pool: web_spawn spawner ready".into());
    web_sys::console::log_1(&format!("[rust] init_thread_pool: building rayon pool with {} threads in worker...", thread_count).into());

    // Initialize rayon in a worker thread (Atomics.wait is allowed there)
    let init_status = Arc::new(AtomicU8::new(INIT_PENDING));
    let init_status_clone = init_status.clone();

    web_spawn::spawn(move || {
        web_sys::console::log_1(&"[rust] worker: starting rayon init...".into());
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .spawn_handler(|thread| {
                web_sys::console::log_1(&"[rust] rayon spawn_handler called".into());
                let _ = web_spawn::spawn(move || thread.run());
                Ok(())
            })
            .build_global();

        match result {
            Ok(_) => {
                web_sys::console::log_1(&"[rust] worker: rayon init success".into());
                init_status_clone.store(INIT_SUCCESS, Ordering::SeqCst);
            }
            Err(e) => {
                web_sys::console::log_1(&format!("[rust] worker: rayon init failed: {}", e).into());
                init_status_clone.store(INIT_FAILED, Ordering::SeqCst);
            }
        }
    });

    // Poll for completion (non-blocking on main thread)
    loop {
        match init_status.load(Ordering::SeqCst) {
            INIT_SUCCESS => {
                web_sys::console::log_1(&"[rust] init_thread_pool: complete".into());
                return Ok(());
            }
            INIT_FAILED => {
                return Err(JsValue::from_str("rayon thread pool initialization failed"));
            }
            _ => {
                // Yield to event loop
                JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await?;
            }
        }
    }
}
