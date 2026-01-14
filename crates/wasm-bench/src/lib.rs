//! WASM benchmarks for mpz libraries.
//!
//! This crate exposes benchmarks as WASM-callable functions
//! for browser performance testing.
//!
//! Modules:
//! - `garble`: Garbled circuits benchmarks (core + protocol)
//! - `zk`: QuickSilver ZK benchmarks (core + protocol + prover/verifier)
//! - `ot`: Oblivious transfer benchmarks (Ferret)

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// NOTE: Only compiling jv module (no rayon/web_spawn dependency)
// garble, ot, zk modules disabled (they require rayon/web_spawn)
#[cfg(target_arch = "wasm32")]
mod jv;

// Re-export jv functions only
#[cfg(target_arch = "wasm32")]
pub use jv::*;

/// Common benchmark result containing timing and work done.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BenchResult {
    pub elapsed_ms: f64,
    pub and_gates: u64,
}

// NOTE: init_thread_pool and test_mt_context_only removed
// We're not using rayon/web_spawn for jv_vm_prover_main_thread (GPU on main thread)
