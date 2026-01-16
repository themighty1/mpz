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

/// Get GPU adapter info - useful for debugging WebGPU support
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn get_gpu_info() -> Result<String, JsValue> {
    use wgpu;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| JsValue::from_str("No GPU adapter found"))?;

    let info = adapter.get_info();
    let limits = adapter.limits();

    let result = format!(
        r#"GPU Adapter Info:
  Name: {}
  Backend: {:?}
  DeviceType: {:?}
  Vendor: 0x{:04X}
  Device: 0x{:04X}

Limits:
  maxComputeWorkgroupStorageSize: {} bytes ({} KB)
  maxStorageBufferBindingSize: {} bytes ({} MB)
  maxComputeWorkgroupSizeX: {}
  maxComputeWorkgroupSizeY: {}
  maxComputeWorkgroupSizeZ: {}
  maxComputeInvocationsPerWorkgroup: {}
  maxComputeWorkgroupsPerDimension: {}

DeviceType Legend:
  Other = Unknown
  IntegratedGpu = Integrated GPU (shared memory)
  DiscreteGpu = Discrete GPU (dedicated memory)
  VirtualGpu = Virtual/Hosted GPU
  Cpu = Software fallback (NO HARDWARE GPU!)"#,
        info.name,
        info.backend,
        info.device_type,
        info.vendor,
        info.device,
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_workgroup_storage_size / 1024,
        limits.max_storage_buffer_binding_size,
        limits.max_storage_buffer_binding_size / (1024 * 1024),
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroups_per_dimension,
    );

    // Also log to console
    web_sys::console::log_1(&result.clone().into());

    Ok(result)
}
