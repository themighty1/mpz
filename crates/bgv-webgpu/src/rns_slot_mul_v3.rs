//! Four-Step NTT RNS Slot Multiplication V3 for BGV using WebGPU.
//!
//! This version uses the four-step (Cooley-Tukey) NTT algorithm which decomposes
//! a large N-point NTT (N = n1 × n2) into:
//!   - n1 independent n2-point NTTs (row NTTs)
//!   - Twiddle factor multiplication
//!   - n2 independent n1-point NTTs (column NTTs)
//!
//! For N=8192 = 32 × 256:
//!   - n1 = 32 rows, n2 = 256 columns
//!   - Row NTTs: 32 independent 256-point NTTs (8 stages, barriers)
//!   - Cross twiddles: multiply by omega^(row * col)
//!   - Column NTTs: 256 independent 32-point NTTs (5 stages, barriers)
//!
//! Benefits:
//!   - Better GPU occupancy: 128K workgroups vs 400 for V2
//!   - Fewer barriers per workgroup (8 or 5 vs 13)
//!   - Better fit for GPU hardware (smaller shared memory per workgroup)
//!
//! Trade-offs:
//!   - 3x global memory round-trips vs shared-memory-only in V2
//!   - More kernel dispatches
//!
//! # Status: Work-in-Progress
//!
//! The four-step NTT infrastructure is complete (buffers, pipelines, bind groups),
//! but the numerical output does not yet match V2. Further debugging needed.
//!
//! Actual order implemented:
//!   - Forward: col_ntt → cross_twiddle → row_ntt
//!   - Inverse: row_intt → cross_inv → col_intt

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, BindGroup, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;
use crate::rns_slot_mul::RnsBatchParams;

#[cfg(target_arch = "wasm32")]
use futures::TryFutureExt;

/// Uniform buffer for batch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
}

/// Uniform buffer for multi-CT batch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBatchParamsMultiCt {
    n: u32,
    n1: u32,              // 32 for four-step NTT
    n2: u32,              // 256 for four-step NTT
    num_batches: u32,
    num_moduli: u32,
    num_cts: u32,
    batches_per_ct: u32,
    _pad: u32,
}

/// Uniform buffer for modulus data (per-modulus).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform buffer for four-step NTT parameters.
/// NOTE: Different shaders interpret this differently:
/// - Row NTT/INTT shaders: expect log_n2 (8) at position 3
/// - Col NTT/INTT shaders: expect log_n1 (5) at position 3
/// So we need separate params buffers or modify shaders.
/// For now, we add both log values.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFourStepParams {
    n: u32,           // Total size (8192)
    n1: u32,          // Number of rows (32)
    n2: u32,          // Number of columns (256)
    log_n2: u32,      // log2(n2) = 8 (for row NTT/INTT)
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,      // log2(n1) = 5 (for col NTT/INTT)
    _pad: u32,
}

/// Maximum number of batches for pre-allocated buffers.
const MAX_SLOT_MUL_BATCHES: usize = 512;

/// GPU context for Four-Step NTT RNS batched slot multiplication.
///
/// Four-step decomposition for N=8192 = 32 × 256:
///   Forward NTT: twist → row_ntt (256-pt) → cross_twiddle → col_ntt (32-pt)
///   Inverse NTT: col_intt (32-pt) → cross_twiddle_inv → row_intt (256-pt) → untwist
///
/// Key benefits:
/// 1. Better GPU occupancy: many more workgroups
/// 2. Fewer barriers per workgroup (8 for rows, 5 for cols vs 13 for full)
/// 3. Smaller shared memory per workgroup (256 or 32 elements vs 8192)
///
/// Trade-offs:
/// - Multiple global memory round-trips between steps
/// - More kernel dispatches
pub struct RnsSlotMulGpuV3 {
    device: Device,
    queue: Queue,

    // Four-step slot encode pipelines (Goldilocks INTT decomposed)
    slot_encode_col_intt_pipeline: ComputePipeline,      // 32-point INTT on columns
    slot_encode_cross_twiddle_inv_pipeline: ComputePipeline,  // Cross-twiddle inverse
    slot_encode_row_intt_pipeline: ComputePipeline,      // 256-point INTT on rows + n^-1

    // Four-step NTT pipelines (RNS moduli)
    twist_pipeline: ComputePipeline,              // Coefficients → twisted + RNS expanded
    row_ntt_pipeline: ComputePipeline,            // 256-point NTTs (step 1)
    cross_twiddle_pipeline: ComputePipeline,      // Multiply by omega^(row*col) (step 2)
    col_ntt_pipeline: ComputePipeline,            // 32-point NTTs (step 3)
    // Inverse pipelines for INTT
    col_intt_pipeline: ComputePipeline,           // 32-point INTTs
    cross_twiddle_inv_pipeline: ComputePipeline,  // Multiply by omega_inv^(row*col)
    row_intt_pipeline: ComputePipeline,           // 256-point INTTs + untwist
    // Fused multiply kernel
    pointwise_mul_pipeline: ComputePipeline,      // pt * ct -> result

    // Parameters
    params: RnsBatchParams,
    n1: usize,  // 32 rows
    n2: usize,  // 256 columns

    // Precomputed buffers (uploaded once)
    plaintext_twiddles_buffer: Buffer,
    plaintext_params_buffer: Buffer,

    // Four-step twiddle buffers for all k moduli
    // Row NTT: uses omega_n2 = omega^(n/n2) = omega^32 (256th root)
    all_row_twiddles_buffer: Buffer,        // k * n2 * 2 u32s (omega_n2 powers)
    all_row_inv_twiddles_buffer: Buffer,    // k * n2 * 2 u32s (omega_n2_inv powers)
    // Col NTT: uses omega_n1 = omega^(n/n1) = omega^256 (32nd root)
    all_col_twiddles_buffer: Buffer,        // k * n1 * 2 u32s (omega_n1 powers)
    all_col_inv_twiddles_buffer: Buffer,    // k * n1 * 2 u32s (omega_n1_inv powers)
    // Cross twiddles: omega^(row * col) for row in 0..n1, col in 0..n2
    all_cross_twiddles_buffer: Buffer,      // k * n1 * n2 * 2 u32s
    all_cross_inv_twiddles_buffer: Buffer,  // k * n1 * n2 * 2 u32s
    // Psi powers for twist/untwist
    all_psi_buffer: Buffer,                 // k * n * 2 u32s (psi^i)
    all_psi_inv_buffer: Buffer,             // k * n * 2 u32s (psi_inv^i)
    // Modulus params
    all_rns_params_buffer: Buffer,          // k * 8 u32s (ModulusParams for all k)

    // Pre-allocated buffers
    preallocated_encoded_buffer: Buffer,
    preallocated_temp_buffer: Buffer,       // Intermediate buffer for 4-step (n * k * max_batches)
    preallocated_pt_ntt_buffer: Buffer,     // max_batches * k * n
    preallocated_out_c0_buffer: Buffer,
    preallocated_out_c1_buffer: Buffer,
    preallocated_batch_params_buffer: Buffer,       // 16 bytes (GpuBatchParams)
    preallocated_fourstep_params_buffer: Buffer,   // 32 bytes (GpuFourStepParams)
    preallocated_fused_params_buffer: Buffer,       // 32 bytes (GpuBatchParamsMultiCt)
    preallocated_slots_buffer: Buffer,
    preallocated_cts_c0_buffer: Buffer,
    preallocated_cts_c1_buffer: Buffer,

    // Goldilocks twiddle buffers for four-step slot encoding
    goldilocks_col_inv_twiddles_buffer: Buffer,   // 32 elements (zeta_inv^(256*i))
    goldilocks_row_inv_twiddles_buffer: Buffer,   // 256 elements (zeta_inv^(32*i))
    // Note: cross_inv_twiddles reuses plaintext_twiddles_buffer (zeta_inv^(row*col) = zeta_inv^idx)

    // Bind groups (one per step for flexibility)
    slot_encode_col_intt_bind_group: BindGroup,
    slot_encode_cross_twiddle_inv_bind_group: BindGroup,
    slot_encode_row_intt_bind_group: BindGroup,
    twist_bind_group: BindGroup,
    row_ntt_bind_group: BindGroup,
    cross_twiddle_bind_group: BindGroup,
    col_ntt_bind_group: BindGroup,
    pointwise_mul_bind_group: BindGroup,
    col_intt_bind_group: BindGroup,
    cross_twiddle_inv_bind_group: BindGroup,
    row_intt_bind_group: BindGroup,
    // Bind groups for c1 INTT (same pipelines, different data buffer)
    col_intt_c1_bind_group: BindGroup,
    cross_twiddle_inv_c1_bind_group: BindGroup,
    row_intt_c1_bind_group: BindGroup,
}

/// Four-step decomposition: n1 rows × n2 columns
/// For n=8192: n1=32, n2=256
const FOUR_STEP_N1: usize = 32;   // Number of rows
const FOUR_STEP_N2: usize = 256;  // Number of columns
const FOUR_STEP_LOG_N1: u32 = 5;  // log2(32)
const FOUR_STEP_LOG_N2: u32 = 8;  // log2(256)

impl std::fmt::Debug for RnsSlotMulGpuV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RnsSlotMulGpuV3")
            .field("n", &self.params.n)
            .field("k", &self.params.k)
            .field("n1", &self.n1)
            .field("n2", &self.n2)
            .finish_non_exhaustive()
    }
}

impl RnsSlotMulGpuV3 {
    /// Returns a reference to the GPU device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the GPU queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the batch parameters.
    pub fn params(&self) -> &RnsBatchParams {
        &self.params
    }
}

impl RnsSlotMulGpuV3 {
    /// Creates a new four-step NTT GPU context for RNS batched slot multiplication.
    ///
    /// NOTE: This implementation is a work-in-progress. The four-step NTT approach
    /// decomposes the 8192-point NTT into row (256-pt) and column (32-pt) NTTs
    /// for better GPU occupancy.
    pub fn new(params: RnsBatchParams) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(params))
    }

    /// Creates a new GPU context asynchronously (required for WASM).
    pub async fn new_async(params: RnsBatchParams) -> Result<Self, GpuError> {
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
            .ok_or(GpuError::AdapterNotFound)?;

        // Print GPU limits for debugging
        let limits = adapter.limits();
        let info = adapter.get_info();
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&format!(
                "[V3 GPU] Adapter: {} ({:?})",
                info.name, info.backend
            ).into());
            web_sys::console::log_1(&format!(
                "[V3 GPU] DeviceType: {:?} (Cpu = software fallback)",
                info.device_type
            ).into());
            web_sys::console::log_1(&format!(
                "[V3 GPU] maxComputeWorkgroupStorageSize: {} bytes ({} KB)",
                limits.max_compute_workgroup_storage_size,
                limits.max_compute_workgroup_storage_size / 1024
            ).into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!(
            "[V3 GPU] maxComputeWorkgroupStorageSize: {} bytes ({} KB)",
            limits.max_compute_workgroup_storage_size,
            limits.max_compute_workgroup_storage_size / 1024
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rns-slot-mul-v3 device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroups_per_dimension: 65535,
                        max_storage_buffer_binding_size: 1024 * 1024 * 512,
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Self::new_with_device(device, queue, params)
    }

    /// Creates context with existing device/queue.
    ///
    /// The four-step approach requires:
    /// 1. Separate twiddle buffers for row NTTs (omega_n2), col NTTs (omega_n1), and cross (omega^(r*c))
    /// 2. Separate pipelines for twist, row_ntt, cross_twiddle, col_ntt (forward)
    /// 3. Separate pipelines for col_intt, cross_twiddle_inv, row_intt_untwist (inverse)
    /// 4. Temporary buffers for intermediate results
    #[allow(unused_variables)]
    pub fn new_with_device(
        device: Device,
        queue: Queue,
        params: RnsBatchParams,
    ) -> Result<Self, GpuError> {
        let n = params.n;
        let k = params.k;
        let log_n = (n as u32).trailing_zeros();

        // Four-step decomposition constants
        let n1 = FOUR_STEP_N1;  // 32 rows
        let n2 = FOUR_STEP_N2;  // 256 columns
        assert_eq!(n1 * n2, n, "Four-step decomposition requires n = n1 * n2");

        // Compile shaders using naga_oil composition
        // Four-step slot encode shaders (Goldilocks INTT)
        let slot_encode_col_intt_shader = crate::shader_math::create_shader_module(
            &device,
            SLOT_ENCODE_COL_INTT_SHADER,
            "slot_encode_col_intt_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let slot_encode_cross_twiddle_inv_shader = crate::shader_math::create_shader_module(
            &device,
            SLOT_ENCODE_CROSS_TWIDDLE_INV_SHADER,
            "slot_encode_cross_twiddle_inv_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let slot_encode_row_intt_shader = crate::shader_math::create_shader_module(
            &device,
            SLOT_ENCODE_ROW_INTT_SHADER,
            "slot_encode_row_intt_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let twist_shader = crate::shader_math::create_shader_module(
            &device,
            TWIST_SHADER,
            "twist_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let row_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            ROW_NTT_SHADER,
            "row_ntt_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let cross_twiddle_shader = crate::shader_math::create_shader_module(
            &device,
            CROSS_TWIDDLE_SHADER,
            "cross_twiddle_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let col_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            COL_NTT_SHADER,
            "col_ntt_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let col_intt_shader = crate::shader_math::create_shader_module(
            &device,
            COL_INTT_SHADER,
            "col_intt_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let row_intt_shader = crate::shader_math::create_shader_module(
            &device,
            ROW_INTT_UNTWIST_SHADER,
            "row_intt_untwist_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let pointwise_mul_shader = crate::shader_math::create_shader_module(
            &device,
            POINTWISE_MUL_SHADER,
            "pointwise_mul_v3.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        // Create pipelines
        // Four-step slot encode pipelines
        let slot_encode_col_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_encode_col_intt_v3 pipeline"),
                layout: None,
                module: &slot_encode_col_intt_shader,
                entry_point: Some("slot_encode_col_intt"),
                compilation_options: Default::default(),
                cache: None,
            });

        let slot_encode_cross_twiddle_inv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_encode_cross_twiddle_inv_v3 pipeline"),
                layout: None,
                module: &slot_encode_cross_twiddle_inv_shader,
                entry_point: Some("slot_encode_cross_twiddle_inv"),
                compilation_options: Default::default(),
                cache: None,
            });

        let slot_encode_row_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_encode_row_intt_v3 pipeline"),
                layout: None,
                module: &slot_encode_row_intt_shader,
                entry_point: Some("slot_encode_row_intt"),
                compilation_options: Default::default(),
                cache: None,
            });

        let row_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("row_ntt_v3 pipeline"),
                layout: None,
                module: &row_ntt_shader,
                entry_point: Some("row_ntt"),
                compilation_options: Default::default(),
                cache: None,
            });

        let cross_twiddle_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cross_twiddle_v3 pipeline"),
                layout: None,
                module: &cross_twiddle_shader,
                entry_point: Some("cross_twiddle"),
                compilation_options: Default::default(),
                cache: None,
            });

        let col_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("col_ntt_v3 pipeline"),
                layout: None,
                module: &col_ntt_shader,
                entry_point: Some("col_ntt"),
                compilation_options: Default::default(),
                cache: None,
            });

        let col_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("col_intt_v3 pipeline"),
                layout: None,
                module: &col_intt_shader,
                entry_point: Some("col_intt"),
                compilation_options: Default::default(),
                cache: None,
            });

        let cross_twiddle_inv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cross_twiddle_inv_v3 pipeline"),
                layout: None,
                module: &cross_twiddle_shader,  // Same shader, different twiddles
                entry_point: Some("cross_twiddle"),
                compilation_options: Default::default(),
                cache: None,
            });

        let row_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("row_intt_untwist_v3 pipeline"),
                layout: None,
                module: &row_intt_shader,
                entry_point: Some("row_intt_untwist"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pointwise_mul_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pointwise_mul_v3 pipeline"),
                layout: None,
                module: &pointwise_mul_shader,
                entry_point: Some("pointwise_mul"),
                compilation_options: Default::default(),
                cache: None,
            });

        let twist_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("twist_v3 pipeline"),
                layout: None,
                module: &twist_shader,
                entry_point: Some("twist"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create plaintext parameter buffer (for slot_encode)
        let pt_data = &params.plaintext_data;
        let plaintext_params = GpuModulusParams {
            modulus_lo: pt_data.t as u32,
            modulus_hi: (pt_data.t >> 32) as u32,
            mu_lo: pt_data.mu as u32,
            mu_hi: (pt_data.mu >> 32) as u32,
            n_inv_lo: pt_data.n_inv as u32,
            n_inv_hi: (pt_data.n_inv >> 32) as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let plaintext_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plaintext_params_v3"),
            contents: bytemuck::bytes_of(&plaintext_params),
            usage: BufferUsages::UNIFORM,
        });

        // Create plaintext twiddle buffer (inverse powers for INTT in slot_encode)
        let pt_twiddles_flat: Vec<u32> = pt_data
            .zeta_inv_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let plaintext_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_twiddles_v3"),
                contents: bytemuck::cast_slice(&pt_twiddles_flat),
                usage: BufferUsages::STORAGE,
            });

        // Create Goldilocks twiddle buffers for four-step slot encoding
        // Col INTT: zeta_inv^(256*i) for i = 0..32 (32-point INTT twiddles)
        let mut goldilocks_col_inv_twiddles: Vec<u32> = Vec::with_capacity(n1 * 2);
        for i in 0..n1 {
            let idx = (256 * i) % n;
            let zeta_inv = pt_data.zeta_inv_powers[idx];
            goldilocks_col_inv_twiddles.push(zeta_inv as u32);
            goldilocks_col_inv_twiddles.push((zeta_inv >> 32) as u32);
        }
        let goldilocks_col_inv_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goldilocks_col_inv_twiddles_v3"),
                contents: bytemuck::cast_slice(&goldilocks_col_inv_twiddles),
                usage: BufferUsages::STORAGE,
            });

        // Row INTT: zeta_inv^(32*j) for j = 0..256 (256-point INTT twiddles)
        let mut goldilocks_row_inv_twiddles: Vec<u32> = Vec::with_capacity(n2 * 2);
        for j in 0..n2 {
            let idx = (32 * j) % n;
            let zeta_inv = pt_data.zeta_inv_powers[idx];
            goldilocks_row_inv_twiddles.push(zeta_inv as u32);
            goldilocks_row_inv_twiddles.push((zeta_inv >> 32) as u32);
        }
        let goldilocks_row_inv_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goldilocks_row_inv_twiddles_v3"),
                contents: bytemuck::cast_slice(&goldilocks_row_inv_twiddles),
                usage: BufferUsages::STORAGE,
            });

        // Create RNS modulus parameters buffer (all k moduli)
        let mut all_rns_params: Vec<GpuModulusParams> = Vec::with_capacity(k);
        for rns_data in params.rns_data.iter() {
            all_rns_params.push(GpuModulusParams {
                modulus_lo: rns_data.modulus as u32,
                modulus_hi: (rns_data.modulus >> 32) as u32,
                mu_lo: rns_data.mu as u32,
                mu_hi: (rns_data.mu >> 32) as u32,
                n_inv_lo: rns_data.n_inv as u32,
                n_inv_hi: (rns_data.n_inv >> 32) as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }
        let all_rns_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_rns_params_v3"),
            contents: bytemuck::cast_slice(&all_rns_params),
            usage: BufferUsages::STORAGE,
        });

        // Compute four-step twiddle factors
        // Row NTT uses omega_n2 = omega^(n/n2) = omega^32 (256th root of unity)
        // Col NTT uses omega_n1 = omega^(n/n1) = omega^256 (32nd root of unity)
        // Cross twiddles use omega^(row * col)

        // Row twiddles: omega_n2^j = omega_n^(32*j) for j = 0..n2
        let mut all_row_twiddles: Vec<u32> = Vec::with_capacity(k * n2 * 2);
        let mut all_row_inv_twiddles: Vec<u32> = Vec::with_capacity(k * n2 * 2);
        for rns_data in params.rns_data.iter() {
            for j in 0..n2 {
                // omega_n2^j = omega_n^(32*j) = omega_powers[(32*j) % n]
                let idx = (32 * j) % n;
                let omega = rns_data.omega_powers[idx];
                all_row_twiddles.push(omega as u32);
                all_row_twiddles.push((omega >> 32) as u32);

                let omega_inv = rns_data.omega_inv_powers[idx];
                all_row_inv_twiddles.push(omega_inv as u32);
                all_row_inv_twiddles.push((omega_inv >> 32) as u32);
            }
        }
        let all_row_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_row_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_row_twiddles),
            usage: BufferUsages::STORAGE,
        });
        let all_row_inv_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_row_inv_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_row_inv_twiddles),
            usage: BufferUsages::STORAGE,
        });

        // Col twiddles: omega_n1^i = omega_n^(256*i) for i = 0..n1
        let mut all_col_twiddles: Vec<u32> = Vec::with_capacity(k * n1 * 2);
        let mut all_col_inv_twiddles: Vec<u32> = Vec::with_capacity(k * n1 * 2);
        for rns_data in params.rns_data.iter() {
            for i in 0..n1 {
                // omega_n1^i = omega_n^(256*i) = omega_powers[(256*i) % n]
                let idx = (256 * i) % n;
                let omega = rns_data.omega_powers[idx];
                all_col_twiddles.push(omega as u32);
                all_col_twiddles.push((omega >> 32) as u32);

                let omega_inv = rns_data.omega_inv_powers[idx];
                all_col_inv_twiddles.push(omega_inv as u32);
                all_col_inv_twiddles.push((omega_inv >> 32) as u32);
            }
        }
        let all_col_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_col_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_col_twiddles),
            usage: BufferUsages::STORAGE,
        });
        let all_col_inv_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_col_inv_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_col_inv_twiddles),
            usage: BufferUsages::STORAGE,
        });

        // Cross twiddles: omega^(row * col) for all row in 0..n1, col in 0..n2
        let mut all_cross_twiddles: Vec<u32> = Vec::with_capacity(k * n * 2);
        let mut all_cross_inv_twiddles: Vec<u32> = Vec::with_capacity(k * n * 2);
        for rns_data in params.rns_data.iter() {
            for row in 0..n1 {
                for col in 0..n2 {
                    let idx = (row * col) % n;
                    let omega = rns_data.omega_powers[idx];
                    all_cross_twiddles.push(omega as u32);
                    all_cross_twiddles.push((omega >> 32) as u32);

                    let omega_inv = rns_data.omega_inv_powers[idx];
                    all_cross_inv_twiddles.push(omega_inv as u32);
                    all_cross_inv_twiddles.push((omega_inv >> 32) as u32);
                }
            }
        }
        let all_cross_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_cross_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_cross_twiddles),
            usage: BufferUsages::STORAGE,
        });
        let all_cross_inv_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_cross_inv_twiddles_v3"),
            contents: bytemuck::cast_slice(&all_cross_inv_twiddles),
            usage: BufferUsages::STORAGE,
        });

        // Psi powers for twist/untwist
        let mut all_psi: Vec<u32> = Vec::with_capacity(k * n * 2);
        let mut all_psi_inv: Vec<u32> = Vec::with_capacity(k * n * 2);
        for rns_data in params.rns_data.iter() {
            for &psi in &rns_data.psi_powers {
                all_psi.push(psi as u32);
                all_psi.push((psi >> 32) as u32);
            }
            for &psi_inv in &rns_data.psi_inv_powers {
                all_psi_inv.push(psi_inv as u32);
                all_psi_inv.push((psi_inv >> 32) as u32);
            }
        }
        let all_psi_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_psi_v3"),
            contents: bytemuck::cast_slice(&all_psi),
            usage: BufferUsages::STORAGE,
        });
        let all_psi_inv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_psi_inv_v3"),
            contents: bytemuck::cast_slice(&all_psi_inv),
            usage: BufferUsages::STORAGE,
        });

        // Pre-allocated buffers
        let max_batches = MAX_SLOT_MUL_BATCHES;
        let slot_buffer_size = (max_batches * n * 2 * 4) as u64;
        let rns_buffer_size = (max_batches * k * n * 2 * 4) as u64;

        let preallocated_slots_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slots_input_v3"),
            size: slot_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_encoded_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_v3"),
            size: slot_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Temp buffer for intermediate four-step results
        let preallocated_temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp_v3"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_pt_ntt_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_ntt_v3"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // CT input buffers
        const MAX_CT_CHUNKS: usize = 8;
        let ct_buffer_size = (MAX_CT_CHUNKS * k * n * 2 * 4) as u64;

        let preallocated_cts_c0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cts_c0_input_v3"),
            size: ct_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_cts_c1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cts_c1_input_v3"),
            size: ct_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_out_c0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c0_v3"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_out_c1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c1_v3"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Batch params buffer
        let initial_batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: max_batches as u32,
            num_moduli: k as u32,
        };
        let preallocated_batch_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_params_v3"),
                contents: bytemuck::bytes_of(&initial_batch_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Four-step params buffer (for row/col NTT)
        let initial_fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            log_n2: FOUR_STEP_LOG_N2,
            num_batches: max_batches as u32,
            num_moduli: k as u32,
            log_n1: FOUR_STEP_LOG_N1,
            _pad: 0,
        };
        let preallocated_fourstep_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fourstep_params_v3"),
                contents: bytemuck::bytes_of(&initial_fourstep_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Fused params buffer
        let initial_fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            n1: FOUR_STEP_N1 as u32,
            n2: FOUR_STEP_N2 as u32,
            num_batches: max_batches as u32,
            num_moduli: k as u32,
            num_cts: 1,
            batches_per_ct: max_batches as u32,
            _pad: 0,
        };
        let preallocated_fused_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_params_v3"),
                contents: bytemuck::bytes_of(&initial_fused_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Create bind groups for four-step slot encoding
        // Step 1: Col INTT - reads slots, writes to temp
        let slot_encode_col_intt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_col_intt_bind_group_v3"),
            layout: &slot_encode_col_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: goldilocks_col_inv_twiddles_buffer.as_entire_binding(),
                },
            ],
        });

        // Step 2: Cross-twiddle inverse - in-place on temp
        let slot_encode_cross_twiddle_inv_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_cross_twiddle_inv_bind_group_v3"),
            layout: &slot_encode_cross_twiddle_inv_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: plaintext_twiddles_buffer.as_entire_binding(),  // Full zeta_inv powers
                },
            ],
        });

        // Step 3: Row INTT - reads temp, writes to encoded
        let slot_encode_row_intt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_row_intt_bind_group_v3"),
            layout: &slot_encode_row_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: goldilocks_row_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Twist bind group: reads from encoded, writes to temp (with RNS expansion + transpose)
        // Uses FourStepParams because we need n1/n2 for the transpose
        let twist_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("twist_bind_group_v3"),
            layout: &twist_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_psi_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Row NTT bind group: reads from temp, writes to pt_ntt
        let row_ntt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("row_ntt_bind_group_v3"),
            layout: &row_ntt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_pt_ntt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_row_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Forward NTT data flow: twist → row_ntt → cross_twiddle → col_ntt
        // twist: encoded → temp
        // row_ntt: temp → pt_ntt
        // cross_twiddle: pt_ntt → pt_ntt (in-place)
        // col_ntt: pt_ntt → pt_ntt (in-place)

        // Cross twiddle bind group: in-place modification on pt_ntt buffer
        // (operates AFTER row_ntt, BEFORE col_ntt)
        let cross_twiddle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_twiddle_bind_group_v3"),
            layout: &cross_twiddle_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_pt_ntt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: all_cross_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Col NTT bind group: in-place modification on pt_ntt buffer
        // (operates LAST in forward NTT: row_ntt → cross → col_ntt)
        let col_ntt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("col_ntt_bind_group_v3"),
            layout: &col_ntt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_pt_ntt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: all_col_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Pointwise mul bind group
        let pointwise_mul_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pointwise_mul_bind_group_v3"),
            layout: &pointwise_mul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fused_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_pt_ntt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_cts_c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: preallocated_cts_c1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: preallocated_out_c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: preallocated_out_c1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // INTT order: col_intt → cross_inv → row_intt (reverse of forward)
        //
        // Col INTT bind group (for c0): reads from out_c0, writes to temp
        // This is the FIRST step of INTT
        let col_intt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("col_intt_bind_group_v3"),
            layout: &col_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_out_c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_col_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Cross twiddle inv bind group: in-place modification on temp
        let cross_twiddle_inv_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_twiddle_inv_bind_group_v3"),
            layout: &cross_twiddle_inv_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: all_cross_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Row INTT bind group (for c0): reads from temp, writes to out_c0 (with un-transpose)
        // This is the LAST step of INTT - writes final output directly
        let row_intt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("row_intt_bind_group_v3"),
            layout: &row_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_out_c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_row_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_psi_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // =========================================================================
        // C1 INTT bind groups (same pipelines, different data buffer)
        // =========================================================================

        // Col INTT bind group for c1: reads from out_c1, writes to temp
        let col_intt_c1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("col_intt_c1_bind_group_v3"),
            layout: &col_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_out_c1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_col_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Cross twiddle inv bind group for c1: in-place modification on temp
        // (temp is shared after col_intt_c1 writes to it)
        let cross_twiddle_inv_c1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_twiddle_inv_c1_bind_group_v3"),
            layout: &cross_twiddle_inv_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: all_cross_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Row INTT bind group for c1: reads from temp, writes to out_c1 (with un-transpose)
        let row_intt_c1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("row_intt_c1_bind_group_v3"),
            layout: &row_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_fourstep_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_out_c1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_row_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_psi_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            slot_encode_col_intt_pipeline,
            slot_encode_cross_twiddle_inv_pipeline,
            slot_encode_row_intt_pipeline,
            twist_pipeline,
            row_ntt_pipeline,
            cross_twiddle_pipeline,
            col_ntt_pipeline,
            col_intt_pipeline,
            cross_twiddle_inv_pipeline,
            row_intt_pipeline,
            pointwise_mul_pipeline,
            params,
            n1,
            n2,
            plaintext_twiddles_buffer,
            plaintext_params_buffer,
            goldilocks_col_inv_twiddles_buffer,
            goldilocks_row_inv_twiddles_buffer,
            all_row_twiddles_buffer,
            all_row_inv_twiddles_buffer,
            all_col_twiddles_buffer,
            all_col_inv_twiddles_buffer,
            all_cross_twiddles_buffer,
            all_cross_inv_twiddles_buffer,
            all_psi_buffer,
            all_psi_inv_buffer,
            all_rns_params_buffer,
            preallocated_encoded_buffer,
            preallocated_temp_buffer,
            preallocated_pt_ntt_buffer,
            preallocated_out_c0_buffer,
            preallocated_out_c1_buffer,
            preallocated_batch_params_buffer,
            preallocated_fourstep_params_buffer,
            preallocated_fused_params_buffer,
            preallocated_slots_buffer,
            preallocated_cts_c0_buffer,
            preallocated_cts_c1_buffer,
            slot_encode_col_intt_bind_group,
            slot_encode_cross_twiddle_inv_bind_group,
            slot_encode_row_intt_bind_group,
            twist_bind_group,
            row_ntt_bind_group,
            cross_twiddle_bind_group,
            col_ntt_bind_group,
            pointwise_mul_bind_group,
            col_intt_bind_group,
            cross_twiddle_inv_bind_group,
            row_intt_bind_group,
            col_intt_c1_bind_group,
            cross_twiddle_inv_c1_bind_group,
            row_intt_c1_bind_group,
        })
    }

    // =========================================================================
    // Warmup (Force Shader Compilation)
    // =========================================================================

    /// Forces shader compilation by dispatching minimal workloads through all pipelines.
    /// Call this after context creation to avoid compilation delays during actual work.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn warmup(&self) {
        pollster::block_on(self.warmup_async());
    }

    /// Async warmup for WASM - dispatches through all pipelines to force compilation.
    pub async fn warmup_async(&self) {
        let n = self.params.n;
        let k = self.params.k;
        let n1 = self.n1;
        let n2 = self.n2;

        // Use minimal batch count (1) for warmup
        let num_batches = 1usize;
        let log_n = (n as u32).trailing_zeros();

        // Update batch params for warmup
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Update fourstep params for warmup
        let fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            log_n2: (n2 as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            log_n1: (n1 as u32).trailing_zeros(),
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fourstep_params_buffer,
            0,
            bytemuck::bytes_of(&fourstep_params),
        );

        // Update fused params for warmup
        let fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            num_cts: 1,
            batches_per_ct: num_batches as u32,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fused_params_buffer,
            0,
            bytemuck::bytes_of(&fused_params),
        );

        let wg_per_n = (n + 255) / 256;

        // Create encoder and dispatch through all pipelines
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warmup_v3"),
        });

        // 1. Slot encode (four-step decomposition)
        // Step 1: Column INTT (32-point INTT on each of 256 columns)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_slot_encode_col_intt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
            pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
            pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
        }
        // Step 2: Cross-twiddle inverse
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_slot_encode_cross_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
            pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
            pass.dispatch_workgroups(wg_per_n as u32, num_batches as u32, 1);
        }
        // Step 3: Row INTT (256-point INTT on each of 32 rows)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_slot_encode_row_intt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
            pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
            pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
        }

        // 2. Twist (forward)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_twist"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.twist_pipeline);
            pass.set_bind_group(0, &self.twist_bind_group, &[]);
            pass.dispatch_workgroups(wg_per_n as u32, num_batches as u32, k as u32);
        }

        // 3. Row NTT
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_row_ntt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.row_ntt_pipeline);
            pass.set_bind_group(0, &self.row_ntt_bind_group, &[]);
            pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
        }

        // 4. Cross twiddle (forward)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_cross_twiddle"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cross_twiddle_pipeline);
            pass.set_bind_group(0, &self.cross_twiddle_bind_group, &[]);
            pass.dispatch_workgroups(wg_per_n as u32, num_batches as u32, k as u32);
        }

        // 5. Col NTT
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_col_ntt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.col_ntt_pipeline);
            pass.set_bind_group(0, &self.col_ntt_bind_group, &[]);
            pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
        }

        // 6. Pointwise mul
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_pointwise_mul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pointwise_mul_pipeline);
            pass.set_bind_group(0, &self.pointwise_mul_bind_group, &[]);
            pass.dispatch_workgroups(wg_per_n as u32, num_batches as u32, k as u32);
        }

        // 7. Col INTT
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_col_intt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.col_intt_pipeline);
            pass.set_bind_group(0, &self.col_intt_bind_group, &[]);
            pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
        }

        // 8. Cross twiddle inv
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_cross_twiddle_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cross_twiddle_pipeline);
            pass.set_bind_group(0, &self.cross_twiddle_inv_bind_group, &[]);
            pass.dispatch_workgroups(wg_per_n as u32, num_batches as u32, k as u32);
        }

        // 9. Row INTT
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup_row_intt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.row_intt_pipeline);
            pass.set_bind_group(0, &self.row_intt_bind_group, &[]);
            pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Force GPU to complete warmup before returning
        self.device.poll(wgpu::Maintain::Wait);
    }

    // =========================================================================
    // Execution Pipeline
    // =========================================================================

    /// Performs batched slot multiplication using four-step NTT.
    #[cfg(target_arch = "wasm32")]
    pub async fn mul_batched_multi_ct(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        self.mul_batched_multi_ct_async(slots, cts_c0_ntt, cts_c1_ntt, batches_per_ct)
            .await
    }

    /// Performs batched slot multiplication using four-step NTT.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mul_batched_multi_ct(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        pollster::block_on(self.mul_batched_multi_ct_async(slots, cts_c0_ntt, cts_c1_ntt, batches_per_ct))
    }

    /// Core async implementation of four-step NTT batched multiplication.
    async fn mul_batched_multi_ct_async(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let n1 = self.n1;
        let n2 = self.n2;
        let num_batches = slots.len();
        let num_cts = cts_c0_ntt.len();

        if num_batches > MAX_SLOT_MUL_BATCHES {
            return Err(GpuError::BatchSizeExceeded {
                requested: num_batches,
                max: MAX_SLOT_MUL_BATCHES,
            });
        }

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Flatten CT components
        let ct_flat_size = num_cts * k * n;
        let mut all_c0_flat: Vec<u64> = vec![0u64; ct_flat_size];
        let mut all_c1_flat: Vec<u64> = vec![0u64; ct_flat_size];

        for (ct_idx, (c0_ct, c1_ct)) in cts_c0_ntt.iter().zip(cts_c1_ntt.iter()).enumerate() {
            for (mod_idx, (c0_residue, c1_residue)) in c0_ct.iter().zip(c1_ct.iter()).enumerate() {
                let base = (ct_idx * k + mod_idx) * n;
                all_c0_flat[base..base + n].copy_from_slice(c0_residue);
                all_c1_flat[base..base + n].copy_from_slice(c1_residue);
            }
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&slots_flat),
        );
        self.queue.write_buffer(
            &self.preallocated_cts_c0_buffer,
            0,
            bytemuck::cast_slice(&all_c0_flat),
        );
        self.queue.write_buffer(
            &self.preallocated_cts_c1_buffer,
            0,
            bytemuck::cast_slice(&all_c1_flat),
        );

        // Update batch params
        let log_n = (n as u32).trailing_zeros();
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Update four-step params
        let fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            log_n2: FOUR_STEP_LOG_N2,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            log_n1: FOUR_STEP_LOG_N1,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fourstep_params_buffer,
            0,
            bytemuck::bytes_of(&fourstep_params),
        );

        // Update fused params (for pointwise mul)
        let fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            n1: self.n1 as u32,
            n2: self.n2 as u32,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            num_cts: num_cts as u32,
            batches_per_ct: batches_per_ct.get(0).copied().unwrap_or(num_batches) as u32,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fused_params_buffer,
            0,
            bytemuck::bytes_of(&fused_params),
        );

        // Execute GPU pipeline
        // =====================================================================
        // Phase 1: Slot encode (plaintext INTT) - four-step decomposition
        // Step 1: 32-point col INTT, Step 2: cross-twiddle inv, Step 3: 256-point row INTT
        // =====================================================================
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"[V3] Starting slot_encode phase".into());

        let wg_per_n = ((n + 255) / 256) as u32;
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_v3_encoder"),
            });

            // Step 1: Column INTT (32-point INTT on each of 256 columns)
            // Dispatch: (n2, num_batches, 1) = (256, num_batches, 1)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_col_intt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
            }

            // Step 2: Cross-twiddle inverse (multiply by zeta_inv^(row*col))
            // Dispatch: (n/256, num_batches, 1) = (32, num_batches, 1) with workgroup_size(256)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_cross_twiddle_inv"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, 1);
            }

            // Step 3: Row INTT (256-point INTT on each of 32 rows) + n^-1 scaling
            // Dispatch: (n1, num_batches, 1) = (32, num_batches, 1)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_row_intt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"[V3] slot_encode submitted, starting NTT phase".into());

        // =====================================================================
        // Phase 2: Four-step forward NTT + pointwise mul + INTT
        // =====================================================================
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fourstep_ntt_v3_encoder"),
            });

            // Dispatch dimensions for element-wise operations
            let wg_per_n = ((n + 255) / 256) as u32;

            // 2a. Twist: encoded → temp (apply psi^idx, expand to k moduli)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("twist_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.twist_pipeline);
                pass.set_bind_group(0, &self.twist_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Standard four-step NTT order (with transpose in twist):
            //   row NTT (256-pt) -> cross twiddle -> col NTT (32-pt)

            // 2b. Row NTT: 256-point NTTs on each row (n1=32 rows per batch per modulus)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_ntt_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_ntt_pipeline);
                pass.set_bind_group(0, &self.row_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            // 2c. Cross twiddle: multiply by omega^(row * col)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // 2d. Col NTT: 32-point NTTs on each column (n2=256 columns per batch per modulus)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_ntt_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_ntt_pipeline);
                pass.set_bind_group(0, &self.col_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            // 2e. Pointwise multiply: pt_ntt × ct → out_c0, out_c1
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pointwise_mul_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pointwise_mul_pipeline);
                pass.set_bind_group(0, &self.pointwise_mul_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // =====================================================================
            // INTT for c0: col_intt → cross_inv → row_intt (reverse of forward)
            // row_intt also applies un-transpose + untwist at the end
            // =====================================================================

            // 2f. Col INTT for c0: out_c0 → temp (32-point INTTs on columns)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_intt_c0_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_intt_pipeline);
                pass.set_bind_group(0, &self.col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            // 2g. Cross twiddle inv for c0: temp → temp (in-place)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_inv_c0_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // 2h. Row INTT + un-transpose + untwist for c0: temp → out_c0
            // The row_intt_untwist shader un-transposes and applies psi_inv, writes to out_c0
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_intt_c0_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_intt_pipeline);
                pass.set_bind_group(0, &self.row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            // =====================================================================
            // INTT for c1: col_intt → cross_inv → row_intt
            // =====================================================================

            // 2i. Col INTT for c1: out_c1 → temp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_intt_c1_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_intt_pipeline);
                pass.set_bind_group(0, &self.col_intt_c1_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            // 2j. Cross twiddle inv for c1: temp → temp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_inv_c1_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_inv_c1_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // 2k. Row INTT + un-transpose + untwist for c1: temp → out_c1
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_intt_c1_v3"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_intt_pipeline);
                pass.set_bind_group(0, &self.row_intt_c1_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back results
        self.read_rns_batch_async(num_batches).await
    }

    /// Reads back results from GPU.
    async fn read_rns_batch_async(
        &self,
        num_batches: usize,
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let result_size = (num_batches * k * n * 2 * 4) as u64;

        // Create staging buffers
        let staging_c0 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c0_v3"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_c1 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c1_v3"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"[V3] NTT phase done, copying results".into());

        // Copy from output buffers to staging
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_results_v3"),
        });
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c0_buffer, 0, &staging_c0, 0, result_size);
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c1_buffer, 0, &staging_c1, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read - use futures oneshot channels for proper async await
        let c0_slice = staging_c0.slice(..);
        let c1_slice = staging_c1.slice(..);

        let (tx0, rx0) = futures::channel::oneshot::channel();
        let (tx1, rx1) = futures::channel::oneshot::channel();

        c0_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx0.send(result);
        });
        c1_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx1.send(result);
        });

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"[V3] Awaiting buffer maps...".into());

        // In WASM, we need to poll the device and yield to event loop repeatedly
        #[cfg(target_arch = "wasm32")]
        {
            use std::pin::Pin;
            use std::task::{Context, Poll};
            use futures::Future;
            use wasm_bindgen::prelude::*;
            use wasm_bindgen_futures::JsFuture;

            // Helper to yield to event loop
            async fn yield_now() {
                let promise = js_sys::Promise::resolve(&JsValue::undefined());
                let _ = JsFuture::from(promise).await;
            }

            let mut rx0 = rx0;
            let mut rx1 = rx1;
            let mut poll_count = 0u32;

            // Poll rx0 until ready
            loop {
                self.device.poll(wgpu::Maintain::Poll);
                poll_count += 1;
                if poll_count % 100 == 0 {
                    web_sys::console::log_1(&format!("[V3] Poll iteration {}", poll_count).into());
                }
                let waker = futures::task::noop_waker();
                let mut cx = Context::from_waker(&waker);
                match Pin::new(&mut rx0).poll(&mut cx) {
                    Poll::Ready(result) => {
                        web_sys::console::log_1(&format!("[V3] C0 ready after {} polls", poll_count).into());
                        result.map_err(|_| GpuError::MapFailed)?
                            .map_err(|_| GpuError::MapFailed)?;
                        break;
                    }
                    Poll::Pending => {
                        // Yield to event loop
                        yield_now().await;
                    }
                }
            }
            web_sys::console::log_1(&"[V3] C0 map complete".into());

            // Poll rx1 until ready
            loop {
                self.device.poll(wgpu::Maintain::Poll);
                let waker = futures::task::noop_waker();
                let mut cx = Context::from_waker(&waker);
                match Pin::new(&mut rx1).poll(&mut cx) {
                    Poll::Ready(result) => {
                        result.map_err(|_| GpuError::MapFailed)?
                            .map_err(|_| GpuError::MapFailed)?;
                        break;
                    }
                    Poll::Pending => {
                        yield_now().await;
                    }
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.poll(wgpu::Maintain::Wait);
            rx0.await
                .map_err(|_| GpuError::MapFailed)?
                .map_err(|_| GpuError::MapFailed)?;
            rx1.await
                .map_err(|_| GpuError::MapFailed)?
                .map_err(|_| GpuError::MapFailed)?;
        }

        let c0_data: Vec<u32> = bytemuck::cast_slice(&c0_slice.get_mapped_range()).to_vec();
        let c1_data: Vec<u32> = bytemuck::cast_slice(&c1_slice.get_mapped_range()).to_vec();

        // Convert to structured output
        let mut result_c0: Vec<Vec<Vec<u64>>> = Vec::with_capacity(num_batches);
        let mut result_c1: Vec<Vec<Vec<u64>>> = Vec::with_capacity(num_batches);

        for batch_idx in 0..num_batches {
            let mut batch_c0: Vec<Vec<u64>> = Vec::with_capacity(k);
            let mut batch_c1: Vec<Vec<u64>> = Vec::with_capacity(k);

            for mod_idx in 0..k {
                let mut residue_c0: Vec<u64> = Vec::with_capacity(n);
                let mut residue_c1: Vec<u64> = Vec::with_capacity(n);

                for j in 0..n {
                    let offset = ((batch_idx * k + mod_idx) * n + j) * 2;
                    let c0_val = (c0_data[offset] as u64) | ((c0_data[offset + 1] as u64) << 32);
                    let c1_val = (c1_data[offset] as u64) | ((c1_data[offset + 1] as u64) << 32);
                    residue_c0.push(c0_val);
                    residue_c1.push(c1_val);
                }

                batch_c0.push(residue_c0);
                batch_c1.push(residue_c1);
            }

            result_c0.push(batch_c0);
            result_c1.push(batch_c1);
        }

        Ok((result_c0, result_c1))
    }

    /// Test helper: Run only slot_encode and return the encoded coefficients.
    #[cfg(test)]
    pub fn test_slot_encode_only(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        let n = self.params.n;
        let num_batches = slots.len();

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload slots
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&slots_flat),
        );

        // Update batch params
        let log_n = (n as u32).trailing_zeros();
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: self.params.k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Execute slot_encode (four-step decomposition)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_only_test"),
            });
            let n1 = self.n1;
            let n2 = self.n2;
            let wg_per_n = ((n + 255) / 256) as u32;

            // Step 1: Column INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_col_intt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
            }
            // Step 2: Cross-twiddle inverse
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_cross_inv"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, 1);
            }
            // Step 3: Row INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_row_intt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back encoded buffer
        let result_size = (num_batches * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_encoded_v3"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoded"),
        });
        encoder.copy_buffer_to_buffer(&self.preallocated_encoded_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv().unwrap().unwrap();

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Convert to structured output
        let mut result: Vec<Vec<u64>> = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch: Vec<u64> = Vec::with_capacity(n);
            for j in 0..n {
                let offset = (batch_idx * n + j) * 2;
                let val = (data[offset] as u64) | ((data[offset + 1] as u64) << 32);
                batch.push(val);
            }
            result.push(batch);
        }

        Ok(result)
    }

    /// Test helper: Run only the forward NTT (slot_encode + twist + row_ntt + cross_twiddle + col_ntt)
    /// and return the pt_ntt values for debugging.
    #[cfg(test)]
    pub fn test_forward_ntt_only(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let n1 = self.n1;
        let n2 = self.n2;
        let num_batches = slots.len();

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload slots
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&slots_flat),
        );

        // Update batch params
        let log_n = (n as u32).trailing_zeros();
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Update four-step params
        let fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            log_n2: FOUR_STEP_LOG_N2,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            log_n1: FOUR_STEP_LOG_N1,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fourstep_params_buffer,
            0,
            bytemuck::bytes_of(&fourstep_params),
        );

        // Execute forward NTT pipeline
        // 1. Slot encode (four-step decomposition)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_v3_test"),
            });
            let wg_per_n = ((n + 255) / 256) as u32;

            // Step 1: Column INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_col_intt_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
            }
            // Step 2: Cross-twiddle inverse
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_cross_inv_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, 1);
            }
            // Step 3: Row INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_row_intt_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // 2. Forward NTT (twist + row_ntt + cross_twiddle + col_ntt)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_ntt_v3_test"),
            });
            let wg_per_n = ((n + 255) / 256) as u32;

            // Twist: encoded → temp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("twist_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.twist_pipeline);
                pass.set_bind_group(0, &self.twist_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Row NTT: temp → pt_ntt
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_ntt_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_ntt_pipeline);
                pass.set_bind_group(0, &self.row_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            // Cross twiddle: pt_ntt (in-place)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Col NTT: pt_ntt (in-place)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_ntt_test"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_ntt_pipeline);
                pass.set_bind_group(0, &self.col_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back pt_ntt
        let result_size = (num_batches * k * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_pt_ntt_v3"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_pt_ntt"),
        });
        encoder.copy_buffer_to_buffer(&self.preallocated_pt_ntt_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv().unwrap().unwrap();

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Convert to structured output
        let mut result: Vec<Vec<Vec<u64>>> = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch: Vec<Vec<u64>> = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let mut residue: Vec<u64> = Vec::with_capacity(n);
                for j in 0..n {
                    let offset = ((batch_idx * k + mod_idx) * n + j) * 2;
                    let val = (data[offset] as u64) | ((data[offset + 1] as u64) << 32);
                    residue.push(val);
                }
                batch.push(residue);
            }
            result.push(batch);
        }

        Ok(result)
    }

    /// Test helper: Run slot_encode + twist only and return the temp_buffer values.
    /// This helps isolate whether bugs are in twist or row_ntt.
    #[cfg(test)]
    pub fn test_twist_only(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let num_batches = slots.len();

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload slots
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&slots_flat),
        );

        // Update batch params
        let log_n = (n as u32).trailing_zeros();
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Update four-step params
        let fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: self.n1 as u32,
            n2: self.n2 as u32,
            log_n2: FOUR_STEP_LOG_N2,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            log_n1: FOUR_STEP_LOG_N1,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fourstep_params_buffer,
            0,
            bytemuck::bytes_of(&fourstep_params),
        );

        // Execute slot_encode (four-step decomposition)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_twist_test"),
            });
            let n1 = self.n1;
            let n2 = self.n2;
            let wg_per_n = ((n + 255) / 256) as u32;

            // Step 1: Column INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_col_intt_twist"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
            }
            // Step 2: Cross-twiddle inverse
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_cross_inv_twist"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, 1);
            }
            // Step 3: Row INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_row_intt_twist"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Execute twist only
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twist_only_test"),
            });
            let wg_per_n = ((n + 255) / 256) as u32;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("twist_only"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.twist_pipeline);
                pass.set_bind_group(0, &self.twist_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back temp_buffer
        let result_size = (num_batches * k * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_temp_v3"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_temp"),
        });
        encoder.copy_buffer_to_buffer(&self.preallocated_temp_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv().unwrap().unwrap();

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Convert to structured output [batch][mod][element]
        let mut result: Vec<Vec<Vec<u64>>> = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch: Vec<Vec<u64>> = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let mut residue: Vec<u64> = Vec::with_capacity(n);
                for j in 0..n {
                    let offset = ((batch_idx * k + mod_idx) * n + j) * 2;
                    let val = (data[offset] as u64) | ((data[offset + 1] as u64) << 32);
                    residue.push(val);
                }
                batch.push(residue);
            }
            result.push(batch);
        }

        Ok(result)
    }

    /// Test helper: Run slot_encode + NTT + INTT and return the round-trip result.
    /// This verifies that NTT and INTT are inverse operations.
    #[cfg(test)]
    pub fn test_ntt_intt_roundtrip(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let n1 = self.n1;
        let n2 = self.n2;
        let num_batches = slots.len();

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload slots
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&slots_flat),
        );

        // Update batch params
        let log_n = (n as u32).trailing_zeros();
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        self.queue.write_buffer(
            &self.preallocated_batch_params_buffer,
            0,
            bytemuck::bytes_of(&batch_params),
        );

        // Update four-step params
        let fourstep_params = GpuFourStepParams {
            n: n as u32,
            n1: n1 as u32,
            n2: n2 as u32,
            log_n2: FOUR_STEP_LOG_N2,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            log_n1: FOUR_STEP_LOG_N1,
            _pad: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fourstep_params_buffer,
            0,
            bytemuck::bytes_of(&fourstep_params),
        );

        // Execute slot_encode (four-step decomposition)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_roundtrip"),
            });
            let wg_per_n_encode = ((n + 255) / 256) as u32;

            // Step 1: Column INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_col_intt_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_col_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, 1);
            }
            // Step 2: Cross-twiddle inverse
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_cross_inv_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.slot_encode_cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n_encode, num_batches as u32, 1);
            }
            // Step 3: Row INTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("slot_encode_row_intt_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.slot_encode_row_intt_pipeline);
                pass.set_bind_group(0, &self.slot_encode_row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Execute forward NTT (twist → row_ntt → cross → col_ntt)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_ntt_roundtrip"),
            });
            let wg_per_n = ((n + 255) / 256) as u32;

            // Twist
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("twist_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.twist_pipeline);
                pass.set_bind_group(0, &self.twist_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Row NTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_ntt_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_ntt_pipeline);
                pass.set_bind_group(0, &self.row_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            // Cross twiddle
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Col NTT
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_ntt_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_ntt_pipeline);
                pass.set_bind_group(0, &self.col_ntt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Copy pt_ntt to out_c0 (to feed the INTT path which reads from out_c0)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_for_intt_roundtrip"),
            });
            let copy_size = (num_batches * k * n * 2 * 4) as u64;
            encoder.copy_buffer_to_buffer(&self.preallocated_pt_ntt_buffer, 0, &self.preallocated_out_c0_buffer, 0, copy_size);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Execute inverse NTT (col_intt → cross_inv → row_intt_untwist)
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("inverse_ntt_roundtrip"),
            });
            let wg_per_n = ((n + 255) / 256) as u32;

            // Col INTT: out_c0 → temp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("col_intt_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.col_intt_pipeline);
                pass.set_bind_group(0, &self.col_intt_bind_group, &[]);
                pass.dispatch_workgroups(n2 as u32, num_batches as u32, k as u32);
            }

            // Cross twiddle inv: temp (in-place)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cross_twiddle_inv_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cross_twiddle_inv_pipeline);
                pass.set_bind_group(0, &self.cross_twiddle_inv_bind_group, &[]);
                pass.dispatch_workgroups(wg_per_n, num_batches as u32, k as u32);
            }

            // Row INTT + untwist: temp → out_c0
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("row_intt_untwist_rt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.row_intt_pipeline);
                pass.set_bind_group(0, &self.row_intt_bind_group, &[]);
                pass.dispatch_workgroups(n1 as u32, num_batches as u32, k as u32);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back out_c0
        let result_size = (num_batches * k * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_roundtrip"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_roundtrip_result"),
        });
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c0_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv().unwrap().unwrap();

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Convert to structured output [batch][mod][element]
        let mut result: Vec<Vec<Vec<u64>>> = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch: Vec<Vec<u64>> = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let mut residue: Vec<u64> = Vec::with_capacity(n);
                for j in 0..n {
                    let offset = ((batch_idx * k + mod_idx) * n + j) * 2;
                    let val = (data[offset] as u64) | ((data[offset + 1] as u64) << 32);
                    residue.push(val);
                }
                batch.push(residue);
            }
            result.push(batch);
        }

        Ok(result)
    }
}
// =============================================================================
// Four-Step Slot Encode Shaders (Goldilocks INTT)
// For N=8192 = 32 rows × 256 columns, we decompose INTT into:
//   1. 32-point INTT on each of 256 columns (shared mem: 256 bytes)
//   2. Inverse cross-twiddle multiplication
//   3. 256-point INTT on each of 32 rows (shared mem: 2KB)
//   4. Scale by n^-1
// =============================================================================

/// Step 1: Column INTT - 32-point INTT on each column
/// Input layout: slots[batch * n + row * n2 + col] (row-major)
/// Each workgroup processes one column of one batch
/// Dispatch: (n2, num_batches, 1) = (256, num_batches, 1)
const SLOT_ENCODE_COL_INTT_SHADER: &str = r#"
#import math

struct FourStepParams {
    n: u32,           // 8192
    n1: u32,          // 32 (rows)
    n2: u32,          // 256 (columns)
    log_n2: u32,      // 8
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,      // 5
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FourStepParams;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> temp: array<u32>;
@group(0) @binding(3) var<storage, read> col_inv_twiddles: array<u32>;

var<workgroup> shared_lo: array<u32, 32>;
var<workgroup> shared_hi: array<u32, 32>;

@compute @workgroup_size(32, 1, 1)
fn slot_encode_col_intt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let col = wg_id.x;
    let batch_idx = wg_id.y;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n1 = params.log_n1;
    let n = params.n;

    if col >= n2 || batch_idx >= params.num_batches { return; }

    // Load column elements with bit-reversal
    // Input: slots[batch * n + row * n2 + col]
    let batch_offset = batch_idx * n;
    let row = tid;
    let in_idx = batch_offset + row * n2 + col;
    let in_base = in_idx * 2u;
    let val = vec2<u32>(slots[in_base], slots[in_base + 1u]);
    let rev_row = math::bit_reverse(row, log_n1);
    shared_lo[rev_row] = val.x;
    shared_hi[rev_row] = val.y;
    workgroupBarrier();

    // 32-point INTT (5 stages)
    for (var stage = 0u; stage < log_n1; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        if tid < (n1 >> 1u) {
            let butterfly_idx = tid;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n1 / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(col_inv_twiddles[tw_base], col_inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::goldilocks_mul(v, twiddle);
            let new_u = math::goldilocks_add(u, tw_v);
            let new_v = math::goldilocks_sub(u, tw_v);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Store to temp (same layout: temp[batch * n + row * n2 + col])
    let out_val = vec2<u32>(shared_lo[row], shared_hi[row]);
    let out_idx = batch_offset + row * n2 + col;
    let out_base = out_idx * 2u;
    temp[out_base] = out_val.x;
    temp[out_base + 1u] = out_val.y;
}
"#;

/// Step 2: Inverse cross-twiddle multiplication
/// Multiply each element by zeta_inv^(row * col)
/// Dispatch: (n / 256, num_batches, 1) = (32, num_batches, 1) with workgroup_size(256)
const SLOT_ENCODE_CROSS_TWIDDLE_INV_SHADER: &str = r#"
#import math

struct FourStepParams {
    n: u32,
    n1: u32,
    n2: u32,
    log_n2: u32,
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FourStepParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> cross_inv_twiddles: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn slot_encode_cross_twiddle_inv(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let idx = global_id.x;
    let batch_idx = wg_id.y;
    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;

    if idx >= n || batch_idx >= params.num_batches { return; }

    let row = idx / n2;
    let col = idx % n2;

    // Load value
    let batch_offset = batch_idx * n;
    let data_idx = batch_offset + idx;
    let base = data_idx * 2u;
    let val = vec2<u32>(data[base], data[base + 1u]);

    // Get inverse cross-twiddle: zeta_inv^(row * col)
    let tw_idx = row * col;  // This wraps naturally for powers
    let tw_base = (tw_idx % n) * 2u;
    let twiddle = vec2<u32>(cross_inv_twiddles[tw_base], cross_inv_twiddles[tw_base + 1u]);

    // Multiply
    let result = math::goldilocks_mul(val, twiddle);

    // Store
    data[base] = result.x;
    data[base + 1u] = result.y;
}
"#;

/// Step 3: Row INTT - 256-point INTT on each row + n^-1 scaling
/// Each workgroup processes one row of one batch
/// Dispatch: (n1, num_batches, 1) = (32, num_batches, 1)
const SLOT_ENCODE_ROW_INTT_SHADER: &str = r#"
#import math

struct FourStepParams {
    n: u32,
    n1: u32,
    n2: u32,
    log_n2: u32,
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FourStepParams;
@group(0) @binding(1) var<storage, read> temp: array<u32>;
@group(0) @binding(2) var<storage, read_write> coeffs: array<u32>;
@group(0) @binding(3) var<storage, read> row_inv_twiddles: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 256>;
var<workgroup> shared_hi: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn slot_encode_row_intt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let row = wg_id.x;
    let batch_idx = wg_id.y;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n2 = params.log_n2;
    let n = params.n;

    if row >= n1 || batch_idx >= params.num_batches { return; }

    // Load row elements with bit-reversal
    // Input: temp[batch * n + row * n2 + col]
    let batch_offset = batch_idx * n;
    let col = tid;
    let in_idx = batch_offset + row * n2 + col;
    let in_base = in_idx * 2u;
    let val = vec2<u32>(temp[in_base], temp[in_base + 1u]);
    let rev_col = math::bit_reverse(col, log_n2);
    shared_lo[rev_col] = val.x;
    shared_hi[rev_col] = val.y;
    workgroupBarrier();

    // 256-point INTT (8 stages)
    for (var stage = 0u; stage < log_n2; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        if tid < (n2 >> 1u) {
            let butterfly_idx = tid;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n2 / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(row_inv_twiddles[tw_base], row_inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::goldilocks_mul(v, twiddle);
            let new_u = math::goldilocks_add(u, tw_v);
            let new_v = math::goldilocks_sub(u, tw_v);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Scale by n^-1 and store to coeffs
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    var out_val = vec2<u32>(shared_lo[col], shared_hi[col]);
    out_val = math::goldilocks_mul(out_val, n_inv);
    let out_idx = batch_offset + row * n2 + col;
    let out_base = out_idx * 2u;
    coeffs[out_base] = out_val.x;
    coeffs[out_base + 1u] = out_val.y;
}
"#;

// =============================================================================
// Four-Step NTT Shaders
// For N=8192 = 32 rows × 256 columns
// =============================================================================

/// Twist shader with transpose - applies psi^idx to input coefficients AND transposes.
///
/// Four-step NTT requires column-major input indexing:
///   matrix[j1*n2 + j2] = input[j2*n1 + j1]
///
/// Combined with psi twist:
///   output[idx] = input[transposed_idx] * psi^transposed_idx
/// where:
///   idx = j1*n2 + j2 (output position in row-major matrix)
///   transposed_idx = j2*n1 + j1 (original coefficient position)
const TWIST_SHADER: &str = r#"
#import math

struct FourStepParams {
    n: u32,           // Total size (8192)
    n1: u32,          // Number of rows (32)
    n2: u32,          // Number of columns (256)
    log_n2: u32,      // log2(n2) = 8
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,      // log2(n1) = 5
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FourStepParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> psi_powers: array<u32>;  // k * n psi^i values
@group(0) @binding(4) var<storage, read> mod_params: array<ModulusParams>;

@compute @workgroup_size(256, 1, 1)
fn twist(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let batch_idx = wg_id.y;
    let mod_idx = wg_id.z;
    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let k = params.num_moduli;

    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let idx = global_id.x;
    if idx >= n { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    // Four-step transpose: output[j1*n2 + j2] reads from input[j2*n1 + j1]
    // idx = j1*n2 + j2, so j1 = idx/n2, j2 = idx%n2
    let j1 = idx / n2;
    let j2 = idx % n2;
    let transposed_idx = j2 * n1 + j1;

    // Input: encoded[batch_idx * n + transposed_idx]
    // The encoded value is a Goldilocks field element (can be up to 2^64)
    let in_base = (batch_idx * n + transposed_idx) * 2u;
    let val = vec2<u32>(input[in_base], input[in_base + 1u]);

    // Reduce val mod q first (val can be 64-bit, q is 60-bit)
    let val_reduced = math::reduce_mod_64(val, q);

    // Psi power for the ORIGINAL coefficient position (transposed_idx)
    let psi_base = (mod_idx * n + transposed_idx) * 2u;
    let psi = vec2<u32>(psi_powers[psi_base], psi_powers[psi_base + 1u]);

    // Twist: val_reduced * psi^transposed_idx (both are now < q)
    let twisted = math::mulmod_60bit(val_reduced, psi, q);

    // Output: temp[(batch_idx * k + mod_idx) * n + idx]
    let out_offset = (batch_idx * k + mod_idx) * n + idx;
    let out_base = out_offset * 2u;
    output[out_base] = twisted.x;
    output[out_base + 1u] = twisted.y;
}
"#;

/// Row NTT shader - performs n1 independent n2-point NTTs (256-point each).
/// Each workgroup handles one row of one batch of one modulus.
/// Workgroup layout: (row_idx, batch_idx, mod_idx)
const ROW_NTT_SHADER: &str = r#"
#import math

struct Params {
    n: u32,           // Total size (8192)
    n1: u32,          // Number of rows (32)
    n2: u32,          // Number of columns (256)
    log_n2: u32,      // log2(n2) = 8
    num_batches: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;  // k * n2 twiddles (omega_n2 powers)
@group(0) @binding(4) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 256>;
var<workgroup> shared_hi: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn row_ntt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let row_idx = wg_id.x;     // Which row (0..n1)
    let batch_idx = wg_id.y;   // Which batch
    let mod_idx = wg_id.z;     // Which modulus

    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n2 = params.log_n2;
    let k = params.num_moduli;

    if row_idx >= n1 { return; }
    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    // Twiddle offset for this modulus (omega_n2 powers)
    let tw_mod_offset = mod_idx * n2 * 2u;

    // Input/output offset for this (batch, modulus, row)
    let data_offset = (batch_idx * k + mod_idx) * n + row_idx * n2;

    // Load row into shared memory with bit-reversal
    let col = tid;  // Each thread handles one column
    if col < n2 {
        let in_base = (data_offset + col) * 2u;
        let val = vec2<u32>(input[in_base], input[in_base + 1u]);
        let rev_col = math::bit_reverse(col, log_n2);
        shared_lo[rev_col] = val.x;
        shared_hi[rev_col] = val.y;
    }
    workgroupBarrier();

    // NTT butterfly stages (log_n2 = 8 stages for 256-point NTT)
    for (var stage = 0u; stage < log_n2; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        // Each thread does one butterfly
        let butterfly_idx = tid;
        if butterfly_idx < n2 / 2u {
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n2 / m);
            let tw_base = tw_mod_offset + twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::mulmod_60bit(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Store result back to same location
    if col < n2 {
        let val = vec2<u32>(shared_lo[col], shared_hi[col]);
        let out_base = (data_offset + col) * 2u;
        output[out_base] = val.x;
        output[out_base + 1u] = val.y;
    }
}
"#;

/// Cross twiddle shader - multiplies by omega^(row * col).
/// Processes one element per thread.
const CROSS_TWIDDLE_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    n1: u32,
    n2: u32,
    _pad0: u32,
    num_batches: u32,
    num_moduli: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> cross_twiddles: array<u32>;  // k * n twiddles
@group(0) @binding(3) var<storage, read> mod_params: array<ModulusParams>;

@compute @workgroup_size(256, 1, 1)
fn cross_twiddle(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let batch_idx = wg_id.y;
    let mod_idx = wg_id.z;
    let n = params.n;
    let k = params.num_moduli;

    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let idx = global_id.x;
    if idx >= n { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    // Data offset
    let data_offset = (batch_idx * k + mod_idx) * n + idx;
    let data_base = data_offset * 2u;
    let val = vec2<u32>(data[data_base], data[data_base + 1u]);

    // Cross twiddle: omega^(row * col) where idx = row * n2 + col
    let tw_base = (mod_idx * n + idx) * 2u;
    let twiddle = vec2<u32>(cross_twiddles[tw_base], cross_twiddles[tw_base + 1u]);

    let result = math::mulmod_60bit(val, twiddle, q);

    data[data_base] = result.x;
    data[data_base + 1u] = result.y;
}
"#;

/// Column NTT shader - performs n2 independent n1-point NTTs (32-point each).
/// Each workgroup handles one column of one batch of one modulus.
/// Uses strided access pattern (stride = n2).
const COL_NTT_SHADER: &str = r#"
#import math

struct Params {
    n: u32,           // Total size (8192)
    n1: u32,          // Number of rows (32)
    n2: u32,          // Number of columns (256)
    log_n2: u32,      // log2(n2) = 8 (unused in col NTT)
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,      // log2(n1) = 5 (used here)
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;  // k * n1 twiddles (omega_n1 powers)
@group(0) @binding(3) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 32>;
var<workgroup> shared_hi: array<u32, 32>;

@compute @workgroup_size(32, 1, 1)
fn col_ntt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let col_idx = wg_id.x;     // Which column (0..n2)
    let batch_idx = wg_id.y;   // Which batch
    let mod_idx = wg_id.z;     // Which modulus

    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n1 = params.log_n1;
    let k = params.num_moduli;

    if col_idx >= n2 { return; }
    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    // Twiddle offset for this modulus (omega_n1 powers)
    let tw_mod_offset = mod_idx * n1 * 2u;

    // Base offset for this (batch, modulus)
    let batch_mod_offset = (batch_idx * k + mod_idx) * n;

    // Load column into shared memory with bit-reversal
    // Column elements are at: col_idx, col_idx + n2, col_idx + 2*n2, ...
    let row = tid;
    if row < n1 {
        let data_idx = batch_mod_offset + row * n2 + col_idx;
        let data_base = data_idx * 2u;
        let val = vec2<u32>(data[data_base], data[data_base + 1u]);
        let rev_row = math::bit_reverse(row, log_n1);
        shared_lo[rev_row] = val.x;
        shared_hi[rev_row] = val.y;
    }
    workgroupBarrier();

    // NTT butterfly stages (log_n1 = 5 stages for 32-point NTT)
    for (var stage = 0u; stage < log_n1; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let butterfly_idx = tid;
        if butterfly_idx < n1 / 2u {
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n1 / m);
            let tw_base = tw_mod_offset + twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::mulmod_60bit(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Store result back (strided access)
    if row < n1 {
        let val = vec2<u32>(shared_lo[row], shared_hi[row]);
        let data_idx = batch_mod_offset + row * n2 + col_idx;
        let data_base = data_idx * 2u;
        data[data_base] = val.x;
        data[data_base + 1u] = val.y;
    }
}
"#;

/// Column INTT shader - performs n2 independent n1-point INTTs (32-point each).
/// Has separate input and output buffers for INTT path:
///   - INTT reads from out_c0/c1 (NTT result), writes to temp
const COL_INTT_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    n1: u32,
    n2: u32,
    log_n2: u32,      // unused in col INTT
    num_batches: u32,
    num_moduli: u32,
    log_n1: u32,      // used here
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> inv_twiddles: array<u32>;  // k * n1 inverse twiddles
@group(0) @binding(4) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 32>;
var<workgroup> shared_hi: array<u32, 32>;

@compute @workgroup_size(32, 1, 1)
fn col_intt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let col_idx = wg_id.x;
    let batch_idx = wg_id.y;
    let mod_idx = wg_id.z;

    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n1 = params.log_n1;
    let k = params.num_moduli;

    if col_idx >= n2 { return; }
    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    let tw_mod_offset = mod_idx * n1 * 2u;
    let batch_mod_offset = (batch_idx * k + mod_idx) * n;

    // Load from input with bit-reversal
    let row = tid;
    if row < n1 {
        let data_idx = batch_mod_offset + row * n2 + col_idx;
        let data_base = data_idx * 2u;
        let val = vec2<u32>(input[data_base], input[data_base + 1u]);
        let rev_row = math::bit_reverse(row, log_n1);
        shared_lo[rev_row] = val.x;
        shared_hi[rev_row] = val.y;
    }
    workgroupBarrier();

    // INTT butterfly stages
    for (var stage = 0u; stage < log_n1; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let butterfly_idx = tid;
        if butterfly_idx < n1 / 2u {
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n1 / m);
            let tw_base = tw_mod_offset + twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            // INTT butterfly: same as NTT (DIT style)
            let tw_v = math::mulmod_60bit(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Note: n1^-1 scaling is deferred to the final row INTT
    // Store to output without scaling
    if row < n1 {
        let val = vec2<u32>(shared_lo[row], shared_hi[row]);
        let data_idx = batch_mod_offset + row * n2 + col_idx;
        let data_base = data_idx * 2u;
        output[data_base] = val.x;
        output[data_base + 1u] = val.y;
    }
}
"#;

/// Row INTT + un-transpose + untwist shader.
/// Performs n1 independent n2-point INTTs, then un-transposes and applies psi_inv^idx.
/// Also applies the final n^-1 scaling.
///
/// After four-step INTT, data is still transposed: matrix[j1*n2 + j2] = coeff[j2*n1 + j1]
/// This shader:
///   1. Computes row INTT for row j1 (256-point INTT)
///   2. Un-transposes: writes to output[j2*n1 + j1] instead of matrix[j1*n2 + j2]
///   3. Applies psi_inv^(j2*n1 + j1) to get the original coefficient
const ROW_INTT_UNTWIST_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    n1: u32,
    n2: u32,
    log_n2: u32,
    num_batches: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> inv_twiddles: array<u32>;  // k * n2 inverse twiddles
@group(0) @binding(4) var<storage, read> psi_inv_powers: array<u32>;  // k * n psi_inv^i
@group(0) @binding(5) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 256>;
var<workgroup> shared_hi: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn row_intt_untwist(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let row_idx = wg_id.x;   // j1 in transposed matrix
    let batch_idx = wg_id.y;
    let mod_idx = wg_id.z;

    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let log_n2 = params.log_n2;
    let k = params.num_moduli;

    if row_idx >= n1 { return; }
    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);
    let n_inv = vec2<u32>(mp.n_inv_lo, mp.n_inv_hi);

    let tw_mod_offset = mod_idx * n2 * 2u;
    let batch_mod_offset = (batch_idx * k + mod_idx) * n;
    let input_row_offset = batch_mod_offset + row_idx * n2;

    // Load row with bit-reversal
    let col = tid;  // j2 in transposed matrix
    if col < n2 {
        let in_base = (input_row_offset + col) * 2u;
        let val = vec2<u32>(input[in_base], input[in_base + 1u]);
        let rev_col = math::bit_reverse(col, log_n2);
        shared_lo[rev_col] = val.x;
        shared_hi[rev_col] = val.y;
    }
    workgroupBarrier();

    // INTT butterfly stages
    for (var stage = 0u; stage < log_n2; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let butterfly_idx = tid;
        if butterfly_idx < n2 / 2u {
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n2 / m);
            let tw_base = tw_mod_offset + twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::mulmod_60bit(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Apply n^-1 scaling, psi_inv untwist, and un-transpose, then store
    if col < n2 {
        var val = vec2<u32>(shared_lo[col], shared_hi[col]);

        // Scale by n^-1
        val = math::mulmod_60bit(val, n_inv, q);

        // Un-transpose: the original coefficient index is j2*n1 + j1 = col*n1 + row_idx
        let original_idx = col * n1 + row_idx;

        // Untwist by psi_inv^original_idx
        let psi_inv_base = (mod_idx * n + original_idx) * 2u;
        let psi_inv = vec2<u32>(psi_inv_powers[psi_inv_base], psi_inv_powers[psi_inv_base + 1u]);
        val = math::mulmod_60bit(val, psi_inv, q);

        // Write to un-transposed position: output[batch_mod_offset + original_idx]
        let out_base = (batch_mod_offset + original_idx) * 2u;
        output[out_base] = val.x;
        output[out_base + 1u] = val.y;
    }
}
"#;

/// Pointwise multiplication shader - multiplies pt_ntt with CT coefficients.
/// pt_ntt is in four-step NTT order: pt_ntt[k1*n2 + k2] corresponds to standard NTT index [k1 + k2*n1]
/// CT is in standard NTT order, so we need to permute the CT access.
const POINTWISE_MUL_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    n1: u32,           // 32 for four-step (number of rows)
    n2: u32,           // 256 for four-step (number of columns)
    num_batches: u32,
    num_moduli: u32,
    num_cts: u32,
    batches_per_ct: u32,
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pt_ntt: array<u32>;
@group(0) @binding(2) var<storage, read> ct_c0: array<u32>;
@group(0) @binding(3) var<storage, read> ct_c1: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_c0: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_c1: array<u32>;
@group(0) @binding(6) var<storage, read> mod_params: array<ModulusParams>;

@compute @workgroup_size(256, 1, 1)
fn pointwise_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let batch_idx = wg_id.y;
    let mod_idx = wg_id.z;
    let n = params.n;
    let n1 = params.n1;
    let n2 = params.n2;
    let k = params.num_moduli;

    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let idx = global_id.x;
    if idx >= n { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);

    // Determine which CT this batch belongs to
    let ct_idx = batch_idx / params.batches_per_ct;

    // pt_ntt at [(batch_idx * k + mod_idx) * n + idx]
    // With twist-transpose, the four-step NTT output is in STANDARD order
    // (the transpose in twist and un-transpose in INTT cancel out)
    let pt_offset = (batch_idx * k + mod_idx) * n + idx;
    let pt_base = pt_offset * 2u;
    let pt = vec2<u32>(pt_ntt[pt_base], pt_ntt[pt_base + 1u]);

    // CT at [(ct_idx * k + mod_idx) * n + idx] (same index as pt_ntt, no permutation needed)
    let ct_offset = (ct_idx * k + mod_idx) * n + idx;
    let ct_base = ct_offset * 2u;
    let c0 = vec2<u32>(ct_c0[ct_base], ct_c0[ct_base + 1u]);
    let c1 = vec2<u32>(ct_c1[ct_base], ct_c1[ct_base + 1u]);

    // Multiply
    let prod_c0 = math::mulmod_60bit(pt, c0, q);
    let prod_c1 = math::mulmod_60bit(pt, c1, q);

    // Output at [(batch_idx * k + mod_idx) * n + idx]
    let out_base = pt_base;  // Same offset as pt
    out_c0[out_base] = prod_c0.x;
    out_c0[out_base + 1u] = prod_c0.y;
    out_c1[out_base] = prod_c1.x;
    out_c1[out_base + 1u] = prod_c1.y;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rns_slot_mul::RnsBatchParams;
    use crate::rns_slot_mul_v2::RnsSlotMulGpuV2;

    #[test]
    fn test_v3_context_creation() {
        // Use the standard Goldilocks params
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };

        let result = RnsSlotMulGpuV3::new(params);
        match result {
            Ok(ctx) => {
                assert_eq!(ctx.n1, 32);
                assert_eq!(ctx.n2, 256);
                println!("V3 context created successfully with n1={}, n2={}", ctx.n1, ctx.n2);
            }
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
            }
        }
    }

    #[test]
    fn test_v3_vs_v2_comparison() {
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };
        let n = params.n;
        let k = params.k;

        // Create V2 context
        let v2_ctx = match RnsSlotMulGpuV2::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V2 GPU not available ({})", e);
                return;
            }
        };

        // Create V3 context
        let v3_ctx = match RnsSlotMulGpuV3::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
                return;
            }
        };

        println!("V2 and V3 contexts created successfully");
        println!("V3 uses four-step NTT with n1={} rows × n2={} cols", v3_ctx.n1, v3_ctx.n2);

        // Create test data
        let num_batches = 2;
        let t = params.plaintext_data.t;

        let slots: Vec<Vec<u64>> = (0..num_batches)
            .map(|batch_idx| {
                (0..n)
                    .map(|j| ((batch_idx * 1000 + j) as u64) % t)
                    .collect()
            })
            .collect();

        let cts_c0: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|mod_idx| {
                let q = params.rns_data[mod_idx].modulus;
                (0..n).map(|j| ((100 + j) as u64) % q).collect()
            })
            .collect()];

        let cts_c1: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|mod_idx| {
                let q = params.rns_data[mod_idx].modulus;
                (0..n).map(|j| ((200 + j) as u64) % q).collect()
            })
            .collect()];

        let batches_per_ct = vec![num_batches];

        // Test V2
        let (v2_c0, v2_c1) = match v2_ctx.mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct) {
            Ok(result) => {
                println!("V2 mul_batched_multi_ct succeeded");
                println!("  Output shape: {} batches × {} moduli × {} elements", result.0.len(), result.0[0].len(), result.0[0][0].len());
                result
            }
            Err(e) => {
                println!("V2 mul_batched_multi_ct failed: {}", e);
                return;
            }
        };

        // Test V3
        let (v3_c0, v3_c1) = match v3_ctx.mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct) {
            Ok(result) => {
                println!("V3 mul_batched_multi_ct succeeded");
                println!("  Output shape: {} batches × {} moduli × {} elements", result.0.len(), result.0[0].len(), result.0[0][0].len());
                result
            }
            Err(e) => {
                println!("V3 mul_batched_multi_ct failed: {}", e);
                return;
            }
        };

        // Compare results
        let mut total_mismatches = 0;
        let mut first_mismatch: Option<(usize, usize, usize, u64, u64)> = None;

        for batch_idx in 0..num_batches {
            for mod_idx in 0..k {
                for j in 0..n {
                    if v2_c0[batch_idx][mod_idx][j] != v3_c0[batch_idx][mod_idx][j] {
                        total_mismatches += 1;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((batch_idx, mod_idx, j, v2_c0[batch_idx][mod_idx][j], v3_c0[batch_idx][mod_idx][j]));
                        }
                    }
                    if v2_c1[batch_idx][mod_idx][j] != v3_c1[batch_idx][mod_idx][j] {
                        total_mismatches += 1;
                    }
                }
            }
        }

        if total_mismatches == 0 {
            println!("✓ V3 output matches V2 exactly!");
        } else {
            println!("✗ V3 vs V2 mismatch: {} total differences", total_mismatches);
            if let Some((b, m, j, v2_val, v3_val)) = first_mismatch {
                println!("  First mismatch at batch={}, mod={}, idx={}: V2={} vs V3={}", b, m, j, v2_val, v3_val);
            }
            // Print a few sample values for debugging
            println!("  Sample V2 c0[0][0][0..5]: {:?}", &v2_c0[0][0][0..5]);
            println!("  Sample V3 c0[0][0][0..5]: {:?}", &v3_c0[0][0][0..5]);

            // Check if V3's values exist in V2 at different indices (permutation search)
            println!("\n  Searching for V3 values in V2 output (batch=0, mod=0, first 10 V3 indices):");
            let n1 = 32usize;
            let n2 = 256usize;
            for v3_idx in 0..std::cmp::min(10, n) {
                let v3_val = v3_c0[0][0][v3_idx];
                let mut found_at: Option<usize> = None;
                for v2_idx in 0..n {
                    if v2_c0[0][0][v2_idx] == v3_val {
                        found_at = Some(v2_idx);
                        break;
                    }
                }
                match found_at {
                    Some(v2_idx) => {
                        let v3_k1 = v3_idx / n2;
                        let v3_k2 = v3_idx % n2;
                        let v2_k1 = v2_idx / n2;
                        let v2_k2 = v2_idx % n2;
                        println!("    V3[{}] (k1={},k2={}) = {} found at V2[{}] (k1={},k2={})",
                                 v3_idx, v3_k1, v3_k2, v3_val, v2_idx, v2_k1, v2_k2);
                    }
                    None => {
                        println!("    V3[{}] = {} NOT FOUND in V2 output", v3_idx, v3_val);
                    }
                }
            }
        }
    }

    /// CPU reference implementation of four-step NTT for verification.
    /// Returns the NTT result using the four-step algorithm.
    fn cpu_four_step_ntt(input: &[u64], n: usize, n1: usize, n2: usize, omega: u64, q: u64) -> Vec<u64> {
        assert_eq!(n, n1 * n2);
        let mut data = input.to_vec();

        // Helper: modular multiplication
        let mulmod = |a: u64, b: u64| -> u64 {
            ((a as u128 * b as u128) % q as u128) as u64
        };

        // Helper: modular addition (handles overflow)
        let addmod = |a: u64, b: u64| -> u64 {
            let (sum, overflow) = a.overflowing_add(b);
            if overflow || sum >= q { sum.wrapping_sub(q) } else { sum }
        };

        // Helper: modular subtraction (handles underflow)
        let submod = |a: u64, b: u64| -> u64 {
            if a >= b { a - b } else { a.wrapping_add(q).wrapping_sub(b) }
        };

        // Helper: compute omega^exp mod q
        let pow_mod = |base: u64, mut exp: u64| -> u64 {
            let mut result = 1u64;
            let mut base = base;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = mulmod(result, base);
                }
                base = mulmod(base, base);
                exp >>= 1;
            }
            result
        };

        // Helper: bit-reverse
        let bit_reverse = |x: usize, bits: usize| -> usize {
            let mut result = 0;
            let mut x = x;
            for _ in 0..bits {
                result = (result << 1) | (x & 1);
                x >>= 1;
            }
            result
        };

        // Helper: single NTT (DIT Cooley-Tukey)
        let ntt_inplace = |data: &mut [u64], omega_n: u64| {
            let n = data.len();
            let log_n = (n as f64).log2() as usize;

            // Bit-reverse permutation
            for i in 0..n {
                let j = bit_reverse(i, log_n);
                if i < j {
                    data.swap(i, j);
                }
            }

            // Butterfly stages
            for stage in 0..log_n {
                let m = 1 << (stage + 1);
                let half_m = 1 << stage;
                let step = n / m;
                let twiddle_base = pow_mod(omega_n, step as u64);

                for group in 0..(n / m) {
                    let mut twiddle = 1u64;
                    for j in 0..half_m {
                        let idx1 = group * m + j;
                        let idx2 = idx1 + half_m;
                        let u = data[idx1];
                        let v = mulmod(data[idx2], twiddle);
                        data[idx1] = addmod(u, v);
                        data[idx2] = submod(u, v);
                        twiddle = mulmod(twiddle, twiddle_base);
                    }
                }
            }
        };

        // omega_n1 = omega^n2 (primitive n1-th root of unity, for column n1-point NTT)
        let omega_n1 = pow_mod(omega, n2 as u64);
        // omega_n2 = omega^n1 (primitive n2-th root of unity, for row n2-point NTT)
        let omega_n2 = pow_mod(omega, n1 as u64);

        println!("CPU four-step NTT:");
        println!("  n={}, n1={}, n2={}", n, n1, n2);
        println!("  omega = {}", omega);
        println!("  omega_n1 = omega^{} = {}", n2, omega_n1);
        println!("  omega_n2 = omega^{} = {}", n1, omega_n2);

        // Correct four-step NTT algorithm:
        // Input is viewed as n1×n2 matrix in COLUMN-MAJOR order:
        //   A[j1][j2] = x[j2*n1 + j1]   where j1 in 0..n1, j2 in 0..n2
        //
        // This is equivalent to transposing the input if it's stored row-major.
        //
        // Algorithm:
        // 1. Compute n1 row-wise n2-point DFTs using omega_n2 = omega^n1
        // 2. Multiply by cross-twiddles omega^{j1*k2}
        // 3. Compute n2 column-wise n1-point DFTs using omega_n1 = omega^n2
        // 4. Output is in row-major order: X[k1*n2 + k2]

        // Step 0: Rearrange input from row-major x[j] to column-major A[j1][j2]
        // where j = j1*n2 + j2 (row-major) becomes A[j1][j2] = x[j2*n1 + j1] (column-major)
        // In memory as row-major n1×n2: A_flat[row*n2 + col] = x[col*n1 + row]
        let mut matrix = vec![0u64; n];
        for j1 in 0..n1 {
            for j2 in 0..n2 {
                // Column-major: A[j1][j2] = input[j2*n1 + j1]
                // Store in row-major: matrix[j1*n2 + j2]
                matrix[j1 * n2 + j2] = data[j2 * n1 + j1];
            }
        }

        // Step 1: Row NTTs (n1 independent n2-point NTTs using omega_n2)
        for row in 0..n1 {
            let start = row * n2;
            let mut row_data: Vec<u64> = matrix[start..start + n2].to_vec();
            ntt_inplace(&mut row_data, omega_n2);
            matrix[start..start + n2].copy_from_slice(&row_data);
        }

        // Step 2: Cross twiddle multiplication by omega^(j1 * k2)
        // j1 = row index, k2 = column index (frequency after row NTT)
        for j1 in 0..n1 {
            for k2 in 0..n2 {
                let twiddle = pow_mod(omega, (j1 * k2) as u64);
                let idx = j1 * n2 + k2;
                matrix[idx] = mulmod(matrix[idx], twiddle);
            }
        }

        // Step 3: Column NTTs (n2 independent n1-point NTTs using omega_n1)
        for k2 in 0..n2 {
            let mut col_data: Vec<u64> = (0..n1).map(|j1| matrix[j1 * n2 + k2]).collect();
            ntt_inplace(&mut col_data, omega_n1);
            for k1 in 0..n1 {
                matrix[k1 * n2 + k2] = col_data[k1];
            }
        }

        // Output is already in row-major order: matrix[k1*n2 + k2] = X[k1*n2 + k2]
        matrix
    }

    /// Direct NTT (not four-step) for comparison.
    fn cpu_direct_ntt(input: &[u64], n: usize, omega: u64, q: u64) -> Vec<u64> {
        let mulmod = |a: u64, b: u64| -> u64 {
            ((a as u128 * b as u128) % q as u128) as u64
        };
        let addmod = |a: u64, b: u64| -> u64 {
            let (sum, overflow) = a.overflowing_add(b);
            if overflow || sum >= q { sum.wrapping_sub(q) } else { sum }
        };
        let submod = |a: u64, b: u64| -> u64 {
            if a >= b { a - b } else { a.wrapping_add(q).wrapping_sub(b) }
        };
        let pow_mod = |base: u64, mut exp: u64| -> u64 {
            let mut result = 1u64;
            let mut base = base;
            while exp > 0 {
                if exp & 1 == 1 { result = mulmod(result, base); }
                base = mulmod(base, base);
                exp >>= 1;
            }
            result
        };
        let bit_reverse = |x: usize, bits: usize| -> usize {
            let mut result = 0;
            let mut x = x;
            for _ in 0..bits { result = (result << 1) | (x & 1); x >>= 1; }
            result
        };

        let mut data = input.to_vec();
        let log_n = (n as f64).log2() as usize;

        // Bit-reverse permutation
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j { data.swap(i, j); }
        }

        // Butterfly stages
        for stage in 0..log_n {
            let m = 1 << (stage + 1);
            let half_m = 1 << stage;
            let step = n / m;
            let twiddle_base = pow_mod(omega, step as u64);

            for group in 0..(n / m) {
                let mut twiddle = 1u64;
                for j in 0..half_m {
                    let idx1 = group * m + j;
                    let idx2 = idx1 + half_m;
                    let u = data[idx1];
                    let v = mulmod(data[idx2], twiddle);
                    data[idx1] = addmod(u, v);
                    data[idx2] = submod(u, v);
                    twiddle = mulmod(twiddle, twiddle_base);
                }
            }
        }

        data
    }

    #[test]
    fn test_cpu_four_step_small() {
        // Small example for debugging: n=8 = 2 × 4
        // Use q=17, primitive 8th root of unity is 2 (2^8 = 256 ≡ 1 mod 17)
        let q: u64 = 17;
        let n = 8;
        let n1 = 2; // rows
        let n2 = 4; // columns
        let omega: u64 = 2; // primitive 8th root of unity mod 17

        // Verify omega is an 8th root: omega^8 = 1 mod 17
        let omega8 = (0..8).fold(1u64, |acc, _| (acc * omega) % q);
        assert_eq!(omega8, 1, "omega should be 8th root of unity");
        let omega4 = (0..4).fold(1u64, |acc, _| (acc * omega) % q);
        assert_ne!(omega4, 1, "omega^4 should not be 1");

        println!("\n=== Small Four-Step NTT Test (n=8 = 2×4) ===");
        println!("q={}, omega={}", q, omega);

        // Simple input
        let input: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        println!("Input: {:?}", input);

        // Direct DFT (not FFT) for comparison
        let direct_dft: Vec<u64> = (0..n).map(|k| {
            (0..n).fold(0u64, |acc, j| {
                let omega_jk = (0..(j*k)).fold(1u64, |a, _| (a * omega) % q);
                (acc + input[j] * omega_jk) % q
            })
        }).collect();
        println!("Direct DFT: {:?}", direct_dft);

        // Now trace through four-step manually
        // Input layout as 2×4 matrix (row-major):
        //   Row 0: [1, 2, 3, 4]  (indices 0,1,2,3)
        //   Row 1: [5, 6, 7, 8]  (indices 4,5,6,7)

        // omega_n2 = omega^(n/n2) = omega^2 (4th root of unity)
        let omega_n2 = (omega * omega) % q;
        // omega_n1 = omega^(n/n1) = omega^4 (2nd root of unity)
        let omega_n1 = (0..4).fold(1u64, |acc, _| (acc * omega) % q);
        println!("omega_n2 = omega^2 = {} (4th root)", omega_n2);
        println!("omega_n1 = omega^4 = {} (2nd root)", omega_n1);

        // Our four-step implementation
        let fourstep = cpu_four_step_ntt(&input, n, n1, n2, omega, q);
        println!("Four-step result: {:?}", fourstep);

        // Direct NTT (FFT-style)
        let direct_fft = cpu_direct_ntt(&input, n, omega, q);
        println!("Direct FFT result: {:?}", direct_fft);

        // Check which indices match
        println!("\nComparison:");
        for i in 0..n {
            let matches = if fourstep[i] == direct_fft[i] { "✓" } else { "✗" };
            println!("  [{}]: fourstep={}, direct_fft={}, direct_dft={} {}",
                     i, fourstep[i], direct_fft[i], direct_dft[i], matches);
        }

        // The four-step result might be in a different index order
        // Try to find the permutation
        println!("\nLooking for permutation:");
        for i in 0..n {
            for j in 0..n {
                if fourstep[i] == direct_fft[j] {
                    println!("  fourstep[{}] = direct_fft[{}] = {}", i, j, fourstep[i]);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_cpu_four_step_vs_direct() {
        // Use a small example first for debugging
        let q: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
        let n = 8192;
        let n1 = 32;
        let n2 = 256;

        // Find primitive n-th root of unity
        // For Goldilocks, omega = 7^((q-1)/n)
        let omega = {
            let exp = (q - 1) / n as u64;
            let mut result = 1u64;
            let mut base = 7u64;
            let mut e = exp;
            while e > 0 {
                if e & 1 == 1 {
                    result = ((result as u128 * base as u128) % q as u128) as u64;
                }
                base = ((base as u128 * base as u128) % q as u128) as u64;
                e >>= 1;
            }
            result
        };

        println!("Testing CPU four-step NTT vs direct NTT");
        println!("n={}, n1={}, n2={}, q={}, omega={}", n, n1, n2, q, omega);

        // Simple input: [1, 2, 3, ..., n]
        let input: Vec<u64> = (1..=n as u64).collect();

        let direct_result = cpu_direct_ntt(&input, n, omega, q);
        let fourstep_result = cpu_four_step_ntt(&input, n, n1, n2, omega, q);

        // Compare
        let mut mismatches = 0;
        for i in 0..n {
            if direct_result[i] != fourstep_result[i] {
                mismatches += 1;
                if mismatches <= 5 {
                    println!("  Mismatch at {}: direct={} vs fourstep={}", i, direct_result[i], fourstep_result[i]);
                }
            }
        }

        if mismatches == 0 {
            println!("✓ CPU four-step NTT matches direct NTT!");
        } else {
            println!("✗ {} mismatches between four-step and direct NTT", mismatches);
            println!("  Direct[0..5]: {:?}", &direct_result[0..5]);
            println!("  FourStep[0..5]: {:?}", &fourstep_result[0..5]);
        }

        assert_eq!(mismatches, 0, "Four-step NTT should match direct NTT");
    }

    /// Test that V3's forward NTT followed by INTT gives back the original (identity test).
    /// This isolates the NTT/INTT correctness from the slot encoding and pointwise mul.
    #[test]
    fn test_v3_ntt_identity() {
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };

        let v3_ctx = match RnsSlotMulGpuV3::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
                return;
            }
        };

        println!("Testing V3 NTT identity (NTT followed by INTT should give original)");

        // Create simple test data: all 1s for slot 0, 0s elsewhere
        let n = params.n;
        let k = params.k;
        let t = params.plaintext_data.t;

        // Simple test: slots[i] = i % t
        let slots: Vec<Vec<u64>> = vec![(0..n).map(|i| (i as u64) % t).collect()];

        // Create identity ciphertext (c0 = NTT(1), c1 = NTT(0))
        // For identity test, we use ct[i] = 1 for all i in NTT domain
        let cts_c0: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|_| vec![1u64; n])
            .collect()];
        let cts_c1: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|_| vec![0u64; n])
            .collect()];

        let batches_per_ct = vec![1];

        // Run V3
        let result = v3_ctx.mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct);
        match result {
            Ok((c0_out, _c1_out)) => {
                // When multiplied by NTT(1), the result should be NTT(plaintext) after INTT
                // which equals the original plaintext coefficients
                println!("V3 mul succeeded");
                println!("  Output c0[0][0][0..10]: {:?}", &c0_out[0][0][0..10]);

                // The output should be related to the input slots
                // After slot_encode (INTT), we get coefficients
                // After NTT, pointwise mul by 1, INTT, we should get back coefficients
            }
            Err(e) => {
                println!("V3 mul failed: {}", e);
            }
        }
    }

    #[test]
    fn test_v3_ntt_intt_roundtrip() {
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };
        let n = params.n;
        let k = params.k;
        let t = params.plaintext_data.t;

        let v3_ctx = match RnsSlotMulGpuV3::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
                return;
            }
        };

        println!("Testing V3 NTT -> INTT round-trip");

        // Create test slots
        let slots: Vec<Vec<u64>> = vec![(0..n).map(|i| (i as u64) % t).collect()];

        // First, get the slot_encode output (this is what we expect after NTT -> INTT)
        let encoded = v3_ctx.test_slot_encode_only(&slots).expect("slot encode failed");
        println!("  slot_encode produced {} values", encoded[0].len());

        // Run NTT -> INTT round-trip
        let roundtrip = v3_ctx.test_ntt_intt_roundtrip(&slots).expect("round-trip failed");
        println!("  round-trip produced {} batches × {} moduli × {} values",
                 roundtrip.len(), roundtrip[0].len(), roundtrip[0][0].len());

        // Compare with slot_encode output
        // The round-trip output should match the encoded values reduced mod each q
        let mod_idx = 0;  // Test first modulus
        let q = params.rns_data[mod_idx].modulus;

        let mut mismatches = 0;
        let mut first_mismatch: Option<(usize, u64, u64)> = None;
        for i in 0..n {
            // Reduce encoded value mod q for comparison
            let expected = encoded[0][i] % q;
            let actual = roundtrip[0][mod_idx][i];
            if expected != actual {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, expected, actual));
                }
            }
        }

        if mismatches == 0 {
            println!("  ✓ NTT -> INTT round-trip matches encoded (mod q): all {} elements correct", n);
        } else {
            println!("  ✗ NTT -> INTT round-trip: {} mismatches out of {}", mismatches, n);
            if let Some((i, exp, act)) = first_mismatch {
                println!("    First mismatch at i={}: expected {} (encoded mod q), got {}", i, exp, act);
            }
            println!("    encoded[0..5] mod q: {:?}", &encoded[0][0..5].iter().map(|&x| x % q).collect::<Vec<_>>());
            println!("    roundtrip[0..5]:     {:?}", &roundtrip[0][mod_idx][0..5]);
        }
    }

    #[test]
    fn test_v3_full_pipeline_with_identity_ct() {
        // This test verifies the full pipeline by multiplying with identity ciphertext
        // (all 1s in NTT form). Result should equal the NTT->INTT round-trip.
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };
        let n = params.n;
        let k = params.k;
        let t = params.plaintext_data.t;

        let v3_ctx = match RnsSlotMulGpuV3::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
                return;
            }
        };

        println!("Testing V3 full pipeline with identity ciphertext");

        // Create test slots
        let slots: Vec<Vec<u64>> = vec![(0..n).map(|i| (i as u64) % t).collect()];

        // Get the NTT -> INTT round-trip result (expected)
        let roundtrip = v3_ctx.test_ntt_intt_roundtrip(&slots).expect("round-trip failed");
        println!("  Round-trip reference computed");

        // Create identity ciphertext: all 1s (multiplying by 1 in NTT domain is like
        // multiplying by a polynomial that has all coefficients = 1 in coefficient domain,
        // which is NOT the identity. But it should still be consistent.)
        // For true identity, we need ct[i] = 1 for all i, which corresponds to
        // INTT([1,0,0,...]) in coefficient domain. Actually, this is complex.
        //
        // Simpler test: use ct that encodes the same value at each position,
        // then verify the output is consistent.
        //
        // Even simpler: compare V3 full pipeline with round-trip when ct = all 1s
        let cts_c0: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|_| vec![1u64; n])
            .collect()];
        let cts_c1: Vec<Vec<Vec<u64>>> = vec![(0..k)
            .map(|_| vec![0u64; n])
            .collect()];
        let batches_per_ct = vec![1];

        // Run full pipeline
        let (c0_out, _c1_out) = v3_ctx.mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct)
            .expect("full pipeline failed");
        println!("  Full pipeline completed");

        // Compare c0_out with roundtrip
        // Since we multiplied by ct=1 (in NTT form), the result is:
        // INTT(NTT(pt) * 1) = INTT(NTT(pt)) = pt (coefficients)
        // So c0_out should equal roundtrip (which is also INTT(NTT(pt)))
        let mod_idx = 0;
        let mut mismatches = 0;
        let mut first_mismatch: Option<(usize, u64, u64)> = None;
        for i in 0..n {
            let expected = roundtrip[0][mod_idx][i];
            let actual = c0_out[0][mod_idx][i];
            if expected != actual {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, expected, actual));
                }
            }
        }

        if mismatches == 0 {
            println!("  ✓ Full pipeline with ct=1 matches round-trip: all {} elements correct", n);
        } else {
            println!("  ✗ Full pipeline with ct=1 vs round-trip: {} mismatches", mismatches);
            if let Some((i, exp, act)) = first_mismatch {
                println!("    First mismatch at i={}: expected {}, got {}", i, exp, act);
            }
            println!("    roundtrip[0..5]:  {:?}", &roundtrip[0][mod_idx][0..5]);
            println!("    full_pipe[0..5]:  {:?}", &c0_out[0][mod_idx][0..5]);
        }
    }

    // Helper: CPU NTT in-place using DIT Cooley-Tukey
    fn cpu_ntt_inplace(data: &mut [u64], omega_n: u64, q: u64) {
        let n = data.len();
        let log_n = (n as f64).log2() as usize;

        let mulmod = |a: u64, b: u64| -> u64 {
            ((a as u128 * b as u128) % q as u128) as u64
        };
        let addmod = |a: u64, b: u64| -> u64 {
            let sum = a + b;
            if sum >= q { sum - q } else { sum }
        };
        let submod = |a: u64, b: u64| -> u64 {
            if a >= b { a - b } else { a.wrapping_add(q).wrapping_sub(b) }
        };
        let pow_mod = |base: u64, mut exp: u64| -> u64 {
            let mut result = 1u64;
            let mut base = base;
            while exp > 0 {
                if exp & 1 == 1 { result = mulmod(result, base); }
                base = mulmod(base, base);
                exp >>= 1;
            }
            result
        };
        let bit_reverse = |x: usize, bits: usize| -> usize {
            let mut result = 0;
            let mut x = x;
            for _ in 0..bits { result = (result << 1) | (x & 1); x >>= 1; }
            result
        };

        // Bit-reverse permutation
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j { data.swap(i, j); }
        }

        // Butterfly stages
        for stage in 0..log_n {
            let m = 1 << (stage + 1);
            let half_m = 1 << stage;
            let step = n / m;
            let twiddle_base = pow_mod(omega_n, step as u64);

            for group in 0..(n / m) {
                let mut twiddle = 1u64;
                for j in 0..half_m {
                    let idx1 = group * m + j;
                    let idx2 = idx1 + half_m;
                    let u = data[idx1];
                    let v = mulmod(data[idx2], twiddle);
                    data[idx1] = addmod(u, v);
                    data[idx2] = submod(u, v);
                    twiddle = mulmod(twiddle, twiddle_base);
                }
            }
        }
    }

    // Helper: CPU standard NTT (not four-step, for reference)
    fn cpu_standard_ntt(data: &mut [u64], omega: u64, psi_powers: &[u64], q: u64) {
        let n = data.len();
        let log_n = (n as f64).log2() as usize;

        let mulmod = |a: u64, b: u64| -> u64 {
            ((a as u128 * b as u128) % q as u128) as u64
        };
        let addmod = |a: u64, b: u64| -> u64 {
            let sum = a + b;
            if sum >= q { sum - q } else { sum }
        };
        let submod = |a: u64, b: u64| -> u64 {
            if a >= b { a - b } else { a.wrapping_add(q).wrapping_sub(b) }
        };
        let pow_mod = |base: u64, mut exp: u64| -> u64 {
            let mut result = 1u64;
            let mut base = base;
            while exp > 0 {
                if exp & 1 == 1 { result = mulmod(result, base); }
                base = mulmod(base, base);
                exp >>= 1;
            }
            result
        };
        let bit_reverse = |x: usize, bits: usize| -> usize {
            let mut result = 0;
            let mut x = x;
            for _ in 0..bits { result = (result << 1) | (x & 1); x >>= 1; }
            result
        };

        // Twist by psi
        for i in 0..n {
            data[i] = mulmod(data[i], psi_powers[i]);
        }

        // Bit-reverse permutation
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j { data.swap(i, j); }
        }

        // Butterfly stages
        for stage in 0..log_n {
            let m = 1 << (stage + 1);
            let half_m = 1 << stage;
            let step = n / m;
            let twiddle_base = pow_mod(omega, step as u64);

            for group in 0..(n / m) {
                let mut twiddle = 1u64;
                for j in 0..half_m {
                    let idx1 = group * m + j;
                    let idx2 = idx1 + half_m;
                    let u = data[idx1];
                    let v = mulmod(data[idx2], twiddle);
                    data[idx1] = addmod(u, v);
                    data[idx2] = submod(u, v);
                    twiddle = mulmod(twiddle, twiddle_base);
                }
            }
        }
    }

    #[test]
    fn test_four_step_vs_standard_cpu() {
        // This test verifies that the four-step NTT algorithm produces the same
        // result as the standard NTT (up to a permutation)
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };
        let n = params.n;
        let mod_idx = 0;
        let q = params.rns_data[mod_idx].modulus;
        let omega = params.rns_data[mod_idx].omega;
        let psi_powers = &params.rns_data[mod_idx].psi_powers;
        let omega_powers = &params.rns_data[mod_idx].omega_powers;

        println!("\n=== Testing Four-Step vs Standard CPU NTT ===");
        println!("n = {}, n1 = 32, n2 = 256", n);
        println!("q = {}", q);

        // Simple test input
        let input: Vec<u64> = (0..n).map(|i| (i as u64) % q).collect();

        // Standard CPU NTT
        let mut standard_result = input.clone();
        cpu_standard_ntt(&mut standard_result, omega, psi_powers, q);
        println!("Standard NTT[0..5]: {:?}", &standard_result[0..5]);

        // Four-step CPU NTT (same as in test_v3_forward_ntt_vs_v2)
        let n1 = 32usize;
        let n2 = 256usize;

        // Step 1: twist + transpose
        let mut four_step_data = vec![0u64; n];
        for idx in 0..n {
            let j1 = idx / n2;
            let j2 = idx % n2;
            let transposed_idx = j2 * n1 + j1;
            let val = input[transposed_idx];
            let twisted = ((val as u128 * psi_powers[transposed_idx] as u128) % q as u128) as u64;
            four_step_data[idx] = twisted;
        }

        // Step 2: row NTTs (n1=32 rows of n2=256 elements)
        let omega_n2 = omega_powers[n1];  // omega^32, primitive 256th root
        for row in 0..n1 {
            let start = row * n2;
            let mut row_data: Vec<u64> = four_step_data[start..start + n2].to_vec();
            cpu_ntt_inplace(&mut row_data, omega_n2, q);
            four_step_data[start..start + n2].copy_from_slice(&row_data);
        }

        // Step 3: cross twiddle (omega^(j1 * k2))
        for j1 in 0..n1 {
            for k2 in 0..n2 {
                let idx = j1 * n2 + k2;
                let twiddle_power = (j1 * k2) % n;
                let twiddle = omega_powers[twiddle_power];
                four_step_data[idx] = ((four_step_data[idx] as u128 * twiddle as u128) % q as u128) as u64;
            }
        }

        // Step 4: column NTTs (n2=256 columns of n1=32 elements)
        let omega_n1 = omega_powers[n2];  // omega^256, primitive 32nd root
        for k2 in 0..n2 {
            let mut col_data: Vec<u64> = (0..n1).map(|j1| four_step_data[j1 * n2 + k2]).collect();
            cpu_ntt_inplace(&mut col_data, omega_n1, q);
            for k1 in 0..n1 {
                four_step_data[k1 * n2 + k2] = col_data[k1];
            }
        }
        println!("Four-step NTT[0..5]: {:?}", &four_step_data[0..5]);

        // Check if four-step is a permutation of standard
        // Four-step output[k1*n2 + k2] should equal standard[k1 + k2*n1]
        let mut matches = 0;
        let mut first_mismatch: Option<(usize, u64, u64)> = None;
        for k1 in 0..n1 {
            for k2 in 0..n2 {
                let four_step_idx = k1 * n2 + k2;
                let standard_idx = k1 + k2 * n1;
                if four_step_data[four_step_idx] == standard_result[standard_idx] {
                    matches += 1;
                } else if first_mismatch.is_none() {
                    first_mismatch = Some((four_step_idx, four_step_data[four_step_idx], standard_result[standard_idx]));
                }
            }
        }

        println!("Permutation matches: {}/{}", matches, n);
        if matches == n {
            println!("✓ Four-step NTT is correct (output is permutation of standard)");
        } else {
            println!("✗ Four-step doesn't match standard after permutation");
            if let Some((idx, four, std)) = first_mismatch {
                println!("  First mismatch at idx {}: four_step={} vs standard={}", idx, four, std);
            }
        }

        // Also check direct equality (which should NOT match due to permutation)
        let direct_matches: usize = (0..n).filter(|&i| four_step_data[i] == standard_result[i]).count();
        println!("Direct matches (without permutation): {}/{}", direct_matches, n);
        if direct_matches == n {
            println!("  (Unexpectedly, they match directly!)");
        } else {
            println!("  (As expected, four-step output order differs from standard)");
        }
    }

    #[test]
    fn test_v3_forward_ntt_vs_v2() {
        let params = match RnsBatchParams::goldilocks(8192) {
            Some(p) => p,
            None => {
                println!("Skipping test: goldilocks params not available");
                return;
            }
        };
        let n = params.n;
        let k = params.k;
        let t = params.plaintext_data.t;

        // Create V2 and V3 contexts
        let v2_ctx = match RnsSlotMulGpuV2::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V2 GPU not available ({})", e);
                return;
            }
        };
        let v3_ctx = match RnsSlotMulGpuV3::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V3 GPU not available ({})", e);
                return;
            }
        };

        println!("\n=== Testing V3 Forward NTT vs V2 ===");

        // Create simple test slots
        let slots: Vec<Vec<u64>> = vec![(0..n).map(|i| (i as u64) % t).collect()];

        // V2: Get slot_encode output
        let v2_encoded = v2_ctx.test_slot_encode(&slots).expect("V2 slot encode failed");
        println!("V2 slot encode completed, got {} batches × {} elements",
                 v2_encoded.len(), v2_encoded[0].len());

        // V3: Get slot_encode output (for comparison)
        let v3_encoded = v3_ctx.test_slot_encode_only(&slots).expect("V3 slot encode failed");
        println!("V3 slot encode completed, got {} batches × {} elements",
                 v3_encoded.len(), v3_encoded[0].len());

        // Compare slot_encode outputs
        let mut encode_mismatches = 0;
        for i in 0..n {
            if v2_encoded[0][i] != v3_encoded[0][i] {
                encode_mismatches += 1;
            }
        }
        if encode_mismatches == 0 {
            println!("✓ V3 slot_encode matches V2 exactly!");
        } else {
            println!("✗ V3 slot_encode vs V2: {} mismatches", encode_mismatches);
            println!("  V2 encoded[0..5]: {:?}", &v2_encoded[0][0..5]);
            println!("  V3 encoded[0..5]: {:?}", &v3_encoded[0][0..5]);
        }

        // V3: Test twist output (slot_encode + twist only)
        let v3_twisted = v3_ctx.test_twist_only(&slots).expect("V3 twist failed");
        println!("V3 twist completed, got {} batches × {} moduli × {} elements",
                 v3_twisted.len(), v3_twisted[0].len(), v3_twisted[0][0].len());

        // Compute CPU twist for comparison (for mod_idx=0)
        let n1 = 32usize;
        let n2 = 256usize;
        let mod_idx_check = 0;
        let q_check = params.rns_data[mod_idx_check].modulus;
        let psi_powers_check = &params.rns_data[mod_idx_check].psi_powers;
        let mut cpu_twisted_check = vec![0u64; n];
        for idx in 0..n {
            let j1 = idx / n2;
            let j2 = idx % n2;
            let transposed_idx = j2 * n1 + j1;
            let val = v2_encoded[0][transposed_idx];
            let psi_power = psi_powers_check[transposed_idx];
            let twisted = ((val as u128 * psi_power as u128) % q_check as u128) as u64;
            cpu_twisted_check[idx] = twisted;
        }

        // Check V3 twist output vs CPU (for mod_idx=0, batch=0)
        let mut twist_mismatches = 0;
        let mut first_twist_mismatch: Option<(usize, u64, u64)> = None;
        for idx in 0..n {
            if v3_twisted[0][mod_idx_check][idx] != cpu_twisted_check[idx] {
                twist_mismatches += 1;
                if first_twist_mismatch.is_none() {
                    first_twist_mismatch = Some((idx, v3_twisted[0][mod_idx_check][idx], cpu_twisted_check[idx]));
                }
            }
        }
        if twist_mismatches == 0 {
            println!("✓ V3 twist matches CPU exactly!");
        } else {
            println!("✗ V3 twist vs CPU: {} mismatches", twist_mismatches);
            if let Some((idx, gpu, cpu)) = first_twist_mismatch {
                println!("  First mismatch at idx {}: GPU={} vs CPU={}", idx, gpu, cpu);
            }
            // Show around index 256-260
            println!("  V3 twist[254..262]: {:?}", &v3_twisted[0][mod_idx_check][254..262]);
            println!("  CPU twist[254..262]: {:?}", &cpu_twisted_check[254..262]);
        }

        // Also check if twist output values are valid (< q)
        let twist_overflows: Vec<_> = v3_twisted[0][mod_idx_check].iter().enumerate()
            .filter(|(_, &v)| v >= q_check)
            .take(5)
            .collect();
        if twist_overflows.is_empty() {
            println!("  ✓ All V3 twist values are < q");
        } else {
            println!("  ✗ V3 twist has {} values >= q! Examples: {:?}", twist_overflows.len(), twist_overflows);
        }

        // V3: Get forward NTT output (includes slot_encode + twist + 4-step NTT)
        let v3_ntt = v3_ctx.test_forward_ntt_only(&slots).expect("V3 forward NTT failed");
        println!("V3 forward NTT completed, got {} batches × {} moduli × {} elements",
                 v3_ntt.len(), v3_ntt[0].len(), v3_ntt[0][0].len());

        // Run V2 forward NTT for each modulus
        let mut v2_ntt: Vec<Vec<u64>> = Vec::with_capacity(k);
        for mod_idx in 0..k {
            let ntt = v2_ctx.test_forward_ntt(&v2_encoded[0], mod_idx).expect("V2 forward NTT failed");
            v2_ntt.push(ntt);
        }
        println!("V2 forward NTT completed, got {} moduli × {} elements",
                 v2_ntt.len(), v2_ntt[0].len());

        // Also compute CPU four-step NTT for comparison
        // This uses the same algorithm as the V3 GPU implementation
        let mod_idx = 0;  // Use first modulus for testing
        let q = params.rns_data[mod_idx].modulus;
        let omega = params.rns_data[mod_idx].omega_powers[1];  // omega = omega_powers[1]
        let psi = params.rns_data[mod_idx].psi_powers[1];  // psi = psi_powers[1]

        println!("\nCPU Four-Step NTT Reference (mod_idx=0):");
        println!("  q = {}", q);
        println!("  omega = {}", omega);
        println!("  psi = {}", psi);

        // CPU: twist + transpose (matches twist shader)
        let n1 = 32usize;
        let n2 = 256usize;
        let mut cpu_twisted = vec![0u64; n];
        let psi_powers = &params.rns_data[mod_idx].psi_powers;
        let omega_powers = &params.rns_data[mod_idx].omega_powers;
        for idx in 0..n {
            let j1 = idx / n2;
            let j2 = idx % n2;
            let transposed_idx = j2 * n1 + j1;
            // temp[j1*n2 + j2] = encoded[j2*n1 + j1] * psi^(j2*n1 + j1)
            let val = v2_encoded[0][transposed_idx];
            let psi_power = psi_powers[transposed_idx];
            let twisted = ((val as u128 * psi_power as u128) % q as u128) as u64;
            cpu_twisted[idx] = twisted;
        }
        println!("  CPU twisted (first 5): {:?}", &cpu_twisted[0..5]);

        // CPU: row NTTs (n1=32 rows of n2=256 elements each)
        let omega_n2 = omega_powers[n1];  // omega^n1 = omega^32, primitive 256th root
        let mut cpu_after_row = cpu_twisted.clone();
        for row in 0..n1 {
            let start = row * n2;
            let mut row_data: Vec<u64> = cpu_after_row[start..start + n2].to_vec();
            // Do 256-point NTT with omega_n2
            cpu_ntt_inplace(&mut row_data, omega_n2, q);
            cpu_after_row[start..start + n2].copy_from_slice(&row_data);
        }
        println!("  CPU after row NTT (first 5): {:?}", &cpu_after_row[0..5]);

        // CPU: cross twiddle (multiply by omega^(j1 * k2))
        let mut cpu_after_cross = cpu_after_row.clone();
        for j1 in 0..n1 {
            for k2 in 0..n2 {
                let idx = j1 * n2 + k2;
                let twiddle_power = (j1 * k2) % n;
                let twiddle = omega_powers[twiddle_power];
                cpu_after_cross[idx] = ((cpu_after_cross[idx] as u128 * twiddle as u128) % q as u128) as u64;
            }
        }
        println!("  CPU after cross twiddle (first 5): {:?}", &cpu_after_cross[0..5]);

        // CPU: column NTTs (n2=256 columns of n1=32 elements each)
        let omega_n1 = omega_powers[n2];  // omega^n2 = omega^256, primitive 32nd root
        let mut cpu_after_col = cpu_after_cross.clone();
        for k2 in 0..n2 {
            let mut col_data: Vec<u64> = (0..n1).map(|j1| cpu_after_col[j1 * n2 + k2]).collect();
            cpu_ntt_inplace(&mut col_data, omega_n1, q);
            for k1 in 0..n1 {
                cpu_after_col[k1 * n2 + k2] = col_data[k1];
            }
        }
        println!("  CPU after col NTT (first 5): {:?}", &cpu_after_col[0..5]);

        // Verify all CPU values are < q
        let cpu_overflows: Vec<_> = cpu_after_col.iter().enumerate()
            .filter(|(_, &v)| v >= q)
            .take(5)
            .collect();
        if cpu_overflows.is_empty() {
            println!("  ✓ All CPU NTT values are < q");
        } else {
            println!("  ✗ CPU NTT has {} values >= q! Examples: {:?}", cpu_overflows.len(), cpu_overflows);
        }

        // Check if V2 values are > q (which would indicate wrong modulus or buffer issue)
        let v2_overflows: Vec<_> = v2_ntt[mod_idx].iter().enumerate()
            .filter(|(_, &v)| v >= q)
            .take(5)
            .collect();
        if v2_overflows.is_empty() {
            println!("  ✓ All V2 NTT values are < q");
        } else {
            println!("  ✗ V2 NTT has {} values >= q! Examples: {:?}", v2_overflows.len(), v2_overflows);
            println!("    This suggests V2 test helper is reading wrong data");
        }

        // Check V3 values
        let v3_overflows: Vec<_> = v3_ntt[0][mod_idx].iter().enumerate()
            .filter(|(_, &v)| v >= q)
            .take(5)
            .collect();
        if v3_overflows.is_empty() {
            println!("  ✓ All V3 NTT values are < q");
        } else {
            println!("  ✗ V3 NTT has {} values >= q! Examples: {:?}", v3_overflows.len(), v3_overflows);
        }

        // Compare CPU with V2 and V3
        println!("\nComparison with CPU reference:");
        println!("  V2 NTT[0][0..5]:  {:?}", &v2_ntt[mod_idx][0..5]);
        println!("  V3 NTT[0][0..5]:  {:?}", &v3_ntt[0][mod_idx][0..5]);
        println!("  CPU NTT[0..5]:    {:?}", &cpu_after_col[0..5]);

        let mut cpu_v2_match = 0;
        let mut cpu_v3_match = 0;
        for i in 0..n {
            if cpu_after_col[i] == v2_ntt[mod_idx][i] { cpu_v2_match += 1; }
            if cpu_after_col[i] == v3_ntt[0][mod_idx][i] { cpu_v3_match += 1; }
        }
        println!("  CPU matches V2: {}/{}", cpu_v2_match, n);
        println!("  CPU matches V3: {}/{}", cpu_v3_match, n);

        // Compare V3 and V2 forward NTT outputs for batch 0
        let mut total_mismatches = 0;
        let mut first_mismatch: Option<(usize, usize, u64, u64)> = None;

        for mod_idx in 0..k {
            for j in 0..n {
                let v2_val = v2_ntt[mod_idx][j];
                let v3_val = v3_ntt[0][mod_idx][j];
                if v2_val != v3_val {
                    total_mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((mod_idx, j, v2_val, v3_val));
                    }
                }
            }
        }

        if total_mismatches == 0 {
            println!("\n✓ V3 forward NTT matches V2 exactly!");
        } else {
            println!("✗ V3 vs V2 forward NTT mismatch: {} total differences", total_mismatches);
            if let Some((m, j, v2_val, v3_val)) = first_mismatch {
                println!("  First mismatch at mod={}, idx={}: V2={} vs V3={}", m, j, v2_val, v3_val);
            }
            // Print sample values
            println!("  Sample V2 NTT[0][0..5]: {:?}", &v2_ntt[0][0..5]);
            println!("  Sample V3 NTT[0][0][0..5]: {:?}", &v3_ntt[0][0][0..5]);

            // Try to find if V3 values exist in V2 but at different indices (permutation)
            let mut found_matches = 0;
            for j in 0..std::cmp::min(5, n) {
                let v3_val = v3_ntt[0][0][j];
                for k in 0..n {
                    if v2_ntt[0][k] == v3_val {
                        println!("  V3[0][0][{}] = {} found at V2[0][{}]", j, v3_val, k);
                        found_matches += 1;
                        break;
                    }
                }
            }
            if found_matches > 0 {
                println!("  Found {} index permutation matches (possible ordering issue)", found_matches);
            }
        }

        // Test expected permutation: V3[k1*n2+k2] should equal standard NTT at k1+k2*n1
        // Compute CPU standard NTT (no transpose) for verification
        println!("\n=== Testing Permutation Relationship ===");
        let mut cpu_std_ntt = vec![0u64; n];
        let omega_powers_0 = &params.rns_data[0].omega_powers;
        let psi_powers_0 = &params.rns_data[0].psi_powers;
        let q0 = params.rns_data[0].modulus;

        // Apply twist first
        let mut twisted_input = vec![0u64; n];
        for j in 0..n {
            twisted_input[j] = ((v2_encoded[0][j] as u128 * psi_powers_0[j] as u128) % q0 as u128) as u64;
        }

        // Standard NTT: X[k] = sum_j twisted[j] * omega^(j*k)
        for k in 0..n {
            let mut sum: u128 = 0;
            for j in 0..n {
                let omega_jk = omega_powers_0[(j * k) % n];
                sum = (sum + (twisted_input[j] as u128 * omega_jk as u128) % q0 as u128) % q0 as u128;
            }
            cpu_std_ntt[k] = sum as u64;
        }

        // Check permutation: V3[k1*n2+k2] == cpu_std_ntt[k1+k2*n1]
        let mut permute_matches_v1 = 0;  // V3[idx] == std[k1+k2*n1] where k1=idx/n2, k2=idx%n2
        let mut permute_matches_v2 = 0;  // V3[idx] == std[k2*n1+k1]
        let mut permute_matches_v3 = 0;  // V3[idx] == std[idx] (no permutation)

        for idx in 0..n {
            let k1 = idx / n2;
            let k2 = idx % n2;
            let std_idx_v1 = k1 + k2 * n1;  // Current formula
            let std_idx_v2 = k2 * n1 + k1;  // Alternative (same as v1 for this case)

            if v3_ntt[0][0][idx] == cpu_std_ntt[std_idx_v1] {
                permute_matches_v1 += 1;
            }
            if v3_ntt[0][0][idx] == cpu_std_ntt[idx] {
                permute_matches_v3 += 1;
            }
        }

        println!("  V3[idx] == std[k1+k2*n1] matches: {}/{}", permute_matches_v1, n);
        println!("  V3[idx] == std[idx] (no permutation) matches: {}/{}", permute_matches_v3, n);

        // Print sample comparisons
        println!("  Sample comparisons (idx, k1, k2, std_idx, V3_val, std_val):");
        for idx in [0, 1, 2, 256, 257, 512].iter() {
            if *idx < n {
                let k1 = idx / n2;
                let k2 = idx % n2;
                let std_idx = k1 + k2 * n1;
                println!("    idx={}: k1={}, k2={}, std_idx={}, V3={}, std={}",
                         idx, k1, k2, std_idx, v3_ntt[0][0][*idx], cpu_std_ntt[std_idx]);
            }
        }

        // Search for V3 values in CPU std NTT to find the permutation
        println!("\n  Searching for V3 values in CPU std NTT (first 10 V3 indices):");
        for v3_idx in 0..10 {
            let v3_val = v3_ntt[0][0][v3_idx];
            let mut found = false;
            for std_idx in 0..n {
                if cpu_std_ntt[std_idx] == v3_val {
                    let std_k1 = std_idx / n2;
                    let std_k2 = std_idx % n2;
                    println!("    V3[{}] = {} found at std[{}] (k1={}, k2={})",
                             v3_idx, v3_val, std_idx, std_k1, std_k2);
                    found = true;
                    break;
                }
            }
            if !found {
                println!("    V3[{}] = {} NOT FOUND in std NTT", v3_idx, v3_val);
            }
        }
    }
}
