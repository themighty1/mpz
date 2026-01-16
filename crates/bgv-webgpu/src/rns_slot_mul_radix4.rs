//! Radix-4 RNS Slot Multiplication for BGV using WebGPU.
//!
//! This version uses radix-4 NTT butterflies instead of radix-2.
//! For n=8192 (2^13), this reduces stages from 13 to 7 (6 radix-4 + 1 radix-2).
//! Fewer stages = fewer barriers = better GPU utilization.
//!
//! Pipeline: slot_encode -> forward_ntt -> fused_mul_intt
//! Same as V2, but with radix-4 butterflies.

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, BindGroup, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;
use crate::rns_slot_mul::RnsBatchParams;

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
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
    num_cts: u32,
    batches_per_ct: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform buffer for modulus data (per-modulus).
/// Extended for radix-4 with both forward and inverse omega^(n/4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    // omega^(n/4) for forward NTT radix-4 butterfly
    omega_quarter_lo: u32,
    omega_quarter_hi: u32,
    // omega_inv^(n/4) for inverse NTT radix-4 butterfly
    omega_quarter_inv_lo: u32,
    omega_quarter_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Maximum number of batches for pre-allocated buffers.
const MAX_SLOT_MUL_BATCHES: usize = 512;

/// GPU context for Radix-4 RNS batched slot multiplication.
///
/// Key optimization: Radix-4 butterflies reduce NTT stages from 13 to 7.
/// This means 7 barriers instead of 14, improving GPU occupancy.
pub struct RnsSlotMulGpuRadix4 {
    device: Device,
    queue: Queue,

    // Pipelines
    slot_encode_pipeline: ComputePipeline,
    forward_ntt_pipeline: ComputePipeline,
    fused_mul_intt_pipeline: ComputePipeline,
    standard_intt_pipeline: ComputePipeline,  // For polynomial interpolation (standard INTT)

    // Parameters
    params: RnsBatchParams,

    // Precomputed buffers
    plaintext_twiddles_buffer: Buffer,        // zeta_inv_powers for twisted INTT
    standard_intt_twiddles_buffer: Buffer,    // omega_inv_powers for standard INTT
    plaintext_params_buffer: Buffer,
    all_rns_twiddles_buffer: Buffer,
    all_rns_inv_twiddles_buffer: Buffer,
    all_rns_params_buffer: Buffer,

    // Pre-allocated buffers
    preallocated_encoded_buffer: Buffer,
    preallocated_pt_ntt_buffer: Buffer,
    preallocated_out_c0_buffer: Buffer,
    preallocated_out_c1_buffer: Buffer,
    preallocated_batch_params_buffer: Buffer,
    preallocated_fused_params_buffer: Buffer,
    preallocated_slots_buffer: Buffer,
    preallocated_cts_c0_buffer: Buffer,
    preallocated_cts_c1_buffer: Buffer,

    // Bind groups
    forward_ntt_bind_group: BindGroup,
    fused_mul_intt_bind_group: BindGroup,
}

impl std::fmt::Debug for RnsSlotMulGpuRadix4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RnsSlotMulGpuRadix4")
            .field("n", &self.params.n)
            .field("k", &self.params.k)
            .finish_non_exhaustive()
    }
}

impl RnsSlotMulGpuRadix4 {
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

/// Modular exponentiation: base^exp mod m
fn mod_pow(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut base = (base as u128) % (m as u128);
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        exp >>= 1;
        base = (base * base) % m;
    }
    result as u64
}

/// Modular inverse using extended Euclidean algorithm: a^(-1) mod m
fn mod_inv(a: u64, m: u64) -> u64 {
    // Use Fermat's little theorem: a^(-1) = a^(m-2) mod m (for prime m)
    mod_pow(a, m - 2, m)
}

impl RnsSlotMulGpuRadix4 {
    /// Creates a new Radix-4 GPU context for RNS batched slot multiplication.
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

        // Print adapter info for debugging
        #[cfg(target_arch = "wasm32")]
        {
            let info = adapter.get_info();
            let limits = adapter.limits();
            web_sys::console::log_1(&format!(
                "[GPU] Adapter: {} ({:?})",
                info.name, info.backend
            ).into());
            web_sys::console::log_1(&format!(
                "[GPU] DeviceType: {:?} (Cpu = software fallback)",
                info.device_type
            ).into());
            web_sys::console::log_1(&format!(
                "[GPU] maxComputeWorkgroupStorageSize: {} bytes ({} KB)",
                limits.max_compute_workgroup_storage_size,
                limits.max_compute_workgroup_storage_size / 1024
            ).into());
            web_sys::console::log_1(&format!(
                "[GPU] maxStorageBufferBindingSize: {} bytes ({} MB)",
                limits.max_storage_buffer_binding_size,
                limits.max_storage_buffer_binding_size / (1024 * 1024)
            ).into());
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rns-slot-mul-radix4 device"),
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
    pub fn new_with_device(
        device: Device,
        queue: Queue,
        params: RnsBatchParams,
    ) -> Result<Self, GpuError> {
        let n = params.n;
        let k = params.k;
        let log_n = (n as u32).trailing_zeros();

        // Compile shaders
        let slot_encode_shader = crate::shader_math::create_shader_module(
            &device,
            SLOT_ENCODE_RADIX4_SHADER,
            "slot_encode_radix4.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let forward_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            FORWARD_NTT_RADIX4_SHADER,
            "forward_ntt_radix4.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let fused_mul_intt_shader = crate::shader_math::create_shader_module(
            &device,
            FUSED_MUL_INTT_RADIX4_SHADER,
            "fused_mul_intt_radix4.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        // Create pipelines
        let slot_encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_encode_radix4 pipeline"),
                layout: None,
                module: &slot_encode_shader,
                entry_point: Some("slot_encode_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        let forward_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("forward_ntt_radix4 pipeline"),
                layout: None,
                module: &forward_ntt_shader,
                entry_point: Some("forward_ntt_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        let fused_mul_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fused_mul_intt_radix4 pipeline"),
                layout: None,
                module: &fused_mul_intt_shader,
                entry_point: Some("fused_mul_intt"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create standard INTT pipeline (same shader as slot_encode, just uses different twiddles)
        let standard_intt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("standard_intt_radix4 pipeline"),
                layout: None,
                module: &slot_encode_shader,
                entry_point: Some("slot_encode_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create plaintext parameter buffer with omega_quarter values
        // For plaintext: zeta is 2n-th root, so omega = zeta^2 is n-th root
        // omega_quarter for forward NTT = omega^(n/4) = zeta^(n/2)
        // omega_quarter_inv for INTT = omega_inv^(n/4) = zeta_inv^(n/2)
        // Note: slot_encode uses INTT, so it needs omega_quarter_inv
        let pt_data = &params.plaintext_data;
        // For INTT, we use zeta_inv_powers[n/2] = omega_inv^(n/4)
        let pt_omega_quarter_inv = pt_data.zeta_inv_powers[n / 2];
        // Compute omega^(n/4) = (zeta^2)^(n/4) = zeta^(n/2)
        // We can compute it as modular inverse of omega_quarter_inv
        let pt_omega_quarter = mod_inv(pt_omega_quarter_inv, pt_data.t);
        let plaintext_params = GpuModulusParams {
            modulus_lo: pt_data.t as u32,
            modulus_hi: (pt_data.t >> 32) as u32,
            mu_lo: pt_data.mu as u32,
            mu_hi: (pt_data.mu >> 32) as u32,
            n_inv_lo: pt_data.n_inv as u32,
            n_inv_hi: (pt_data.n_inv >> 32) as u32,
            omega_quarter_lo: pt_omega_quarter as u32,
            omega_quarter_hi: (pt_omega_quarter >> 32) as u32,
            omega_quarter_inv_lo: pt_omega_quarter_inv as u32,
            omega_quarter_inv_hi: (pt_omega_quarter_inv >> 32) as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let plaintext_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plaintext_params_radix4"),
            contents: bytemuck::bytes_of(&plaintext_params),
            usage: BufferUsages::UNIFORM,
        });

        // Create plaintext twiddle buffer (zeta_inv_powers for twisted INTT / slot encoding)
        let pt_twiddles_flat: Vec<u32> = pt_data
            .zeta_inv_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let plaintext_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_twiddles_radix4"),
                contents: bytemuck::cast_slice(&pt_twiddles_flat),
                usage: BufferUsages::STORAGE,
            });

        // Create standard INTT twiddle buffer (omega_inv_powers for polynomial interpolation)
        let standard_intt_twiddles_flat: Vec<u32> = pt_data
            .omega_inv_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let standard_intt_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("standard_intt_twiddles_radix4"),
                contents: bytemuck::cast_slice(&standard_intt_twiddles_flat),
                usage: BufferUsages::STORAGE,
            });

        // Create combined RNS modulus parameters with omega_quarter for each
        // omega_powers[k] = omega^k, so omega_powers[n/4] = omega^(n/4)
        // omega_inv_powers[k] = omega_inv^k, so omega_inv_powers[n/4] = omega_inv^(n/4)
        let mut all_rns_params: Vec<GpuModulusParams> = Vec::with_capacity(k);
        for rns_data in params.rns_data.iter() {
            let omega_quarter = rns_data.omega_powers[n / 4];
            let omega_quarter_inv = rns_data.omega_inv_powers[n / 4];
            all_rns_params.push(GpuModulusParams {
                modulus_lo: rns_data.modulus as u32,
                modulus_hi: (rns_data.modulus >> 32) as u32,
                mu_lo: rns_data.mu as u32,
                mu_hi: (rns_data.mu >> 32) as u32,
                n_inv_lo: rns_data.n_inv as u32,
                n_inv_hi: (rns_data.n_inv >> 32) as u32,
                omega_quarter_lo: omega_quarter as u32,
                omega_quarter_hi: (omega_quarter >> 32) as u32,
                omega_quarter_inv_lo: omega_quarter_inv as u32,
                omega_quarter_inv_hi: (omega_quarter_inv >> 32) as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }
        let all_rns_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_rns_params_radix4"),
            contents: bytemuck::cast_slice(&all_rns_params),
            usage: BufferUsages::STORAGE,
        });

        // Create combined forward twiddles buffer
        let mut all_fwd_twiddles: Vec<u32> = Vec::with_capacity(k * n * 4);
        for rns_data in params.rns_data.iter() {
            for &x in &rns_data.psi_powers {
                all_fwd_twiddles.push(x as u32);
                all_fwd_twiddles.push((x >> 32) as u32);
            }
            for &x in &rns_data.omega_powers {
                all_fwd_twiddles.push(x as u32);
                all_fwd_twiddles.push((x >> 32) as u32);
            }
        }
        let all_rns_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_rns_twiddles_radix4"),
            contents: bytemuck::cast_slice(&all_fwd_twiddles),
            usage: BufferUsages::STORAGE,
        });

        // Create combined inverse twiddles buffer
        let mut all_inv_twiddles: Vec<u32> = Vec::with_capacity(k * n * 4);
        for rns_data in params.rns_data.iter() {
            for &x in &rns_data.psi_inv_powers {
                all_inv_twiddles.push(x as u32);
                all_inv_twiddles.push((x >> 32) as u32);
            }
            for &x in &rns_data.omega_inv_powers {
                all_inv_twiddles.push(x as u32);
                all_inv_twiddles.push((x >> 32) as u32);
            }
        }
        let all_rns_inv_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_rns_inv_twiddles_radix4"),
            contents: bytemuck::cast_slice(&all_inv_twiddles),
            usage: BufferUsages::STORAGE,
        });

        // Create pre-allocated buffers
        let max_batches = MAX_SLOT_MUL_BATCHES;
        let slot_buffer_size = (max_batches * n * 2 * 4) as u64;
        let rns_buffer_size = (max_batches * k * n * 2 * 4) as u64;

        let preallocated_slots_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slots_input_radix4"),
            size: slot_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_encoded_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_radix4"),
            size: slot_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_pt_ntt_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_ntt_radix4"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        const MAX_CT_CHUNKS: usize = 8;
        let ct_buffer_size = (MAX_CT_CHUNKS * k * n * 2 * 4) as u64;

        let preallocated_cts_c0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cts_c0_input_radix4"),
            size: ct_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_cts_c1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cts_c1_input_radix4"),
            size: ct_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let preallocated_out_c0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c0_radix4"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let preallocated_out_c1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c1_radix4"),
            size: rns_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let initial_batch_params = GpuBatchParams {
            n: n as u32,
            log_n,
            num_batches: max_batches as u32,
            num_moduli: k as u32,
        };
        let preallocated_batch_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_params_radix4"),
                contents: bytemuck::bytes_of(&initial_batch_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let initial_fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            log_n,
            num_batches: max_batches as u32,
            num_moduli: k as u32,
            num_cts: 1,
            batches_per_ct: max_batches as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let preallocated_fused_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_params_radix4"),
                contents: bytemuck::bytes_of(&initial_fused_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        // Create bind groups
        let forward_ntt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward_ntt_bind_group_radix4"),
            layout: &forward_ntt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preallocated_pt_ntt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: all_rns_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        let fused_mul_intt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_mul_intt_bind_group_radix4"),
            layout: &fused_mul_intt_pipeline.get_bind_group_layout(0),
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
                    resource: all_rns_inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: all_rns_params_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            slot_encode_pipeline,
            forward_ntt_pipeline,
            fused_mul_intt_pipeline,
            standard_intt_pipeline,
            params,
            plaintext_twiddles_buffer,
            standard_intt_twiddles_buffer,
            plaintext_params_buffer,
            all_rns_twiddles_buffer,
            all_rns_inv_twiddles_buffer,
            all_rns_params_buffer,
            preallocated_encoded_buffer,
            preallocated_pt_ntt_buffer,
            preallocated_out_c0_buffer,
            preallocated_out_c1_buffer,
            preallocated_batch_params_buffer,
            preallocated_fused_params_buffer,
            preallocated_slots_buffer,
            preallocated_cts_c0_buffer,
            preallocated_cts_c1_buffer,
            forward_ntt_bind_group,
            fused_mul_intt_bind_group,
        })
    }

    /// Performs batched slot multiplication with multiple ciphertexts.
    #[cfg(target_arch = "wasm32")]
    pub async fn mul_batched_multi_ct(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
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

        // Update fused params
        let fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            num_cts: num_cts as u32,
            batches_per_ct: batches_per_ct.get(0).copied().unwrap_or(num_batches) as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fused_params_buffer,
            0,
            bytemuck::bytes_of(&fused_params),
        );

        // Create slot_encode bind group
        let slot_encode_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_bind_group_radix4"),
            layout: &self.slot_encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.plaintext_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute GPU pipeline
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("slot_encode_radix4_encoder"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode_radix4"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &slot_encode_bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
            drop(pass);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ntt_mul_radix4_encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward_ntt_radix4_all"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_ntt_pipeline);
                pass.set_bind_group(0, &self.forward_ntt_bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, k as u32, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fused_mul_intt_radix4_all"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.fused_mul_intt_pipeline);
                pass.set_bind_group(0, &self.fused_mul_intt_bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, k as u32, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back results
        self.read_rns_batch_async(num_batches).await
    }

    #[cfg(target_arch = "wasm32")]
    async fn read_rns_batch_async(
        &self,
        num_batches: usize,
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let result_size = (num_batches * k * n * 2 * 4) as u64;

        let staging_c0 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c0_radix4"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_c1 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c1_radix4"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_results_radix4"),
            });
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c0_buffer, 0, &staging_c0, 0, result_size);
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c1_buffer, 0, &staging_c1, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

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

        self.device.poll(wgpu::Maintain::Wait);

        rx0.await
            .map_err(|_| GpuError::MapFailed)?
            .map_err(|_| GpuError::MapFailed)?;
        rx1.await
            .map_err(|_| GpuError::MapFailed)?
            .map_err(|_| GpuError::MapFailed)?;

        let c0_data: Vec<u32> = bytemuck::cast_slice(&c0_slice.get_mapped_range()).to_vec();
        let c1_data: Vec<u32> = bytemuck::cast_slice(&c1_slice.get_mapped_range()).to_vec();

        let c0_u64: &[u64] = bytemuck::cast_slice(&c0_data);
        let c1_u64: &[u64] = bytemuck::cast_slice(&c1_data);

        let mut c0_results: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; n]; k]; num_batches];
        let mut c1_results: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; n]; k]; num_batches];

        for batch_idx in 0..num_batches {
            for mod_idx in 0..k {
                let base = (batch_idx * k + mod_idx) * n;
                c0_results[batch_idx][mod_idx].copy_from_slice(&c0_u64[base..base + n]);
                c1_results[batch_idx][mod_idx].copy_from_slice(&c1_u64[base..base + n]);
            }
        }

        Ok((c0_results, c1_results))
    }

    /// Native (non-WASM) version.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mul_batched_multi_ct(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        pollster::block_on(async {
            self.mul_batched_multi_ct_async(slots, cts_c0_ntt, cts_c1_ntt, batches_per_ct)
                .await
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn mul_batched_multi_ct_async(
        &self,
        slots: &[Vec<u64>],
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        batches_per_ct: &[usize],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
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

        // Update fused params
        let fused_params = GpuBatchParamsMultiCt {
            n: n as u32,
            log_n,
            num_batches: num_batches as u32,
            num_moduli: k as u32,
            num_cts: num_cts as u32,
            batches_per_ct: batches_per_ct.get(0).copied().unwrap_or(num_batches) as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue.write_buffer(
            &self.preallocated_fused_params_buffer,
            0,
            bytemuck::bytes_of(&fused_params),
        );

        // Create slot_encode bind group
        let slot_encode_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_bind_group_radix4_native"),
            layout: &self.slot_encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.plaintext_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute GPU pipeline
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("slot_encode_radix4_encoder_native"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode_radix4_native"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &slot_encode_bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
            drop(pass);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ntt_mul_radix4_encoder_native"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward_ntt_radix4_native_all"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_ntt_pipeline);
                pass.set_bind_group(0, &self.forward_ntt_bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, k as u32, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fused_mul_intt_radix4_native_all"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.fused_mul_intt_pipeline);
                pass.set_bind_group(0, &self.fused_mul_intt_bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, k as u32, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        self.read_rns_batch_async_native(num_batches).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn read_rns_batch_async_native(
        &self,
        num_batches: usize,
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let n = self.params.n;
        let k = self.params.k;
        let result_size = (num_batches * k * n * 2 * 4) as u64;

        let staging_c0 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c0_radix4_native"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_c1 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_c1_radix4_native"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_results_radix4_native"),
            });
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c0_buffer, 0, &staging_c0, 0, result_size);
        encoder.copy_buffer_to_buffer(&self.preallocated_out_c1_buffer, 0, &staging_c1, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let c0_slice = staging_c0.slice(..);
        let (tx0, rx0) = std::sync::mpsc::channel();
        c0_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx0.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx0.recv().map_err(|_| GpuError::MapFailed)?.map_err(|_| GpuError::MapFailed)?;

        let c1_slice = staging_c1.slice(..);
        let (tx1, rx1) = std::sync::mpsc::channel();
        c1_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx1.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx1.recv().map_err(|_| GpuError::MapFailed)?.map_err(|_| GpuError::MapFailed)?;

        let c0_data: Vec<u32> = bytemuck::cast_slice(&c0_slice.get_mapped_range()).to_vec();
        let c1_data: Vec<u32> = bytemuck::cast_slice(&c1_slice.get_mapped_range()).to_vec();

        let c0_u64: &[u64] = bytemuck::cast_slice(&c0_data);
        let c1_u64: &[u64] = bytemuck::cast_slice(&c1_data);

        let mut c0_results: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; n]; k]; num_batches];
        let mut c1_results: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; n]; k]; num_batches];

        for batch_idx in 0..num_batches {
            for mod_idx in 0..k {
                let base = (batch_idx * k + mod_idx) * n;
                c0_results[batch_idx][mod_idx].copy_from_slice(&c0_u64[base..base + n]);
                c1_results[batch_idx][mod_idx].copy_from_slice(&c1_u64[base..base + n]);
            }
        }

        Ok((c0_results, c1_results))
    }

    /// Test helper: Run only slot_encode and return the encoded coefficients.
    #[cfg(test)]
    pub fn test_slot_encode(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        let n = self.params.n;
        let num_batches = slots.len();

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload to GPU
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

        // Create slot_encode bind group
        let slot_encode_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_slot_encode_bind_group_radix4"),
            layout: &self.slot_encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.plaintext_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Run slot_encode
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_slot_encode_encoder_radix4"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test_slot_encode_radix4"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &slot_encode_bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back encoded buffer
        let result_size = (num_batches * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_slot_encode_staging_radix4"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_slot_encode_copy_radix4"),
            });
        encoder.copy_buffer_to_buffer(&self.preallocated_encoded_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|_| GpuError::MapFailed)?.map_err(|_| GpuError::MapFailed)?;

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        let data_u64: &[u64] = bytemuck::cast_slice(&data);

        // Reshape to [num_batches][n]
        let mut results: Vec<Vec<u64>> = vec![vec![0u64; n]; num_batches];
        for batch_idx in 0..num_batches {
            let base = batch_idx * n;
            results[batch_idx].copy_from_slice(&data_u64[base..base + n]);
        }

        Ok(results)
    }

    /// Batched slot encoding (INTT) for Goldilocks field - async version for WASM.
    ///
    /// This is the GPU-accelerated inverse NTT that converts slot values to polynomial
    /// coefficients. Operates on the plaintext modulus t (Goldilocks).
    ///
    /// # Arguments
    /// * `slots` - Vector of slot vectors, each of length n
    ///
    /// # Returns
    /// Vector of coefficient vectors, each of length n (polynomial coefficients mod t)
    pub async fn slot_encode_batched_async(&self, slots: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        let n = self.params.n;
        let num_batches = slots.len();

        if num_batches == 0 {
            return Ok(Vec::new());
        }

        // Validate input sizes
        for (i, slot_vec) in slots.iter().enumerate() {
            if slot_vec.len() != n {
                return Err(GpuError::InvalidParams(format!(
                    "Slot vector {} has {} elements, expected {}", i, slot_vec.len(), n
                )));
            }
        }

        // Flatten slots
        let mut slots_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, slot_vec) in slots.iter().enumerate() {
            let base = batch_idx * n;
            slots_flat[base..base + n].copy_from_slice(slot_vec);
        }

        // Upload to GPU
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

        // Create slot_encode bind group
        let slot_encode_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_encode_batched_bind_group_radix4"),
            layout: &self.slot_encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.plaintext_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Run slot_encode
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_batched_encoder_radix4"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode_batched_radix4"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &slot_encode_bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back encoded buffer asynchronously
        let result_size = (num_batches * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot_encode_batched_staging_radix4"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_encode_batched_copy_radix4"),
            });
        encoder.copy_buffer_to_buffer(&self.preallocated_encoded_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Async buffer mapping
        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::MapFailed)?
            .map_err(|_| GpuError::MapFailed)?;

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        let data_u64: &[u64] = bytemuck::cast_slice(&data);

        // Reshape to [num_batches][n]
        let mut results: Vec<Vec<u64>> = vec![vec![0u64; n]; num_batches];
        for batch_idx in 0..num_batches {
            let base = batch_idx * n;
            results[batch_idx].copy_from_slice(&data_u64[base..base + n]);
        }

        Ok(results)
    }

    /// Batched standard INTT for Goldilocks field - async version for WASM.
    ///
    /// This is the GPU-accelerated standard inverse NTT for polynomial interpolation.
    /// Unlike `slot_encode_batched_async` which uses twisted INTT (for slot encoding),
    /// this uses standard INTT (for interpolation at n-th roots of unity).
    ///
    /// # Arguments
    /// * `evals` - Vector of evaluation vectors, each of length n (values at ω^0, ω^1, ..., ω^{n-1})
    ///
    /// # Returns
    /// Vector of coefficient vectors, each of length n (polynomial coefficients mod t)
    pub async fn standard_intt_batched_async(&self, evals: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        let n = self.params.n;
        let num_batches = evals.len();

        if num_batches == 0 {
            return Ok(Vec::new());
        }

        // Validate input sizes
        for (i, eval_vec) in evals.iter().enumerate() {
            if eval_vec.len() != n {
                return Err(GpuError::InvalidParams(format!(
                    "Eval vector {} has {} elements, expected {}", i, eval_vec.len(), n
                )));
            }
        }

        // Flatten evals
        let mut evals_flat: Vec<u64> = vec![0u64; num_batches * n];
        for (batch_idx, eval_vec) in evals.iter().enumerate() {
            let base = batch_idx * n;
            evals_flat[base..base + n].copy_from_slice(eval_vec);
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.preallocated_slots_buffer,
            0,
            bytemuck::cast_slice(&evals_flat),
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

        // Create bind group using standard_intt_twiddles_buffer (omega_inv_powers)
        let standard_intt_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("standard_intt_batched_bind_group_radix4"),
            layout: &self.standard_intt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.preallocated_batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.preallocated_slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.preallocated_encoded_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.standard_intt_twiddles_buffer.as_entire_binding(), // omega_inv_powers
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.plaintext_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Run standard INTT
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("standard_intt_batched_encoder_radix4"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("standard_intt_batched_radix4"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.standard_intt_pipeline);
            pass.set_bind_group(0, &standard_intt_bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back encoded buffer asynchronously
        let result_size = (num_batches * n * 2 * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("standard_intt_batched_staging_radix4"),
            size: result_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("standard_intt_batched_copy_radix4"),
            });
        encoder.copy_buffer_to_buffer(&self.preallocated_encoded_buffer, 0, &staging, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Async buffer mapping
        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::MapFailed)?
            .map_err(|_| GpuError::MapFailed)?;

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        let data_u64: &[u64] = bytemuck::cast_slice(&data);

        // Reshape to [num_batches][n]
        let mut results: Vec<Vec<u64>> = vec![vec![0u64; n]; num_batches];
        for batch_idx in 0..num_batches {
            let base = batch_idx * n;
            results[batch_idx].copy_from_slice(&data_u64[base..base + n]);
        }

        Ok(results)
    }
}

// ============================================================================
// RADIX-4 SHADERS
// ============================================================================

/// Slot encode shader with radix-4 INTT.
/// For n=8192: 6 radix-4 stages + 1 radix-2 stage = 7 barriers (vs 13 in radix-2)
const SLOT_ENCODE_RADIX4_SHADER: &str = r#"
#import math

struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    omega_quarter_lo: u32,
    omega_quarter_hi: u32,
    omega_quarter_inv_lo: u32,
    omega_quarter_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> coeffs: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Radix-4 INTT butterfly: equivalent to two cascaded radix-2 DIT INTT stages
// This combines stages s (inner) and s+1 (outer) of a radix-2 INTT
fn radix4_intt_butterfly(
    a0: vec2<u32>, a1: vec2<u32>, a2: vec2<u32>, a3: vec2<u32>,
    w_inner: vec2<u32>,   // twiddle for inner radix-2 stage
    w_even: vec2<u32>,    // twiddle for outer stage, even position
    w_odd: vec2<u32>,     // twiddle for outer stage, odd position
    q: vec2<u32>
) -> array<vec2<u32>, 4> {
    // Inner radix-2 stage: butterflies on (a0, a1) and (a2, a3)
    let tw_a1 = math::mulmod_60bit(a1, w_inner, q);
    let tw_a3 = math::mulmod_60bit(a3, w_inner, q);

    let t0 = math::addmod(a0, tw_a1, q);
    let t1 = math::submod(a0, tw_a1, q);
    let t2 = math::addmod(a2, tw_a3, q);
    let t3 = math::submod(a2, tw_a3, q);

    // Outer radix-2 stage: butterflies on (t0, t2) and (t1, t3)
    let tw_t2 = math::mulmod_60bit(t2, w_even, q);
    let tw_t3 = math::mulmod_60bit(t3, w_odd, q);

    let b0 = math::addmod(t0, tw_t2, q);
    let b2 = math::submod(t0, tw_t2, q);
    let b1 = math::addmod(t1, tw_t3, q);
    let b3 = math::submod(t1, tw_t3, q);

    return array<vec2<u32>, 4>(b0, b1, b2, b3);
}

@compute @workgroup_size(256, 1, 1)
fn slot_encode_batched(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches { return; }

    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let j_inv = vec2<u32>(mod_params.omega_quarter_inv_lo, mod_params.omega_quarter_inv_hi);
    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load with bit-reversal (same as radix-2)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(slots[base], slots[base + 1u]);
        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // Radix-4 INTT stages (6 stages for n=8192, processing 2 bits per stage)
    let num_radix4_stages = log_n / 2u;

    for (var stage = 0u; stage < num_radix4_stages; stage++) {
        let m = 1u << ((stage + 1u) * 2u);  // 4, 16, 64, 256, 1024, 4096
        let quarter_m = m / 4u;              // 1, 4, 16, 64, 256, 1024

        // Each thread handles multiple radix-4 butterflies
        let butterflies_per_stage = n / 4u;  // 2048 butterflies
        let butterflies_per_thread = butterflies_per_stage / 256u;  // 8 per thread

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / quarter_m;
            let idx_in_group = butterfly_idx % quarter_m;

            let i0 = group * m + idx_in_group;
            let i1 = i0 + quarter_m;
            let i2 = i0 + 2u * quarter_m;
            let i3 = i0 + 3u * quarter_m;

            // Load 4 values
            let a0 = vec2<u32>(shared_lo[i0], shared_hi[i0]);
            let a1 = vec2<u32>(shared_lo[i1], shared_hi[i1]);
            let a2 = vec2<u32>(shared_lo[i2], shared_hi[i2]);
            let a3 = vec2<u32>(shared_lo[i3], shared_hi[i3]);

            // Compute twiddle indices for cascaded radix-2 INTT
            // Inner stage (combines pairs): w_inner at idx 2k * n / m
            // Outer stage even position: w_even at idx k * n / m
            // Outer stage odd position: w_odd at idx (k + quarter_m) * n / m = tw_even + n/4
            let k = idx_in_group;
            let tw_even_idx = k * (n / m);
            let tw_inner_idx = (2u * tw_even_idx) % n;
            let tw_odd_idx = (tw_even_idx + n / 4u) % n;

            let tw_even_base = tw_even_idx * 2u;
            let tw_inner_base = tw_inner_idx * 2u;
            let tw_odd_base = tw_odd_idx * 2u;

            let w_even = vec2<u32>(twiddles[tw_even_base], twiddles[tw_even_base + 1u]);
            let w_inner = vec2<u32>(twiddles[tw_inner_base], twiddles[tw_inner_base + 1u]);
            let w_odd = vec2<u32>(twiddles[tw_odd_base], twiddles[tw_odd_base + 1u]);

            // Apply radix-4 butterfly (cascaded radix-2)
            let result = radix4_intt_butterfly(a0, a1, a2, a3, w_inner, w_even, w_odd, q);

            // Store results
            shared_lo[i0] = result[0].x;
            shared_hi[i0] = result[0].y;
            shared_lo[i1] = result[1].x;
            shared_hi[i1] = result[1].y;
            shared_lo[i2] = result[2].x;
            shared_hi[i2] = result[2].y;
            shared_lo[i3] = result[3].x;
            shared_hi[i3] = result[3].y;
        }
        workgroupBarrier();
    }

    // Final radix-2 stage if log_n is odd (13 is odd, so we need this)
    if (log_n % 2u) == 1u {
        let m = n;
        let half_m = n / 2u;
        let butterflies_per_thread = half_m / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let ii = butterfly_idx;
            let jj = butterfly_idx + half_m;

            let twiddle_idx = butterfly_idx;
            let tw_base = twiddle_idx * 2u;
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

    // Scale by n^-1 and store
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        val = math::mulmod_60bit(val, n_inv, q);
        let out_base = (batch_offset + idx) * 2u;
        coeffs[out_base] = val.x;
        coeffs[out_base + 1u] = val.y;
    }
}
"#;

/// Forward NTT shader with radix-4 butterflies.
const FORWARD_NTT_RADIX4_SHADER: &str = r#"
#import math

struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo: u32,
    mu_hi: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
    omega_quarter_lo: u32,
    omega_quarter_hi: u32,
    omega_quarter_inv_lo: u32,
    omega_quarter_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> coeffs: array<u32>;
@group(0) @binding(2) var<storage, read_write> ntt_out: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;
@group(0) @binding(4) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Radix-4 NTT butterfly: equivalent to two cascaded radix-2 DIT NTT stages
// This is the same structure as INTT, just uses omega_powers instead of omega_inv_powers
fn radix4_ntt_butterfly(
    a0: vec2<u32>, a1: vec2<u32>, a2: vec2<u32>, a3: vec2<u32>,
    w_inner: vec2<u32>,   // twiddle for inner radix-2 stage
    w_even: vec2<u32>,    // twiddle for outer stage, even position
    w_odd: vec2<u32>,     // twiddle for outer stage, odd position
    q: vec2<u32>
) -> array<vec2<u32>, 4> {
    // Inner radix-2 stage: butterflies on (a0, a1) and (a2, a3)
    let tw_a1 = math::mulmod_60bit(a1, w_inner, q);
    let tw_a3 = math::mulmod_60bit(a3, w_inner, q);

    let t0 = math::addmod(a0, tw_a1, q);
    let t1 = math::submod(a0, tw_a1, q);
    let t2 = math::addmod(a2, tw_a3, q);
    let t3 = math::submod(a2, tw_a3, q);

    // Outer radix-2 stage: butterflies on (t0, t2) and (t1, t3)
    let tw_t2 = math::mulmod_60bit(t2, w_even, q);
    let tw_t3 = math::mulmod_60bit(t3, w_odd, q);

    let b0 = math::addmod(t0, tw_t2, q);
    let b2 = math::submod(t0, tw_t2, q);
    let b1 = math::addmod(t1, tw_t3, q);
    let b3 = math::submod(t1, tw_t3, q);

    return array<vec2<u32>, 4>(b0, b1, b2, b3);
}

@compute @workgroup_size(256, 1, 1)
fn forward_ntt_batched(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let mod_idx = wg_id.y;
    let n = params.n;
    let log_n = params.log_n;
    let k = params.num_moduli;

    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);
    let j = vec2<u32>(mp.omega_quarter_lo, mp.omega_quarter_hi);

    let twiddle_base_offset = mod_idx * n * 4u;
    let batch_offset = batch_idx * n;
    let out_batch_offset = (batch_idx * k + mod_idx) * n;
    let elements_per_thread = n / 256u;

    // Load, twist, and bit-reverse
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let coeff_base = (batch_offset + idx) * 2u;
        var val = vec2<u32>(coeffs[coeff_base], coeffs[coeff_base + 1u]);

        // Twist by psi^idx
        let psi_base = twiddle_base_offset + idx * 2u;
        let psi_power = vec2<u32>(twiddles[psi_base], twiddles[psi_base + 1u]);
        val = math::mulmod_60bit(val, psi_power, q);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // Radix-4 NTT stages
    let num_radix4_stages = log_n / 2u;

    for (var stage = 0u; stage < num_radix4_stages; stage++) {
        let m = 1u << ((stage + 1u) * 2u);
        let quarter_m = m / 4u;

        let butterflies_per_stage = n / 4u;
        let butterflies_per_thread = butterflies_per_stage / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / quarter_m;
            let idx_in_group = butterfly_idx % quarter_m;

            let i0 = group * m + idx_in_group;
            let i1 = i0 + quarter_m;
            let i2 = i0 + 2u * quarter_m;
            let i3 = i0 + 3u * quarter_m;

            let a0 = vec2<u32>(shared_lo[i0], shared_hi[i0]);
            let a1 = vec2<u32>(shared_lo[i1], shared_hi[i1]);
            let a2 = vec2<u32>(shared_lo[i2], shared_hi[i2]);
            let a3 = vec2<u32>(shared_lo[i3], shared_hi[i3]);

            // Compute twiddle indices for cascaded radix-2 NTT
            // Inner stage: w_inner at idx 2k * n / m
            // Outer stage even position: w_even at idx k * n / m
            // Outer stage odd position: w_odd at idx (k + quarter_m) * n / m = tw_even + n/4
            let k = idx_in_group;
            let tw_even_idx = k * (n / m);
            let tw_inner_idx = (2u * tw_even_idx) % n;
            let tw_odd_idx = (tw_even_idx + n / 4u) % n;
            let omega_offset = twiddle_base_offset + n * 2u;

            let tw_even_base = omega_offset + tw_even_idx * 2u;
            let tw_inner_base = omega_offset + tw_inner_idx * 2u;
            let tw_odd_base = omega_offset + tw_odd_idx * 2u;

            let w_even = vec2<u32>(twiddles[tw_even_base], twiddles[tw_even_base + 1u]);
            let w_inner = vec2<u32>(twiddles[tw_inner_base], twiddles[tw_inner_base + 1u]);
            let w_odd = vec2<u32>(twiddles[tw_odd_base], twiddles[tw_odd_base + 1u]);

            let result = radix4_ntt_butterfly(a0, a1, a2, a3, w_inner, w_even, w_odd, q);

            shared_lo[i0] = result[0].x;
            shared_hi[i0] = result[0].y;
            shared_lo[i1] = result[1].x;
            shared_hi[i1] = result[1].y;
            shared_lo[i2] = result[2].x;
            shared_hi[i2] = result[2].y;
            shared_lo[i3] = result[3].x;
            shared_hi[i3] = result[3].y;
        }
        workgroupBarrier();
    }

    // Final radix-2 stage if needed
    if (log_n % 2u) == 1u {
        let half_m = n / 2u;
        let butterflies_per_thread = half_m / 256u;
        let omega_offset = twiddle_base_offset + n * 2u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let ii = butterfly_idx;
            let jj = butterfly_idx + half_m;

            let tw_base = omega_offset + butterfly_idx * 2u;
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

    // Store results
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        let out_base = (out_batch_offset + idx) * 2u;
        ntt_out[out_base] = val.x;
        ntt_out[out_base + 1u] = val.y;
    }
}
"#;

/// Fused multiply + INTT shader with radix-4 butterflies.
const FUSED_MUL_INTT_RADIX4_SHADER: &str = r#"
#import math

struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
    num_cts: u32,
    batches_per_ct: u32,
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
    omega_quarter_lo: u32,
    omega_quarter_hi: u32,
    omega_quarter_inv_lo: u32,
    omega_quarter_inv_hi: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> pt_ntt: array<u32>;
@group(0) @binding(2) var<storage, read> ct_c0: array<u32>;
@group(0) @binding(3) var<storage, read> ct_c1: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_c0: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_c1: array<u32>;
@group(0) @binding(6) var<storage, read> inv_twiddles: array<u32>;
@group(0) @binding(7) var<storage, read> mod_params: array<ModulusParams>;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Radix-4 INTT butterfly: equivalent to two cascaded radix-2 DIT INTT stages
fn radix4_intt_butterfly(
    a0: vec2<u32>, a1: vec2<u32>, a2: vec2<u32>, a3: vec2<u32>,
    w_inner: vec2<u32>,   // twiddle for inner radix-2 stage
    w_even: vec2<u32>,    // twiddle for outer stage, even position
    w_odd: vec2<u32>,     // twiddle for outer stage, odd position
    q: vec2<u32>
) -> array<vec2<u32>, 4> {
    // Inner radix-2 stage: butterflies on (a0, a1) and (a2, a3)
    let tw_a1 = math::mulmod_60bit(a1, w_inner, q);
    let tw_a3 = math::mulmod_60bit(a3, w_inner, q);

    let t0 = math::addmod(a0, tw_a1, q);
    let t1 = math::submod(a0, tw_a1, q);
    let t2 = math::addmod(a2, tw_a3, q);
    let t3 = math::submod(a2, tw_a3, q);

    // Outer radix-2 stage: butterflies on (t0, t2) and (t1, t3)
    let tw_t2 = math::mulmod_60bit(t2, w_even, q);
    let tw_t3 = math::mulmod_60bit(t3, w_odd, q);

    let b0 = math::addmod(t0, tw_t2, q);
    let b2 = math::submod(t0, tw_t2, q);
    let b1 = math::addmod(t1, tw_t3, q);
    let b3 = math::submod(t1, tw_t3, q);

    return array<vec2<u32>, 4>(b0, b1, b2, b3);
}

@compute @workgroup_size(256, 1, 1)
fn fused_mul_intt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let mod_idx = wg_id.y;
    let n = params.n;
    let log_n = params.log_n;
    let k = params.num_moduli;

    if batch_idx >= params.num_batches { return; }
    if mod_idx >= k { return; }

    let mp = mod_params[mod_idx];
    let q = vec2<u32>(mp.modulus_lo, mp.modulus_hi);
    let n_inv = vec2<u32>(mp.n_inv_lo, mp.n_inv_hi);
    let j_inv = vec2<u32>(mp.omega_quarter_inv_lo, mp.omega_quarter_inv_hi);

    let twiddle_base_offset = mod_idx * n * 4u;
    let elements_per_thread = n / 256u;

    let ct_idx = batch_idx / params.batches_per_ct;
    let ct_base_offset = (ct_idx * k + mod_idx) * n;
    let out_base_offset = (batch_idx * k + mod_idx) * n;
    let pt_ntt_offset = (batch_idx * k + mod_idx) * n;

    // =========== Process C0 ===========
    // Load, multiply, bit-reverse
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;

        let pt_base = (pt_ntt_offset + idx) * 2u;
        let pt = vec2<u32>(pt_ntt[pt_base], pt_ntt[pt_base + 1u]);

        let ct_base = (ct_base_offset + idx) * 2u;
        let c0 = vec2<u32>(ct_c0[ct_base], ct_c0[ct_base + 1u]);

        let prod = math::mulmod_60bit(c0, pt, q);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = prod.x;
        shared_hi[rev_idx] = prod.y;
    }
    workgroupBarrier();

    // Radix-4 INTT stages
    let num_radix4_stages = log_n / 2u;

    for (var stage = 0u; stage < num_radix4_stages; stage++) {
        let m = 1u << ((stage + 1u) * 2u);
        let quarter_m = m / 4u;

        let butterflies_per_stage = n / 4u;
        let butterflies_per_thread = butterflies_per_stage / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / quarter_m;
            let idx_in_group = butterfly_idx % quarter_m;

            let i0 = group * m + idx_in_group;
            let i1 = i0 + quarter_m;
            let i2 = i0 + 2u * quarter_m;
            let i3 = i0 + 3u * quarter_m;

            let a0 = vec2<u32>(shared_lo[i0], shared_hi[i0]);
            let a1 = vec2<u32>(shared_lo[i1], shared_hi[i1]);
            let a2 = vec2<u32>(shared_lo[i2], shared_hi[i2]);
            let a3 = vec2<u32>(shared_lo[i3], shared_hi[i3]);

            // Compute twiddle indices for cascaded radix-2 INTT
            let k = idx_in_group;
            let tw_even_idx = k * (n / m);
            let tw_inner_idx = (2u * tw_even_idx) % n;
            let tw_odd_idx = (tw_even_idx + n / 4u) % n;
            let omega_inv_offset = twiddle_base_offset + n * 2u;

            let tw_even_base = omega_inv_offset + tw_even_idx * 2u;
            let tw_inner_base = omega_inv_offset + tw_inner_idx * 2u;
            let tw_odd_base = omega_inv_offset + tw_odd_idx * 2u;

            let w_even = vec2<u32>(inv_twiddles[tw_even_base], inv_twiddles[tw_even_base + 1u]);
            let w_inner = vec2<u32>(inv_twiddles[tw_inner_base], inv_twiddles[tw_inner_base + 1u]);
            let w_odd = vec2<u32>(inv_twiddles[tw_odd_base], inv_twiddles[tw_odd_base + 1u]);

            let result = radix4_intt_butterfly(a0, a1, a2, a3, w_inner, w_even, w_odd, q);

            shared_lo[i0] = result[0].x;
            shared_hi[i0] = result[0].y;
            shared_lo[i1] = result[1].x;
            shared_hi[i1] = result[1].y;
            shared_lo[i2] = result[2].x;
            shared_hi[i2] = result[2].y;
            shared_lo[i3] = result[3].x;
            shared_hi[i3] = result[3].y;
        }
        workgroupBarrier();
    }

    // Final radix-2 stage if needed
    if (log_n % 2u) == 1u {
        let half_m = n / 2u;
        let butterflies_per_thread = half_m / 256u;
        let omega_inv_offset = twiddle_base_offset + n * 2u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let ii = butterfly_idx;
            let jj = butterfly_idx + half_m;

            let tw_base = omega_inv_offset + butterfly_idx * 2u;
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

    // Scale by n^-1, untwist, and store c0
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        val = math::mulmod_60bit(val, n_inv, q);
        let psi_inv_base = twiddle_base_offset + idx * 2u;
        let psi_inv = vec2<u32>(inv_twiddles[psi_inv_base], inv_twiddles[psi_inv_base + 1u]);
        val = math::mulmod_60bit(val, psi_inv, q);

        let out_base = (out_base_offset + idx) * 2u;
        out_c0[out_base] = val.x;
        out_c0[out_base + 1u] = val.y;
    }
    workgroupBarrier();

    // =========== Process C1 ===========
    // Load, multiply, bit-reverse
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;

        let pt_base = (pt_ntt_offset + idx) * 2u;
        let pt = vec2<u32>(pt_ntt[pt_base], pt_ntt[pt_base + 1u]);

        let ct_base = (ct_base_offset + idx) * 2u;
        let c1 = vec2<u32>(ct_c1[ct_base], ct_c1[ct_base + 1u]);

        let prod = math::mulmod_60bit(c1, pt, q);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = prod.x;
        shared_hi[rev_idx] = prod.y;
    }
    workgroupBarrier();

    // Radix-4 INTT stages for c1
    for (var stage = 0u; stage < num_radix4_stages; stage++) {
        let m = 1u << ((stage + 1u) * 2u);
        let quarter_m = m / 4u;

        let butterflies_per_stage = n / 4u;
        let butterflies_per_thread = butterflies_per_stage / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / quarter_m;
            let idx_in_group = butterfly_idx % quarter_m;

            let i0 = group * m + idx_in_group;
            let i1 = i0 + quarter_m;
            let i2 = i0 + 2u * quarter_m;
            let i3 = i0 + 3u * quarter_m;

            let a0 = vec2<u32>(shared_lo[i0], shared_hi[i0]);
            let a1 = vec2<u32>(shared_lo[i1], shared_hi[i1]);
            let a2 = vec2<u32>(shared_lo[i2], shared_hi[i2]);
            let a3 = vec2<u32>(shared_lo[i3], shared_hi[i3]);

            // Compute twiddle indices for cascaded radix-2 INTT
            let k = idx_in_group;
            let tw_even_idx = k * (n / m);
            let tw_inner_idx = (2u * tw_even_idx) % n;
            let tw_odd_idx = (tw_even_idx + n / 4u) % n;
            let omega_inv_offset = twiddle_base_offset + n * 2u;

            let tw_even_base = omega_inv_offset + tw_even_idx * 2u;
            let tw_inner_base = omega_inv_offset + tw_inner_idx * 2u;
            let tw_odd_base = omega_inv_offset + tw_odd_idx * 2u;

            let w_even = vec2<u32>(inv_twiddles[tw_even_base], inv_twiddles[tw_even_base + 1u]);
            let w_inner = vec2<u32>(inv_twiddles[tw_inner_base], inv_twiddles[tw_inner_base + 1u]);
            let w_odd = vec2<u32>(inv_twiddles[tw_odd_base], inv_twiddles[tw_odd_base + 1u]);

            let result = radix4_intt_butterfly(a0, a1, a2, a3, w_inner, w_even, w_odd, q);

            shared_lo[i0] = result[0].x;
            shared_hi[i0] = result[0].y;
            shared_lo[i1] = result[1].x;
            shared_hi[i1] = result[1].y;
            shared_lo[i2] = result[2].x;
            shared_hi[i2] = result[2].y;
            shared_lo[i3] = result[3].x;
            shared_hi[i3] = result[3].y;
        }
        workgroupBarrier();
    }

    // Final radix-2 stage for c1 if needed
    if (log_n % 2u) == 1u {
        let half_m = n / 2u;
        let butterflies_per_thread = half_m / 256u;
        let omega_inv_offset = twiddle_base_offset + n * 2u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let ii = butterfly_idx;
            let jj = butterfly_idx + half_m;

            let tw_base = omega_inv_offset + butterfly_idx * 2u;
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

    // Scale by n^-1, untwist, and store c1
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        val = math::mulmod_60bit(val, n_inv, q);
        let psi_inv_base = twiddle_base_offset + idx * 2u;
        let psi_inv = vec2<u32>(inv_twiddles[psi_inv_base], inv_twiddles[psi_inv_base + 1u]);
        val = math::mulmod_60bit(val, psi_inv, q);

        let out_base = (out_base_offset + idx) * 2u;
        out_c1[out_base] = val.x;
        out_c1[out_base + 1u] = val.y;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rns_slot_mul::RnsBatchParams;
    use crate::rns_slot_mul_v2::RnsSlotMulGpuV2;

    #[test]
    fn test_radix4_gpu_context_creation() {
        let params = RnsBatchParams::goldilocks(8192);
        assert!(params.is_some(), "Failed to create RnsBatchParams");

        let params = params.unwrap();
        let result = RnsSlotMulGpuRadix4::new(params);

        match result {
            Ok(gpu) => {
                assert_eq!(gpu.params().n, 8192);
                assert_eq!(gpu.params().k, 5);
                println!("Radix-4 GPU context created successfully!");
            }
            Err(e) => {
                println!("Radix-4 GPU context creation failed (expected in CI): {:?}", e);
            }
        }
    }

    #[test]
    fn test_omega_quarter_computation() {
        // For Goldilocks prime, verify omega_quarter values are correct
        let params = RnsBatchParams::goldilocks(8192).unwrap();
        let n = params.n;

        for rns_data in &params.rns_data {
            let omega = rns_data.omega_powers[1];
            let q = rns_data.modulus;
            // omega_quarter = omega^(n/4) = omega_powers[n/4]
            let omega_quarter = rns_data.omega_powers[n / 4];
            let omega_quarter_inv = rns_data.omega_inv_powers[n / 4];

            // omega^(n/4) should satisfy: (omega^(n/4))^4 = omega^n = 1
            let omega_quarter_4 = mod_pow(omega_quarter, 4, q);
            let omega_n = mod_pow(omega, n as u64, q);

            assert_eq!(omega_n, 1, "omega^n should be 1");
            // omega^(n/4)^4 = omega^n = 1
            assert_eq!(omega_quarter_4, omega_n, "omega_quarter^4 should equal omega^n");

            // Verify omega_quarter * omega_quarter_inv = 1 (mod q)
            let product = ((omega_quarter as u128 * omega_quarter_inv as u128) % q as u128) as u64;
            assert_eq!(product, 1, "omega_quarter * omega_quarter_inv should be 1");

            println!("Modulus {}: omega_quarter = {}, omega_quarter_inv = {}", q, omega_quarter, omega_quarter_inv);
        }
    }

    /// Test that radix-4 produces the same output as V2 for the same inputs.
    #[test]
    fn test_radix4_vs_v2_correctness() {
        let params = RnsBatchParams::goldilocks(8192).unwrap();
        let n = params.n;
        let k = params.k;

        // Create V2 and Radix-4 contexts
        let v2_ctx = match RnsSlotMulGpuV2::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V2 GPU not available ({})", e);
                return;
            }
        };

        let radix4_ctx = match RnsSlotMulGpuRadix4::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: Radix-4 GPU context creation failed ({})", e);
                return;
            }
        };

        // Create test data
        let num_batches = 4;
        let num_cts = 1;

        // Slots with deterministic values
        let slots: Vec<Vec<u64>> = (0..num_batches)
            .map(|batch_idx| {
                let t = params.plaintext_data.t;
                (0..n)
                    .map(|j| ((batch_idx * 1000 + j) as u64) % t)
                    .collect()
            })
            .collect();

        // Ciphertexts with small deterministic values
        let cts_c0: Vec<Vec<Vec<u64>>> = (0..num_cts)
            .map(|ct_idx| {
                (0..k)
                    .map(|mod_idx| {
                        let q = params.rns_data[mod_idx].modulus;
                        (0..n)
                            .map(|j| ((ct_idx * 1000 + mod_idx * 100 + j) as u64) % q)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let cts_c1: Vec<Vec<Vec<u64>>> = (0..num_cts)
            .map(|ct_idx| {
                (0..k)
                    .map(|mod_idx| {
                        let q = params.rns_data[mod_idx].modulus;
                        (0..n)
                            .map(|j| ((ct_idx * 2000 + mod_idx * 200 + j + 500) as u64) % q)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let batches_per_ct = vec![num_batches];

        // Run V2
        let (v2_c0, v2_c1) = v2_ctx
            .mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct)
            .expect("V2 mul_batched_multi_ct failed");

        // Run Radix-4
        let (radix4_c0, radix4_c1) = radix4_ctx
            .mul_batched_multi_ct(&slots, &cts_c0, &cts_c1, &batches_per_ct)
            .expect("Radix-4 mul_batched_multi_ct failed");

        // Compare outputs
        let mut mismatches = 0;
        for batch_idx in 0..num_batches {
            for mod_idx in 0..k {
                for i in 0..n {
                    if v2_c0[batch_idx][mod_idx][i] != radix4_c0[batch_idx][mod_idx][i] {
                        if mismatches < 10 {
                            println!(
                                "C0 mismatch at batch={}, mod={}, i={}: V2={}, Radix4={}",
                                batch_idx, mod_idx, i,
                                v2_c0[batch_idx][mod_idx][i],
                                radix4_c0[batch_idx][mod_idx][i]
                            );
                        }
                        mismatches += 1;
                    }
                    if v2_c1[batch_idx][mod_idx][i] != radix4_c1[batch_idx][mod_idx][i] {
                        if mismatches < 10 {
                            println!(
                                "C1 mismatch at batch={}, mod={}, i={}: V2={}, Radix4={}",
                                batch_idx, mod_idx, i,
                                v2_c1[batch_idx][mod_idx][i],
                                radix4_c1[batch_idx][mod_idx][i]
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
        }

        if mismatches > 0 {
            panic!("Radix-4 vs V2: {} mismatches out of {} elements",
                   mismatches, num_batches * k * n * 2);
        }

        println!("Radix-4 vs V2: All {} elements match!", num_batches * k * n * 2);
    }

    /// Test slot_encode output comparison between radix-4 and V2.
    #[test]
    fn test_radix4_slot_encode_vs_v2() {
        let params = RnsBatchParams::goldilocks(8192).unwrap();
        let n = params.n;

        // Create V2 and Radix-4 contexts
        let v2_ctx = match RnsSlotMulGpuV2::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: V2 GPU not available ({})", e);
                return;
            }
        };

        let radix4_ctx = match RnsSlotMulGpuRadix4::new(params.clone()) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: Radix-4 GPU context creation failed ({})", e);
                return;
            }
        };

        // Create test data - single batch of slots
        let t = params.plaintext_data.t;
        let slots: Vec<Vec<u64>> = vec![
            (0..n).map(|j| (j as u64) % t).collect()
        ];

        // Run slot_encode on both
        let v2_encoded = v2_ctx.test_slot_encode(&slots).expect("V2 slot_encode failed");
        let radix4_encoded = radix4_ctx.test_slot_encode(&slots).expect("Radix4 slot_encode failed");

        // Compare
        let mut mismatches = 0;
        for i in 0..n {
            if v2_encoded[0][i] != radix4_encoded[0][i] {
                if mismatches < 10 {
                    println!(
                        "slot_encode mismatch at i={}: V2={}, Radix4={}",
                        i, v2_encoded[0][i], radix4_encoded[0][i]
                    );
                }
                mismatches += 1;
            }
        }

        if mismatches > 0 {
            panic!("slot_encode: {} mismatches out of {} elements", mismatches, n);
        }
        println!("slot_encode: All {} elements match!", n);
    }

    // Note: V1 (RnsSlotMulGpu) has a different API that expects ciphertexts already in NTT domain,
    // while V2 and radix-4 take raw polynomial coefficients and do NTT internally.
    // Therefore, a direct comparison between V1 and radix-4 is not valid without first
    // transforming inputs to NTT form. Radix-4 matches V2, which is the current reference.
}
