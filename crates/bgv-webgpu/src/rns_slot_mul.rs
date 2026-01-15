//! GPU-accelerated batched RNS slot multiplication.
//!
//! This module provides GPU acceleration for the core IT-PAC operation:
//! multiplying a single ciphertext by many plaintext slot vectors across
//! all RNS moduli.
//!
//! # Performance Target
//!
//! CPU baseline (B+C=180 rows, k=4 moduli, n=8192):
//! - Total: ~3-4 seconds
//!
//! GPU target: < 200ms for the full batch

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;
use crate::math::{compute_powers, mod_mul, mod_inverse, find_psi, find_primitive_root};

#[cfg(target_arch = "wasm32")]
use futures::TryFutureExt;

/// Precomputed NTT data for a single RNS modulus.
#[derive(Clone)]
pub struct RnsModulusNttData {
    /// The modulus q_i.
    pub modulus: u64,
    /// Primitive 2n-th root of unity (psi).
    pub psi: u64,
    /// n-th root of unity (omega = psi^2).
    pub omega: u64,
    /// Inverse of psi.
    pub psi_inv: u64,
    /// Inverse of omega.
    pub omega_inv: u64,
    /// 1/n mod q.
    pub n_inv: u64,
    /// Barrett mu = floor(2^128 / q).
    pub mu: u128,
    /// Forward twiddle factors (powers of psi for twist).
    pub psi_powers: Vec<u64>,
    /// Inverse twiddle factors.
    pub psi_inv_powers: Vec<u64>,
    /// Forward NTT twiddles (powers of omega).
    pub omega_powers: Vec<u64>,
    /// Inverse NTT twiddles.
    pub omega_inv_powers: Vec<u64>,
}

impl RnsModulusNttData {
    /// Creates NTT data for a single modulus.
    pub fn new(n: usize, modulus: u64, psi: u64) -> Self {
        let omega = mod_mul(psi, psi, modulus);
        let psi_inv = mod_inverse(psi, modulus);
        let omega_inv = mod_inverse(omega, modulus);
        let n_inv = mod_inverse(n as u64, modulus);
        let mu = (1u128 << 127) / (modulus as u128) * 2;

        let psi_powers = compute_powers(psi, n, modulus);
        let psi_inv_powers = compute_powers(psi_inv, n, modulus);
        let omega_powers = compute_powers(omega, n, modulus);
        let omega_inv_powers = compute_powers(omega_inv, n, modulus);

        Self {
            modulus,
            psi,
            omega,
            psi_inv,
            omega_inv,
            n_inv,
            mu,
            psi_powers,
            psi_inv_powers,
            omega_powers,
            omega_inv_powers,
        }
    }
}

/// Precomputed data for plaintext modulus (Goldilocks).
#[derive(Clone)]
pub struct PlaintextNttData {
    /// Plaintext modulus t.
    pub t: u64,
    /// Primitive 2n-th root of unity for slot encoding.
    pub zeta: u64,
    /// Inverse of zeta.
    pub zeta_inv: u64,
    /// 1/n mod t.
    pub n_inv: u64,
    /// Barrett mu for t.
    pub mu: u128,
    /// Inverse twiddle factors for INTT (slot encoding).
    pub zeta_inv_powers: Vec<u64>,
}

impl PlaintextNttData {
    /// Creates plaintext NTT data for slot encoding.
    pub fn new(n: usize, t: u64) -> Option<Self> {
        // Check if slot packing is supported: t ≡ 1 (mod 2n)
        let order = 2 * n as u64;
        if (t - 1) % order != 0 {
            return None;
        }

        let zeta = find_primitive_root(n, t)?;
        let zeta_inv = mod_inverse(zeta, t);
        let n_inv = mod_inverse(n as u64, t);
        let mu = (1u128 << 127) / (t as u128) * 2;

        // Compute inverse powers for INTT
        let zeta_inv_powers = compute_powers(zeta_inv, n, t);

        Some(Self {
            t,
            zeta,
            zeta_inv,
            n_inv,
            mu,
            zeta_inv_powers,
        })
    }
}

/// Parameters for RNS batched slot multiplication.
#[derive(Clone)]
pub struct RnsBatchParams {
    /// Ring dimension (8192).
    pub n: usize,
    /// Number of RNS moduli (typically 4).
    pub k: usize,
    /// Plaintext modulus (Goldilocks).
    pub t: u64,
    /// Plaintext NTT data.
    pub plaintext_data: PlaintextNttData,
    /// RNS moduli NTT data.
    pub rns_data: Vec<RnsModulusNttData>,
}

impl RnsBatchParams {
    /// Creates parameters for Goldilocks with standard RNS moduli.
    pub fn goldilocks(n: usize) -> Option<Self> {
        let t = 0xFFFFFFFF00000001u64; // Goldilocks prime

        // Standard RNS moduli for BGV (same as NTT_PRIMES_8192 in RnsParams)
        let rns_moduli: Vec<(u64, u64)> = vec![
            (1152921504606994433, find_psi(n, 1152921504606994433)?),
            (1152921504607191041, find_psi(n, 1152921504607191041)?),
            (1152921504607223809, find_psi(n, 1152921504607223809)?),
            (1152921504607338497, find_psi(n, 1152921504607338497)?),
            (1152921504607518721, find_psi(n, 1152921504607518721)?),
        ];

        let plaintext_data = PlaintextNttData::new(n, t)?;
        let rns_data: Vec<_> = rns_moduli
            .iter()
            .map(|&(q, psi)| RnsModulusNttData::new(n, q, psi))
            .collect();

        Some(Self {
            n,
            k: rns_data.len(),
            t,
            plaintext_data,
            rns_data,
        })
    }

    /// Creates parameters from existing RNS params.
    pub fn from_moduli(n: usize, t: u64, moduli: &[(u64, u64)]) -> Option<Self> {
        let plaintext_data = PlaintextNttData::new(n, t)?;
        let rns_data: Vec<_> = moduli
            .iter()
            .map(|&(q, psi)| RnsModulusNttData::new(n, q, psi))
            .collect();

        Some(Self {
            n,
            k: rns_data.len(),
            t,
            plaintext_data,
            rns_data,
        })
    }
}

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
    mod_idx: u32,
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

/// GPU context for RNS batched slot multiplication.
///
/// This accelerates the core IT-PAC operation: multiplying one ciphertext
/// by many plaintexts across all RNS moduli.
pub struct RnsSlotMulGpu {
    device: Device,
    queue: Queue,

    // Pipelines
    slot_encode_pipeline: ComputePipeline,
    forward_ntt_pipeline: ComputePipeline,
    pointwise_mul_pipeline: ComputePipeline,
    pointwise_mul_multi_ct_pipeline: ComputePipeline,
    inverse_ntt_pipeline: ComputePipeline,

    // Parameters
    params: RnsBatchParams,

    // Precomputed buffers (uploaded once)
    plaintext_twiddles_buffer: Buffer,
    plaintext_params_buffer: Buffer,
    rns_twiddles_buffers: Vec<Buffer>,
    rns_inv_twiddles_buffers: Vec<Buffer>,
    rns_params_buffers: Vec<Buffer>,
}

impl std::fmt::Debug for RnsSlotMulGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RnsSlotMulGpu")
            .field("n", &self.params.n)
            .field("k", &self.params.k)
            .finish_non_exhaustive()
    }
}

impl RnsSlotMulGpu {
    /// Returns a reference to the GPU device (for sharing with other GPU contexts).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the GPU queue (for sharing with other GPU contexts).
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

impl RnsSlotMulGpu {
    /// Creates a new GPU context for RNS batched slot multiplication.
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rns-slot-mul device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroups_per_dimension: 65535,
                        max_storage_buffer_binding_size: 1024 * 1024 * 512, // 512MB
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        // Compile shaders using naga_oil composition for shared math module
        let slot_encode_shader = crate::shader_math::create_shader_module(
            &device,
            SLOT_ENCODE_SHADER,
            "slot_encode.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let forward_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            FORWARD_NTT_SHADER,
            "forward_ntt.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let pointwise_mul_shader = crate::shader_math::create_shader_module(
            &device,
            POINTWISE_MUL_SHADER,
            "pointwise_mul.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let inverse_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            INVERSE_NTT_SHADER,
            "inverse_ntt.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        // Create pipelines
        let slot_encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_encode pipeline"),
                layout: None,
                module: &slot_encode_shader,
                entry_point: Some("slot_encode_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        let forward_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("forward_ntt pipeline"),
                layout: None,
                module: &forward_ntt_shader,
                entry_point: Some("forward_ntt_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pointwise_mul_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pointwise_mul pipeline"),
                layout: None,
                module: &pointwise_mul_shader,
                entry_point: Some("pointwise_mul_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pointwise_mul_multi_ct_shader = crate::shader_math::create_shader_module(
            &device,
            POINTWISE_MUL_MULTI_CT_SHADER,
            "pointwise_mul_multi_ct.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let pointwise_mul_multi_ct_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pointwise_mul_multi_ct pipeline"),
                layout: None,
                module: &pointwise_mul_multi_ct_shader,
                entry_point: Some("pointwise_mul_multi_ct"),
                compilation_options: Default::default(),
                cache: None,
            });

        let inverse_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("inverse_ntt pipeline"),
                layout: None,
                module: &inverse_ntt_shader,
                entry_point: Some("inverse_ntt_batched"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Upload plaintext twiddles
        let pt_twiddles_u32: Vec<u32> = params
            .plaintext_data
            .zeta_inv_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let plaintext_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_twiddles"),
                contents: bytemuck::cast_slice(&pt_twiddles_u32),
                usage: BufferUsages::STORAGE,
            });

        let pt_params = GpuModulusParams {
            modulus_lo: params.plaintext_data.t as u32,
            modulus_hi: (params.plaintext_data.t >> 32) as u32,
            mu_lo: params.plaintext_data.mu as u32,
            mu_hi: (params.plaintext_data.mu >> 32) as u32,
            n_inv_lo: params.plaintext_data.n_inv as u32,
            n_inv_hi: (params.plaintext_data.n_inv >> 32) as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let plaintext_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_params"),
                contents: bytemuck::bytes_of(&pt_params),
                usage: BufferUsages::UNIFORM,
            });

        // Upload RNS twiddles for each modulus
        let mut rns_twiddles_buffers = Vec::with_capacity(params.k);
        let mut rns_inv_twiddles_buffers = Vec::with_capacity(params.k);
        let mut rns_params_buffers = Vec::with_capacity(params.k);

        for (i, rns) in params.rns_data.iter().enumerate() {
            // Forward twiddles (psi powers for twist + omega powers for NTT)
            let fwd_twiddles: Vec<u32> = rns
                .psi_powers
                .iter()
                .chain(rns.omega_powers.iter())
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect();

            rns_twiddles_buffers.push(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("rns_twiddles_{}", i)),
                    contents: bytemuck::cast_slice(&fwd_twiddles),
                    usage: BufferUsages::STORAGE,
                },
            ));

            // Inverse twiddles
            let inv_twiddles: Vec<u32> = rns
                .psi_inv_powers
                .iter()
                .chain(rns.omega_inv_powers.iter())
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect();

            rns_inv_twiddles_buffers.push(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("rns_inv_twiddles_{}", i)),
                    contents: bytemuck::cast_slice(&inv_twiddles),
                    usage: BufferUsages::STORAGE,
                },
            ));

            // Modulus params
            let rns_params = GpuModulusParams {
                modulus_lo: rns.modulus as u32,
                modulus_hi: (rns.modulus >> 32) as u32,
                mu_lo: rns.mu as u32,
                mu_hi: (rns.mu >> 32) as u32,
                n_inv_lo: rns.n_inv as u32,
                n_inv_hi: (rns.n_inv >> 32) as u32,
                _pad0: 0,
                _pad1: 0,
            };

            rns_params_buffers.push(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("rns_params_{}", i)),
                    contents: bytemuck::bytes_of(&rns_params),
                    usage: BufferUsages::UNIFORM,
                },
            ));
        }

        Ok(Self {
            device,
            queue,
            slot_encode_pipeline,
            forward_ntt_pipeline,
            pointwise_mul_pipeline,
            pointwise_mul_multi_ct_pipeline,
            inverse_ntt_pipeline,
            params,
            plaintext_twiddles_buffer,
            plaintext_params_buffer,
            rns_twiddles_buffers,
            rns_inv_twiddles_buffers,
            rns_params_buffers,
        })
    }

    /// Performs batched RNS slot multiplication.
    ///
    /// Multiplies one ciphertext (in NTT domain) by many plaintext slot vectors.
    ///
    /// # Arguments
    /// * `ct_c0_ntt` - Ciphertext c0 in NTT domain, shape [k][n]
    /// * `ct_c1_ntt` - Ciphertext c1 in NTT domain, shape [k][n]
    /// * `plaintext_slots` - Batch of plaintext slot vectors, shape [num_batches][n]
    ///
    /// # Returns
    /// (c0_results, c1_results) where each is [num_batches][k][n] in coefficient domain
    pub fn mul_batched(
        &self,
        ct_c0_ntt: &[Vec<u64>],
        ct_c1_ntt: &[Vec<u64>],
        plaintext_slots: &[Vec<u64>],
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let num_batches = plaintext_slots.len();
        if num_batches == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let n = self.params.n;
        let k = self.params.k;

        // Validate inputs
        if ct_c0_ntt.len() != k || ct_c1_ntt.len() != k {
            return Err(GpuError::InvalidParams(format!(
                "CT NTT should have {} moduli",
                k
            )));
        }

        // === Step 1: Upload CT NTT to GPU ===
        let ct_c0_buffer = self.upload_rns_poly(ct_c0_ntt)?;
        let ct_c1_buffer = self.upload_rns_poly(ct_c1_ntt)?;

        // === Step 2: Upload plaintext slots ===
        let slots_flat: Vec<u32> = plaintext_slots
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let slots_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_slots"),
                contents: bytemuck::cast_slice(&slots_flat),
                usage: BufferUsages::STORAGE,
            });

        // === Step 3: Allocate intermediate and output buffers ===
        let batch_size = num_batches * n * 2 * std::mem::size_of::<u32>();
        let rns_batch_size = num_batches * k * n * 2 * std::mem::size_of::<u32>();

        // Encoded coefficients (after slot encode)
        let encoded_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_coeffs"),
            size: batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // NTT domain plaintexts (for each RNS modulus)
        let pt_ntt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_ntt"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Output c0 and c1 (RNS, coefficient domain)
        let out_c0_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c0"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c1_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c1"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Batch params
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };

        let batch_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batch_params"),
                    contents: bytemuck::bytes_of(&batch_params),
                    usage: BufferUsages::UNIFORM,
                });

        // === Step 4: Execute pipeline ===
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rns_slot_mul encoder"),
            });

        // 4a: Slot encode (INTT mod t)
        {
            let bind_group_layout = self.slot_encode_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slot_encode bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: slots_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: encoded_buffer.as_entire_binding(),
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

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        // 4b-4d: For each RNS modulus, do forward NTT, pointwise mul, inverse NTT
        for mod_idx in 0..k {
            // Forward NTT
            {
                let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("forward_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: encoded_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("forward_ntt_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, 1, 1);
            }

            // Pointwise multiplication
            {
                let bind_group_layout = self.pointwise_mul_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("pointwise_mul_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: ct_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ct_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let workgroups = ((num_batches * n) as u32 + 255) / 256;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("pointwise_mul_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pointwise_mul_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Inverse NTT
            {
                let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("inverse_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_inv_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("inverse_ntt_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inverse_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_batches as u32, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // === Step 5: Read back results ===
        let c0_results = self.read_rns_batch(&out_c0_buffer, num_batches, k, n)?;
        let c1_results = self.read_rns_batch(&out_c1_buffer, num_batches, k, n)?;

        Ok((c0_results, c1_results))
    }

    /// Performs batched RNS slot multiplication with multiple ciphertexts.
    ///
    /// This is an optimized version that batches all operations across multiple CTs
    /// in a single GPU dispatch, reducing per-dispatch overhead.
    ///
    /// # Arguments
    /// * `cts_c0_ntt` - Multiple ciphertexts c0 in NTT domain, shape [num_cts][k][n]
    /// * `cts_c1_ntt` - Multiple ciphertexts c1 in NTT domain, shape [num_cts][k][n]
    /// * `plaintext_slots` - Batch of plaintext slot vectors, shape [total_batches][n]
    /// * `batches_per_ct` - Number of plaintext batches per ciphertext
    ///
    /// # Returns
    /// (c0_results, c1_results) where each is [total_batches][k][n] in coefficient domain
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mul_batched_multi_ct(
        &self,
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        plaintext_slots: &[Vec<u64>],
        batches_per_ct: usize,
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let num_cts = cts_c0_ntt.len();
        let total_batches = plaintext_slots.len();

        if num_cts == 0 || total_batches == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let n = self.params.n;
        let k = self.params.k;

        // Validate inputs
        if cts_c1_ntt.len() != num_cts {
            return Err(GpuError::InvalidParams(
                "c0 and c1 CT counts must match".to_string(),
            ));
        }

        for (i, ct) in cts_c0_ntt.iter().enumerate() {
            if ct.len() != k {
                return Err(GpuError::InvalidParams(format!(
                    "CT {} c0 should have {} moduli, got {}",
                    i,
                    k,
                    ct.len()
                )));
            }
        }

        // === Step 1: Upload all CTs NTT to GPU (concatenated) ===
        // Layout: [ct0_mod0, ct0_mod1, ..., ct0_modk, ct1_mod0, ...]
        let all_c0_flat: Vec<u32> = cts_c0_ntt
            .iter()
            .flat_map(|ct| {
                ct.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            })
            .collect();

        let all_c1_flat: Vec<u32> = cts_c1_ntt
            .iter()
            .flat_map(|ct| {
                ct.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            })
            .collect();

        let cts_c0_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi_ct_c0_ntt"),
                contents: bytemuck::cast_slice(&all_c0_flat),
                usage: BufferUsages::STORAGE,
            });

        let cts_c1_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi_ct_c1_ntt"),
                contents: bytemuck::cast_slice(&all_c1_flat),
                usage: BufferUsages::STORAGE,
            });

        // === Step 2: Upload plaintext slots ===
        let slots_flat: Vec<u32> = plaintext_slots
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let slots_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_slots"),
                contents: bytemuck::cast_slice(&slots_flat),
                usage: BufferUsages::STORAGE,
            });

        // === Step 3: Allocate intermediate and output buffers ===
        let batch_size = total_batches * n * 2 * std::mem::size_of::<u32>();
        let rns_batch_size = total_batches * k * n * 2 * std::mem::size_of::<u32>();

        let encoded_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_coeffs"),
            size: batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pt_ntt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_ntt"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c0_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c0"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c1_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c1"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Also create regular params for slot_encode, forward_ntt, inverse_ntt
        let regular_batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: total_batches as u32,
            num_moduli: k as u32,
        };

        let regular_batch_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batch_params"),
                    contents: bytemuck::bytes_of(&regular_batch_params),
                    usage: BufferUsages::UNIFORM,
                });

        // === Step 4: Execute pipeline ===
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("multi_ct_slot_mul encoder"),
            });

        // 4a: Slot encode (INTT mod t)
        {
            let bind_group_layout = self.slot_encode_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slot_encode bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: regular_batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: slots_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: encoded_buffer.as_entire_binding(),
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

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_batches as u32, 1, 1);
        }

        // 4b-4d: For each RNS modulus, do forward NTT, pointwise mul (multi-CT), inverse NTT
        for mod_idx in 0..k {
            // Forward NTT
            {
                let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("forward_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: regular_batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: encoded_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("forward_ntt_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }

            // Pointwise multiplication (multi-CT version)
            {
                // Create params buffer with current mod_idx
                let multi_ct_params = GpuBatchParamsMultiCt {
                    n: n as u32,
                    log_n: (n as u32).trailing_zeros(),
                    num_batches: total_batches as u32,
                    num_moduli: k as u32,
                    num_cts: num_cts as u32,
                    batches_per_ct: batches_per_ct as u32,
                    mod_idx: mod_idx as u32,
                    _pad: 0,
                };

                let multi_ct_params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("batch_params_multi_ct_{}", mod_idx)),
                            contents: bytemuck::bytes_of(&multi_ct_params),
                            usage: BufferUsages::UNIFORM,
                        });

                let bind_group_layout = self.pointwise_mul_multi_ct_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("pointwise_mul_multi_ct_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: multi_ct_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: cts_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: cts_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let workgroups = ((total_batches * n) as u32 + 255) / 256;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("pointwise_mul_multi_ct_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pointwise_mul_multi_ct_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Inverse NTT
            {
                let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("inverse_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: regular_batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_inv_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("inverse_ntt_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inverse_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // === Step 5: Read back results ===
        let c0_results = self.read_rns_batch(&out_c0_buffer, total_batches, k, n)?;
        let c1_results = self.read_rns_batch(&out_c1_buffer, total_batches, k, n)?;

        Ok((c0_results, c1_results))
    }

    /// WASM async version of mul_batched_multi_ct (avoids blocking on main thread)
    #[cfg(target_arch = "wasm32")]
    pub async fn mul_batched_multi_ct(
        &self,
        cts_c0_ntt: &[Vec<Vec<u64>>],
        cts_c1_ntt: &[Vec<Vec<u64>>],
        plaintext_slots: &[Vec<u64>],
        batches_per_ct: usize,
    ) -> Result<(Vec<Vec<Vec<u64>>>, Vec<Vec<Vec<u64>>>), GpuError> {
        let num_cts = cts_c0_ntt.len();
        let total_batches = plaintext_slots.len();

        if num_cts == 0 || total_batches == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let n = self.params.n;
        let k = self.params.k;

        // Validate inputs
        if cts_c1_ntt.len() != num_cts {
            return Err(GpuError::InvalidParams(
                "c0 and c1 CT counts must match".to_string(),
            ));
        }

        for (i, ct) in cts_c0_ntt.iter().enumerate() {
            if ct.len() != k {
                return Err(GpuError::InvalidParams(format!(
                    "CT {} c0 should have {} moduli, got {}",
                    i,
                    k,
                    ct.len()
                )));
            }
        }

        // === Step 1: Upload all CTs NTT to GPU (concatenated) ===
        let all_c0_flat: Vec<u32> = cts_c0_ntt
            .iter()
            .flat_map(|ct| {
                ct.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            })
            .collect();

        let all_c1_flat: Vec<u32> = cts_c1_ntt
            .iter()
            .flat_map(|ct| {
                ct.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            })
            .collect();

        let cts_c0_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi_ct_c0_ntt"),
                contents: bytemuck::cast_slice(&all_c0_flat),
                usage: BufferUsages::STORAGE,
            });

        let cts_c1_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi_ct_c1_ntt"),
                contents: bytemuck::cast_slice(&all_c1_flat),
                usage: BufferUsages::STORAGE,
            });

        // === Step 2: Upload plaintext slots ===
        let slots_flat: Vec<u32> = plaintext_slots
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let slots_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plaintext_slots"),
                contents: bytemuck::cast_slice(&slots_flat),
                usage: BufferUsages::STORAGE,
            });

        // === Step 3: Allocate intermediate and output buffers ===
        let batch_size = total_batches * n * 2 * std::mem::size_of::<u32>();
        let rns_batch_size = total_batches * k * n * 2 * std::mem::size_of::<u32>();

        let encoded_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded_coeffs"),
            size: batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pt_ntt_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pt_ntt"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c0_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c0"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c1_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_c1"),
            size: rns_batch_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let regular_batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: total_batches as u32,
            num_moduli: k as u32,
        };

        let regular_batch_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batch_params"),
                    contents: bytemuck::bytes_of(&regular_batch_params),
                    usage: BufferUsages::UNIFORM,
                });

        // === Step 4: Execute pipeline ===
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("multi_ct_slot_mul encoder"),
            });

        // 4a: Slot encode (INTT mod t)
        {
            let bind_group_layout = self.slot_encode_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slot_encode bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: regular_batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: slots_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: encoded_buffer.as_entire_binding(),
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

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slot_encode pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_batches as u32, 1, 1);
        }

        // 4b-4d: For each RNS modulus, do forward NTT, pointwise mul (multi-CT), inverse NTT
        for mod_idx in 0..k {
            // Forward NTT
            {
                let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("forward_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: regular_batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: encoded_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("forward_ntt_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }

            // Pointwise multiplication (multi-CT version)
            {
                let multi_ct_params = GpuBatchParamsMultiCt {
                    n: n as u32,
                    log_n: (n as u32).trailing_zeros(),
                    num_batches: total_batches as u32,
                    num_moduli: k as u32,
                    num_cts: num_cts as u32,
                    batches_per_ct: batches_per_ct as u32,
                    mod_idx: mod_idx as u32,
                    _pad: 0,
                };

                let multi_ct_params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("batch_params_multi_ct_{}", mod_idx)),
                            contents: bytemuck::bytes_of(&multi_ct_params),
                            usage: BufferUsages::UNIFORM,
                        });

                let bind_group_layout = self.pointwise_mul_multi_ct_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("pointwise_mul_multi_ct_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: multi_ct_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: cts_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: cts_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pt_ntt_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("pointwise_mul_multi_ct_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pointwise_mul_multi_ct_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }

            // Inverse NTT
            {
                let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("inverse_ntt_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: regular_batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_c0_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("inverse_ntt_c0_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inverse_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }

            {
                let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("inverse_ntt_c1_{} bind group", mod_idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: regular_batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out_c1_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("inverse_ntt_c1_{} pass", mod_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.inverse_ntt_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(total_batches as u32, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        // === Step 5: Read back results using async ===
        let c0_results = self.read_rns_batch_async(&out_c0_buffer, total_batches, k, n).await?;
        let c1_results = self.read_rns_batch_async(&out_c1_buffer, total_batches, k, n).await?;

        Ok((c0_results, c1_results))
    }

    /// Performs batched INTT (inverse NTT) mod Goldilocks.
    ///
    /// This is useful for polynomial interpolation: given evaluation values,
    /// compute polynomial coefficients via INTT.
    ///
    /// # Arguments
    /// * `values` - Batch of evaluation vectors, shape [num_batches][n]
    ///
    /// # Returns
    /// Coefficient vectors [num_batches][n] representing the interpolated polynomials
    pub fn batched_intt(&self, values: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        let num_batches = values.len();
        if num_batches == 0 {
            return Ok(Vec::new());
        }

        let n = self.params.n;

        // Validate input sizes
        for (i, v) in values.iter().enumerate() {
            if v.len() != n {
                return Err(GpuError::InvalidParams(format!(
                    "Batch {} has {} elements, expected {}",
                    i,
                    v.len(),
                    n
                )));
            }
        }

        // Upload input values
        let values_flat: Vec<u32> = values
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let values_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("intt_input"),
                contents: bytemuck::cast_slice(&values_flat),
                usage: BufferUsages::STORAGE,
            });

        // Allocate output buffer
        let output_size = num_batches * n * 2 * std::mem::size_of::<u32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("intt_output"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create batch params
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            num_moduli: 1, // Not used for INTT
        };

        let batch_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("intt_batch_params"),
                    contents: bytemuck::bytes_of(&batch_params),
                    usage: BufferUsages::UNIFORM,
                });

        // Execute INTT (slot_encode pipeline does INTT mod t)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched_intt encoder"),
            });

        {
            let bind_group_layout = self.slot_encode_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batched_intt bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: values_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
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

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batched_intt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.slot_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back results
        self.read_batch(&output_buffer, num_batches, n)
    }

    /// Reads back a batch of polynomials from GPU buffer.
    fn read_batch(
        &self,
        buffer: &Buffer,
        num_batches: usize,
        n: usize,
    ) -> Result<Vec<Vec<u64>>, GpuError> {
        let size = (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..size);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(e.to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Reshape: [num_batches][n]
        let mut results = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut coeffs = Vec::with_capacity(n);
            for elem_idx in 0..n {
                let idx = (batch_idx * n + elem_idx) * 2;
                let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
                coeffs.push(val);
            }
            results.push(coeffs);
        }

        drop(data);
        staging.unmap();

        Ok(results)
    }

    // WASM async version of read_batch (avoids blocking)
    #[cfg(target_arch = "wasm32")]
    async fn read_batch_async(
        &self,
        buffer: &Buffer,
        num_batches: usize,
        n: usize,
    ) -> Result<Vec<Vec<u64>>, GpuError> {
        let size = (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..size);

        // Use wasm_bindgen_futures for proper async
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        receiver
            .await
            .map_err(|_| GpuError::ExecutionFailed("map_async canceled".to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Reshape: [num_batches][n]
        let mut results = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut coeffs = Vec::with_capacity(n);
            for elem_idx in 0..n {
                let idx = (batch_idx * n + elem_idx) * 2;
                let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
                coeffs.push(val);
            }
            results.push(coeffs);
        }

        drop(data);
        staging.unmap();

        Ok(results)
    }

    fn upload_rns_poly(&self, poly: &[Vec<u64>]) -> Result<Buffer, GpuError> {
        let flat: Vec<u32> = poly
            .iter()
            .flat_map(|residue| residue.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rns_poly"),
                contents: bytemuck::cast_slice(&flat),
                usage: BufferUsages::STORAGE,
            }))
    }

    fn read_rns_batch(
        &self,
        buffer: &Buffer,
        num_batches: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let size = (num_batches * k * n * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..size);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(e.to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Reshape: [num_batches][k][n]
        let mut results = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let mut residue = Vec::with_capacity(n);
                for elem_idx in 0..n {
                    let idx = ((batch_idx * k + mod_idx) * n + elem_idx) * 2;
                    let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
                    residue.push(val);
                }
                batch.push(residue);
            }
            results.push(batch);
        }

        drop(data);
        staging.unmap();

        Ok(results)
    }

    // WASM async version of read_rns_batch (avoids blocking)
    #[cfg(target_arch = "wasm32")]
    async fn read_rns_batch_async(
        &self,
        buffer: &Buffer,
        num_batches: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let size = (num_batches * k * n * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..size);

        // Use wasm_bindgen_futures for proper async
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        receiver
            .await
            .map_err(|_| GpuError::ExecutionFailed("map_async canceled".to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Reshape: [num_batches][k][n]
        let mut results = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let mut batch = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let mut residue = Vec::with_capacity(n);
                for elem_idx in 0..n {
                    let idx = ((batch_idx * k + mod_idx) * n + elem_idx) * 2;
                    let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
                    residue.push(val);
                }
                batch.push(residue);
            }
            results.push(batch);
        }

        drop(data);
        staging.unmap();

        Ok(results)
    }

    // =========================================================================
    // Test-only APIs (not for production use)
    // =========================================================================

    /// Runs forward NTT on GPU for a single modulus.
    ///
    /// **Test API** - not optimized for production use.
    ///
    /// Input: polynomial coefficients in natural order
    /// Output: NTT values
    pub fn test_forward_ntt(
        &self,
        input: &[u64],
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = self.params.n;
        let k = self.params.k;

        if input.len() != n {
            return Err(GpuError::InvalidParams(format!(
                "Input length {} != n {}",
                input.len(),
                n
            )));
        }
        if mod_idx >= k {
            return Err(GpuError::InvalidParams(format!(
                "mod_idx {} >= k {}",
                mod_idx, k
            )));
        }

        // Create input buffer (single batch at offset 0)
        let num_batches = 1;
        let mut input_data = vec![0u32; num_batches * n * 2];
        for elem_idx in 0..n {
            let idx = elem_idx * 2;
            input_data[idx] = input[elem_idx] as u32;
            input_data[idx + 1] = (input[elem_idx] >> 32) as u32;
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_forward_ntt input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_forward_ntt output"),
            size: (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Batch params
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        let batch_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test batch params"),
            contents: bytemuck::bytes_of(&batch_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Run forward NTT
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_forward_ntt encoder"),
        });

        {
            let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test_forward_ntt bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.rns_twiddles_buffers[mod_idx].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test_forward_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        // Copy to staging buffer
        let output_size = (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging,
            0,
            output_size,
        );

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back using channel pattern
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Extract output (single batch at offset 0)
        let mut result = Vec::with_capacity(n);
        for elem_idx in 0..n {
            let idx = elem_idx * 2;
            let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
            result.push(val);
        }

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Runs inverse NTT on GPU for a single modulus.
    ///
    /// **Test API** - not optimized for production use.
    pub fn test_inverse_ntt(
        &self,
        input: &[u64],
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = self.params.n;
        let k = self.params.k;

        if input.len() != n {
            return Err(GpuError::InvalidParams(format!(
                "Input length {} != n {}",
                input.len(),
                n
            )));
        }
        if mod_idx >= k {
            return Err(GpuError::InvalidParams(format!(
                "mod_idx {} >= k {}",
                mod_idx, k
            )));
        }

        // Create input buffer (single batch at offset 0)
        let num_batches = 1;
        let mut input_data = vec![0u32; num_batches * n * 2];
        for elem_idx in 0..n {
            let idx = elem_idx * 2;
            input_data[idx] = input[elem_idx] as u32;
            input_data[idx + 1] = (input[elem_idx] >> 32) as u32;
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_inverse_ntt input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_inverse_ntt output"),
            size: (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Batch params
        let batch_params = GpuBatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            num_moduli: k as u32,
        };
        let batch_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test batch params"),
            contents: bytemuck::bytes_of(&batch_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Run inverse NTT
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_inverse_ntt encoder"),
        });

        {
            let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test_inverse_ntt bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: batch_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.rns_inv_twiddles_buffers[mod_idx].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.rns_params_buffers[mod_idx].as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test_inverse_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        // Copy from input_buffer (inverse NTT is in-place on c0_data = binding 1)
        let output_size = (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &input_buffer,  // Read from input (in-place result)
            0,
            &staging,
            0,
            output_size,
        );

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back using channel pattern
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        // Extract output (single batch at offset 0)
        let mut result = Vec::with_capacity(n);
        for elem_idx in 0..n {
            let idx = elem_idx * 2;
            let val = (u32_data[idx] as u64) | ((u32_data[idx + 1] as u64) << 32);
            result.push(val);
        }

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Tests GPU mulmod against CPU reference.
    ///
    /// **Test API** - verifies mulmod implementation correctness.
    pub fn test_mulmod(
        &self,
        a_values: &[u64],
        b_values: &[u64],
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = a_values.len();
        if b_values.len() != n {
            return Err(GpuError::InvalidParams("a and b must have same length".to_string()));
        }
        if mod_idx >= self.params.k {
            return Err(GpuError::InvalidParams(format!("mod_idx {} >= k {}", mod_idx, self.params.k)));
        }

        let q = self.params.rns_data[mod_idx].modulus;

        // Create shader for mulmod test
        let shader_src = r#"
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

@group(0) @binding(0) var<storage, read> a_vals: array<u32>;
@group(0) @binding(1) var<storage, read> b_vals: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_vals: array<u32>;
@group(0) @binding(3) var<uniform> mod_params: ModulusParams;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    var lo = p0;
    var hi = p3;
    let mid = p1 + p2;
    let mid_lo = (mid & 0xFFFFu) << 16u;
    let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo;
    if new_lo < lo { hi = hi + 1u; }
    lo = new_lo;
    hi = hi + mid_hi;
    if p1 > 0xFFFFFFFFu - p2 { hi = hi + 0x10000u; }
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);

    var r0 = p00.x; var r1 = p00.y;
    var r2 = p11.x; var r3 = p11.y;

    var t = r1 + p01.x; var c: u32 = 0u;
    if t < r1 { c = 1u; } r1 = t;
    t = r1 + p10.x; if t < r1 { c = c + 1u; } r1 = t;

    t = r2 + p01.y; var c2: u32 = 0u;
    if t < r2 { c2 = 1u; } r2 = t;
    t = r2 + p10.y; if t < r2 { c2 = c2 + 1u; } r2 = t;
    t = r2 + c; if t < r2 { c2 = c2 + 1u; } r2 = t;
    r3 = r3 + c2;

    return vec4<u32>(r0, r1, r2, r3);
}

// Compare 128-bit: returns true if a >= b
fn ge128(a: vec4<u32>, b: vec4<u32>) -> bool {
    if a.w != b.w { return a.w > b.w; }
    if a.z != b.z { return a.z > b.z; }
    if a.y != b.y { return a.y > b.y; }
    return a.x >= b.x;
}

// 128-bit subtraction: a - b
fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r0 = a.x; var r1 = a.y; var r2 = a.z; var r3 = a.w;
    var borrow = 0u;

    if r0 >= b.x { r0 = r0 - b.x; }
    else { r0 = 0xFFFFFFFFu - (b.x - r0 - 1u); borrow = 1u; }

    var new_r1 = r1 - b.y - borrow;
    if new_r1 > r1 { borrow = 1u; } else { borrow = 0u; }
    r1 = new_r1;

    var new_r2 = r2 - b.z - borrow;
    if new_r2 > r2 { borrow = 1u; } else { borrow = 0u; }
    r2 = new_r2;

    r3 = r3 - b.w - borrow;

    return vec4<u32>(r0, r1, r2, r3);
}

// mulmod for 60-bit inputs: a * b mod q where a, b < q and q is ~60 bits
fn mulmod_60bit(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    let q128 = vec4<u32>(q.x, q.y, 0u, 0u);
    var r = prod;

    // Shift-and-subtract reduction for 128-bit product mod 60-bit q
    for (var shift = 60u; shift > 0u; shift = shift - 1u) {
        var q_shifted: vec4<u32>;
        if shift >= 64u {
            let s = shift - 64u;
            if s == 0u {
                q_shifted = vec4<u32>(0u, 0u, q.x, q.y);
            } else {
                q_shifted = vec4<u32>(0u, 0u, q.x << s, (q.y << s) | (q.x >> (32u - s)));
            }
        } else if shift >= 32u {
            let s = shift - 32u;
            if s == 0u {
                q_shifted = vec4<u32>(0u, q.x, q.y, 0u);
            } else {
                q_shifted = vec4<u32>(0u, q.x << s, (q.y << s) | (q.x >> (32u - s)), q.y >> (32u - s));
            }
        } else {
            q_shifted = vec4<u32>(q.x << shift, (q.y << shift) | (q.x >> (32u - shift)), q.y >> (32u - shift), 0u);
        }

        if ge128(r, q_shifted) {
            r = sub128(r, q_shifted);
        }
    }

    // Final reductions
    for (var i = 0u; i < 3u; i++) {
        if !ge128(r, q128) { break; }
        r = sub128(r, q128);
    }

    return vec2<u32>(r.x, r.y);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);

    let a = vec2<u32>(a_vals[idx * 2u], a_vals[idx * 2u + 1u]);
    let b = vec2<u32>(b_vals[idx * 2u], b_vals[idx * 2u + 1u]);

    let result = mulmod_60bit(a, b, q);

    out_vals[idx * 2u] = result.x;
    out_vals[idx * 2u + 1u] = result.y;
}
"#;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mulmod_test"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mulmod_test pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers
        let a_data: Vec<u32> = a_values.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();
        let b_data: Vec<u32> = b_values.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a_buffer"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b_buffer"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buffer"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mulmod_test bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.rns_params_buffers[mod_idx].as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mulmod_test encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mulmod_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            result.push(val);
        }

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Tests GPU addmod and submod against CPU reference.
    ///
    /// **Test API** - verifies addmod/submod implementation correctness.
    pub fn test_addmod_submod(
        &self,
        a_values: &[u64],
        b_values: &[u64],
        mod_idx: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), GpuError> {
        let n = a_values.len();
        if b_values.len() != n {
            return Err(GpuError::InvalidParams("a and b must have same length".to_string()));
        }
        if mod_idx >= self.params.k {
            return Err(GpuError::InvalidParams(format!("mod_idx {} >= k {}", mod_idx, self.params.k)));
        }

        // Create shader for addmod/submod test
        let shader_src = r#"
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

@group(0) @binding(0) var<storage, read> a_vals: array<u32>;
@group(0) @binding(1) var<storage, read> b_vals: array<u32>;
@group(0) @binding(2) var<storage, read_write> add_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> sub_out: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

fn addmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var sum_lo = a.x + b.x;
    var carry = 0u;
    if sum_lo < a.x { carry = 1u; }
    var sum_hi = a.y + b.y + carry;

    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x { sum_lo = sum_lo - q.x; }
        else { sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u); sum_hi = sum_hi - 1u; }
        sum_hi = sum_hi - q.y;
    }
    return vec2<u32>(sum_lo, sum_hi);
}

fn submod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        return vec2<u32>(diff_lo, diff_hi);
    } else {
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;
        var res_lo = q.x - diff_lo;
        borrow = 0u;
        if q.x < diff_lo { borrow = 1u; }
        var res_hi = q.y - diff_hi - borrow;
        return vec2<u32>(res_lo, res_hi);
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let a = vec2<u32>(a_vals[idx * 2u], a_vals[idx * 2u + 1u]);
    let b = vec2<u32>(b_vals[idx * 2u], b_vals[idx * 2u + 1u]);
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);

    let add_result = addmod(a, b, q);
    add_out[idx * 2u] = add_result.x;
    add_out[idx * 2u + 1u] = add_result.y;

    let sub_result = submod(a, b, q);
    sub_out[idx * 2u] = sub_result.x;
    sub_out[idx * 2u + 1u] = sub_result.y;
}
"#;

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("addmod_submod_test shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("addmod_submod_test pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let a_data: Vec<u32> = a_values.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();
        let b_data: Vec<u32> = b_values.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a_buffer"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b_buffer"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let add_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("add_out_buffer"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sub_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sub_out_buffer"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("addmod_submod_test bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: add_out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sub_out_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.rns_params_buffers[mod_idx].as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("addmod_submod_test encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("addmod_submod_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let add_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("add_staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sub_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sub_staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&add_out_buffer, 0, &add_staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);
        encoder.copy_buffer_to_buffer(&sub_out_buffer, 0, &sub_staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read add results
        let buffer_slice = add_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let mut add_result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            add_result.push(val);
        }
        drop(data);
        add_staging.unmap();

        // Read sub results
        let buffer_slice = sub_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let mut sub_result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            sub_result.push(val);
        }
        drop(data);
        sub_staging.unmap();

        Ok((add_result, sub_result))
    }

    /// Test bit_reverse function in isolation.
    pub fn test_bit_reverse(
        &self,
        indices: &[u32],
        log_n: u32,
    ) -> Result<Vec<u32>, GpuError> {
        let n = indices.len();

        let shader_source = format!(r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn bit_reverse(x: u32, bits: u32) -> u32 {{
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {{
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }}
    return r;
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= {n}u {{ return; }}
    output[idx] = bit_reverse(input[idx], {log_n}u);
}}
"#, n = n, log_n = log_n);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bit_reverse_test shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bit_reverse_test pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (n * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bit_reverse_test bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bit_reverse_test encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bit_reverse_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (n * std::mem::size_of::<u32>()) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Test twiddle factor (omega_inv^i) computation in isolation.
    pub fn test_twiddle_factors(
        &self,
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = self.params.n;
        let data = &self.params.rns_data[mod_idx];
        let q = data.modulus;
        let omega_inv = data.omega_inv;

        // Create buffer with omega_inv powers (what should be in twiddles)
        // We'll compute them on GPU and compare
        let shader_source = format!(r#"
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {{
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    var lo = p0;
    var hi = p3;
    let mid = p1 + p2;
    let mid_lo = (mid & 0xFFFFu) << 16u;
    let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo;
    if new_lo < lo {{ hi = hi + 1u; }}
    lo = new_lo;
    hi = hi + mid_hi;
    if p1 > 0xFFFFFFFFu - p2 {{ hi = hi + 0x10000u; }}
    return vec2<u32>(lo, hi);
}}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {{
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x;
    var r1 = p00.y;
    var r2 = p11.x;
    var r3 = p11.y;
    var carry: u32 = 0u;
    var sum = r1 + p01.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r1 + p10.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p01.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p10.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    return vec4<u32>(r0, r1, r2, r3);
}}

fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {{
    var r: vec4<u32>;
    var borrow: u32 = 0u;
    if a.x >= b.x {{ r.x = a.x - b.x; borrow = 0u; }}
    else {{ r.x = 0xFFFFFFFFu - (b.x - a.x - 1u); borrow = 1u; }}
    let t1 = a.y - borrow;
    borrow = select(0u, 1u, a.y < borrow);
    if t1 >= b.y {{ r.y = t1 - b.y; }}
    else {{ r.y = 0xFFFFFFFFu - (b.y - t1 - 1u); borrow = borrow + 1u; }}
    let t2 = a.z - borrow;
    borrow = select(0u, 1u, a.z < borrow);
    if t2 >= b.z {{ r.z = t2 - b.z; }}
    else {{ r.z = 0xFFFFFFFFu - (b.z - t2 - 1u); borrow = borrow + 1u; }}
    r.w = a.w - b.w - borrow;
    return r;
}}

fn ge128(a: vec4<u32>, b: vec4<u32>) -> bool {{
    if a.w != b.w {{ return a.w > b.w; }}
    if a.z != b.z {{ return a.z > b.z; }}
    if a.y != b.y {{ return a.y > b.y; }}
    return a.x >= b.x;
}}

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {{
    var prod = mul64(a, b);
    let q128 = vec4<u32>(q.x, q.y, 0u, 0u);
    for (var shift = 63; shift >= 0; shift--) {{
        if !ge128(prod, q128) {{ break; }}
        var shifted_q: vec4<u32>;
        let s = u32(shift);
        if s >= 64u {{
            let ss = s - 64u;
            shifted_q.x = 0u;
            shifted_q.y = 0u;
            shifted_q.z = q.x << ss;
            shifted_q.w = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            if ss > 0u {{ shifted_q.z = shifted_q.z | select(0u, q.y >> (32u - ss), ss < 32u); }}
        }} else if s >= 32u {{
            let ss = s - 32u;
            shifted_q.x = 0u;
            shifted_q.y = q.x << ss;
            shifted_q.z = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            shifted_q.w = select(0u, q.y >> (32u - ss), ss > 0u);
        }} else if s > 0u {{
            shifted_q.x = q.x << s;
            shifted_q.y = (q.x >> (32u - s)) | (q.y << s);
            shifted_q.z = q.y >> (32u - s);
            shifted_q.w = 0u;
        }} else {{
            shifted_q = q128;
        }}
        if ge128(prod, shifted_q) {{
            prod = sub128(prod, shifted_q);
        }}
    }}
    return vec2<u32>(prod.x, prod.y);
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= {n}u {{ return; }}

    // Compute omega_inv^idx iteratively
    let q = vec2<u32>({q_lo}u, {q_hi}u);
    let omega_inv = vec2<u32>({omega_inv_lo}u, {omega_inv_hi}u);

    var result = vec2<u32>(1u, 0u);  // Start with 1
    var base = omega_inv;
    var exp = idx;

    while exp > 0u {{
        if (exp & 1u) == 1u {{
            result = mulmod(result, base, q);
        }}
        exp = exp >> 1u;
        base = mulmod(base, base, q);
    }}

    output[idx * 2u] = result.x;
    output[idx * 2u + 1u] = result.y;
}}
"#, n = n,
   q_lo = q as u32, q_hi = (q >> 32) as u32,
   omega_inv_lo = omega_inv as u32, omega_inv_hi = (omega_inv >> 32) as u32);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twiddle_test shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("twiddle_test pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("twiddle_test bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("twiddle_test encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twiddle_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            result.push(val);
        }
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Test untwist operation (multiply by psi_inv^i) in isolation.
    pub fn test_untwist(
        &self,
        values: &[u64],
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = values.len();
        let data = &self.params.rns_data[mod_idx];
        let q = data.modulus;
        let psi_inv = data.psi_inv;

        // Pack input values
        let mut input_data: Vec<u32> = Vec::with_capacity(n * 2);
        for &v in values {
            input_data.push(v as u32);
            input_data.push((v >> 32) as u32);
        }

        let shader_source = format!(r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {{
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    var lo = p0;
    var hi = p3;
    let mid = p1 + p2;
    let mid_lo = (mid & 0xFFFFu) << 16u;
    let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo;
    if new_lo < lo {{ hi = hi + 1u; }}
    lo = new_lo;
    hi = hi + mid_hi;
    if p1 > 0xFFFFFFFFu - p2 {{ hi = hi + 0x10000u; }}
    return vec2<u32>(lo, hi);
}}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {{
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x;
    var r1 = p00.y;
    var r2 = p11.x;
    var r3 = p11.y;
    var carry: u32 = 0u;
    var sum = r1 + p01.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r1 + p10.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p01.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p10.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    return vec4<u32>(r0, r1, r2, r3);
}}

fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {{
    var r: vec4<u32>;
    var borrow: u32 = 0u;
    if a.x >= b.x {{ r.x = a.x - b.x; borrow = 0u; }}
    else {{ r.x = 0xFFFFFFFFu - (b.x - a.x - 1u); borrow = 1u; }}
    let t1 = a.y - borrow;
    borrow = select(0u, 1u, a.y < borrow);
    if t1 >= b.y {{ r.y = t1 - b.y; }}
    else {{ r.y = 0xFFFFFFFFu - (b.y - t1 - 1u); borrow = borrow + 1u; }}
    let t2 = a.z - borrow;
    borrow = select(0u, 1u, a.z < borrow);
    if t2 >= b.z {{ r.z = t2 - b.z; }}
    else {{ r.z = 0xFFFFFFFFu - (b.z - t2 - 1u); borrow = borrow + 1u; }}
    r.w = a.w - b.w - borrow;
    return r;
}}

fn ge128(a: vec4<u32>, b: vec4<u32>) -> bool {{
    if a.w != b.w {{ return a.w > b.w; }}
    if a.z != b.z {{ return a.z > b.z; }}
    if a.y != b.y {{ return a.y > b.y; }}
    return a.x >= b.x;
}}

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {{
    var prod = mul64(a, b);
    let q128 = vec4<u32>(q.x, q.y, 0u, 0u);
    for (var shift = 63; shift >= 0; shift--) {{
        if !ge128(prod, q128) {{ break; }}
        var shifted_q: vec4<u32>;
        let s = u32(shift);
        if s >= 64u {{
            let ss = s - 64u;
            shifted_q.x = 0u;
            shifted_q.y = 0u;
            shifted_q.z = q.x << ss;
            shifted_q.w = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            if ss > 0u {{ shifted_q.z = shifted_q.z | select(0u, q.y >> (32u - ss), ss < 32u); }}
        }} else if s >= 32u {{
            let ss = s - 32u;
            shifted_q.x = 0u;
            shifted_q.y = q.x << ss;
            shifted_q.z = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            shifted_q.w = select(0u, q.y >> (32u - ss), ss > 0u);
        }} else if s > 0u {{
            shifted_q.x = q.x << s;
            shifted_q.y = (q.x >> (32u - s)) | (q.y << s);
            shifted_q.z = q.y >> (32u - s);
            shifted_q.w = 0u;
        }} else {{
            shifted_q = q128;
        }}
        if ge128(prod, shifted_q) {{
            prod = sub128(prod, shifted_q);
        }}
    }}
    return vec2<u32>(prod.x, prod.y);
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= {n}u {{ return; }}

    let q = vec2<u32>({q_lo}u, {q_hi}u);
    let psi_inv = vec2<u32>({psi_inv_lo}u, {psi_inv_hi}u);

    // Compute psi_inv^idx
    var psi_inv_power = vec2<u32>(1u, 0u);
    var base = psi_inv;
    var exp = idx;
    while exp > 0u {{
        if (exp & 1u) == 1u {{
            psi_inv_power = mulmod(psi_inv_power, base, q);
        }}
        exp = exp >> 1u;
        base = mulmod(base, base, q);
    }}

    // Read input value and multiply by psi_inv^idx
    let val = vec2<u32>(input[idx * 2u], input[idx * 2u + 1u]);
    let result = mulmod(val, psi_inv_power, q);

    output[idx * 2u] = result.x;
    output[idx * 2u + 1u] = result.y;
}}
"#, n = n,
   q_lo = q as u32, q_hi = (q >> 32) as u32,
   psi_inv_lo = psi_inv as u32, psi_inv_hi = (psi_inv >> 32) as u32);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("untwist_test shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("untwist_test pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("untwist_test bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("untwist_test encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("untwist_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            result.push(val);
        }
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Test n_inv scaling in isolation.
    pub fn test_n_inv_scaling(
        &self,
        values: &[u64],
        mod_idx: usize,
    ) -> Result<Vec<u64>, GpuError> {
        let n = values.len();
        let data = &self.params.rns_data[mod_idx];
        let q = data.modulus;
        let n_inv = data.n_inv;

        // Pack input values
        let mut input_data: Vec<u32> = Vec::with_capacity(n * 2);
        for &v in values {
            input_data.push(v as u32);
            input_data.push((v >> 32) as u32);
        }

        let shader_source = format!(r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {{
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    var lo = p0;
    var hi = p3;
    let mid = p1 + p2;
    let mid_lo = (mid & 0xFFFFu) << 16u;
    let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo;
    if new_lo < lo {{ hi = hi + 1u; }}
    lo = new_lo;
    hi = hi + mid_hi;
    if p1 > 0xFFFFFFFFu - p2 {{ hi = hi + 0x10000u; }}
    return vec2<u32>(lo, hi);
}}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {{
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x;
    var r1 = p00.y;
    var r2 = p11.x;
    var r3 = p11.y;
    var carry: u32 = 0u;
    var sum = r1 + p01.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r1 + p10.x;
    if sum < r1 {{ carry = 1u; }}
    r1 = sum;
    sum = r2 + carry;
    carry = 0u;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p01.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    carry = 0u;
    sum = r2 + p10.y;
    if sum < r2 {{ carry = 1u; }}
    r2 = sum;
    r3 = r3 + carry;
    return vec4<u32>(r0, r1, r2, r3);
}}

fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {{
    var r: vec4<u32>;
    var borrow: u32 = 0u;
    if a.x >= b.x {{ r.x = a.x - b.x; borrow = 0u; }}
    else {{ r.x = 0xFFFFFFFFu - (b.x - a.x - 1u); borrow = 1u; }}
    let t1 = a.y - borrow;
    borrow = select(0u, 1u, a.y < borrow);
    if t1 >= b.y {{ r.y = t1 - b.y; }}
    else {{ r.y = 0xFFFFFFFFu - (b.y - t1 - 1u); borrow = borrow + 1u; }}
    let t2 = a.z - borrow;
    borrow = select(0u, 1u, a.z < borrow);
    if t2 >= b.z {{ r.z = t2 - b.z; }}
    else {{ r.z = 0xFFFFFFFFu - (b.z - t2 - 1u); borrow = borrow + 1u; }}
    r.w = a.w - b.w - borrow;
    return r;
}}

fn ge128(a: vec4<u32>, b: vec4<u32>) -> bool {{
    if a.w != b.w {{ return a.w > b.w; }}
    if a.z != b.z {{ return a.z > b.z; }}
    if a.y != b.y {{ return a.y > b.y; }}
    return a.x >= b.x;
}}

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {{
    var prod = mul64(a, b);
    let q128 = vec4<u32>(q.x, q.y, 0u, 0u);
    for (var shift = 63; shift >= 0; shift--) {{
        if !ge128(prod, q128) {{ break; }}
        var shifted_q: vec4<u32>;
        let s = u32(shift);
        if s >= 64u {{
            let ss = s - 64u;
            shifted_q.x = 0u;
            shifted_q.y = 0u;
            shifted_q.z = q.x << ss;
            shifted_q.w = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            if ss > 0u {{ shifted_q.z = shifted_q.z | select(0u, q.y >> (32u - ss), ss < 32u); }}
        }} else if s >= 32u {{
            let ss = s - 32u;
            shifted_q.x = 0u;
            shifted_q.y = q.x << ss;
            shifted_q.z = select(0u, q.x >> (32u - ss), ss > 0u) | (q.y << ss);
            shifted_q.w = select(0u, q.y >> (32u - ss), ss > 0u);
        }} else if s > 0u {{
            shifted_q.x = q.x << s;
            shifted_q.y = (q.x >> (32u - s)) | (q.y << s);
            shifted_q.z = q.y >> (32u - s);
            shifted_q.w = 0u;
        }} else {{
            shifted_q = q128;
        }}
        if ge128(prod, shifted_q) {{
            prod = sub128(prod, shifted_q);
        }}
    }}
    return vec2<u32>(prod.x, prod.y);
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if idx >= {n}u {{ return; }}

    let q = vec2<u32>({q_lo}u, {q_hi}u);
    let n_inv = vec2<u32>({n_inv_lo}u, {n_inv_hi}u);

    let val = vec2<u32>(input[idx * 2u], input[idx * 2u + 1u]);
    let result = mulmod(val, n_inv, q);

    output[idx * 2u] = result.x;
    output[idx * 2u + 1u] = result.y;
}}
"#, n = n,
   q_lo = q as u32, q_hi = (q >> 32) as u32,
   n_inv_lo = n_inv as u32, n_inv_hi = (n_inv >> 32) as u32);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("n_inv_test shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("n_inv_test pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("n_inv_test bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("n_inv_test encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("n_inv_test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (n * 2 * std::mem::size_of::<u32>()) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {:?}", e)))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let val = (u32_data[i * 2] as u64) | ((u32_data[i * 2 + 1] as u64) << 32);
            result.push(val);
        }
        drop(data);
        staging.unmap();

        Ok(result)
    }
}
// ============================================================================
// WGSL Shaders
// ============================================================================

/// Slot encoding shader (INTT mod t to convert slots to coefficients).
const SLOT_ENCODE_SHADER: &str = r#"
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
    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load with bit-reversal (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(slots[base], slots[base + 1u]);
        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // INTT butterfly stages
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;
        let butterflies_per_thread = (n >> 1u) / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::mulmod(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Scale by n^-1 and store (inline array access)
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        val = math::mulmod(val, n_inv, q);
        let out_base = (batch_offset + idx) * 2u;
        coeffs[out_base] = val.x;
        coeffs[out_base + 1u] = val.y;
    }
}
"#;

/// Forward NTT shader (twist + NTT for RNS modulus).
const FORWARD_NTT_SHADER: &str = r#"
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
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> coeffs: array<u32>;
@group(0) @binding(2) var<storage, read_write> ntt_out: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

@compute @workgroup_size(256, 1, 1)
fn forward_ntt_batched(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches { return; }

    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load, twist, and bit-reverse (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let coeff_base = (batch_offset + idx) * 2u;
        var val = vec2<u32>(coeffs[coeff_base], coeffs[coeff_base + 1u]);

        // Twist: multiply by psi^idx
        let psi_base = idx * 2u;
        let psi_power = vec2<u32>(twiddles[psi_base], twiddles[psi_base + 1u]);
        val = math::mulmod(val, psi_power, q);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // Forward NTT butterfly stages (Cooley-Tukey)
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;
        let butterflies_per_thread = (n >> 1u) / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            // Omega powers at offset n in twiddles buffer
            let twiddle_idx = n + idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = math::mulmod(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Store results (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let val = vec2<u32>(shared_lo[idx], shared_hi[idx]);
        let out_base = (batch_offset + idx) * 2u;
        ntt_out[out_base] = val.x;
        ntt_out[out_base + 1u] = val.y;
    }
}
"#;

/// Pointwise multiplication shader.
const POINTWISE_MUL_SHADER: &str = r#"
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
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> ct_c0_ntt: array<u32>;
@group(0) @binding(2) var<storage, read> ct_c1_ntt: array<u32>;
@group(0) @binding(3) var<storage, read> pt_ntt: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_c0: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_c1: array<u32>;
@group(0) @binding(6) var<uniform> mod_params: ModulusParams;

@compute @workgroup_size(256, 1, 1)
fn pointwise_mul_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.n * params.num_batches;

    if idx >= total { return; }

    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);

    // CT is single (not batched), PT is batched
    let elem_idx = idx % params.n;
    let base_ct = elem_idx * 2u;
    let base_pt = idx * 2u;

    let c0 = vec2<u32>(ct_c0_ntt[base_ct], ct_c0_ntt[base_ct + 1u]);
    let c1 = vec2<u32>(ct_c1_ntt[base_ct], ct_c1_ntt[base_ct + 1u]);
    let pt = vec2<u32>(pt_ntt[base_pt], pt_ntt[base_pt + 1u]);

    let out0 = math::mulmod(c0, pt, q);
    let out1 = math::mulmod(c1, pt, q);

    out_c0[base_pt] = out0.x;
    out_c0[base_pt + 1u] = out0.y;
    out_c1[base_pt] = out1.x;
    out_c1[base_pt + 1u] = out1.y;
}
"#;

/// Multi-CT pointwise multiplication shader.
/// Supports batching across multiple ciphertexts for improved GPU utilization.
const POINTWISE_MUL_MULTI_CT_SHADER: &str = r#"
#import math

struct BatchParamsMultiCt {
    n: u32,
    log_n: u32,
    num_batches: u32,
    num_moduli: u32,
    num_cts: u32,
    batches_per_ct: u32,
    mod_idx: u32,
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

@group(0) @binding(0) var<uniform> params: BatchParamsMultiCt;
@group(0) @binding(1) var<storage, read> cts_c0_ntt: array<u32>;  // All CTs c0, layout: [ct0_mod0, ct0_mod1, ..., ct1_mod0, ...]
@group(0) @binding(2) var<storage, read> cts_c1_ntt: array<u32>;  // All CTs c1
@group(0) @binding(3) var<storage, read> pt_ntt: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_c0: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_c1: array<u32>;
@group(0) @binding(6) var<uniform> mod_params: ModulusParams;

@compute @workgroup_size(256, 1, 1)
fn pointwise_mul_multi_ct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.n * params.num_batches;

    if idx >= total { return; }

    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let n = params.n;
    let k = params.num_moduli;

    // Determine which batch and element we're processing
    let batch_idx = idx / n;
    let elem_idx = idx % n;

    // Determine which CT this batch belongs to
    let ct_idx = batch_idx / params.batches_per_ct;

    // CT buffer layout: [ct0_mod0[n], ct0_mod1[n], ..., ct0_modk-1[n], ct1_mod0[n], ...]
    // For CT ct_idx, modulus mod_idx, element elem_idx:
    // Index = ((ct_idx * k + mod_idx) * n + elem_idx) * 2
    let base_ct = ((ct_idx * k + params.mod_idx) * n + elem_idx) * 2u;

    // PT buffer layout: [batch0[n], batch1[n], ...]
    let base_pt = idx * 2u;

    let c0 = vec2<u32>(cts_c0_ntt[base_ct], cts_c0_ntt[base_ct + 1u]);
    let c1 = vec2<u32>(cts_c1_ntt[base_ct], cts_c1_ntt[base_ct + 1u]);
    let pt = vec2<u32>(pt_ntt[base_pt], pt_ntt[base_pt + 1u]);

    let out0 = math::mulmod(c0, pt, q);
    let out1 = math::mulmod(c1, pt, q);

    out_c0[base_pt] = out0.x;
    out_c0[base_pt + 1u] = out0.y;
    out_c1[base_pt] = out1.x;
    out_c1[base_pt + 1u] = out1.y;
}
"#;

/// Inverse NTT shader (INTT + untwist for RNS modulus).
const INVERSE_NTT_SHADER: &str = r#"
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
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read_write> c0_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> c1_data: array<u32>;
@group(0) @binding(3) var<storage, read> inv_twiddles: array<u32>; // [psi_inv_powers, omega_inv_powers]
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Process c0 for a batch - CT DIT inverse (matches forward NTT structure)
@compute @workgroup_size(256, 1, 1)
fn inverse_ntt_batched(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches { return; }

    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Process c0: load with bit-reversal (CT DIT takes bit-reversed input)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let data_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(c0_data[data_base], c0_data[data_base + 1u]);
        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // CT DIT INTT butterfly stages with omega_inv
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;
        let butterflies_per_thread = (n >> 1u) / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            // Omega_inv powers at offset n
            let twiddle_idx = n + idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            // CT DIT butterfly: new_u = u + w*v, new_v = u - w*v
            let tw_v = math::mulmod(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Scale by n^-1, untwist, and store
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Scale by n^-1
        val = math::mulmod(val, n_inv, q);

        // Untwist: multiply by psi_inv^idx
        let psi_inv_base = idx * 2u;
        let psi_inv_power = vec2<u32>(inv_twiddles[psi_inv_base], inv_twiddles[psi_inv_base + 1u]);
        val = math::mulmod(val, psi_inv_power, q);

        let out_base = (batch_offset + idx) * 2u;
        c0_data[out_base] = val.x;
        c0_data[out_base + 1u] = val.y;
    }

    workgroupBarrier();

    // Process c1 similarly - CT DIT inverse
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let data_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(c1_data[data_base], c1_data[data_base + 1u]);
        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;
        let butterflies_per_thread = (n >> 1u) / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;
            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            let tw_idx = n + idx_in_group * (n / m);
            let tw_base2 = tw_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base2], inv_twiddles[tw_base2 + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            // CT DIT butterfly: new_u = u + w*v, new_v = u - w*v
            let tw_v = math::mulmod(v, twiddle, q);
            let new_u = math::addmod(u, tw_v, q);
            let new_v = math::submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Scale by n^-1
        val = math::mulmod(val, n_inv, q);

        // Untwist: multiply by psi_inv^idx
        let psi_inv_base2 = idx * 2u;
        let psi_inv_power = vec2<u32>(inv_twiddles[psi_inv_base2], inv_twiddles[psi_inv_base2 + 1u]);
        val = math::mulmod(val, psi_inv_power, q);

        let out_base2 = (batch_offset + idx) * 2u;
        c1_data[out_base2] = val.x;
        c1_data[out_base2 + 1u] = val.y;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_batch_params() {
        let params = RnsBatchParams::goldilocks(8192);
        assert!(params.is_some());
        let params = params.unwrap();
        assert_eq!(params.n, 8192);
        assert_eq!(params.k, 5); // 5 RNS moduli in goldilocks config
    }
}
