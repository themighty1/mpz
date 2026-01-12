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

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;

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

        // Standard RNS moduli for BGV (same as RnsParams)
        let rns_moduli: Vec<(u64, u64)> = vec![
            (1152921504606994433, find_psi(n, 1152921504606994433)?),
            (1152921504607191041, find_psi(n, 1152921504607191041)?),
            (1152921504607223809, find_psi(n, 1152921504607223809)?),
            (1152921504607338497, find_psi(n, 1152921504607338497)?),
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

impl RnsSlotMulGpu {
    /// Creates a new GPU context for RNS batched slot multiplication.
    pub fn new(params: RnsBatchParams) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(params))
    }

    async fn new_async(params: RnsBatchParams) -> Result<Self, GpuError> {
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

        // Compile shaders
        let slot_encode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slot_encode"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SLOT_ENCODE_SHADER)),
        });

        let forward_ntt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_ntt"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(FORWARD_NTT_SHADER)),
        });

        let pointwise_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointwise_mul"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(POINTWISE_MUL_SHADER)),
        });

        let inverse_ntt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("inverse_ntt"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(INVERSE_NTT_SHADER)),
        });

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

        let pointwise_mul_multi_ct_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pointwise_mul_multi_ct"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(POINTWISE_MUL_MULTI_CT_SHADER)),
            });

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
}

// Helper functions

fn compute_powers(base: u64, n: usize, modulus: u64) -> Vec<u64> {
    let mut powers = Vec::with_capacity(n);
    let mut current = 1u64;
    for _ in 0..n {
        powers.push(current);
        current = mod_mul(current, base, modulus);
    }
    powers
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

fn mod_inverse(a: u64, modulus: u64) -> u64 {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = modulus as i128;
    let mut new_r: i128 = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        let temp = t - quotient * new_t;
        t = new_t;
        new_t = temp;
        let temp = r - quotient * new_r;
        r = new_r;
        new_r = temp;
    }

    if t < 0 {
        (t + modulus as i128) as u64
    } else {
        t as u64
    }
}

fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
    let order = 2 * n as u64;
    if (q - 1) % order != 0 {
        return None;
    }
    let exp = (q - 1) / order;
    for g in 2..1000u64 {
        let root = mod_pow(g, exp, q);
        let root_n = mod_pow(root, n as u64, q);
        if root_n == q - 1 {
            return Some(root);
        }
    }
    None
}

fn find_psi(n: usize, q: u64) -> Option<u64> {
    find_primitive_root(n, q)
}

// ============================================================================
// WGSL Shaders
// ============================================================================

/// Slot encoding shader (INTT mod t to convert slots to coefficients).
const SLOT_ENCODE_SHADER: &str = r#"
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

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

// 64-bit modular arithmetic helpers
fn addmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var sum_lo = a.x + b.x;
    var carry = 0u;
    if sum_lo < a.x { carry = 1u; }
    var sum_hi = a.y + b.y + carry;

    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
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

    if p1 > 0xFFFFFFFFu - p2 {
        hi = hi + 0x10000u;
    }

    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);

    var r0 = p00.x;
    var r1 = p00.y;
    var r2 = p11.x;
    var r3 = p11.y;

    var t = r1 + p01.x;
    var c: u32 = 0u;
    if t < r1 { c = 1u; }
    r1 = t;
    t = r1 + p10.x;
    if t < r1 { c = c + 1u; }
    r1 = t;

    t = r2 + p01.y;
    var c2: u32 = 0u;
    if t < r2 { c2 = 1u; }
    r2 = t;
    t = r2 + p10.y;
    if t < r2 { c2 = c2 + 1u; }
    r2 = t;
    t = r2 + c;
    if t < r2 { c2 = c2 + 1u; }
    r2 = t;

    r3 = r3 + c2;

    return vec4<u32>(r0, r1, r2, r3);
}

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);

    // Fast path: product fits in 64 bits
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x;
        var r1 = prod.y;
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
            var borrow = 0u;
            if r0 >= q.x { r0 = r0 - q.x; }
            else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }

    // For larger products, use simple repeated subtraction (slow but correct)
    var r0 = prod.x;
    var r1 = prod.y;
    for (var i = 0u; i < 10u; i++) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
        var borrow = 0u;
        if r0 >= q.x { r0 = r0 - q.x; }
        else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
        r1 = r1 - q.y - borrow;
    }
    return vec2<u32>(r0, r1);
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
    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load with bit-reversal (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(slots[base], slots[base + 1u]);
        let rev_idx = bit_reverse(idx, log_n);
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

            let tw_v = mulmod(v, twiddle, q);
            let new_u = addmod(u, tw_v, q);
            let new_v = submod(u, tw_v, q);

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
        val = mulmod(val, n_inv, q);
        let out_base = (batch_offset + idx) * 2u;
        coeffs[out_base] = val.x;
        coeffs[out_base + 1u] = val.y;
    }
}
"#;

/// Forward NTT shader (twist + NTT for RNS modulus).
const FORWARD_NTT_SHADER: &str = r#"
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

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

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

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x; var r1 = prod.y;
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
            var borrow = 0u;
            if r0 >= q.x { r0 = r0 - q.x; }
            else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }
    var r0 = prod.x; var r1 = prod.y;
    for (var i = 0u; i < 10u; i++) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
        var borrow = 0u;
        if r0 >= q.x { r0 = r0 - q.x; }
        else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
        r1 = r1 - q.y - borrow;
    }
    return vec2<u32>(r0, r1);
}

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
        val = mulmod(val, psi_power, q);

        let rev_idx = bit_reverse(idx, log_n);
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

            let tw_v = mulmod(v, twiddle, q);
            let new_u = addmod(u, tw_v, q);
            let new_v = submod(u, tw_v, q);

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

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x; var r1 = prod.y;
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
            var borrow = 0u;
            if r0 >= q.x { r0 = r0 - q.x; }
            else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }
    var r0 = prod.x; var r1 = prod.y;
    for (var i = 0u; i < 10u; i++) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
        var borrow = 0u;
        if r0 >= q.x { r0 = r0 - q.x; }
        else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
        r1 = r1 - q.y - borrow;
    }
    return vec2<u32>(r0, r1);
}

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

    let out0 = mulmod(c0, pt, q);
    let out1 = mulmod(c1, pt, q);

    out_c0[base_pt] = out0.x;
    out_c0[base_pt + 1u] = out0.y;
    out_c1[base_pt] = out1.x;
    out_c1[base_pt + 1u] = out1.y;
}
"#;

/// Multi-CT pointwise multiplication shader.
/// Supports batching across multiple ciphertexts for improved GPU utilization.
const POINTWISE_MUL_MULTI_CT_SHADER: &str = r#"
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

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x; var r1 = prod.y;
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
            var borrow = 0u;
            if r0 >= q.x { r0 = r0 - q.x; }
            else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }
    var r0 = prod.x; var r1 = prod.y;
    for (var i = 0u; i < 10u; i++) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
        var borrow = 0u;
        if r0 >= q.x { r0 = r0 - q.x; }
        else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
        r1 = r1 - q.y - borrow;
    }
    return vec2<u32>(r0, r1);
}

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

    let out0 = mulmod(c0, pt, q);
    let out1 = mulmod(c1, pt, q);

    out_c0[base_pt] = out0.x;
    out_c0[base_pt + 1u] = out0.y;
    out_c1[base_pt] = out1.x;
    out_c1[base_pt + 1u] = out1.y;
}
"#;

/// Inverse NTT shader (INTT + untwist for RNS modulus).
const INVERSE_NTT_SHADER: &str = r#"
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

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

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

fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x; var r1 = prod.y;
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
            var borrow = 0u;
            if r0 >= q.x { r0 = r0 - q.x; }
            else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }
    var r0 = prod.x; var r1 = prod.y;
    for (var i = 0u; i < 10u; i++) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) { break; }
        var borrow = 0u;
        if r0 >= q.x { r0 = r0 - q.x; }
        else { r0 = 0xFFFFFFFFu - (q.x - r0 - 1u); borrow = 1u; }
        r1 = r1 - q.y - borrow;
    }
    return vec2<u32>(r0, r1);
}

// Process c0 for a batch
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

    // Process c0: load with bit-reversal (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let data_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(c0_data[data_base], c0_data[data_base + 1u]);
        let rev_idx = bit_reverse(idx, log_n);
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

            // Omega_inv powers at offset n (inline array access)
            let twiddle_idx = n + idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = mulmod(v, twiddle, q);
            let new_u = addmod(u, tw_v, q);
            let new_v = submod(u, tw_v, q);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Untwist, scale by n^-1, and store (inline array access)
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Untwist: multiply by psi_inv^idx
        let psi_inv_base = idx * 2u;
        let psi_inv_power = vec2<u32>(inv_twiddles[psi_inv_base], inv_twiddles[psi_inv_base + 1u]);
        val = mulmod(val, psi_inv_power, q);

        // Scale by n^-1
        val = mulmod(val, n_inv, q);

        let out_base = (batch_offset + idx) * 2u;
        c0_data[out_base] = val.x;
        c0_data[out_base + 1u] = val.y;
    }

    workgroupBarrier();

    // Process c1 similarly (inline array access)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let data_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(c1_data[data_base], c1_data[data_base + 1u]);
        let rev_idx = bit_reverse(idx, log_n);
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

            let tw_v = mulmod(v, twiddle, q);
            let new_u = addmod(u, tw_v, q);
            let new_v = submod(u, tw_v, q);

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

        let psi_inv_base2 = idx * 2u;
        let psi_inv_power = vec2<u32>(inv_twiddles[psi_inv_base2], inv_twiddles[psi_inv_base2 + 1u]);
        val = mulmod(val, psi_inv_power, q);
        val = mulmod(val, n_inv, q);

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
        assert_eq!(params.k, 4);
    }
}
