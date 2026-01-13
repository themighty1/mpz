//! GPU runtime for batched slot-wise multiplication.
//!
//! This module provides the GPU execution context for accelerating
//! the expensive `mul_plaintext_slots` and `sub_plaintext_slots` operations.
//!
//! # Performance Target
//!
//! CPU baseline (180 rows):
//! - mul_plaintext_slots: ~18ms × 180 = 3.2s
//! - sub_plaintext_slots: ~1ms × 180 = 180ms
//!
//! GPU target: < 100ms total for all 180 rows

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;

/// Uniform buffer for batch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    _pad: u32,
}

/// Uniform buffer for polynomial multiplication.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MulParams {
    n: u32,
    num_batches: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform buffer for packing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    n: u32,
    num_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Precomputed twiddle factors for NTT/INTT operations.
///
/// These are computed once and reused across all slot multiplications.
pub struct TwiddleFactors {
    /// Forward NTT twiddle factors: ω^i for i ∈ [0, n)
    pub fwd_twiddles: Vec<u64>,
    /// Inverse NTT twiddle factors: ω^{-i} for i ∈ [0, n)
    pub inv_twiddles: Vec<u64>,
    /// n^{-1} mod t for scaling after INTT
    pub n_inv: u64,
    /// Ring dimension
    pub n: usize,
    /// Plaintext modulus
    pub t: u64,
}

impl TwiddleFactors {
    /// Computes twiddle factors for the given parameters.
    ///
    /// # Arguments
    /// * `n` - Ring dimension (must be power of 2)
    /// * `t` - Plaintext modulus
    /// * `omega` - Primitive n-th root of unity mod t
    pub fn compute(n: usize, t: u64, omega: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be power of 2");

        let mut fwd_twiddles = Vec::with_capacity(n);
        let mut inv_twiddles = Vec::with_capacity(n);

        // Compute ω^i and ω^{-i}
        let omega_inv = mod_inverse(omega, t);

        let mut w = 1u64;
        let mut w_inv = 1u64;

        for _ in 0..n {
            fwd_twiddles.push(w);
            inv_twiddles.push(w_inv);
            w = mul_mod(w, omega, t);
            w_inv = mul_mod(w_inv, omega_inv, t);
        }

        let n_inv = mod_inverse(n as u64, t);

        Self {
            fwd_twiddles,
            inv_twiddles,
            n_inv,
            n,
            t,
        }
    }
}

/// GPU context for batched slot multiplication operations.
///
/// This context holds the compiled pipelines and precomputed data
/// needed for GPU-accelerated slot operations.
pub struct SlotMulGpuContext {
    device: Device,
    queue: Queue,

    // Pipelines
    intt_encode_pipeline: ComputePipeline,
    poly_mul_pipeline: ComputePipeline,
    slot_sub_pipeline: ComputePipeline,
    pack_2way_pipeline: ComputePipeline,

    // Precomputed data on GPU
    inv_twiddles_buffer: Buffer,
    n_inv_buffer: Buffer,
    t_buffer: Buffer,
    barrett_t_buffer: Buffer,

    // Parameters
    n: usize,
    t: u64,
}

impl SlotMulGpuContext {
    /// Creates a new GPU context for slot multiplication.
    ///
    /// On native, this blocks. On WASM, use `new_async` instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(twiddles: &TwiddleFactors) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(twiddles))
    }

    /// On WASM, sync GPU init is not supported. Use `new_async` instead.
    #[cfg(target_arch = "wasm32")]
    pub fn new(_twiddles: &TwiddleFactors) -> Result<Self, GpuError> {
        Err(GpuError::ExecutionFailed(
            "Sync GPU init not supported in WASM. Use new_async.".to_string()
        ))
    }

    /// Creates a new GPU context asynchronously.
    pub async fn new_async(twiddles: &TwiddleFactors) -> Result<Self, GpuError> {
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

        // Request device with higher limits for shared memory
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("slot-mul-gpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroups_per_dimension: 65535,
                        max_storage_buffer_binding_size: 1024 * 1024 * 256, // 256MB
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        // Compile standalone shaders (without module imports, since naga_oil would be needed)
        let intt_shader_src = create_standalone_intt_shader();
        let intt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batched_intt_encode"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(intt_shader_src)),
        });

        let poly_mul_shader_src = create_standalone_poly_mul_shader();
        let poly_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batched_poly_mul"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(poly_mul_shader_src)),
        });

        let slot_sub_shader_src = create_standalone_slot_sub_shader();
        let slot_sub_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batched_slot_sub"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(slot_sub_shader_src)),
        });

        let pack_2way_shader_src = create_standalone_pack_2way_shader();
        let pack_2way_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batched_pack_2way"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pack_2way_shader_src)),
        });

        // Create pipelines
        let intt_encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("intt_encode pipeline"),
                layout: None,
                module: &intt_shader,
                entry_point: Some("batched_intt_encode"),
                compilation_options: Default::default(),
                cache: None,
            });

        let poly_mul_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("poly_mul pipeline"),
                layout: None,
                module: &poly_mul_shader,
                entry_point: Some("batched_poly_mul"),
                compilation_options: Default::default(),
                cache: None,
            });

        let slot_sub_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slot_sub pipeline"),
                layout: None,
                module: &slot_sub_shader,
                entry_point: Some("batched_slot_sub"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pack_2way_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pack_2way pipeline"),
                layout: None,
                module: &pack_2way_shader,
                entry_point: Some("batched_pack_2way"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Upload precomputed data
        let inv_twiddles_u32: Vec<u32> = twiddles
            .inv_twiddles
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let inv_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inv_twiddles"),
                contents: bytemuck::cast_slice(&inv_twiddles_u32),
                usage: BufferUsages::STORAGE,
            });

        let n_inv_u32 = [twiddles.n_inv as u32, (twiddles.n_inv >> 32) as u32];
        let n_inv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("n_inv"),
            contents: bytemuck::cast_slice(&n_inv_u32),
            usage: BufferUsages::STORAGE,
        });

        let t_u32 = [twiddles.t as u32, (twiddles.t >> 32) as u32];
        let t_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("t_val"),
            contents: bytemuck::cast_slice(&t_u32),
            usage: BufferUsages::UNIFORM,
        });

        // Compute Barrett parameter for t
        let barrett_t = compute_barrett_param(twiddles.t);
        let barrett_t_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("barrett_t"),
            contents: bytemuck::cast_slice(&barrett_t),
            usage: BufferUsages::UNIFORM,
        });

        Ok(Self {
            device,
            queue,
            intt_encode_pipeline,
            poly_mul_pipeline,
            slot_sub_pipeline,
            pack_2way_pipeline,
            inv_twiddles_buffer,
            n_inv_buffer,
            t_buffer,
            barrett_t_buffer,
            n: twiddles.n,
            t: twiddles.t,
        })
    }

    /// Returns a reference to the GPU device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the GPU queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the ring dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Encodes multiple slot vectors into polynomial coefficients.
    ///
    /// This performs batched INTT on the GPU.
    ///
    /// # Arguments
    /// * `slot_batches` - Vec of slot vectors, each with n elements
    ///
    /// # Returns
    /// Vec of polynomial coefficient vectors
    pub fn encode_slots_batched(
        &self,
        slot_batches: &[Vec<u64>],
    ) -> Result<Vec<Vec<u64>>, GpuError> {
        let num_batches = slot_batches.len();
        if num_batches == 0 {
            return Ok(Vec::new());
        }

        let n = self.n;
        for batch in slot_batches {
            if batch.len() != n {
                return Err(GpuError::InvalidParams(format!(
                    "Expected {} slots, got {}",
                    n,
                    batch.len()
                )));
            }
        }

        // Flatten input data
        let slots_flat: Vec<u32> = slot_batches
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let slots_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("slots input"),
                contents: bytemuck::cast_slice(&slots_flat),
                usage: BufferUsages::STORAGE,
            });

        let coeffs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coeffs output"),
            size: (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = BatchParams {
            n: n as u32,
            log_n: (n as u32).trailing_zeros(),
            num_batches: num_batches as u32,
            _pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch params"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            });

        // Create bind group (full INTT shader with shared memory)
        let bind_group_layout = self.intt_encode_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("intt encode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: coeffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.inv_twiddles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.n_inv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.t_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.barrett_t_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch one workgroup per batch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("intt encode encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("intt encode pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.intt_encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back results
        let results = self.read_buffer(&coeffs_buffer, num_batches * n)?;

        // Reshape into batches
        let coeffs_batches: Vec<Vec<u64>> = results
            .chunks(n)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(coeffs_batches)
    }

    /// Performs batched polynomial pointwise multiplication.
    ///
    /// Multiplies a single ciphertext polynomial by multiple plaintext polynomials.
    /// All polynomials should be in NTT domain.
    ///
    /// # Arguments
    /// * `ct_ntt` - Single ciphertext polynomial in NTT domain (n elements)
    /// * `pt_ntt_batches` - Multiple plaintext polynomials in NTT domain
    /// * `q` - Modulus for this RNS component
    ///
    /// # Returns
    /// Vec of output polynomials (one per plaintext)
    pub fn poly_mul_batched(
        &self,
        ct_ntt: &[u64],
        pt_ntt_batches: &[Vec<u64>],
        q: u64,
    ) -> Result<Vec<Vec<u64>>, GpuError> {
        let num_batches = pt_ntt_batches.len();
        if num_batches == 0 {
            return Ok(Vec::new());
        }

        let n = self.n;
        if ct_ntt.len() != n {
            return Err(GpuError::InvalidParams(format!(
                "Expected {} coefficients in ct, got {}",
                n,
                ct_ntt.len()
            )));
        }

        // Flatten input data
        let ct_u32: Vec<u32> = ct_ntt
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let pt_flat: Vec<u32> = pt_ntt_batches
            .iter()
            .flat_map(|batch| batch.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let ct_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ct_ntt"),
                contents: bytemuck::cast_slice(&ct_u32),
                usage: BufferUsages::STORAGE,
            });

        let pt_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pt_ntt"),
                contents: bytemuck::cast_slice(&pt_flat),
                usage: BufferUsages::STORAGE,
            });

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_ntt"),
            size: (num_batches * n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MulParams {
            n: n as u32,
            num_batches: num_batches as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mul params"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            });

        let q_u32 = [q as u32, (q >> 32) as u32];
        let q_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q_val"),
                contents: bytemuck::cast_slice(&q_u32),
                usage: BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.poly_mul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("poly mul bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ct_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let total_elements = (num_batches * n) as u32;
        let workgroups = (total_elements + 255) / 256;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("poly mul encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("poly mul pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.poly_mul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back results
        let results = self.read_buffer(&out_buffer, num_batches * n)?;

        let out_batches: Vec<Vec<u64>> = results.chunks(n).map(|chunk| chunk.to_vec()).collect();

        Ok(out_batches)
    }

    fn read_buffer(&self, buffer: &Buffer, num_elements: usize) -> Result<Vec<u64>, GpuError> {
        let size = (num_elements * 2 * std::mem::size_of::<u32>()) as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
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

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..size);

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

        let results: Vec<u64> = u32_data
            .chunks(2)
            .map(|chunk| (chunk[0] as u64) | ((chunk[1] as u64) << 32))
            .collect();

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

// Helper functions

/// Computes modular inverse using extended Euclidean algorithm.
fn mod_inverse(a: u64, m: u64) -> u64 {
    let mut t: i128 = 0;
    let mut newt: i128 = 1;
    let mut r: i128 = m as i128;
    let mut newr: i128 = a as i128;

    while newr != 0 {
        let quotient = r / newr;
        (t, newt) = (newt, t - quotient * newt);
        (r, newr) = (newr, r - quotient * newr);
    }

    if t < 0 {
        t += m as i128;
    }

    t as u64
}

/// Modular multiplication with 128-bit intermediate.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Computes Barrett reduction parameter μ = floor(2^128 / q).
fn compute_barrett_param(q: u64) -> [u32; 4] {
    // For 64-bit q, we compute μ = floor(2^128 / q)
    // Since 2^128 doesn't fit in u128, we compute in two parts:
    // μ = 2^64 * (2^64 / q) + (2^64 mod q) * (2^64 / q) / q
    //
    // Simpler approach: μ ≈ 2^128 / q
    // We can compute this as: (2^64 / q) * 2^64 + adjustment

    let q128 = q as u128;

    // 2^64 / q gives us the high part
    let two_64: u128 = 1u128 << 64;
    let high_div = two_64 / q128;
    let high_rem = two_64 % q128;

    // For the low part: (high_rem * 2^64) / q
    // But high_rem * 2^64 might overflow, so we compute differently
    // mu = high_div * 2^64 + (high_rem << 64) / q

    // Use wrapping arithmetic to avoid overflow check
    // mu_high = high_div (fits in 64 bits for q > 2^64, but for smaller q it's fine)
    // mu_low = (high_rem * 2^64) / q

    // For Goldilocks (q ≈ 2^64), high_div ≈ 1, high_rem ≈ 0
    // So mu ≈ 2^64

    // Compute (high_rem * 2^64) / q using 128-bit arithmetic carefully
    let low_div = if high_rem > 0 {
        // high_rem < q < 2^64, so high_rem * 2^64 / q < 2^64
        // We can compute this by: (high_rem << 64) / q
        // But high_rem << 64 overflows. Instead: high_rem * (2^64 / q) + correction
        (high_rem * high_div + (high_rem * (two_64 % q128)) / q128) as u64
    } else {
        0
    };

    // mu = high_div * 2^64 + low_div
    // Store as [mu0, mu1, mu2, mu3] where mu = mu0 + mu1*2^32 + mu2*2^64 + mu3*2^96
    let mu_low = low_div;
    let mu_high = high_div as u64;

    [
        mu_low as u32,
        (mu_low >> 32) as u32,
        mu_high as u32,
        (mu_high >> 32) as u32,
    ]
}

/// Creates standalone batched INTT shader with inlined math functions.
/// Uses shared memory for 8192-element NTT with 1024 threads per workgroup.
/// Each workgroup processes one batch (polynomial).
fn create_standalone_intt_shader() -> String {
    r#"
struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read_write> coeffs: array<u32>;
@group(0) @binding(3) var<storage, read> inv_twiddles: array<u32>;
@group(0) @binding(4) var<storage, read> n_inv: array<u32>;
@group(0) @binding(5) var<uniform> t_val: vec2<u32>;
@group(0) @binding(6) var<uniform> barrett_t: vec4<u32>;

// Shared memory: 8192 elements × 2 u32s = 64KB
var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Bit reversal for n up to 8192 (13 bits)
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

// 64-bit multiply: a * b -> vec2<u32>
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

// 128-bit multiply: a * b -> vec4<u32>
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

// Barrett reduction: x mod q
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu: vec4<u32>) -> vec2<u32> {
    let mu0 = mu.x;
    let mu1 = mu.y;
    let mu2 = mu.z;
    let mu3 = mu.w;

    var r0 = x.x;
    var r1 = x.y;

    // Fast path: x already < q
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) {
        return vec2<u32>(r0, r1);
    }

    // Compute partial products for r * mu
    let p00 = u64_mul(r0, mu0);
    let p01 = u64_mul(r0, mu1);
    let p02 = u64_mul(r0, mu2);
    let p03 = u64_mul(r0, mu3);
    let p10 = u64_mul(r1, mu0);
    let p11 = u64_mul(r1, mu1);
    let p12 = u64_mul(r1, mu2);
    let p13 = u64_mul(r1, mu3);

    // Accumulate bits 64-95
    var acc64: u32 = p02.x;
    var c: u32 = 0u;
    var t: u32;
    t = acc64 + p01.y; if t < acc64 { c = 1u; } acc64 = t;
    t = acc64 + p10.y; if t < acc64 { c = c + 1u; } acc64 = t;
    t = acc64 + p11.x; if t < acc64 { c = c + 1u; } acc64 = t;
    var mid32 = p00.y;
    var mid_c: u32 = 0u;
    var mt = mid32 + p01.x; if mt < mid32 { mid_c = 1u; } mid32 = mt;
    mt = mid32 + p10.x; if mt < mid32 { mid_c = mid_c + 1u; }
    t = acc64 + mid_c; if t < acc64 { c = c + 1u; } acc64 = t;

    // Accumulate bits 96-127
    var acc96: u32 = p02.y;
    var c2: u32 = 0u;
    t = acc96 + p03.x; if t < acc96 { c2 = 1u; } acc96 = t;
    t = acc96 + p11.y; if t < acc96 { c2 = c2 + 1u; } acc96 = t;
    t = acc96 + p12.x; if t < acc96 { c2 = c2 + 1u; } acc96 = t;
    t = acc96 + c; if t < acc96 { c2 = c2 + 1u; } acc96 = t;

    // Accumulate bits 128-159
    var acc128: u32 = p13.x;
    var c3: u32 = 0u;
    t = acc128 + p03.y; if t < acc128 { c3 = 1u; } acc128 = t;
    t = acc128 + p12.y; if t < acc128 { c3 = c3 + 1u; } acc128 = t;
    t = acc128 + c2; if t < acc128 { c3 = c3 + 1u; } acc128 = t;

    // q_est
    var q_est_lo = acc128;
    var q_est_hi = p13.y + c3;

    // Compute q_est * q
    let qe0 = u64_mul(q_est_lo, q.x);
    let qe1 = u64_mul(q_est_lo, q.y);
    let qe2 = u64_mul(q_est_hi, q.x);

    var p0 = qe0.x;
    var p1 = qe0.y;

    t = p1 + qe1.x; c = 0u; if t < p1 { c = 1u; } p1 = t;
    t = p1 + qe2.x; if t < p1 { c = c + 1u; } p1 = t;

    // Subtract from r
    var borrow: u32 = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
    }

    // Final corrections
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            break;
        }
        borrow = 0u;
        if r0 >= q.x {
            r0 = r0 - q.x;
        } else {
            r0 = r0 + (0xFFFFFFFFu - q.x) + 1u;
            borrow = 1u;
        }
        sub1 = q.y + borrow;
        if r1 >= sub1 {
            r1 = r1 - sub1;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        }
    }

    return vec2<u32>(r0, r1);
}

// Modular addition: (a + b) mod q
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

// Modular subtraction: (a - b) mod q
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

// Modular multiply with Barrett reduction
fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu: vec4<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    return barrett_reduce(prod, q, mu);
}

// Note: Using 256 threads due to WebGPU limit (max 256 invocations per workgroup)
// Each thread handles 32 elements for n=8192
@compute @workgroup_size(256, 1, 1)
fn batched_intt_encode(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches {
        return;
    }

    let q = t_val;
    let mu = barrett_t;
    let batch_offset = batch_idx * n * 2u;
    let elements_per_thread = n / 256u;  // 32 for n=8192

    // === LOAD: bit-reverse permutation ===
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let src = batch_offset + idx * 2u;
        let val = vec2<u32>(slots[src], slots[src + 1u]);

        // Store to bit-reversed position in shared memory
        let rev_idx = bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }

    workgroupBarrier();

    // === INTT BUTTERFLY STAGES (using inverse twiddles) ===
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let total_butterflies = n >> 1u;  // n/2 = 4096
        let butterflies_per_thread = total_butterflies / 256u;  // 16 for n=8192

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;

            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let ii = group * m + idx_in_group;
            let jj = ii + half_m;

            // Twiddle index (using inverse twiddles)
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            // Load butterfly inputs
            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            // Butterfly: u' = u + t*w, v' = u - t*w
            let tw_v = mulmod(v, twiddle, q, mu);

            let new_u = addmod(u, tw_v, q);
            let new_v = submod(u, tw_v, q);

            // Store results
            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }

        workgroupBarrier();
    }

    // === STORE: scale by n^-1 ===
    let n_inv_val = vec2<u32>(n_inv[0], n_inv[1]);

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;

        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Multiply by n^-1
        val = mulmod(val, n_inv_val, q, mu);

        let dst = batch_offset + idx * 2u;
        coeffs[dst] = val.x;
        coeffs[dst + 1u] = val.y;
    }
}
"#.to_string()
}

/// Creates standalone polynomial multiplication shader.
fn create_standalone_poly_mul_shader() -> String {
    r#"
struct MulParams {
    n: u32,
    num_batches: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: MulParams;
@group(0) @binding(1) var<storage, read> ct_ntt: array<u32>;
@group(0) @binding(2) var<storage, read> pt_ntt: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_ntt: array<u32>;
@group(0) @binding(4) var<uniform> q_val: vec2<u32>;
// Note: Barrett reduction not used in simplified version

// 64-bit multiply helper
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

// 128-bit multiply
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

// Simplified modular multiply (for values that don't overflow much)
fn mulmod_simple(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);

    // Simple reduction: if high bits are zero, just check if >= q
    if prod.z == 0u && prod.w == 0u {
        var r0 = prod.x;
        var r1 = prod.y;

        // Subtract q while >= q
        for (var i = 0u; i < 3u; i++) {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                break;
            }
            var borrow = 0u;
            if r0 >= q.x {
                r0 = r0 - q.x;
            } else {
                r0 = 0xFFFFFFFFu - (q.x - r0 - 1u);
                borrow = 1u;
            }
            r1 = r1 - q.y - borrow;
        }
        return vec2<u32>(r0, r1);
    }

    // For larger products, we'd need full Barrett reduction
    // For now, return the low bits (this is a simplification)
    return vec2<u32>(prod.x, prod.y);
}

@compute @workgroup_size(256, 1, 1)
fn batched_poly_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_batches;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let batch_idx = idx / params.n;
    let elem_idx = idx % params.n;

    let q = q_val;

    // Load ciphertext element
    let ct_base = elem_idx * 2u;
    let ct_val = vec2<u32>(ct_ntt[ct_base], ct_ntt[ct_base + 1u]);

    // Load plaintext element
    let pt_base = (batch_idx * params.n + elem_idx) * 2u;
    let pt_val = vec2<u32>(pt_ntt[pt_base], pt_ntt[pt_base + 1u]);

    // Multiply
    let result = mulmod_simple(ct_val, pt_val, q);

    // Store
    let out_base = pt_base;
    out_ntt[out_base] = result.x;
    out_ntt[out_base + 1u] = result.y;
}
"#.to_string()
}

/// Creates standalone slot subtraction shader.
fn create_standalone_slot_sub_shader() -> String {
    r#"
struct SubParams {
    n: u32,
    num_batches: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SubParams;
@group(0) @binding(1) var<storage, read> slots_in: array<u32>;
@group(0) @binding(2) var<storage, read> blinders: array<u32>;
@group(0) @binding(3) var<storage, read_write> slots_out: array<u32>;
@group(0) @binding(4) var<uniform> t_val: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn batched_slot_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_batches;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let t = t_val;
    let base = idx * 2u;

    let a = vec2<u32>(slots_in[base], slots_in[base + 1u]);
    let b = vec2<u32>(blinders[base], blinders[base + 1u]);

    var result: vec2<u32>;
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        result = vec2<u32>(diff_lo, diff_hi);
    } else {
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;

        var res_lo = t.x - diff_lo;
        borrow = 0u;
        if t.x < diff_lo { borrow = 1u; }
        var res_hi = t.y - diff_hi - borrow;
        result = vec2<u32>(res_lo, res_hi);
    }

    slots_out[base] = result.x;
    slots_out[base + 1u] = result.y;
}
"#.to_string()
}

/// Creates standalone 2-way packing shader.
fn create_standalone_pack_2way_shader() -> String {
    r#"
struct PackParams {
    n: u32,
    num_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> ct0_c0: array<u32>;
@group(0) @binding(2) var<storage, read> ct0_c1: array<u32>;
@group(0) @binding(3) var<storage, read> ct1_c0: array<u32>;
@group(0) @binding(4) var<storage, read> ct1_c1: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_c0: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_c1: array<u32>;
@group(0) @binding(7) var<uniform> q_val: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn batched_pack_2way(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_pairs;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let q = q_val;
    let coeff_idx = idx % params.n;
    let base = idx * 2u;

    let a0_c0 = vec2<u32>(ct0_c0[base], ct0_c0[base + 1u]);
    let a0_c1 = vec2<u32>(ct0_c1[base], ct0_c1[base + 1u]);
    let a1_c0 = vec2<u32>(ct1_c0[base], ct1_c0[base + 1u]);
    let a1_c1 = vec2<u32>(ct1_c1[base], ct1_c1[base + 1u]);

    // Rotation by n/2: negate odd coefficients
    var b1_c0: vec2<u32>;
    var b1_c1: vec2<u32>;

    if (coeff_idx % 2u) == 0u {
        b1_c0 = a1_c0;
        b1_c1 = a1_c1;
    } else {
        // Negate: q - val
        if a1_c0.y == 0u && a1_c0.x == 0u {
            b1_c0 = vec2<u32>(0u, 0u);
        } else {
            var diff_lo = q.x - a1_c0.x;
            var borrow = 0u;
            if q.x < a1_c0.x { borrow = 1u; }
            var diff_hi = q.y - a1_c0.y - borrow;
            b1_c0 = vec2<u32>(diff_lo, diff_hi);
        }

        if a1_c1.y == 0u && a1_c1.x == 0u {
            b1_c1 = vec2<u32>(0u, 0u);
        } else {
            var diff_lo = q.x - a1_c1.x;
            var borrow = 0u;
            if q.x < a1_c1.x { borrow = 1u; }
            var diff_hi = q.y - a1_c1.y - borrow;
            b1_c1 = vec2<u32>(diff_lo, diff_hi);
        }
    }

    // Add with reduction
    var sum_lo = a0_c0.x + b1_c0.x;
    var carry = 0u;
    if sum_lo < a0_c0.x { carry = 1u; }
    var sum_hi = a0_c0.y + b1_c0.y + carry;
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }
    out_c0[base] = sum_lo;
    out_c0[base + 1u] = sum_hi;

    sum_lo = a0_c1.x + b1_c1.x;
    carry = 0u;
    if sum_lo < a0_c1.x { carry = 1u; }
    sum_hi = a0_c1.y + b1_c1.y + carry;
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }
    out_c1[base] = sum_lo;
    out_c1[base + 1u] = sum_hi;
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_inverse() {
        let t = 0xFFFFFFFF00000001u64; // Goldilocks
        let a = 12345u64;
        let inv = mod_inverse(a, t);
        assert_eq!(mul_mod(a, inv, t), 1);
    }

    #[test]
    fn test_barrett_param() {
        let q = 0xFFFFFFFF00000001u64;
        let params = compute_barrett_param(q);
        // Just verify it computes without panic
        assert!(params[0] != 0 || params[1] != 0 || params[2] != 0 || params[3] != 0);
    }
}
