//! GPU-accelerated NTT for Goldilocks field (p = 2^64 - 2^32 + 1).
//!
//! Provides batched forward and inverse NTT operations for polynomial multiplication
//! in the JustVengers ZK protocol.

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;

/// Goldilocks prime: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS: u64 = 0xFFFFFFFF00000001;

/// GPU parameters for Goldilocks NTT.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    _pad: u32,
}

/// Modulus parameters for GPU shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu0: u32,
    mu1: u32,
    mu2: u32,
    mu3: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
}

/// GPU context for Goldilocks NTT operations.
pub struct GoldilocksNttGpu {
    device: Device,
    queue: Queue,
    n: usize,
    log_n: u32,

    // Pipelines
    forward_ntt_pipeline: ComputePipeline,
    inverse_ntt_pipeline: ComputePipeline,

    // Precomputed buffers
    modulus_params_buffer: Buffer,
    forward_twiddles_buffer: Buffer,  // omega powers for forward NTT
    inverse_twiddles_buffer: Buffer,  // omega_inv powers for inverse NTT
}

/// Computes modular inverse using extended Euclidean algorithm.
fn mod_inverse(a: u64, m: u64) -> u64 {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = m as i128;
    let mut new_r: i128 = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if t < 0 {
        t += m as i128;
    }
    t as u64
}

/// Computes a * b mod m using 128-bit arithmetic.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Computes base^exp mod m.
fn mod_pow(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128;
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    result as u64
}

/// Finds a primitive n-th root of unity in Goldilocks field.
/// Returns omega such that omega^n = 1 and omega^(n/2) = -1.
fn find_primitive_nth_root(n: usize) -> Option<u64> {
    let log_n = n.trailing_zeros();
    if log_n > 32 {
        return None; // Goldilocks supports up to 2^32
    }

    // 7 is a primitive root mod Goldilocks
    // omega = 7^((p-1) / n)
    let p = GOLDILOCKS;
    let exp = (p - 1) / (n as u64);
    let omega = mod_pow(7, exp, p);

    // Verify: omega^n = 1
    let check = mod_pow(omega, n as u64, p);
    if check != 1 {
        return None;
    }

    // Verify: omega^(n/2) = p - 1 (i.e., -1 mod p)
    if n > 1 {
        let half_check = mod_pow(omega, (n / 2) as u64, p);
        if half_check != p - 1 {
            return None;
        }
    }

    Some(omega)
}

/// Computes powers: [1, base, base^2, ..., base^(n-1)] mod p
fn compute_powers(base: u64, n: usize, p: u64) -> Vec<u64> {
    let mut powers = Vec::with_capacity(n);
    let mut current = 1u64;
    for _ in 0..n {
        powers.push(current);
        current = mod_mul(current, base, p);
    }
    powers
}

impl GoldilocksNttGpu {
    /// Creates a new GPU context for Goldilocks NTT.
    ///
    /// # Arguments
    /// * `n` - NTT size (must be power of 2, max 8192 due to shared memory)
    pub fn new(n: usize) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(n))
    }

    /// Creates a new GPU context asynchronously (required for WASM).
    pub async fn new_async(n: usize) -> Result<Self, GpuError> {
        // Validate n
        if !n.is_power_of_two() {
            return Err(GpuError::InvalidParams(format!("n={} must be power of 2", n)));
        }
        if n > 8192 {
            return Err(GpuError::InvalidParams(format!(
                "n={} exceeds max 8192 (shared memory limit)", n
            )));
        }

        let log_n = n.trailing_zeros();

        // Initialize GPU
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
                    label: Some("goldilocks-ntt device"),
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

        // Compile shaders
        let forward_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_FORWARD_NTT_SHADER,
            "goldilocks_forward_ntt.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let inverse_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_INVERSE_NTT_SHADER,
            "goldilocks_inverse_ntt.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        // Create pipelines
        let forward_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("goldilocks_forward_ntt pipeline"),
                layout: None,
                module: &forward_ntt_shader,
                entry_point: Some("forward_ntt_goldilocks"),
                compilation_options: Default::default(),
                cache: None,
            });

        let inverse_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("goldilocks_inverse_ntt pipeline"),
                layout: None,
                module: &inverse_ntt_shader,
                entry_point: Some("inverse_ntt_goldilocks"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Compute Goldilocks parameters
        let p = GOLDILOCKS;
        let mu: u128 = ((1u128 << 127) / (p as u128)) * 2;
        let n_inv = mod_inverse(n as u64, p);

        let modulus_params = GpuModulusParams {
            modulus_lo: p as u32,
            modulus_hi: (p >> 32) as u32,
            mu0: mu as u32,
            mu1: (mu >> 32) as u32,
            mu2: (mu >> 64) as u32,
            mu3: (mu >> 96) as u32,
            n_inv_lo: n_inv as u32,
            n_inv_hi: (n_inv >> 32) as u32,
        };

        let modulus_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goldilocks_modulus_params"),
                contents: bytemuck::bytes_of(&modulus_params),
                usage: BufferUsages::UNIFORM,
            });

        // Compute twiddle factors - omega is primitive n-th root of unity
        let omega = find_primitive_nth_root(n)
            .ok_or_else(|| GpuError::InvalidParams(format!("No primitive root for n={}", n)))?;
        let omega_inv = mod_inverse(omega, p);

        // Forward twiddles: [omega^0, omega^1, ..., omega^(n-1)]
        let omega_powers = compute_powers(omega, n, p);

        let forward_twiddles: Vec<u32> = omega_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let forward_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goldilocks_forward_twiddles"),
                contents: bytemuck::cast_slice(&forward_twiddles),
                usage: BufferUsages::STORAGE,
            });

        // Inverse twiddles: [omega_inv^0, omega_inv^1, ..., omega_inv^(n-1)]
        let omega_inv_powers = compute_powers(omega_inv, n, p);

        let inverse_twiddles: Vec<u32> = omega_inv_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let inverse_twiddles_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("goldilocks_inverse_twiddles"),
                contents: bytemuck::cast_slice(&inverse_twiddles),
                usage: BufferUsages::STORAGE,
            });

        Ok(Self {
            device,
            queue,
            n,
            log_n,
            forward_ntt_pipeline,
            inverse_ntt_pipeline,
            modulus_params_buffer,
            forward_twiddles_buffer,
            inverse_twiddles_buffer,
        })
    }

    /// Performs batched forward NTT on multiple polynomials.
    ///
    /// # Arguments
    /// * `polys` - Slice of polynomials, each with n coefficients
    ///
    /// # Returns
    /// Vector of NTT evaluations, each with n values
    pub fn batched_forward_ntt(&self, polys: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        if polys.is_empty() {
            return Ok(Vec::new());
        }

        let num_batches = polys.len();

        // Validate inputs
        for (i, poly) in polys.iter().enumerate() {
            if poly.len() != self.n {
                return Err(GpuError::InvalidParams(format!(
                    "Poly {} has {} coeffs, expected {}", i, poly.len(), self.n
                )));
            }
        }

        // Upload input
        let input_flat: Vec<u32> = polys
            .iter()
            .flat_map(|p| p.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ntt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: BufferUsages::STORAGE,
        });

        // Allocate output
        let output_size = num_batches * self.n * 2 * std::mem::size_of::<u32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt_output"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Batch params
        let batch_params = GpuBatchParams {
            n: self.n as u32,
            log_n: self.log_n,
            num_batches: num_batches as u32,
            _pad: 0,
        };
        let batch_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_params"),
            contents: bytemuck::bytes_of(&batch_params),
            usage: BufferUsages::UNIFORM,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_ntt encoder"),
        });

        {
            let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward_ntt bind group"),
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
                        resource: self.forward_twiddles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.modulus_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {}", e)))?
            .map_err(GpuError::BufferMapping)?;

        let data = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&data);

        // Convert back to Vec<Vec<u64>>
        let mut results = Vec::with_capacity(num_batches);
        for batch in 0..num_batches {
            let mut vals = Vec::with_capacity(self.n);
            for i in 0..self.n {
                let idx = (batch * self.n + i) * 2;
                let lo = output_u32[idx] as u64;
                let hi = output_u32[idx + 1] as u64;
                vals.push(lo | (hi << 32));
            }
            results.push(vals);
        }

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Performs batched inverse NTT on multiple evaluation vectors.
    ///
    /// # Arguments
    /// * `evals` - Slice of evaluation vectors, each with n values
    ///
    /// # Returns
    /// Vector of polynomials (coefficient form), each with n coefficients
    pub fn batched_inverse_ntt(&self, evals: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        if evals.is_empty() {
            return Ok(Vec::new());
        }

        let num_batches = evals.len();

        // Validate inputs
        for (i, eval) in evals.iter().enumerate() {
            if eval.len() != self.n {
                return Err(GpuError::InvalidParams(format!(
                    "Eval {} has {} values, expected {}", i, eval.len(), self.n
                )));
            }
        }

        // Upload input
        let input_flat: Vec<u32> = evals
            .iter()
            .flat_map(|e| e.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("intt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: BufferUsages::STORAGE,
        });

        // Allocate output
        let output_size = num_batches * self.n * 2 * std::mem::size_of::<u32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("intt_output"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Batch params
        let batch_params = GpuBatchParams {
            n: self.n as u32,
            log_n: self.log_n,
            num_batches: num_batches as u32,
            _pad: 0,
        };
        let batch_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_params"),
            contents: bytemuck::bytes_of(&batch_params),
            usage: BufferUsages::UNIFORM,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inverse_ntt encoder"),
        });

        {
            let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inverse_ntt bind group"),
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
                        resource: self.inverse_twiddles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.modulus_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inverse_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(format!("Channel recv failed: {}", e)))?
            .map_err(GpuError::BufferMapping)?;

        let data = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&data);

        // Convert back to Vec<Vec<u64>>
        let mut results = Vec::with_capacity(num_batches);
        for batch in 0..num_batches {
            let mut coeffs = Vec::with_capacity(self.n);
            for i in 0..self.n {
                let idx = (batch * self.n + i) * 2;
                let lo = output_u32[idx] as u64;
                let hi = output_u32[idx + 1] as u64;
                coeffs.push(lo | (hi << 32));
            }
            results.push(coeffs);
        }

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

// =============================================================================
// Shaders
// =============================================================================

/// Forward NTT shader for Goldilocks field (standard NTT, no twist).
/// Uses Cooley-Tukey DIT with Barrett reduction for 64-bit modular multiplication.
const GOLDILOCKS_FORWARD_NTT_SHADER: &str = r#"
#import math

struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu0: u32,
    mu1: u32,
    mu2: u32,
    mu3: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> coeffs: array<u32>;
@group(0) @binding(2) var<storage, read_write> ntt_out: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Barrett modular multiplication for Goldilocks (64-bit modulus)
fn mulmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let prod = math::mul64(a, b);
    return math::barrett_reduce_64bit(prod, q, mod_params.mu0, mod_params.mu1, mod_params.mu2, mod_params.mu3);
}

fn addmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    return math::addmod(a, b, q);
}

fn submod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    return math::submod(a, b, q);
}

@compute @workgroup_size(256, 1, 1)
fn forward_ntt_goldilocks(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches { return; }

    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load and bit-reverse (no twist for standard NTT)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let coeff_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(coeffs[coeff_base], coeffs[coeff_base + 1u]);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // Forward NTT butterfly stages (Cooley-Tukey DIT)
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

            // twiddle = omega^(idx_in_group * n / m)
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = mulmod_goldilocks(v, twiddle);
            let new_u = addmod_goldilocks(u, tw_v);
            let new_v = submod_goldilocks(u, tw_v);

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
        let out_base = (batch_offset + idx) * 2u;
        ntt_out[out_base] = val.x;
        ntt_out[out_base + 1u] = val.y;
    }
}
"#;

/// Inverse NTT shader for Goldilocks field (standard NTT with inverse twiddles).
/// Uses Cooley-Tukey DIT with omega_inv, then scales by 1/n.
const GOLDILOCKS_INVERSE_NTT_SHADER: &str = r#"
#import math

struct BatchParams {
    n: u32,
    log_n: u32,
    num_batches: u32,
    _pad: u32,
}

struct ModulusParams {
    modulus_lo: u32,
    modulus_hi: u32,
    mu0: u32,
    mu1: u32,
    mu2: u32,
    mu3: u32,
    n_inv_lo: u32,
    n_inv_hi: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> evals: array<u32>;
@group(0) @binding(2) var<storage, read_write> coeffs_out: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;
@group(0) @binding(4) var<uniform> mod_params: ModulusParams;

var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

// Barrett modular multiplication for Goldilocks (64-bit modulus)
fn mulmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    let prod = math::mul64(a, b);
    return math::barrett_reduce_64bit(prod, q, mod_params.mu0, mod_params.mu1, mod_params.mu2, mod_params.mu3);
}

fn addmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    return math::addmod(a, b, q);
}

fn submod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = vec2<u32>(mod_params.modulus_lo, mod_params.modulus_hi);
    return math::submod(a, b, q);
}

@compute @workgroup_size(256, 1, 1)
fn inverse_ntt_goldilocks(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches { return; }

    let batch_offset = batch_idx * n;
    let elements_per_thread = n / 256u;

    // Load with bit-reverse (same as forward NTT)
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let eval_base = (batch_offset + idx) * 2u;
        let val = vec2<u32>(evals[eval_base], evals[eval_base + 1u]);

        let rev_idx = math::bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }
    workgroupBarrier();

    // Inverse NTT butterfly stages (Cooley-Tukey DIT with omega_inv)
    // Same structure as forward, but using inverse twiddles
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

            // twiddle = omega_inv^(idx_in_group * n / m)
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
            let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

            let tw_v = mulmod_goldilocks(v, twiddle);
            let new_u = addmod_goldilocks(u, tw_v);
            let new_v = submod_goldilocks(u, tw_v);

            shared_lo[ii] = new_u.x;
            shared_hi[ii] = new_u.y;
            shared_lo[jj] = new_v.x;
            shared_hi[jj] = new_v.y;
        }
        workgroupBarrier();
    }

    // Scale by 1/n and store (no untwist for standard NTT)
    let n_inv = vec2<u32>(mod_params.n_inv_lo, mod_params.n_inv_hi);
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Scale by 1/n
        val = mulmod_goldilocks(val, n_inv);

        let out_base = (batch_offset + idx) * 2u;
        coeffs_out[out_base] = val.x;
        coeffs_out[out_base + 1u] = val.y;
    }
}
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_inverse() {
        let p = GOLDILOCKS;
        let a = 12345u64;
        let inv = mod_inverse(a, p);
        let product = mod_mul(a, inv, p);
        assert_eq!(product, 1);
    }

    #[test]
    fn test_find_primitive_nth_root() {
        for log_n in 1..=16 {
            let n = 1usize << log_n;
            let omega = find_primitive_nth_root(n).expect(&format!("No root for n={}", n));
            let p = GOLDILOCKS;

            // omega^n = 1
            assert_eq!(mod_pow(omega, n as u64, p), 1);

            // omega^(n/2) = -1
            assert_eq!(mod_pow(omega, (n / 2) as u64, p), p - 1);
        }
    }

    #[test]
    fn test_gpu_ntt_roundtrip() {
        let n = 1024;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // Create test polynomial
        let poly: Vec<u64> = (0..n as u64).map(|i| i * 12345 % GOLDILOCKS).collect();

        // Forward NTT
        let evals = gpu.batched_forward_ntt(&[poly.clone()])
            .expect("Forward NTT failed");

        // Inverse NTT
        let recovered = gpu.batched_inverse_ntt(&evals)
            .expect("Inverse NTT failed");

        // Compare
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].len(), n);
        for i in 0..n {
            assert_eq!(
                recovered[0][i], poly[i],
                "Mismatch at index {}: got {}, expected {}",
                i, recovered[0][i], poly[i]
            );
        }
    }

    #[test]
    fn test_gpu_ntt_vs_cpu() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 1024;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // Create test polynomial
        let poly: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 3) % GOLDILOCKS).collect();

        // GPU forward NTT
        let gpu_evals = gpu.batched_forward_ntt(&[poly.clone()])
            .expect("GPU NTT failed");

        // CPU forward NTT
        let mut cpu_vals: Vec<Goldilocks> = poly.iter()
            .map(|&x| Goldilocks::new(x))
            .collect();
        Goldilocks::ntt(&mut cpu_vals);
        let cpu_evals: Vec<u64> = cpu_vals.iter().map(|x| x.inner()).collect();

        // Compare
        for i in 0..n {
            assert_eq!(
                gpu_evals[0][i], cpu_evals[i],
                "NTT mismatch at index {}: GPU={}, CPU={}",
                i, gpu_evals[0][i], cpu_evals[i]
            );
        }
    }

    #[test]
    fn test_batched_ntt() {
        let n = 512;
        let num_batches = 10;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // Create multiple test polynomials
        let polys: Vec<Vec<u64>> = (0..num_batches)
            .map(|batch| {
                (0..n as u64)
                    .map(|i| (i * (batch as u64 + 1) * 123) % GOLDILOCKS)
                    .collect()
            })
            .collect();

        // Forward + Inverse should recover original
        let evals = gpu.batched_forward_ntt(&polys).expect("Forward failed");
        let recovered = gpu.batched_inverse_ntt(&evals).expect("Inverse failed");

        for batch in 0..num_batches {
            for i in 0..n {
                assert_eq!(
                    recovered[batch][i], polys[batch][i],
                    "Batch {} index {} mismatch",
                    batch, i
                );
            }
        }
    }
}

#[cfg(test)]
mod barrett_tests {
    use super::*;
    use wgpu::util::DeviceExt;

    const U64_MUL_TEST_SHADER: &str = r#"
#import math

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn test_u64_mul() {
    let a = input[0u];
    let b = input[1u];
    let result = math::u64_mul(a, b);
    output[0u] = result.x;
    output[1u] = result.y;
}
"#;

    #[test]
    fn test_gpu_u64_mul() {
        let test_cases: Vec<(u32, u32)> = vec![
            (2, 3),
            (0xFFFFFFFF, 2),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
        ];

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let adapter = match adapter { Some(a) => a, None => { println!("No GPU"); return; } };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

        let shader = crate::shader_math::create_shader_module(&device, U64_MUL_TEST_SHADER, "u64_mul_test.wgsl").unwrap();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &shader, entry_point: Some("test_u64_mul"),
            compilation_options: Default::default(), cache: None,
        });

        for (a, b) in test_cases {
            let expected = (a as u64) * (b as u64);
            let input_data: [u32; 2] = [a, b];
            let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&input_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 8, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 8, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default()); pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]); pass.dispatch_workgroups(1,1,1); }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, 8);
            queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let out: &[u32] = bytemuck::cast_slice(&data);
            let gpu_result = (out[0] as u64) | ((out[1] as u64) << 32);
            drop(data); staging.unmap();

            println!("{:#x} * {:#x} = {:#x}, gpu = {:#x}", a, b, expected, gpu_result);
            assert_eq!(gpu_result, expected, "u64_mul mismatch");
            println!("  PASS");
        }
    }

    const BARRETT_TEST_SHADER: &str = r#"
#import math

struct ModParams {
    q_lo: u32, q_hi: u32,
    mu0: u32, mu1: u32, mu2: u32, mu3: u32,
    _pad0: u32, _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ModParams;
@group(0) @binding(1) var<storage, read> input_a: array<u32>;
@group(0) @binding(2) var<storage, read> input_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn test_mulmod() {
    let a = vec2<u32>(input_a[0u], input_a[1u]);
    let b = vec2<u32>(input_b[0u], input_b[1u]);
    let q = vec2<u32>(params.q_lo, params.q_hi);
    let x = math::mul64(a, b);
    let result = math::barrett_reduce_64bit(x, q, params.mu0, params.mu1, params.mu2, params.mu3);

    output[0u] = result.x;
    output[1u] = result.y;
    output[2u] = x.x;
    output[3u] = x.y;
    output[4u] = x.z;
    output[5u] = x.w;
}
"#;

    #[test]
    fn test_gpu_barrett_reduction() {
        let p = GOLDILOCKS;
        let mu: u128 = ((1u128 << 127) / (p as u128)) * 2;

        let test_cases: Vec<(u64, u64)> = vec![
            (2, 3),
            (p - 1, 2),
            (p - 1, p - 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (1 << 32, 1 << 32),
            (p / 2, p / 2),
        ];

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let adapter = match adapter {
            Some(a) => a,
            None => { println!("No GPU, skipping"); return; }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

        let shader = crate::shader_math::create_shader_module(&device, BARRETT_TEST_SHADER, "barrett_test.wgsl").unwrap();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("barrett_test"), layout: None, module: &shader,
            entry_point: Some("test_mulmod"), compilation_options: Default::default(), cache: None,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ModParams { q_lo: u32, q_hi: u32, mu0: u32, mu1: u32, mu2: u32, mu3: u32, _pad0: u32, _pad1: u32 }

        let params = ModParams {
            q_lo: p as u32, q_hi: (p >> 32) as u32,
            mu0: mu as u32, mu1: (mu >> 32) as u32, mu2: (mu >> 64) as u32, mu3: (mu >> 96) as u32,
            _pad0: 0, _pad1: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });

        for (a, b) in &test_cases {
            let (a, b) = (*a, *b);
            let expected = ((a as u128 * b as u128) % p as u128) as u64;

            let a_data: [u32; 2] = [a as u32, (a >> 32) as u32];
            let b_data: [u32; 2] = [b as u32, (b >> 32) as u32];
            let input_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&a_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let input_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&b_data), usage: wgpu::BufferUsages::STORAGE,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 24, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 24, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: input_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: input_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default()); pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]); pass.dispatch_workgroups(1, 1, 1); }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 24);
            queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let d: &[u32] = bytemuck::cast_slice(&data);
            let gpu_result = (d[0] as u64) | ((d[1] as u64) << 32);
            let gpu_prod = (d[2] as u128) | ((d[3] as u128) << 32) | ((d[4] as u128) << 64) | ((d[5] as u128) << 96);
            drop(data);
            staging.unmap();

            let expected_prod = (a as u128) * (b as u128);
            println!("{} * {} mod p", a, b);
            println!("  prod:   expected {:#x}, gpu {:#x} {}", expected_prod, gpu_prod, if expected_prod == gpu_prod { "OK" } else { "WRONG" });
            println!("  result: expected {}, gpu {}", expected, gpu_result);
            if gpu_result != expected {
                println!("  ERROR: diff = {}", (expected as i128) - (gpu_result as i128));
            }
            assert_eq!(gpu_result, expected, "GPU Barrett mismatch");
            println!("  PASS\n");
        }
    }

    /// CPU Barrett reduction using the same algorithm as the shader
    fn barrett_reduce_cpu(x: u128, q: u64, mu: u128) -> u64 {
        // x * mu could be up to 256 bits
        let x_lo = x & 0xFFFFFFFFFFFFFFFF;
        let x_hi = x >> 64;
        let mu_lo = mu & 0xFFFFFFFFFFFFFFFF;
        let mu_hi = mu >> 64;

        // x * mu = x_lo*mu_lo + (x_lo*mu_hi + x_hi*mu_lo)<<64 + x_hi*mu_hi<<128
        let p0 = x_lo * mu_lo;  // bits 0-127
        let p1 = x_lo * mu_hi;  // bits 64-191
        let p2 = x_hi * mu_lo;  // bits 64-191
        let p3 = x_hi * mu_hi;  // bits 128-255

        // Accumulate bits 128+ to get q_est
        let col2 = (p0 >> 64) + (p1 & 0xFFFFFFFFFFFFFFFF) + (p2 & 0xFFFFFFFFFFFFFFFF);
        let col3 = (p1 >> 64) + (p2 >> 64) + p3 + (col2 >> 64);

        // q_est is the lower 64 bits of col3 (this matches what shader does)
        let q_est = (col3 & 0xFFFFFFFFFFFFFFFF) as u64;

        // r = x - q_est * q
        let mut r = x.wrapping_sub((q_est as u128) * (q as u128));

        // Correction loop
        while r >= q as u128 {
            r -= q as u128;
        }

        r as u64
    }

    #[test]
    fn test_mu_calculation() {
        let p = GOLDILOCKS;

        // How goldilocks_ntt.rs computes mu (line 231)
        let mu_current: u128 = ((1u128 << 127) / (p as u128)) * 2;

        // Correct mu: floor(2^128 / p)
        let mu_via_max = u128::MAX / (p as u128);
        let mu_correct = mu_via_max + 1;

        println!("p = {:#x}", p);
        println!("mu_current  = {:#x}", mu_current);
        println!("mu_correct  = {:#x}", mu_correct);
        println!("difference  = {}", mu_correct as i128 - mu_current as i128);

        println!("\nmu_current as 4 x u32:");
        println!("  mu0 = {:#x}", (mu_current as u32));
        println!("  mu1 = {:#x}", ((mu_current >> 32) as u32));
        println!("  mu2 = {:#x}", ((mu_current >> 64) as u32));
        println!("  mu3 = {:#x}", ((mu_current >> 96) as u32));
    }

    /// Mimics the shader's u64_mul: two u32 -> u64 as (lo, hi)
    fn u64_mul(a: u32, b: u32) -> (u32, u32) {
        let a_lo = a & 0xFFFF;
        let a_hi = a >> 16;
        let b_lo = b & 0xFFFF;
        let b_hi = b >> 16;

        let p0 = a_lo * b_lo;
        let p1 = a_lo * b_hi;
        let p2 = a_hi * b_lo;
        let p3 = a_hi * b_hi;

        let mut lo = p0;
        let mut hi = p3;

        let mid = p1.wrapping_add(p2);  // May overflow - WGSL wraps
        let mid_lo = (mid & 0xFFFF) << 16;
        let mid_hi = mid >> 16;

        let new_lo = lo.wrapping_add(mid_lo);
        if new_lo < lo { hi = hi.wrapping_add(1); }
        lo = new_lo;
        hi = hi.wrapping_add(mid_hi);

        // Check for carry from p1 + p2 overflow
        if p1 > 0xFFFFFFFFu32.wrapping_sub(p2) {
            hi = hi.wrapping_add(0x10000);
        }

        (lo, hi)
    }

    /// Mimics shader's mul64: two 64-bit -> 128-bit as (r0, r1, r2, r3)
    fn mul64(a: (u32, u32), b: (u32, u32)) -> (u32, u32, u32, u32) {
        let p00 = u64_mul(a.0, b.0);
        let p01 = u64_mul(a.0, b.1);
        let p10 = u64_mul(a.1, b.0);
        let p11 = u64_mul(a.1, b.1);

        let mut r0 = p00.0;
        let mut r1 = p00.1;
        let mut r2 = p11.0;
        let mut r3 = p11.1;

        let mut t = r1.wrapping_add(p01.0);
        let mut c: u32 = if t < r1 { 1 } else { 0 };
        r1 = t;
        t = r1.wrapping_add(p10.0);
        if t < r1 { c = c.wrapping_add(1); }
        r1 = t;

        t = r2.wrapping_add(p01.1);
        let mut c2: u32 = if t < r2 { 1 } else { 0 };
        r2 = t;
        t = r2.wrapping_add(p10.1);
        if t < r2 { c2 = c2.wrapping_add(1); }
        r2 = t;
        t = r2.wrapping_add(c);
        if t < r2 { c2 = c2.wrapping_add(1); }
        r2 = t;

        r3 = r3.wrapping_add(c2);

        (r0, r1, r2, r3)
    }

    #[test]
    fn test_mul64() {
        let test_cases: Vec<(u64, u64)> = vec![
            (2, 3),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFFFFFFFFFF, 2),
            (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
            (0xDEADBEEF12345678, 0xCAFEBABE87654321),
        ];

        for (a, b) in test_cases {
            let expected = (a as u128) * (b as u128);
            let a_split = (a as u32, (a >> 32) as u32);
            let b_split = (b as u32, (b >> 32) as u32);
            let (r0, r1, r2, r3) = mul64(a_split, b_split);
            let result = (r0 as u128) | ((r1 as u128) << 32) | ((r2 as u128) << 64) | ((r3 as u128) << 96);

            println!("{:#x} * {:#x} = {:#x}", a, b, expected);
            println!("  shader: r0={:#x}, r1={:#x}, r2={:#x}, r3={:#x}", r0, r1, r2, r3);
            println!("  result: {:#x}", result);

            assert_eq!(result, expected, "mul64 mismatch");
            println!("  PASS\n");
        }
    }

    #[test]
    fn test_u64_mul() {
        let test_cases: Vec<(u32, u32)> = vec![
            (2, 3),
            (0xFFFFFFFF, 2),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
        ];

        for (a, b) in test_cases {
            let expected = (a as u64) * (b as u64);
            let (lo, hi) = u64_mul(a, b);
            let result = (lo as u64) | ((hi as u64) << 32);

            println!("{}u32 * {}u32 = {}", a, b, expected);
            println!("  shader: lo={:#x}, hi={:#x} -> {}", lo, hi, result);

            assert_eq!(result, expected, "u64_mul mismatch for {} * {}", a, b);
            println!("  PASS\n");
        }
    }

    #[test]
    fn test_barrett_vs_cpu_modmul() {
        let p = GOLDILOCKS;
        let mu: u128 = ((1u128 << 127) / (p as u128)) * 2;

        let test_cases: Vec<(u64, u64)> = vec![
            (2, 3),
            (p - 1, 2),
            (p - 1, p - 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (1 << 32, 1 << 32),
            (p / 2, p / 2),
            (12345678901234567, 98765432109876543 % p),
            (0xDEADBEEF12345678 % p, 0xCAFEBABE87654321 % p),
        ];

        println!("Testing Barrett reduction vs CPU modmul\n");

        for (a, b) in test_cases {
            let prod = (a as u128) * (b as u128);
            let expected = (prod % (p as u128)) as u64;
            let barrett_result = barrett_reduce_cpu(prod, p, mu);

            println!("a={}, b={}", a, b);
            println!("  product = {:#x}", prod);
            println!("  expected = {}", expected);
            println!("  barrett  = {}", barrett_result);

            assert_eq!(barrett_result, expected,
                "Barrett mismatch for a={}, b={}: got {}, expected {}",
                a, b, barrett_result, expected);
            println!("  PASS\n");
        }
    }
}
