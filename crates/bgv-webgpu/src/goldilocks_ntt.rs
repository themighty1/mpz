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

/// Parameters for twiddle factor multiplication (four-step FFT).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTwiddleParams {
    n: u32,            // Total NTT size (N = N1 * N2)
    n1: u32,           // Row size (number of columns)
    num_elements: u32, // Total elements to process
    _pad: u32,
}

/// Parameters for batched small NTT (Pass 2 of four-step FFT).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBatchedNttParams {
    n: u32,            // NTT size (power of 2, 2 to 2048)
    log_n: u32,        // log2(n)
    batch_per_wg: u32, // How many NTTs per workgroup
    total_batches: u32,// Total number of NTTs to process
    stride: u32,       // Stride between consecutive elements in global memory
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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
    twiddle_pipeline: ComputePipeline,
    batched_ntt_pipeline: ComputePipeline,

    // Precomputed buffers
    modulus_params_buffer: Buffer,
    forward_twiddles_buffer: Buffer,  // omega powers for forward NTT
    inverse_twiddles_buffer: Buffer,  // omega_inv powers for inverse NTT
}

impl std::fmt::Debug for GoldilocksNttGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoldilocksNttGpu")
            .field("n", &self.n)
            .field("log_n", &self.log_n)
            .finish_non_exhaustive()
    }
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
    /// Returns the NTT size this context was created with.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Creates a new GPU context for Goldilocks NTT.
    ///
    /// # Arguments
    /// * `n` - NTT size (must be power of 2, max 4096 due to shared memory)
    pub fn new(n: usize) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(n))
    }

    /// Creates a new GPU context asynchronously (required for WASM).
    pub async fn new_async(n: usize) -> Result<Self, GpuError> {
        // Validate n
        if !n.is_power_of_two() {
            return Err(GpuError::InvalidParams(format!("n={} must be power of 2", n)));
        }
        // Shared memory limit is 32KB on most GPUs, but browser WebGPU may have overhead.
        // Use 2048 max (16KB) to be safe.
        if n > 2048 {
            return Err(GpuError::InvalidParams(format!(
                "n={} exceeds max 2048", n
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

        // Log GPU limits
        #[cfg(target_arch = "wasm32")]
        {
            let info = adapter.get_info();
            let limits = adapter.limits();
            web_sys::console::log_1(&format!(
                "[GPU] Adapter: {} ({:?})",
                info.name, info.backend
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

        let twiddle_shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_TWIDDLE_SHADER,
            "goldilocks_twiddle.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let twiddle_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("goldilocks_twiddle pipeline"),
                layout: None,
                module: &twiddle_shader,
                entry_point: Some("apply_twiddle"),
                compilation_options: Default::default(),
                cache: None,
            });

        let batched_ntt_shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_BATCHED_NTT_SHADER,
            "goldilocks_batched_ntt.wgsl",
        )
        .map_err(GpuError::ShaderCompilation)?;

        let batched_ntt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("goldilocks_batched_ntt pipeline"),
                layout: None,
                module: &batched_ntt_shader,
                entry_point: Some("batched_forward_ntt"),
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
            twiddle_pipeline,
            batched_ntt_pipeline,
            modulus_params_buffer,
            forward_twiddles_buffer,
            inverse_twiddles_buffer,
        })
    }

    /// Creates a new GPU NTT context using an existing device and queue.
    ///
    /// NOTE: This function is currently not supported because wgpu Device/Queue
    /// don't implement Clone. Use `new_async` instead to create a new context
    /// with its own device.
    #[allow(unused_variables)]
    pub fn from_device(_device: &Device, _queue: &Queue, n: usize) -> Result<Self, GpuError> {
        // Validate n for better error messages
        if !n.is_power_of_two() {
            return Err(GpuError::InvalidParams(format!("n={} must be power of 2", n)));
        }
        // Shared memory limit is 32KB on most GPUs = 4096 u64 elements.
        if n > 4096 {
            return Err(GpuError::InvalidParams(format!(
                "n={} exceeds max 4096", n
            )));
        }

        // Note: wgpu Device/Queue don't implement Clone, so we can't share them.
        // This function is currently non-functional - use new_async instead.
        Err(GpuError::InvalidParams("from_device not supported - Device/Queue don't impl Clone. Use new_async instead.".into()))
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

    /// Batched polynomial multiplication using GPU NTT.
    ///
    /// Computes `result[i] = a[i] * b[i]` for each pair of polynomials.
    /// Uses NTT for O(n log n) multiplication instead of O(n²) schoolbook.
    ///
    /// # Arguments
    /// * `pairs` - Slice of (a, b) polynomial pairs to multiply
    ///
    /// # Returns
    /// Vector of product polynomials, one per input pair
    pub fn batched_poly_mul(&self, pairs: &[(&[u64], &[u64])]) -> Result<Vec<Vec<u64>>, GpuError> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Compute result lengths and padded length
        let mut result_lens = Vec::with_capacity(pairs.len());
        let mut max_result_len = 0;
        for (a, b) in pairs {
            let rlen = if a.is_empty() || b.is_empty() {
                0
            } else {
                a.len() + b.len() - 1
            };
            result_lens.push(rlen);
            max_result_len = max_result_len.max(rlen);
        }

        // NTT size must be power of 2 and >= max result length
        let ntt_size = max_result_len.next_power_of_two().max(self.n);

        // Prepare padded polynomials for NTT
        let mut a_polys = Vec::with_capacity(pairs.len());
        let mut b_polys = Vec::with_capacity(pairs.len());
        for (a, b) in pairs {
            let mut a_padded = vec![0u64; ntt_size];
            let mut b_padded = vec![0u64; ntt_size];
            a_padded[..a.len()].copy_from_slice(a);
            b_padded[..b.len()].copy_from_slice(b);
            a_polys.push(a_padded);
            b_polys.push(b_padded);
        }

        // Forward NTT on both sets
        let a_evals = self.batched_forward_ntt(&a_polys)?;
        let b_evals = self.batched_forward_ntt(&b_polys)?;

        // Pointwise multiplication (CPU for now, could be GPU)
        let mut prod_evals = Vec::with_capacity(pairs.len());
        for (a_eval, b_eval) in a_evals.iter().zip(b_evals.iter()) {
            let mut prod = Vec::with_capacity(ntt_size);
            for i in 0..ntt_size {
                let p = ((a_eval[i] as u128 * b_eval[i] as u128) % GOLDILOCKS as u128) as u64;
                prod.push(p);
            }
            prod_evals.push(prod);
        }

        // Inverse NTT
        let products = self.batched_inverse_ntt(&prod_evals)?;

        // Trim to actual result lengths
        let mut results = Vec::with_capacity(pairs.len());
        for (prod, &rlen) in products.iter().zip(result_lens.iter()) {
            if rlen == 0 {
                results.push(vec![]);
            } else {
                results.push(prod[..rlen].to_vec());
            }
        }

        Ok(results)
    }

    /// Single polynomial multiplication using GPU NTT.
    ///
    /// Convenience method for multiplying two polynomials.
    pub fn poly_mul(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, GpuError> {
        let results = self.batched_poly_mul(&[(a, b)])?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    // =========================================================================
    // Async versions for WASM (non-blocking GPU operations)
    // =========================================================================

    /// Async version of batched_forward_ntt for WASM.
    pub async fn batched_forward_ntt_async(&self, polys: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        if polys.is_empty() {
            return Ok(Vec::new());
        }

        let num_batches = polys.len();

        for (i, poly) in polys.iter().enumerate() {
            if poly.len() != self.n {
                return Err(GpuError::InvalidParams(format!(
                    "Poly {} has {} coeffs, expected {}", i, poly.len(), self.n
                )));
            }
        }

        let input_flat: Vec<u32> = polys
            .iter()
            .flat_map(|p| p.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        // DEBUG: Check input_flat for forward NTT
        #[cfg(target_arch = "wasm32")]
        {
            let input_nz = input_flat.iter().filter(|&&x| x != 0).count();
            web_sys::console::log_1(&format!(
                "[Forward NTT async] input: len={}, nonzero={}, n={}, batches={}",
                input_flat.len(), input_nz, self.n, num_batches
            ).into());
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ntt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: BufferUsages::STORAGE,
        });

        // Extra 8 bytes for magic sentinel (2 u32s: 0xDEADBEEF_CAFEBABE)
        let output_size = (num_batches * self.n * 2 + 2) * std::mem::size_of::<u32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt_output"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

        let bind_group_layout = self.forward_ntt_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward_ntt bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: batch_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.forward_twiddles_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.modulus_params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_ntt encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // In WASM, browser handles GPU callback scheduling
        // In native, we need to poll the device
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::ExecutionFailed("Channel cancelled".into()))?
            .map_err(GpuError::BufferMapping)?;

        let data = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&data);

        // Check magic sentinel to verify shader actually executed
        let magic_idx = num_batches * self.n * 2;
        let magic_lo = output_u32[magic_idx];
        let magic_hi = output_u32[magic_idx + 1];
        if magic_lo != 0xCAFEBABE || magic_hi != 0xDEADBEEF {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(&format!(
                "[Forward NTT] MAGIC CHECK FAILED! Expected 0xDEADBEEF_CAFEBABE, got 0x{:08X}_{:08X}",
                magic_hi, magic_lo
            ).into());
            panic!(
                "Forward NTT shader did not execute! Magic check failed: expected 0xDEADBEEF_CAFEBABE, got 0x{:08X}_{:08X}",
                magic_hi, magic_lo
            );
        }

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"[Forward NTT] Magic check PASSED - shader executed".into());

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

        // DEBUG: Check output
        #[cfg(target_arch = "wasm32")]
        if !results.is_empty() {
            let out_nz = results[0].iter().filter(|&&x| x != 0).count();
            web_sys::console::log_1(&format!(
                "[Forward NTT async] output: results[0] has {} nonzero (out of {})",
                out_nz, self.n
            ).into());
        }

        Ok(results)
    }

    /// Async version of batched_inverse_ntt for WASM.
    pub async fn batched_inverse_ntt_async(&self, evals: &[Vec<u64>]) -> Result<Vec<Vec<u64>>, GpuError> {
        if evals.is_empty() {
            return Ok(Vec::new());
        }

        let num_batches = evals.len();

        for (i, eval) in evals.iter().enumerate() {
            if eval.len() != self.n {
                return Err(GpuError::InvalidParams(format!(
                    "Eval {} has {} values, expected {}", i, eval.len(), self.n
                )));
            }
        }

        let input_flat: Vec<u32> = evals
            .iter()
            .flat_map(|e| e.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("intt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: BufferUsages::STORAGE,
        });

        let output_size = num_batches * self.n * 2 * std::mem::size_of::<u32>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("intt_output"),
            size: output_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

        let bind_group_layout = self.inverse_ntt_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inverse_ntt bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: batch_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.inverse_twiddles_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.modulus_params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inverse_ntt encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inverse_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_batches as u32, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // In WASM, browser handles GPU callback scheduling
        // In native, we need to poll the device
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::ExecutionFailed("Channel cancelled".into()))?
            .map_err(GpuError::BufferMapping)?;

        let data = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&data);

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

    /// Async version of batched_poly_mul for WASM.
    pub async fn batched_poly_mul_async(&self, pairs: &[(&[u64], &[u64])]) -> Result<Vec<Vec<u64>>, GpuError> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let mut result_lens = Vec::with_capacity(pairs.len());
        let mut max_result_len = 0;
        for (a, b) in pairs {
            let rlen = if a.is_empty() || b.is_empty() {
                0
            } else {
                a.len() + b.len() - 1
            };
            result_lens.push(rlen);
            max_result_len = max_result_len.max(rlen);
        }

        let ntt_size = max_result_len.next_power_of_two().max(self.n);

        let mut a_polys = Vec::with_capacity(pairs.len());
        let mut b_polys = Vec::with_capacity(pairs.len());
        for (a, b) in pairs {
            let mut a_padded = vec![0u64; ntt_size];
            let mut b_padded = vec![0u64; ntt_size];
            a_padded[..a.len()].copy_from_slice(a);
            b_padded[..b.len()].copy_from_slice(b);
            a_polys.push(a_padded);
            b_polys.push(b_padded);
        }

        // DEBUG: Find first nonzero input and check it
        #[cfg(target_arch = "wasm32")]
        let debug_idx = {
            let mut idx = 0;
            for (i, a) in a_polys.iter().enumerate() {
                if a.iter().any(|&x| x != 0) {
                    idx = i;
                    break;
                }
            }
            idx
        };

        let a_evals = self.batched_forward_ntt_async(&a_polys).await?;
        let b_evals = self.batched_forward_ntt_async(&b_polys).await?;

        // DEBUG: Check NTT results for the first nonzero input
        #[cfg(target_arch = "wasm32")]
        {
            let a_in_nz = a_polys[debug_idx].iter().filter(|&&x| x != 0).count();
            let a_out_nz = a_evals[debug_idx].iter().filter(|&&x| x != 0).count();
            web_sys::console::log_1(&format!(
                "[GPU poly_mul] idx={}: input has {} nonzero, NTT output has {} nonzero (n={})",
                debug_idx, a_in_nz, a_out_nz, ntt_size
            ).into());
        }

        let mut prod_evals = Vec::with_capacity(pairs.len());
        for (a_eval, b_eval) in a_evals.iter().zip(b_evals.iter()) {
            let mut prod = Vec::with_capacity(ntt_size);
            for i in 0..ntt_size {
                let p = ((a_eval[i] as u128 * b_eval[i] as u128) % GOLDILOCKS as u128) as u64;
                prod.push(p);
            }
            prod_evals.push(prod);
        }

        let products = self.batched_inverse_ntt_async(&prod_evals).await?;

        let mut results = Vec::with_capacity(pairs.len());
        for (prod, &rlen) in products.iter().zip(result_lens.iter()) {
            if rlen == 0 {
                results.push(vec![]);
            } else {
                results.push(prod[..rlen].to_vec());
            }
        }

        Ok(results)
    }

    /// Applies twiddle factor multiplication for four-step FFT.
    ///
    /// After Pass 1 (row NTTs of size N1), multiplies each element at position
    /// (row, col) by ω_N^(row * col) where N = N1 * N2.
    ///
    /// # Arguments
    /// * `data` - In-place buffer containing the data (N elements as u64)
    /// * `n` - Total NTT size (N = N1 * N2)
    /// * `n1` - Row size (number of columns, typically 2048)
    pub async fn apply_twiddle_async(
        &self,
        data: &mut [u64],
        n: usize,
        n1: usize,
    ) -> Result<(), GpuError> {
        if data.len() != n {
            return Err(GpuError::InvalidParams(format!(
                "Data length {} != n={}", data.len(), n
            )));
        }
        if n % n1 != 0 {
            return Err(GpuError::InvalidParams(format!(
                "n={} not divisible by n1={}", n, n1
            )));
        }

        // Create data buffer (read-write)
        let data_flat: Vec<u32> = data
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddle_data"),
            contents: bytemuck::cast_slice(&data_flat),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        // Create twiddles for size N (need ω_N^k for k = 0..N-1)
        // We can reuse forward_twiddles if N == self.n, otherwise compute new ones
        let twiddles_buffer = if n == self.n {
            // Reuse existing twiddles
            &self.forward_twiddles_buffer
        } else {
            // Need to compute twiddles for the larger size N
            let omega = find_primitive_nth_root(n)
                .ok_or_else(|| GpuError::InvalidParams(format!("No primitive root for n={}", n)))?;
            let omega_powers = compute_powers(omega, n, GOLDILOCKS);
            let twiddles: Vec<u32> = omega_powers
                .iter()
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect();

            // Create temporary buffer
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("twiddle_factors_temp"),
                contents: bytemuck::cast_slice(&twiddles),
                usage: BufferUsages::STORAGE,
            });
            // Note: We leak this buffer for now. For production, we'd cache it.
            Box::leak(Box::new(buf))
        };

        // Create params buffer
        let params = GpuTwiddleParams {
            n: n as u32,
            n1: n1 as u32,
            num_elements: n as u32,
            _pad: 0,
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group_layout = self.twiddle_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("twiddle bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: twiddles_buffer.as_entire_binding() },
            ],
        });

        // Dispatch
        let num_workgroups = (n + 255) / 256;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("twiddle encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twiddle pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.twiddle_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        // Copy back to staging
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("twiddle_staging"),
            size: (n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &data_buffer,
            0,
            &staging_buffer,
            0,
            (n * 2 * std::mem::size_of::<u32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map and read back
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::ExecutionFailed("Channel cancelled".into()))?
            .map_err(GpuError::BufferMapping)?;

        let mapped = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&mapped);

        // Write back to data
        for i in 0..n {
            let lo = output_u32[i * 2] as u64;
            let hi = output_u32[i * 2 + 1] as u64;
            data[i] = lo | (hi << 32);
        }

        drop(mapped);
        staging_buffer.unmap();

        Ok(())
    }

    /// Multi-pass forward NTT for sizes larger than 1024.
    /// Uses four-step FFT algorithm: N = N1 × N2
    /// 1. N2 row NTTs of size N1
    /// 2. Twiddle multiplication
    /// 3. N1 column NTTs of size N2
    pub async fn multipass_forward_ntt_async(
        &self,
        data: &[u64],
        target_n: usize,
    ) -> Result<Vec<u64>, GpuError> {
        // Validate target_n
        if target_n == 0 || (target_n & (target_n - 1)) != 0 {
            return Err(GpuError::InvalidParams(format!(
                "target_n={} must be a power of 2", target_n
            )));
        }

        // Pad input to target_n
        let mut work_data: Vec<u64> = data.to_vec();
        work_data.resize(target_n, 0);

        // Max single-pass size is 1024 (limited by shared memory)
        const MAX_SINGLE_PASS: usize = 1024;

        if target_n <= MAX_SINGLE_PASS {
            // Use single-pass NTT
            if target_n == self.n {
                let result = self.batched_forward_ntt_async(&[work_data]).await?;
                return Ok(result.into_iter().next().unwrap());
            } else {
                // Need a temporary GPU context for the target size
                let temp_gpu = GoldilocksNttGpu::new_async(target_n).await?;
                let result = temp_gpu.batched_forward_ntt_async(&[work_data]).await?;
                return Ok(result.into_iter().next().unwrap());
            }
        }

        // Four-step FFT: N = N1 × N2 where both ≤ 1024
        // Maximize N1 (up to 1024) to use efficient single-pass NTT for rows
        let n1 = MAX_SINGLE_PASS.min(target_n);  // N1 = min(1024, N)
        let n2 = target_n / n1;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!(
            "[Multipass NTT] N={} = N1={} × N2={}", target_n, n1, n2
        ).into());

        if n1 > MAX_SINGLE_PASS || n2 > MAX_SINGLE_PASS {
            return Err(GpuError::InvalidParams(format!(
                "NTT size {} too large, N1={} or N2={} exceeds max {}",
                target_n, n1, n2, MAX_SINGLE_PASS
            )));
        }

        // =====================================================================
        // Bailey's Four-Step FFT (correct order):
        // Data: N2 rows × N1 columns (row-major), N = N1 * N2
        // 1. Column NTTs: N1 NTTs of size N2
        // 2. Twiddle: element at (row, col) *= ω_N^(col * row)
        // 3. Row NTTs: N2 NTTs of size N1
        // 4. Read output column-major: X[col * N2 + row] = D[row][col]
        // =====================================================================

        // =====================================================================
        // Pass 1: N1 column NTTs of size N2 (stride = N1)
        // =====================================================================
        work_data = self.batched_column_ntt_async(&work_data, n1, n2, target_n).await?;

        // =====================================================================
        // Twiddle multiplication: element at (row, col) *= ω_N^(col * row)
        // =====================================================================
        self.apply_twiddle_async(&mut work_data, target_n, n1).await?;

        // =====================================================================
        // Pass 2: N2 row NTTs of size N1
        // =====================================================================
        let rows: Vec<Vec<u64>> = (0..n2)
            .map(|row| work_data[row * n1..(row + 1) * n1].to_vec())
            .collect();

        let row_ntt_results = if n1 == self.n {
            self.batched_forward_ntt_async(&rows).await?
        } else {
            let temp_gpu = GoldilocksNttGpu::new_async(n1).await?;
            temp_gpu.batched_forward_ntt_async(&rows).await?
        };

        for (row, row_data) in row_ntt_results.iter().enumerate() {
            for (col, &val) in row_data.iter().enumerate() {
                work_data[row * n1 + col] = val;
            }
        }

        // =====================================================================
        // Read output column-major: X[col * N2 + row] = D[row][col]
        // =====================================================================
        let mut result = vec![0u64; target_n];
        for row in 0..n2 {
            for col in 0..n1 {
                let output_idx = col * n2 + row;
                let src_idx = row * n1 + col;
                result[output_idx] = work_data[src_idx];
            }
        }

        Ok(result)
    }

    /// Apply batched column NTTs using the batched_ntt_pipeline.
    /// Each column is an NTT of size n2, with elements spaced n1 apart.
    async fn batched_column_ntt_async(
        &self,
        data: &[u64],
        n1: usize, // Number of columns (stride between elements)
        n2: usize, // Column height (NTT size)
        total_n: usize,
    ) -> Result<Vec<u64>, GpuError> {
        if data.len() != total_n {
            return Err(GpuError::InvalidParams(format!(
                "Data length {} != total_n={}", data.len(), total_n
            )));
        }

        // Flatten input
        let input_flat: Vec<u32> = data
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("column_ntt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("column_ntt_output"),
            size: (total_n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Twiddles for column NTT size n2
        let omega = find_primitive_nth_root(n2)
            .ok_or_else(|| GpuError::InvalidParams(format!("No primitive root for n2={}", n2)))?;
        let omega_powers = compute_powers(omega, n2, GOLDILOCKS);
        let twiddles_flat: Vec<u32> = omega_powers
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let twiddles_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("column_twiddles"),
            contents: bytemuck::cast_slice(&twiddles_flat),
            usage: BufferUsages::STORAGE,
        });

        // Configure batched NTT params:
        // - n = n2 (NTT size per column)
        // - total_batches = n1 (number of columns)
        // - stride = n1 (distance between consecutive elements of a column)
        // - batch_per_wg: fit as many columns as we can in shared memory (2048 elements)
        let max_shared_elements = 2048usize;
        let batch_per_wg = (max_shared_elements / n2).max(1);

        let params = GpuBatchedNttParams {
            n: n2 as u32,
            log_n: n2.trailing_zeros(),
            batch_per_wg: batch_per_wg as u32,
            total_batches: n1 as u32,
            stride: n1 as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("column_ntt_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group_layout = self.batched_ntt_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("column_ntt bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: twiddles_buffer.as_entire_binding() },
            ],
        });

        // Dispatch
        let num_workgroups = (n1 + batch_per_wg - 1) / batch_per_wg;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("column_ntt encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("column_ntt pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batched_ntt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        // Copy to staging
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("column_ntt_staging"),
            size: (total_n * 2 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_n * 2 * std::mem::size_of::<u32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map and read back
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| GpuError::ExecutionFailed("Channel cancelled".into()))?
            .map_err(GpuError::BufferMapping)?;

        let mapped = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&mapped);

        let mut result = vec![0u64; total_n];
        for i in 0..total_n {
            let lo = output_u32[i * 2] as u64;
            let hi = output_u32[i * 2 + 1] as u64;
            result[i] = lo | (hi << 32);
        }

        drop(mapped);
        staging_buffer.unmap();

        Ok(result)
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

var<workgroup> shared_lo: array<u32, 2048>;
var<workgroup> shared_hi: array<u32, 2048>;

// Goldilocks modular multiplication using fixed Goldilocks-specific reduction
// (uses the special structure of p = 2^64 - 2^32 + 1 for efficient reduction)
fn mulmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_mul(a, b);
}

fn addmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_add(a, b);
}

fn submod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_sub(a, b);
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

    // Keep mod_params binding alive (shared bind group layout with inverse NTT)
    let _keep = mod_params.modulus_lo * 0u;

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

    // Write magic sentinel at the end of buffer (only thread 0 of last workgroup)
    // Magic value: 0xDEADBEEF_CAFEBABE
    if tid == 0u && batch_idx == params.num_batches - 1u {
        let magic_base = params.num_batches * n * 2u;
        ntt_out[magic_base] = 0xCAFEBABEu;
        ntt_out[magic_base + 1u] = 0xDEADBEEFu;
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

// WebGPU workgroup memory limit: use 16KB to be safe in browser
var<workgroup> shared_lo: array<u32, 2048>;
var<workgroup> shared_hi: array<u32, 2048>;

// Goldilocks modular multiplication using fixed Goldilocks-specific reduction
fn mulmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_mul(a, b);
}

fn addmod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_add(a, b);
}

fn submod_goldilocks(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return math::goldilocks_sub(a, b);
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

/// Batched small NTT shader for Pass 2 of four-step FFT.
/// Handles NTT sizes from 2 to 2048 by batching multiple NTTs per workgroup.
/// For n < 256, multiple NTTs share the workgroup to maximize parallelism.
const GOLDILOCKS_BATCHED_NTT_SHADER: &str = r#"
#import math

struct BatchedNttParams {
    n: u32,              // NTT size (must be power of 2, 2 to 2048)
    log_n: u32,          // log2(n)
    batch_per_wg: u32,   // How many NTTs per workgroup
    total_batches: u32,  // Total number of NTTs to process
    stride: u32,         // Stride between consecutive elements of one NTT in global memory
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: BatchedNttParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> twiddles: array<u32>;

var<workgroup> shared_lo: array<u32, 2048>;
var<workgroup> shared_hi: array<u32, 2048>;

// Bit reverse for small values
fn bit_reverse_small(x: u32, bits: u32) -> u32 {
    var result = 0u;
    var val = x;
    for (var i = 0u; i < bits; i++) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

@compute @workgroup_size(256, 1, 1)
fn batched_forward_ntt(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let n = params.n;
    let log_n = params.log_n;
    let batch_per_wg = params.batch_per_wg;
    let stride = params.stride;

    let wg_batch_start = wg_id.x * batch_per_wg;
    let total_elements = n * batch_per_wg;

    // Load into shared memory with bit-reversal
    // Each thread handles elements at indices tid, tid+256, tid+512, ...
    var local_idx = tid;
    while (local_idx < total_elements) {
        let batch_in_wg = local_idx / n;
        let pos_in_ntt = local_idx % n;
        let global_batch = wg_batch_start + batch_in_wg;

        if (global_batch < params.total_batches) {
            // For column NTTs in four-step: element j of NTT i is at position i + j * stride
            let global_idx = global_batch + pos_in_ntt * stride;
            let val_lo = input[global_idx * 2u];
            let val_hi = input[global_idx * 2u + 1u];

            // Bit-reverse within the NTT
            let rev_pos = bit_reverse_small(pos_in_ntt, log_n);
            let shared_idx = batch_in_wg * n + rev_pos;
            shared_lo[shared_idx] = val_lo;
            shared_hi[shared_idx] = val_hi;
        }
        local_idx += 256u;
    }
    workgroupBarrier();

    // NTT butterfly stages
    let butterflies_per_wg = total_elements / 2u;

    for (var s = 0u; s < log_n; s++) {
        let m = 1u << (s + 1u);
        let half_m = m >> 1u;

        var butterfly_idx = tid;
        while (butterfly_idx < butterflies_per_wg) {
            let batch_in_wg = butterfly_idx / (n / 2u);
            let bf_in_ntt = butterfly_idx % (n / 2u);

            let global_batch = wg_batch_start + batch_in_wg;
            if (global_batch < params.total_batches) {
                // Which butterfly group and position within group
                let group = bf_in_ntt / half_m;
                let pos = bf_in_ntt % half_m;

                let ii = batch_in_wg * n + group * m + pos;
                let jj = ii + half_m;

                let u = vec2<u32>(shared_lo[ii], shared_hi[ii]);
                let v = vec2<u32>(shared_lo[jj], shared_hi[jj]);

                // Twiddle factor: omega^(pos * n / m)
                let twiddle_idx = pos * (n / m);
                let tw = vec2<u32>(twiddles[twiddle_idx * 2u], twiddles[twiddle_idx * 2u + 1u]);

                let tw_v = math::goldilocks_mul(v, tw);
                let new_u = math::goldilocks_add(u, tw_v);
                let new_v = math::goldilocks_sub(u, tw_v);

                shared_lo[ii] = new_u.x;
                shared_hi[ii] = new_u.y;
                shared_lo[jj] = new_v.x;
                shared_hi[jj] = new_v.y;
            }
            butterfly_idx += 256u;
        }
        workgroupBarrier();
    }

    // Store back to global memory (with stride)
    local_idx = tid;
    while (local_idx < total_elements) {
        let batch_in_wg = local_idx / n;
        let pos_in_ntt = local_idx % n;
        let global_batch = wg_batch_start + batch_in_wg;

        if (global_batch < params.total_batches) {
            let global_idx = global_batch + pos_in_ntt * stride;
            let shared_idx = batch_in_wg * n + pos_in_ntt;
            output[global_idx * 2u] = shared_lo[shared_idx];
            output[global_idx * 2u + 1u] = shared_hi[shared_idx];
        }
        local_idx += 256u;
    }
}
"#;

/// Twiddle factor multiplication shader for four-step FFT.
/// Multiplies element at position (row, col) by ω_N^(row * col).
const GOLDILOCKS_TWIDDLE_SHADER: &str = r#"
#import math

struct TwiddleParams {
    n: u32,           // Total NTT size (N = N1 * N2)
    n1: u32,          // Row size (number of columns)
    num_elements: u32, // Total elements to process
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: TwiddleParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn apply_twiddle(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= params.num_elements) { return; }

    let n = params.n;
    let n1 = params.n1;

    // Compute row and column in the conceptual 2D array
    let row = idx / n1;
    let col = idx % n1;

    // Twiddle factor index: (row * col) mod N
    let twiddle_idx = (row * col) % n;

    // Load element (stored as two u32s for u64)
    let elem_base = idx * 2u;
    let elem = vec2<u32>(data[elem_base], data[elem_base + 1u]);

    // Load twiddle factor
    let tw_base = twiddle_idx * 2u;
    let tw = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

    // Multiply
    let result = math::goldilocks_mul(elem, tw);

    // Store
    data[elem_base] = result.x;
    data[elem_base + 1u] = result.y;
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

    #[test]
    fn test_poly_mul_simple() {
        let n = 1024;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
        let a = vec![1u64, 2];
        let b = vec![3u64, 4];
        let result = gpu.poly_mul(&a, &b).expect("poly_mul failed");

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 3);  // constant term
        assert_eq!(result[1], 10); // x coefficient
        assert_eq!(result[2], 8);  // x² coefficient
    }

    #[test]
    fn test_poly_mul_vs_cpu() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 1024;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // Create random-ish polynomials
        let a: Vec<u64> = (0..100).map(|i| (i * 12345 + 67) % GOLDILOCKS).collect();
        let b: Vec<u64> = (0..80).map(|i| (i * 54321 + 89) % GOLDILOCKS).collect();

        // GPU multiplication
        let gpu_result = gpu.poly_mul(&a, &b).expect("GPU poly_mul failed");

        // CPU multiplication using NTT
        let result_len = a.len() + b.len() - 1;
        let ntt_n = result_len.next_power_of_two();

        let mut a_ntt: Vec<Goldilocks> = a.iter().map(|&x| Goldilocks::new(x)).collect();
        let mut b_ntt: Vec<Goldilocks> = b.iter().map(|&x| Goldilocks::new(x)).collect();
        a_ntt.resize(ntt_n, Goldilocks::new(0));
        b_ntt.resize(ntt_n, Goldilocks::new(0));

        Goldilocks::ntt(&mut a_ntt);
        Goldilocks::ntt(&mut b_ntt);
        for i in 0..ntt_n {
            a_ntt[i] = a_ntt[i] * b_ntt[i];
        }
        Goldilocks::intt(&mut a_ntt);

        let cpu_result: Vec<u64> = a_ntt.iter().take(result_len).map(|x| x.inner()).collect();

        // Compare
        assert_eq!(gpu_result.len(), cpu_result.len());
        for i in 0..result_len {
            assert_eq!(
                gpu_result[i], cpu_result[i],
                "poly_mul mismatch at {}: GPU={}, CPU={}",
                i, gpu_result[i], cpu_result[i]
            );
        }
    }

    #[test]
    fn test_batched_poly_mul() {
        let n = 1024;
        let gpu = GoldilocksNttGpu::new(n).expect("Failed to create GPU context");

        // Multiple polynomial pairs
        let pairs: Vec<(Vec<u64>, Vec<u64>)> = vec![
            (vec![1, 2], vec![3, 4]),           // (1+2x)(3+4x) = 3+10x+8x²
            (vec![1, 1, 1], vec![1, 1]),        // (1+x+x²)(1+x) = 1+2x+2x²+x³
            (vec![5], vec![7]),                 // 5 * 7 = 35
        ];

        let pairs_ref: Vec<(&[u64], &[u64])> = pairs.iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();

        let results = gpu.batched_poly_mul(&pairs_ref).expect("batched_poly_mul failed");

        assert_eq!(results.len(), 3);

        // Check first: 3 + 10x + 8x²
        assert_eq!(results[0], vec![3, 10, 8]);

        // Check second: 1 + 2x + 2x² + x³
        assert_eq!(results[1], vec![1, 2, 2, 1]);

        // Check third: 35
        assert_eq!(results[2], vec![35]);
    }

    #[test]
    fn test_twiddle_shader() {
        // Test the twiddle shader: multiply element at (row, col) by ω_N^(row * col)
        let n = 8usize;   // Total size N = N1 * N2
        let n1 = 4usize;  // Row size (4 columns)
        let n2 = 2usize;  // 2 rows

        // Create GPU context
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let adapter = match adapter {
            Some(a) => a,
            None => { println!("No GPU available, skipping test"); return; }
        };
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
        ).unwrap();

        // Compile shader
        let shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_TWIDDLE_SHADER,
            "twiddle_test.wgsl",
        ).expect("Failed to compile twiddle shader");

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("twiddle_test_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("apply_twiddle"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create test data: all ones
        let mut data: Vec<u64> = vec![1u64; n];

        // Compute expected result on CPU
        let omega = find_primitive_nth_root(n).expect("No primitive root");
        let mut expected = vec![0u64; n];
        for idx in 0..n {
            let row = idx / n1;
            let col = idx % n1;
            let twiddle_exp = (row * col) % n;
            let tw = mod_pow(omega, twiddle_exp as u64, GOLDILOCKS);
            expected[idx] = mod_mul(data[idx], tw, GOLDILOCKS);
        }

        // Flatten data for GPU
        let data_flat: Vec<u32> = data.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddle_data"),
            contents: bytemuck::cast_slice(&data_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create twiddles buffer
        let omega_powers = compute_powers(omega, n, GOLDILOCKS);
        let twiddles_flat: Vec<u32> = omega_powers.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddles"),
            contents: bytemuck::cast_slice(&twiddles_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create params buffer
        let params = GpuTwiddleParams {
            n: n as u32,
            n1: n1 as u32,
            num_elements: n as u32,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("twiddle_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: twiddles_buffer.as_entire_binding() },
            ],
        });

        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy to staging
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 2 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging, 0, (n * 2 * 4) as u64);
        queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&mapped);

        // Compare
        for i in 0..n {
            let lo = output_u32[i * 2] as u64;
            let hi = output_u32[i * 2 + 1] as u64;
            let got = lo | (hi << 32);
            assert_eq!(
                got, expected[i],
                "Twiddle mismatch at idx={} (row={}, col={}): got {}, expected {}",
                i, i / n1, i % n1, got, expected[i]
            );
        }

        drop(mapped);
        staging.unmap();
        println!("Twiddle shader test passed!");
    }

    #[test]
    fn test_batched_ntt_shader() {
        // Test the batched NTT shader with small NTTs
        let n = 8usize;          // NTT size
        let batch_per_wg = 4;    // 4 NTTs per workgroup (32 elements total, fits in shared)
        let total_batches = 4;   // Total NTTs
        let stride = total_batches; // Column layout: element j of NTT i at position i + j * stride

        // Create GPU context
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let adapter = match adapter {
            Some(a) => a,
            None => { println!("No GPU available, skipping test"); return; }
        };
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
        ).unwrap();

        // Compile shader
        let shader = crate::shader_math::create_shader_module(
            &device,
            GOLDILOCKS_BATCHED_NTT_SHADER,
            "batched_ntt_test.wgsl",
        ).expect("Failed to compile batched NTT shader");

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batched_ntt_test_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("batched_forward_ntt"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create test data: batch_per_wg polynomials, each of size n
        // Layout: element j of NTT i is at index i + j * stride
        let total_elements = n * total_batches;
        let mut input_data = vec![0u64; total_elements];

        // Fill with simple test values: NTT i gets values [i, i+1, i+2, ...]
        for batch in 0..total_batches {
            for j in 0..n {
                let idx = batch + j * stride;
                input_data[idx] = ((batch * n + j) as u64) % GOLDILOCKS;
            }
        }

        // Compute expected NTT results on CPU
        let omega = find_primitive_nth_root(n).expect("No primitive root");
        let mut expected = vec![vec![0u64; n]; total_batches];
        for batch in 0..total_batches {
            // Extract input for this batch
            let mut poly = vec![0u64; n];
            for j in 0..n {
                poly[j] = input_data[batch + j * stride];
            }
            // Compute NTT
            for k in 0..n {
                let mut sum = 0u128;
                for j in 0..n {
                    let exp = (j * k) % n;
                    let tw = mod_pow(omega, exp as u64, GOLDILOCKS);
                    sum += (poly[j] as u128 * tw as u128) % GOLDILOCKS as u128;
                }
                expected[batch][k] = (sum % GOLDILOCKS as u128) as u64;
            }
        }

        // Flatten input for GPU
        let input_flat: Vec<u32> = input_data.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batched_ntt_input"),
            contents: bytemuck::cast_slice(&input_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batched_ntt_output"),
            size: (total_elements * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create twiddles buffer for size n
        let omega_powers = compute_powers(omega, n, GOLDILOCKS);
        let twiddles_flat: Vec<u32> = omega_powers.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddles"),
            contents: bytemuck::cast_slice(&twiddles_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create params buffer
        let params = GpuBatchedNttParams {
            n: n as u32,
            log_n: n.trailing_zeros(),
            batch_per_wg: batch_per_wg as u32,
            total_batches: total_batches as u32,
            stride: stride as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batched_ntt_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: twiddles_buffer.as_entire_binding() },
            ],
        });

        // Dispatch (1 workgroup handles all 4 NTTs)
        let num_workgroups = (total_batches + batch_per_wg - 1) / batch_per_wg;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        // Copy to staging
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (total_elements * 2 * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (total_elements * 2 * 4) as u64);
        queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();
        let output_u32: &[u32] = bytemuck::cast_slice(&mapped);

        // Compare
        for batch in 0..total_batches {
            for k in 0..n {
                let idx = batch + k * stride;
                let lo = output_u32[idx * 2] as u64;
                let hi = output_u32[idx * 2 + 1] as u64;
                let got = lo | (hi << 32);
                assert_eq!(
                    got, expected[batch][k],
                    "Batched NTT mismatch at batch={}, k={}: got {}, expected {}",
                    batch, k, got, expected[batch][k]
                );
            }
        }

        drop(mapped);
        staging.unmap();
        println!("Batched NTT shader test passed!");
    }

    #[test]
    fn test_multipass_ntt_2048() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 2048;  // Larger than max single-pass (1024)

        // Create GPU context (can be any size >= 1024 for the row NTTs)
        let gpu = GoldilocksNttGpu::new(1024).expect("Failed to create GPU context");

        // Create test polynomial
        let poly: Vec<u64> = (0..n as u64).map(|i| (i * 12345 + 67) % GOLDILOCKS).collect();

        // GPU multi-pass forward NTT
        let gpu_result = pollster::block_on(gpu.multipass_forward_ntt_async(&poly, n))
            .expect("Multipass NTT failed");

        // CPU reference NTT
        let mut cpu_vals: Vec<Goldilocks> = poly.iter()
            .map(|&x| Goldilocks::new(x))
            .collect();
        Goldilocks::ntt(&mut cpu_vals);
        let cpu_result: Vec<u64> = cpu_vals.iter().map(|x| x.inner()).collect();

        // Compare
        assert_eq!(gpu_result.len(), cpu_result.len());
        for i in 0..n {
            assert_eq!(
                gpu_result[i], cpu_result[i],
                "Multipass NTT mismatch at index {}: GPU={}, CPU={}",
                i, gpu_result[i], cpu_result[i]
            );
        }

        println!("Multipass NTT 2048 test passed!");
    }

    #[test]
    fn test_multipass_ntt_4096() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 4096;  // N=4096 = 64×64

        let gpu = GoldilocksNttGpu::new(1024).expect("Failed to create GPU context");

        let poly: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 3) % GOLDILOCKS).collect();

        let gpu_result = pollster::block_on(gpu.multipass_forward_ntt_async(&poly, n))
            .expect("Multipass NTT failed");

        let mut cpu_vals: Vec<Goldilocks> = poly.iter()
            .map(|&x| Goldilocks::new(x))
            .collect();
        Goldilocks::ntt(&mut cpu_vals);
        let cpu_result: Vec<u64> = cpu_vals.iter().map(|x| x.inner()).collect();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for i in 0..n {
            assert_eq!(
                gpu_result[i], cpu_result[i],
                "Multipass NTT mismatch at index {}: GPU={}, CPU={}",
                i, gpu_result[i], cpu_result[i]
            );
        }

        println!("Multipass NTT 4096 test passed!");
    }

    #[test]
    fn test_multipass_ntt_8192() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 8192;  // N=8192 = 1024×8

        let gpu = GoldilocksNttGpu::new(1024).expect("Failed to create GPU context");

        let poly: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 3) % GOLDILOCKS).collect();

        let gpu_result = pollster::block_on(gpu.multipass_forward_ntt_async(&poly, n))
            .expect("Multipass NTT failed");

        let mut cpu_vals: Vec<Goldilocks> = poly.iter()
            .map(|&x| Goldilocks::new(x))
            .collect();
        Goldilocks::ntt(&mut cpu_vals);
        let cpu_result: Vec<u64> = cpu_vals.iter().map(|x| x.inner()).collect();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for i in 0..n {
            assert_eq!(
                gpu_result[i], cpu_result[i],
                "Multipass NTT mismatch at index {}: GPU={}, CPU={}",
                i, gpu_result[i], cpu_result[i]
            );
        }

        println!("Multipass NTT 8192 test passed!");
    }

    #[test]
    fn test_multipass_ntt_16384() {
        use mpz_fields::goldilocks::Goldilocks;

        let n = 16384;  // N=16384 = 1024×16

        let gpu = GoldilocksNttGpu::new(1024).expect("Failed to create GPU context");

        let poly: Vec<u64> = (0..n as u64).map(|i| (i * 11 + 5) % GOLDILOCKS).collect();

        let gpu_result = pollster::block_on(gpu.multipass_forward_ntt_async(&poly, n))
            .expect("Multipass NTT failed");

        let mut cpu_vals: Vec<Goldilocks> = poly.iter()
            .map(|&x| Goldilocks::new(x))
            .collect();
        Goldilocks::ntt(&mut cpu_vals);
        let cpu_result: Vec<u64> = cpu_vals.iter().map(|x| x.inner()).collect();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for i in 0..n {
            assert_eq!(
                gpu_result[i], cpu_result[i],
                "Multipass NTT mismatch at index {}: GPU={}, CPU={}",
                i, gpu_result[i], cpu_result[i]
            );
        }

        println!("Multipass NTT 16384 test passed!");
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

    /// CPU polynomial multiplication for comparison
    fn cpu_poly_mul(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let result_len = a.len() + b.len() - 1;
        let mut result = vec![0u64; result_len];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = ((ai as u128) * (bj as u128)) % (modulus as u128);
                result[i + j] = ((result[i + j] as u128 + prod) % (modulus as u128)) as u64;
            }
        }
        result
    }

    #[test]
    fn test_gpu_poly_mul_correctness() {
        let p = GOLDILOCKS;

        // Test various sizes
        for &size in &[4usize, 8, 16, 32, 64, 128, 256, 512, 1024] {
            println!("\n=== Testing poly_mul size {} ===", size);

            // Create NTT context
            let ntt_size = (2 * size).next_power_of_two().max(1024);
            if ntt_size > 4096 {
                println!("Skipping size {} - product would exceed NTT limit", size);
                continue;
            }

            let ntt = GoldilocksNttGpu::new(ntt_size).expect("Failed to create NTT context");

            // Generate test polynomials
            let a: Vec<u64> = (0..size).map(|i| (i as u64 * 12345 + 67890) % p).collect();
            let b: Vec<u64> = (0..size).map(|i| (i as u64 * 98765 + 43210) % p).collect();

            // CPU multiplication
            let cpu_result = cpu_poly_mul(&a, &b, p);

            // GPU multiplication
            let gpu_results = ntt.batched_poly_mul(&[(&a, &b)]).expect("GPU poly_mul failed");
            let gpu_result = &gpu_results[0];

            // Compare lengths
            assert_eq!(cpu_result.len(), gpu_result.len(),
                "Length mismatch: CPU={}, GPU={}", cpu_result.len(), gpu_result.len());

            // Compare values
            let mut mismatches = 0;
            for i in 0..cpu_result.len() {
                if cpu_result[i] != gpu_result[i] {
                    if mismatches < 5 {
                        println!("  Mismatch at [{}]: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
                    }
                    mismatches += 1;
                }
            }

            if mismatches > 0 {
                panic!("FAIL: {} mismatches out of {} coefficients", mismatches, cpu_result.len());
            }

            println!("  PASS: All {} coefficients match", cpu_result.len());
        }
    }

    #[test]
    fn test_gpu_poly_mul_random() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let p = GOLDILOCKS;
        let mut rng = StdRng::seed_from_u64(42);

        for size in [64usize, 256, 1024] {
            println!("\n=== Testing random poly_mul size {} ===", size);

            let ntt_size = (2 * size).next_power_of_two().max(1024);
            let ntt = GoldilocksNttGpu::new(ntt_size).expect("Failed to create NTT context");

            // Random polynomials
            let a: Vec<u64> = (0..size).map(|_| rng.gen::<u64>() % p).collect();
            let b: Vec<u64> = (0..size).map(|_| rng.gen::<u64>() % p).collect();

            let cpu_result = cpu_poly_mul(&a, &b, p);
            let gpu_results = ntt.batched_poly_mul(&[(&a, &b)]).expect("GPU poly_mul failed");
            let gpu_result = &gpu_results[0];

            assert_eq!(cpu_result.len(), gpu_result.len());

            for i in 0..cpu_result.len() {
                assert_eq!(cpu_result[i], gpu_result[i],
                    "Mismatch at [{}]: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
            }

            println!("  PASS: All {} coefficients match", cpu_result.len());
        }
    }

    #[test]
    fn test_gpu_poly_mul_async() {
        let p = GOLDILOCKS;

        for &size in &[64usize, 128, 256] {
            println!("\n=== Testing ASYNC poly_mul size {} ===", size);

            let ntt_size = (2 * size).next_power_of_two().max(1024);
            let ntt = pollster::block_on(GoldilocksNttGpu::new_async(ntt_size))
                .expect("Failed to create NTT context");

            // Generate test polynomials with nonzero values
            let a: Vec<u64> = (0..size).map(|i| (i as u64 * 12345 + 67890) % p).collect();
            let b: Vec<u64> = (0..size).map(|i| (i as u64 * 98765 + 43210) % p).collect();

            // CPU multiplication
            let cpu_result = cpu_poly_mul(&a, &b, p);

            // GPU ASYNC multiplication
            let gpu_results = pollster::block_on(ntt.batched_poly_mul_async(&[(&a, &b)]))
                .expect("GPU async poly_mul failed");
            let gpu_result = &gpu_results[0];

            // Compare
            assert_eq!(cpu_result.len(), gpu_result.len(),
                "Length mismatch: CPU={}, GPU={}", cpu_result.len(), gpu_result.len());

            let mut mismatches = 0;
            for i in 0..cpu_result.len() {
                if cpu_result[i] != gpu_result[i] {
                    if mismatches < 5 {
                        println!("  Mismatch at [{}]: CPU={}, GPU={}", i, cpu_result[i], gpu_result[i]);
                    }
                    mismatches += 1;
                }
            }

            if mismatches > 0 {
                panic!("FAIL: {} mismatches out of {} coefficients", mismatches, cpu_result.len());
            }

            println!("  PASS: All {} coefficients match (async)", cpu_result.len());
        }
    }
}
