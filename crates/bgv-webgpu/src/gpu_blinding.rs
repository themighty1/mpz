//! GPU-accelerated blinding for BGV ciphertexts.
//!
//! This module provides GPU acceleration for the blinding operation in BGV:
//! 1. INTT of blinder slots in Goldilocks field
//! 2. Scale coefficients by delta_i = q_i / t for each RNS modulus
//! 3. Subtract from c0 component of ciphertext
//!
//! This replaces the CPU-bound blinding operation which was ~30% of total time.

use wgpu::{Buffer, BufferUsages, ComputePipeline, Device, Queue};
use wgpu::util::DeviceExt;

use crate::error::GpuError;
use crate::shader_math::compose_shader;
use crate::GoldilocksNttGpu;

/// Parameters for GPU blinding operation.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlindingParams {
    /// Ring dimension (number of coefficients per polynomial)
    n: u32,
    /// Number of RNS moduli
    k: u32,
    /// Number of polynomials to blind in this batch
    num_polys: u32,
    /// Padding for alignment
    _pad: u32,
}

/// GPU context for blinding operations.
///
/// Reuses the existing `GoldilocksNttGpu` for INTT and adds shaders for
/// scaling and subtraction.
pub struct GpuBlindingContext {
    device: Device,
    queue: Queue,
    n: usize,
    k: usize,

    // NTT context for INTT in Goldilocks field
    ntt_ctx: GoldilocksNttGpu,

    // Pipelines
    scale_pipeline: ComputePipeline,
    subtract_pipeline: ComputePipeline,

    // Precomputed delta values buffer (k moduli)
    delta_buffer: Buffer,

    // RNS moduli buffer (k moduli)
    moduli_buffer: Buffer,

    // Params buffer
    params_buffer: Buffer,

    // Preallocated working buffers
    blinder_coeffs_buffer: Buffer,   // After INTT: num_polys * n * 8 bytes
    scaled_blinders_buffer: Buffer,  // After scaling: num_polys * k * n * 8 bytes
}

// Shader for scaling coefficients by delta_i for each RNS modulus
const SCALE_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    k: u32,
    num_polys: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<vec2<u32>>;      // Goldilocks coefficients [num_polys * n]
@group(0) @binding(2) var<storage, read> deltas: array<vec2<u32>>;      // delta_i for each modulus [k]
@group(0) @binding(3) var<storage, read> moduli: array<vec2<u32>>;      // q_i for each modulus [k]
@group(0) @binding(4) var<storage, read_write> scaled: array<vec2<u32>>; // Output [num_polys * k * n]

// Scale coefficient by delta_i mod q_i
// Input: coefficient c in Goldilocks (may be up to 2^64 - 2^32)
// Output: (c % t) * delta_i mod q_i where t is Goldilocks
@compute @workgroup_size(256, 1, 1)
fn scale_coefficients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let k = params.k;
    let num_polys = params.num_polys;

    // Global index covers all (poly_idx, mod_idx, coeff_idx) combinations
    let total_work = num_polys * k * n;
    let idx = gid.x;
    if idx >= total_work { return; }

    // Decode indices
    let coeff_idx = idx % n;
    let mod_idx = (idx / n) % k;
    let poly_idx = idx / (n * k);

    // Get input coefficient (already reduced mod Goldilocks from INTT)
    let c = coeffs[poly_idx * n + coeff_idx];

    // Get delta and modulus for this RNS component
    let delta_i = deltas[mod_idx];
    let q_i = moduli[mod_idx];

    // Reduce c mod q_i first (c may be up to Goldilocks ~2^64, q_i is ~2^60)
    let c_reduced = math::reduce_mod_64(c, q_i);

    // Multiply by delta_i mod q_i
    let result = math::mulmod_60bit(c_reduced, delta_i, q_i);

    // Store in output buffer
    let out_idx = poly_idx * k * n + mod_idx * n + coeff_idx;
    scaled[out_idx] = result;
}
"#;

// Shader for subtracting scaled blinders from c0
const SUBTRACT_SHADER: &str = r#"
#import math

struct Params {
    n: u32,
    k: u32,
    num_polys: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> c0: array<vec2<u32>>;     // c0 coefficients [num_polys * k * n]
@group(0) @binding(2) var<storage, read> blinders: array<vec2<u32>>;     // Scaled blinders [num_polys * k * n]
@group(0) @binding(3) var<storage, read> moduli: array<vec2<u32>>;       // q_i for each modulus [k]

// Subtract blinder from c0: c0 = c0 - blinder mod q_i
@compute @workgroup_size(256, 1, 1)
fn subtract_blinders(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let k = params.k;
    let num_polys = params.num_polys;

    let total_work = num_polys * k * n;
    let idx = gid.x;
    if idx >= total_work { return; }

    // Decode mod_idx for getting the right modulus
    let coeff_idx = idx % n;
    let mod_idx = (idx / n) % k;

    let q_i = moduli[mod_idx];
    let c0_val = c0[idx];
    let blinder_val = blinders[idx];

    // c0 = c0 - blinder mod q_i
    c0[idx] = math::submod(c0_val, blinder_val, q_i);
}
"#;

impl GpuBlindingContext {
    /// Creates a new GPU blinding context.
    ///
    /// # Arguments
    /// * `n` - Ring dimension
    /// * `k` - Number of RNS moduli
    /// * `moduli` - The RNS moduli q_i
    /// * `t` - Plaintext modulus (Goldilocks)
    /// * `max_polys` - Maximum number of polynomials to blind in one batch
    pub async fn new(
        n: usize,
        k: usize,
        moduli: &[u64],
        t: u64,
        max_polys: usize,
    ) -> Result<Self, GpuError> {
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
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu_blinding_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceCreation)?;

        Self::from_device_queue(&device, &queue, n, k, moduli, t, max_polys)
    }

    /// Creates a blinding context from existing device and queue.
    ///
    /// # Arguments
    /// * `device` - Reference to existing wgpu Device (will be cloned)
    /// * `queue` - Reference to existing wgpu Queue (will be cloned)
    /// * `n` - Ring dimension
    /// * `k` - Number of RNS moduli
    /// * `moduli` - The RNS moduli q_i
    /// * `t` - Plaintext modulus (Goldilocks)
    /// * `max_polys` - Maximum number of polynomials to blind in one batch
    pub fn from_device_queue(
        device: &Device,
        queue: &Queue,
        n: usize,
        k: usize,
        moduli: &[u64],
        t: u64,
        max_polys: usize,
    ) -> Result<Self, GpuError> {
        let device = device.clone();
        let queue = queue.clone();
        assert_eq!(moduli.len(), k, "moduli length must equal k");

        // Create NTT context for INTT
        let ntt_ctx = GoldilocksNttGpu::from_device(&device, &queue, n)?;

        // Compute delta_i = q_i / t for each modulus
        let deltas: Vec<u64> = moduli.iter().map(|&q_i| q_i / t).collect();

        // Create delta buffer
        let delta_flat: Vec<u32> = deltas.iter()
            .flat_map(|&d| [d as u32, (d >> 32) as u32])
            .collect();
        let delta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("delta_buffer"),
            contents: bytemuck::cast_slice(&delta_flat),
            usage: BufferUsages::STORAGE,
        });

        // Create moduli buffer
        let moduli_flat: Vec<u32> = moduli.iter()
            .flat_map(|&q| [q as u32, (q >> 32) as u32])
            .collect();
        let moduli_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moduli_buffer"),
            contents: bytemuck::cast_slice(&moduli_flat),
            usage: BufferUsages::STORAGE,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blinding_params"),
            size: std::mem::size_of::<BlindingParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create working buffers
        let blinder_coeffs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blinder_coeffs"),
            size: (max_polys * n * 8) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scaled_blinders_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scaled_blinders"),
            size: (max_polys * k * n * 8) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Compile shaders
        let scale_shader = compose_shader(SCALE_SHADER, "scale_coefficients.wgsl")
            .map_err(|e| GpuError::ShaderCompilation(e))?;
        let scale_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale_shader"),
            source: wgpu::ShaderSource::Wgsl(scale_shader.into()),
        });

        let subtract_shader = compose_shader(SUBTRACT_SHADER, "subtract_blinders.wgsl")
            .map_err(|e| GpuError::ShaderCompilation(e))?;
        let subtract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subtract_shader"),
            source: wgpu::ShaderSource::Wgsl(subtract_shader.into()),
        });

        // Create pipelines
        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale_pipeline"),
            layout: None,
            module: &scale_module,
            entry_point: Some("scale_coefficients"),
            compilation_options: Default::default(),
            cache: None,
        });

        let subtract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("subtract_pipeline"),
            layout: None,
            module: &subtract_module,
            entry_point: Some("subtract_blinders"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            n,
            k,
            ntt_ctx,
            scale_pipeline,
            subtract_pipeline,
            delta_buffer,
            moduli_buffer,
            params_buffer,
            blinder_coeffs_buffer,
            scaled_blinders_buffer,
        })
    }

    /// Applies blinding to c0 coefficients on GPU.
    ///
    /// # Arguments
    /// * `c0_coeffs` - c0 coefficients in coefficient domain [num_polys][k][n]
    /// * `blinder_slots` - Blinder slot values in Goldilocks [num_polys][n]
    ///
    /// # Returns
    /// Blinded c0 coefficients [num_polys][k][n]
    #[cfg(target_arch = "wasm32")]
    pub async fn apply_blinding(
        &self,
        c0_coeffs: &[Vec<Vec<u64>>],
        blinder_slots: &[Vec<u64>],
    ) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let num_polys = c0_coeffs.len();
        assert_eq!(blinder_slots.len(), num_polys);

        if num_polys == 0 {
            return Ok(vec![]);
        }

        let n = self.n;
        let k = self.k;

        // Step 1: INTT blinder slots to get coefficients in Goldilocks
        // Use batched INTT for all polynomials at once
        let blinder_coeffs = self.ntt_ctx.batched_inverse_ntt_async(blinder_slots).await?;

        // Step 2: Upload blinder coefficients to GPU
        let blinder_flat: Vec<u32> = blinder_coeffs.iter()
            .flat_map(|poly| poly.iter().flat_map(|&c| [c as u32, (c >> 32) as u32]))
            .collect();
        self.queue.write_buffer(&self.blinder_coeffs_buffer, 0, bytemuck::cast_slice(&blinder_flat));

        // Step 3: Upload c0 coefficients to GPU
        let c0_flat: Vec<u32> = c0_coeffs.iter()
            .flat_map(|poly| poly.iter().flat_map(|residue|
                residue.iter().flat_map(|&c| [c as u32, (c >> 32) as u32])
            ))
            .collect();
        let c0_size = (num_polys * k * n * 8) as u64;
        let c0_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c0_buffer"),
            contents: bytemuck::cast_slice(&c0_flat),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Step 4: Update params
        let params = BlindingParams {
            n: n as u32,
            k: k as u32,
            num_polys: num_polys as u32,
            _pad: 0,
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Step 5: Run scale shader
        let scale_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bind_group"),
            layout: &self.scale_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.blinder_coeffs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.delta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.scaled_blinders_buffer.as_entire_binding() },
            ],
        });

        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scale_encoder"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &scale_bind_group, &[]);
            let total_work = (num_polys * k * n) as u32;
            let workgroups = (total_work + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 6: Run subtract shader
        let subtract_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subtract_bind_group"),
            layout: &self.subtract_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: c0_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scaled_blinders_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.moduli_buffer.as_entire_binding() },
            ],
        });

        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("subtract_encoder"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("subtract_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.subtract_pipeline);
            pass.set_bind_group(0, &subtract_bind_group, &[]);
            let total_work = (num_polys * k * n) as u32;
            let workgroups = (total_work + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 7: Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blinding_staging"),
            size: c0_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
            encoder.copy_buffer_to_buffer(&c0_buffer, 0, &staging_buffer, 0, c0_size);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let slice = staging_buffer.slice(..);
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

        // Reshape results
        let mut result = Vec::with_capacity(num_polys);
        for poly_idx in 0..num_polys {
            let mut poly_result = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let base = poly_idx * k * n + mod_idx * n;
                poly_result.push(data_u64[base..base + n].to_vec());
            }
            result.push(poly_result);
        }

        Ok(result)
    }

    /// Non-WASM version (synchronous).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_blinding(
        &self,
        c0_coeffs: &[Vec<Vec<u64>>],
        blinder_slots: &[Vec<u64>],
    ) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        pollster::block_on(self.apply_blinding_async(c0_coeffs, blinder_slots))
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn apply_blinding_async(
        &self,
        c0_coeffs: &[Vec<Vec<u64>>],
        blinder_slots: &[Vec<u64>],
    ) -> Result<Vec<Vec<Vec<u64>>>, GpuError> {
        let num_polys = c0_coeffs.len();
        assert_eq!(blinder_slots.len(), num_polys);

        if num_polys == 0 {
            return Ok(vec![]);
        }

        let n = self.n;
        let k = self.k;

        // Step 1: INTT blinder slots
        let blinder_coeffs = self.ntt_ctx.batched_inverse_ntt(blinder_slots)?;

        // Step 2: Upload blinder coefficients
        let blinder_flat: Vec<u32> = blinder_coeffs.iter()
            .flat_map(|poly| poly.iter().flat_map(|&c| [c as u32, (c >> 32) as u32]))
            .collect();
        self.queue.write_buffer(&self.blinder_coeffs_buffer, 0, bytemuck::cast_slice(&blinder_flat));

        // Step 3: Upload c0
        let c0_flat: Vec<u32> = c0_coeffs.iter()
            .flat_map(|poly| poly.iter().flat_map(|residue|
                residue.iter().flat_map(|&c| [c as u32, (c >> 32) as u32])
            ))
            .collect();
        let c0_size = (num_polys * k * n * 8) as u64;
        let c0_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("c0_buffer"),
            contents: bytemuck::cast_slice(&c0_flat),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Step 4: Update params
        let params = BlindingParams {
            n: n as u32,
            k: k as u32,
            num_polys: num_polys as u32,
            _pad: 0,
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Step 5 & 6: Run scale and subtract shaders
        let scale_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale_bind_group"),
            layout: &self.scale_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.blinder_coeffs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.delta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.scaled_blinders_buffer.as_entire_binding() },
            ],
        });

        let subtract_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subtract_bind_group"),
            layout: &self.subtract_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: c0_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scaled_blinders_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.moduli_buffer.as_entire_binding() },
            ],
        });

        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blinding_encoder"),
            });

            // Scale pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("scale_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &scale_bind_group, &[]);
                let total_work = (num_polys * k * n) as u32;
                let workgroups = (total_work + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Subtract pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("subtract_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.subtract_pipeline);
                pass.set_bind_group(0, &subtract_bind_group, &[]);
                let total_work = (num_polys * k * n) as u32;
                let workgroups = (total_work + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 7: Read back
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blinding_staging"),
            size: c0_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
            encoder.copy_buffer_to_buffer(&c0_buffer, 0, &staging_buffer, 0, c0_size);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        let data_u64: &[u64] = bytemuck::cast_slice(&data);

        // Reshape
        let mut result = Vec::with_capacity(num_polys);
        for poly_idx in 0..num_polys {
            let mut poly_result = Vec::with_capacity(k);
            for mod_idx in 0..k {
                let base = poly_idx * k * n + mod_idx * n;
                poly_result.push(data_u64[base..base + n].to_vec());
            }
            result.push(poly_result);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blinding_params_size() {
        assert_eq!(std::mem::size_of::<BlindingParams>(), 16);
    }
}
