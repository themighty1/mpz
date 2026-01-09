//! GPU context and operations for BGV ciphertext manipulation.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::{error::GpuError, shader, BatchParams};

/// Uniform buffer parameters for the scalar multiplication shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    n: u32,
    num_batches: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU context for WebGPU operations.
pub struct GpuContext {
    device: Device,
    queue: Queue,
    scalar_mul_pipeline: ComputePipeline,
}

impl GpuContext {
    /// Creates a new GPU context.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
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
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bgv-webgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await?;

        // Compile scalar multiplication shader
        let scalar_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scalar_mul shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::SCALAR_MUL_SHADER)),
        });

        let scalar_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scalar_mul pipeline"),
            layout: None,
            module: &scalar_mul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            scalar_mul_pipeline,
        })
    }

    /// Returns a reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

/// A BGV ciphertext stored on the GPU.
pub struct GpuCiphertext {
    c0_buffer: Buffer,
    c1_buffer: Buffer,
    n: usize,
}

impl GpuCiphertext {
    /// Creates a GPU ciphertext from coefficient slices.
    pub fn from_coeffs(ctx: &GpuContext, c0: &[u64], c1: &[u64]) -> Result<Self, GpuError> {
        if c0.len() != c1.len() {
            return Err(GpuError::InvalidParams(
                "c0 and c1 must have same length".into(),
            ));
        }

        let n = c0.len();

        let c0_u32: Vec<u32> = c0.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();
        let c1_u32: Vec<u32> = c1.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]).collect();

        let c0_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ciphertext c0"),
            contents: bytemuck::cast_slice(&c0_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let c1_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ciphertext c1"),
            contents: bytemuck::cast_slice(&c1_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        Ok(Self { c0_buffer, c1_buffer, n })
    }

    /// Returns the ring dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }
}

/// Slot-wise scalar multiplication.
///
/// Computes: output[batch][slot] = ct[slot] * scalars[batch][slot] mod q
pub struct SlotWiseMul {
    params: BatchParams,
    params_buffer: Buffer,
    out_c0_buffer: Buffer,
    out_c1_buffer: Buffer,
    staging_buffer: Buffer,
}

impl SlotWiseMul {
    /// Creates a new slot-wise multiplication context.
    ///
    /// - `n`: Ring dimension (8192 for Goldilocks)
    /// - `num_batches`: Number of output ciphertexts (80)
    pub fn new(ctx: &GpuContext, params: BatchParams) -> Result<Self, GpuError> {
        let output_size = params.num_batches * params.n * 2; // u32 count

        let out_c0_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output c0"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_c1_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output c1"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: (output_size * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_params = ShaderParams {
            n: params.n as u32,
            num_batches: params.num_batches as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shader params"),
            contents: bytemuck::bytes_of(&shader_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Ok(Self {
            params,
            params_buffer,
            out_c0_buffer,
            out_c1_buffer,
            staging_buffer,
        })
    }

    /// Runs slot-wise multiplication.
    ///
    /// - `ct`: Input ciphertext (n slots)
    /// - `scalars`: num_batches × n scalars (flattened row-major)
    ///
    /// Returns: (c0_outputs, c1_outputs) where each is num_batches vectors of n u64s.
    pub fn run(
        &self,
        ctx: &GpuContext,
        ct: &GpuCiphertext,
        scalars: &[u64],
    ) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), GpuError> {
        let expected_scalars = self.params.num_batches * self.params.n;
        if scalars.len() != expected_scalars {
            return Err(GpuError::BufferSizeMismatch {
                expected: expected_scalars,
                actual: scalars.len(),
            });
        }

        if ct.dimension() != self.params.n {
            return Err(GpuError::InvalidParams(format!(
                "Ciphertext dimension {} doesn't match expected {}",
                ct.dimension(),
                self.params.n
            )));
        }

        // Upload scalars
        let scalars_u32: Vec<u32> = scalars
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let scalars_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalars buffer"),
            contents: bytemuck::cast_slice(&scalars_u32),
            usage: BufferUsages::STORAGE,
        });

        // Create bind group
        let bind_group_layout = ctx.scalar_mul_pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scalar mul bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ct.c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ct.c1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.out_c0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.out_c1_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let total_threads = (self.params.num_batches * self.params.n) as u32;
        let workgroups = (total_threads + 255) / 256;

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scalar mul encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scalar mul pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.scalar_mul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        // Read back results
        self.read_results(ctx)
    }

    fn read_results(&self, ctx: &GpuContext) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), GpuError> {
        let n = self.params.n;
        let num_batches = self.params.num_batches;
        let output_u32_count = num_batches * n * 2;
        let buffer_size = (output_u32_count * std::mem::size_of::<u32>()) as u64;

        let c0_results = self.read_buffer(ctx, &self.out_c0_buffer, buffer_size)?;
        let c1_results = self.read_buffer(ctx, &self.out_c1_buffer, buffer_size)?;

        let c0_vecs: Vec<Vec<u64>> = (0..num_batches)
            .map(|batch| {
                (0..n)
                    .map(|slot| {
                        let idx = (batch * n + slot) * 2;
                        (c0_results[idx] as u64) | ((c0_results[idx + 1] as u64) << 32)
                    })
                    .collect()
            })
            .collect();

        let c1_vecs: Vec<Vec<u64>> = (0..num_batches)
            .map(|batch| {
                (0..n)
                    .map(|slot| {
                        let idx = (batch * n + slot) * 2;
                        (c1_results[idx] as u64) | ((c1_results[idx + 1] as u64) << 32)
                    })
                    .collect()
            })
            .collect();

        Ok((c0_vecs, c1_vecs))
    }

    fn read_buffer(&self, ctx: &GpuContext, buffer: &Buffer, size: u64) -> Result<Vec<u32>, GpuError> {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read buffer encoder"),
        });

        encoder.copy_buffer_to_buffer(buffer, 0, &self.staging_buffer, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..size);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        ctx.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(e.to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GOLDILOCKS_Q;

    #[test]
    fn test_gpu_context_creation() {
        match GpuContext::new() {
            Ok(_) => println!("GPU context created successfully"),
            Err(e) => println!("GPU not available: {}", e),
        }
    }

    #[test]
    fn test_print_gpu_info() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        match adapter {
            Some(adapter) => {
                let info = adapter.get_info();
                println!("\n=== GPU Adapter Info ===");
                println!("  Name: {}", info.name);
                println!("  Vendor: 0x{:x}", info.vendor);
                println!("  Device: 0x{:x}", info.device);
                println!("  Device Type: {:?}", info.device_type);
                println!("  Backend: {:?}", info.backend);
                println!("========================\n");
            }
            None => println!("No GPU adapter found"),
        }
    }

    #[test]
    fn test_slot_wise_mul() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 256;
        let num_batches = 4;

        let params = BatchParams::new(n, GOLDILOCKS_Q, num_batches);

        // Create test ciphertext: all slots = 2
        let ct_c0: Vec<u64> = vec![2; n];
        let ct_c1: Vec<u64> = vec![3; n];
        let ct = GpuCiphertext::from_coeffs(&ctx, &ct_c0, &ct_c1).unwrap();

        // Scalars: batch i, slot j = i + 1
        let scalars: Vec<u64> = (0..num_batches)
            .flat_map(|batch| vec![(batch + 1) as u64; n])
            .collect();

        let mul = SlotWiseMul::new(&ctx, params).unwrap();
        let (c0_results, c1_results) = mul.run(&ctx, &ct, &scalars).unwrap();

        assert_eq!(c0_results.len(), num_batches);
        assert_eq!(c0_results[0].len(), n);

        // Check: batch 0 should be 2 * 1 = 2, batch 1 should be 2 * 2 = 4, etc.
        assert_eq!(c0_results[0][0], 2, "batch 0: 2 * 1 = 2");
        assert_eq!(c0_results[1][0], 4, "batch 1: 2 * 2 = 4");
        assert_eq!(c0_results[2][0], 6, "batch 2: 2 * 3 = 6");
        assert_eq!(c0_results[3][0], 8, "batch 3: 2 * 4 = 8");

        // c1: 3 * scalar
        assert_eq!(c1_results[0][0], 3, "batch 0: 3 * 1 = 3");
        assert_eq!(c1_results[1][0], 6, "batch 1: 3 * 2 = 6");
        assert_eq!(c1_results[2][0], 9, "batch 2: 3 * 3 = 9");
        assert_eq!(c1_results[3][0], 12, "batch 3: 3 * 4 = 12");
    }

    #[test]
    fn test_goldilocks_reduction() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 4;
        let num_batches = 1;
        let q = GOLDILOCKS_Q;

        let params = BatchParams::new(n, q, num_batches);

        // Test with values near q
        let ct_c0: Vec<u64> = vec![q - 1, q - 2, 1000, 0];
        let ct_c1: Vec<u64> = vec![q - 1, q - 2, 1000, 0];
        let ct = GpuCiphertext::from_coeffs(&ctx, &ct_c0, &ct_c1).unwrap();

        // Multiply by 2
        let scalars: Vec<u64> = vec![2; n];

        let mul = SlotWiseMul::new(&ctx, params).unwrap();
        let (c0_results, _) = mul.run(&ctx, &ct, &scalars).unwrap();

        // Expected: 2 * (q-1) mod q = q - 2
        assert_eq!(c0_results[0][0], q - 2, "2*(q-1) mod q should be q-2");
        assert_eq!(c0_results[0][1], q - 4, "2*(q-2) mod q should be q-4");
        assert_eq!(c0_results[0][2], 2000, "2*1000 should be 2000");
        assert_eq!(c0_results[0][3], 0, "2*0 should be 0");
    }
}
