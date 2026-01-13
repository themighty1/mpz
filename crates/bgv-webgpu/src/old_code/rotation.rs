//! GPU-accelerated rotation and slot summation for RNS BGV ciphertexts.
//!
//! This module provides GPU implementations of:
//! - Automorphism (σ_k): Permutes polynomial coefficients
//! - NTT-based polynomial multiplication
//! - Key-switching with HYBRID decomposition
//! - sum_slots: Sums all slots using log(n) rotations

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, BufferUsages, ComputePipeline, Device, Queue};

use crate::error::GpuError;
use super::rotation_shader::{
    BIT_REVERSE_SHADER, DIGIT_DECOMPOSE_SHADER, NTT_BUTTERFLY_SHADER, POINTWISE_MUL_SHADER,
    SCALE_SHADER, TWIST_SHADER,
};
// New modular fused shaders with shared math (composed via naga_oil)
use super::rotation_shader::{fused, create_shader_module};

/// Parameters for RNS BGV on GPU.
#[derive(Clone, Debug)]
pub struct GpuRnsParams {
    /// Ring dimension (8192).
    pub n: usize,
    /// RNS moduli.
    pub moduli: Vec<u64>,
    /// Number of digits per RNS limb for key-switching.
    pub digits_per_limb: usize,
    /// Decomposition base (2^15).
    pub decomp_base: u64,
}

impl GpuRnsParams {
    /// Creates parameters for Goldilocks field.
    ///
    /// Uses 5 RNS moduli (~300 bit q) for sufficient noise budget.
    /// This matches the CPU `RnsBgvParams::goldilocks()` parameters.
    ///
    /// The noise budget must support:
    /// 1. Slot-wise scalar multiplication
    /// 2. sum_slots: 13 rotations with key-switching (log2(8192) = 13)
    /// 3. Masking: plaintext multiplication to zero out all slots except one
    /// 4. CT additions (~80 for wasm zkVM)
    ///
    /// Each rotation adds significant noise due to key-switching, so 5 moduli
    /// provides sufficient margin for correctness.
    pub fn goldilocks() -> Self {
        Self {
            n: 8192,
            // Same primes as CPU RnsParams::NTT_PRIMES_8192[0..5]
            moduli: vec![
                1152921504606994433,  // q₀ ≡ 1 (mod 16384)
                1152921504607191041,  // q₁ ≡ 1 (mod 16384)
                1152921504607223809,  // q₂ ≡ 1 (mod 16384)
                1152921504607338497,  // q₃ ≡ 1 (mod 16384)
                1152921504607518721,  // q₄ ≡ 1 (mod 16384)
            ],
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        }
    }
}

/// Precomputed NTT data for a single modulus.
#[derive(Clone, Debug)]
pub struct NttModulusData {
    /// The modulus q.
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
    /// Barrett mu = floor(2^128 / q) (low 64 bits).
    pub mu_lo: u64,
    /// Barrett mu (high 64 bits).
    pub mu_hi: u64,
    /// Powers of psi: [psi^0, psi^1, ..., psi^(n-1)].
    pub psi_powers: Vec<u64>,
    /// Powers of psi_inv: [psi_inv^0, psi_inv^1, ..., psi_inv^(n-1)].
    pub psi_inv_powers: Vec<u64>,
    /// Twiddle factors for forward NTT: [omega^0, omega^1, ..., omega^(n-1)].
    pub twiddles: Vec<u64>,
    /// Twiddle factors for inverse NTT.
    pub inv_twiddles: Vec<u64>,
}

impl NttModulusData {
    /// Creates NTT data for a single modulus.
    pub fn new(n: usize, modulus: u64, psi: u64) -> Self {
        let omega = mod_mul(psi, psi, modulus); // omega = psi^2
        let psi_inv = mod_inverse(psi, modulus);
        let omega_inv = mod_inverse(omega, modulus);
        let n_inv = mod_inverse(n as u64, modulus);

        // Compute Barrett mu
        let mu = u128::MAX / (modulus as u128);
        let mu_lo = mu as u64;
        let mu_hi = (mu >> 64) as u64;

        // Precompute powers
        let psi_powers = compute_powers(psi, n, modulus);
        let psi_inv_powers = compute_powers(psi_inv, n, modulus);
        let twiddles = compute_powers(omega, n, modulus);
        let inv_twiddles = compute_powers(omega_inv, n, modulus);

        Self {
            modulus,
            psi,
            omega,
            psi_inv,
            omega_inv,
            n_inv,
            mu_lo,
            mu_hi,
            psi_powers,
            psi_inv_powers,
            twiddles,
            inv_twiddles,
        }
    }
}

/// Computes [base^0, base^1, ..., base^(n-1)] mod modulus.
fn compute_powers(base: u64, n: usize, modulus: u64) -> Vec<u64> {
    let mut powers = Vec::with_capacity(n);
    let mut current = 1u64;
    for _ in 0..n {
        powers.push(current);
        current = mod_mul(current, base, modulus);
    }
    powers
}

/// Modular multiplication.
#[inline]
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation.
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

/// Modular inverse using extended GCD.
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

/// Finds a primitive 2n-th root of unity modulo q.
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

/// Uniform parameters for automorphism shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AutoParams {
    n: u32,
    two_n: u32,
    k: u32,
    num_moduli: u32,
}

/// Uniform parameters for addition shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for bit-reversal permutation shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BitRevParams {
    n: u32,
    log_n: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for NTT butterfly shader.
/// Note: modulus and mu are passed as pairs of u32s representing u64 values.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NttButterflyParams {
    n: u32,
    stage: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo_lo: u32,   // Low 32 bits of mu_lo (which is low 64 bits of 128-bit mu)
    mu_lo_hi: u32,   // High 32 bits of mu_lo
    mu_hi_lo: u32,   // Low 32 bits of mu_hi (which is high 64 bits of 128-bit mu)
    mu_hi_hi: u32,   // High 32 bits of mu_hi
}

/// Parameters for pointwise multiplication shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointwiseParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    _pad0: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
}

/// Parameters for scale shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    scalar_lo: u32,
    scalar_hi: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for twist shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TwistParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    _pad0: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
}

/// Parameters for digit decomposition shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecomposeParams {
    n: u32,
    digit_idx: u32,
    decomp_base_log: u32,
    _pad: u32,
}

/// Parameters for fused twist shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedTwistParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for fused bit-reverse shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedBitrevParams {
    n: u32,
    log_n: u32,
    num_moduli: u32,
    _pad: u32,
}

/// Parameters for fused butterfly shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedButterflyParams {
    n: u32,
    stage: u32,
    num_moduli: u32,
    _pad: u32,
}

/// Parameters for fused pointwise shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedPointwiseParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for fused scale shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusedScaleParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for shared memory NTT shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SharedMemNttParams {
    n: u32,
    log_n: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU context for rotation operations with NTT support.
pub struct GpuRotationContext {
    device: Device,
    queue: Queue,
    params: GpuRnsParams,
    // Existing pipelines
    auto_pipeline: ComputePipeline,
    add_pipeline: ComputePipeline,
    moduli_buffer: Buffer,
    // NTT pipelines
    bit_rev_pipeline: ComputePipeline,
    ntt_butterfly_pipeline: ComputePipeline,
    pointwise_mul_pipeline: ComputePipeline,
    scale_pipeline: ComputePipeline,
    twist_pipeline: ComputePipeline,
    // Key-switching pipeline
    digit_decompose_pipeline: ComputePipeline,
    // Precomputed NTT data per modulus
    ntt_data: Vec<NttModulusData>,
    // GPU buffers for twiddle factors (one per modulus)
    twiddle_buffers: Vec<Buffer>,
    inv_twiddle_buffers: Vec<Buffer>,
    psi_power_buffers: Vec<Buffer>,
    psi_inv_power_buffers: Vec<Buffer>,
    // Fused pipelines (process all moduli in single dispatch)
    fused_twist_pipeline: ComputePipeline,
    fused_bitrev_pipeline: ComputePipeline,
    fused_butterfly_pipeline: ComputePipeline,
    fused_pointwise_pipeline: ComputePipeline,
    fused_scale_pipeline: ComputePipeline,
    // Combined buffers for fused operations
    all_moduli_buffer: Buffer,        // [mod*2]: q_lo, q_hi for each modulus
    all_barrett_buffer: Buffer,       // [mod*4]: mu_lo_lo, mu_lo_hi, mu_hi_lo, mu_hi_hi
    all_psi_powers_buffer: Buffer,    // [mod*n*2]: all psi powers concatenated
    all_psi_inv_powers_buffer: Buffer,
    all_twiddles_buffer: Buffer,      // [mod*n*2]: all twiddles concatenated
    all_inv_twiddles_buffer: Buffer,
    all_n_inv_buffer: Buffer,         // [mod*2]: n_inv for each modulus
    // Shared memory NTT (all stages in one dispatch)
    shared_mem_ntt_fwd_pipeline: ComputePipeline,
    shared_mem_ntt_inv_pipeline: ComputePipeline,
    max_shared_mem: u32,              // GPU's max workgroup storage size
}

impl GpuRotationContext {
    /// Creates a new GPU rotation context (blocks on async).
    /// Use `new_async` for WASM environments.
    pub fn new(params: GpuRnsParams) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(params))
    }

    /// Creates a new GPU rotation context asynchronously.
    /// Required for WASM where blocking is not allowed.
    pub async fn new_async(params: GpuRnsParams) -> Result<Self, GpuError> {
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

        // Request higher limits for shared memory NTT (1024 threads per workgroup)
        let mut limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x.min(1024);
        limits.max_compute_invocations_per_workgroup = adapter_limits.max_compute_invocations_per_workgroup.min(1024);
        limits.max_compute_workgroup_storage_size = adapter_limits.max_compute_workgroup_storage_size;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rotation device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        // Compile automorphism shader
        let auto_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("automorphism shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(AUTOMORPHISM_SHADER)),
        });

        let auto_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("automorphism pipeline"),
            layout: None,
            module: &auto_shader,
            entry_point: Some("apply_automorphism"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Compile addition shader
        let add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADD_SHADER)),
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add pipeline"),
            layout: None,
            module: &add_shader,
            entry_point: Some("add_polys"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Compile NTT shaders
        let bit_rev_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bit reverse shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BIT_REVERSE_SHADER)),
        });
        let bit_rev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bit reverse pipeline"),
            layout: None,
            module: &bit_rev_shader,
            entry_point: Some("bit_reverse_permutation"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ntt_butterfly_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ntt butterfly shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(NTT_BUTTERFLY_SHADER)),
        });
        let ntt_butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ntt butterfly pipeline"),
            layout: None,
            module: &ntt_butterfly_shader,
            entry_point: Some("ntt_butterfly"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pointwise_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointwise mul shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(POINTWISE_MUL_SHADER)),
        });
        let pointwise_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pointwise mul pipeline"),
            layout: None,
            module: &pointwise_mul_shader,
            entry_point: Some("pointwise_mul"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scale_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCALE_SHADER)),
        });
        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale pipeline"),
            layout: None,
            module: &scale_shader,
            entry_point: Some("scale"),
            compilation_options: Default::default(),
            cache: None,
        });

        let twist_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twist shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TWIST_SHADER)),
        });
        let twist_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("twist pipeline"),
            layout: None,
            module: &twist_shader,
            entry_point: Some("twist"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Compile digit decomposition shader for key-switching
        let digit_decompose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("digit decompose shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DIGIT_DECOMPOSE_SHADER)),
        });
        let digit_decompose_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("digit decompose pipeline"),
            layout: None,
            module: &digit_decompose_shader,
            entry_point: Some("digit_decompose"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Compile fused shaders (process all moduli in single dispatch)
        // Use naga_oil composer to import shared math module
        let fused_twist_shader = create_shader_module(&device, fused::FUSED_TWIST_SHADER, "fused_twist.wgsl")
            .expect("Failed to compose fused twist shader");
        let fused_twist_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused twist pipeline"),
            layout: None,
            module: &fused_twist_shader,
            entry_point: Some("fused_twist"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fused_bitrev_shader = create_shader_module(&device, fused::FUSED_BITREV_SHADER, "fused_bitrev.wgsl")
            .expect("Failed to compose fused bitrev shader");
        let fused_bitrev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused bitrev pipeline"),
            layout: None,
            module: &fused_bitrev_shader,
            entry_point: Some("fused_bit_reverse"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fused_butterfly_shader = create_shader_module(&device, fused::FUSED_BUTTERFLY_SHADER, "fused_butterfly.wgsl")
            .expect("Failed to compose fused butterfly shader");
        let fused_butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused butterfly pipeline"),
            layout: None,
            module: &fused_butterfly_shader,
            entry_point: Some("fused_butterfly"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fused_pointwise_shader = create_shader_module(&device, fused::FUSED_POINTWISE_SHADER, "fused_pointwise.wgsl")
            .expect("Failed to compose fused pointwise shader");
        let fused_pointwise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused pointwise pipeline"),
            layout: None,
            module: &fused_pointwise_shader,
            entry_point: Some("fused_pointwise"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fused_scale_shader = create_shader_module(&device, fused::FUSED_SCALE_SHADER, "fused_scale.wgsl")
            .expect("Failed to compose fused scale shader");
        let fused_scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused scale pipeline"),
            layout: None,
            module: &fused_scale_shader,
            entry_point: Some("fused_scale"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Shared memory NTT - all stages in one dispatch
        use super::rotation_shader::shared_mem_ntt;
        let shared_mem_ntt_fwd_shader = create_shader_module(&device, shared_mem_ntt::SHARED_MEM_NTT_FWD_SHADER, "shared_mem_ntt_fwd.wgsl")
            .expect("Failed to compose shared memory forward NTT shader");
        let shared_mem_ntt_fwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shared mem ntt fwd pipeline"),
            layout: None,
            module: &shared_mem_ntt_fwd_shader,
            entry_point: Some("shared_mem_ntt_fwd"),
            compilation_options: Default::default(),
            cache: None,
        });

        let shared_mem_ntt_inv_shader = create_shader_module(&device, shared_mem_ntt::SHARED_MEM_NTT_INV_SHADER, "shared_mem_ntt_inv.wgsl")
            .expect("Failed to compose shared memory inverse NTT shader");
        let shared_mem_ntt_inv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shared mem ntt inv pipeline"),
            layout: None,
            module: &shared_mem_ntt_inv_shader,
            entry_point: Some("shared_mem_ntt_inv"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Query GPU shared memory limit for adaptive strategy
        let max_shared_mem = device.limits().max_compute_workgroup_storage_size;

        // Upload moduli
        let moduli_u32: Vec<u32> = params
            .moduli
            .iter()
            .flat_map(|&m| [m as u32, (m >> 32) as u32])
            .collect();

        let moduli_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moduli buffer"),
            contents: bytemuck::cast_slice(&moduli_u32),
            usage: BufferUsages::STORAGE,
        });

        // Precompute NTT data for each modulus
        let n = params.n;
        let mut ntt_data = Vec::with_capacity(params.moduli.len());
        let mut twiddle_buffers = Vec::with_capacity(params.moduli.len());
        let mut inv_twiddle_buffers = Vec::with_capacity(params.moduli.len());
        let mut psi_power_buffers = Vec::with_capacity(params.moduli.len());
        let mut psi_inv_power_buffers = Vec::with_capacity(params.moduli.len());

        for &modulus in &params.moduli {
            // Find primitive 2n-th root of unity
            let psi = find_primitive_root(n, modulus)
                .ok_or_else(|| GpuError::InvalidParams(format!(
                    "No primitive 2n-th root of unity for modulus {}",
                    modulus
                )))?;

            let data = NttModulusData::new(n, modulus, psi);

            // Upload twiddle factors to GPU
            let twiddles_u32: Vec<u32> = data.twiddles.iter()
                .flat_map(|&t| [t as u32, (t >> 32) as u32])
                .collect();
            let twiddle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("twiddles"),
                contents: bytemuck::cast_slice(&twiddles_u32),
                usage: BufferUsages::STORAGE,
            });

            let inv_twiddles_u32: Vec<u32> = data.inv_twiddles.iter()
                .flat_map(|&t| [t as u32, (t >> 32) as u32])
                .collect();
            let inv_twiddle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inv_twiddles"),
                contents: bytemuck::cast_slice(&inv_twiddles_u32),
                usage: BufferUsages::STORAGE,
            });

            let psi_powers_u32: Vec<u32> = data.psi_powers.iter()
                .flat_map(|&p| [p as u32, (p >> 32) as u32])
                .collect();
            let psi_power_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("psi_powers"),
                contents: bytemuck::cast_slice(&psi_powers_u32),
                usage: BufferUsages::STORAGE,
            });

            let psi_inv_powers_u32: Vec<u32> = data.psi_inv_powers.iter()
                .flat_map(|&p| [p as u32, (p >> 32) as u32])
                .collect();
            let psi_inv_power_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("psi_inv_powers"),
                contents: bytemuck::cast_slice(&psi_inv_powers_u32),
                usage: BufferUsages::STORAGE,
            });

            twiddle_buffers.push(twiddle_buf);
            inv_twiddle_buffers.push(inv_twiddle_buf);
            psi_power_buffers.push(psi_power_buf);
            psi_inv_power_buffers.push(psi_inv_power_buf);
            ntt_data.push(data);
        }

        // Create combined buffers for fused operations
        // all_moduli_buffer: [mod*2] = q_lo, q_hi for each modulus
        let all_moduli_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| [d.modulus as u32, (d.modulus >> 32) as u32])
            .collect();
        let all_moduli_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_moduli"),
            contents: bytemuck::cast_slice(&all_moduli_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_barrett_buffer: [mod*4] = mu_lo_lo, mu_lo_hi, mu_hi_lo, mu_hi_hi
        let all_barrett_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| [
                d.mu_lo as u32, (d.mu_lo >> 32) as u32,
                d.mu_hi as u32, (d.mu_hi >> 32) as u32,
            ])
            .collect();
        let all_barrett_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_barrett"),
            contents: bytemuck::cast_slice(&all_barrett_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_psi_powers_buffer: [mod*n*2] - all psi powers concatenated
        let all_psi_powers_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| d.psi_powers.iter().flat_map(|&p| [p as u32, (p >> 32) as u32]))
            .collect();
        let all_psi_powers_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_psi_powers"),
            contents: bytemuck::cast_slice(&all_psi_powers_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_psi_inv_powers_buffer
        let all_psi_inv_powers_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| d.psi_inv_powers.iter().flat_map(|&p| [p as u32, (p >> 32) as u32]))
            .collect();
        let all_psi_inv_powers_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_psi_inv_powers"),
            contents: bytemuck::cast_slice(&all_psi_inv_powers_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_twiddles_buffer: [mod*n*2] - all twiddles concatenated
        let all_twiddles_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| d.twiddles.iter().flat_map(|&t| [t as u32, (t >> 32) as u32]))
            .collect();
        let all_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_twiddles"),
            contents: bytemuck::cast_slice(&all_twiddles_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_inv_twiddles_buffer
        let all_inv_twiddles_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| d.inv_twiddles.iter().flat_map(|&t| [t as u32, (t >> 32) as u32]))
            .collect();
        let all_inv_twiddles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_inv_twiddles"),
            contents: bytemuck::cast_slice(&all_inv_twiddles_u32),
            usage: BufferUsages::STORAGE,
        });

        // all_n_inv_buffer: [mod*2] = n_inv for each modulus
        let all_n_inv_u32: Vec<u32> = ntt_data.iter()
            .flat_map(|d| [d.n_inv as u32, (d.n_inv >> 32) as u32])
            .collect();
        let all_n_inv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_n_inv"),
            contents: bytemuck::cast_slice(&all_n_inv_u32),
            usage: BufferUsages::STORAGE,
        });

        Ok(Self {
            device,
            queue,
            params,
            auto_pipeline,
            add_pipeline,
            moduli_buffer,
            bit_rev_pipeline,
            ntt_butterfly_pipeline,
            pointwise_mul_pipeline,
            scale_pipeline,
            twist_pipeline,
            digit_decompose_pipeline,
            ntt_data,
            twiddle_buffers,
            inv_twiddle_buffers,
            psi_power_buffers,
            psi_inv_power_buffers,
            // Fused pipelines
            fused_twist_pipeline,
            fused_bitrev_pipeline,
            fused_butterfly_pipeline,
            fused_pointwise_pipeline,
            fused_scale_pipeline,
            // Combined buffers
            all_moduli_buffer,
            all_barrett_buffer,
            all_psi_powers_buffer,
            all_psi_inv_powers_buffer,
            all_twiddles_buffer,
            all_inv_twiddles_buffer,
            all_n_inv_buffer,
            // Shared memory NTT
            shared_mem_ntt_fwd_pipeline,
            shared_mem_ntt_inv_pipeline,
            max_shared_mem,
        })
    }

    /// Returns the NTT data for a specific modulus.
    pub fn ntt_data(&self, mod_idx: usize) -> &NttModulusData {
        &self.ntt_data[mod_idx]
    }

    /// Returns the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the parameters.
    pub fn params(&self) -> &GpuRnsParams {
        &self.params
    }
}

/// An RNS polynomial stored on GPU.
/// Layout: [modulus_0_coeffs..., modulus_1_coeffs..., ...]
/// Each coefficient is stored as two u32s (lo, hi).
pub struct GpuRnsPoly {
    buffer: Buffer,
    n: usize,
    num_moduli: usize,
}

impl GpuRnsPoly {
    /// Creates a GPU polynomial from RNS residues.
    /// `residues[mod_idx][coeff_idx]` is the coefficient.
    pub fn from_residues(
        ctx: &GpuRotationContext,
        residues: &[Vec<u64>],
    ) -> Result<Self, GpuError> {
        let num_moduli = residues.len();
        if num_moduli == 0 {
            return Err(GpuError::InvalidParams("empty residues".into()));
        }

        let n = residues[0].len();
        for r in residues {
            if r.len() != n {
                return Err(GpuError::InvalidParams("inconsistent residue lengths".into()));
            }
        }

        // Flatten to u32 array
        let data: Vec<u32> = residues
            .iter()
            .flat_map(|r| r.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]))
            .collect();

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rns poly"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        Ok(Self {
            buffer,
            n,
            num_moduli,
        })
    }

    /// Creates an uninitialized GPU polynomial.
    pub fn new_uninit(ctx: &GpuRotationContext, n: usize, num_moduli: usize) -> Self {
        let size = n * num_moduli * 2 * std::mem::size_of::<u32>();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rns poly uninit"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            n,
            num_moduli,
        }
    }

    /// Reads the polynomial back to CPU.
    pub fn to_residues(&self, ctx: &GpuRotationContext) -> Result<Vec<Vec<u64>>, GpuError> {
        let size = (self.n * self.num_moduli * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ExecutionFailed(e.to_string()))?
            .map_err(|e| GpuError::ExecutionFailed(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let u32_data: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // Convert to residues
        let mut residues = vec![vec![0u64; self.n]; self.num_moduli];
        for mod_idx in 0..self.num_moduli {
            for coeff_idx in 0..self.n {
                let base = (mod_idx * self.n + coeff_idx) * 2;
                residues[mod_idx][coeff_idx] =
                    (u32_data[base] as u64) | ((u32_data[base + 1] as u64) << 32);
            }
        }

        Ok(residues)
    }

    /// Reads the polynomial back to CPU asynchronously.
    /// Required for WASM where blocking is not allowed.
    pub async fn to_residues_async(&self, ctx: &GpuRotationContext) -> Result<Vec<Vec<u64>>, GpuError> {
        let size = (self.n * self.num_moduli * 2 * std::mem::size_of::<u32>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_async"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read encoder async"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);

        #[cfg(target_arch = "wasm32")]
        {
            // For WASM: use a JS Promise to await the map_async callback
            use wasm_bindgen::prelude::*;
            use std::cell::RefCell;
            use std::rc::Rc;

            let result: Rc<RefCell<Option<Result<(), wgpu::BufferAsyncError>>>> = Rc::new(RefCell::new(None));
            let result_clone = result.clone();

            // Create a JS Promise that resolves when the buffer is mapped
            let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                let resolve = Rc::new(resolve);
                let resolve_clone = resolve.clone();
                let result_inner = result_clone.clone();

                buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                    *result_inner.borrow_mut() = Some(r);
                    resolve_clone.call0(&JsValue::NULL).ok();
                });
            });

            // Await the promise
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| GpuError::ExecutionFailed(format!("JS await failed: {:?}", e)))?;

            // Check the result
            let map_result = result.borrow_mut().take()
                .ok_or_else(|| GpuError::ExecutionFailed("map_async callback not called".into()))?;
            map_result.map_err(|e| GpuError::ExecutionFailed(format!("{:?}", e)))?;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // For native: use flume channel with polling
            let (tx, rx) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            loop {
                ctx.device.poll(wgpu::Maintain::Poll);
                match rx.try_recv() {
                    Ok(result) => {
                        result.map_err(|e| GpuError::ExecutionFailed(format!("{:?}", e)))?;
                        break;
                    }
                    Err(flume::TryRecvError::Empty) => {
                        std::thread::yield_now();
                    }
                    Err(flume::TryRecvError::Disconnected) => {
                        return Err(GpuError::ExecutionFailed("channel disconnected".into()));
                    }
                }
            }
        }

        let data = buffer_slice.get_mapped_range();
        let u32_data: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // Convert to residues
        let mut residues = vec![vec![0u64; self.n]; self.num_moduli];
        for mod_idx in 0..self.num_moduli {
            for coeff_idx in 0..self.n {
                let base = (mod_idx * self.n + coeff_idx) * 2;
                residues[mod_idx][coeff_idx] =
                    (u32_data[base] as u64) | ((u32_data[base + 1] as u64) << 32);
            }
        }

        Ok(residues)
    }

    /// Returns the buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Creates a zero-initialized GPU polynomial.
    pub fn new_zero(ctx: &GpuRotationContext, n: usize, num_moduli: usize) -> Self {
        let data = vec![0u32; n * num_moduli * 2];

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rns poly zero"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            n,
            num_moduli,
        }
    }

    /// Creates a GPU polynomial from a single-modulus buffer, replicating to all moduli.
    pub fn from_single_residue(
        ctx: &GpuRotationContext,
        single_buffer: &Buffer,
        n: usize,
        num_moduli: usize,
    ) -> Result<Self, GpuError> {
        let single_size = (n * 2 * std::mem::size_of::<u32>()) as u64;
        let total_size = single_size * num_moduli as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rns poly replicated"),
            size: total_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy the single residue to each modulus position
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("replicate encoder"),
        });
        for mod_idx in 0..num_moduli {
            let offset = mod_idx as u64 * single_size;
            encoder.copy_buffer_to_buffer(single_buffer, 0, &buffer, offset, single_size);
        }
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        Ok(Self {
            buffer,
            n,
            num_moduli,
        })
    }

    /// Creates a GPU polynomial by copying from an existing buffer.
    pub fn from_buffer(ctx: &GpuRotationContext, source: &Buffer, n: usize, num_moduli: usize) -> Self {
        let size = (n * num_moduli * 2 * std::mem::size_of::<u32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rns poly from buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("from buffer encoder"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        Self {
            buffer,
            n,
            num_moduli,
        }
    }
}

/// An RNS ciphertext (c0, c1) stored on GPU.
pub struct GpuRnsCiphertext {
    pub c0: GpuRnsPoly,
    pub c1: GpuRnsPoly,
}

impl GpuRnsCiphertext {
    /// Creates a GPU ciphertext from RNS residues.
    pub fn from_residues(
        ctx: &GpuRotationContext,
        c0_residues: &[Vec<u64>],
        c1_residues: &[Vec<u64>],
    ) -> Result<Self, GpuError> {
        Ok(Self {
            c0: GpuRnsPoly::from_residues(ctx, c0_residues)?,
            c1: GpuRnsPoly::from_residues(ctx, c1_residues)?,
        })
    }

    /// Reads ciphertext back to CPU.
    pub fn to_residues(
        &self,
        ctx: &GpuRotationContext,
    ) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>), GpuError> {
        Ok((self.c0.to_residues(ctx)?, self.c1.to_residues(ctx)?))
    }
}

/// Applies automorphism σ_k to a polynomial on GPU.
pub fn gpu_automorphism(
    ctx: &GpuRotationContext,
    input: &GpuRnsPoly,
    k: usize,
) -> Result<GpuRnsPoly, GpuError> {
    let n = input.n;
    let num_moduli = input.num_moduli;

    let output = GpuRnsPoly::new_uninit(ctx, n, num_moduli);

    let params = AutoParams {
        n: n as u32,
        two_n: (2 * n) as u32,
        k: k as u32,
        num_moduli: num_moduli as u32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auto params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

    let bind_group_layout = ctx.auto_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("auto bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ctx.moduli_buffer.as_entire_binding(),
            },
        ],
    });

    let total_threads = (n * num_moduli) as u32;
    let workgroups = (total_threads + 255) / 256;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("auto pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.auto_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(output)
}

/// Adds two RNS polynomials on GPU.
pub fn gpu_add(
    ctx: &GpuRotationContext,
    a: &GpuRnsPoly,
    b: &GpuRnsPoly,
) -> Result<GpuRnsPoly, GpuError> {
    if a.n != b.n || a.num_moduli != b.num_moduli {
        return Err(GpuError::InvalidParams("dimension mismatch".into()));
    }

    let n = a.n;
    let num_moduli = a.num_moduli;

    let result = GpuRnsPoly::new_uninit(ctx, n, num_moduli);

    let params = AddParams {
        n: n as u32,
        num_moduli: num_moduli as u32,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

    let bind_group_layout = ctx.add_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ctx.moduli_buffer.as_entire_binding(),
            },
        ],
    });

    let total_threads = (n * num_moduli) as u32;
    let workgroups = (total_threads + 255) / 256;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.add_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(result)
}

/// Multiplies two polynomials using NTT (for a single RNS modulus).
///
/// This performs negacyclic convolution in Z_q[X]/(X^n + 1) using:
/// 1. Pre-multiply by psi powers (twist for negacyclic)
/// 2. Forward NTT (bit-reverse + log_n butterfly passes)
/// 3. Pointwise multiply
/// 4. Inverse NTT (bit-reverse + log_n butterfly passes)
/// 5. Scale by 1/n and post-multiply by psi_inv powers
pub fn gpu_ntt_mul_single(
    ctx: &GpuRotationContext,
    a: &Buffer,
    b: &Buffer,
    mod_idx: usize,
) -> Result<Buffer, GpuError> {
    let n = ctx.params.n;
    let log_n = (n as f64).log2() as u32;
    let ntt = &ctx.ntt_data[mod_idx];

    // Create temporary buffers for twisted inputs and NTT results
    let buffer_size = (n * 2 * std::mem::size_of::<u32>()) as u64;

    let a_twisted = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("a_twisted"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_twisted = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("b_twisted"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let a_ntt = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("a_ntt"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_ntt = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("b_ntt"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let result_ntt = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result_ntt"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let result = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy inputs to working buffers
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ntt_mul setup encoder"),
    });
    encoder.copy_buffer_to_buffer(a, 0, &a_twisted, 0, buffer_size);
    encoder.copy_buffer_to_buffer(b, 0, &b_twisted, 0, buffer_size);
    ctx.queue.submit(Some(encoder.finish()));

    // Step 1: Pre-multiply by psi powers (twist for negacyclic)
    apply_twist(ctx, &a_twisted, mod_idx, false)?;
    apply_twist(ctx, &b_twisted, mod_idx, false)?;

    // Step 2: Forward NTT for both inputs
    // Bit-reverse permutation
    apply_bit_reverse(ctx, &a_twisted, &a_ntt)?;
    apply_bit_reverse(ctx, &b_twisted, &b_ntt)?;

    // log_n butterfly stages
    for stage in 0..log_n {
        apply_ntt_butterfly(ctx, &a_ntt, mod_idx, stage, false)?;
        apply_ntt_butterfly(ctx, &b_ntt, mod_idx, stage, false)?;
    }

    // Step 3: Pointwise multiplication
    apply_pointwise_mul(ctx, &a_ntt, &b_ntt, &result_ntt, mod_idx)?;

    // Step 4: Inverse NTT
    // Bit-reverse permutation
    apply_bit_reverse(ctx, &result_ntt, &result)?;

    // log_n butterfly stages with inverse twiddles
    for stage in 0..log_n {
        apply_ntt_butterfly(ctx, &result, mod_idx, stage, true)?;
    }

    // Step 5: Scale by 1/n and post-multiply by psi_inv powers
    apply_scale(ctx, &result, mod_idx, ntt.n_inv)?;
    apply_twist(ctx, &result, mod_idx, true)?;

    Ok(result)
}

/// Applies twist (multiply by psi powers or psi_inv powers).
fn apply_twist(
    ctx: &GpuRotationContext,
    buffer: &Buffer,
    mod_idx: usize,
    inverse: bool,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = TwistParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        _pad0: 0,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("twist params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let psi_buffer = if inverse {
        &ctx.psi_inv_power_buffers[mod_idx]
    } else {
        &ctx.psi_power_buffers[mod_idx]
    };

    let bind_group_layout = ctx.twist_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("twist bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: psi_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("twist encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("twist pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.twist_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Applies bit-reverse permutation.
fn apply_bit_reverse(
    ctx: &GpuRotationContext,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let log_n = (n as f64).log2() as u32;

    let params = BitRevParams {
        n: n as u32,
        log_n,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitrev params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.bit_rev_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bitrev bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bitrev encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bitrev pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.bit_rev_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Applies one stage of NTT butterfly.
fn apply_ntt_butterfly(
    ctx: &GpuRotationContext,
    data: &Buffer,
    mod_idx: usize,
    stage: u32,
    inverse: bool,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = NttButterflyParams {
        n: n as u32,
        stage,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("butterfly params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let twiddle_buffer = if inverse {
        &ctx.inv_twiddle_buffers[mod_idx]
    } else {
        &ctx.twiddle_buffers[mod_idx]
    };

    let bind_group_layout = ctx.ntt_butterfly_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("butterfly bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddle_buffer.as_entire_binding(),
            },
        ],
    });

    // n/2 butterflies per stage
    let workgroups = ((n / 2) as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("butterfly encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("butterfly pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.ntt_butterfly_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Applies pointwise multiplication.
fn apply_pointwise_mul(
    ctx: &GpuRotationContext,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
    mod_idx: usize,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = PointwiseParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        _pad0: 0,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pointwise params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.pointwise_mul_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pointwise bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pointwise encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pointwise pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pointwise_mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Applies scalar multiplication (scale by 1/n).
fn apply_scale(
    ctx: &GpuRotationContext,
    buffer: &Buffer,
    mod_idx: usize,
    scalar: u64,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = ScaleParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        scalar_lo: scalar as u32,
        scalar_hi: (scalar >> 32) as u32,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scale params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.scale_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scale encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.scale_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

/// Extracts a specific digit from each coefficient of a polynomial.
/// Used for HYBRID key-switching decomposition.
fn apply_digit_decompose(
    ctx: &GpuRotationContext,
    input: &Buffer,
    output: &Buffer,
    digit_idx: u32,
) -> Result<(), GpuError> {
    let n = ctx.params.n;
    let decomp_base_log = (ctx.params.decomp_base as f64).log2() as u32;

    let params = DecomposeParams {
        n: n as u32,
        digit_idx,
        decomp_base_log,
        _pad: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("decompose params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.digit_decompose_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decompose bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("decompose encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("decompose pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.digit_decompose_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);

    Ok(())
}

// ============================================================================
// Key-Switching with HYBRID Decomposition
// ============================================================================

/// GPU representation of a Galois key for automorphism σ_k.
/// Contains key-switching keys for HYBRID decomposition.
pub struct GpuGaloisKey {
    /// Automorphism exponent k.
    pub k: usize,
    /// Key components: keys_b[limb][digit] and keys_a[limb][digit].
    /// Each key is an RNS polynomial stored in a GPU buffer.
    pub keys_b: Vec<Vec<Buffer>>,
    pub keys_a: Vec<Vec<Buffer>>,
}

impl GpuGaloisKey {
    /// Creates a GPU Galois key from coefficient data.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `k` - Automorphism exponent
    /// * `keys_b_data` - keys_b[limb][digit] as Vec<Vec<u64>> (coefficients)
    /// * `keys_a_data` - keys_a[limb][digit] as Vec<Vec<u64>> (coefficients)
    pub fn from_coefficients(
        ctx: &GpuRotationContext,
        k: usize,
        keys_b_data: &[Vec<Vec<Vec<u64>>>],  // [limb][digit][mod_idx][coeff]
        keys_a_data: &[Vec<Vec<Vec<u64>>>],
    ) -> Result<Self, GpuError> {
        let num_limbs = keys_b_data.len();
        let digits_per_limb = if num_limbs > 0 { keys_b_data[0].len() } else { 0 };

        let mut keys_b = Vec::with_capacity(num_limbs);
        let mut keys_a = Vec::with_capacity(num_limbs);

        for limb_idx in 0..num_limbs {
            let mut limb_keys_b = Vec::with_capacity(digits_per_limb);
            let mut limb_keys_a = Vec::with_capacity(digits_per_limb);

            for digit_idx in 0..digits_per_limb {
                // Convert coefficients to u32 pairs and upload
                let b_data = &keys_b_data[limb_idx][digit_idx];
                let a_data = &keys_a_data[limb_idx][digit_idx];

                let b_u32: Vec<u32> = b_data.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&c| [c as u32, (c >> 32) as u32]))
                    .collect();
                let a_u32: Vec<u32> = a_data.iter()
                    .flat_map(|residue| residue.iter().flat_map(|&c| [c as u32, (c >> 32) as u32]))
                    .collect();

                let b_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("galois_key_b[{}][{}]", limb_idx, digit_idx)),
                    contents: bytemuck::cast_slice(&b_u32),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                });
                let a_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("galois_key_a[{}][{}]", limb_idx, digit_idx)),
                    contents: bytemuck::cast_slice(&a_u32),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                });

                limb_keys_b.push(b_buffer);
                limb_keys_a.push(a_buffer);
            }

            keys_b.push(limb_keys_b);
            keys_a.push(limb_keys_a);
        }

        Ok(Self { k, keys_b, keys_a })
    }
}

/// Performs key-switching on c1_auto using HYBRID decomposition.
/// Returns (ks_c0, ks_c1).
pub fn gpu_key_switch(
    ctx: &GpuRotationContext,
    c1_auto: &GpuRnsPoly,
    galois_key: &GpuGaloisKey,
) -> Result<(GpuRnsPoly, GpuRnsPoly), GpuError> {
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();
    let num_limbs = galois_key.keys_b.len();
    let digits_per_limb = ctx.params.digits_per_limb;
    let buffer_size = (n * num_moduli * 2 * std::mem::size_of::<u32>()) as u64;
    let single_mod_size = (n * 2 * std::mem::size_of::<u32>()) as u64;

    // Initialize accumulators to zero
    let c0_acc = GpuRnsPoly::new_zero(ctx, n, num_moduli);
    let c1_acc = GpuRnsPoly::new_zero(ctx, n, num_moduli);

    // Temporary buffers for digit extraction and multiplication
    let digit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("digit_buffer"),
        size: single_mod_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // For each RNS limb
    for limb_idx in 0..num_limbs.min(num_moduli) {
        // Extract the limb's residue from c1_auto
        let offset = (limb_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

        let limb_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("limb_buffer"),
            size: single_mod_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy the limb from c1_auto
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy limb encoder"),
        });
        encoder.copy_buffer_to_buffer(&c1_auto.buffer, offset, &limb_buffer, 0, single_mod_size);
        ctx.queue.submit(Some(encoder.finish()));

        // For each digit
        for digit_idx in 0..digits_per_limb {
            // Extract digit from the limb
            apply_digit_decompose(ctx, &limb_buffer, &digit_buffer, digit_idx as u32)?;

            // Create RNS polynomial from digit (replicate to all moduli)
            let digit_poly = GpuRnsPoly::from_single_residue(ctx, &digit_buffer, n, num_moduli)?;

            // Multiply digit_poly by galois_key component
            // term_b = digit_poly * keys_b[limb][digit]
            // term_a = digit_poly * keys_a[limb][digit]
            let key_b = GpuRnsPoly::from_buffer(
                ctx,
                &galois_key.keys_b[limb_idx][digit_idx],
                n,
                num_moduli,
            );
            let key_a = GpuRnsPoly::from_buffer(
                ctx,
                &galois_key.keys_a[limb_idx][digit_idx],
                n,
                num_moduli,
            );

            let term_b = gpu_ntt_mul(ctx, &digit_poly, &key_b)?;
            let term_a = gpu_ntt_mul(ctx, &digit_poly, &key_a)?;

            // Accumulate: c0_acc += term_b, c1_acc += term_a
            let c0_new = gpu_add(ctx, &c0_acc, &term_b)?;
            let c1_new = gpu_add(ctx, &c1_acc, &term_a)?;

            // Copy back to accumulators
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accumulate encoder"),
            });
            encoder.copy_buffer_to_buffer(&c0_new.buffer, 0, &c0_acc.buffer, 0, buffer_size);
            encoder.copy_buffer_to_buffer(&c1_new.buffer, 0, &c1_acc.buffer, 0, buffer_size);
            ctx.queue.submit(Some(encoder.finish()));
        }
    }

    ctx.device.poll(wgpu::Maintain::Wait);
    Ok((c0_acc, c1_acc))
}

/// Applies automorphism σ_k with key-switching on a GPU ciphertext.
pub fn gpu_apply_automorphism(
    ctx: &GpuRotationContext,
    ct: &GpuRnsCiphertext,
    galois_key: &GpuGaloisKey,
) -> Result<GpuRnsCiphertext, GpuError> {
    let k = galois_key.k;

    // Step 1: Apply automorphism σ_k to both components
    let c0_auto = gpu_automorphism(ctx, &ct.c0, k)?;
    let c1_auto = gpu_automorphism(ctx, &ct.c1, k)?;

    // Step 2: Key-switch c1_auto
    let (ks_c0, ks_c1) = gpu_key_switch(ctx, &c1_auto, galois_key)?;

    // Step 3: Combine: new_c0 = c0_auto + ks_c0, new_c1 = ks_c1
    let new_c0 = gpu_add(ctx, &c0_auto, &ks_c0)?;

    Ok(GpuRnsCiphertext {
        c0: new_c0,
        c1: ks_c1,
    })
}

// ============================================================================
// Sum Slots Operation
// ============================================================================

/// GPU representation of all Galois keys needed for sum_slots.
/// Contains log2(n) keys for rotations by powers of 2.
pub struct GpuGaloisKeys {
    /// Keys for rotation by 2^i slots, i = 0, 1, ..., log2(n)-1.
    /// keys[i] is for rotation by 2^i slots.
    pub keys: Vec<GpuGaloisKey>,
}

impl GpuGaloisKeys {
    /// Gets the key for rotation by the given step (must be a power of 2).
    pub fn get_key(&self, step: usize) -> Option<&GpuGaloisKey> {
        if step == 0 || !step.is_power_of_two() {
            return None;
        }
        let idx = step.trailing_zeros() as usize;
        self.keys.get(idx)
    }
}

/// Computes the rotation exponent k for σ_k that rotates slots by `step` positions.
/// For power-of-2 cyclotomics, rotation by `step` slots is σ_{5^step} mod 2n.
pub fn rotation_exponent(step: usize, n: usize) -> usize {
    // In the cyclotomic ring Z[X]/(X^n + 1), the Galois automorphism σ_k maps X → X^k.
    // For slot rotation by `step`, we need k = 5^step mod 2n.
    let two_n = 2 * n;
    let mut k = 1usize;
    let base = 5usize;

    // Compute 5^step mod 2n using repeated squaring
    let mut exp = step;
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            k = (k * b) % two_n;
        }
        b = (b * b) % two_n;
        exp >>= 1;
    }

    k
}

/// Sums all slots of a ciphertext using log2(n) rotations.
///
/// After calling this function, all slots contain the sum of the original slot values.
///
/// Algorithm:
/// ```text
/// for i in 0..log2(n):
///     step = 2^i
///     ct = ct + rotate(ct, step)
/// ```
pub fn gpu_sum_slots(
    ctx: &GpuRotationContext,
    ct: &GpuRnsCiphertext,
    galois_keys: &GpuGaloisKeys,
) -> Result<GpuRnsCiphertext, GpuError> {
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();
    let buffer_size = (n * num_moduli * 2 * std::mem::size_of::<u32>()) as u64;
    let num_keys = galois_keys.keys.len();

    // Copy input to working ciphertext
    let current = GpuRnsCiphertext {
        c0: GpuRnsPoly::new_uninit(ctx, n, num_moduli),
        c1: GpuRnsPoly::new_uninit(ctx, n, num_moduli),
    };

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sum_slots init encoder"),
    });
    encoder.copy_buffer_to_buffer(&ct.c0.buffer, 0, &current.c0.buffer, 0, buffer_size);
    encoder.copy_buffer_to_buffer(&ct.c1.buffer, 0, &current.c1.buffer, 0, buffer_size);
    ctx.queue.submit(Some(encoder.finish()));

    // The sum_slots algorithm uses the group structure of (Z/2n)*:
    // - Phase 1: Tree summation with σ_{5^{2^i}} for i=0..log(n)-2 (covers <5> subgroup)
    // - Phase 2: Add conjugate σ_{2n-1} to cover the -1·<5> coset
    //
    // Keys are organized as:
    // - Keys 0 to num_keys-2: power-of-5 automorphisms σ_{5^{2^i}}
    // - Key num_keys-1: conjugation automorphism σ_{2n-1}

    // Phase 1: Tree-based summation using powers of 5
    // This covers the <5> subgroup of (Z/2n)*
    // After log2(n)-1 iterations, each slot contains sum of n/2 slots
    for i in 0..(num_keys - 1) {
        let galois_key = galois_keys.keys.get(i).ok_or_else(|| {
            GpuError::InvalidParams(format!("Missing Galois key at index {}", i))
        })?;

        // Apply automorphism and add
        let permuted = gpu_apply_automorphism(ctx, &current, galois_key)?;
        let new_c0 = gpu_add(ctx, &current.c0, &permuted.c0)?;
        let new_c1 = gpu_add(ctx, &current.c1, &permuted.c1)?;

        // Update current (copy buffers)
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sum_slots phase1 encoder"),
        });
        encoder.copy_buffer_to_buffer(&new_c0.buffer, 0, &current.c0.buffer, 0, buffer_size);
        encoder.copy_buffer_to_buffer(&new_c1.buffer, 0, &current.c1.buffer, 0, buffer_size);
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Phase 2: Add conjugate to cover the -1·<5> coset
    // This doubles the sum to include all n slots
    if num_keys > 0 {
        let conj_key = galois_keys.keys.get(num_keys - 1).ok_or_else(|| {
            GpuError::InvalidParams("Missing conjugation key".to_string())
        })?;

        let conjugated = gpu_apply_automorphism(ctx, &current, conj_key)?;
        let new_c0 = gpu_add(ctx, &current.c0, &conjugated.c0)?;
        let new_c1 = gpu_add(ctx, &current.c1, &conjugated.c1)?;

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sum_slots phase2 encoder"),
        });
        encoder.copy_buffer_to_buffer(&new_c0.buffer, 0, &current.c0.buffer, 0, buffer_size);
        encoder.copy_buffer_to_buffer(&new_c1.buffer, 0, &current.c1.buffer, 0, buffer_size);
        ctx.queue.submit(Some(encoder.finish()));
    }

    ctx.device.poll(wgpu::Maintain::Wait);
    Ok(current)
}

/// Multiplies two RNS polynomials using NTT.
/// Performs NTT multiplication independently for each RNS modulus.
pub fn gpu_ntt_mul(
    ctx: &GpuRotationContext,
    a: &GpuRnsPoly,
    b: &GpuRnsPoly,
) -> Result<GpuRnsPoly, GpuError> {
    if a.n != b.n || a.num_moduli != b.num_moduli {
        return Err(GpuError::InvalidParams("dimension mismatch".into()));
    }

    let n = a.n;
    let num_moduli = a.num_moduli;

    // Allocate result buffer
    let result = GpuRnsPoly::new_uninit(ctx, n, num_moduli);
    let buffer_size_per_mod = (n * 2 * std::mem::size_of::<u32>()) as u64;

    // Process each modulus independently
    for mod_idx in 0..num_moduli {
        // Extract single-modulus buffers
        let offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

        // Create temporary single-modulus buffers
        let a_single = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("a_single"),
            size: buffer_size_per_mod,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_single = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b_single"),
            size: buffer_size_per_mod,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from RNS buffers
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract encoder"),
        });
        encoder.copy_buffer_to_buffer(&a.buffer, offset, &a_single, 0, buffer_size_per_mod);
        encoder.copy_buffer_to_buffer(&b.buffer, offset, &b_single, 0, buffer_size_per_mod);
        ctx.queue.submit(Some(encoder.finish()));

        // Multiply using NTT
        let result_single = gpu_ntt_mul_single(ctx, &a_single, &b_single, mod_idx)?;

        // Copy result back to RNS buffer
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("merge encoder"),
        });
        encoder.copy_buffer_to_buffer(&result_single, 0, &result.buffer, offset, buffer_size_per_mod);
        ctx.queue.submit(Some(encoder.finish()));
    }

    ctx.device.poll(wgpu::Maintain::Wait);
    Ok(result)
}

// ============================================================================
// Shaders (inline for simplicity)
// ============================================================================

const AUTOMORPHISM_SHADER: &str = r#"
struct AutoParams {
    n: u32,
    two_n: u32,
    k: u32,
    num_moduli: u32,
}

@group(0) @binding(0) var<uniform> params: AutoParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> moduli: array<u32>;

fn load_modulus(mod_idx: u32) -> vec2<u32> {
    return vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
}

fn negate_mod(x: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if x.x == 0u && x.y == 0u {
        return x;
    }
    if q.x >= x.x {
        return vec2<u32>(q.x - x.x, q.y - x.y);
    } else {
        return vec2<u32>(0xFFFFFFFFu - (x.x - q.x - 1u), q.y - x.y - 1u);
    }
}

@compute @workgroup_size(256, 1, 1)
fn apply_automorphism(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let two_n = params.two_n;
    let k = params.k;
    let num_moduli = params.num_moduli;

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    if mod_idx >= num_moduli {
        return;
    }

    let q = load_modulus(mod_idx);

    // For output coefficient at position coeff_idx, find source position
    // We need source_idx such that (source_idx * k) mod two_n gives coeff_idx or coeff_idx + n
    // Inverse: source_idx = coeff_idx * k_inv mod two_n (but k_inv is complex)

    // Alternative: do forward mapping - each thread reads its position and writes to target
    // output[target] = ±input[source] where target = (source * k) mod 2n

    // Read input coefficient at coeff_idx
    let in_base = (mod_idx * n + coeff_idx) * 2u;
    let in_val = vec2<u32>(input[in_base], input[in_base + 1u]);

    // Compute target position
    let target_exp = (coeff_idx * k) % two_n;
    var final_idx = target_exp;
    var negate = false;

    if target_exp >= n {
        final_idx = target_exp - n;
        negate = true;
    }

    var out_val = in_val;
    if negate {
        out_val = negate_mod(in_val, q);
    }

    // Write to output
    let out_base = (mod_idx * n + final_idx) * 2u;
    output[out_base] = out_val.x;
    output[out_base + 1u] = out_val.y;
}
"#;

/// Polynomial multiplication shader (schoolbook).
/// For ring Z[X]/(X^n + 1): result[k] = Σ_{i+j≡k} sign(i,j) * a[i] * b[j]
const POLY_MUL_SHADER: &str = r#"
struct PolyMulParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: PolyMulParams;
@group(0) @binding(1) var<storage, read> poly_a: array<u32>;
@group(0) @binding(2) var<storage, read> poly_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

fn load_a(idx: u32) -> vec2<u32> {
    return vec2<u32>(poly_a[idx * 2u], poly_a[idx * 2u + 1u]);
}

fn load_b(idx: u32) -> vec2<u32> {
    return vec2<u32>(poly_b[idx * 2u], poly_b[idx * 2u + 1u]);
}

// 32x32 -> 64 multiplication
fn mul32(a: u32, b: u32) -> vec2<u32> {
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

    if mid < p1 { hi = hi + 0x10000u; }

    return vec2<u32>(lo, hi);
}

// 64x64 -> 128 multiplication (returns low 128 bits as vec4)
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p0 = mul32(a.x, b.x);
    let p1 = mul32(a.x, b.y);
    let p2 = mul32(a.y, b.x);
    let p3 = mul32(a.y, b.y);

    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    // Add p1 << 32
    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;
    t = w2 + p1.y + c;
    if t < p1.y { w3 = w3 + 1u; }
    w2 = t;

    // Add p2 << 32
    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;
    t = w2 + p2.y + c;
    if t < p2.y || (c == 1u && t == p2.y) { w3 = w3 + 1u; }
    w2 = t;

    // Add p3 << 64
    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return vec4<u32>(w0, w1, w2, w3);
}

// Modular addition
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

// Modular subtraction
fn submod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        return vec2<u32>(diff_lo, diff_hi);
    } else {
        // a < b: return q - (b - a)
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;

        var res_lo = q.x - diff_lo;
        borrow = 0u;
        if q.x < diff_lo {
            borrow = 1u;
            res_lo = 0xFFFFFFFFu - (diff_lo - q.x - 1u);
        }
        var res_hi = q.y - diff_hi - borrow;
        return vec2<u32>(res_lo, res_hi);
    }
}

// Barrett-style reduction of 128-bit to 64-bit mod q
fn reduce128(x: vec4<u32>, q: vec2<u32>) -> vec2<u32> {
    // Simplified reduction: iteratively subtract q
    // This is slow but correct - can optimize with Barrett later
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;

    // While high bits are non-zero, subtract q << appropriate shift
    // For correctness, do simple loop
    for (var iter = 0u; iter < 128u; iter = iter + 1u) {
        if r2 == 0u && r3 == 0u {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                return vec2<u32>(r0, r1);
            }
        }

        // Subtract q
        if r0 >= q.x {
            r0 = r0 - q.x;
        } else {
            r0 = 0xFFFFFFFFu - (q.x - r0 - 1u);
            if r1 > 0u { r1 = r1 - 1u; }
            else if r2 > 0u { r1 = 0xFFFFFFFFu; r2 = r2 - 1u; }
            else if r3 > 0u { r1 = 0xFFFFFFFFu; r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
        }

        if r1 >= q.y {
            r1 = r1 - q.y;
        } else {
            if r2 > 0u { r2 = r2 - 1u; }
            else if r3 > 0u { r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
            else {
                // Underflow - should not happen if input was valid
                r1 = 0xFFFFFFFFu - (q.y - r1 - 1u);
            }
        }
    }

    return vec2<u32>(r0, r1);
}

// Modular multiplication
fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    return reduce128(prod, q);
}

@compute @workgroup_size(256, 1, 1)
fn poly_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let n = params.n;
    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);

    if out_idx >= n {
        return;
    }

    var acc = vec2<u32>(0u, 0u);

    // result[out_idx] = sum_{i} sign(i) * a[i] * b[j]
    // where j = (out_idx - i) mod n, and sign = -1 if wrap-around
    for (var i = 0u; i < n; i = i + 1u) {
        var j: u32;
        var negate: bool;

        if out_idx >= i {
            j = out_idx - i;
            negate = false;
        } else {
            j = out_idx + n - i;
            negate = true;
        }

        let a_val = load_a(i);
        let b_val = load_b(j);
        let prod = mulmod(a_val, b_val, q);

        if negate {
            acc = submod(acc, prod, q);
        } else {
            acc = addmod(acc, prod, q);
        }
    }

    result[out_idx * 2u] = acc.x;
    result[out_idx * 2u + 1u] = acc.y;
}
"#;

const ADD_SHADER: &str = r#"
struct AddParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;
@group(0) @binding(4) var<storage, read> moduli: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn add_polys(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);

    let base = (mod_idx * n + coeff_idx) * 2u;
    let a_val = vec2<u32>(a[base], a[base + 1u]);
    let b_val = vec2<u32>(b[base], b[base + 1u]);

    var sum_lo = a_val.x + b_val.x;
    var carry = 0u;
    if sum_lo < a_val.x { carry = 1u; }
    var sum_hi = a_val.y + b_val.y + carry;

    // Reduce if >= q
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }

    result[base] = sum_lo;
    result[base + 1u] = sum_hi;
}
"#;

// ============================================================================
// Batched sum_slots Implementation
// ============================================================================
//
// This version pre-allocates all working buffers and encodes the entire
// sum_slots operation into a SINGLE command buffer submission, eliminating
// CPU-GPU synchronization overhead.

/// Pre-allocated workspace for batched sum_slots operation.
///
/// This struct holds all the GPU buffers needed for the complete sum_slots
/// operation, allowing everything to be encoded into a single command buffer.
pub struct SumSlotsWorkspace {
    // Ring parameters
    n: usize,
    num_moduli: usize,
    digits_per_limb: usize,
    log_n: u32,

    // Buffer sizes
    rns_buffer_size: u64,
    single_mod_size: u64,

    // Current ciphertext (ping-pong buffers)
    pub current_c0: Buffer,
    pub current_c1: Buffer,
    temp_c0: Buffer,
    temp_c1: Buffer,

    // Automorphism outputs
    c0_auto: Buffer,
    c1_auto: Buffer,

    // Key-switch accumulators
    ks_c0_acc: Buffer,
    ks_c1_acc: Buffer,

    // NTT working buffers (per modulus)
    // For key-switching, we need to do NTT muls for each digit
    // We'll reuse these buffers across iterations
    ntt_a_twisted: Vec<Buffer>,
    ntt_b_twisted: Vec<Buffer>,
    ntt_a_out: Vec<Buffer>,
    ntt_b_out: Vec<Buffer>,
    ntt_result: Vec<Buffer>,
    ntt_result_inv: Vec<Buffer>,

    // Digit extraction buffers
    limb_buffer: Buffer,
    digit_buffer: Buffer,
    digit_rns: Buffer,  // digit replicated to all moduli

    // Term accumulation
    term_b: Buffer,
    term_a: Buffer,

    // Temporary buffer for in-place operations
    add_temp: Buffer,

    // =========================================================================
    // Pre-allocated bind groups for NTT operations
    // =========================================================================
    // These eliminate the ~2.4s of bind group creation overhead

    // Pre-created params buffers (static, depend only on modulus)
    twist_params_fwd: Vec<Buffer>,      // [mod_idx] - forward twist params
    twist_params_inv: Vec<Buffer>,      // [mod_idx] - inverse twist params
    bitrev_params: Buffer,              // single params (n, log_n always same)
    butterfly_params_fwd: Vec<Vec<Buffer>>,  // [mod_idx][stage] - forward butterfly
    butterfly_params_inv: Vec<Vec<Buffer>>,  // [mod_idx][stage] - inverse butterfly
    pointwise_params: Vec<Buffer>,      // [mod_idx]
    scale_params: Vec<Buffer>,          // [mod_idx]

    // Pre-created bind groups for NTT on workspace buffers
    // Twist bind groups: [buffer_type][mod_idx] where buffer_type = 0:a_twisted, 1:b_twisted, 2:result_inv
    twist_bg_a_fwd: Vec<wgpu::BindGroup>,      // forward twist on ntt_a_twisted
    twist_bg_b_fwd: Vec<wgpu::BindGroup>,      // forward twist on ntt_b_twisted
    twist_bg_result_inv: Vec<wgpu::BindGroup>, // inverse twist on ntt_result_inv

    // BitRev bind groups
    bitrev_bg_a: Vec<wgpu::BindGroup>,         // ntt_a_twisted -> ntt_a_out
    bitrev_bg_b: Vec<wgpu::BindGroup>,         // ntt_b_twisted -> ntt_b_out
    bitrev_bg_result: Vec<wgpu::BindGroup>,    // ntt_result -> ntt_result_inv

    // Butterfly bind groups: [mod_idx][stage]
    butterfly_bg_a_fwd: Vec<Vec<wgpu::BindGroup>>,     // forward on ntt_a_out
    butterfly_bg_b_fwd: Vec<Vec<wgpu::BindGroup>>,     // forward on ntt_b_out
    butterfly_bg_result_inv: Vec<Vec<wgpu::BindGroup>>, // inverse on ntt_result_inv

    // Pointwise and scale bind groups: [mod_idx]
    pointwise_bg: Vec<wgpu::BindGroup>,        // a_out × b_out -> result
    scale_bg: Vec<wgpu::BindGroup>,            // scale on ntt_result_inv

    // =========================================================================
    // Fused NTT buffers and bind groups (process all moduli in single dispatch)
    // =========================================================================
    // These reduce dispatch count by 3× (one dispatch for all 3 moduli)

    // Fused working buffers (RNS-sized, hold all moduli)
    fused_a_twisted: Buffer,      // n * num_moduli for twisted a
    fused_b_twisted: Buffer,      // n * num_moduli for twisted b
    fused_a_out: Buffer,          // n * num_moduli for bit-reversed a
    fused_b_out: Buffer,          // n * num_moduli for bit-reversed b
    fused_result: Buffer,         // n * num_moduli for pointwise result
    fused_result_inv: Buffer,     // n * num_moduli for inverse NTT output

    // Fused params buffers
    fused_twist_params: Buffer,
    fused_bitrev_params: Buffer,
    fused_butterfly_params: Vec<Buffer>,  // [stage] - one per stage
    fused_pointwise_params: Buffer,
    fused_scale_params: Buffer,

    // Fused bind groups
    fused_twist_a_fwd_bg: wgpu::BindGroup,
    fused_twist_b_fwd_bg: wgpu::BindGroup,
    fused_twist_result_inv_bg: wgpu::BindGroup,
    fused_bitrev_a_bg: wgpu::BindGroup,
    fused_bitrev_b_bg: wgpu::BindGroup,
    fused_bitrev_result_bg: wgpu::BindGroup,
    fused_butterfly_a_fwd_bg: Vec<wgpu::BindGroup>,   // [stage]
    fused_butterfly_b_fwd_bg: Vec<wgpu::BindGroup>,   // [stage]
    fused_butterfly_result_inv_bg: Vec<wgpu::BindGroup>, // [stage]
    fused_pointwise_bg: wgpu::BindGroup,
    fused_scale_bg: wgpu::BindGroup,

    // =========================================================================
    // Shared memory NTT bind groups (entire NTT in single dispatch)
    // =========================================================================
    // Pre-allocated uniform buffers for shared memory NTT
    shared_mem_ntt_params: Buffer,              // n, log_n (same for all)
    shared_mem_q_buffers: Vec<Buffer>,          // [mod_idx] - q as vec2<u32>
    shared_mem_barrett_buffers: Vec<Buffer>,    // [mod_idx] - barrett params as vec4<u32>
    shared_mem_n_inv_buffers: Vec<Buffer>,      // [mod_idx] - n^-1 for inverse NTT

    // Pre-allocated bind groups for shared memory NTT on workspace buffers
    shared_mem_ntt_fwd_a_bg: Vec<wgpu::BindGroup>,      // [mod_idx] forward on ntt_a_twisted
    shared_mem_ntt_fwd_b_bg: Vec<wgpu::BindGroup>,      // [mod_idx] forward on ntt_b_twisted
    shared_mem_ntt_inv_result_bg: Vec<wgpu::BindGroup>, // [mod_idx] inverse on ntt_result
}

impl SumSlotsWorkspace {
    /// Creates a new workspace with all buffers and bind groups pre-allocated.
    pub fn new(ctx: &GpuRotationContext) -> Self {
        let n = ctx.params.n;
        let num_moduli = ctx.params.moduli.len();
        let digits_per_limb = ctx.params.digits_per_limb;
        let log_n = (n as f64).log2() as u32;

        let rns_buffer_size = (n * num_moduli * 2 * std::mem::size_of::<u32>()) as u64;
        let single_mod_size = (n * 2 * std::mem::size_of::<u32>()) as u64;

        let create_rns_buffer = |label: &str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: rns_buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let create_single_buffer = |label: &str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: single_mod_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Create NTT buffers for each modulus
        let mut ntt_a_twisted = Vec::with_capacity(num_moduli);
        let mut ntt_b_twisted = Vec::with_capacity(num_moduli);
        let mut ntt_a_out = Vec::with_capacity(num_moduli);
        let mut ntt_b_out = Vec::with_capacity(num_moduli);
        let mut ntt_result = Vec::with_capacity(num_moduli);
        let mut ntt_result_inv = Vec::with_capacity(num_moduli);

        for i in 0..num_moduli {
            ntt_a_twisted.push(create_single_buffer(&format!("ntt_a_twisted_{}", i)));
            ntt_b_twisted.push(create_single_buffer(&format!("ntt_b_twisted_{}", i)));
            ntt_a_out.push(create_single_buffer(&format!("ntt_a_out_{}", i)));
            ntt_b_out.push(create_single_buffer(&format!("ntt_b_out_{}", i)));
            ntt_result.push(create_single_buffer(&format!("ntt_result_{}", i)));
            ntt_result_inv.push(create_single_buffer(&format!("ntt_result_inv_{}", i)));
        }

        // =================================================================
        // Pre-create params buffers (static, created once)
        // =================================================================

        // Twist params: one per modulus, for forward and inverse
        let mut twist_params_fwd = Vec::with_capacity(num_moduli);
        let mut twist_params_inv = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            let ntt = &ctx.ntt_data[mod_idx];
            let params = TwistParams {
                n: n as u32,
                modulus_lo: ntt.modulus as u32,
                modulus_hi: (ntt.modulus >> 32) as u32,
                _pad0: 0,
                mu_lo_lo: ntt.mu_lo as u32,
                mu_lo_hi: (ntt.mu_lo >> 32) as u32,
                mu_hi_lo: ntt.mu_hi as u32,
                mu_hi_hi: (ntt.mu_hi >> 32) as u32,
            };
            // Forward and inverse use same params (different psi buffer)
            twist_params_fwd.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("twist_params_fwd_{}", mod_idx)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            }));
            twist_params_inv.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("twist_params_inv_{}", mod_idx)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            }));
        }

        // BitRev params: single buffer (n and log_n are constant)
        let bitrev_params_data = BitRevParams {
            n: n as u32,
            log_n,
            _pad0: 0,
            _pad1: 0,
        };
        let bitrev_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bitrev_params"),
            contents: bytemuck::bytes_of(&bitrev_params_data),
            usage: BufferUsages::UNIFORM,
        });

        // Butterfly params: [mod_idx][stage] for forward and inverse
        let mut butterfly_params_fwd = Vec::with_capacity(num_moduli);
        let mut butterfly_params_inv = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            let ntt = &ctx.ntt_data[mod_idx];
            let mut stages_fwd = Vec::with_capacity(log_n as usize);
            let mut stages_inv = Vec::with_capacity(log_n as usize);
            for stage in 0..log_n {
                let params = NttButterflyParams {
                    n: n as u32,
                    stage,
                    modulus_lo: ntt.modulus as u32,
                    modulus_hi: (ntt.modulus >> 32) as u32,
                    mu_lo_lo: ntt.mu_lo as u32,
                    mu_lo_hi: (ntt.mu_lo >> 32) as u32,
                    mu_hi_lo: ntt.mu_hi as u32,
                    mu_hi_hi: (ntt.mu_hi >> 32) as u32,
                };
                stages_fwd.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("butterfly_params_fwd_{}_{}", mod_idx, stage)),
                    contents: bytemuck::bytes_of(&params),
                    usage: BufferUsages::UNIFORM,
                }));
                stages_inv.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("butterfly_params_inv_{}_{}", mod_idx, stage)),
                    contents: bytemuck::bytes_of(&params),
                    usage: BufferUsages::UNIFORM,
                }));
            }
            butterfly_params_fwd.push(stages_fwd);
            butterfly_params_inv.push(stages_inv);
        }

        // Pointwise params: [mod_idx]
        let mut pointwise_params = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            let ntt = &ctx.ntt_data[mod_idx];
            let params = PointwiseParams {
                n: n as u32,
                modulus_lo: ntt.modulus as u32,
                modulus_hi: (ntt.modulus >> 32) as u32,
                _pad0: 0,
                mu_lo_lo: ntt.mu_lo as u32,
                mu_lo_hi: (ntt.mu_lo >> 32) as u32,
                mu_hi_lo: ntt.mu_hi as u32,
                mu_hi_hi: (ntt.mu_hi >> 32) as u32,
            };
            pointwise_params.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("pointwise_params_{}", mod_idx)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            }));
        }

        // Scale params: [mod_idx]
        let mut scale_params = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            let ntt = &ctx.ntt_data[mod_idx];
            let params = ScaleParams {
                n: n as u32,
                modulus_lo: ntt.modulus as u32,
                modulus_hi: (ntt.modulus >> 32) as u32,
                scalar_lo: ntt.n_inv as u32,
                scalar_hi: (ntt.n_inv >> 32) as u32,
                mu_lo_lo: ntt.mu_lo as u32,
                mu_lo_hi: (ntt.mu_lo >> 32) as u32,
                mu_hi_lo: ntt.mu_hi as u32,
                mu_hi_hi: (ntt.mu_hi >> 32) as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            scale_params.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("scale_params_{}", mod_idx)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            }));
        }

        // =================================================================
        // Pre-create bind groups for NTT operations
        // =================================================================

        // Twist bind groups
        let mut twist_bg_a_fwd = Vec::with_capacity(num_moduli);
        let mut twist_bg_b_fwd = Vec::with_capacity(num_moduli);
        let mut twist_bg_result_inv = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            // Forward twist on ntt_a_twisted
            twist_bg_a_fwd.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("twist_bg_a_fwd_{}", mod_idx)),
                layout: &ctx.twist_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: twist_params_fwd[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_a_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.psi_power_buffers[mod_idx].as_entire_binding() },
                ],
            }));
            // Forward twist on ntt_b_twisted
            twist_bg_b_fwd.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("twist_bg_b_fwd_{}", mod_idx)),
                layout: &ctx.twist_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: twist_params_fwd[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_b_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.psi_power_buffers[mod_idx].as_entire_binding() },
                ],
            }));
            // Inverse twist on ntt_result_inv
            twist_bg_result_inv.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("twist_bg_result_inv_{}", mod_idx)),
                layout: &ctx.twist_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: twist_params_inv[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_result_inv[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.psi_inv_power_buffers[mod_idx].as_entire_binding() },
                ],
            }));
        }

        // BitRev bind groups
        let mut bitrev_bg_a = Vec::with_capacity(num_moduli);
        let mut bitrev_bg_b = Vec::with_capacity(num_moduli);
        let mut bitrev_bg_result = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            bitrev_bg_a.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bitrev_bg_a_{}", mod_idx)),
                layout: &ctx.bit_rev_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bitrev_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_a_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ntt_a_out[mod_idx].as_entire_binding() },
                ],
            }));
            bitrev_bg_b.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bitrev_bg_b_{}", mod_idx)),
                layout: &ctx.bit_rev_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bitrev_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_b_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ntt_b_out[mod_idx].as_entire_binding() },
                ],
            }));
            bitrev_bg_result.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bitrev_bg_result_{}", mod_idx)),
                layout: &ctx.bit_rev_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bitrev_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_result[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ntt_result_inv[mod_idx].as_entire_binding() },
                ],
            }));
        }

        // Butterfly bind groups: [mod_idx][stage]
        let mut butterfly_bg_a_fwd = Vec::with_capacity(num_moduli);
        let mut butterfly_bg_b_fwd = Vec::with_capacity(num_moduli);
        let mut butterfly_bg_result_inv = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            let mut stages_a = Vec::with_capacity(log_n as usize);
            let mut stages_b = Vec::with_capacity(log_n as usize);
            let mut stages_result = Vec::with_capacity(log_n as usize);
            for stage in 0..log_n as usize {
                stages_a.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("butterfly_bg_a_fwd_{}_{}", mod_idx, stage)),
                    layout: &ctx.ntt_butterfly_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: butterfly_params_fwd[mod_idx][stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: ntt_a_out[mod_idx].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: ctx.twiddle_buffers[mod_idx].as_entire_binding() },
                    ],
                }));
                stages_b.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("butterfly_bg_b_fwd_{}_{}", mod_idx, stage)),
                    layout: &ctx.ntt_butterfly_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: butterfly_params_fwd[mod_idx][stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: ntt_b_out[mod_idx].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: ctx.twiddle_buffers[mod_idx].as_entire_binding() },
                    ],
                }));
                stages_result.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("butterfly_bg_result_inv_{}_{}", mod_idx, stage)),
                    layout: &ctx.ntt_butterfly_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: butterfly_params_inv[mod_idx][stage].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: ntt_result_inv[mod_idx].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: ctx.inv_twiddle_buffers[mod_idx].as_entire_binding() },
                    ],
                }));
            }
            butterfly_bg_a_fwd.push(stages_a);
            butterfly_bg_b_fwd.push(stages_b);
            butterfly_bg_result_inv.push(stages_result);
        }

        // Pointwise bind groups
        let mut pointwise_bg = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            pointwise_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("pointwise_bg_{}", mod_idx)),
                layout: &ctx.pointwise_mul_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pointwise_params[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_a_out[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ntt_b_out[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ntt_result[mod_idx].as_entire_binding() },
                ],
            }));
        }

        // Scale bind groups
        let mut scale_bg = Vec::with_capacity(num_moduli);
        for mod_idx in 0..num_moduli {
            scale_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("scale_bg_{}", mod_idx)),
                layout: &ctx.scale_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scale_params[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_result_inv[mod_idx].as_entire_binding() },
                ],
            }));
        }

        // =====================================================================
        // Fused buffers and bind groups (process all moduli in single dispatch)
        // =====================================================================

        // Create fused working buffers (RNS-sized)
        let fused_a_twisted = create_rns_buffer("fused_a_twisted");
        let fused_b_twisted = create_rns_buffer("fused_b_twisted");
        let fused_a_out = create_rns_buffer("fused_a_out");
        let fused_b_out = create_rns_buffer("fused_b_out");
        let fused_result = create_rns_buffer("fused_result");
        let fused_result_inv = create_rns_buffer("fused_result_inv");

        // Create fused params buffers
        let fused_twist_params_data = FusedTwistParams {
            n: n as u32,
            num_moduli: num_moduli as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let fused_twist_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused_twist_params"),
            contents: bytemuck::bytes_of(&fused_twist_params_data),
            usage: BufferUsages::UNIFORM,
        });

        let fused_bitrev_params_data = FusedBitrevParams {
            n: n as u32,
            log_n,
            num_moduli: num_moduli as u32,
            _pad: 0,
        };
        let fused_bitrev_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused_bitrev_params"),
            contents: bytemuck::bytes_of(&fused_bitrev_params_data),
            usage: BufferUsages::UNIFORM,
        });

        let mut fused_butterfly_params = Vec::with_capacity(log_n as usize);
        for stage in 0..log_n {
            let params = FusedButterflyParams {
                n: n as u32,
                stage,
                num_moduli: num_moduli as u32,
                _pad: 0,
            };
            fused_butterfly_params.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("fused_butterfly_params_{}", stage)),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            }));
        }

        let fused_pointwise_params_data = FusedPointwiseParams {
            n: n as u32,
            num_moduli: num_moduli as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let fused_pointwise_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused_pointwise_params"),
            contents: bytemuck::bytes_of(&fused_pointwise_params_data),
            usage: BufferUsages::UNIFORM,
        });

        let fused_scale_params_data = FusedScaleParams {
            n: n as u32,
            num_moduli: num_moduli as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let fused_scale_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused_scale_params"),
            contents: bytemuck::bytes_of(&fused_scale_params_data),
            usage: BufferUsages::UNIFORM,
        });

        // Create fused bind groups
        // Twist bind groups use: params, data, psi_powers, moduli, barrett_params
        let fused_twist_a_fwd_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_twist_a_fwd_bg"),
            layout: &ctx.fused_twist_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_twist_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_a_twisted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.all_psi_powers_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
            ],
        });

        let fused_twist_b_fwd_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_twist_b_fwd_bg"),
            layout: &ctx.fused_twist_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_twist_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_b_twisted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.all_psi_powers_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
            ],
        });

        let fused_twist_result_inv_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_twist_result_inv_bg"),
            layout: &ctx.fused_twist_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_twist_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_result_inv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.all_psi_inv_powers_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
            ],
        });

        // Bitrev bind groups: params, input, output
        let fused_bitrev_a_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_bitrev_a_bg"),
            layout: &ctx.fused_bitrev_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_bitrev_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_a_twisted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fused_a_out.as_entire_binding() },
            ],
        });

        let fused_bitrev_b_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_bitrev_b_bg"),
            layout: &ctx.fused_bitrev_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_bitrev_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_b_twisted.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fused_b_out.as_entire_binding() },
            ],
        });

        let fused_bitrev_result_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_bitrev_result_bg"),
            layout: &ctx.fused_bitrev_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_bitrev_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fused_result_inv.as_entire_binding() },
            ],
        });

        // Butterfly bind groups: params, data, twiddles, moduli, barrett_params
        let mut fused_butterfly_a_fwd_bg = Vec::with_capacity(log_n as usize);
        let mut fused_butterfly_b_fwd_bg = Vec::with_capacity(log_n as usize);
        let mut fused_butterfly_result_inv_bg = Vec::with_capacity(log_n as usize);
        for stage in 0..log_n as usize {
            fused_butterfly_a_fwd_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fused_butterfly_a_fwd_bg_{}", stage)),
                layout: &ctx.fused_butterfly_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: fused_butterfly_params[stage].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fused_a_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.all_twiddles_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
                ],
            }));
            fused_butterfly_b_fwd_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fused_butterfly_b_fwd_bg_{}", stage)),
                layout: &ctx.fused_butterfly_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: fused_butterfly_params[stage].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fused_b_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.all_twiddles_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
                ],
            }));
            fused_butterfly_result_inv_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fused_butterfly_result_inv_bg_{}", stage)),
                layout: &ctx.fused_butterfly_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: fused_butterfly_params[stage].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fused_result_inv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.all_inv_twiddles_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
                ],
            }));
        }

        // Pointwise bind group: params, a, b, result, moduli, barrett_params
        let fused_pointwise_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_pointwise_bg"),
            layout: &ctx.fused_pointwise_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_pointwise_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_a_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fused_b_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fused_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.all_moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ctx.all_barrett_buffer.as_entire_binding() },
            ],
        });

        // Scale bind group: params, data, n_inv, moduli, barrett_params
        let fused_scale_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_scale_bg"),
            layout: &ctx.fused_scale_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fused_scale_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fused_result_inv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.all_n_inv_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.all_moduli_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctx.all_barrett_buffer.as_entire_binding() },
            ],
        });

        // =========================================================================
        // Shared memory NTT buffers and bind groups
        // =========================================================================
        // Params buffer (same for all NTTs)
        let shared_mem_params = SharedMemNttParams {
            n: n as u32,
            log_n,
            _pad0: 0,
            _pad1: 0,
        };
        let shared_mem_ntt_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shared_mem_ntt_params"),
            contents: bytemuck::bytes_of(&shared_mem_params),
            usage: BufferUsages::UNIFORM,
        });

        // Per-modulus uniform buffers
        let mut shared_mem_q_buffers = Vec::with_capacity(num_moduli);
        let mut shared_mem_barrett_buffers = Vec::with_capacity(num_moduli);
        let mut shared_mem_n_inv_buffers = Vec::with_capacity(num_moduli);

        for mod_idx in 0..num_moduli {
            let ntt = &ctx.ntt_data[mod_idx];

            // q as vec2<u32>
            let q_data = [ntt.modulus as u32, (ntt.modulus >> 32) as u32];
            shared_mem_q_buffers.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("shared_mem_q_{}", mod_idx)),
                contents: bytemuck::cast_slice(&q_data),
                usage: BufferUsages::UNIFORM,
            }));

            // Barrett as vec4<u32>
            let barrett_data = [
                ntt.mu_lo as u32,
                (ntt.mu_lo >> 32) as u32,
                ntt.mu_hi as u32,
                (ntt.mu_hi >> 32) as u32,
            ];
            shared_mem_barrett_buffers.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("shared_mem_barrett_{}", mod_idx)),
                contents: bytemuck::cast_slice(&barrett_data),
                usage: BufferUsages::UNIFORM,
            }));

            // n_inv as storage (for inverse NTT)
            let n_inv_data = [ntt.n_inv as u32, (ntt.n_inv >> 32) as u32];
            shared_mem_n_inv_buffers.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("shared_mem_n_inv_{}", mod_idx)),
                contents: bytemuck::cast_slice(&n_inv_data),
                usage: BufferUsages::STORAGE,
            }));
        }

        // Shared memory NTT bind groups
        // Forward NTT bindings: 0=params, 1=data, 2=twiddles, 3=psi, 4=q, 5=barrett
        // Inverse NTT bindings: 0=params, 1=data, 2=inv_twiddles, 3=inv_psi, 4=n_inv, 5=q, 6=barrett
        let mut shared_mem_ntt_fwd_a_bg = Vec::with_capacity(num_moduli);
        let mut shared_mem_ntt_fwd_b_bg = Vec::with_capacity(num_moduli);
        let mut shared_mem_ntt_inv_result_bg = Vec::with_capacity(num_moduli);

        for mod_idx in 0..num_moduli {
            // Forward NTT on ntt_a_twisted
            shared_mem_ntt_fwd_a_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shared_mem_ntt_fwd_a_{}", mod_idx)),
                layout: &ctx.shared_mem_ntt_fwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: shared_mem_ntt_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_a_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.twiddle_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.psi_power_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: shared_mem_q_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: shared_mem_barrett_buffers[mod_idx].as_entire_binding() },
                ],
            }));

            // Forward NTT on ntt_b_twisted
            shared_mem_ntt_fwd_b_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shared_mem_ntt_fwd_b_{}", mod_idx)),
                layout: &ctx.shared_mem_ntt_fwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: shared_mem_ntt_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_b_twisted[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.twiddle_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.psi_power_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: shared_mem_q_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: shared_mem_barrett_buffers[mod_idx].as_entire_binding() },
                ],
            }));

            // Inverse NTT on ntt_result
            shared_mem_ntt_inv_result_bg.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shared_mem_ntt_inv_result_{}", mod_idx)),
                layout: &ctx.shared_mem_ntt_inv_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: shared_mem_ntt_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ntt_result[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.inv_twiddle_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.psi_inv_power_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: shared_mem_n_inv_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: shared_mem_q_buffers[mod_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: shared_mem_barrett_buffers[mod_idx].as_entire_binding() },
                ],
            }));
        }

        let current_c0 = create_rns_buffer("current_c0");
        let current_c1 = create_rns_buffer("current_c1");
        let temp_c0 = create_rns_buffer("temp_c0");
        let temp_c1 = create_rns_buffer("temp_c1");
        let c0_auto = create_rns_buffer("c0_auto");
        let c1_auto = create_rns_buffer("c1_auto");
        let ks_c0_acc = create_rns_buffer("ks_c0_acc");
        let ks_c1_acc = create_rns_buffer("ks_c1_acc");
        let limb_buffer = create_single_buffer("limb_buffer");
        let digit_buffer = create_single_buffer("digit_buffer");
        let digit_rns = create_rns_buffer("digit_rns");
        let term_b = create_rns_buffer("term_b");
        let term_a = create_rns_buffer("term_a");
        let add_temp = create_rns_buffer("add_temp");

        Self {
            n,
            num_moduli,
            digits_per_limb,
            log_n,
            rns_buffer_size,
            single_mod_size,

            current_c0,
            current_c1,
            temp_c0,
            temp_c1,
            c0_auto,
            c1_auto,
            ks_c0_acc,
            ks_c1_acc,

            ntt_a_twisted,
            ntt_b_twisted,
            ntt_a_out,
            ntt_b_out,
            ntt_result,
            ntt_result_inv,

            limb_buffer,
            digit_buffer,
            digit_rns,
            term_b,
            term_a,
            add_temp,

            // Pre-allocated params buffers
            twist_params_fwd,
            twist_params_inv,
            bitrev_params,
            butterfly_params_fwd,
            butterfly_params_inv,
            pointwise_params,
            scale_params,

            // Pre-allocated bind groups
            twist_bg_a_fwd,
            twist_bg_b_fwd,
            twist_bg_result_inv,
            bitrev_bg_a,
            bitrev_bg_b,
            bitrev_bg_result,
            butterfly_bg_a_fwd,
            butterfly_bg_b_fwd,
            butterfly_bg_result_inv,
            pointwise_bg,
            scale_bg,

            // Fused buffers
            fused_a_twisted,
            fused_b_twisted,
            fused_a_out,
            fused_b_out,
            fused_result,
            fused_result_inv,

            // Fused params
            fused_twist_params,
            fused_bitrev_params,
            fused_butterfly_params,
            fused_pointwise_params,
            fused_scale_params,

            // Fused bind groups
            fused_twist_a_fwd_bg,
            fused_twist_b_fwd_bg,
            fused_twist_result_inv_bg,
            fused_bitrev_a_bg,
            fused_bitrev_b_bg,
            fused_bitrev_result_bg,
            fused_butterfly_a_fwd_bg,
            fused_butterfly_b_fwd_bg,
            fused_butterfly_result_inv_bg,
            fused_pointwise_bg,
            fused_scale_bg,

            // Shared memory NTT
            shared_mem_ntt_params,
            shared_mem_q_buffers,
            shared_mem_barrett_buffers,
            shared_mem_n_inv_buffers,
            shared_mem_ntt_fwd_a_bg,
            shared_mem_ntt_fwd_b_bg,
            shared_mem_ntt_inv_result_bg,
        }
    }
}

/// Batched sum_slots that encodes operations per-rotation into command buffers.
///
/// This reduces CPU-GPU synchronization from ~1000 submits to 13 submits
/// (one per rotation), while avoiding the overhead of creating 48,000 buffers
/// in a single encoder.
pub fn gpu_sum_slots_batched(
    ctx: &GpuRotationContext,
    ct: &GpuRnsCiphertext,
    galois_keys: &GpuGaloisKeys,
    workspace: &SumSlotsWorkspace,
) -> Result<GpuRnsCiphertext, GpuError> {
    gpu_sum_slots_batched_inner(ctx, ct, galois_keys, workspace, false)
}

/// Batched sum_slots with optional profiling output.
pub fn gpu_sum_slots_batched_profiled(
    ctx: &GpuRotationContext,
    ct: &GpuRnsCiphertext,
    galois_keys: &GpuGaloisKeys,
    workspace: &SumSlotsWorkspace,
) -> Result<GpuRnsCiphertext, GpuError> {
    gpu_sum_slots_batched_inner(ctx, ct, galois_keys, workspace, true)
}

fn gpu_sum_slots_batched_inner(
    ctx: &GpuRotationContext,
    ct: &GpuRnsCiphertext,
    galois_keys: &GpuGaloisKeys,
    workspace: &SumSlotsWorkspace,
    profile: bool,
) -> Result<GpuRnsCiphertext, GpuError> {
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;

    #[cfg(not(target_arch = "wasm32"))]
    let total_start = Instant::now();
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();
    let num_keys = galois_keys.keys.len();

    #[cfg(not(target_arch = "wasm32"))]
    let mut encode_time = std::time::Duration::ZERO;
    #[cfg(not(target_arch = "wasm32"))]
    let mut submit_time = std::time::Duration::ZERO;

    // Initial copy: input -> workspace
    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("init_copy"),
        });
        encoder.copy_buffer_to_buffer(
            &ct.c0.buffer, 0,
            &workspace.current_c0, 0,
            workspace.rns_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &ct.c1.buffer, 0,
            &workspace.current_c1, 0,
            workspace.rns_buffer_size,
        );
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Process each rotation with ONE command buffer per rotation
    for key_idx in 0..num_keys {
        #[cfg(not(target_arch = "wasm32"))]
        let rotation_start = Instant::now();
        let galois_key = &galois_keys.keys[key_idx];
        let k = galois_key.k;

        #[cfg(not(target_arch = "wasm32"))]
        let encode_start = Instant::now();
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("rotation_{}", key_idx)),
        });

        // === Step 1: Automorphism on c0 and c1 ===
        encode_automorphism(
            ctx, &mut encoder,
            &workspace.current_c0, &workspace.c0_auto,
            k,
        );
        encode_automorphism(
            ctx, &mut encoder,
            &workspace.current_c1, &workspace.c1_auto,
            k,
        );

        // === Step 2: Key-switch c1_auto ===
        // Zero the accumulators
        encoder.clear_buffer(&workspace.ks_c0_acc, 0, Some(workspace.rns_buffer_size));
        encoder.clear_buffer(&workspace.ks_c1_acc, 0, Some(workspace.rns_buffer_size));

        // For each RNS limb and digit
        for limb_idx in 0..num_moduli.min(galois_key.keys_b.len()) {
            let offset = (limb_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

            // Copy limb from c1_auto
            encoder.copy_buffer_to_buffer(
                &workspace.c1_auto, offset,
                &workspace.limb_buffer, 0,
                workspace.single_mod_size,
            );

            for digit_idx in 0..workspace.digits_per_limb {
                // Extract digit
                encode_digit_decompose(
                    ctx, &mut encoder,
                    &workspace.limb_buffer,
                    &workspace.digit_buffer,
                    digit_idx as u32,
                );

                // Replicate digit to all moduli
                for mod_idx in 0..num_moduli {
                    let dest_offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;
                    encoder.copy_buffer_to_buffer(
                        &workspace.digit_buffer, 0,
                        &workspace.digit_rns, dest_offset,
                        workspace.single_mod_size,
                    );
                }

                // NTT multiply: term_b = digit * keys_b[limb][digit]
                // NTT multiply: term_a = digit * keys_a[limb][digit]
                // Using shared memory NTT for 4x faster encode time
                encode_ntt_mul_shared_mem(
                    ctx, &mut encoder, workspace,
                    &workspace.digit_rns,
                    &galois_key.keys_b[limb_idx][digit_idx],
                    &workspace.term_b,
                );
                encode_ntt_mul_shared_mem(
                    ctx, &mut encoder, workspace,
                    &workspace.digit_rns,
                    &galois_key.keys_a[limb_idx][digit_idx],
                    &workspace.term_a,
                );

                // Accumulate: ks_c0_acc += term_b, ks_c1_acc += term_a
                encode_add_inplace(ctx, &mut encoder, workspace, &workspace.ks_c0_acc, &workspace.term_b);
                encode_add_inplace(ctx, &mut encoder, workspace, &workspace.ks_c1_acc, &workspace.term_a);
            }
        }

        // === Step 3: Combine ===
        // new_c0 = c0_auto + ks_c0_acc
        encode_add(ctx, &mut encoder, &workspace.c0_auto, &workspace.ks_c0_acc, &workspace.temp_c0);
        // new_c1 = ks_c1_acc
        encoder.copy_buffer_to_buffer(
            &workspace.ks_c1_acc, 0,
            &workspace.temp_c1, 0,
            workspace.rns_buffer_size,
        );

        // === Step 4: Add rotated to current ===
        // current = current + rotated
        encode_add(ctx, &mut encoder, &workspace.current_c0, &workspace.temp_c0, &workspace.c0_auto);
        encode_add(ctx, &mut encoder, &workspace.current_c1, &workspace.temp_c1, &workspace.c1_auto);

        // Swap: current <- result
        encoder.copy_buffer_to_buffer(
            &workspace.c0_auto, 0,
            &workspace.current_c0, 0,
            workspace.rns_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &workspace.c1_auto, 0,
            &workspace.current_c1, 0,
            workspace.rns_buffer_size,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            encode_time += encode_start.elapsed();
        }

        // Submit this rotation's commands
        #[cfg(not(target_arch = "wasm32"))]
        let submit_start = Instant::now();
        ctx.queue.submit(Some(encoder.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        {
            submit_time += submit_start.elapsed();
        }

        #[cfg(not(target_arch = "wasm32"))]
        if profile {
            eprintln!("  Rotation {}: encode={:?}, submit={:?}, total={:?}",
                key_idx, encode_start.elapsed(), submit_start.elapsed(), rotation_start.elapsed());
        }
    }

    // Final copy: workspace -> result
    let result_c0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result_c0"),
        size: workspace.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let result_c1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result_c1"),
        size: workspace.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("final_copy"),
        });
        encoder.copy_buffer_to_buffer(
            &workspace.current_c0, 0,
            &result_c0, 0,
            workspace.rns_buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &workspace.current_c1, 0,
            &result_c1, 0,
            workspace.rns_buffer_size,
        );
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Wait for all operations to complete (native only - WASM handles this via browser)
    #[cfg(not(target_arch = "wasm32"))]
    let poll_start = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::Maintain::Wait);
    #[cfg(not(target_arch = "wasm32"))]
    let poll_time = poll_start.elapsed();

    // Profiling output (native only)
    #[cfg(not(target_arch = "wasm32"))]
    if profile {
        let total_time = total_start.elapsed();

        // Calculate actual dispatch count with shared memory NTT
        // Per NTT multiply: 4 dispatches per modulus (2 fwd + pointwise + inv)
        // Per rotation: 24 NTT muls × 3 moduli × 4 = 288 dispatches + automorphisms + adds
        let ntt_dispatches_per_rotation = num_moduli * workspace.digits_per_limb * 2 * num_moduli * 4;
        let other_dispatches = 2 + 2 + 4; // 2 auto + 2 clear + ~4 adds
        let total_dispatches = num_keys * (ntt_dispatches_per_rotation + other_dispatches);

        eprintln!("\n=== Sum Slots Profiling Summary ===");
        eprintln!();
        eprintln!("Pipeline timing (encode overlaps with GPU):");
        eprintln!("  Encode time (CPU command building): {:?}", encode_time);
        eprintln!("  Submit time (queue submission):     {:?}", submit_time);
        eprintln!("  Poll time (wait for GPU):           {:?}", poll_time);
        eprintln!("  Total wall-clock time:              {:?}", total_time);
        eprintln!();
        eprintln!("Analysis:");
        eprintln!("  If encode >> poll: CPU-bound (GPU starving)");
        eprintln!("  If poll >> encode: GPU-bound (GPU saturated) ✓");
        eprintln!("  Current: encode={:.0}ms, poll={:.0}ms",
            encode_time.as_secs_f64() * 1000.0,
            poll_time.as_secs_f64() * 1000.0);
        eprintln!();
        eprintln!("Dispatch count:");
        eprintln!("  - Rotations: {}", num_keys);
        eprintln!("  - Dispatches per rotation: ~{}", ntt_dispatches_per_rotation + other_dispatches);
        eprintln!("  - Total dispatches: ~{}", total_dispatches);
    }

    // Suppress unused variable warnings in WASM
    #[cfg(target_arch = "wasm32")]
    let _ = profile;

    Ok(GpuRnsCiphertext {
        c0: GpuRnsPoly {
            buffer: result_c0,
            n,
            num_moduli,
        },
        c1: GpuRnsPoly {
            buffer: result_c1,
            n,
            num_moduli,
        },
    })
}

/// Processes TWO ciphertexts in parallel to test GPU saturation.
/// Both ciphertexts' work is encoded in the same command buffer,
/// allowing GPU to execute them concurrently if it has spare capacity.
pub fn gpu_sum_slots_batched_2x(
    ctx: &GpuRotationContext,
    ct1: &GpuRnsCiphertext,
    ct2: &GpuRnsCiphertext,
    galois_keys: &GpuGaloisKeys,
    workspace1: &SumSlotsWorkspace,
    workspace2: &SumSlotsWorkspace,
) -> Result<(GpuRnsCiphertext, GpuRnsCiphertext), GpuError> {
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();
    let num_keys = galois_keys.keys.len();

    // Initial copy: input -> workspace for both ciphertexts
    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("init_copy_2x"),
        });
        // Ciphertext 1
        encoder.copy_buffer_to_buffer(&ct1.c0.buffer, 0, &workspace1.current_c0, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&ct1.c1.buffer, 0, &workspace1.current_c1, 0, workspace1.rns_buffer_size);
        // Ciphertext 2
        encoder.copy_buffer_to_buffer(&ct2.c0.buffer, 0, &workspace2.current_c0, 0, workspace2.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&ct2.c1.buffer, 0, &workspace2.current_c1, 0, workspace2.rns_buffer_size);
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Process each rotation - encode BOTH ciphertexts in same command buffer
    for key_idx in 0..num_keys {
        let galois_key = &galois_keys.keys[key_idx];
        let k = galois_key.k;

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("rotation_2x_{}", key_idx)),
        });

        // === Encode rotation for ciphertext 1 ===
        encode_automorphism(ctx, &mut encoder, &workspace1.current_c0, &workspace1.c0_auto, k);
        encode_automorphism(ctx, &mut encoder, &workspace1.current_c1, &workspace1.c1_auto, k);

        // === Encode rotation for ciphertext 2 (in SAME command buffer) ===
        encode_automorphism(ctx, &mut encoder, &workspace2.current_c0, &workspace2.c0_auto, k);
        encode_automorphism(ctx, &mut encoder, &workspace2.current_c1, &workspace2.c1_auto, k);

        // Zero accumulators for both
        encoder.clear_buffer(&workspace1.ks_c0_acc, 0, Some(workspace1.rns_buffer_size));
        encoder.clear_buffer(&workspace1.ks_c1_acc, 0, Some(workspace1.rns_buffer_size));
        encoder.clear_buffer(&workspace2.ks_c0_acc, 0, Some(workspace2.rns_buffer_size));
        encoder.clear_buffer(&workspace2.ks_c1_acc, 0, Some(workspace2.rns_buffer_size));

        // Key-switching for both ciphertexts
        for limb_idx in 0..num_moduli.min(galois_key.keys_b.len()) {
            let offset = (limb_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

            // Copy limbs for both
            encoder.copy_buffer_to_buffer(&workspace1.c1_auto, offset, &workspace1.limb_buffer, 0, workspace1.single_mod_size);
            encoder.copy_buffer_to_buffer(&workspace2.c1_auto, offset, &workspace2.limb_buffer, 0, workspace2.single_mod_size);

            for digit_idx in 0..workspace1.digits_per_limb {
                // Extract digits for both
                encode_digit_decompose(ctx, &mut encoder, &workspace1.limb_buffer, &workspace1.digit_buffer, digit_idx as u32);
                encode_digit_decompose(ctx, &mut encoder, &workspace2.limb_buffer, &workspace2.digit_buffer, digit_idx as u32);

                // Replicate digits to all moduli for both
                for mod_idx in 0..num_moduli {
                    let dest_offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;
                    encoder.copy_buffer_to_buffer(&workspace1.digit_buffer, 0, &workspace1.digit_rns, dest_offset, workspace1.single_mod_size);
                    encoder.copy_buffer_to_buffer(&workspace2.digit_buffer, 0, &workspace2.digit_rns, dest_offset, workspace2.single_mod_size);
                }

                // NTT multiplies for ciphertext 1
                encode_ntt_mul_shared_mem(ctx, &mut encoder, workspace1, &workspace1.digit_rns, &galois_key.keys_b[limb_idx][digit_idx], &workspace1.term_b);
                encode_ntt_mul_shared_mem(ctx, &mut encoder, workspace1, &workspace1.digit_rns, &galois_key.keys_a[limb_idx][digit_idx], &workspace1.term_a);

                // NTT multiplies for ciphertext 2 (GPU can run these in parallel!)
                encode_ntt_mul_shared_mem(ctx, &mut encoder, workspace2, &workspace2.digit_rns, &galois_key.keys_b[limb_idx][digit_idx], &workspace2.term_b);
                encode_ntt_mul_shared_mem(ctx, &mut encoder, workspace2, &workspace2.digit_rns, &galois_key.keys_a[limb_idx][digit_idx], &workspace2.term_a);

                // Accumulate for both
                encode_add_inplace(ctx, &mut encoder, workspace1, &workspace1.ks_c0_acc, &workspace1.term_b);
                encode_add_inplace(ctx, &mut encoder, workspace1, &workspace1.ks_c1_acc, &workspace1.term_a);
                encode_add_inplace(ctx, &mut encoder, workspace2, &workspace2.ks_c0_acc, &workspace2.term_b);
                encode_add_inplace(ctx, &mut encoder, workspace2, &workspace2.ks_c1_acc, &workspace2.term_a);
            }
        }

        // Combine for both
        encode_add(ctx, &mut encoder, &workspace1.c0_auto, &workspace1.ks_c0_acc, &workspace1.temp_c0);
        encode_add(ctx, &mut encoder, &workspace2.c0_auto, &workspace2.ks_c0_acc, &workspace2.temp_c0);
        encoder.copy_buffer_to_buffer(&workspace1.ks_c1_acc, 0, &workspace1.temp_c1, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace2.ks_c1_acc, 0, &workspace2.temp_c1, 0, workspace2.rns_buffer_size);

        // Add rotated to current for both
        encode_add(ctx, &mut encoder, &workspace1.current_c0, &workspace1.temp_c0, &workspace1.c0_auto);
        encode_add(ctx, &mut encoder, &workspace1.current_c1, &workspace1.temp_c1, &workspace1.c1_auto);
        encode_add(ctx, &mut encoder, &workspace2.current_c0, &workspace2.temp_c0, &workspace2.c0_auto);
        encode_add(ctx, &mut encoder, &workspace2.current_c1, &workspace2.temp_c1, &workspace2.c1_auto);

        // Swap for both
        encoder.copy_buffer_to_buffer(&workspace1.c0_auto, 0, &workspace1.current_c0, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace1.c1_auto, 0, &workspace1.current_c1, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace2.c0_auto, 0, &workspace2.current_c0, 0, workspace2.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace2.c1_auto, 0, &workspace2.current_c1, 0, workspace2.rns_buffer_size);

        ctx.queue.submit(Some(encoder.finish()));
    }

    // Create result buffers for both
    let result1_c0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result1_c0"), size: workspace1.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let result1_c1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result1_c1"), size: workspace1.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let result2_c0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result2_c0"), size: workspace2.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let result2_c1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result2_c1"), size: workspace2.rns_buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    // Final copy for both
    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("final_copy_2x") });
        encoder.copy_buffer_to_buffer(&workspace1.current_c0, 0, &result1_c0, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace1.current_c1, 0, &result1_c1, 0, workspace1.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace2.current_c0, 0, &result2_c0, 0, workspace2.rns_buffer_size);
        encoder.copy_buffer_to_buffer(&workspace2.current_c1, 0, &result2_c1, 0, workspace2.rns_buffer_size);
        ctx.queue.submit(Some(encoder.finish()));
    }

    ctx.device.poll(wgpu::Maintain::Wait);

    Ok((
        GpuRnsCiphertext { c0: GpuRnsPoly { buffer: result1_c0, n, num_moduli }, c1: GpuRnsPoly { buffer: result1_c1, n, num_moduli } },
        GpuRnsCiphertext { c0: GpuRnsPoly { buffer: result2_c0, n, num_moduli }, c1: GpuRnsPoly { buffer: result2_c1, n, num_moduli } },
    ))
}

/// Encodes automorphism σ_k into the command encoder (no submit).
fn encode_automorphism(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    k: usize,
) {
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();

    let params = AutoParams {
        n: n as u32,
        two_n: (2 * n) as u32,
        k: k as u32,
        num_moduli: num_moduli as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("auto params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("auto bind group"),
        layout: &ctx.auto_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ctx.moduli_buffer.as_entire_binding() },
        ],
    });

    let workgroups = ((n * num_moduli) as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("auto pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.auto_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes digit decomposition into the command encoder.
fn encode_digit_decompose(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    input: &Buffer,
    output: &Buffer,
    digit_idx: u32,
) {
    let n = ctx.params.n;
    let decomp_base_log = (ctx.params.decomp_base as f64).log2() as u32;

    let params = DecomposeParams {
        n: n as u32,
        digit_idx,
        decomp_base_log,
        _pad: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("decompose params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decompose bind group"),
        layout: &ctx.digit_decompose_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("decompose pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.digit_decompose_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes RNS polynomial addition into the command encoder.
fn encode_add(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
) {
    let n = ctx.params.n;
    let num_moduli = ctx.params.moduli.len();

    let params = AddParams {
        n: n as u32,
        num_moduli: num_moduli as u32,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("add params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add bind group"),
        layout: &ctx.add_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: result.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ctx.moduli_buffer.as_entire_binding() },
        ],
    });

    let workgroups = ((n * num_moduli) as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("add pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.add_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes in-place addition (a += b) using a temporary buffer.
/// WebGPU doesn't allow using the same buffer for both read and write in one dispatch,
/// so we use: temp = a + b; copy temp -> a
fn encode_add_inplace(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    workspace: &SumSlotsWorkspace,
    a: &Buffer,
    b: &Buffer,
) {
    // a + b -> temp
    encode_add(ctx, encoder, a, b, &workspace.add_temp);
    // copy temp -> a
    encoder.copy_buffer_to_buffer(
        &workspace.add_temp, 0,
        a, 0,
        workspace.rns_buffer_size,
    );
}

/// Encodes NTT multiplication for RNS polynomials.
fn encode_ntt_mul_rns(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    workspace: &SumSlotsWorkspace,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
) {
    let n = workspace.n;
    let num_moduli = workspace.num_moduli;
    let log_n = workspace.log_n;

    // Process each modulus
    for mod_idx in 0..num_moduli {
        let offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

        // Copy inputs to working buffers
        encoder.copy_buffer_to_buffer(
            a, offset,
            &workspace.ntt_a_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );
        encoder.copy_buffer_to_buffer(
            b, offset,
            &workspace.ntt_b_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );

        // Forward NTT on a
        encode_twist(ctx, encoder, &workspace.ntt_a_twisted[mod_idx], mod_idx, false);
        encode_bit_reverse(ctx, encoder, &workspace.ntt_a_twisted[mod_idx], &workspace.ntt_a_out[mod_idx]);
        for stage in 0..log_n {
            encode_ntt_butterfly(ctx, encoder, &workspace.ntt_a_out[mod_idx], mod_idx, stage, false);
        }

        // Forward NTT on b
        encode_twist(ctx, encoder, &workspace.ntt_b_twisted[mod_idx], mod_idx, false);
        encode_bit_reverse(ctx, encoder, &workspace.ntt_b_twisted[mod_idx], &workspace.ntt_b_out[mod_idx]);
        for stage in 0..log_n {
            encode_ntt_butterfly(ctx, encoder, &workspace.ntt_b_out[mod_idx], mod_idx, stage, false);
        }

        // Pointwise multiply
        encode_pointwise_mul(
            ctx, encoder,
            &workspace.ntt_a_out[mod_idx],
            &workspace.ntt_b_out[mod_idx],
            &workspace.ntt_result[mod_idx],
            mod_idx,
        );

        // Inverse NTT
        encode_bit_reverse(ctx, encoder, &workspace.ntt_result[mod_idx], &workspace.ntt_result_inv[mod_idx]);
        for stage in 0..log_n {
            encode_ntt_butterfly(ctx, encoder, &workspace.ntt_result_inv[mod_idx], mod_idx, stage, true);
        }
        encode_scale(ctx, encoder, &workspace.ntt_result_inv[mod_idx], mod_idx);
        encode_twist(ctx, encoder, &workspace.ntt_result_inv[mod_idx], mod_idx, true);

        // Copy result back
        encoder.copy_buffer_to_buffer(
            &workspace.ntt_result_inv[mod_idx], 0,
            result, offset,
            workspace.single_mod_size,
        );
    }
}

/// Fast NTT multiplication using pre-allocated bind groups.
/// This eliminates the ~50μs overhead of creating params buffers and bind groups per dispatch.
fn encode_ntt_mul_rns_fast(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    workspace: &SumSlotsWorkspace,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
) {
    let n = workspace.n;
    let num_moduli = workspace.num_moduli;
    let log_n = workspace.log_n;
    let twist_workgroups = (n as u32 + 255) / 256;
    let butterfly_workgroups = ((n / 2) as u32 + 255) / 256;

    // Process each modulus
    for mod_idx in 0..num_moduli {
        let offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

        // Copy inputs to working buffers
        encoder.copy_buffer_to_buffer(
            a, offset,
            &workspace.ntt_a_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );
        encoder.copy_buffer_to_buffer(
            b, offset,
            &workspace.ntt_b_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );

        // Forward NTT on a: twist
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twist_a_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.twist_pipeline);
            pass.set_bind_group(0, &workspace.twist_bg_a_fwd[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Forward NTT on a: bit-reverse
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitrev_a"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.bit_rev_pipeline);
            pass.set_bind_group(0, &workspace.bitrev_bg_a[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Forward NTT on a: butterflies
        for stage in 0..log_n as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("butterfly_a_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.ntt_butterfly_pipeline);
            pass.set_bind_group(0, &workspace.butterfly_bg_a_fwd[mod_idx][stage], &[]);
            pass.dispatch_workgroups(butterfly_workgroups, 1, 1);
        }

        // Forward NTT on b: twist
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twist_b_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.twist_pipeline);
            pass.set_bind_group(0, &workspace.twist_bg_b_fwd[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Forward NTT on b: bit-reverse
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitrev_b"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.bit_rev_pipeline);
            pass.set_bind_group(0, &workspace.bitrev_bg_b[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Forward NTT on b: butterflies
        for stage in 0..log_n as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("butterfly_b_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.ntt_butterfly_pipeline);
            pass.set_bind_group(0, &workspace.butterfly_bg_b_fwd[mod_idx][stage], &[]);
            pass.dispatch_workgroups(butterfly_workgroups, 1, 1);
        }

        // Pointwise multiply
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pointwise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pointwise_mul_pipeline);
            pass.set_bind_group(0, &workspace.pointwise_bg[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Inverse NTT: bit-reverse
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitrev_result"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.bit_rev_pipeline);
            pass.set_bind_group(0, &workspace.bitrev_bg_result[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Inverse NTT: butterflies
        for stage in 0..log_n as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("butterfly_result_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.ntt_butterfly_pipeline);
            pass.set_bind_group(0, &workspace.butterfly_bg_result_inv[mod_idx][stage], &[]);
            pass.dispatch_workgroups(butterfly_workgroups, 1, 1);
        }

        // Inverse NTT: scale
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.scale_pipeline);
            pass.set_bind_group(0, &workspace.scale_bg[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Inverse NTT: untwist
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twist_result_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.twist_pipeline);
            pass.set_bind_group(0, &workspace.twist_bg_result_inv[mod_idx], &[]);
            pass.dispatch_workgroups(twist_workgroups, 1, 1);
        }

        // Copy result back
        encoder.copy_buffer_to_buffer(
            &workspace.ntt_result_inv[mod_idx], 0,
            result, offset,
            workspace.single_mod_size,
        );
    }
}

/// Fused NTT multiply - processes all moduli in single dispatches.
/// This reduces dispatch count from 141 to 47 per NTT multiply (3× reduction).
fn encode_ntt_mul_rns_fused(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    workspace: &SumSlotsWorkspace,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
) {
    let n = workspace.n;
    let num_moduli = workspace.num_moduli;
    let log_n = workspace.log_n;

    // Fused workgroups: process n * num_moduli elements
    let fused_twist_workgroups = ((n * num_moduli) as u32 + 255) / 256;
    // For butterfly: half_n * num_moduli threads
    let fused_butterfly_workgroups = (((n / 2) * num_moduli) as u32 + 255) / 256;

    // Copy inputs to fused working buffers
    encoder.copy_buffer_to_buffer(
        a, 0,
        &workspace.fused_a_twisted, 0,
        workspace.rns_buffer_size,
    );
    encoder.copy_buffer_to_buffer(
        b, 0,
        &workspace.fused_b_twisted, 0,
        workspace.rns_buffer_size,
    );

    // Forward NTT on a: fused twist
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_twist_a_fwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_twist_pipeline);
        pass.set_bind_group(0, &workspace.fused_twist_a_fwd_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Forward NTT on a: fused bit-reverse
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_bitrev_a"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_bitrev_pipeline);
        pass.set_bind_group(0, &workspace.fused_bitrev_a_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Forward NTT on a: fused butterflies
    for stage in 0..log_n as usize {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_butterfly_a_fwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_butterfly_pipeline);
        pass.set_bind_group(0, &workspace.fused_butterfly_a_fwd_bg[stage], &[]);
        pass.dispatch_workgroups(fused_butterfly_workgroups, 1, 1);
    }

    // Forward NTT on b: fused twist
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_twist_b_fwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_twist_pipeline);
        pass.set_bind_group(0, &workspace.fused_twist_b_fwd_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Forward NTT on b: fused bit-reverse
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_bitrev_b"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_bitrev_pipeline);
        pass.set_bind_group(0, &workspace.fused_bitrev_b_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Forward NTT on b: fused butterflies
    for stage in 0..log_n as usize {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_butterfly_b_fwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_butterfly_pipeline);
        pass.set_bind_group(0, &workspace.fused_butterfly_b_fwd_bg[stage], &[]);
        pass.dispatch_workgroups(fused_butterfly_workgroups, 1, 1);
    }

    // Fused pointwise multiply
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_pointwise"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_pointwise_pipeline);
        pass.set_bind_group(0, &workspace.fused_pointwise_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Inverse NTT: fused bit-reverse
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_bitrev_result"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_bitrev_pipeline);
        pass.set_bind_group(0, &workspace.fused_bitrev_result_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Inverse NTT: fused butterflies
    for stage in 0..log_n as usize {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_butterfly_result_inv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_butterfly_pipeline);
        pass.set_bind_group(0, &workspace.fused_butterfly_result_inv_bg[stage], &[]);
        pass.dispatch_workgroups(fused_butterfly_workgroups, 1, 1);
    }

    // Inverse NTT: fused scale
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_scale"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_scale_pipeline);
        pass.set_bind_group(0, &workspace.fused_scale_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Inverse NTT: fused untwist
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_twist_result_inv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.fused_twist_pipeline);
        pass.set_bind_group(0, &workspace.fused_twist_result_inv_bg, &[]);
        pass.dispatch_workgroups(fused_twist_workgroups, 1, 1);
    }

    // Copy result back
    encoder.copy_buffer_to_buffer(
        &workspace.fused_result_inv, 0,
        result, 0,
        workspace.rns_buffer_size,
    );
}

/// Encodes twist operation (multiply by psi powers).
fn encode_twist(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    buffer: &Buffer,
    mod_idx: usize,
    inverse: bool,
) {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = TwistParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        _pad0: 0,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("twist params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let psi_buffer = if inverse {
        &ctx.psi_inv_power_buffers[mod_idx]
    } else {
        &ctx.psi_power_buffers[mod_idx]
    };

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("twist bind group"),
        layout: &ctx.twist_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: psi_buffer.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("twist pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.twist_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes bit-reverse permutation.
fn encode_bit_reverse(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    input: &Buffer,
    output: &Buffer,
) {
    let n = ctx.params.n;
    let log_n = (n as f64).log2() as u32;

    let params = BitRevParams {
        n: n as u32,
        log_n,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitrev params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bitrev bind group"),
        layout: &ctx.bit_rev_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("bitrev pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.bit_rev_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes NTT butterfly stage.
fn encode_ntt_butterfly(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    buffer: &Buffer,
    mod_idx: usize,
    stage: u32,
    inverse: bool,
) {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = NttButterflyParams {
        n: n as u32,
        stage,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ntt params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let twiddle_buffer = if inverse {
        &ctx.inv_twiddle_buffers[mod_idx]
    } else {
        &ctx.twiddle_buffers[mod_idx]
    };

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ntt bind group"),
        layout: &ctx.ntt_butterfly_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: twiddle_buffer.as_entire_binding() },
        ],
    });

    let workgroups = ((n / 2) as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("ntt pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.ntt_butterfly_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes pointwise multiplication.
fn encode_pointwise_mul(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
    mod_idx: usize,
) {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = PointwiseParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        _pad0: 0,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pointwise params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pointwise bind group"),
        layout: &ctx.pointwise_mul_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: result.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pointwise pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.pointwise_mul_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes scale by n_inv operation.
fn encode_scale(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    buffer: &Buffer,
    mod_idx: usize,
) {
    let n = ctx.params.n;
    let ntt = &ctx.ntt_data[mod_idx];

    let params = ScaleParams {
        n: n as u32,
        modulus_lo: ntt.modulus as u32,
        modulus_hi: (ntt.modulus >> 32) as u32,
        scalar_lo: ntt.n_inv as u32,
        scalar_hi: (ntt.n_inv >> 32) as u32,
        mu_lo_lo: ntt.mu_lo as u32,
        mu_lo_hi: (ntt.mu_lo >> 32) as u32,
        mu_hi_lo: ntt.mu_hi as u32,
        mu_hi_hi: (ntt.mu_hi >> 32) as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scale params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale bind group"),
        layout: &ctx.scale_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer.as_entire_binding() },
        ],
    });

    let workgroups = (n as u32 + 255) / 256;

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("scale pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ctx.scale_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Encodes a full NTT using shared memory (single dispatch for all 13 stages).
/// This is much faster than 13 separate butterfly dispatches.
///
/// Performs: data = NTT(data) or data = INTT(data) depending on is_inverse.
/// The data buffer is modified in-place.
#[allow(dead_code)]
fn encode_shared_mem_ntt(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    data: &Buffer,
    mod_idx: usize,
    is_inverse: bool,
) {
    let n = ctx.params.n;
    let log_n = (n as f64).log2() as u32;
    let ntt = &ctx.ntt_data[mod_idx];

    // Create params buffer (same for both forward and inverse)
    let params = SharedMemNttParams {
        n: n as u32,
        log_n,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shared mem ntt params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    // q as vec2<u32>
    let q_data = [ntt.modulus as u32, (ntt.modulus >> 32) as u32];
    let q_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q buffer"),
        contents: bytemuck::cast_slice(&q_data),
        usage: BufferUsages::UNIFORM,
    });

    // Barrett params as vec4<u32>
    let barrett_data = [
        ntt.mu_lo as u32,
        (ntt.mu_lo >> 32) as u32,
        ntt.mu_hi as u32,
        (ntt.mu_hi >> 32) as u32,
    ];
    let barrett_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("barrett buffer"),
        contents: bytemuck::cast_slice(&barrett_data),
        usage: BufferUsages::UNIFORM,
    });

    if is_inverse {
        // Inverse NTT: bindings are 0=params, 1=data, 2=inv_twiddles, 3=inv_psi, 4=n_inv, 5=q, 6=barrett
        let n_inv_data = [ntt.n_inv as u32, (ntt.n_inv >> 32) as u32];
        let n_inv_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("n_inv buffer"),
            contents: bytemuck::cast_slice(&n_inv_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared mem ntt inv bind group"),
            layout: &ctx.shared_mem_ntt_inv_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.inv_twiddle_buffers[mod_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.psi_inv_power_buffers[mod_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: n_inv_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: q_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: barrett_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shared mem ntt inv pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.shared_mem_ntt_inv_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    } else {
        // Forward NTT: bindings are 0=params, 1=data, 2=twiddles, 3=psi, 4=q, 5=barrett
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared mem ntt fwd bind group"),
            layout: &ctx.shared_mem_ntt_fwd_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.twiddle_buffers[mod_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctx.psi_power_buffers[mod_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: q_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: barrett_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shared mem ntt fwd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.shared_mem_ntt_fwd_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// Encodes NTT multiply using shared memory NTT with pre-allocated bind groups.
/// Uses 4 dispatches per modulus (2 forward NTT + pointwise + inverse NTT).
/// Performs: result = INTT(NTT(a) * NTT(b))
fn encode_ntt_mul_shared_mem(
    ctx: &GpuRotationContext,
    encoder: &mut wgpu::CommandEncoder,
    workspace: &SumSlotsWorkspace,
    a: &Buffer,
    b: &Buffer,
    result: &Buffer,
) {
    let n = workspace.n;
    let num_moduli = workspace.num_moduli;

    for mod_idx in 0..num_moduli {
        let offset = (mod_idx * n * 2 * std::mem::size_of::<u32>()) as u64;

        // Copy inputs to working buffers
        encoder.copy_buffer_to_buffer(
            a, offset,
            &workspace.ntt_a_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );
        encoder.copy_buffer_to_buffer(
            b, offset,
            &workspace.ntt_b_twisted[mod_idx], 0,
            workspace.single_mod_size,
        );

        // Forward NTT on a (1 dispatch, pre-allocated bind group)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shared_mem_ntt_fwd_a"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.shared_mem_ntt_fwd_pipeline);
            pass.set_bind_group(0, &workspace.shared_mem_ntt_fwd_a_bg[mod_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Forward NTT on b (1 dispatch, pre-allocated bind group)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shared_mem_ntt_fwd_b"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.shared_mem_ntt_fwd_pipeline);
            pass.set_bind_group(0, &workspace.shared_mem_ntt_fwd_b_bg[mod_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pointwise multiply (pre-allocated bind group)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pointwise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pointwise_mul_pipeline);
            pass.set_bind_group(0, &workspace.pointwise_bg[mod_idx], &[]);
            let workgroups = (n as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Inverse NTT on result (1 dispatch, pre-allocated bind group)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shared_mem_ntt_inv_result"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.shared_mem_ntt_inv_pipeline);
            pass.set_bind_group(0, &workspace.shared_mem_ntt_inv_result_bg[mod_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result back
        encoder.copy_buffer_to_buffer(
            &workspace.ntt_result[mod_idx], 0,
            result, offset,
            workspace.single_mod_size,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_automorphism() {
        let params = GpuRnsParams {
            n: 8,
            moduli: vec![17, 19], // Small primes for testing
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        };

        let ctx = match GpuRotationContext::new(params) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: {:?}", e);
                return;
            }
        };

        // Create test polynomial: [0, 1, 2, 3, 4, 5, 6, 7] for each modulus
        let residues = vec![
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 6, 7],
        ];

        let poly = GpuRnsPoly::from_residues(&ctx, &residues).unwrap();

        // Apply σ_5: X → X^5
        let result = gpu_automorphism(&ctx, &poly, 5).unwrap();
        let result_residues = result.to_residues(&ctx).unwrap();

        println!("Input: {:?}", residues[0]);
        println!("After σ_5: {:?}", result_residues[0]);
    }

    #[test]
    fn test_gpu_add() {
        let params = GpuRnsParams {
            n: 4,
            moduli: vec![17],
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        };

        let ctx = match GpuRotationContext::new(params) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: {:?}", e);
                return;
            }
        };

        let a_residues = vec![vec![1, 2, 3, 4]];
        let b_residues = vec![vec![5, 6, 7, 8]];

        let a = GpuRnsPoly::from_residues(&ctx, &a_residues).unwrap();
        let b = GpuRnsPoly::from_residues(&ctx, &b_residues).unwrap();

        let result = gpu_add(&ctx, &a, &b).unwrap();
        let result_residues = result.to_residues(&ctx).unwrap();

        println!("a: {:?}", a_residues[0]);
        println!("b: {:?}", b_residues[0]);
        println!("a + b: {:?}", result_residues[0]);

        // Expected: [6, 8, 10, 12] mod 17 = [6, 8, 10, 12]
        assert_eq!(result_residues[0], vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_ntt_precomputation() {
        // Verify NTT precomputation is correct for a simple case
        let n = 8;
        let q = 17u64;

        let psi = find_primitive_root(n, q).expect("should find root");
        let data = NttModulusData::new(n, q, psi);

        println!("Testing NTT precomputation for n={}, q={}", n, q);
        println!("psi = {}", data.psi);
        println!("omega = {}", data.omega);
        println!("psi_inv = {}", data.psi_inv);
        println!("n_inv = {} (should be {} * {} ≡ 1 mod {})",
            data.n_inv, n, data.n_inv, q);

        // Verify psi is primitive 2n-th root
        assert_eq!(mod_pow(data.psi, n as u64, q), q - 1, "psi^n should be -1");
        assert_eq!(mod_pow(data.psi, 2 * n as u64, q), 1, "psi^(2n) should be 1");

        // Verify omega is primitive n-th root
        assert_eq!(mod_pow(data.omega, n as u64, q), 1, "omega^n should be 1");

        // Verify inverses
        assert_eq!(mod_mul(data.psi, data.psi_inv, q), 1, "psi * psi_inv should be 1");
        assert_eq!(mod_mul(data.omega, data.omega_inv, q), 1, "omega * omega_inv should be 1");
        assert_eq!(mod_mul(n as u64, data.n_inv, q), 1, "n * n_inv should be 1");

        // Verify twiddle factors
        for i in 0..n {
            let expected = mod_pow(data.omega, i as u64, q);
            assert_eq!(data.twiddles[i], expected, "twiddle[{}] mismatch", i);
        }

        println!("NTT precomputation verified successfully!");
    }

    #[test]
    fn test_gpu_ntt_mul() {
        // Use a larger prime that is NTT-friendly: q ≡ 1 (mod 2n)
        // For n=64, we need q ≡ 1 (mod 128)
        // 257 = 2*128 + 1, so 257 ≡ 1 (mod 128)
        let params = GpuRnsParams {
            n: 64,
            moduli: vec![257], // Prime with 257 ≡ 1 (mod 128)
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        };

        let ctx = match GpuRotationContext::new(params) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: {:?}", e);
                return;
            }
        };

        // First, verify simple data roundtrip
        println!("=== Step 1: Data roundtrip test ===");
        let mut test_coeffs = vec![0u64; 64];
        test_coeffs[0] = 1;
        test_coeffs[1] = 1;
        let test_residues = vec![test_coeffs.clone()];
        let test_poly = GpuRnsPoly::from_residues(&ctx, &test_residues).unwrap();
        let roundtrip = test_poly.to_residues(&ctx).unwrap();
        println!("Input: {:?}", &test_coeffs[..4]);
        println!("Roundtrip: {:?}", &roundtrip[0][..4]);
        assert_eq!(roundtrip[0][0], 1, "roundtrip coeff 0");
        assert_eq!(roundtrip[0][1], 1, "roundtrip coeff 1");
        println!("Data roundtrip: OK");

        // Test: (1 + X) * (1 + X) = 1 + 2X + X^2 in Z_257[X]/(X^64 + 1)
        println!("\n=== Step 2: NTT multiplication ===");
        let mut a_coeffs = vec![0u64; 64];
        a_coeffs[0] = 1; // constant term
        a_coeffs[1] = 1; // X term

        let a_residues = vec![a_coeffs.clone()];
        let a = GpuRnsPoly::from_residues(&ctx, &a_residues).unwrap();

        let result = gpu_ntt_mul(&ctx, &a, &a).unwrap();
        let result_residues = result.to_residues(&ctx).unwrap();

        println!("Input (1 + X): {:?}", &a_coeffs[..4]);
        println!("Result (1 + X)^2: {:?}", &result_residues[0][..8]);

        // Verify: (1 + X)^2 = 1 + 2X + X^2
        assert_eq!(result_residues[0][0], 1, "constant term");
        assert_eq!(result_residues[0][1], 2, "X term");
        assert_eq!(result_residues[0][2], 1, "X^2 term");
        assert_eq!(result_residues[0][3], 0, "X^3 term should be 0");
    }

    #[test]
    fn test_gpu_ntt_debug_steps() {
        // Detailed debug test to isolate NTT pipeline issues
        use wgpu::BufferUsages;

        let n = 8; // Very small for easier debugging
        let q = 17u64; // Small prime: 17 ≡ 1 (mod 16) supports n=8

        let params = GpuRnsParams {
            n,
            moduli: vec![q],
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        };

        let ctx = match GpuRotationContext::new(params) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: {:?}", e);
                return;
            }
        };

        let ntt = &ctx.ntt_data[0];
        println!("=== NTT Debug Test (n={}, q={}) ===", n, q);
        println!("psi = {}", ntt.psi);
        println!("omega = {}", ntt.omega);
        println!("psi_inv = {}", ntt.psi_inv);
        println!("n_inv = {}", ntt.n_inv);
        println!("mu_lo = 0x{:016X}", ntt.mu_lo);
        println!("mu_hi = 0x{:016X}", ntt.mu_hi);
        println!("psi_powers: {:?}", ntt.psi_powers);
        println!("twiddles: {:?}", ntt.twiddles);

        // Create simple input: [1, 0, 0, 0, 0, 0, 0, 0]
        let input: Vec<u64> = vec![1, 0, 0, 0, 0, 0, 0, 0];
        println!("\nInput polynomial: {:?}", input);

        // Upload to GPU
        let input_u32: Vec<u32> = input.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let buffer_size = (n * 2 * std::mem::size_of::<u32>()) as u64;

        let input_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(&input_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Verify data uploaded correctly
        let readback = read_buffer_to_u64(&ctx, &input_buf, n);
        println!("After upload: {:?}", readback);
        assert_eq!(readback, input, "upload roundtrip failed");

        // Test twist (multiply by psi^i)
        println!("\n--- Testing twist (multiply by psi^i) ---");
        apply_twist(&ctx, &input_buf, 0, false).unwrap();
        let after_twist = read_buffer_to_u64(&ctx, &input_buf, n);
        println!("After twist: {:?}", after_twist);

        // Expected: [1*psi^0, 0*psi^1, ...] = [1, 0, 0, 0, 0, 0, 0, 0] (since input is [1,0,...])
        // But input[0] * psi_powers[0] = 1 * 1 = 1
        // So after twist, we should have [1, 0, 0, 0, 0, 0, 0, 0]
        assert_eq!(after_twist[0], 1, "twist: coeff 0 should be 1");
        for i in 1..n {
            assert_eq!(after_twist[i], 0, "twist: coeff {} should be 0", i);
        }
        println!("Twist: OK");

        // Test untwist (multiply by psi_inv^i)
        println!("\n--- Testing untwist (multiply by psi_inv^i) ---");
        apply_twist(&ctx, &input_buf, 0, true).unwrap();
        let after_untwist = read_buffer_to_u64(&ctx, &input_buf, n);
        println!("After untwist: {:?}", after_untwist);

        // After twist then untwist, should be back to original
        assert_eq!(after_untwist, input, "twist/untwist roundtrip failed");
        println!("Untwist: OK");

        // Test bit-reverse permutation
        println!("\n--- Testing bit-reverse ---");
        let br_input: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let br_input_u32: Vec<u32> = br_input.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let br_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("br_input"),
            contents: bytemuck::cast_slice(&br_input_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let br_output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("br_output"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        apply_bit_reverse(&ctx, &br_buf, &br_output_buf).unwrap();
        let br_result = read_buffer_to_u64(&ctx, &br_output_buf, n);
        println!("Bit-reverse input: {:?}", br_input);
        println!("Bit-reverse output: {:?}", br_result);
        // For n=8, log_n=3, bit-reverse permutation is:
        // 0 (000) -> 0 (000), 1 (001) -> 4 (100), 2 (010) -> 2 (010), 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001), 5 (101) -> 5 (101), 6 (110) -> 3 (011), 7 (111) -> 7 (111)
        let expected_br: Vec<u64> = vec![0, 4, 2, 6, 1, 5, 3, 7];
        assert_eq!(br_result, expected_br, "bit-reverse failed");
        println!("Bit-reverse: OK");

        // Test a single NTT butterfly stage
        println!("\n--- Testing NTT butterfly (stage 0, forward) ---");
        // Create input: [1, 2, 3, 4, 5, 6, 7, 8] - simple increasing values
        let butterfly_input: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let butterfly_input_u32: Vec<u32> = butterfly_input.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let butterfly_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("butterfly_input"),
            contents: bytemuck::cast_slice(&butterfly_input_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let before = read_buffer_to_u64(&ctx, &butterfly_buf, n);
        println!("Before butterfly stage 0: {:?}", before);

        apply_ntt_butterfly(&ctx, &butterfly_buf, 0, 0, false).unwrap();
        let after_s0 = read_buffer_to_u64(&ctx, &butterfly_buf, n);
        println!("After butterfly stage 0: {:?}", after_s0);

        // Stage 0: m=2, half_m=1. Butterflies at (0,1), (2,3), (4,5), (6,7)
        // Twiddle for j=0: omega^(0 * 8 / 2) = omega^0 = 1
        // Butterfly (0,1): t = 1 * 2 = 2; data[0] = 1+2=3, data[1] = 1-2=-1=16 (mod 17)
        // Butterfly (2,3): t = 1 * 4 = 4; data[2] = 3+4=7, data[3] = 3-4=-1=16 (mod 17)
        // Butterfly (4,5): t = 1 * 6 = 6; data[4] = 5+6=11, data[5] = 5-6=-1=16 (mod 17)
        // Butterfly (6,7): t = 1 * 8 = 8; data[6] = 7+8=15, data[7] = 7-8=-1=16 (mod 17)
        let expected_s0: Vec<u64> = vec![3, 16, 7, 16, 11, 16, 15, 16];
        println!("Expected after stage 0: {:?}", expected_s0);
        if after_s0 != expected_s0 {
            println!("MISMATCH at stage 0!");
            for i in 0..n {
                if after_s0[i] != expected_s0[i] {
                    println!("  Index {}: got {}, expected {}", i, after_s0[i], expected_s0[i]);
                }
            }
        }
        assert_eq!(after_s0, expected_s0, "butterfly stage 0 failed");
        println!("Butterfly stage 0: OK");

        // Continue with stages 1 and 2
        println!("\n--- Testing NTT butterfly (stage 1, forward) ---");
        apply_ntt_butterfly(&ctx, &butterfly_buf, 0, 1, false).unwrap();
        let after_s1 = read_buffer_to_u64(&ctx, &butterfly_buf, n);
        println!("After butterfly stage 1: {:?}", after_s1);

        println!("\n--- Testing NTT butterfly (stage 2, forward) ---");
        apply_ntt_butterfly(&ctx, &butterfly_buf, 0, 2, false).unwrap();
        let after_s2 = read_buffer_to_u64(&ctx, &butterfly_buf, n);
        println!("After butterfly stage 2 (final NTT): {:?}", after_s2);

        // Test pointwise multiplication
        println!("\n--- Testing pointwise multiplication ---");
        // Create two simple inputs and multiply them
        let pw_a: Vec<u64> = vec![2, 0, 0, 0, 0, 0, 0, 0];
        let pw_b: Vec<u64> = vec![3, 0, 0, 0, 0, 0, 0, 0];
        let pw_a_u32: Vec<u32> = pw_a.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let pw_b_u32: Vec<u32> = pw_b.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let pw_a_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pw_a"),
            contents: bytemuck::cast_slice(&pw_a_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let pw_b_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pw_b"),
            contents: bytemuck::cast_slice(&pw_b_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let pw_result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pw_result"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        apply_pointwise_mul(&ctx, &pw_a_buf, &pw_b_buf, &pw_result_buf, 0).unwrap();
        let pw_result = read_buffer_to_u64(&ctx, &pw_result_buf, n);
        println!("Pointwise a: {:?}", pw_a);
        println!("Pointwise b: {:?}", pw_b);
        println!("Pointwise result: {:?}", pw_result);
        // Expected: 2 * 3 = 6 mod 17
        assert_eq!(pw_result[0], 6, "pointwise mul: 2*3 should be 6");
        for i in 1..n {
            assert_eq!(pw_result[i], 0, "pointwise mul: coeff {} should be 0", i);
        }
        println!("Pointwise multiplication: OK");

        // Test scale operation
        println!("\n--- Testing scale operation ---");
        let scale_input: Vec<u64> = vec![8, 16, 4, 2, 1, 9, 13, 15];
        let scale_input_u32: Vec<u32> = scale_input.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let scale_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scale_input"),
            contents: bytemuck::cast_slice(&scale_input_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let n_inv = ntt.n_inv;
        println!("scale_input: {:?}", scale_input);
        println!("n_inv = {}", n_inv);
        apply_scale(&ctx, &scale_buf, 0, n_inv).unwrap();
        let scale_result = read_buffer_to_u64(&ctx, &scale_buf, n);
        println!("After scale by n_inv: {:?}", scale_result);

        // Expected: each value * n_inv mod 17, where n_inv = 15
        // 8*15 mod 17 = 120 mod 17 = 1
        // 16*15 mod 17 = 240 mod 17 = 2
        // etc.
        let expected_scale: Vec<u64> = scale_input.iter()
            .map(|&x| (x * n_inv) % q)
            .collect();
        println!("Expected scale: {:?}", expected_scale);
        assert_eq!(scale_result, expected_scale, "scale operation failed");
        println!("Scale operation: OK");

        // Test full NTT roundtrip: forward NTT then inverse NTT
        println!("\n--- Testing full NTT roundtrip ---");
        let roundtrip_input: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let roundtrip_input_u32: Vec<u32> = roundtrip_input.iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();

        let rt_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt_input"),
            contents: bytemuck::cast_slice(&roundtrip_input_u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let rt_ntt_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt_ntt"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rt_result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt_result"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        println!("Roundtrip input: {:?}", roundtrip_input);

        // Step 1: Twist
        apply_twist(&ctx, &rt_buf, 0, false).unwrap();
        let after_rt_twist = read_buffer_to_u64(&ctx, &rt_buf, n);
        println!("After twist: {:?}", after_rt_twist);

        // Step 2: Bit-reverse into NTT buffer
        apply_bit_reverse(&ctx, &rt_buf, &rt_ntt_buf).unwrap();
        let after_rt_br = read_buffer_to_u64(&ctx, &rt_ntt_buf, n);
        println!("After bit-reverse: {:?}", after_rt_br);

        // Step 3: Forward NTT butterflies
        for stage in 0..3 {
            apply_ntt_butterfly(&ctx, &rt_ntt_buf, 0, stage, false).unwrap();
        }
        let after_rt_ntt = read_buffer_to_u64(&ctx, &rt_ntt_buf, n);
        println!("After forward NTT: {:?}", after_rt_ntt);

        // Step 4: Bit-reverse for inverse
        apply_bit_reverse(&ctx, &rt_ntt_buf, &rt_result_buf).unwrap();

        // Step 5: Inverse NTT butterflies
        for stage in 0..3 {
            apply_ntt_butterfly(&ctx, &rt_result_buf, 0, stage, true).unwrap();
        }
        let after_rt_intt = read_buffer_to_u64(&ctx, &rt_result_buf, n);
        println!("After inverse NTT (before scale): {:?}", after_rt_intt);

        // Step 6: Scale by n_inv
        apply_scale(&ctx, &rt_result_buf, 0, n_inv).unwrap();
        let after_rt_scale = read_buffer_to_u64(&ctx, &rt_result_buf, n);
        println!("After scale: {:?}", after_rt_scale);

        // Step 7: Untwist
        apply_twist(&ctx, &rt_result_buf, 0, true).unwrap();
        let rt_final = read_buffer_to_u64(&ctx, &rt_result_buf, n);
        println!("After untwist (final): {:?}", rt_final);

        assert_eq!(rt_final, roundtrip_input, "NTT roundtrip failed");
        println!("NTT roundtrip: OK");

        println!("\n=== Debug test passed ===");
    }

    #[test]
    fn test_gpu_ntt_goldilocks_small() {
        // Test with small n but Goldilocks prime to isolate issues
        use wgpu::BufferUsages;

        let n = 64; // Small n for debugging
        let q = 1152921504606994433u64; // First Goldilocks prime

        // Verify q ≡ 1 (mod 2n)
        assert_eq!((q - 1) % (2 * n as u64), 0, "Prime not NTT-friendly for this n");

        let params = GpuRnsParams {
            n,
            moduli: vec![q],
            digits_per_limb: 4,
            decomp_base: 1 << 15,
        };

        let ctx = match GpuRotationContext::new(params) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("Skipping test: {:?}", e);
                return;
            }
        };

        let ntt = &ctx.ntt_data[0];
        println!("=== Goldilocks Small Test (n={}, q={}) ===", n, q);
        println!("psi = {}", ntt.psi);
        println!("omega = {}", ntt.omega);
        println!("n_inv = {}", ntt.n_inv);
        println!("mu_lo = 0x{:016X}", ntt.mu_lo);
        println!("mu_hi = 0x{:016X}", ntt.mu_hi);

        // Test data roundtrip
        println!("\n--- Data roundtrip test ---");
        let mut test_coeffs = vec![0u64; n];
        test_coeffs[0] = 1;
        test_coeffs[1] = 1;
        let test_residues = vec![test_coeffs.clone()];
        let test_poly = GpuRnsPoly::from_residues(&ctx, &test_residues).unwrap();
        let roundtrip = test_poly.to_residues(&ctx).unwrap();
        println!("Input: {:?}", &test_coeffs[..4]);
        println!("Roundtrip: {:?}", &roundtrip[0][..4]);
        assert_eq!(roundtrip[0][0], 1, "roundtrip coeff 0");
        assert_eq!(roundtrip[0][1], 1, "roundtrip coeff 1");
        println!("Data roundtrip: OK");

        // Test NTT multiplication: (1 + X) * (1 + X) = 1 + 2X + X^2
        println!("\n--- NTT multiplication test ---");
        let mut a_coeffs = vec![0u64; n];
        a_coeffs[0] = 1;
        a_coeffs[1] = 1;
        let a_residues = vec![a_coeffs.clone()];
        let a = GpuRnsPoly::from_residues(&ctx, &a_residues).unwrap();

        let result = gpu_ntt_mul(&ctx, &a, &a).unwrap();
        let result_residues = result.to_residues(&ctx).unwrap();

        println!("Input (1 + X): {:?}", &a_coeffs[..4]);
        println!("Result (1 + X)^2: {:?}", &result_residues[0][..8]);

        // Check if values are in correct range
        let max_val = result_residues[0].iter().max().unwrap();
        println!("Max value in result: {}", max_val);
        assert!(*max_val < q, "Result values should be < q");

        // Verify expected result
        assert_eq!(result_residues[0][0], 1, "constant term");
        assert_eq!(result_residues[0][1], 2, "X term");
        assert_eq!(result_residues[0][2], 1, "X^2 term");
        assert_eq!(result_residues[0][3], 0, "X^3 term should be 0");

        println!("\n=== Goldilocks small test passed ===");
    }

    /// Helper function to read a GPU buffer back to Vec<u64>
    fn read_buffer_to_u64(ctx: &GpuRotationContext, buffer: &wgpu::Buffer, n: usize) -> Vec<u64> {
        let size = (n * 2 * std::mem::size_of::<u32>()) as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let lo = u32_data[i * 2] as u64;
            let hi = u32_data[i * 2 + 1] as u64;
            result.push(lo | (hi << 32));
        }
        result
    }
}
