//! **DEPRECATED: NOT USED IN PRODUCTION**
//!
//! This module contains batched sum_slots implementation that was an attempt to
//! optimize GPU rotations by encoding multiple operations into single command buffers.
//! 
//! **Status**: This code has a known bug (test_gpu_sum_slots_batched fails) and is
//! not used in production. The non-batched `gpu_sum_slots` in rotation.rs works correctly.
//!
//! **Why kept**: Preserved for reference and potential future debugging/optimization.
//!
//! This module provides:
//! - `SumSlotsWorkspace`: Pre-allocated GPU buffers for batched operations
//! - `gpu_sum_slots_batched`: Batched version with one command buffer per rotation
//! - `gpu_sum_slots_batched_2x`: Processes two ciphertexts in parallel
//! - Various `encode_*` helpers for command buffer encoding

use wgpu::{Buffer, BufferUsages};

use crate::error::GpuError;
use crate::rotation::{
    GpuRotationContext, GpuRnsCiphertext, GpuRnsPoly, GpuGaloisKeys,
    AutoParams, DecomposeParams,
};

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
