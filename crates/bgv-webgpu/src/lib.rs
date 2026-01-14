#![allow(dead_code)]
//! WebGPU-accelerated BGV ciphertext scalar multiplication.
//!
//! This crate provides GPU-accelerated slot-wise scalar multiplication for BGV ciphertexts.
//!
//! # Use Case
//!
//! In JustVengers IT-PAC, the prover computes:
//! - Input: 1 ciphertext with 8192 slots
//! - Input: 80 × 8192 random scalars
//! - Output: 80 ciphertexts (one per batch)
//!
//! Each output batch is: `output[batch][slot] = ct[slot] * scalar[batch][slot] mod q`

mod error;
mod gpu;
mod math;
mod shader;
pub mod shader_math;
pub mod slot_mul_shader;
pub mod slot_mul_gpu;
pub mod rns_slot_mul;
pub mod goldilocks_ntt;

// DEPRECATED: rotation-based operations not used in production
pub mod old_code;

pub use error::GpuError;
pub use gpu::{GpuContext, GpuCiphertext, SlotWiseMul};
pub use slot_mul_gpu::{SlotMulGpuContext, TwiddleFactors};
pub use rns_slot_mul::{RnsSlotMulGpu, RnsBatchParams, RnsModulusNttData, PlaintextNttData};
pub use goldilocks_ntt::GoldilocksNttGpu;

// Re-export old rotation types for backwards compatibility
pub use old_code::{
    GpuRotationContext, GpuRnsParams, GpuRnsPoly, GpuRnsCiphertext, NttModulusData,
    GpuGaloisKey, GpuGaloisKeys, gpu_automorphism, gpu_add, gpu_ntt_mul, gpu_ntt_mul_single,
    gpu_key_switch, gpu_apply_automorphism, rotation_exponent,
};

/// The modulus for Goldilocks field operations.
pub const GOLDILOCKS_Q: u64 = 0xFFFFFFFF00000001;

/// Parameters for slot-wise multiplication.
#[derive(Clone, Debug)]
pub struct BatchParams {
    /// Ring dimension (number of slots per ciphertext).
    pub n: usize,
    /// Ciphertext modulus q.
    pub q: u64,
    /// Number of output ciphertexts (batches).
    pub num_batches: usize,
}

impl BatchParams {
    /// Creates parameters for the standard JustVengers configuration.
    ///
    /// - n = 8192 (ring dimension for Goldilocks)
    /// - q = Goldilocks prime
    /// - num_batches = 80
    pub fn justvengers_standard() -> Self {
        Self {
            n: 8192,
            q: GOLDILOCKS_Q,
            num_batches: 80,
        }
    }

    /// Creates custom parameters.
    pub fn new(n: usize, q: u64, num_batches: usize) -> Self {
        Self { n, q, num_batches }
    }
}
