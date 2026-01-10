//! Additively Homomorphic Encryption.
//!
//! This module provides BGV lattice-based homomorphic encryption with:
//! - Ring-LWE security
//! - Slot packing for SIMD operations
//! - Noise management requirements
//!
//! # Structure
//!
//! - `params`: Parameter sets for different security levels
//! - `ring`: Polynomial ring arithmetic R_q = Z_q[X]/(X^n + 1)
//! - `keys`: Key generation
//! - `ciphertext`: Ciphertext type and homomorphic operations

mod params;
mod ring;
mod keys;
mod ciphertext;
mod sample;
mod slot;
mod rns;
mod rns_bgv;

pub use params::{BgvParams, ParamSet, RnsBgvParams, GOLDILOCKS};
pub use ring::{BarrettReducer, RingPoly};
pub use keys::{SecretKey, PublicKey, KeyPair};
pub use ciphertext::Ciphertext;
pub use sample::DiscreteGaussian;
pub use slot::SlotEncoder;
pub use rns::{RnsParams, RnsParamSet, RnsPoly};
pub use rns_bgv::{
    RnsSecretKey, RnsPublicKey, RnsKeyPair, RnsCiphertext,
    RnsGaloisKey, RnsGaloisKeys,
    SlotPackedEncryptedPowers, SlotPackedCiphertextBatch,
    decrypt_batched_evaluation,
};

#[cfg(test)]
mod tests;
