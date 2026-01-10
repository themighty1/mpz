//! Additively Homomorphic Encryption.
//!
//! This module provides two homomorphic encryption schemes:
//!
//! ## CL Encryption (Class Groups)
//!
//! The Castagnos-Laguillaumie scheme based on class groups of imaginary quadratic
//! orders. Provides linear homomorphism (addition and scalar multiplication) with:
//! - No noise accumulation (unlike BGV)
//! - Deterministic decryption
//! - Security from class group order computation hardness
//!
//! See [`cl_ahe`] module for CL-based encryption.
//!
//! ## BGV Encryption (Legacy)
//!
//! Traditional BGV lattice-based encryption with:
//! - Ring-LWE security
//! - Slot packing for SIMD operations
//! - Noise management requirements
//!
//! # Structure
//!
//! - `cl_ahe`: CL-based homomorphic encryption (recommended for addition)
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

#[cfg(feature = "cl-scheme")]
pub mod cl_ahe;

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

// Re-export CL types at top level for convenience
#[cfg(feature = "cl-scheme")]
pub use cl_ahe::{CLGroup, CLSecretKey, CLPublicKey, CLKeyPair, CLCiphertext};

#[cfg(test)]
mod tests;
