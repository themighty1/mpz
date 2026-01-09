//! Additively Homomorphic Encryption (BGV-style).
//!
//! This module implements BGV encryption with the following properties required
//! by the Justvengers protocol:
//! - CPA security (from Ring-LWE hardness)
//! - Circuit privacy (re-randomization)
//! - Linear targeted malleability (can compute a*c + b on encrypted c)
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

pub use params::{BgvParams, ParamSet, RnsBgvParams, GOLDILOCKS};
pub use ring::{BarrettReducer, RingPoly};
pub use keys::{SecretKey, PublicKey, KeyPair};
pub use ciphertext::Ciphertext;
pub use sample::DiscreteGaussian;
pub use slot::SlotEncoder;
pub use rns::{RnsParams, RnsParamSet, RnsPoly};

#[cfg(test)]
mod tests;
