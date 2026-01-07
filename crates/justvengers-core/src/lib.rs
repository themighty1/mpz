//! Core primitives for the Justvengers VOLE-based ZK protocol.
//!
//! Justvengers is a protocol for batched disjunctive ZK statements achieving
//! O(R + B + C) communication complexity, where:
//! - R is the number of repetitions
//! - B is the number of branches
//! - C is the circuit size per branch
//!
//! This crate provides the foundational building blocks:
//! - Polynomial arithmetic with Lagrange interpolation
//! - NTT (Number Theoretic Transform) for O(n log n) polynomial multiplication
//! - AHE (Additively Homomorphic Encryption) - BGV-style
//! - IT-PAC (Information-Theoretic Polynomial Authentication Codes)
//! - Topology vectors for circuit linearization

#![deny(missing_docs, unreachable_pub, unused_must_use)]
#![deny(clippy::all)]

pub mod ahe;
pub mod itmac;
pub mod itpac;
pub mod ntt;
pub mod poly;

pub use ahe::{BgvParams, Ciphertext, KeyPair, ParamSet, PublicKey, SecretKey};
pub use itmac::{GlobalKey, ItMac, ItMacBatch, ItMacField, ProverShare, VerifierShare, VolePool};
pub use itpac::{EncryptedPowers, ItPac, ItPacGenerator, ItPacVerifier};
pub use ntt::{Ntt, NttField};
pub use poly::Poly;
