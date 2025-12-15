//! Half-Gates Garbling Scheme
//!
//! This module implements "half-gate" garbled circuits from the
//! [Two Halves Make a Whole \[ZRE15\]](https://eprint.iacr.org/2014/756) paper.
//!
//! ## Gate Size
//!
//! AND gates require **2κ bits** (two ciphertexts).
//!
//! ## Overview
//!
//! The half-gates technique splits an AND gate into two "half-gates":
//! - **Garbler half-gate**: Known to the garbler
//! - **Evaluator half-gate**: Known to the evaluator
//!
//! Each half-gate requires only one ciphertext, and combining them gives
//! the correct AND gate output while maintaining Free-XOR compatibility.

/// Evaluator for half-gates garbled circuits.
pub mod evaluator;
/// Garbler for half-gates garbled circuits.
pub mod garbler;
/// Cache-optimized garbler using wire remapping.
pub mod remapped_garbler;

pub use evaluator::{
    EncryptedGateBatchConsumer, EncryptedGateConsumer, Evaluator, EvaluatorError, EvaluatorOutput,
    evaluate_garbled_circuits,
};
pub use garbler::{
    EncryptedGateBatchIter, EncryptedGateIter, Garbler, GarblerError, GarblerOutput,
};
pub use remapped_garbler::{
    RemappedEncryptedGateBatchIter, RemappedEncryptedGateIter, RemappedGarbler,
    RemappedGarblerError, RemappedGarblerOutput, WireRemapping,
};
