//! # Three Halves Make a Whole - Garbled Circuit Implementation
//!
//! This module implements the "Three Halves Make a Whole" garbling scheme from:
//!
//! **Paper**: "Three Halves Make a Whole? Beating the Half-Gates Lower Bound for Garbled Circuits"
//! **Authors**: Mike Rosulek, Lawrence Roy
//! **Published**: Eurocrypt 2021
//! **ePrint**: <https://eprint.iacr.org/2021/749>
//!
//! ## Overview
//!
//! This scheme reduces AND gate size from 2κ bits (half-gates) to 1.5κ + 5 bits
//! using two key techniques:
//!
//! 1. **Slicing**: Wire labels are split into left/right halves (κ/2 bits each),
//!    and the evaluator computes each half using potentially different linear combinations.
//!
//! 2. **Dicing**: The evaluator decrypts "control bits" that determine which linear
//!    combinations to apply. These control bits are randomized to hide the gate's
//!    truth table.
//!
//! ## Module Structure
//!
//! - [`matrices`]: Core matrices (K, V, M) that define the linear algebraic structure
//! - [`control`]: Control matrix system (R, S₁, S₂) for the "dicing" technique
//! - [`slicing`]: Wire label slicing utilities
//! - [`garbler`]: Garbling functions
//! - [`evaluator`]: Evaluation functions
//!
//! ## Usage
//!
//! ```ignore
//! use mpz_garble_core::three_halves::{Garbler, Evaluator, GarblerOutput, EvaluatorOutput};
//!
//! let mut gb = Garbler::default();
//! let mut ev = Evaluator::default();
//!
//! let mut gb_iter = gb.generate(&circuit, delta, &input_keys, &mut rng)?;
//! let mut ev_consumer = ev.evaluate(&circuit, &input_macs)?;
//!
//! while let Some(gate) = gb_iter.next() {
//!     ev_consumer.next(gate);
//! }
//!
//! let gb_output = gb_iter.finish()?;
//! let ev_output = ev_consumer.finish()?;
//! ```

pub mod control;
/// Evaluator for three-halves garbled circuits.
pub mod evaluator;
/// Garbler for three-halves garbled circuits.
pub mod garbler;
pub mod matrices;
pub mod slicing;

// Re-export main types from garbler
pub use garbler::{
    ControlBits,
    EncryptedGate,
    EncryptedGateBatch,
    EncryptedGateBatchIter,
    EncryptedGateIter,
    Garbler,
    GarblerError,
    GarblerOutput,
    ThreeHalvesGate,
};

// Re-export main types from evaluator
pub use evaluator::{
    EncryptedGateBatchConsumer,
    EncryptedGateConsumer,
    Evaluator,
    EvaluatorError,
    EvaluatorOutput,
};

#[cfg(test)]
mod tests;
