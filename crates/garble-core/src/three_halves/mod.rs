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
//!
//! ## Key Equations (Paper Section 5)
//!
//! The main garbling equation (Equation 4) is:
//!
//! ```text
//! V · [C; G⃗] = M · H⃗ ⊕ (R ⊕ [0 0 t]) · [A₀; B₀; Δ]
//! ```
//!
//! Where:
//! - `C` = output wire label (2 halves: C_L, C_R)
//! - `G⃗` = gate ciphertexts (3 values, each κ/2 bits)
//! - `H⃗` = hash outputs [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]ᵀ
//! - `t` = truth table (8×2 matrix encoding which input gives true output)
//! - `R` = control matrix (randomized, determines linear combinations)

pub mod matrices;
pub mod control;
pub mod slicing;

#[cfg(test)]
mod tests;
