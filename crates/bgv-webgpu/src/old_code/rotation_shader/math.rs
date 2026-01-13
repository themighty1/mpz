//! Re-export shared math functions from the top-level shader_math module.
//!
//! This module exists for backward compatibility. New code should use `crate::shader_math` directly.

pub use crate::shader_math::MATH_MODULE;
