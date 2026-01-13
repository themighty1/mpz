//! **DEPRECATED: NOT USED IN PRODUCTION**
//!
//! This module contains rotation-based operations (sum_slots, automorphisms, key-switching)
//! that are NOT used in production. Production uses RNS slot multiplication directly.
//!
//! Kept for reference and potential future use.

mod rotation;

pub use rotation::*;
