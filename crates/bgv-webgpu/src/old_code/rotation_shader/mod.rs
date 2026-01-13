//! WGSL shaders for GPU-accelerated BGV rotation and slot summation.
//!
//! This module uses naga_oil for shader composition with shared math functions.

pub mod math;
pub mod fused;
pub mod shared_mem_ntt;

// Re-export compose functions from shader_math for backward compatibility
pub use crate::shader_math::{compose_shader, create_shader_module};

// Re-export the old shaders for backward compatibility
// TODO: Remove these once all code is migrated to use the new modular shaders
pub use super::rotation_shader_legacy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_fused_twist() {
        let result = compose_shader(fused::FUSED_TWIST_SHADER, "fused_twist.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn fused_twist"), "Missing entry point");
    }

    #[test]
    fn test_compose_fused_butterfly() {
        let result = compose_shader(fused::FUSED_BUTTERFLY_SHADER, "fused_butterfly.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn fused_butterfly"), "Missing entry point");
    }

    #[test]
    fn test_compose_fused_pointwise() {
        let result = compose_shader(fused::FUSED_POINTWISE_SHADER, "fused_pointwise.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn fused_pointwise"), "Missing entry point");
    }

    #[test]
    fn test_compose_fused_scale() {
        let result = compose_shader(fused::FUSED_SCALE_SHADER, "fused_scale.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn fused_scale"), "Missing entry point");
    }

    #[test]
    fn test_compose_fused_bitrev() {
        // Bitrev doesn't import math, should work without composer
        let result = compose_shader(fused::FUSED_BITREV_SHADER, "fused_bitrev.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_compose_shared_mem_ntt_fwd() {
        let result = compose_shader(shared_mem_ntt::SHARED_MEM_NTT_FWD_SHADER, "shared_mem_ntt_fwd.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn shared_mem_ntt_fwd"), "Missing entry point");
    }

    #[test]
    fn test_compose_shared_mem_ntt_inv() {
        let result = compose_shader(shared_mem_ntt::SHARED_MEM_NTT_INV_SHADER, "shared_mem_ntt_inv.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn shared_mem_ntt_inv"), "Missing entry point");
    }

    #[test]
    fn test_compose_shared_mem_ntt_partial() {
        let result = compose_shader(shared_mem_ntt::SHARED_MEM_NTT_PARTIAL_SHADER, "shared_mem_ntt_partial.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn shared_mem_ntt_partial"), "Missing entry point");
    }
}
