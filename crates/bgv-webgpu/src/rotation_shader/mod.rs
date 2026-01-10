//! WGSL shaders for GPU-accelerated BGV rotation and slot summation.
//!
//! This module uses naga_oil for shader composition with shared math functions.

pub mod math;
pub mod fused;
pub mod shared_mem_ntt;

use naga_oil::compose::{Composer, ComposableModuleDescriptor, NagaModuleDescriptor, ShaderLanguage, ShaderType};
use std::collections::HashMap;

/// Composes a shader with the math module imported.
/// Returns the composed WGSL source string.
pub fn compose_shader(shader_source: &str, shader_name: &str) -> Result<String, String> {
    let mut composer = Composer::default();

    // Add the math module
    if let Err(e) = composer.add_composable_module(ComposableModuleDescriptor {
        source: math::MATH_MODULE,
        file_path: "math.wgsl",
        language: ShaderLanguage::Wgsl,
        shader_defs: HashMap::new(),
        ..Default::default()
    }) {
        return Err(format!("Failed to add math module: {}", e.emit_to_string(&composer)));
    }

    // Compose the shader
    let naga_module = match composer.make_naga_module(NagaModuleDescriptor {
        source: shader_source,
        file_path: shader_name,
        shader_type: ShaderType::Wgsl,
        shader_defs: HashMap::new(),
        ..Default::default()
    }) {
        Ok(m) => m,
        Err(e) => return Err(format!("Failed to compose {}: {}", shader_name, e.emit_to_string(&composer))),
    };

    // Convert back to WGSL string
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&naga_module)
    .map_err(|e| format!("Validation failed for {}: {:?}", shader_name, e))?;

    naga::back::wgsl::write_string(
        &naga_module,
        &info,
        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
    )
    .map_err(|e| format!("Failed to write WGSL for {}: {:?}", shader_name, e))
}

/// Creates a wgpu shader module from composed shader source.
pub fn create_shader_module(
    device: &wgpu::Device,
    shader_source: &str,
    shader_name: &str,
) -> Result<wgpu::ShaderModule, String> {
    let composed = compose_shader(shader_source, shader_name)?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_name),
        source: wgpu::ShaderSource::Wgsl(composed.into()),
    }))
}

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
    fn test_compose_shared_mem_ntt() {
        let result = compose_shader(shared_mem_ntt::SHARED_MEM_NTT_SHADER, "shared_mem_ntt.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn shared_mem_ntt"), "Missing entry point");
    }

    #[test]
    fn test_compose_shared_mem_ntt_partial() {
        let result = compose_shader(shared_mem_ntt::SHARED_MEM_NTT_PARTIAL_SHADER, "shared_mem_ntt_partial.wgsl");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn shared_mem_ntt_partial"), "Missing entry point");
    }
}
