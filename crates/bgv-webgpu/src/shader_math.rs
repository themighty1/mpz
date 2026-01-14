//! Shared math functions for WGSL shaders.
//!
//! This module provides WGSL math functions that can be imported by any shader using naga_oil.
//! Use `#import math` in your shader to access these functions.

use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage, ShaderType};
use std::collections::HashMap;

/// Shared math module that can be imported by other shaders.
/// Use `#import math` in your shader and call functions directly.
pub const MATH_MODULE: &str = r#"
#define_import_path math

// Modular addition: (a + b) mod q
fn addmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var sum_lo = a.x + b.x;
    var carry = 0u;
    if sum_lo < a.x { carry = 1u; }
    var sum_hi = a.y + b.y + carry;

    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }

    return vec2<u32>(sum_lo, sum_hi);
}

// Modular subtraction: (a - b) mod q
fn submod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        return vec2<u32>(diff_lo, diff_hi);
    } else {
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;

        var res_lo = q.x - diff_lo;
        borrow = 0u;
        if q.x < diff_lo { borrow = 1u; }
        var res_hi = q.y - diff_hi - borrow;
        return vec2<u32>(res_lo, res_hi);
    }
}

// Multiply two u32 to get u64 as vec2<u32>
fn u64_mul(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    var lo = p0;
    var hi = p3;

    let mid = p1 + p2;
    let mid_lo = (mid & 0xFFFFu) << 16u;
    let mid_hi = mid >> 16u;

    let new_lo = lo + mid_lo;
    if new_lo < lo { hi = hi + 1u; }
    lo = new_lo;
    hi = hi + mid_hi;

    if p1 > 0xFFFFFFFFu - p2 {
        hi = hi + 0x10000u;
    }

    return vec2<u32>(lo, hi);
}

// Multiply two 64-bit values to get 128-bit result as vec4<u32>
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);

    var r0 = p00.x;
    var r1 = p00.y;
    var r2 = p11.x;
    var r3 = p11.y;

    var t = r1 + p01.x;
    var c: u32 = 0u;
    if t < r1 { c = 1u; }
    r1 = t;
    t = r1 + p10.x;
    if t < r1 { c = c + 1u; }
    r1 = t;

    t = r2 + p01.y;
    var c2: u32 = 0u;
    if t < r2 { c2 = 1u; }
    r2 = t;
    t = r2 + p10.y;
    if t < r2 { c2 = c2 + 1u; }
    r2 = t;
    t = r2 + c;
    if t < r2 { c2 = c2 + 1u; }
    r2 = t;

    r3 = r3 + c2;

    return vec4<u32>(r0, r1, r2, r3);
}

// 128-bit subtraction: a - b
fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r0 = a.x; var r1 = a.y; var r2 = a.z; var r3 = a.w;
    var borrow = 0u;

    if r0 >= b.x { r0 = r0 - b.x; }
    else { r0 = 0xFFFFFFFFu - (b.x - r0 - 1u); borrow = 1u; }

    var new_r1 = r1 - b.y - borrow;
    if new_r1 > r1 { borrow = 1u; } else { borrow = 0u; }
    r1 = new_r1;

    var new_r2 = r2 - b.z - borrow;
    if new_r2 > r2 { borrow = 1u; } else { borrow = 0u; }
    r2 = new_r2;

    r3 = r3 - b.w - borrow;

    return vec4<u32>(r0, r1, r2, r3);
}

// Compare 128-bit: returns true if a >= b
fn ge128(a: vec4<u32>, b: vec4<u32>) -> bool {
    if a.w != b.w { return a.w > b.w; }
    if a.z != b.z { return a.z > b.z; }
    if a.y != b.y { return a.y > b.y; }
    return a.x >= b.x;
}

// Modular multiplication using shift-and-subtract reduction
// For 60-bit modulus: a * b mod q where a, b < q
fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    let q128 = vec4<u32>(q.x, q.y, 0u, 0u);
    var r = prod;

    // Shift-and-subtract reduction for 128-bit product mod 60-bit q
    for (var shift = 60u; shift > 0u; shift = shift - 1u) {
        var q_shifted: vec4<u32>;
        if shift >= 64u {
            let s = shift - 64u;
            if s == 0u {
                q_shifted = vec4<u32>(0u, 0u, q.x, q.y);
            } else {
                q_shifted = vec4<u32>(0u, 0u, q.x << s, (q.y << s) | (q.x >> (32u - s)));
            }
        } else if shift >= 32u {
            let s = shift - 32u;
            if s == 0u {
                q_shifted = vec4<u32>(0u, q.x, q.y, 0u);
            } else {
                q_shifted = vec4<u32>(0u, q.x << s, (q.y << s) | (q.x >> (32u - s)), q.y >> (32u - s));
            }
        } else {
            q_shifted = vec4<u32>(q.x << shift, (q.y << shift) | (q.x >> (32u - shift)), q.y >> (32u - shift), 0u);
        }

        if ge128(r, q_shifted) {
            r = sub128(r, q_shifted);
        }
    }

    // Final reductions
    for (var i = 0u; i < 3u; i++) {
        if !ge128(r, q128) { break; }
        r = sub128(r, q128);
    }

    return vec2<u32>(r.x, r.y);
}

// Barrett reduction for 128-bit value mod 64-bit q
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;

    // Fast path: x already < q
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) {
        return vec2<u32>(r0, r1);
    }

    // Compute all 8 partial products for r * mu
    let p00 = u64_mul(r0, mu0);
    let p01 = u64_mul(r0, mu1);
    let p02 = u64_mul(r0, mu2);
    let p03 = u64_mul(r0, mu3);
    let p10 = u64_mul(r1, mu0);
    let p11 = u64_mul(r1, mu1);
    let p12 = u64_mul(r1, mu2);
    let p13 = u64_mul(r1, mu3);

    // Accumulate bits 64-95 with carries
    var acc64: u32 = p02.x;
    var c: u32 = 0u;
    var t: u32;
    t = acc64 + p01.y; if t < acc64 { c = 1u; } acc64 = t;
    t = acc64 + p10.y; if t < acc64 { c = c + 1u; } acc64 = t;
    t = acc64 + p11.x; if t < acc64 { c = c + 1u; } acc64 = t;
    var mid32 = p00.y;
    var mid_c: u32 = 0u;
    var mt = mid32 + p01.x; if mt < mid32 { mid_c = 1u; } mid32 = mt;
    mt = mid32 + p10.x; if mt < mid32 { mid_c = mid_c + 1u; }
    t = acc64 + mid_c; if t < acc64 { c = c + 1u; } acc64 = t;

    // Accumulate bits 96-127
    var acc96: u32 = p02.y;
    var c2: u32 = 0u;
    t = acc96 + p03.x; if t < acc96 { c2 = 1u; } acc96 = t;
    t = acc96 + p11.y; if t < acc96 { c2 = c2 + 1u; } acc96 = t;
    t = acc96 + p12.x; if t < acc96 { c2 = c2 + 1u; } acc96 = t;
    t = acc96 + c; if t < acc96 { c2 = c2 + 1u; } acc96 = t;

    // Accumulate bits 128-159
    var acc128: u32 = p13.x;
    var c3: u32 = 0u;
    t = acc128 + p03.y; if t < acc128 { c3 = 1u; } acc128 = t;
    t = acc128 + p12.y; if t < acc128 { c3 = c3 + 1u; } acc128 = t;
    t = acc128 + c2; if t < acc128 { c3 = c3 + 1u; } acc128 = t;

    // Accumulate bits 160-191
    var acc160: u32 = p13.y + c3;

    // q_est = acc160 * 2^32 + acc128
    var q_est_lo = acc128;
    var q_est_hi = acc160;

    // Compute q_est * q
    let qe0 = u64_mul(q_est_lo, q.x);
    let qe1 = u64_mul(q_est_lo, q.y);
    let qe2 = u64_mul(q_est_hi, q.x);

    var p0 = qe0.x;
    var p1 = qe0.y;
    var p2: u32 = 0u;

    t = p1 + qe1.x; c = 0u; if t < p1 { c = 1u; } p1 = t; p2 = qe1.y + c;
    t = p1 + qe2.x; c = 0u; if t < p1 { c = 1u; } p1 = t; p2 = p2 + qe2.y + c;

    // Subtract p from r
    var borrow: u32 = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
    }

    // Final correction: subtract q while r >= q
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            break;
        }
        borrow = 0u;
        if r0 >= q.x {
            r0 = r0 - q.x;
        } else {
            r0 = r0 + (0xFFFFFFFFu - q.x) + 1u;
            borrow = 1u;
        }
        sub1 = q.y + borrow;
        if r1 >= sub1 {
            r1 = r1 - sub1;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        }
    }

    return vec2<u32>(r0, r1);
}

// Bit-reverse index
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}
"#;

/// Composes a shader with the math module imported.
/// Returns the composed WGSL source string.
pub fn compose_shader(shader_source: &str, shader_name: &str) -> Result<String, String> {
    let mut composer = Composer::default();

    // Add the math module
    if let Err(e) = composer.add_composable_module(ComposableModuleDescriptor {
        source: MATH_MODULE,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_shader_with_math_prefix() {
        // Test that math::addmod/submod/mulmod/bit_reverse work with module prefix
        let test_shader = r#"
#import math

@compute @workgroup_size(64, 1, 1)
fn test_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let rev = math::bit_reverse(idx, 8u);
    let a = vec2<u32>(1u, 0u);
    let b = vec2<u32>(2u, 0u);
    let q = vec2<u32>(0xFFFFFFFFu, 0u);
    let sum = math::addmod(a, b, q);
    let diff = math::submod(a, b, q);
    let prod = math::mulmod(a, b, q);
}
"#;
        let result = compose_shader(test_shader, "test.wgsl");
        match result {
            Ok(composed) => {
                println!("Composed shader:\n{}", composed);
            }
            Err(e) => panic!("Failed to compose shader: {}", e),
        }
    }
}
