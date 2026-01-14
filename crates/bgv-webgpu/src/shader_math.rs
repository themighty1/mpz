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
    var carry: u32 = 0u;
    if sum_lo < a.x { carry = 1u; }

    var sum_hi = a.y + b.y;
    var carry_hi: u32 = 0u;
    if sum_hi < a.y { carry_hi = 1u; }
    let tmp = sum_hi + carry;
    if tmp < sum_hi { carry_hi = carry_hi + 1u; }
    sum_hi = tmp;

    // If carry_hi > 0, sum >= 2^64 > q, so we need to subtract q
    // Also subtract if sum >= q (normal case)
    if carry_hi > 0u || sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = sum_lo + (0xFFFFFFFFu - q.x) + 1u;
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

// Barrett reduction for 60-bit RNS moduli (optimized)
// Only uses lower 64 bits of input - sufficient when q < 2^60 and inputs < q
// For products of two values < 2^60, the result fits in ~120 bits but we only
// need to reduce values that are already partially reduced.
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce_60bit(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
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

// Barrett reduction for 64-bit Goldilocks modulus (full 128-bit input)
// Handles the full 128-bit product when multiplying two 64-bit Goldilocks elements.
// Uses all four 32-bit words of the input (x.x, x.y, x.z, x.w).
// mu is passed as four 32-bit words: mu ≈ 2^128 / q
fn barrett_reduce_64bit(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    // Fast path: x already < q (high 64 bits are zero)
    if x.w == 0u && x.z == 0u && (x.y < q.y || (x.y == q.y && x.x < q.x)) {
        return vec2<u32>(x.x, x.y);
    }

    // We need to compute floor(x * mu / 2^128) where x is 128-bit and mu is 128-bit
    // This gives us a quotient estimate q_est, then r = x - q_est * q
    //
    // x = x0 + x1*2^32 + x2*2^64 + x3*2^96  (128-bit)
    // mu = mu0 + mu1*2^32 + mu2*2^64 + mu3*2^96  (128-bit)
    //
    // We need bits 128-255 of x * mu (i.e., floor(x * mu / 2^128))

    // Compute all 16 partial products (32x32 -> 64)
    let p00 = u64_mul(x.x, mu0);
    let p01 = u64_mul(x.x, mu1);
    let p02 = u64_mul(x.x, mu2);
    let p03 = u64_mul(x.x, mu3);
    let p10 = u64_mul(x.y, mu0);
    let p11 = u64_mul(x.y, mu1);
    let p12 = u64_mul(x.y, mu2);
    let p13 = u64_mul(x.y, mu3);
    let p20 = u64_mul(x.z, mu0);
    let p21 = u64_mul(x.z, mu1);
    let p22 = u64_mul(x.z, mu2);
    let p23 = u64_mul(x.z, mu3);
    let p30 = u64_mul(x.w, mu0);
    let p31 = u64_mul(x.w, mu1);
    let p32 = u64_mul(x.w, mu2);
    let p33 = u64_mul(x.w, mu3);

    // Accumulate columns to compute q_est = bits 128-191 of x*mu
    // pij = xi * muj, pij.x contributes to column (i+j), pij.y to column (i+j+1)
    var t: u32;

    // Column 1 (bits 32-63): pij.x where i+j=1, pij.y where i+j=0
    var col1: u32 = p00.y;
    var c1: u32 = 0u;
    t = col1 + p01.x; if t < col1 { c1 = 1u; } col1 = t;
    t = col1 + p10.x; if t < col1 { c1 = c1 + 1u; } col1 = t;

    // Column 2 (bits 64-95): pij.x where i+j=2, pij.y where i+j=1
    var col2: u32 = p01.y;
    var c2: u32 = 0u;
    t = col2 + p10.y; if t < col2 { c2 = 1u; } col2 = t;
    t = col2 + p02.x; if t < col2 { c2 = c2 + 1u; } col2 = t;
    t = col2 + p11.x; if t < col2 { c2 = c2 + 1u; } col2 = t;
    t = col2 + p20.x; if t < col2 { c2 = c2 + 1u; } col2 = t;
    t = col2 + c1; if t < col2 { c2 = c2 + 1u; } col2 = t;

    // Column 3 (bits 96-127): pij.x where i+j=3, pij.y where i+j=2
    var col3: u32 = p02.y;
    var c3: u32 = 0u;
    t = col3 + p11.y; if t < col3 { c3 = 1u; } col3 = t;
    t = col3 + p20.y; if t < col3 { c3 = c3 + 1u; } col3 = t;
    t = col3 + p03.x; if t < col3 { c3 = c3 + 1u; } col3 = t;
    t = col3 + p12.x; if t < col3 { c3 = c3 + 1u; } col3 = t;
    t = col3 + p21.x; if t < col3 { c3 = c3 + 1u; } col3 = t;
    t = col3 + p30.x; if t < col3 { c3 = c3 + 1u; } col3 = t;
    t = col3 + c2; if t < col3 { c3 = c3 + 1u; } col3 = t;

    // Column 4 (bits 128-159): pij.x where i+j=4, pij.y where i+j=3
    var col4: u32 = p03.y;
    var c4: u32 = 0u;
    t = col4 + p12.y; if t < col4 { c4 = 1u; } col4 = t;
    t = col4 + p21.y; if t < col4 { c4 = c4 + 1u; } col4 = t;
    t = col4 + p30.y; if t < col4 { c4 = c4 + 1u; } col4 = t;
    t = col4 + p13.x; if t < col4 { c4 = c4 + 1u; } col4 = t;
    t = col4 + p22.x; if t < col4 { c4 = c4 + 1u; } col4 = t;
    t = col4 + p31.x; if t < col4 { c4 = c4 + 1u; } col4 = t;
    t = col4 + c3; if t < col4 { c4 = c4 + 1u; } col4 = t;

    // Column 5 (bits 160-191): pij.x where i+j=5, pij.y where i+j=4
    var col5: u32 = p13.y;
    var c5: u32 = 0u;
    t = col5 + p22.y; if t < col5 { c5 = 1u; } col5 = t;
    t = col5 + p31.y; if t < col5 { c5 = c5 + 1u; } col5 = t;
    t = col5 + p23.x; if t < col5 { c5 = c5 + 1u; } col5 = t;
    t = col5 + p32.x; if t < col5 { c5 = c5 + 1u; } col5 = t;
    t = col5 + c4; if t < col5 { c5 = c5 + 1u; } col5 = t;

    // Column 6 (bits 192-223): pij.x where i+j=6, pij.y where i+j=5
    var col6: u32 = p23.y;
    var c6: u32 = 0u;
    t = col6 + p32.y; if t < col6 { c6 = 1u; } col6 = t;
    t = col6 + p33.x; if t < col6 { c6 = c6 + 1u; } col6 = t;
    t = col6 + c5; if t < col6 { c6 = c6 + 1u; } col6 = t;

    // Column 7 (bits 224-255): pij.y where i+j=6
    var col7: u32 = p33.y;
    t = col7 + c6; col7 = t;

    // q_est = (col7, col6, col5, col4) >> 64 = (col7, col6) as 64-bit value
    // Actually we want bits 128-191 of x*mu as our q_est (64-bit)
    var q_est_lo = col4;
    var q_est_hi = col5;

    // Compute q_est * q (64-bit * 64-bit = 128-bit, but we only need low 128 bits)
    let qe00 = u64_mul(q_est_lo, q.x);
    let qe01 = u64_mul(q_est_lo, q.y);
    let qe10 = u64_mul(q_est_hi, q.x);
    let qe11 = u64_mul(q_est_hi, q.y);

    var sub0 = qe00.x;
    var sub1 = qe00.y;
    var sub2: u32 = 0u;
    var sub3 = qe11.y;
    var sc: u32 = 0u;

    t = sub1 + qe01.x; if t < sub1 { sc = 1u; } sub1 = t;
    t = sub1 + qe10.x; if t < sub1 { sc = sc + 1u; } sub1 = t;

    var sc2: u32 = 0u;
    t = sub2 + qe01.y; if t < sub2 { sc2 = 1u; } sub2 = t;
    t = sub2 + qe10.y; if t < sub2 { sc2 = sc2 + 1u; } sub2 = t;
    t = sub2 + qe11.x; if t < sub2 { sc2 = sc2 + 1u; } sub2 = t;
    t = sub2 + sc; if t < sub2 { sc2 = sc2 + 1u; } sub2 = t;

    // Propagate carry to sub3
    t = sub3 + sc2; sub3 = t;

    // r = x - q_est * q
    // sub = (sub0, sub1, sub2, ...) is q_est * q (up to 128+ bits)
    // x = (x.x, x.y, x.z, x.w) is the full 128-bit input
    // Result r should be < 2q, so fits in 65 bits

    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;
    var borrow: u32 = 0u;

    // Subtract sub0 from r0
    if r0 >= sub0 {
        r0 = r0 - sub0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - sub0) + 1u;
        borrow = 1u;
    }

    // Subtract sub1 + borrow from r1
    var sub1b = sub1 + borrow;
    borrow = 0u;
    if sub1b < sub1 { borrow = 1u; }  // sub1 + old_borrow overflowed
    if r1 >= sub1b {
        r1 = r1 - sub1b;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1b) + 1u;
        borrow = borrow + 1u;
    }

    // Subtract sub2 + borrow from r2
    var sub2b = sub2 + borrow;
    borrow = 0u;
    if sub2b < sub2 { borrow = 1u; }
    if r2 >= sub2b {
        r2 = r2 - sub2b;
    } else {
        r2 = r2 + (0xFFFFFFFFu - sub2b) + 1u;
        borrow = borrow + 1u;
    }

    // Subtract sub3 + borrow from r3
    var sub3b = sub3 + borrow;
    if r3 >= sub3b {
        r3 = r3 - sub3b;
    } else {
        r3 = r3 + (0xFFFFFFFFu - sub3b) + 1u;
    }

    // Now r = (r0, r1, r2, r3) but result should fit in ~65 bits
    // If r2 or r3 are non-zero, we need to reduce further (shouldn't happen for correct q_est)
    // For safety, check if r >= q and correct

    // Final corrections: subtract q while r >= q
    // Since q_est might be off by 1-2, r could be in [0, 3q)
    for (var i = 0u; i < 3u; i = i + 1u) {
        // Check if (r1, r0) >= (q.y, q.x) considering r2 might have a bit set
        if r2 > 0u || r1 > q.y || (r1 == q.y && r0 >= q.x) {
            borrow = 0u;
            if r0 >= q.x {
                r0 = r0 - q.x;
            } else {
                r0 = r0 + (0xFFFFFFFFu - q.x) + 1u;
                borrow = 1u;
            }
            var qyb = q.y + borrow;
            if r1 >= qyb {
                r1 = r1 - qyb;
            } else {
                r1 = r1 + (0xFFFFFFFFu - qyb) + 1u;
                if r2 > 0u { r2 = r2 - 1u; }
            }
        } else {
            break;
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

// ============================================================================
// Goldilocks-specific arithmetic (p = 2^64 - 2^32 + 1)
// Based on recmo/goldilocks with bug fixes for overflow/underflow handling.
// Attribution: Original WGSL implementation from https://github.com/recmo/goldilocks
// ============================================================================

// Goldilocks prime constants
const GOLDILOCKS_P_LO: u32 = 0x00000001u;  // Low 32 bits of p
const GOLDILOCKS_P_HI: u32 = 0xFFFFFFFFu;  // High 32 bits of p
const GOLDILOCKS_EPSILON: u32 = 0xFFFFFFFFu;  // 2^64 mod p = 2^32 - 1

// Goldilocks addition: (a + b) mod p
fn goldilocks_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var r = a + b;
    var carry = u32(r.x < a.x);
    r.y = r.y + carry;

    // Check for overflow past 2^64
    if (r.y < a.y) {
        // Add (2^64 mod p) = EPSILON = 2^32 - 1
        let old_x = r.x;
        r.x = r.x + GOLDILOCKS_EPSILON;
        if (r.x < old_x) {
            r.y = r.y + 1u;
        }
    }

    // Reduce if r >= p
    if (r.y == GOLDILOCKS_P_HI && r.x >= GOLDILOCKS_P_LO) {
        var borrow = u32(r.x < GOLDILOCKS_P_LO);
        r.x = r.x - GOLDILOCKS_P_LO;
        r.y = r.y - GOLDILOCKS_P_HI - borrow;
    }

    return r;
}

// Goldilocks subtraction: (a - b) mod p
fn goldilocks_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var r = a - b;
    r.y -= u32(r.x > a.x);
    if (r.y > a.y) {
        // Underflow: add p by subtracting (2^64 - p) = 2^32 - 1 = EPSILON
        // Actually we need to add p = 2^64 - 2^32 + 1
        // In wrapped arithmetic: result += p means result -= (2^64 - p) = result -= (2^32 - 1)
        // But since we underflowed, the wrapped value is result + 2^64
        // We need (result + 2^64) - 2^64 + p = result + p
        // So add p by: r.x += 1, r.y -= 1 (accounting for carry)
        r.x += 1u;
        r.y -= u32(r.x != 0u);
    }
    return r;
}

// Helper: compute (a + b) / 2 without overflow (for carry detection)
fn goldilocks_hadd(a: u32, b: u32) -> u32 {
    return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

// 32x32 -> 64-bit multiplication
fn goldilocks_mul64(a: u32, b: u32) -> vec2<u32> {
    var a0 = (a << 16u) >> 16u;
    var a1 = a >> 16u;
    var b0 = (b << 16u) >> 16u;
    var b1 = b >> 16u;

    var a0b0 = a0 * b0;
    var a0b1 = a0 * b1;
    var a1b0 = a1 * b0;
    var a1b1 = a1 * b1;

    var r: vec2<u32>;
    r.x = a0b0 + (a1b0 << 16u) + (a0b1 << 16u);
    r.y = a1b1 + (goldilocks_hadd((a0b0 >> 16u) + a0b1, a1b0) >> 15u);
    return r;
}

// 64x64 -> 128-bit multiplication
fn goldilocks_mul128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    var a0b0 = goldilocks_mul64(a.x, b.x);
    var a0b1 = goldilocks_mul64(a.x, b.y);
    var a1b0 = goldilocks_mul64(a.y, b.x);
    var a1b1 = goldilocks_mul64(a.y, b.y);

    var r = vec4<u32>(a0b0, a1b1);

    r.y += a0b1.x;
    if (r.y < a0b1.x) {
        a0b1.y += 1u;
    }
    r.z += a0b1.y;
    if (r.z < a0b1.y) {
        r.w += 1u;
    }

    r.y += a1b0.x;
    if (r.y < a1b0.x) {
        a1b0.y += 1u;
    }
    r.z += a1b0.y;
    if (r.z < a1b0.y) {
        r.w += 1u;
    }

    return r;
}

// Goldilocks 128-bit to 64-bit reduction
// Reduces n = n.x + n.y*2^32 + n.z*2^64 + n.w*2^96 mod p
// Using: 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p)
fn goldilocks_reduce(n: vec4<u32>) -> vec2<u32> {
    var mid = n.y + n.z;
    var mid_carry = u32(mid < n.y);

    var sub_total = n.z + n.w;
    var sub_carry = u32(sub_total < n.z);

    var r_lo: u32;
    var r_hi: u32;

    if (n.x >= sub_total) {
        r_lo = n.x - sub_total;
        if (mid >= sub_carry) {
            r_hi = mid - sub_carry;
        } else {
            r_hi = mid - sub_carry;
            let old_lo = r_lo;
            r_lo = r_lo + GOLDILOCKS_P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + GOLDILOCKS_P_HI;
        }
    } else {
        r_lo = n.x - sub_total;
        var borrow = sub_carry + 1u;
        if (mid >= borrow) {
            r_hi = mid - borrow;
        } else {
            r_hi = mid - borrow;
            let old_lo = r_lo;
            r_lo = r_lo + GOLDILOCKS_P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + GOLDILOCKS_P_HI;
        }
    }

    if (mid_carry > 0u) {
        let old_lo = r_lo;
        r_lo = r_lo + GOLDILOCKS_EPSILON;
        if (r_lo < old_lo) {
            r_hi = r_hi + 1u;
        }
    }

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (r_hi == GOLDILOCKS_P_HI && r_lo >= GOLDILOCKS_P_LO) {
            var borrow = u32(r_lo < GOLDILOCKS_P_LO);
            r_lo = r_lo - GOLDILOCKS_P_LO;
            r_hi = r_hi - GOLDILOCKS_P_HI - borrow;
        } else {
            break;
        }
    }

    return vec2<u32>(r_lo, r_hi);
}

// Goldilocks multiplication: (a * b) mod p
fn goldilocks_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return goldilocks_reduce(goldilocks_mul128(a, b));
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

    #[test]
    fn test_goldilocks_functions_compile() {
        // Test that Goldilocks-specific functions compile correctly
        let test_shader = r#"
#import math

@compute @workgroup_size(64, 1, 1)
fn test_goldilocks(@builtin(global_invocation_id) id: vec3<u32>) {
    let a = vec2<u32>(0x12345678u, 0xABCDEF01u);
    let b = vec2<u32>(0x87654321u, 0x10FEDCBAu);

    // Test Goldilocks add/sub/mul
    let sum = math::goldilocks_add(a, b);
    let diff = math::goldilocks_sub(a, b);
    let prod = math::goldilocks_mul(a, b);

    // Test reduce with 128-bit input
    let big = vec4<u32>(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
    let reduced = math::goldilocks_reduce(big);
}
"#;
        let result = compose_shader(test_shader, "test_goldilocks.wgsl");
        match result {
            Ok(_composed) => {
                println!("Goldilocks shader compiled successfully!");
            }
            Err(e) => panic!("Failed to compose Goldilocks shader: {}", e),
        }
    }
}
