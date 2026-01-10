//! Shared math functions for WGSL shaders.
//! This module is imported by other shaders using naga_oil.

/// Shared math module that can be imported by other shaders.
/// Use `#import math` in your shader and call functions as `math::func_name()`.
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
"#;
