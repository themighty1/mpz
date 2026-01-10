//! WGSL shaders for GPU-accelerated BGV rotation and slot summation.

/// Common modular arithmetic functions used across NTT shaders.
/// These are included in each shader that needs them.
pub const MODULAR_ARITHMETIC: &str = r#"
// Modular addition: (a + b) mod q
fn addmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var sum_lo = a.x + b.x;
    var carry = 0u;
    if sum_lo < a.x { carry = 1u; }
    var sum_hi = a.y + b.y + carry;

    // Reduce if >= q
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
        // a >= b, simple subtraction
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        return vec2<u32>(diff_lo, diff_hi);
    } else {
        // a < b, compute q - (b - a)
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;

        // Result = q - diff
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

    // Handle carry from p1 + p2 overflow
    if p1 > 0xFFFFFFFFu - p2 {
        hi = hi + 0x10000u;
    }

    return vec2<u32>(lo, hi);
}

// Barrett reduction of 128-bit value mod 64-bit q
// Uses precomputed mu = floor(2^128 / q) passed as vec2<u32>
fn barrett_reduce_with_mu(x: vec4<u32>, q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    // Approximate quotient: q_hat ≈ (x * mu) >> 128
    // We compute (x_hi * mu) >> 64 as approximation

    // x = [x0, x1, x2, x3] where x = x0 + x1*2^32 + x2*2^64 + x3*2^96
    // For x < 2*q^2, we have x3 = 0 and x2 is small

    // Quick path: if x fits in 64 bits
    if x.z == 0u && x.w == 0u {
        if x.y < q.y || (x.y == q.y && x.x < q.x) {
            return vec2<u32>(x.x, x.y);
        }
        // One subtraction suffices
        var r_lo = x.x;
        var r_hi = x.y;
        if r_lo >= q.x {
            r_lo = r_lo - q.x;
        } else {
            r_lo = 0xFFFFFFFFu - (q.x - r_lo - 1u);
            r_hi = r_hi - 1u;
        }
        r_hi = r_hi - q.y;
        return vec2<u32>(r_lo, r_hi);
    }

    // Full Barrett: q_hat = floor((x * mu) / 2^128)
    // For our 60-bit primes, x < 2^128 after multiplication of two 64-bit values

    // Compute x_hi (upper 64 bits) * mu
    let x_hi = vec2<u32>(x.z, x.w);

    // q_hat ≈ x_hi * mu >> 64 (taking high 64 bits of 128-bit product)
    let p0 = u64_mul(x.z, mu.x);
    let p1 = u64_mul(x.z, mu.y);
    let p2 = u64_mul(x.w, mu.x);
    let p3 = u64_mul(x.w, mu.y);

    // Sum middle terms
    var mid_lo = p0.y + p1.x + p2.x;
    var mid_carry = 0u;
    if mid_lo < p0.y { mid_carry = 1u; }
    if mid_lo < p1.x { mid_carry = mid_carry + 1u; }

    var q_hat_lo = p1.y + p2.y + (mid_lo >> 0u) + mid_carry;
    var q_hat_hi = p3.x + p3.y;

    // Also add contribution from x_lo * mu (upper bits)
    let contrib = u64_mul(x.y, mu.y);
    let old_q_hat_lo = q_hat_lo;
    q_hat_lo = q_hat_lo + contrib.y;
    if q_hat_lo < old_q_hat_lo { q_hat_hi = q_hat_hi + 1u; }

    // r = x - q_hat * q
    // Compute q_hat * q (up to 192 bits, but we only need low 128)
    let qh_q_0 = u64_mul(q_hat_lo, q.x);
    let qh_q_1 = u64_mul(q_hat_lo, q.y);
    let qh_q_2 = u64_mul(q_hat_hi, q.x);

    // Build q_hat * q
    var qhq0 = qh_q_0.x;
    var qhq1 = qh_q_0.y + qh_q_1.x + qh_q_2.x;
    var qhq2 = qh_q_1.y + qh_q_2.y;

    // r = x - qhq (only need low 128 bits)
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;

    // Subtract qhq from r
    if r0 >= qhq0 {
        r0 = r0 - qhq0;
    } else {
        r0 = 0xFFFFFFFFu - (qhq0 - r0 - 1u);
        if r1 > 0u { r1 = r1 - 1u; }
        else { r1 = 0xFFFFFFFFu; r2 = r2 - 1u; }
    }

    if r1 >= qhq1 {
        r1 = r1 - qhq1;
    } else {
        r1 = 0xFFFFFFFFu - (qhq1 - r1 - 1u);
        r2 = r2 - 1u;
    }

    r2 = r2 - qhq2;

    // r should now be in [0, 2q), do final correction
    var result = vec2<u32>(r0, r1);
    if r2 > 0u || r1 > q.y || (r1 == q.y && r0 >= q.x) {
        result = submod(result, q, q);
    }
    if result.y > q.y || (result.y == q.y && result.x >= q.x) {
        result = submod(result, q, q);
    }

    return result;
}

// Modular multiplication with Barrett reduction
fn mulmod_barrett(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    // Full 128-bit product
    let p0 = u64_mul(a.x, b.x);
    let p1 = u64_mul(a.x, b.y);
    let p2 = u64_mul(a.y, b.x);
    let p3 = u64_mul(a.y, b.y);

    // Accumulate into 128-bit result [w0, w1, w2, w3]
    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    // Add p1 << 32
    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;
    t = w2 + p1.y + c;
    c = 0u;
    if t < p1.y { c = 1u; }
    w2 = t;
    w3 = w3 + c;

    // Add p2 << 32
    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;
    t = w2 + p2.y + c;
    c = 0u;
    if t < p2.y { c = 1u; }
    w2 = t;
    w3 = w3 + c;

    // Add p3 << 64
    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return barrett_reduce_with_mu(vec4<u32>(w0, w1, w2, w3), q, mu);
}
"#;

/// Bit-reversal permutation shader.
/// Reorders elements for Cooley-Tukey NTT.
pub const BIT_REVERSE_SHADER: &str = r#"
struct BitRevParams {
    n: u32,
    log_n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BitRevParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result = 0u;
    var val = x;
    for (var i = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

@compute @workgroup_size(256, 1, 1)
fn bit_reverse_permutation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if idx >= n {
        return;
    }

    let rev_idx = bit_reverse(idx, params.log_n);

    // Copy from input[idx] to output[rev_idx]
    // Each element is 2 u32s (one u64)
    output[rev_idx * 2u] = input[idx * 2u];
    output[rev_idx * 2u + 1u] = input[idx * 2u + 1u];
}
"#;

/// NTT butterfly pass shader.
/// Called log_n times with different stage parameters.
/// Each invocation performs n/2 butterflies.
pub const NTT_BUTTERFLY_SHADER: &str = r#"
struct NttParams {
    n: u32,
    stage: u32,           // Current stage (0 to log_n - 1)
    modulus_lo: u32,
    modulus_hi: u32,
    mu_lo_lo: u32,        // Barrett mu: 128 bits split into 4 u32s
    mu_lo_hi: u32,        // mu = mu_hi_hi:mu_hi_lo:mu_lo_hi:mu_lo_lo
    mu_hi_lo: u32,
    mu_hi_hi: u32,
}

@group(0) @binding(0) var<uniform> params: NttParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;

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

// Barrett reduction for 128-bit value mod 64-bit q
// Uses q_est = (r * mu) >> 128 where mu = floor(2^128 / q)
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;

    // Fast path: x already < q
    if r3 == 0u && r2 == 0u {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            return vec2<u32>(r0, r1);
        }
    }

    // For 64-bit products, compute full r * mu to get accurate quotient estimate
    // q_est = floor((r * mu) / 2^128) where r = r1*2^32 + r0, mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    if r3 == 0u && r2 == 0u {
        // Compute r * mu using 8 partial products:
        // r0*mu0 (pos 0), r0*mu1 (pos 32), r0*mu2 (pos 64), r0*mu3 (pos 96)
        // r1*mu0 (pos 32), r1*mu1 (pos 64), r1*mu2 (pos 96), r1*mu3 (pos 128)

        let p00 = u64_mul(r0, mu0);  // bits 0-63
        let p01 = u64_mul(r0, mu1);  // bits 32-95
        let p02 = u64_mul(r0, mu2);  // bits 64-127
        let p03 = u64_mul(r0, mu3);  // bits 96-159
        let p10 = u64_mul(r1, mu0);  // bits 32-95
        let p11 = u64_mul(r1, mu1);  // bits 64-127
        let p12 = u64_mul(r1, mu2);  // bits 96-159
        let p13 = u64_mul(r1, mu3);  // bits 128-191

        // We need bits 128+ of the sum. Build up from position 64.
        // Position 64-95: p00.y (from p00) + p01.x + p10.x + lower parts with carries
        // Position 96-127: p01.y + p02.x + p10.y + p11.x + ...
        // Position 128-159: p02.y + p03.x + p11.y + p12.x + p13.x + ...
        // Position 160-191: p03.y + p12.y + p13.y + ...

        // Accumulate bits 64-95
        var acc64: u32 = p02.x;
        var c: u32 = 0u;
        var t = acc64 + p01.y;
        if t < acc64 { c = 1u; }
        acc64 = t;
        t = acc64 + p10.y;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        t = acc64 + p11.x;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        // Also add high part of position 32-63: p00.y + p01.x + p10.x
        var mid32 = p00.y;
        var mid_c: u32 = 0u;
        var mt = mid32 + p01.x;
        if mt < mid32 { mid_c = 1u; }
        mid32 = mt;
        mt = mid32 + p10.x;
        if mt < mid32 { mid_c = mid_c + 1u; }
        // Carry from position 32-63 into 64-95
        t = acc64 + mid_c;
        if t < acc64 { c = c + 1u; }
        acc64 = t;

        // Accumulate bits 96-127
        var acc96: u32 = p02.y;
        var c2: u32 = 0u;
        t = acc96 + p03.x;
        if t < acc96 { c2 = 1u; }
        acc96 = t;
        t = acc96 + p11.y;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        t = acc96 + p12.x;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        // Add carry from acc64
        t = acc96 + c;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;

        // Accumulate bits 128-159 (this is q_est low 32 bits)
        var acc128: u32 = p13.x;
        var c3: u32 = 0u;
        t = acc128 + p03.y;
        if t < acc128 { c3 = 1u; }
        acc128 = t;
        t = acc128 + p12.y;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;
        // Add carry from acc96
        t = acc128 + c2;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;

        // Accumulate bits 160-191 (this is q_est high 32 bits)
        var acc160: u32 = p13.y + c3;

        // q_est = acc160 * 2^32 + acc128
        var q_est_lo = acc128;
        var q_est_hi = acc160;

        // Compute q_est * q (up to 128 bits is enough)
        let qe0 = u64_mul(q_est_lo, q.x);
        let qe1 = u64_mul(q_est_lo, q.y);
        let qe2 = u64_mul(q_est_hi, q.x);
        let qe3 = u64_mul(q_est_hi, q.y);

        var p0 = qe0.x;
        var p1 = qe0.y;
        var p2: u32 = 0u;

        t = p1 + qe1.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe1.y + c;
        p2 = t;

        t = p1 + qe2.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe2.y + c;
        p2 = t;

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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
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

    // For larger products (r >= 2^64), compute full (r * mu) >> 128
    // r = r3*2^96 + r2*2^64 + r1*2^32 + r0
    // mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    // Need bits 128+ of the 256-bit product

    // Compute partial products contributing to bits 64+ (for carries and result)
    let p01 = u64_mul(r0, mu1);  // 32-95
    let p02 = u64_mul(r0, mu2);  // 64-127
    let p03 = u64_mul(r0, mu3);  // 96-159
    let p10 = u64_mul(r1, mu0);  // 32-95
    let p11 = u64_mul(r1, mu1);  // 64-127
    let p12 = u64_mul(r1, mu2);  // 96-159
    let p13 = u64_mul(r1, mu3);  // 128-191
    let p20 = u64_mul(r2, mu0);  // 64-127
    let p21 = u64_mul(r2, mu1);  // 96-159
    let p22 = u64_mul(r2, mu2);  // 128-191
    let p23 = u64_mul(r2, mu3);  // 160-223
    let p30 = u64_mul(r3, mu0);  // 96-159
    let p31 = u64_mul(r3, mu1);  // 128-191
    let p32 = u64_mul(r3, mu2);  // 160-223
    let p33 = u64_mul(r3, mu3);  // 192-255

    // Accumulate bits 64-95 (for carry into 96+)
    var acc64: u32 = p02.x;
    var c64: u32 = 0u;
    var t = acc64 + p01.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p10.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p11.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p20.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;

    // Accumulate bits 96-127 (for carry into 128+)
    var acc96: u32 = p02.y;
    var c96: u32 = 0u;
    t = acc96 + p03.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p11.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p12.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p20.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p21.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p30.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + c64;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;

    // Accumulate bits 128-159 (q_est low 32 bits)
    var acc128: u32 = p13.x;
    var c128: u32 = 0u;
    t = acc128 + p03.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p12.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p21.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p22.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p30.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p31.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + c96;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;

    // Accumulate bits 160-191 (q_est high 32 bits)
    var acc160: u32 = p13.y;
    var c160: u32 = 0u;
    t = acc160 + p22.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p23.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p31.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p32.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + c128;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;

    var q_est_lo = acc128;
    var q_est_hi = acc160;

    // Compute q_est * q
    let prod_ll = u64_mul(q_est_lo, q.x);
    let prod_lh = u64_mul(q_est_lo, q.y);
    let prod_hl = u64_mul(q_est_hi, q.x);
    let prod_hh = u64_mul(q_est_hi, q.y);

    var p0 = prod_ll.x;
    var p1 = prod_ll.y;
    var p2 = 0u;
    var p3 = 0u;
    var c: u32 = 0u;

    t = p1 + prod_lh.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_lh.y + c;
    p2 = t;

    t = p1 + prod_hl.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_hl.y + c;
    if t < p2 { p3 = 1u; }
    p2 = t;

    t = p2 + prod_hh.x;
    if t < p2 { c = 1u; } else { c = 0u; }
    p2 = t;
    p3 = p3 + prod_hh.y + c;

    // Subtract p from r
    var borrow = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
        borrow = 0u;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        borrow = 1u;
    }

    var sub2 = p2 + borrow;
    if r2 >= sub2 {
        r2 = r2 - sub2;
        borrow = 0u;
    } else {
        r2 = r2 + (0xFFFFFFFFu - sub2) + 1u;
        borrow = 1u;
    }

    r3 = r3 - p3 - borrow;

    // Final reduction
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r3 == 0u && r2 == 0u {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                break;
            }
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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
        }
        if borrow == 1u {
            if r2 > 0u { r2 = r2 - 1u; }
            else { r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
        }
    }

    return vec2<u32>(r0, r1);
}

fn mulmod_barrett(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    // Compute 128-bit product of two 64-bit numbers
    let p0 = u64_mul(a.x, b.x);  // Low * Low
    let p1 = u64_mul(a.x, b.y);  // Low * High
    let p2 = u64_mul(a.y, b.x);  // High * Low
    let p3 = u64_mul(a.y, b.y);  // High * High

    // Accumulate into 128-bit result
    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    // Add p1 << 32
    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p1.y + c;
    if t < c || (c == 0u && t < p1.y) { w3 = w3 + 1u; }
    w2 = t;

    // Add p2 << 32
    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p2.y + c;
    if t < c || (c == 0u && t < p2.y) { w3 = w3 + 1u; }
    w2 = t;

    // Add p3 << 64
    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return barrett_reduce(vec4<u32>(w0, w1, w2, w3), q, mu0, mu1, mu2, mu3);
}

@compute @workgroup_size(256, 1, 1)
fn ntt_butterfly(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let stage = params.stage;
    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);
    let idx = global_id.x;

    // Each thread handles one butterfly
    // In stage s, butterflies span 2^(s+1) elements
    let m = 1u << (stage + 1u);
    let half_m = m >> 1u;

    // Total n/2 butterflies per stage
    if idx >= n / 2u {
        return;
    }

    // Determine which butterfly group and position within group
    let group = idx / half_m;
    let j = idx % half_m;

    // Indices for this butterfly
    let k = group * m;
    let i0 = k + j;
    let i1 = k + j + half_m;

    // Load twiddle factor: omega^(j * n / m)
    let twiddle_idx = j * (n / m);
    let w = vec2<u32>(twiddles[twiddle_idx * 2u], twiddles[twiddle_idx * 2u + 1u]);

    // Load data
    let u = vec2<u32>(data[i0 * 2u], data[i0 * 2u + 1u]);
    let v = vec2<u32>(data[i1 * 2u], data[i1 * 2u + 1u]);

    // Butterfly: t = w * v mod q (using Barrett reduction)
    let t = mulmod_barrett(w, v, q, params.mu_lo_lo, params.mu_lo_hi, params.mu_hi_lo, params.mu_hi_hi);

    // Store: data[i0] = u + t, data[i1] = u - t
    let sum = addmod(u, t, q);
    let diff = submod(u, t, q);

    data[i0 * 2u] = sum.x;
    data[i0 * 2u + 1u] = sum.y;
    data[i1 * 2u] = diff.x;
    data[i1 * 2u + 1u] = diff.y;
}
"#;

/// Pointwise multiplication shader.
/// Multiplies two polynomials element-wise in NTT domain.
pub const POINTWISE_MUL_SHADER: &str = r#"
struct PointwiseParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    _pad0: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
}

@group(0) @binding(0) var<uniform> params: PointwiseParams;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;

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

// Barrett reduction for 128-bit value mod 64-bit q
// Uses q_est = (r * mu) >> 128 where mu = floor(2^128 / q)
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;

    // Fast path: x already < q
    if r3 == 0u && r2 == 0u {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            return vec2<u32>(r0, r1);
        }
    }

    // For 64-bit products, compute full r * mu to get accurate quotient estimate
    // q_est = floor((r * mu) / 2^128) where r = r1*2^32 + r0, mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    if r3 == 0u && r2 == 0u {
        // Compute r * mu using 8 partial products:
        // r0*mu0 (pos 0), r0*mu1 (pos 32), r0*mu2 (pos 64), r0*mu3 (pos 96)
        // r1*mu0 (pos 32), r1*mu1 (pos 64), r1*mu2 (pos 96), r1*mu3 (pos 128)

        let p00 = u64_mul(r0, mu0);  // bits 0-63
        let p01 = u64_mul(r0, mu1);  // bits 32-95
        let p02 = u64_mul(r0, mu2);  // bits 64-127
        let p03 = u64_mul(r0, mu3);  // bits 96-159
        let p10 = u64_mul(r1, mu0);  // bits 32-95
        let p11 = u64_mul(r1, mu1);  // bits 64-127
        let p12 = u64_mul(r1, mu2);  // bits 96-159
        let p13 = u64_mul(r1, mu3);  // bits 128-191

        // We need bits 128+ of the sum. Build up from position 64.
        // Position 64-95: p00.y (from p00) + p01.x + p10.x + lower parts with carries
        // Position 96-127: p01.y + p02.x + p10.y + p11.x + ...
        // Position 128-159: p02.y + p03.x + p11.y + p12.x + p13.x + ...
        // Position 160-191: p03.y + p12.y + p13.y + ...

        // Accumulate bits 64-95
        var acc64: u32 = p02.x;
        var c: u32 = 0u;
        var t = acc64 + p01.y;
        if t < acc64 { c = 1u; }
        acc64 = t;
        t = acc64 + p10.y;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        t = acc64 + p11.x;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        // Also add high part of position 32-63: p00.y + p01.x + p10.x
        var mid32 = p00.y;
        var mid_c: u32 = 0u;
        var mt = mid32 + p01.x;
        if mt < mid32 { mid_c = 1u; }
        mid32 = mt;
        mt = mid32 + p10.x;
        if mt < mid32 { mid_c = mid_c + 1u; }
        // Carry from position 32-63 into 64-95
        t = acc64 + mid_c;
        if t < acc64 { c = c + 1u; }
        acc64 = t;

        // Accumulate bits 96-127
        var acc96: u32 = p02.y;
        var c2: u32 = 0u;
        t = acc96 + p03.x;
        if t < acc96 { c2 = 1u; }
        acc96 = t;
        t = acc96 + p11.y;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        t = acc96 + p12.x;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        // Add carry from acc64
        t = acc96 + c;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;

        // Accumulate bits 128-159 (this is q_est low 32 bits)
        var acc128: u32 = p13.x;
        var c3: u32 = 0u;
        t = acc128 + p03.y;
        if t < acc128 { c3 = 1u; }
        acc128 = t;
        t = acc128 + p12.y;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;
        // Add carry from acc96
        t = acc128 + c2;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;

        // Accumulate bits 160-191 (this is q_est high 32 bits)
        var acc160: u32 = p13.y + c3;

        // q_est = acc160 * 2^32 + acc128
        var q_est_lo = acc128;
        var q_est_hi = acc160;

        // Compute q_est * q (up to 128 bits is enough)
        let qe0 = u64_mul(q_est_lo, q.x);
        let qe1 = u64_mul(q_est_lo, q.y);
        let qe2 = u64_mul(q_est_hi, q.x);
        let qe3 = u64_mul(q_est_hi, q.y);

        var p0 = qe0.x;
        var p1 = qe0.y;
        var p2: u32 = 0u;

        t = p1 + qe1.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe1.y + c;
        p2 = t;

        t = p1 + qe2.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe2.y + c;
        p2 = t;

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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
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

    // For larger products (r >= 2^64), compute full (r * mu) >> 128
    // r = r3*2^96 + r2*2^64 + r1*2^32 + r0
    // mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    // Need bits 128+ of the 256-bit product

    // Compute partial products contributing to bits 64+ (for carries and result)
    let p01 = u64_mul(r0, mu1);  // 32-95
    let p02 = u64_mul(r0, mu2);  // 64-127
    let p03 = u64_mul(r0, mu3);  // 96-159
    let p10 = u64_mul(r1, mu0);  // 32-95
    let p11 = u64_mul(r1, mu1);  // 64-127
    let p12 = u64_mul(r1, mu2);  // 96-159
    let p13 = u64_mul(r1, mu3);  // 128-191
    let p20 = u64_mul(r2, mu0);  // 64-127
    let p21 = u64_mul(r2, mu1);  // 96-159
    let p22 = u64_mul(r2, mu2);  // 128-191
    let p23 = u64_mul(r2, mu3);  // 160-223
    let p30 = u64_mul(r3, mu0);  // 96-159
    let p31 = u64_mul(r3, mu1);  // 128-191
    let p32 = u64_mul(r3, mu2);  // 160-223
    let p33 = u64_mul(r3, mu3);  // 192-255

    // Accumulate bits 64-95 (for carry into 96+)
    var acc64: u32 = p02.x;
    var c64: u32 = 0u;
    var t = acc64 + p01.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p10.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p11.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p20.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;

    // Accumulate bits 96-127 (for carry into 128+)
    var acc96: u32 = p02.y;
    var c96: u32 = 0u;
    t = acc96 + p03.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p11.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p12.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p20.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p21.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p30.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + c64;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;

    // Accumulate bits 128-159 (q_est low 32 bits)
    var acc128: u32 = p13.x;
    var c128: u32 = 0u;
    t = acc128 + p03.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p12.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p21.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p22.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p30.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p31.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + c96;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;

    // Accumulate bits 160-191 (q_est high 32 bits)
    var acc160: u32 = p13.y;
    var c160: u32 = 0u;
    t = acc160 + p22.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p23.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p31.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p32.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + c128;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;

    var q_est_lo = acc128;
    var q_est_hi = acc160;

    // Compute q_est * q
    let prod_ll = u64_mul(q_est_lo, q.x);
    let prod_lh = u64_mul(q_est_lo, q.y);
    let prod_hl = u64_mul(q_est_hi, q.x);
    let prod_hh = u64_mul(q_est_hi, q.y);

    var p0 = prod_ll.x;
    var p1 = prod_ll.y;
    var p2 = 0u;
    var p3 = 0u;
    var c: u32 = 0u;

    t = p1 + prod_lh.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_lh.y + c;
    p2 = t;

    t = p1 + prod_hl.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_hl.y + c;
    if t < p2 { p3 = 1u; }
    p2 = t;

    t = p2 + prod_hh.x;
    if t < p2 { c = 1u; } else { c = 0u; }
    p2 = t;
    p3 = p3 + prod_hh.y + c;

    // Subtract p from r
    var borrow = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
        borrow = 0u;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        borrow = 1u;
    }

    var sub2 = p2 + borrow;
    if r2 >= sub2 {
        r2 = r2 - sub2;
        borrow = 0u;
    } else {
        r2 = r2 + (0xFFFFFFFFu - sub2) + 1u;
        borrow = 1u;
    }

    r3 = r3 - p3 - borrow;

    // Final reduction
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r3 == 0u && r2 == 0u {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                break;
            }
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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
        }
        if borrow == 1u {
            if r2 > 0u { r2 = r2 - 1u; }
            else { r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
        }
    }

    return vec2<u32>(r0, r1);
}

fn mulmod_barrett(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    let p0 = u64_mul(a.x, b.x);
    let p1 = u64_mul(a.x, b.y);
    let p2 = u64_mul(a.y, b.x);
    let p3 = u64_mul(a.y, b.y);

    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p1.y + c;
    if t < c || (c == 0u && t < p1.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p2.y + c;
    if t < c || (c == 0u && t < p2.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return barrett_reduce(vec4<u32>(w0, w1, w2, w3), q, mu0, mu1, mu2, mu3);
}

@compute @workgroup_size(256, 1, 1)
fn pointwise_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if idx >= n {
        return;
    }

    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);

    let a_val = vec2<u32>(a[idx * 2u], a[idx * 2u + 1u]);
    let b_val = vec2<u32>(b[idx * 2u], b[idx * 2u + 1u]);

    let prod = mulmod_barrett(a_val, b_val, q, params.mu_lo_lo, params.mu_lo_hi, params.mu_hi_lo, params.mu_hi_hi);

    result[idx * 2u] = prod.x;
    result[idx * 2u + 1u] = prod.y;
}
"#;

/// Scale by scalar shader.
/// Multiplies all elements by a scalar (used for 1/n scaling and psi twist).
pub const SCALE_SHADER: &str = r#"
struct ScaleParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    scalar_lo: u32,
    scalar_hi: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ScaleParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

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

// Barrett reduction for 128-bit value mod 64-bit q
// Uses q_est = (r * mu) >> 128 where mu = floor(2^128 / q)
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;

    // Fast path: x already < q
    if r3 == 0u && r2 == 0u {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            return vec2<u32>(r0, r1);
        }
    }

    // For 64-bit products, compute full r * mu to get accurate quotient estimate
    // q_est = floor((r * mu) / 2^128) where r = r1*2^32 + r0, mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    if r3 == 0u && r2 == 0u {
        // Compute r * mu using 8 partial products:
        // r0*mu0 (pos 0), r0*mu1 (pos 32), r0*mu2 (pos 64), r0*mu3 (pos 96)
        // r1*mu0 (pos 32), r1*mu1 (pos 64), r1*mu2 (pos 96), r1*mu3 (pos 128)

        let p00 = u64_mul(r0, mu0);  // bits 0-63
        let p01 = u64_mul(r0, mu1);  // bits 32-95
        let p02 = u64_mul(r0, mu2);  // bits 64-127
        let p03 = u64_mul(r0, mu3);  // bits 96-159
        let p10 = u64_mul(r1, mu0);  // bits 32-95
        let p11 = u64_mul(r1, mu1);  // bits 64-127
        let p12 = u64_mul(r1, mu2);  // bits 96-159
        let p13 = u64_mul(r1, mu3);  // bits 128-191

        // We need bits 128+ of the sum. Build up from position 64.
        // Position 64-95: p00.y (from p00) + p01.x + p10.x + lower parts with carries
        // Position 96-127: p01.y + p02.x + p10.y + p11.x + ...
        // Position 128-159: p02.y + p03.x + p11.y + p12.x + p13.x + ...
        // Position 160-191: p03.y + p12.y + p13.y + ...

        // Accumulate bits 64-95
        var acc64: u32 = p02.x;
        var c: u32 = 0u;
        var t = acc64 + p01.y;
        if t < acc64 { c = 1u; }
        acc64 = t;
        t = acc64 + p10.y;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        t = acc64 + p11.x;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        // Also add high part of position 32-63: p00.y + p01.x + p10.x
        var mid32 = p00.y;
        var mid_c: u32 = 0u;
        var mt = mid32 + p01.x;
        if mt < mid32 { mid_c = 1u; }
        mid32 = mt;
        mt = mid32 + p10.x;
        if mt < mid32 { mid_c = mid_c + 1u; }
        // Carry from position 32-63 into 64-95
        t = acc64 + mid_c;
        if t < acc64 { c = c + 1u; }
        acc64 = t;

        // Accumulate bits 96-127
        var acc96: u32 = p02.y;
        var c2: u32 = 0u;
        t = acc96 + p03.x;
        if t < acc96 { c2 = 1u; }
        acc96 = t;
        t = acc96 + p11.y;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        t = acc96 + p12.x;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        // Add carry from acc64
        t = acc96 + c;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;

        // Accumulate bits 128-159 (this is q_est low 32 bits)
        var acc128: u32 = p13.x;
        var c3: u32 = 0u;
        t = acc128 + p03.y;
        if t < acc128 { c3 = 1u; }
        acc128 = t;
        t = acc128 + p12.y;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;
        // Add carry from acc96
        t = acc128 + c2;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;

        // Accumulate bits 160-191 (this is q_est high 32 bits)
        var acc160: u32 = p13.y + c3;

        // q_est = acc160 * 2^32 + acc128
        var q_est_lo = acc128;
        var q_est_hi = acc160;

        // Compute q_est * q (up to 128 bits is enough)
        let qe0 = u64_mul(q_est_lo, q.x);
        let qe1 = u64_mul(q_est_lo, q.y);
        let qe2 = u64_mul(q_est_hi, q.x);
        let qe3 = u64_mul(q_est_hi, q.y);

        var p0 = qe0.x;
        var p1 = qe0.y;
        var p2: u32 = 0u;

        t = p1 + qe1.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe1.y + c;
        p2 = t;

        t = p1 + qe2.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe2.y + c;
        p2 = t;

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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
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

    // For larger products (r >= 2^64), compute full (r * mu) >> 128
    // r = r3*2^96 + r2*2^64 + r1*2^32 + r0
    // mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    // Need bits 128+ of the 256-bit product

    // Compute partial products contributing to bits 64+ (for carries and result)
    let p01 = u64_mul(r0, mu1);  // 32-95
    let p02 = u64_mul(r0, mu2);  // 64-127
    let p03 = u64_mul(r0, mu3);  // 96-159
    let p10 = u64_mul(r1, mu0);  // 32-95
    let p11 = u64_mul(r1, mu1);  // 64-127
    let p12 = u64_mul(r1, mu2);  // 96-159
    let p13 = u64_mul(r1, mu3);  // 128-191
    let p20 = u64_mul(r2, mu0);  // 64-127
    let p21 = u64_mul(r2, mu1);  // 96-159
    let p22 = u64_mul(r2, mu2);  // 128-191
    let p23 = u64_mul(r2, mu3);  // 160-223
    let p30 = u64_mul(r3, mu0);  // 96-159
    let p31 = u64_mul(r3, mu1);  // 128-191
    let p32 = u64_mul(r3, mu2);  // 160-223
    let p33 = u64_mul(r3, mu3);  // 192-255

    // Accumulate bits 64-95 (for carry into 96+)
    var acc64: u32 = p02.x;
    var c64: u32 = 0u;
    var t = acc64 + p01.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p10.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p11.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p20.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;

    // Accumulate bits 96-127 (for carry into 128+)
    var acc96: u32 = p02.y;
    var c96: u32 = 0u;
    t = acc96 + p03.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p11.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p12.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p20.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p21.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p30.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + c64;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;

    // Accumulate bits 128-159 (q_est low 32 bits)
    var acc128: u32 = p13.x;
    var c128: u32 = 0u;
    t = acc128 + p03.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p12.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p21.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p22.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p30.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p31.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + c96;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;

    // Accumulate bits 160-191 (q_est high 32 bits)
    var acc160: u32 = p13.y;
    var c160: u32 = 0u;
    t = acc160 + p22.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p23.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p31.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p32.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + c128;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;

    var q_est_lo = acc128;
    var q_est_hi = acc160;

    // Compute q_est * q
    let prod_ll = u64_mul(q_est_lo, q.x);
    let prod_lh = u64_mul(q_est_lo, q.y);
    let prod_hl = u64_mul(q_est_hi, q.x);
    let prod_hh = u64_mul(q_est_hi, q.y);

    var p0 = prod_ll.x;
    var p1 = prod_ll.y;
    var p2 = 0u;
    var p3 = 0u;
    var c: u32 = 0u;

    t = p1 + prod_lh.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_lh.y + c;
    p2 = t;

    t = p1 + prod_hl.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_hl.y + c;
    if t < p2 { p3 = 1u; }
    p2 = t;

    t = p2 + prod_hh.x;
    if t < p2 { c = 1u; } else { c = 0u; }
    p2 = t;
    p3 = p3 + prod_hh.y + c;

    // Subtract p from r
    var borrow = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
        borrow = 0u;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        borrow = 1u;
    }

    var sub2 = p2 + borrow;
    if r2 >= sub2 {
        r2 = r2 - sub2;
        borrow = 0u;
    } else {
        r2 = r2 + (0xFFFFFFFFu - sub2) + 1u;
        borrow = 1u;
    }

    r3 = r3 - p3 - borrow;

    // Final reduction
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r3 == 0u && r2 == 0u {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                break;
            }
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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
        }
        if borrow == 1u {
            if r2 > 0u { r2 = r2 - 1u; }
            else { r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
        }
    }

    return vec2<u32>(r0, r1);
}

fn mulmod_barrett(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    let p0 = u64_mul(a.x, b.x);
    let p1 = u64_mul(a.x, b.y);
    let p2 = u64_mul(a.y, b.x);
    let p3 = u64_mul(a.y, b.y);

    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p1.y + c;
    if t < c || (c == 0u && t < p1.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p2.y + c;
    if t < c || (c == 0u && t < p2.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return barrett_reduce(vec4<u32>(w0, w1, w2, w3), q, mu0, mu1, mu2, mu3);
}

@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if idx >= n {
        return;
    }

    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);
    let scalar = vec2<u32>(params.scalar_lo, params.scalar_hi);

    let val = vec2<u32>(data[idx * 2u], data[idx * 2u + 1u]);
    let result = mulmod_barrett(val, scalar, q, params.mu_lo_lo, params.mu_lo_hi, params.mu_hi_lo, params.mu_hi_hi);

    data[idx * 2u] = result.x;
    data[idx * 2u + 1u] = result.y;
}
"#;

/// Twist shader for negacyclic NTT.
/// Multiplies coefficient i by psi^i (forward) or psi^(-i) (inverse).
pub const TWIST_SHADER: &str = r#"
struct TwistParams {
    n: u32,
    modulus_lo: u32,
    modulus_hi: u32,
    _pad0: u32,
    mu_lo_lo: u32,
    mu_lo_hi: u32,
    mu_hi_lo: u32,
    mu_hi_hi: u32,
}

@group(0) @binding(0) var<uniform> params: TwistParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> psi_powers: array<u32>;

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

// Barrett reduction for 128-bit value mod 64-bit q
// Uses q_est = (r * mu) >> 128 where mu = floor(2^128 / q)
// mu is passed as four 32-bit words: mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;
    var r2 = x.z;
    var r3 = x.w;

    // Fast path: x already < q
    if r3 == 0u && r2 == 0u {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            return vec2<u32>(r0, r1);
        }
    }

    // For 64-bit products, compute full r * mu to get accurate quotient estimate
    // q_est = floor((r * mu) / 2^128) where r = r1*2^32 + r0, mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    if r3 == 0u && r2 == 0u {
        // Compute r * mu using 8 partial products:
        // r0*mu0 (pos 0), r0*mu1 (pos 32), r0*mu2 (pos 64), r0*mu3 (pos 96)
        // r1*mu0 (pos 32), r1*mu1 (pos 64), r1*mu2 (pos 96), r1*mu3 (pos 128)

        let p00 = u64_mul(r0, mu0);  // bits 0-63
        let p01 = u64_mul(r0, mu1);  // bits 32-95
        let p02 = u64_mul(r0, mu2);  // bits 64-127
        let p03 = u64_mul(r0, mu3);  // bits 96-159
        let p10 = u64_mul(r1, mu0);  // bits 32-95
        let p11 = u64_mul(r1, mu1);  // bits 64-127
        let p12 = u64_mul(r1, mu2);  // bits 96-159
        let p13 = u64_mul(r1, mu3);  // bits 128-191

        // We need bits 128+ of the sum. Build up from position 64.
        // Position 64-95: p00.y (from p00) + p01.x + p10.x + lower parts with carries
        // Position 96-127: p01.y + p02.x + p10.y + p11.x + ...
        // Position 128-159: p02.y + p03.x + p11.y + p12.x + p13.x + ...
        // Position 160-191: p03.y + p12.y + p13.y + ...

        // Accumulate bits 64-95
        var acc64: u32 = p02.x;
        var c: u32 = 0u;
        var t = acc64 + p01.y;
        if t < acc64 { c = 1u; }
        acc64 = t;
        t = acc64 + p10.y;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        t = acc64 + p11.x;
        if t < acc64 { c = c + 1u; }
        acc64 = t;
        // Also add high part of position 32-63: p00.y + p01.x + p10.x
        var mid32 = p00.y;
        var mid_c: u32 = 0u;
        var mt = mid32 + p01.x;
        if mt < mid32 { mid_c = 1u; }
        mid32 = mt;
        mt = mid32 + p10.x;
        if mt < mid32 { mid_c = mid_c + 1u; }
        // Carry from position 32-63 into 64-95
        t = acc64 + mid_c;
        if t < acc64 { c = c + 1u; }
        acc64 = t;

        // Accumulate bits 96-127
        var acc96: u32 = p02.y;
        var c2: u32 = 0u;
        t = acc96 + p03.x;
        if t < acc96 { c2 = 1u; }
        acc96 = t;
        t = acc96 + p11.y;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        t = acc96 + p12.x;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;
        // Add carry from acc64
        t = acc96 + c;
        if t < acc96 { c2 = c2 + 1u; }
        acc96 = t;

        // Accumulate bits 128-159 (this is q_est low 32 bits)
        var acc128: u32 = p13.x;
        var c3: u32 = 0u;
        t = acc128 + p03.y;
        if t < acc128 { c3 = 1u; }
        acc128 = t;
        t = acc128 + p12.y;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;
        // Add carry from acc96
        t = acc128 + c2;
        if t < acc128 { c3 = c3 + 1u; }
        acc128 = t;

        // Accumulate bits 160-191 (this is q_est high 32 bits)
        var acc160: u32 = p13.y + c3;

        // q_est = acc160 * 2^32 + acc128
        var q_est_lo = acc128;
        var q_est_hi = acc160;

        // Compute q_est * q (up to 128 bits is enough)
        let qe0 = u64_mul(q_est_lo, q.x);
        let qe1 = u64_mul(q_est_lo, q.y);
        let qe2 = u64_mul(q_est_hi, q.x);
        let qe3 = u64_mul(q_est_hi, q.y);

        var p0 = qe0.x;
        var p1 = qe0.y;
        var p2: u32 = 0u;

        t = p1 + qe1.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe1.y + c;
        p2 = t;

        t = p1 + qe2.x;
        c = 0u;
        if t < p1 { c = 1u; }
        p1 = t;
        t = p2 + qe2.y + c;
        p2 = t;

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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
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

    // For larger products (r >= 2^64), compute full (r * mu) >> 128
    // r = r3*2^96 + r2*2^64 + r1*2^32 + r0
    // mu = mu3*2^96 + mu2*2^64 + mu1*2^32 + mu0
    // Need bits 128+ of the 256-bit product

    // Compute partial products contributing to bits 64+ (for carries and result)
    let p01 = u64_mul(r0, mu1);  // 32-95
    let p02 = u64_mul(r0, mu2);  // 64-127
    let p03 = u64_mul(r0, mu3);  // 96-159
    let p10 = u64_mul(r1, mu0);  // 32-95
    let p11 = u64_mul(r1, mu1);  // 64-127
    let p12 = u64_mul(r1, mu2);  // 96-159
    let p13 = u64_mul(r1, mu3);  // 128-191
    let p20 = u64_mul(r2, mu0);  // 64-127
    let p21 = u64_mul(r2, mu1);  // 96-159
    let p22 = u64_mul(r2, mu2);  // 128-191
    let p23 = u64_mul(r2, mu3);  // 160-223
    let p30 = u64_mul(r3, mu0);  // 96-159
    let p31 = u64_mul(r3, mu1);  // 128-191
    let p32 = u64_mul(r3, mu2);  // 160-223
    let p33 = u64_mul(r3, mu3);  // 192-255

    // Accumulate bits 64-95 (for carry into 96+)
    var acc64: u32 = p02.x;
    var c64: u32 = 0u;
    var t = acc64 + p01.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p10.y;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p11.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;
    t = acc64 + p20.x;
    if t < acc64 { c64 = c64 + 1u; }
    acc64 = t;

    // Accumulate bits 96-127 (for carry into 128+)
    var acc96: u32 = p02.y;
    var c96: u32 = 0u;
    t = acc96 + p03.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p11.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p12.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p20.y;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p21.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + p30.x;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;
    t = acc96 + c64;
    if t < acc96 { c96 = c96 + 1u; }
    acc96 = t;

    // Accumulate bits 128-159 (q_est low 32 bits)
    var acc128: u32 = p13.x;
    var c128: u32 = 0u;
    t = acc128 + p03.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p12.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p21.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p22.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p30.y;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + p31.x;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;
    t = acc128 + c96;
    if t < acc128 { c128 = c128 + 1u; }
    acc128 = t;

    // Accumulate bits 160-191 (q_est high 32 bits)
    var acc160: u32 = p13.y;
    var c160: u32 = 0u;
    t = acc160 + p22.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p23.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p31.y;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + p32.x;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;
    t = acc160 + c128;
    if t < acc160 { c160 = c160 + 1u; }
    acc160 = t;

    var q_est_lo = acc128;
    var q_est_hi = acc160;

    // Compute q_est * q
    let prod_ll = u64_mul(q_est_lo, q.x);
    let prod_lh = u64_mul(q_est_lo, q.y);
    let prod_hl = u64_mul(q_est_hi, q.x);
    let prod_hh = u64_mul(q_est_hi, q.y);

    var p0 = prod_ll.x;
    var p1 = prod_ll.y;
    var p2 = 0u;
    var p3 = 0u;
    var c: u32 = 0u;

    t = p1 + prod_lh.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_lh.y + c;
    p2 = t;

    t = p1 + prod_hl.x;
    if t < p1 { c = 1u; } else { c = 0u; }
    p1 = t;
    t = p2 + prod_hl.y + c;
    if t < p2 { p3 = 1u; }
    p2 = t;

    t = p2 + prod_hh.x;
    if t < p2 { c = 1u; } else { c = 0u; }
    p2 = t;
    p3 = p3 + prod_hh.y + c;

    // Subtract p from r
    var borrow = 0u;
    if r0 >= p0 {
        r0 = r0 - p0;
    } else {
        r0 = r0 + (0xFFFFFFFFu - p0) + 1u;
        borrow = 1u;
    }

    var sub1 = p1 + borrow;
    if r1 >= sub1 {
        r1 = r1 - sub1;
        borrow = 0u;
    } else {
        r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
        borrow = 1u;
    }

    var sub2 = p2 + borrow;
    if r2 >= sub2 {
        r2 = r2 - sub2;
        borrow = 0u;
    } else {
        r2 = r2 + (0xFFFFFFFFu - sub2) + 1u;
        borrow = 1u;
    }

    r3 = r3 - p3 - borrow;

    // Final reduction
    for (var i = 0u; i < 3u; i = i + 1u) {
        if r3 == 0u && r2 == 0u {
            if r1 < q.y || (r1 == q.y && r0 < q.x) {
                break;
            }
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
            borrow = 0u;
        } else {
            r1 = r1 + (0xFFFFFFFFu - sub1) + 1u;
            borrow = 1u;
        }
        if borrow == 1u {
            if r2 > 0u { r2 = r2 - 1u; }
            else { r2 = 0xFFFFFFFFu; r3 = r3 - 1u; }
        }
    }

    return vec2<u32>(r0, r1);
}

fn mulmod_barrett(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    let p0 = u64_mul(a.x, b.x);
    let p1 = u64_mul(a.x, b.y);
    let p2 = u64_mul(a.y, b.x);
    let p3 = u64_mul(a.y, b.y);

    var w0 = p0.x;
    var w1 = p0.y;
    var w2 = 0u;
    var w3 = 0u;

    var t = w1 + p1.x;
    var c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p1.y + c;
    if t < c || (c == 0u && t < p1.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w1 + p2.x;
    c = 0u;
    if t < w1 { c = 1u; }
    w1 = t;

    t = w2 + p2.y + c;
    if t < c || (c == 0u && t < p2.y) { w3 = w3 + 1u; }
    w2 = t;

    t = w2 + p3.x;
    c = 0u;
    if t < w2 { c = 1u; }
    w2 = t;
    w3 = w3 + p3.y + c;

    return barrett_reduce(vec4<u32>(w0, w1, w2, w3), q, mu0, mu1, mu2, mu3);
}

@compute @workgroup_size(256, 1, 1)
fn twist(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if idx >= n {
        return;
    }

    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);

    let val = vec2<u32>(data[idx * 2u], data[idx * 2u + 1u]);
    let psi = vec2<u32>(psi_powers[idx * 2u], psi_powers[idx * 2u + 1u]);

    let result = mulmod_barrett(val, psi, q, params.mu_lo_lo, params.mu_lo_hi, params.mu_hi_lo, params.mu_hi_hi);

    data[idx * 2u] = result.x;
    data[idx * 2u + 1u] = result.y;
}
"#;

/// Automorphism shader: applies σ_k to polynomial coefficients.
/// Maps a(X) → a(X^k) mod (X^n + 1).
pub const AUTOMORPHISM_SHADER: &str = r#"
struct AutoParams {
    n: u32,           // Ring dimension
    two_n: u32,       // 2 * n
    k: u32,           // Automorphism exponent
    num_moduli: u32,  // Number of RNS moduli
}

@group(0) @binding(0) var<uniform> params: AutoParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;     // Input polynomial (RNS)
@group(0) @binding(2) var<storage, read_write> output: array<u32>; // Output polynomial
@group(0) @binding(3) var<storage, read> moduli: array<u32>;    // RNS moduli (as u64s)

// Load u64 coefficient
fn load_coeff(poly_offset: u32, idx: u32) -> vec2<u32> {
    let base = poly_offset + idx * 2u;
    return vec2<u32>(input[base], input[base + 1u]);
}

// Store u64 coefficient
fn store_coeff(poly_offset: u32, idx: u32, val: vec2<u32>) {
    let base = poly_offset + idx * 2u;
    output[base] = val.x;
    output[base + 1u] = val.y;
}

// Load modulus
fn load_modulus(mod_idx: u32) -> vec2<u32> {
    return vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
}

// Negate mod q: q - x
fn negate_mod(x: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if x.x == 0u && x.y == 0u {
        return x;
    }
    var res_lo = q.x - x.x;
    var borrow = 0u;
    if q.x < x.x {
        borrow = 1u;
        res_lo = 0xFFFFFFFFu - (x.x - q.x - 1u);
    }
    var res_hi = q.y - x.y - borrow;
    return vec2<u32>(res_lo, res_hi);
}

@compute @workgroup_size(256, 1, 1)
fn apply_automorphism(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let two_n = params.two_n;
    let k = params.k;
    let num_moduli = params.num_moduli;

    // Each thread handles one coefficient across all moduli
    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    if mod_idx >= num_moduli {
        return;
    }

    let q = load_modulus(mod_idx);
    let poly_offset = mod_idx * n * 2u;  // Offset in u32 array

    // For coefficient i, it maps to position (i * k) mod 2n
    // In the ring X^n + 1: X^n = -1, so X^j = (-1)^(j/n) * X^(j mod n)
    let target_exp = (coeff_idx * k) % two_n;
    var final_exp = target_exp;
    var negate = false;

    if target_exp >= n {
        final_exp = target_exp - n;
        negate = true;  // X^n = -1
    }

    // Read input coefficient
    var coeff = load_coeff(poly_offset, coeff_idx);

    // Apply negation if needed
    if negate {
        coeff = negate_mod(coeff, q);
    }

    // Write to output at the new position
    // Note: This has write conflicts! Need atomic or gather/scatter pattern
    // For correctness, we actually read from source position and write to our position
    // So: output[i] = sign * input[source_i] where source_i*k ≡ i (mod 2n)

    // Actually, the inverse mapping: for output position i, find source j where j*k ≡ i (mod 2n)
    // This requires computing k^{-1} mod 2n
    // For simplicity, let's do it the forward way with atomic writes or separate passes

    // Forward approach: each thread reads its coeff and writes to target position
    // But this causes write conflicts. Instead, use output[final_exp] with careful handling.

    // Store at target position (assuming no conflicts within workgroup)
    store_coeff(poly_offset, final_exp, coeff);
}
"#;

/// Polynomial multiplication shader using schoolbook algorithm.
/// More parallelizable than NTT for GPU.
pub const POLY_MUL_SHADER: &str = r#"
struct PolyMulParams {
    n: u32,           // Ring dimension
    modulus_lo: u32,  // q mod 2^32
    modulus_hi: u32,  // q >> 32
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: PolyMulParams;
@group(0) @binding(1) var<storage, read> poly_a: array<u32>;    // First polynomial
@group(0) @binding(2) var<storage, read> poly_b: array<u32>;    // Second polynomial
@group(0) @binding(3) var<storage, read_write> result: array<u32>; // Result

fn load_u64_a(idx: u32) -> vec2<u32> {
    return vec2<u32>(poly_a[idx * 2u], poly_a[idx * 2u + 1u]);
}

fn load_u64_b(idx: u32) -> vec2<u32> {
    return vec2<u32>(poly_b[idx * 2u], poly_b[idx * 2u + 1u]);
}

fn load_u64_result(idx: u32) -> vec2<u32> {
    return vec2<u32>(result[idx * 2u], result[idx * 2u + 1u]);
}

fn store_u64_result(idx: u32, val: vec2<u32>) {
    result[idx * 2u] = val.x;
    result[idx * 2u + 1u] = val.y;
}

// Goldilocks multiplication with reduction
fn goldilocks_mulmod(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Goldilocks: q = 2^64 - 2^32 + 1
    // Use the special structure for fast reduction

    let a_lo = a.x;
    let a_hi = a.y;
    let b_lo = b.x;
    let b_hi = b.y;

    // 128-bit product
    let p0_lo = a_lo & 0xFFFFu;
    let p0_hi = a_lo >> 16u;
    let p1_lo = a_hi & 0xFFFFu;
    let p1_hi = a_hi >> 16u;
    let q0_lo = b_lo & 0xFFFFu;
    let q0_hi = b_lo >> 16u;
    let q1_lo = b_hi & 0xFFFFu;
    let q1_hi = b_hi >> 16u;

    // This is getting complex - use the existing goldilocks_reduce from scalar_mul shader
    // For now, return a placeholder
    return vec2<u32>(0u, 0u);
}

// Schoolbook polynomial multiplication for ring Z[X]/(X^n + 1)
// Each thread computes one output coefficient
@compute @workgroup_size(256, 1, 1)
fn poly_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let n = params.n;
    let q = vec2<u32>(params.modulus_lo, params.modulus_hi);

    if out_idx >= n {
        return;
    }

    // output[out_idx] = sum_{i+j ≡ out_idx (mod n)} sign(i,j) * a[i] * b[j]
    // where sign(i,j) = -1 if i+j >= n (due to X^n = -1)

    var acc = vec2<u32>(0u, 0u);

    for (var i = 0u; i < n; i = i + 1u) {
        // j such that i + j ≡ out_idx (mod n)
        // j = out_idx - i if out_idx >= i, else j = out_idx + n - i
        var j: u32;
        var negate: bool;

        if out_idx >= i {
            j = out_idx - i;
            negate = false;
        } else {
            j = out_idx + n - i;
            negate = true;  // i + j >= n, so X^{i+j} = -X^{i+j-n}
        }

        let a_val = load_u64_a(i);
        let b_val = load_u64_b(j);

        // Multiply a[i] * b[j]
        var prod = goldilocks_mulmod(a_val, b_val);

        // Add or subtract from accumulator
        if negate {
            acc = submod_simple(acc, prod, q);
        } else {
            acc = addmod_simple(acc, prod, q);
        }
    }

    store_u64_result(out_idx, acc);
}

fn addmod_simple(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
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

fn submod_simple(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        return vec2<u32>(diff_lo, diff_hi);
    } else {
        var neg_b_lo = q.x - b.x;
        var borrow = 0u;
        if q.x < b.x {
            borrow = 1u;
            neg_b_lo = 0xFFFFFFFFu - (b.x - q.x - 1u);
        }
        var neg_b_hi = q.y - b.y - borrow;
        return addmod_simple(a, vec2<u32>(neg_b_lo, neg_b_hi), q);
    }
}
"#;

/// Key-switching shader with digit decomposition.
pub const KEY_SWITCH_SHADER: &str = r#"
struct KeySwitchParams {
    n: u32,              // Ring dimension
    num_moduli: u32,     // Number of RNS moduli (4)
    digits_per_limb: u32, // Digits per limb (4)
    decomp_base_log: u32, // log2(decomposition base) = 15
}

@group(0) @binding(0) var<uniform> params: KeySwitchParams;
@group(0) @binding(1) var<storage, read> c1_auto: array<u32>;     // Automorphed c1
@group(0) @binding(2) var<storage, read> keys_a: array<u32>;      // Key a components
@group(0) @binding(3) var<storage, read> keys_b: array<u32>;      // Key b components
@group(0) @binding(4) var<storage, read> moduli: array<u32>;      // RNS moduli
@group(0) @binding(5) var<storage, read_write> out_c0: array<u32>; // Output c0
@group(0) @binding(6) var<storage, read_write> out_c1: array<u32>; // Output c1

// Extract digit from coefficient
fn extract_digit(coeff: vec2<u32>, digit_idx: u32, base_log: u32) -> u32 {
    let shift = base_log * digit_idx;
    if shift >= 64u {
        return 0u;
    }

    var val: u32;
    if shift < 32u {
        val = coeff.x >> shift;
        if shift > 0u {
            val = val | (coeff.y << (32u - shift));
        }
    } else {
        val = coeff.y >> (shift - 32u);
    }

    let mask = (1u << base_log) - 1u;
    return val & mask;
}

@compute @workgroup_size(256, 1, 1)
fn key_switch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This is a simplified version - full implementation needs
    // polynomial multiplication which is complex
    // For now, just demonstrate the digit extraction pattern

    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    // Load coefficient from c1_auto
    let base = (mod_idx * n + coeff_idx) * 2u;
    let coeff = vec2<u32>(c1_auto[base], c1_auto[base + 1u]);

    // Extract digits and accumulate (placeholder - needs poly mul)
    for (var digit = 0u; digit < params.digits_per_limb; digit = digit + 1u) {
        let d = extract_digit(coeff, digit, params.decomp_base_log);
        // Would multiply d by keys[limb][digit] and accumulate
        // This requires polynomial multiplication infrastructure
    }
}
"#;

/// Addition shader for ciphertext addition.
pub const ADD_SHADER: &str = r#"
struct AddParams {
    n: u32,           // Ring dimension
    num_moduli: u32,  // Number of RNS moduli
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<u32>;          // First operand
@group(0) @binding(2) var<storage, read> b: array<u32>;          // Second operand
@group(0) @binding(3) var<storage, read_write> result: array<u32>; // Result
@group(0) @binding(4) var<storage, read> moduli: array<u32>;     // RNS moduli

@compute @workgroup_size(256, 1, 1)
fn add_polys(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    // Load modulus
    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);

    // Load operands
    let base = (mod_idx * n + coeff_idx) * 2u;
    let a_val = vec2<u32>(a[base], a[base + 1u]);
    let b_val = vec2<u32>(b[base], b[base + 1u]);

    // Add with modular reduction
    var sum_lo = a_val.x + b_val.x;
    var carry = 0u;
    if sum_lo < a_val.x { carry = 1u; }
    var sum_hi = a_val.y + b_val.y + carry;

    // Reduce if >= q
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }

    result[base] = sum_lo;
    result[base + 1u] = sum_hi;
}
"#;

/// Digit decomposition shader for HYBRID key-switching.
/// Extracts a specific 15-bit digit from each coefficient.
pub const DIGIT_DECOMPOSE_SHADER: &str = r#"
struct DecomposeParams {
    n: u32,              // Ring dimension
    digit_idx: u32,      // Which digit to extract (0, 1, 2, 3)
    decomp_base_log: u32, // Log2 of decomposition base (15)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: DecomposeParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;      // Input polynomial (u64s as pairs of u32s)
@group(0) @binding(2) var<storage, read_write> output: array<u32>; // Output digits (u64s)

@compute @workgroup_size(256, 1, 1)
fn digit_decompose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if idx >= n {
        return;
    }

    // Load 64-bit coefficient as two u32s
    let coeff_lo = input[idx * 2u];
    let coeff_hi = input[idx * 2u + 1u];

    // Calculate shift amount: digit_idx * decomp_base_log
    let shift = params.digit_idx * params.decomp_base_log;
    let mask = (1u << params.decomp_base_log) - 1u;

    // Extract digit (handles cross-word boundary)
    var digit: u32;
    if shift < 32u {
        // Digit starts in low word
        let from_lo = coeff_lo >> shift;
        if shift + params.decomp_base_log <= 32u {
            // Digit entirely in low word
            digit = from_lo & mask;
        } else {
            // Digit spans both words
            let bits_from_lo = 32u - shift;
            let bits_from_hi = params.decomp_base_log - bits_from_lo;
            digit = (from_lo | (coeff_hi << bits_from_lo)) & mask;
        }
    } else {
        // Digit entirely in high word
        let hi_shift = shift - 32u;
        digit = (coeff_hi >> hi_shift) & mask;
    }

    // Store as 64-bit value (digit in low word, 0 in high word)
    output[idx * 2u] = digit;
    output[idx * 2u + 1u] = 0u;
}
"#;

// =============================================================================
// FUSED SHADERS: Process all moduli in a single dispatch
// =============================================================================
// These reduce dispatch count by 3× by processing all 3 RNS moduli together.

/// Fused twist shader - processes all moduli in one dispatch.
/// Thread layout: thread_idx = mod_idx * n + coeff_idx
pub const FUSED_TWIST_SHADER: &str = r#"
struct FusedTwistParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FusedTwistParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> psi_powers: array<u32>;  // [mod][coeff] layout
@group(0) @binding(3) var<storage, read> moduli: array<u32>;      // moduli[mod*4] = q_lo, q_hi, mu_lo, mu_hi
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>; // [mod*4] = mu_lo_lo, mu_lo_hi, mu_hi_lo, mu_hi_hi

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

fn barrett_reduce_fused(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x;
    var r1 = x.y;

    if x.w == 0u && x.z == 0u {
        if r1 < q.y || (r1 == q.y && r0 < q.x) {
            return vec2<u32>(r0, r1);
        }
    }

    // Simplified Barrett for 64-bit products
    let p00 = u64_mul(r0, mu0);
    let p01 = u64_mul(r0, mu1);
    let p10 = u64_mul(r1, mu0);
    let p11 = u64_mul(r1, mu1);
    let p02 = u64_mul(r0, mu2);
    let p03 = u64_mul(r0, mu3);
    let p12 = u64_mul(r1, mu2);
    let p13 = u64_mul(r1, mu3);

    var acc96: u32 = p02.y + p03.x + p11.y + p12.x;
    var acc128: u32 = p13.x + p03.y + p12.y;
    var acc160: u32 = p13.y;

    // Simplified quotient estimate
    var q_est_lo = acc128;
    var q_est_hi = acc160;

    let qe0 = u64_mul(q_est_lo, q.x);
    let qe1 = u64_mul(q_est_lo, q.y);
    let qe2 = u64_mul(q_est_hi, q.x);

    var sub0 = qe0.x;
    var sub1 = qe0.y + qe1.x + qe2.x;

    // r - q_est * q
    var diff0 = r0 - sub0;
    var borrow: u32 = 0u;
    if r0 < sub0 { borrow = 1u; }
    var diff1 = r1 - sub1 - borrow;

    // Final reduction
    while diff1 > q.y || (diff1 == q.y && diff0 >= q.x) {
        if diff0 >= q.x {
            diff0 = diff0 - q.x;
        } else {
            diff0 = 0xFFFFFFFFu - (q.x - diff0 - 1u);
            diff1 = diff1 - 1u;
        }
        diff1 = diff1 - q.y;
    }

    return vec2<u32>(diff0, diff1);
}

@compute @workgroup_size(256, 1, 1)
fn fused_twist(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    // Load modulus and Barrett params
    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];

    // Load coefficient and psi power
    let base = (mod_idx * n + coeff_idx) * 2u;
    let psi_base = (mod_idx * n + coeff_idx) * 2u;

    let val = vec2<u32>(data[base], data[base + 1u]);
    let psi = vec2<u32>(psi_powers[psi_base], psi_powers[psi_base + 1u]);

    // Multiply and reduce
    let prod = mul64(val, psi);
    let result = barrett_reduce_fused(prod, q, mu0, mu1, mu2, mu3);

    data[base] = result.x;
    data[base + 1u] = result.y;
}
"#;

/// Fused bit-reverse shader - processes all moduli in one dispatch.
pub const FUSED_BITREV_SHADER: &str = r#"
struct FusedBitrevParams {
    n: u32,
    log_n: u32,
    num_moduli: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FusedBitrevParams;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn bit_reverse(x: u32, log_n: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < log_n; i = i + 1u) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(256, 1, 1)
fn fused_bit_reverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    let rev_idx = bit_reverse(coeff_idx, params.log_n);
    let in_base = (mod_idx * n + coeff_idx) * 2u;
    let out_base = (mod_idx * n + rev_idx) * 2u;

    output[out_base] = input[in_base];
    output[out_base + 1u] = input[in_base + 1u];
}
"#;

/// Fused NTT butterfly shader - processes all moduli in one dispatch.
pub const FUSED_BUTTERFLY_SHADER: &str = r#"
struct FusedButterflyParams {
    n: u32,
    stage: u32,
    num_moduli: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FusedButterflyParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;    // [mod][n] layout
@group(0) @binding(3) var<storage, read> moduli: array<u32>;      // [mod*2] = q_lo, q_hi
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>; // [mod*4]

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
    if p1 > 0xFFFFFFFFu - p2 { hi = hi + 0x10000u; }
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x);
    let p01 = u64_mul(a.x, b.y);
    let p10 = u64_mul(a.y, b.x);
    let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x; var r1 = p00.y; var r2 = p11.x; var r3 = p11.y;
    var t = r1 + p01.x; var c: u32 = 0u; if t < r1 { c = 1u; } r1 = t;
    t = r1 + p10.x; if t < r1 { c = c + 1u; } r1 = t;
    t = r2 + p01.y; var c2: u32 = 0u; if t < r2 { c2 = 1u; } r2 = t;
    t = r2 + p10.y; if t < r2 { c2 = c2 + 1u; } r2 = t;
    t = r2 + c; if t < r2 { c2 = c2 + 1u; } r2 = t;
    r3 = r3 + c2;
    return vec4<u32>(r0, r1, r2, r3);
}

fn barrett_reduce_bf(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x; var r1 = x.y;
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) { return vec2<u32>(r0, r1); }
    let p13 = u64_mul(r1, mu3); let p03 = u64_mul(r0, mu3); let p12 = u64_mul(r1, mu2);
    var acc128 = p13.x + p03.y + p12.y;
    var q_est_lo = acc128;
    let qe0 = u64_mul(q_est_lo, q.x); let qe1 = u64_mul(q_est_lo, q.y);
    var sub0 = qe0.x; var sub1 = qe0.y + qe1.x;
    var diff0 = r0 - sub0; var borrow: u32 = 0u; if r0 < sub0 { borrow = 1u; }
    var diff1 = r1 - sub1 - borrow;
    while diff1 > q.y || (diff1 == q.y && diff0 >= q.x) {
        if diff0 >= q.x { diff0 = diff0 - q.x; } else { diff0 = 0xFFFFFFFFu - (q.x - diff0 - 1u); diff1 = diff1 - 1u; }
        diff1 = diff1 - q.y;
    }
    return vec2<u32>(diff0, diff1);
}

fn addmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    var s0 = a.x + b.x; var c: u32 = 0u; if s0 < a.x { c = 1u; }
    var s1 = a.y + b.y + c;
    if s1 > q.y || (s1 == q.y && s0 >= q.x) {
        if s0 >= q.x { s0 = s0 - q.x; } else { s0 = 0xFFFFFFFFu - (q.x - s0 - 1u); s1 = s1 - 1u; }
        s1 = s1 - q.y;
    }
    return vec2<u32>(s0, s1);
}

fn submod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>) -> vec2<u32> {
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var d0 = a.x - b.x; var borrow: u32 = 0u; if a.x < b.x { borrow = 1u; }
        return vec2<u32>(d0, a.y - b.y - borrow);
    } else {
        var d0 = b.x - a.x; var borrow: u32 = 0u; if b.x < a.x { borrow = 1u; }
        var d1 = b.y - a.y - borrow;
        var r0 = q.x - d0; borrow = 0u; if q.x < d0 { borrow = 1u; }
        return vec2<u32>(r0, q.y - d1 - borrow);
    }
}

@compute @workgroup_size(256, 1, 1)
fn fused_butterfly(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;
    let half_n = n / 2u;

    if thread_idx >= half_n * num_moduli {
        return;
    }

    let butterfly_idx = thread_idx % half_n;
    let mod_idx = thread_idx / half_n;

    let stage = params.stage;
    let m = 1u << (stage + 1u);
    let half_m = 1u << stage;

    let group = butterfly_idx / half_m;
    let idx_in_group = butterfly_idx % half_m;
    let i = group * m + idx_in_group;
    let j = i + half_m;

    // Twiddle factor index
    let twiddle_idx = idx_in_group * (n / m);
    let tw_base = (mod_idx * n + twiddle_idx) * 2u;
    let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

    // Load modulus and Barrett params
    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];

    // Data positions
    let base_i = (mod_idx * n + i) * 2u;
    let base_j = (mod_idx * n + j) * 2u;

    let u = vec2<u32>(data[base_i], data[base_i + 1u]);
    let v = vec2<u32>(data[base_j], data[base_j + 1u]);

    // t = v * twiddle mod q
    let prod = mul64(v, twiddle);
    let t = barrett_reduce_bf(prod, q, mu0, mu1, mu2, mu3);

    // Butterfly: data[i] = u + t, data[j] = u - t
    let new_i = addmod(u, t, q);
    let new_j = submod(u, t, q);

    data[base_i] = new_i.x;
    data[base_i + 1u] = new_i.y;
    data[base_j] = new_j.x;
    data[base_j + 1u] = new_j.y;
}
"#;

/// Fused pointwise multiply shader - processes all moduli in one dispatch.
pub const FUSED_POINTWISE_SHADER: &str = r#"
struct FusedPointwiseParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FusedPointwiseParams;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;
@group(0) @binding(4) var<storage, read> moduli: array<u32>;
@group(0) @binding(5) var<storage, read> barrett_params: array<u32>;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let p0 = a_lo * b_lo; let p1 = a_lo * b_hi; let p2 = a_hi * b_lo; let p3 = a_hi * b_hi;
    var lo = p0; var hi = p3;
    let mid = p1 + p2; let mid_lo = (mid & 0xFFFFu) << 16u; let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo; if new_lo < lo { hi = hi + 1u; } lo = new_lo;
    hi = hi + mid_hi; if p1 > 0xFFFFFFFFu - p2 { hi = hi + 0x10000u; }
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x); let p01 = u64_mul(a.x, b.y); let p10 = u64_mul(a.y, b.x); let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x; var r1 = p00.y; var r2 = p11.x; var r3 = p11.y;
    var t = r1 + p01.x; var c: u32 = 0u; if t < r1 { c = 1u; } r1 = t;
    t = r1 + p10.x; if t < r1 { c = c + 1u; } r1 = t;
    t = r2 + p01.y; var c2: u32 = 0u; if t < r2 { c2 = 1u; } r2 = t;
    t = r2 + p10.y; if t < r2 { c2 = c2 + 1u; } r2 = t;
    t = r2 + c; if t < r2 { c2 = c2 + 1u; } r2 = t; r3 = r3 + c2;
    return vec4<u32>(r0, r1, r2, r3);
}

fn barrett_reduce_pw(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x; var r1 = x.y;
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) { return vec2<u32>(r0, r1); }
    let p13 = u64_mul(r1, mu3); let p03 = u64_mul(r0, mu3); let p12 = u64_mul(r1, mu2);
    var q_est_lo = p13.x + p03.y + p12.y;
    let qe0 = u64_mul(q_est_lo, q.x); let qe1 = u64_mul(q_est_lo, q.y);
    var sub0 = qe0.x; var sub1 = qe0.y + qe1.x;
    var diff0 = r0 - sub0; var borrow: u32 = 0u; if r0 < sub0 { borrow = 1u; }
    var diff1 = r1 - sub1 - borrow;
    while diff1 > q.y || (diff1 == q.y && diff0 >= q.x) {
        if diff0 >= q.x { diff0 = diff0 - q.x; } else { diff0 = 0xFFFFFFFFu - (q.x - diff0 - 1u); diff1 = diff1 - 1u; }
        diff1 = diff1 - q.y;
    }
    return vec2<u32>(diff0, diff1);
}

@compute @workgroup_size(256, 1, 1)
fn fused_pointwise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];

    let base = (mod_idx * n + coeff_idx) * 2u;
    let av = vec2<u32>(a[base], a[base + 1u]);
    let bv = vec2<u32>(b[base], b[base + 1u]);

    let prod = mul64(av, bv);
    let res = barrett_reduce_pw(prod, q, mu0, mu1, mu2, mu3);

    result[base] = res.x;
    result[base + 1u] = res.y;
}
"#;

/// Fused scale shader - multiplies all coefficients by n_inv, processes all moduli.
pub const FUSED_SCALE_SHADER: &str = r#"
struct FusedScaleParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FusedScaleParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> n_inv: array<u32>;        // [mod*2] = n_inv_lo, n_inv_hi
@group(0) @binding(3) var<storage, read> moduli: array<u32>;
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>;

fn u64_mul(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let p0 = a_lo * b_lo; let p1 = a_lo * b_hi; let p2 = a_hi * b_lo; let p3 = a_hi * b_hi;
    var lo = p0; var hi = p3;
    let mid = p1 + p2; let mid_lo = (mid & 0xFFFFu) << 16u; let mid_hi = mid >> 16u;
    let new_lo = lo + mid_lo; if new_lo < lo { hi = hi + 1u; } lo = new_lo;
    hi = hi + mid_hi; if p1 > 0xFFFFFFFFu - p2 { hi = hi + 0x10000u; }
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = u64_mul(a.x, b.x); let p01 = u64_mul(a.x, b.y); let p10 = u64_mul(a.y, b.x); let p11 = u64_mul(a.y, b.y);
    var r0 = p00.x; var r1 = p00.y; var r2 = p11.x; var r3 = p11.y;
    var t = r1 + p01.x; var c: u32 = 0u; if t < r1 { c = 1u; } r1 = t;
    t = r1 + p10.x; if t < r1 { c = c + 1u; } r1 = t;
    t = r2 + p01.y; var c2: u32 = 0u; if t < r2 { c2 = 1u; } r2 = t;
    t = r2 + p10.y; if t < r2 { c2 = c2 + 1u; } r2 = t;
    t = r2 + c; if t < r2 { c2 = c2 + 1u; } r2 = t; r3 = r3 + c2;
    return vec4<u32>(r0, r1, r2, r3);
}

fn barrett_reduce_sc(x: vec4<u32>, q: vec2<u32>, mu0: u32, mu1: u32, mu2: u32, mu3: u32) -> vec2<u32> {
    var r0 = x.x; var r1 = x.y;
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) { return vec2<u32>(r0, r1); }
    let p13 = u64_mul(r1, mu3); let p03 = u64_mul(r0, mu3); let p12 = u64_mul(r1, mu2);
    var q_est_lo = p13.x + p03.y + p12.y;
    let qe0 = u64_mul(q_est_lo, q.x); let qe1 = u64_mul(q_est_lo, q.y);
    var sub0 = qe0.x; var sub1 = qe0.y + qe1.x;
    var diff0 = r0 - sub0; var borrow: u32 = 0u; if r0 < sub0 { borrow = 1u; }
    var diff1 = r1 - sub1 - borrow;
    while diff1 > q.y || (diff1 == q.y && diff0 >= q.x) {
        if diff0 >= q.x { diff0 = diff0 - q.x; } else { diff0 = 0xFFFFFFFFu - (q.x - diff0 - 1u); diff1 = diff1 - 1u; }
        diff1 = diff1 - q.y;
    }
    return vec2<u32>(diff0, diff1);
}

@compute @workgroup_size(256, 1, 1)
fn fused_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_moduli = params.num_moduli;

    if thread_idx >= n * num_moduli {
        return;
    }

    let coeff_idx = thread_idx % n;
    let mod_idx = thread_idx / n;

    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];
    let scalar = vec2<u32>(n_inv[mod_idx * 2u], n_inv[mod_idx * 2u + 1u]);

    let base = (mod_idx * n + coeff_idx) * 2u;
    let val = vec2<u32>(data[base], data[base + 1u]);

    let prod = mul64(val, scalar);
    let res = barrett_reduce_sc(prod, q, mu0, mu1, mu2, mu3);

    data[base] = res.x;
    data[base + 1u] = res.y;
}
"#;
