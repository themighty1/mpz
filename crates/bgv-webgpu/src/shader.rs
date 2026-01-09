//! WGSL shader code for BGV scalar multiplication.

/// The WGSL shader for slot-wise scalar multiplication with Goldilocks reduction.
///
/// This shader performs slot-wise multiplication: out[batch][slot] = ct[slot] * scalar[batch][slot] mod q
/// where q = 2^64 - 2^32 + 1 (Goldilocks prime).
pub const SCALAR_MUL_SHADER: &str = r#"
// Goldilocks prime: q = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
const Q_LO: u32 = 0x00000001u;
const Q_HI: u32 = 0xFFFFFFFFu;

// Parameters passed as uniforms
struct Params {
    n: u32,              // Ring dimension (slots per ciphertext)
    num_batches: u32,    // Number of output ciphertexts
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// Input ciphertext c0 coefficients (n u64s stored as 2n u32s)
@group(0) @binding(1) var<storage, read> ct_c0: array<u32>;

// Input ciphertext c1 coefficients (n u64s stored as 2n u32s)
@group(0) @binding(2) var<storage, read> ct_c1: array<u32>;

// Scalar values: num_batches × n u64s (stored as 2 × num_batches × n u32s)
@group(0) @binding(3) var<storage, read> scalars: array<u32>;

// Output c0: num_batches × n u64s
@group(0) @binding(4) var<storage, read_write> out_c0: array<u32>;

// Output c1: num_batches × n u64s
@group(0) @binding(5) var<storage, read_write> out_c1: array<u32>;

// Multiply two u32s to get a u64 (as vec2<u32>)
fn u64_from_mul(a: u32, b: u32) -> vec2<u32> {
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
    if new_lo < lo {
        hi = hi + 1u;
    }
    lo = new_lo;
    hi = hi + mid_hi;

    return vec2<u32>(lo, hi);
}

// Add with carry
fn add_with_carry_out(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    var carry_out = 0u;
    if sum < a || (carry_in > 0u && sum == a) {
        carry_out = 1u;
    }
    return vec2<u32>(sum, carry_out);
}

// Multiply two u64s to get a 128-bit result (as four u32s)
fn mul_u64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let a_lo = a.x;
    let a_hi = a.y;
    let b_lo = b.x;
    let b_hi = b.y;

    let p0 = u64_from_mul(a_lo, b_lo);
    let p1 = u64_from_mul(a_lo, b_hi);
    let p2 = u64_from_mul(a_hi, b_lo);
    let p3 = u64_from_mul(a_hi, b_hi);

    var w0 = p0.x;

    var t = add_with_carry_out(p0.y, p1.x, 0u);
    var sum = t.x;
    var carry = t.y;
    t = add_with_carry_out(sum, p2.x, 0u);
    var w1 = t.x;
    carry = carry + t.y;

    t = add_with_carry_out(p1.y, p2.y, 0u);
    sum = t.x;
    var carry2 = t.y;
    t = add_with_carry_out(sum, p3.x, 0u);
    sum = t.x;
    carry2 = carry2 + t.y;
    t = add_with_carry_out(sum, carry, 0u);
    var w2 = t.x;
    carry2 = carry2 + t.y;

    var w3 = p3.y + carry2;

    return vec4<u32>(w0, w1, w2, w3);
}

// Final reduction: ensure result is in [0, q)
fn final_reduce(v: vec2<u32>) -> vec2<u32> {
    var r = v;

    if r.y > Q_HI || (r.y == Q_HI && r.x >= Q_LO) {
        if r.x >= Q_LO {
            r.x = r.x - Q_LO;
        } else {
            r.x = 0xFFFFFFFFu - (Q_LO - r.x - 1u);
            r.y = r.y - 1u;
        }
        r.y = r.y - Q_HI;
    }

    if r.y > Q_HI || (r.y == Q_HI && r.x >= Q_LO) {
        if r.x >= Q_LO {
            r.x = r.x - Q_LO;
        } else {
            r.x = 0xFFFFFFFFu - (Q_LO - r.x - 1u);
            r.y = r.y - 1u;
        }
        r.y = r.y - Q_HI;
    }

    return r;
}

// Goldilocks reduction: reduce 128-bit to 64-bit mod q
fn goldilocks_reduce(w: vec4<u32>) -> vec2<u32> {
    var r0 = w.x;
    var r1 = w.y;
    var r2 = 0u;

    var t = add_with_carry_out(r1, w.z, 0u);
    r1 = t.x;
    r2 = t.y + w.w;

    if r0 >= w.z {
        r0 = r0 - w.z;
    } else {
        r0 = 0xFFFFFFFFu - (w.z - r0 - 1u);
        if r1 > 0u {
            r1 = r1 - 1u;
        } else {
            r1 = 0xFFFFFFFFu;
            if r2 > 0u {
                r2 = r2 - 1u;
            }
        }
    }

    if r1 >= w.w {
        r1 = r1 - w.w;
    } else {
        r1 = 0xFFFFFFFFu - (w.w - r1 - 1u);
        if r2 > 0u {
            r2 = r2 - 1u;
        }
    }

    if r2 > 0u {
        t = add_with_carry_out(r1, r2, 0u);
        r1 = t.x;
        let overflow = t.y;

        if r0 >= r2 {
            r0 = r0 - r2;
        } else {
            r0 = 0xFFFFFFFFu - (r2 - r0 - 1u);
            if r1 > 0u {
                r1 = r1 - 1u;
            }
        }

        if overflow > 0u {
            t = add_with_carry_out(r1, 1u, 0u);
            r1 = t.x;
            if r0 >= 1u {
                r0 = r0 - 1u;
            } else {
                r0 = 0xFFFFFFFFu;
                r1 = r1 - 1u;
            }
        }
    }

    return final_reduce(vec2<u32>(r0, r1));
}

// Main compute shader: slot-wise multiplication
// Thread layout: thread_idx = batch_idx * n + slot_idx
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = params.n;
    let num_batches = params.num_batches;

    if thread_idx >= n * num_batches {
        return;
    }

    let batch_idx = thread_idx / n;
    let slot_idx = thread_idx % n;

    // Load ciphertext coefficient for this slot
    let c0_lo = ct_c0[slot_idx * 2u];
    let c0_hi = ct_c0[slot_idx * 2u + 1u];
    let c0_coeff = vec2<u32>(c0_lo, c0_hi);

    let c1_lo = ct_c1[slot_idx * 2u];
    let c1_hi = ct_c1[slot_idx * 2u + 1u];
    let c1_coeff = vec2<u32>(c1_lo, c1_hi);

    // Load scalar for this batch and slot
    let scalar_idx = batch_idx * n + slot_idx;
    let scalar_lo = scalars[scalar_idx * 2u];
    let scalar_hi = scalars[scalar_idx * 2u + 1u];
    let scalar = vec2<u32>(scalar_lo, scalar_hi);

    // Compute slot-wise product: out = ct * scalar mod q
    let prod0 = goldilocks_reduce(mul_u64(c0_coeff, scalar));
    let prod1 = goldilocks_reduce(mul_u64(c1_coeff, scalar));

    // Store result
    let out_idx = batch_idx * n + slot_idx;
    out_c0[out_idx * 2u] = prod0.x;
    out_c0[out_idx * 2u + 1u] = prod0.y;
    out_c1[out_idx * 2u] = prod1.x;
    out_c1[out_idx * 2u + 1u] = prod1.y;
}
"#;
