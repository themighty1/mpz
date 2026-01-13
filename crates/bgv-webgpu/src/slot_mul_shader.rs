//! GPU shader for batched slot-wise multiplication.
//!
//! This shader accelerates the expensive `mul_plaintext_slots` operation by:
//! 1. Batching all row encodings (INTT to convert slots → polynomial)
//! 2. Batching all polynomial multiplications (NTT-based)
//! 3. Reusing precomputed twiddle factors across all operations
//!
//! For the JV prover with 180 rows, this reduces:
//! - 180 SlotEncoder creations → 1 GPU buffer with twiddles
//! - 180 sequential INTTs → 1 batched GPU dispatch
//! - 360+ polynomial muls → batched dispatches per modulus

/// Math module import for 64-bit modular arithmetic.
/// Uses the same math functions as existing NTT shaders.
pub const SLOT_MUL_MATH: &str = r#"
#define_import_path slot_mul_math

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
fn barrett_reduce(x: vec4<u32>, q: vec2<u32>, mu: vec4<u32>) -> vec2<u32> {
    let mu0 = mu.x;
    let mu1 = mu.y;
    let mu2 = mu.z;
    let mu3 = mu.w;

    var r0 = x.x;
    var r1 = x.y;

    // Fast path: x already < q
    if x.w == 0u && x.z == 0u && (r1 < q.y || (r1 == q.y && r0 < q.x)) {
        return vec2<u32>(r0, r1);
    }

    // Compute partial products for r * mu
    let p00 = u64_mul(r0, mu0);
    let p01 = u64_mul(r0, mu1);
    let p02 = u64_mul(r0, mu2);
    let p03 = u64_mul(r0, mu3);
    let p10 = u64_mul(r1, mu0);
    let p11 = u64_mul(r1, mu1);
    let p12 = u64_mul(r1, mu2);
    let p13 = u64_mul(r1, mu3);

    // Accumulate bits 64-95
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

    // q_est
    var q_est_lo = acc128;
    var q_est_hi = p13.y + c3;

    // Compute q_est * q
    let qe0 = u64_mul(q_est_lo, q.x);
    let qe1 = u64_mul(q_est_lo, q.y);
    let qe2 = u64_mul(q_est_hi, q.x);

    var p0 = qe0.x;
    var p1 = qe0.y;

    t = p1 + qe1.x; c = 0u; if t < p1 { c = 1u; } p1 = t;
    t = p1 + qe2.x; if t < p1 { c = c + 1u; } p1 = t;

    // Subtract from r
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

    // Final corrections
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

// Modular multiply with Barrett reduction
fn mulmod(a: vec2<u32>, b: vec2<u32>, q: vec2<u32>, mu: vec4<u32>) -> vec2<u32> {
    let prod = mul64(a, b);
    return barrett_reduce(prod, q, mu);
}
"#;

/// Batched INTT shader for slot encoding.
///
/// Encodes multiple slot vectors → polynomial coefficients in parallel.
/// Each workgroup processes one slot vector (8192 elements).
///
/// Input: num_batches × n slot values
/// Output: num_batches × n polynomial coefficients
pub const BATCHED_INTT_ENCODE_SHADER: &str = r#"
struct BatchParams {
    n: u32,              // Ring dimension (8192)
    log_n: u32,          // log2(n) = 13
    num_batches: u32,    // Number of slot vectors to encode
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: BatchParams;
@group(0) @binding(1) var<storage, read> slots: array<u32>;           // Input: batches × n × 2 (u64 as 2×u32)
@group(0) @binding(2) var<storage, read_write> coeffs: array<u32>;    // Output: batches × n × 2
@group(0) @binding(3) var<storage, read> inv_twiddles: array<u32>;    // INTT twiddle factors: n × 2
@group(0) @binding(4) var<storage, read> n_inv: array<u32>;           // n^-1 mod t: 2 words
@group(0) @binding(5) var<uniform> t_val: vec2<u32>;                  // Plaintext modulus t
@group(0) @binding(6) var<uniform> barrett_t: vec4<u32>;              // Barrett parameter for t

// Shared memory for one batch (64KB for n=8192)
var<workgroup> shared_lo: array<u32, 8192>;
var<workgroup> shared_hi: array<u32, 8192>;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(256, 1, 1)
fn batched_intt_encode(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let batch_idx = wg_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if batch_idx >= params.num_batches {
        return;
    }

    let t = t_val;
    let mu = barrett_t;
    let batch_offset = batch_idx * n * 2u;
    let elements_per_thread = n / 256u;  // 32 for n=8192

    // Load slots with bit-reversal
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let rev_idx = bit_reverse(idx, log_n);
        let src = batch_offset + idx * 2u;

        shared_lo[rev_idx] = slots[src];
        shared_hi[rev_idx] = slots[src + 1u];
    }

    workgroupBarrier();

    // INTT butterfly stages (same as forward NTT with inverse twiddles)
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let total_butterflies = n >> 1u;
        let butterflies_per_thread = total_butterflies / 256u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;

            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let i = group * m + idx_in_group;
            let j = i + half_m;

            // Twiddle index
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            // Load butterfly inputs
            let u = vec2<u32>(shared_lo[i], shared_hi[i]);
            let v = vec2<u32>(shared_lo[j], shared_hi[j]);

            // Butterfly with modular arithmetic
            let prod = mul64(v, twiddle);
            let tw = barrett_reduce(prod, t, mu);

            let new_u = addmod(u, tw, t);
            let new_v = submod(u, tw, t);

            shared_lo[i] = new_u.x;
            shared_hi[i] = new_u.y;
            shared_lo[j] = new_v.x;
            shared_hi[j] = new_v.y;
        }

        workgroupBarrier();
    }

    // Scale by n^-1 and store
    let n_inv_val = vec2<u32>(n_inv[0], n_inv[1]);

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;

        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Multiply by n^-1
        let prod = mul64(val, n_inv_val);
        val = barrett_reduce(prod, t, mu);

        let dst = batch_offset + idx * 2u;
        coeffs[dst] = val.x;
        coeffs[dst + 1u] = val.y;
    }
}
"#;

/// Batched polynomial multiplication shader (coefficient domain).
///
/// For RNS-BGV, each ciphertext component is stored per-modulus in NTT form.
/// This shader does pointwise multiplication for one modulus across all batches.
///
/// ct_out[batch] = ct_in × pt_poly[batch]
///
/// The polynomials should already be in NTT domain.
pub const BATCHED_POLY_MUL_SHADER: &str = r#"
struct MulParams {
    n: u32,              // Ring dimension (8192)
    num_batches: u32,    // Number of polynomial pairs to multiply
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: MulParams;
@group(0) @binding(1) var<storage, read> ct_ntt: array<u32>;          // Ciphertext in NTT: n × 2 (single, replicated across batches)
@group(0) @binding(2) var<storage, read> pt_ntt: array<u32>;          // Plaintexts in NTT: batches × n × 2
@group(0) @binding(3) var<storage, read_write> out_ntt: array<u32>;   // Output: batches × n × 2
@group(0) @binding(4) var<uniform> q_val: vec2<u32>;                  // Modulus q
@group(0) @binding(5) var<uniform> barrett_q: vec4<u32>;              // Barrett parameter for q

@compute @workgroup_size(256, 1, 1)
fn batched_poly_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_batches;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let batch_idx = idx / params.n;
    let elem_idx = idx % params.n;

    let q = q_val;
    let mu = barrett_q;

    // Load ciphertext element (same for all batches)
    let ct_base = elem_idx * 2u;
    let ct_val = vec2<u32>(ct_ntt[ct_base], ct_ntt[ct_base + 1u]);

    // Load plaintext element for this batch
    let pt_base = (batch_idx * params.n + elem_idx) * 2u;
    let pt_val = vec2<u32>(pt_ntt[pt_base], pt_ntt[pt_base + 1u]);

    // Multiply: out = ct * pt mod q
    let prod = mul64(ct_val, pt_val);
    let result = barrett_reduce(prod, q, mu);

    // Store result
    let out_base = (batch_idx * params.n + elem_idx) * 2u;
    out_ntt[out_base] = result.x;
    out_ntt[out_base + 1u] = result.y;
}
"#;

/// Batched subtraction shader for VOLE blinding.
///
/// Subtracts blinder values from slot positions after multiplication.
/// out[batch][slot] = ct[batch][slot] - blinder[batch][slot] mod t
pub const BATCHED_SLOT_SUB_SHADER: &str = r#"
struct SubParams {
    n: u32,              // Ring dimension (8192)
    num_batches: u32,    // Number of slot vectors
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SubParams;
@group(0) @binding(1) var<storage, read> slots_in: array<u32>;        // Input slots: batches × n × 2
@group(0) @binding(2) var<storage, read> blinders: array<u32>;        // Blinders: batches × n × 2
@group(0) @binding(3) var<storage, read_write> slots_out: array<u32>; // Output: batches × n × 2
@group(0) @binding(4) var<uniform> t_val: vec2<u32>;                  // Plaintext modulus t

@compute @workgroup_size(256, 1, 1)
fn batched_slot_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_batches;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let t = t_val;
    let base = idx * 2u;

    let a = vec2<u32>(slots_in[base], slots_in[base + 1u]);
    let b = vec2<u32>(blinders[base], blinders[base + 1u]);

    // Modular subtraction
    var result: vec2<u32>;
    if a.y > b.y || (a.y == b.y && a.x >= b.x) {
        var diff_lo = a.x - b.x;
        var borrow = 0u;
        if a.x < b.x { borrow = 1u; }
        var diff_hi = a.y - b.y - borrow;
        result = vec2<u32>(diff_lo, diff_hi);
    } else {
        var diff_lo = b.x - a.x;
        var borrow = 0u;
        if b.x < a.x { borrow = 1u; }
        var diff_hi = b.y - a.y - borrow;

        var res_lo = t.x - diff_lo;
        borrow = 0u;
        if t.x < diff_lo { borrow = 1u; }
        var res_hi = t.y - diff_hi - borrow;
        result = vec2<u32>(res_lo, res_hi);
    }

    slots_out[base] = result.x;
    slots_out[base + 1u] = result.y;
}
"#;

/// Batched 2-way packing shader.
///
/// Packs pairs of ciphertexts using rotation-free 2-way packing:
/// out = ct0 + rotate(ct1, n/2)
///
/// Since rotation by n/2 is just coefficient negation for odd indices,
/// this is very efficient on GPU.
pub const BATCHED_PACK_2WAY_SHADER: &str = r#"
struct PackParams {
    n: u32,              // Ring dimension (8192)
    num_pairs: u32,      // Number of pairs to pack (output ciphertexts)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> ct0_c0: array<u32>;          // First cts c0: pairs × n × 2
@group(0) @binding(2) var<storage, read> ct0_c1: array<u32>;          // First cts c1
@group(0) @binding(3) var<storage, read> ct1_c0: array<u32>;          // Second cts c0
@group(0) @binding(4) var<storage, read> ct1_c1: array<u32>;          // Second cts c1
@group(0) @binding(5) var<storage, read_write> out_c0: array<u32>;    // Output c0: pairs × n × 2
@group(0) @binding(6) var<storage, read_write> out_c1: array<u32>;    // Output c1
@group(0) @binding(7) var<uniform> q_val: vec2<u32>;                  // Modulus q

@compute @workgroup_size(256, 1, 1)
fn batched_pack_2way(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.n * params.num_pairs;
    let idx = global_id.x;

    if idx >= total_elements {
        return;
    }

    let q = q_val;
    let pair_idx = idx / params.n;
    let coeff_idx = idx % params.n;
    let half_n = params.n / 2u;

    let base = (pair_idx * params.n + coeff_idx) * 2u;

    // Load ct0 coefficients
    let a0_c0 = vec2<u32>(ct0_c0[base], ct0_c0[base + 1u]);
    let a0_c1 = vec2<u32>(ct0_c1[base], ct0_c1[base + 1u]);

    // Load ct1 coefficients
    let a1_c0 = vec2<u32>(ct1_c0[base], ct1_c0[base + 1u]);
    let a1_c1 = vec2<u32>(ct1_c1[base], ct1_c1[base + 1u]);

    // Rotation by n/2: for coefficient i, multiply by (-1)^i
    // Even indices: add, Odd indices: subtract
    var b1_c0: vec2<u32>;
    var b1_c1: vec2<u32>;

    if (coeff_idx % 2u) == 0u {
        // Even: no sign change
        b1_c0 = a1_c0;
        b1_c1 = a1_c1;
    } else {
        // Odd: negate (q - val)
        if a1_c0.y == 0u && a1_c0.x == 0u {
            b1_c0 = vec2<u32>(0u, 0u);
        } else {
            var diff_lo = q.x - a1_c0.x;
            var borrow = 0u;
            if q.x < a1_c0.x { borrow = 1u; }
            var diff_hi = q.y - a1_c0.y - borrow;
            b1_c0 = vec2<u32>(diff_lo, diff_hi);
        }

        if a1_c1.y == 0u && a1_c1.x == 0u {
            b1_c1 = vec2<u32>(0u, 0u);
        } else {
            var diff_lo = q.x - a1_c1.x;
            var borrow = 0u;
            if q.x < a1_c1.x { borrow = 1u; }
            var diff_hi = q.y - a1_c1.y - borrow;
            b1_c1 = vec2<u32>(diff_lo, diff_hi);
        }
    }

    // Add: out = ct0 + rotated(ct1)
    // c0
    var sum_lo = a0_c0.x + b1_c0.x;
    var carry = 0u;
    if sum_lo < a0_c0.x { carry = 1u; }
    var sum_hi = a0_c0.y + b1_c0.y + carry;
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }
    out_c0[base] = sum_lo;
    out_c0[base + 1u] = sum_hi;

    // c1
    sum_lo = a0_c1.x + b1_c1.x;
    carry = 0u;
    if sum_lo < a0_c1.x { carry = 1u; }
    sum_hi = a0_c1.y + b1_c1.y + carry;
    if sum_hi > q.y || (sum_hi == q.y && sum_lo >= q.x) {
        if sum_lo >= q.x {
            sum_lo = sum_lo - q.x;
        } else {
            sum_lo = 0xFFFFFFFFu - (q.x - sum_lo - 1u);
            sum_hi = sum_hi - 1u;
        }
        sum_hi = sum_hi - q.y;
    }
    out_c1[base] = sum_lo;
    out_c1[base + 1u] = sum_hi;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_syntax() {
        // Just verify the shaders compile by checking they're valid strings
        assert!(!SLOT_MUL_MATH.is_empty());
        assert!(!BATCHED_INTT_ENCODE_SHADER.is_empty());
        assert!(!BATCHED_POLY_MUL_SHADER.is_empty());
        assert!(!BATCHED_SLOT_SUB_SHADER.is_empty());
        assert!(!BATCHED_PACK_2WAY_SHADER.is_empty());
    }
}
