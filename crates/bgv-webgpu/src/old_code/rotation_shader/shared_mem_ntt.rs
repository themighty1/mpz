//! Single-workgroup NTT using shared memory.
//! All 13 butterfly stages in one dispatch with workgroup barriers.

/// Forward NTT in shared memory - twist, bitrev, butterflies all in one kernel.
/// Requires 64KB shared memory for n=8192.
///
/// Algorithm: twist → bit-reverse → butterflies(forward twiddles)
pub const SHARED_MEM_NTT_FWD_SHADER: &str = r#"
#import math

struct NttParams {
    n: u32,
    log_n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: NttParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;       // forward twiddles
@group(0) @binding(3) var<storage, read> psi_powers: array<u32>;     // for twist (psi^i)
@group(0) @binding(4) var<uniform> q_val: vec2<u32>;
@group(0) @binding(5) var<uniform> barrett: vec4<u32>;               // mu0, mu1, mu2, mu3

// Shared memory: 8192 elements × 2 u32s = 64KB
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

@compute @workgroup_size(1024, 1, 1)
fn shared_mem_ntt_fwd(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = params.n;
    let log_n = params.log_n;

    let q = q_val;
    let mu0 = barrett.x;
    let mu1 = barrett.y;
    let mu2 = barrett.z;
    let mu3 = barrett.w;

    let elements_per_thread = n / 1024u;  // 8 for n=8192

    // === LOAD: twist + bit-reverse permutation ===
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = idx * 2u;

        var val = vec2<u32>(data[base], data[base + 1u]);

        // Apply twist (multiply by psi^idx)
        let psi_base = idx * 2u;
        let psi = vec2<u32>(psi_powers[psi_base], psi_powers[psi_base + 1u]);
        let prod = math::mul64(val, psi);
        val = math::barrett_reduce_60bit(prod, q, mu0, mu1, mu2, mu3);

        // Store to bit-reversed position in shared memory
        let rev_idx = bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }

    workgroupBarrier();

    // === NTT BUTTERFLY STAGES ===
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let total_butterflies = n >> 1u;  // n/2 = 4096
        let butterflies_per_thread = total_butterflies / 1024u;  // 4

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;

            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let i = group * m + idx_in_group;
            let j = i + half_m;

            // Twiddle index
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

            // Load butterfly inputs
            let u = vec2<u32>(shared_lo[i], shared_hi[i]);
            let v = vec2<u32>(shared_lo[j], shared_hi[j]);

            // Butterfly: u' = u + t*w, v' = u - t*w
            let prod = math::mul64(v, twiddle);
            let t = math::barrett_reduce_60bit(prod, q, mu0, mu1, mu2, mu3);

            let new_u = math::addmod(u, t, q);
            let new_v = math::submod(u, t, q);

            // Store results
            shared_lo[i] = new_u.x;
            shared_hi[i] = new_u.y;
            shared_lo[j] = new_v.x;
            shared_hi[j] = new_v.y;
        }

        workgroupBarrier();
    }

    // === STORE: Write back ===
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = idx * 2u;
        data[base] = shared_lo[idx];
        data[base + 1u] = shared_hi[idx];
    }
}
"#;

/// Inverse NTT in shared memory - bitrev, butterflies, scale, untwist all in one kernel.
/// Requires 64KB shared memory for n=8192.
///
/// Algorithm: bit-reverse → butterflies(inverse twiddles) → scale → untwist
pub const SHARED_MEM_NTT_INV_SHADER: &str = r#"
#import math

struct NttParams {
    n: u32,
    log_n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: NttParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> inv_twiddles: array<u32>;   // inverse twiddles
@group(0) @binding(3) var<storage, read> inv_psi_powers: array<u32>; // for untwist (psi^-i)
@group(0) @binding(4) var<storage, read> n_inv: array<u32>;          // n^(-1) mod q
@group(0) @binding(5) var<uniform> q_val: vec2<u32>;
@group(0) @binding(6) var<uniform> barrett: vec4<u32>;               // mu0, mu1, mu2, mu3

// Shared memory: 8192 elements × 2 u32s = 64KB
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

@compute @workgroup_size(1024, 1, 1)
fn shared_mem_ntt_inv(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = params.n;
    let log_n = params.log_n;

    let q = q_val;
    let mu0 = barrett.x;
    let mu1 = barrett.y;
    let mu2 = barrett.z;
    let mu3 = barrett.w;

    let elements_per_thread = n / 1024u;  // 8 for n=8192

    // === LOAD: bit-reverse permutation (NO twist for inverse) ===
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;
        let base = idx * 2u;

        let val = vec2<u32>(data[base], data[base + 1u]);

        // Store to bit-reversed position in shared memory
        let rev_idx = bit_reverse(idx, log_n);
        shared_lo[rev_idx] = val.x;
        shared_hi[rev_idx] = val.y;
    }

    workgroupBarrier();

    // === NTT BUTTERFLY STAGES (with inverse twiddles) ===
    for (var stage = 0u; stage < log_n; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        let total_butterflies = n >> 1u;  // n/2 = 4096
        let butterflies_per_thread = total_butterflies / 1024u;  // 4

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_idx = tid * butterflies_per_thread + b;

            let group = butterfly_idx / half_m;
            let idx_in_group = butterfly_idx % half_m;
            let i = group * m + idx_in_group;
            let j = i + half_m;

            // Twiddle index (using inverse twiddles)
            let twiddle_idx = idx_in_group * (n / m);
            let tw_base = twiddle_idx * 2u;
            let twiddle = vec2<u32>(inv_twiddles[tw_base], inv_twiddles[tw_base + 1u]);

            // Load butterfly inputs
            let u = vec2<u32>(shared_lo[i], shared_hi[i]);
            let v = vec2<u32>(shared_lo[j], shared_hi[j]);

            // Butterfly: u' = u + t*w, v' = u - t*w
            let prod = math::mul64(v, twiddle);
            let t = math::barrett_reduce_60bit(prod, q, mu0, mu1, mu2, mu3);

            let new_u = math::addmod(u, t, q);
            let new_v = math::submod(u, t, q);

            // Store results
            shared_lo[i] = new_u.x;
            shared_hi[i] = new_u.y;
            shared_lo[j] = new_v.x;
            shared_hi[j] = new_v.y;
        }

        workgroupBarrier();
    }

    // === STORE: scale by n^-1, then untwist ===
    let n_inv_val = vec2<u32>(n_inv[0], n_inv[1]);

    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = tid * elements_per_thread + i;

        var val = vec2<u32>(shared_lo[idx], shared_hi[idx]);

        // Scale by n^-1
        let prod1 = math::mul64(val, n_inv_val);
        val = math::barrett_reduce_60bit(prod1, q, mu0, mu1, mu2, mu3);

        // Untwist (multiply by psi^-idx)
        let psi_base = idx * 2u;
        let inv_psi = vec2<u32>(inv_psi_powers[psi_base], inv_psi_powers[psi_base + 1u]);
        let prod2 = math::mul64(val, inv_psi);
        val = math::barrett_reduce_60bit(prod2, q, mu0, mu1, mu2, mu3);

        let base = idx * 2u;
        data[base] = val.x;
        data[base + 1u] = val.y;
    }
}
"#;

/// Shared memory NTT for smaller GPUs (32KB shared mem).
/// Splits into 2 dispatches: stages 0-9, then stages 10-12.
pub const SHARED_MEM_NTT_PARTIAL_SHADER: &str = r#"
#import math

struct NttPartialParams {
    n: u32,
    log_n: u32,
    start_stage: u32,
    end_stage: u32,    // exclusive
}

@group(0) @binding(0) var<uniform> params: NttPartialParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;
@group(0) @binding(3) var<uniform> q_val: vec2<u32>;
@group(0) @binding(4) var<uniform> barrett: vec4<u32>;

// Smaller shared memory: 4096 elements for partial NTT
var<workgroup> shared_lo: array<u32, 4096>;
var<workgroup> shared_hi: array<u32, 4096>;

@compute @workgroup_size(1024, 1, 1)
fn shared_mem_ntt_partial(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let wg = wg_id.x;
    let n = params.n;
    let start_stage = params.start_stage;
    let end_stage = params.end_stage;

    let q = q_val;
    let mu0 = barrett.x;
    let mu1 = barrett.y;
    let mu2 = barrett.z;
    let mu3 = barrett.w;

    let chunk_size = 4096u;
    let elements_per_thread = chunk_size / 1024u;  // 4
    let chunk_offset = wg * chunk_size;

    // Load chunk into shared memory
    for (var i = 0u; i < elements_per_thread; i++) {
        let local_idx = tid * elements_per_thread + i;
        let global_idx = chunk_offset + local_idx;
        let base = global_idx * 2u;

        shared_lo[local_idx] = data[base];
        shared_hi[local_idx] = data[base + 1u];
    }

    workgroupBarrier();

    // Butterfly stages within this chunk
    for (var stage = start_stage; stage < end_stage; stage++) {
        let m = 1u << (stage + 1u);
        let half_m = 1u << stage;

        // Only process if butterfly fits within chunk
        if (half_m <= chunk_size / 2u) {
            let butterflies_in_chunk = chunk_size >> 1u;
            let butterflies_per_thread = butterflies_in_chunk / 1024u;

            for (var b = 0u; b < butterflies_per_thread; b++) {
                let butterfly_idx = tid * butterflies_per_thread + b;

                let group = butterfly_idx / half_m;
                let idx_in_group = butterfly_idx % half_m;
                let i = group * m + idx_in_group;
                let j = i + half_m;

                if (j < chunk_size) {
                    let twiddle_idx = idx_in_group * (n / m);
                    let tw_base = twiddle_idx * 2u;
                    let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

                    let u = vec2<u32>(shared_lo[i], shared_hi[i]);
                    let v = vec2<u32>(shared_lo[j], shared_hi[j]);

                    let prod = math::mul64(v, twiddle);
                    let t = math::barrett_reduce_60bit(prod, q, mu0, mu1, mu2, mu3);

                    let new_u = math::addmod(u, t, q);
                    let new_v = math::submod(u, t, q);

                    shared_lo[i] = new_u.x;
                    shared_hi[i] = new_u.y;
                    shared_lo[j] = new_v.x;
                    shared_hi[j] = new_v.y;
                }
            }

            workgroupBarrier();
        }
    }

    // Store chunk back
    for (var i = 0u; i < elements_per_thread; i++) {
        let local_idx = tid * elements_per_thread + i;
        let global_idx = chunk_offset + local_idx;
        let base = global_idx * 2u;

        data[base] = shared_lo[local_idx];
        data[base + 1u] = shared_hi[local_idx];
    }
}
"#;
