//! Fused shaders that process all RNS moduli in a single dispatch.
//! These import from the shared math module.

/// Fused twist shader - multiplies by psi powers, all moduli in one dispatch.
pub const FUSED_TWIST_SHADER: &str = r#"
#import math

struct FusedTwistParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FusedTwistParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> psi_powers: array<u32>;
@group(0) @binding(3) var<storage, read> moduli: array<u32>;
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>;

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

    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];

    let base = (mod_idx * n + coeff_idx) * 2u;
    let psi_base = (mod_idx * n + coeff_idx) * 2u;

    let val = vec2<u32>(data[base], data[base + 1u]);
    let psi = vec2<u32>(psi_powers[psi_base], psi_powers[psi_base + 1u]);

    let prod = math::mul64(val, psi);
    let result = math::barrett_reduce(prod, q, mu0, mu1, mu2, mu3);

    data[base] = result.x;
    data[base + 1u] = result.y;
}
"#;

/// Fused bit-reverse shader - reorders elements for NTT, all moduli in one dispatch.
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

/// Fused NTT butterfly shader - performs butterfly operations, all moduli in one dispatch.
pub const FUSED_BUTTERFLY_SHADER: &str = r#"
#import math

struct FusedButterflyParams {
    n: u32,
    stage: u32,
    num_moduli: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: FusedButterflyParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> twiddles: array<u32>;
@group(0) @binding(3) var<storage, read> moduli: array<u32>;
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>;

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

    let twiddle_idx = idx_in_group * (n / m);
    let tw_base = (mod_idx * n + twiddle_idx) * 2u;
    let twiddle = vec2<u32>(twiddles[tw_base], twiddles[tw_base + 1u]);

    let q = vec2<u32>(moduli[mod_idx * 2u], moduli[mod_idx * 2u + 1u]);
    let mu0 = barrett_params[mod_idx * 4u];
    let mu1 = barrett_params[mod_idx * 4u + 1u];
    let mu2 = barrett_params[mod_idx * 4u + 2u];
    let mu3 = barrett_params[mod_idx * 4u + 3u];

    let base_i = (mod_idx * n + i) * 2u;
    let base_j = (mod_idx * n + j) * 2u;

    let u = vec2<u32>(data[base_i], data[base_i + 1u]);
    let v = vec2<u32>(data[base_j], data[base_j + 1u]);

    let prod = math::mul64(v, twiddle);
    let t = math::barrett_reduce(prod, q, mu0, mu1, mu2, mu3);

    let new_i = math::addmod(u, t, q);
    let new_j = math::submod(u, t, q);

    data[base_i] = new_i.x;
    data[base_i + 1u] = new_i.y;
    data[base_j] = new_j.x;
    data[base_j + 1u] = new_j.y;
}
"#;

/// Fused pointwise multiply shader - element-wise multiplication, all moduli in one dispatch.
pub const FUSED_POINTWISE_SHADER: &str = r#"
#import math

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

    let prod = math::mul64(av, bv);
    let res = math::barrett_reduce(prod, q, mu0, mu1, mu2, mu3);

    result[base] = res.x;
    result[base + 1u] = res.y;
}
"#;

/// Fused scale shader - multiplies by n_inv, all moduli in one dispatch.
pub const FUSED_SCALE_SHADER: &str = r#"
#import math

struct FusedScaleParams {
    n: u32,
    num_moduli: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FusedScaleParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> n_inv: array<u32>;
@group(0) @binding(3) var<storage, read> moduli: array<u32>;
@group(0) @binding(4) var<storage, read> barrett_params: array<u32>;

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

    let prod = math::mul64(val, scalar);
    let res = math::barrett_reduce(prod, q, mu0, mu1, mu2, mu3);

    data[base] = res.x;
    data[base + 1u] = res.y;
}
"#;
