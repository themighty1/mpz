//! Zig's GF(2^128) multiplication ported to Rust.
//!
//! This is a direct port of Zig's ghash_polyval.zig soft implementation
//! for benchmarking comparison.

/// 256-bit intermediate representation
struct I256 {
    hi: u128,
    lo: u128,
    mid: u128,
}

/// Software carryless multiplication of two 32-bit integers.
/// Uses bit-sliced approach with 4-bit interleaving.
#[inline]
fn clmul_soft32(x: u32, y: u32) -> u64 {
    let a0 = (x & 0x11111111) as u64;
    let a1 = (x & 0x22222222) as u64;
    let a2 = (x & 0x44444444) as u64;
    let a3 = (x & 0x88888888) as u64;
    let b0 = (y & 0x11111111) as u64;
    let b1 = (y & 0x22222222) as u64;
    let b2 = (y & 0x44444444) as u64;
    let b3 = (y & 0x88888888) as u64;

    let c0 = (a0 * b0) ^ (a1 * b3) ^ (a2 * b2) ^ (a3 * b1);
    let c1 = (a0 * b1) ^ (a1 * b0) ^ (a2 * b3) ^ (a3 * b2);
    let c2 = (a0 * b2) ^ (a1 * b1) ^ (a2 * b0) ^ (a3 * b3);
    let c3 = (a0 * b3) ^ (a1 * b2) ^ (a2 * b1) ^ (a3 * b0);

    (c0 & 0x1111111111111111)
        | (c1 & 0x2222222222222222)
        | (c2 & 0x4444444444444444)
        | (c3 & 0x8888888888888888)
}

#[derive(Clone, Copy)]
enum Half {
    Lo,
    Hi,
    HiLo,
}

/// Software carryless multiplication using 64-bit decomposition.
/// Karatsuba-style: 3 multiplies instead of 4.
#[inline]
fn clmul_soft128_64(x: u128, y: u128, half: Half) -> u128 {
    let a: u64 = match half {
        Half::Hi | Half::HiLo => (x >> 64) as u64,
        Half::Lo => x as u64,
    };
    let b: u64 = match half {
        Half::Hi => (y >> 64) as u64,
        Half::Lo | Half::HiLo => y as u64,
    };

    let a0 = a as u32;
    let a1 = (a >> 32) as u32;
    let b0 = b as u32;
    let b1 = (b >> 32) as u32;

    let lo = clmul_soft32(a0, b0);
    let hi = clmul_soft32(a1, b1);
    let mid = clmul_soft32(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;

    let res_lo = lo ^ (mid << 32);
    let res_hi = hi ^ (mid >> 32);

    (res_lo as u128) | ((res_hi as u128) << 64)
}

/// Multiply two 128-bit integers in GF(2^128).
/// Returns 256-bit intermediate result.
#[inline]
fn clmul128(x: u128, y: u128) -> I256 {
    I256 {
        hi: clmul_soft128_64(x, y, Half::Hi),
        lo: clmul_soft128_64(x, y, Half::Lo),
        mid: clmul_soft128_64(x, y, Half::HiLo) ^ clmul_soft128_64(y, x, Half::HiLo),
    }
}

/// Reduce 256-bit polynomial modulo x^128 + x^127 + x^126 + x^121 + 1.
/// This is the POLYVAL polynomial (not GCM's polynomial).
#[inline]
fn reduce(x: I256) -> u128 {
    let hi = x.hi ^ (x.mid >> 64);
    let lo = x.lo ^ (x.mid << 64);

    // p64 = ((1 << 121) | (1 << 126) | (1 << 127)) >> 64
    // = (1 << 57) | (1 << 62) | (1 << 63)
    const P64: u128 = (1u128 << 57) | (1u128 << 62) | (1u128 << 63);

    let a = clmul_soft128_64(lo, P64, Half::Lo);
    let b = ((lo << 64) | (lo >> 64)) ^ a;
    let c = clmul_soft128_64(b, P64, Half::Lo);
    let d = ((b << 64) | (b >> 64)) ^ c;

    d ^ hi
}

/// GF(2^128) multiplication with reduction (Zig's algorithm).
/// Uses POLYVAL polynomial: x^128 + x^127 + x^126 + x^121 + 1
#[inline]
pub fn gf128_mul_zig(x: u128, y: u128) -> u128 {
    reduce(clmul128(x, y))
}

// === WASM Benchmark ===

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

/// Benchmark Zig-style GF(2^128) squaring chain.
///
/// Performs n sequential squarings: a → a² → a⁴ → a⁸ → ...
/// Each iteration is a single GF(2^128) multiply with reduction.
///
/// # Arguments
/// * `n` - Number of squarings
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_zig(n: u32) -> BenchResult {
    // Start with a non-trivial value
    let mut a = 0x123456789abcdef0fedcba9876543210_u128;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        a = gf128_mul_zig(a, a); // Single GF-mul (squaring)
    }

    std::hint::black_box(a);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}
