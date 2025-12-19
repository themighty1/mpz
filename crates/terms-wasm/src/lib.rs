//! Minimal WASM module for compute_terms computation.
//! Built WITHOUT atomics so it can run in workers with private memory.
//!
//! This performs the QuickSilver consistency check term computation:
//!   u = x.gfmul(y).gfmul(chi)
//!   v = (a_10 ^ a_11 ^ z).gfmul(chi)
//!   where a_10 = y if x.lsb else 0, a_11 = x if y.lsb else 0

use wasm_bindgen::prelude::*;

// ============================================================================
// POLYVAL soft64 GF(2^128) multiplication - from BearSSL
// This is ~2.5x faster than soft32 on wasm32 because wasm has native i64
// ============================================================================

use core::num::Wrapping;

/// Carryless multiply of two 64-bit integers with 4-bit interleaving.
#[inline]
fn bmul64(x: u64, y: u64) -> u64 {
    let x0 = Wrapping(x & 0x1111_1111_1111_1111);
    let x1 = Wrapping(x & 0x2222_2222_2222_2222);
    let x2 = Wrapping(x & 0x4444_4444_4444_4444);
    let x3 = Wrapping(x & 0x8888_8888_8888_8888);
    let y0 = Wrapping(y & 0x1111_1111_1111_1111);
    let y1 = Wrapping(y & 0x2222_2222_2222_2222);
    let y2 = Wrapping(y & 0x4444_4444_4444_4444);
    let y3 = Wrapping(y & 0x8888_8888_8888_8888);

    let mut z0 = ((x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1)).0;
    let mut z1 = ((x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2)).0;
    let mut z2 = ((x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3)).0;
    let mut z3 = ((x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0)).0;

    z0 &= 0x1111_1111_1111_1111;
    z1 &= 0x2222_2222_2222_2222;
    z2 &= 0x4444_4444_4444_4444;
    z3 &= 0x8888_8888_8888_8888;

    z0 | z1 | z2 | z3
}

/// Bit-reverse a u64 in constant time.
#[inline]
fn rev64(mut x: u64) -> u64 {
    x = ((x & 0x5555_5555_5555_5555) << 1) | ((x >> 1) & 0x5555_5555_5555_5555);
    x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
    x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    x = ((x & 0x00ff_00ff_00ff_00ff) << 8) | ((x >> 8) & 0x00ff_00ff_00ff_00ff);
    x = ((x & 0x0000_ffff_0000_ffff) << 16) | ((x >> 16) & 0x0000_ffff_0000_ffff);
    x.rotate_right(32)
}

/// GF(2^128) multiplication with reduction (BearSSL/polyval soft64 algorithm).
#[inline]
fn gfmul(a: u128, b: u128) -> u128 {
    let h0 = a as u64;
    let h1 = (a >> 64) as u64;
    let h0r = rev64(h0);
    let h1r = rev64(h1);
    let h2 = h0 ^ h1;
    let h2r = h0r ^ h1r;

    let y0 = b as u64;
    let y1 = (b >> 64) as u64;
    let y0r = rev64(y0);
    let y1r = rev64(y1);
    let y2 = y0 ^ y1;
    let y2r = y0r ^ y1r;

    let z0 = bmul64(y0, h0);
    let z1 = bmul64(y1, h1);

    let mut z2 = bmul64(y2, h2);
    let mut z0h = bmul64(y0r, h0r);
    let mut z1h = bmul64(y1r, h1r);
    let mut z2h = bmul64(y2r, h2r);

    z2 ^= z0 ^ z1;
    z2h ^= z0h ^ z1h;
    z0h = rev64(z0h) >> 1;
    z1h = rev64(z1h) >> 1;
    z2h = rev64(z2h) >> 1;

    let v0 = z0;
    let mut v1 = z0h ^ z2;
    let mut v2 = z1 ^ z2h;
    let mut v3 = z1h;

    // Reduction modulo x^128 + x^7 + x^2 + x + 1
    v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7);
    v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57);
    v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7);
    v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57);

    (v2 as u128) | ((v3 as u128) << 64)
}

// ============================================================================
// Terms computation using polyval gfmul
// ============================================================================

/// 128-bit block as u128 for fast operations
#[derive(Copy, Clone, Default)]
struct Block(u128);

impl Block {
    const ZERO: Self = Block(0);

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        Block(u128::from_le_bytes(arr))
    }

    #[inline]
    fn to_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    #[inline]
    fn gfmul(self, other: Self) -> Self {
        Block(gfmul(self.0, other.0))
    }

    #[inline]
    fn lsb(self) -> bool {
        (self.0 & 1) == 1
    }

    #[inline]
    fn xor(self, other: Self) -> Self {
        Block(self.0 ^ other.0)
    }
}

/// Compute terms for a batch of triples.
///
/// Input:
///   triples: flattened triples as bytes (48 bytes per triple: x, y, z each 16 bytes)
///   chis: chi values as bytes (16 bytes each, one per triple)
///
/// Output:
///   32 bytes: accumulated (u, v) where u is first 16 bytes, v is last 16 bytes
///
/// Each triple computes:
///   u_i = x.gfmul(y).gfmul(chi)
///   v_i = (a_10 ^ a_11 ^ z).gfmul(chi)
///   where a_10 = y if x.lsb else 0, a_11 = x if y.lsb else 0
///
/// Final result is XOR of all (u_i, v_i) pairs.
#[wasm_bindgen]
pub fn compute_terms_batch(triples: &[u8], chis: &[u8]) -> Vec<u8> {
    let triple_count = triples.len() / 48;
    let chi_count = chis.len() / 16;

    assert_eq!(triple_count, chi_count, "triples and chis must have same count");

    let mut u_acc = Block::ZERO;
    let mut v_acc = Block::ZERO;

    for i in 0..triple_count {
        let base = i * 48;
        let x = Block::from_bytes(&triples[base..base + 16]);
        let y = Block::from_bytes(&triples[base + 16..base + 32]);
        let z = Block::from_bytes(&triples[base + 32..base + 48]);
        let chi = Block::from_bytes(&chis[i * 16..(i + 1) * 16]);

        // Compute u = x.gfmul(y).gfmul(chi)
        let u = x.gfmul(y).gfmul(chi);

        // Compute v = (a_10 ^ a_11 ^ z).gfmul(chi)
        // a_10 = y if x.lsb else 0
        // a_11 = x if y.lsb else 0
        let a_10 = if x.lsb() { y } else { Block::ZERO };
        let a_11 = if y.lsb() { x } else { Block::ZERO };
        let v = a_10.xor(a_11).xor(z).gfmul(chi);

        // Accumulate via XOR
        u_acc = u_acc.xor(u);
        v_acc = v_acc.xor(v);
    }

    // Return accumulated (u, v) as 32 bytes
    let mut result = Vec::with_capacity(32);
    result.extend_from_slice(&u_acc.to_bytes());
    result.extend_from_slice(&v_acc.to_bytes());
    result
}

/// Compute terms for a single triple (for testing/debugging).
#[wasm_bindgen]
pub fn compute_term_single(x: &[u8], y: &[u8], z: &[u8], chi: &[u8]) -> Vec<u8> {
    let x = Block::from_bytes(x);
    let y = Block::from_bytes(y);
    let z = Block::from_bytes(z);
    let chi = Block::from_bytes(chi);

    // u = x.gfmul(y).gfmul(chi)
    let u = x.gfmul(y).gfmul(chi);

    // v = (a_10 ^ a_11 ^ z).gfmul(chi)
    let a_10 = if x.lsb() { y } else { Block::ZERO };
    let a_11 = if y.lsb() { x } else { Block::ZERO };
    let v = a_10.xor(a_11).xor(z).gfmul(chi);

    let mut result = Vec::with_capacity(32);
    result.extend_from_slice(&u.to_bytes());
    result.extend_from_slice(&v.to_bytes());
    result
}
