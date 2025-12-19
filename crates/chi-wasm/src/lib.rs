//! Minimal WASM module for chi computation.
//! Built WITHOUT atomics so it can run in workers with private memory.

use blake3::Hasher;
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
fn gfmul_u128(a: u128, b: u128) -> u128 {
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
// Block type using polyval gfmul
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
        Block(gfmul_u128(self.0, other.0))
    }
}

// ============================================================================
// WASM exports
// ============================================================================

/// GF(2^128) multiplication.
/// Takes two 16-byte blocks and returns their product.
#[wasm_bindgen]
pub fn gfmul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let a_block = Block::from_bytes(a);
    let b_block = Block::from_bytes(b);
    let result = a_block.gfmul(b_block);
    result.to_bytes().to_vec()
}

/// Compute 16 independent starting points for parallel chi computation.
/// Returns 16 * 16 = 256 bytes (16 blocks).
///
/// Uses the same algorithm as zk-core: bootstrap 16 values via squaring,
/// then hash each to get independent starting points.
#[wasm_bindgen]
pub fn compute_chi_starts(chi: &[u8], segment_size: u32) -> Vec<u8> {
    let chi_block = Block::from_bytes(chi);

    // Bootstrap 16 values via squaring
    let mut bootstrapped = [Block::ZERO; 16];
    let mut current = chi_block;
    for b in &mut bootstrapped {
        *b = current;
        current = current.gfmul(current);
    }

    // Hash each to get independent starting points
    let mut result = Vec::with_capacity(16 * 16);
    for (i, boot) in bootstrapped.iter().enumerate() {
        let mut hasher = Hasher::new();
        hasher.update(&boot.to_bytes());
        hasher.update(&(i as u64).to_le_bytes());
        hasher.update(&(segment_size as u64).to_le_bytes());
        let hash = hasher.finalize();
        result.extend_from_slice(&hash.as_bytes()[..16]);
    }

    result
}

/// Compute chi values sequentially.
/// Returns count * 16 bytes of chi values (chi, chi^2, chi^4, ...).
#[wasm_bindgen]
pub fn compute_chi_segment(start: &[u8], count: u32) -> Vec<u8> {
    let mut current = Block::from_bytes(start);
    let mut result = Vec::with_capacity(count as usize * 16);

    for _ in 0..count {
        result.extend_from_slice(&current.to_bytes());
        current = current.gfmul(current);
    }

    result
}
