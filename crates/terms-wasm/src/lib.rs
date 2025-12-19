//! Minimal WASM module for compute_terms computation.
//! Built WITHOUT atomics so it can run in workers with private memory.
//!
//! This performs the QuickSilver consistency check term computation:
//!   u = x.gfmul(y).gfmul(chi)
//!   v = (a_10 ^ a_11 ^ z).gfmul(chi)
//!   where a_10 = y if x.lsb else 0, a_11 = x if y.lsb else 0

use mpz_core::Block;
use wasm_bindgen::prelude::*;

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

    // Zero-copy reinterpret bytes as Blocks (requires proper alignment)
    // triples layout: [x0, y0, z0, x1, y1, z1, ...] where each is 16 bytes
    let triple_blocks: &[Block] = bytemuck::cast_slice(&triples[..triple_count * 48]);
    let chi_blocks: &[Block] = bytemuck::cast_slice(&chis[..chi_count * 16]);

    let mut u_acc = Block::ZERO;
    let mut v_acc = Block::ZERO;

    for i in 0..triple_count {
        // Zero-copy access to x, y, z
        let x = triple_blocks[i * 3];
        let y = triple_blocks[i * 3 + 1];
        let z = triple_blocks[i * 3 + 2];
        let chi = chi_blocks[i];

        // Compute u = x.gfmul(y).gfmul(chi)
        let u = x.gfmul(y).gfmul(chi);

        // Compute v = (a_10 ^ a_11 ^ z).gfmul(chi)
        // a_10 = y if x.lsb else 0
        // a_11 = x if y.lsb else 0
        let a_10 = if x.lsb() { y } else { Block::ZERO };
        let a_11 = if y.lsb() { x } else { Block::ZERO };
        let v = (a_10 ^ a_11 ^ z).gfmul(chi);

        // Accumulate via XOR
        u_acc ^= u;
        v_acc ^= v;
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
    let x = Block::try_from(x).expect("x must be 16 bytes");
    let y = Block::try_from(y).expect("y must be 16 bytes");
    let z = Block::try_from(z).expect("z must be 16 bytes");
    let chi = Block::try_from(chi).expect("chi must be 16 bytes");

    // u = x.gfmul(y).gfmul(chi)
    let u = x.gfmul(y).gfmul(chi);

    // v = (a_10 ^ a_11 ^ z).gfmul(chi)
    let a_10 = if x.lsb() { y } else { Block::ZERO };
    let a_11 = if y.lsb() { x } else { Block::ZERO };
    let v = (a_10 ^ a_11 ^ z).gfmul(chi);

    let mut result = Vec::with_capacity(32);
    result.extend_from_slice(&u.to_bytes());
    result.extend_from_slice(&v.to_bytes());
    result
}
