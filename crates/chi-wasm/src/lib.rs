//! Minimal WASM module for chi computation.
//! Built WITHOUT atomics so it can run in workers with private memory.

use blake3::Hasher;
use mpz_core::Block;
use wasm_bindgen::prelude::*;

/// GF(2^128) multiplication.
/// Takes two 16-byte blocks and returns their product.
#[wasm_bindgen]
pub fn gfmul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let a_block = Block::try_from(a).expect("a must be 16 bytes");
    let b_block = Block::try_from(b).expect("b must be 16 bytes");
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
    let chi_block = Block::try_from(chi).expect("chi must be 16 bytes");

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
    let mut current = Block::try_from(start).expect("start must be 16 bytes");
    let mut result = Vec::with_capacity(count as usize * 16);

    for _ in 0..count {
        result.extend_from_slice(&current.to_bytes());
        current = current.gfmul(current);
    }

    result
}
