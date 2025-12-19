//! GF(2^128) multiplication comparison benchmarks for WASM.
//!
//! Compares:
//! - mpz-fields Gf2_128 implementation
//! - RustCrypto ghash crate
//! - aes-wasm GCM (derived, includes AES + GHASH)
//! - aes-wasm CTR (AES only, for subtraction)

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

/// Benchmark mpz-fields Gf2_128 squaring chain.
///
/// Performs n sequential squarings: a → a² → a⁴ → a⁸ → ...
/// Each iteration is a single GF(2^128) multiply.
///
/// # Arguments
/// * `n` - Number of squarings
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_mpz(n: u32) -> BenchResult {
    use mpz_fields::gf2_128::Gf2_128;

    // Start with a non-trivial value
    let mut a = Gf2_128::new(0x123456789abcdef0fedcba9876543210_u128);

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        a = a * a; // Single GF-mul (squaring)
    }

    std::hint::black_box(a);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64, // Using iteration count as work unit
    }
}

/// Benchmark RustCrypto ghash crate.
///
/// Performs n sequential GHASH operations with dependency chain.
/// Each iteration: output becomes next key and block, forcing fresh table computation.
///
/// # Arguments
/// * `n` - Number of operations
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_ghash(n: u32) -> BenchResult {
    use ghash_rc::{GHash, universal_hash::{KeyInit, UniversalHash}};

    // Start with non-trivial values
    let mut key = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                   0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10];
    let mut block = [0x42u8; 16];

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        // Create new hasher with current key (forces table recomputation)
        let mut hasher = GHash::new(&key.into());
        hasher.update(&[block.into()]);
        let output: [u8; 16] = hasher.finalize().into();

        // Output becomes next iteration's key and block
        key = output;
        block = output;
    }

    std::hint::black_box(key);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}

/// Benchmark aes-wasm GCM mode (AES + GHASH).
///
/// Performs n GCM encryptions of a single 16-byte block.
/// Each encryption includes ~2 GF-muls (data block + length block) plus AES operations.
///
/// Use with gf128_aes_wasm_ctr to derive GHASH time:
///   ghash_time ≈ gcm_time - ctr_time
///   per_gfmul ≈ ghash_time / 2
///
/// # Arguments
/// * `n` - Number of 1-block encryptions
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_aes_wasm_gcm(n: u32) -> BenchResult {
    use aes_wasm::aes128gcm;

    let key: aes128gcm::Key = [0u8; 16];
    let plaintext = [0u8; 16]; // Single block
    let aad: [u8; 0] = []; // No additional authenticated data

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for i in 0..n {
        // Use different nonce for each encryption (GCM requires unique nonces)
        let mut nonce: aes128gcm::Nonce = [0u8; 12];
        nonce[0..4].copy_from_slice(&i.to_le_bytes());

        let ciphertext = aes128gcm::encrypt(&plaintext, &aad, &key, nonce);
        std::hint::black_box(ciphertext);
    }

    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}

/// Benchmark aes-wasm CTR mode (AES only, no GHASH).
///
/// Performs n CTR encryptions of a single 16-byte block.
/// Each encryption is ~1 AES block encryption.
///
/// Subtract from gf128_aes_wasm_gcm to isolate GHASH time.
///
/// # Arguments
/// * `n` - Number of 1-block encryptions
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_aes_wasm_ctr(n: u32) -> BenchResult {
    use aes_wasm::aes128ctr;

    let key: aes128ctr::Key = [0u8; 16];
    let plaintext = [0u8; 16]; // Single block

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for i in 0..n {
        // Use different IV for each encryption
        let mut iv: aes128ctr::IV = [0u8; 16];
        iv[0..4].copy_from_slice(&i.to_le_bytes());

        let ciphertext = aes128ctr::encrypt(&plaintext, &key, iv);
        std::hint::black_box(ciphertext);
    }

    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}
