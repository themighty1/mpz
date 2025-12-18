//! AES implementation comparison benchmarks for WASM.
//!
//! Compares the standard `aes` crate vs `aes-wasm` crate performance.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

/// Benchmark the standard `aes` crate encrypt_block (in-place, no allocation).
///
/// # Arguments
/// * `n` - Number of iterations (each encrypts BLOCKS_PER_ITER blocks)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn aes_crate_encrypt(n: u32) -> BenchResult {
    use aes::cipher::{BlockCipherEncrypt, KeyInit};
    use aes::Aes128;

    const BLOCKS_PER_ITER: usize = 10_000;

    let key = [0u8; 16];
    let cipher = Aes128::new_from_slice(&key).unwrap();

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        for i in 0..BLOCKS_PER_ITER {
            let mut block = aes::Block::default();
            block[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block);
        }
    }

    let elapsed_ms = performance.now() - start;
    let total_blocks = n as u64 * BLOCKS_PER_ITER as u64;

    BenchResult {
        elapsed_ms,
        and_gates: total_blocks, // Using block count as work unit
    }
}

/// Benchmark the standard `aes` crate encrypt_block with Vec allocation.
///
/// Allocates a Vec<u8> for output each iteration to match aes-wasm behavior.
///
/// # Arguments
/// * `n` - Number of iterations (each encrypts BLOCKS_PER_ITER blocks)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn aes_crate_encrypt_alloc(n: u32) -> BenchResult {
    use aes::cipher::{BlockCipherEncrypt, KeyInit};
    use aes::Aes128;

    const BLOCKS_PER_ITER: usize = 10_000;

    let key = [0u8; 16];
    let cipher = Aes128::new_from_slice(&key).unwrap();

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        for i in 0..BLOCKS_PER_ITER {
            let mut block = aes::Block::default();
            block[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            cipher.encrypt_block(&mut block);
            // Allocate Vec like aes-wasm does
            let output: Vec<u8> = block.to_vec();
            std::hint::black_box(output);
        }
    }

    let elapsed_ms = performance.now() - start;
    let total_blocks = n as u64 * BLOCKS_PER_ITER as u64;

    BenchResult {
        elapsed_ms,
        and_gates: total_blocks,
    }
}

/// Benchmark the standard `aes` crate encrypt_blocks (batch).
///
/// # Arguments
/// * `n` - Number of iterations (each encrypts BLOCKS_PER_ITER blocks in batches)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn aes_crate_encrypt_batch(n: u32) -> BenchResult {
    use aes::cipher::{BlockCipherEncrypt, KeyInit};
    use aes::Aes128;

    const BLOCKS_PER_ITER: usize = 10_000;
    const BATCH_SIZE: usize = 8;

    let key = [0u8; 16];
    let cipher = Aes128::new_from_slice(&key).unwrap();

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        let mut blocks = [aes::Block::default(); BATCH_SIZE];
        for batch_idx in 0..(BLOCKS_PER_ITER / BATCH_SIZE) {
            for (i, block) in blocks.iter_mut().enumerate() {
                let idx = batch_idx * BATCH_SIZE + i;
                block[0..8].copy_from_slice(&(idx as u64).to_le_bytes());
            }
            cipher.encrypt_blocks(&mut blocks);
            std::hint::black_box(&blocks);
        }
    }

    let elapsed_ms = performance.now() - start;
    let total_blocks = n as u64 * BLOCKS_PER_ITER as u64;

    BenchResult {
        elapsed_ms,
        and_gates: total_blocks,
    }
}

/// Benchmark aes-wasm CTR mode (proxy for AES block speed).
///
/// CTR mode does AES-ECB on counter blocks, so this tests the underlying AES.
/// Encrypts one block at a time to match the aes_crate benchmark.
///
/// # Arguments
/// * `n` - Number of iterations
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn aes_wasm_ctr(n: u32) -> BenchResult {
    use aes_wasm::aes128ctr;

    const BLOCKS_PER_ITER: usize = 10_000;

    let key = aes128ctr::Key::default();

    // Single block buffer
    let plaintext = [0u8; 16];

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        for i in 0..BLOCKS_PER_ITER {
            // Use different IV for each block to force actual AES work
            let mut iv = aes128ctr::IV::default();
            iv[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            let ciphertext = aes128ctr::encrypt(&plaintext, &key, iv);
            std::hint::black_box(ciphertext);
        }
    }

    let elapsed_ms = performance.now() - start;
    let total_blocks = n as u64 * BLOCKS_PER_ITER as u64;

    BenchResult {
        elapsed_ms,
        and_gates: total_blocks,
    }
}

/// Benchmark parallel AES using rayon with the standard `aes` crate.
///
/// # Arguments
/// * `n` - Number of iterations
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn aes_crate_parallel(n: u32) -> BenchResult {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    // Run in worker thread (rayon needs Atomics.wait)
    let _handle = web_spawn::spawn(move || {
        use aes::cipher::{BlockCipherEncrypt, KeyInit};
        use aes::Aes128;
        use rayon::prelude::*;

        const BLOCKS_PER_ITER: usize = 100_000;

        let key = [0u8; 16];
        let cipher = Aes128::new_from_slice(&key).unwrap();

        let global = js_sys::global();
        let performance: web_sys::Performance =
            js_sys::Reflect::get(&global, &"performance".into())
                .expect("performance should exist")
                .unchecked_into();

        let start = performance.now();

        for _ in 0..n {
            let blocks: Vec<u64> = (0..BLOCKS_PER_ITER as u64).collect();
            let _results: Vec<aes::Block> = blocks
                .into_par_iter()
                .map(|i| {
                    let mut block = aes::Block::default();
                    block[0..8].copy_from_slice(&i.to_le_bytes());
                    cipher.encrypt_block(&mut block);
                    block
                })
                .collect();
            std::hint::black_box(&_results);
        }

        let elapsed_ms = performance.now() - start;
        let total_blocks = n as u64 * BLOCKS_PER_ITER as u64;

        *result_clone.lock().unwrap() = Some(BenchResult {
            elapsed_ms,
            and_gates: total_blocks,
        });
    });

    // Poll for result
    JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
        .await
        .unwrap();

    loop {
        if let Some(r) = result.lock().unwrap().take() {
            return r;
        }
        JsFuture::from(js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        }))
        .await
        .unwrap();
    }
}
