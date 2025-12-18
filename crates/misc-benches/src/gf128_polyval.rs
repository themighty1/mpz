//! POLYVAL GF(2^128) multiplication - extracted from RustCrypto polyval crate.
//!
//! This is BearSSL's constant-time 64-bit implementation.
//! Source: https://bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul64.c

use core::num::Wrapping;

/// Field element as 2 x u64 (little-endian).
#[derive(Copy, Clone, Default)]
struct FieldElement(u64, u64);

impl FieldElement {
    #[inline]
    fn from_u128(x: u128) -> Self {
        FieldElement(x as u64, (x >> 64) as u64)
    }

    #[inline]
    fn to_u128(self) -> u128 {
        (self.0 as u128) | ((self.1 as u128) << 64)
    }

    /// Carryless multiplication WITHOUT reduction - returns low 128 bits.
    /// For benchmarking to isolate reduction cost.
    #[inline]
    fn mul_no_reduce(self, rhs: Self) -> u128 {
        let h0 = self.0;
        let h1 = self.1;
        let h0r = rev64(h0);
        let h2 = h0 ^ h1;

        let y0 = rhs.0;
        let y1 = rhs.1;
        let y0r = rev64(y0);
        let y2 = y0 ^ y1;

        let z0 = bmul64(y0, h0);
        let z1 = bmul64(y1, h1);

        let mut z2 = bmul64(y2, h2);
        let z0h = bmul64(y0r, h0r);

        z2 ^= z0 ^ z1;
        let z0h = rev64(z0h) >> 1;

        let v0 = z0;
        let v1 = z0h ^ z2;
        // Return low 128 bits (v0, v1) without reduction
        (v0 as u128) | ((v1 as u128) << 64)
    }

    /// Carryless multiplication over GF(2^128) with reduction.
    /// BearSSL constant-time algorithm.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let h0 = self.0;
        let h1 = self.1;
        let h0r = rev64(h0);
        let h1r = rev64(h1);
        let h2 = h0 ^ h1;
        let h2r = h0r ^ h1r;

        let y0 = rhs.0;
        let y1 = rhs.1;
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

        // Reduction
        v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7);
        v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57);
        v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7);
        v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57);

        FieldElement(v2, v3)
    }
}

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
    x = ((x & 0xffff_0000_ffff) << 16) | ((x >> 16) & 0xffff_0000_ffff);
    x.rotate_right(32)
}

/// GF(2^128) multiplication with reduction (BearSSL/polyval algorithm).
#[inline]
pub fn gf128_mul_polyval(x: u128, y: u128) -> u128 {
    let a = FieldElement::from_u128(x);
    let b = FieldElement::from_u128(y);
    a.mul(b).to_u128()
}

/// GF(2^128) multiplication WITHOUT reduction - returns low 128 bits only.
/// For benchmarking to isolate reduction cost.
#[inline]
pub fn gf128_mul_polyval_no_reduce(x: u128, y: u128) -> u128 {
    let a = FieldElement::from_u128(x);
    let b = FieldElement::from_u128(y);
    a.mul_no_reduce(b)
}

// === WASM Benchmark ===

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

/// Benchmark BearSSL/polyval GF(2^128) squaring chain.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_polyval(n: u32) -> BenchResult {
    let mut a = 0x123456789abcdef0fedcba9876543210_u128;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        a = gf128_mul_polyval(a, a);
    }

    std::hint::black_box(a);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}

/// Benchmark GF(2^128) multiplication WITHOUT reduction.
/// Uses low 128 bits of unreduced product as chain value.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn gf128_polyval_no_red(n: u32) -> BenchResult {
    let mut a = 0x123456789abcdef0fedcba9876543210_u128;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    for _ in 0..n {
        a = gf128_mul_polyval_no_reduce(a, a);
    }

    std::hint::black_box(a);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64,
    }
}

/// Benchmark parallel GF(2^128) multiplication with reduction using rayon.
/// Each thread performs independent multiplications.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn gf128_polyval_parallel(n: u32) -> BenchResult {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    const MULS_PER_ITER: usize = 100_000_000;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    // Run in worker thread (rayon needs Atomics.wait)
    let _handle = web_spawn::spawn(move || {
        use rayon::prelude::*;

        let global = js_sys::global();
        let performance: web_sys::Performance =
            js_sys::Reflect::get(&global, &"performance".into())
                .expect("performance should exist")
                .unchecked_into();

        let start = performance.now();

        let num_threads = rayon::current_num_threads();
        web_sys::console::log_1(&format!("[polyval_parallel] num_threads={}", num_threads).into());

        // n iterations, each with num_threads parallel workers
        for iter in 0..n {
            let iter_start = performance.now();
            let _results: Vec<u128> = (0..num_threads)
                .into_par_iter()
                .map(|i| {
                    let mut v = 0x123456789abcdef0fedcba9876543210_u128 ^ (i as u128);
                    for _ in 0..MULS_PER_ITER {
                        v = gf128_mul_polyval(v, v);
                    }
                    v
                })
                .collect();
            std::hint::black_box(&_results);
            let iter_ms = performance.now() - iter_start;
            web_sys::console::log_1(&format!("[polyval_parallel] iter {} took {:.2}ms", iter, iter_ms).into());
        }

        let elapsed_ms = performance.now() - start;
        let total_muls = n as u64 * num_threads as u64 * MULS_PER_ITER as u64;
        web_sys::console::log_1(&format!("[polyval_parallel] total_muls={}", total_muls).into());

        *result_clone.lock().unwrap() = Some(BenchResult {
            elapsed_ms,
            and_gates: total_muls,
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

/// Benchmark parallel GF(2^128) multiplication WITHOUT reduction using rayon.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn gf128_polyval_no_red_parallel(n: u32) -> BenchResult {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    const MULS_PER_ITER: usize = 100_000_000;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    // Run in worker thread (rayon needs Atomics.wait)
    let _handle = web_spawn::spawn(move || {
        use rayon::prelude::*;

        let global = js_sys::global();
        let performance: web_sys::Performance =
            js_sys::Reflect::get(&global, &"performance".into())
                .expect("performance should exist")
                .unchecked_into();

        let start = performance.now();

        let num_threads = rayon::current_num_threads();

        // n iterations, each with num_threads parallel workers
        for _ in 0..n {
            let _results: Vec<u128> = (0..num_threads)
                .into_par_iter()
                .map(|i| {
                    let mut v = 0x123456789abcdef0fedcba9876543210_u128 ^ (i as u128);
                    for _ in 0..MULS_PER_ITER {
                        v = gf128_mul_polyval_no_reduce(v, v);
                    }
                    v
                })
                .collect();
            std::hint::black_box(&_results);
        }

        let elapsed_ms = performance.now() - start;
        let total_muls = n as u64 * num_threads as u64 * MULS_PER_ITER as u64;

        *result_clone.lock().unwrap() = Some(BenchResult {
            elapsed_ms,
            and_gates: total_muls,
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

/// Benchmark trivial parallel loop (wrapping_add) to test WASM threading overhead.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn trivial_parallel(n: u32) -> BenchResult {
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_futures::JsFuture;

    const OPS_PER_ITER: usize = 100_000_000;

    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let _handle = web_spawn::spawn(move || {
        use rayon::prelude::*;

        let global = js_sys::global();
        let performance: web_sys::Performance =
            js_sys::Reflect::get(&global, &"performance".into())
                .expect("performance should exist")
                .unchecked_into();

        let start = performance.now();

        let num_threads = rayon::current_num_threads();

        for _ in 0..n {
            let _results: Vec<u64> = (0..num_threads)
                .into_par_iter()
                .map(|i| {
                    let mut v: u64 = i as u64 | 1;
                    // XOR chain can't be optimized to closed form
                    for _ in 0..OPS_PER_ITER {
                        v ^= v.wrapping_mul(0x9e3779b97f4a7c15);
                    }
                    v
                })
                .collect();
            std::hint::black_box(&_results);
        }

        let elapsed_ms = performance.now() - start;
        let total_ops = n as u64 * num_threads as u64 * OPS_PER_ITER as u64;

        *result_clone.lock().unwrap() = Some(BenchResult {
            elapsed_ms,
            and_gates: total_ops,
        });
    });

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

/// Single-threaded trivial loop for comparison.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn trivial_single(n: u32) -> BenchResult {
    const OPS_PER_ITER: usize = 100_000_000;

    let performance = web_sys::window().unwrap().performance().unwrap();
    let start = performance.now();

    let mut v: u64 = 1;
    for _ in 0..n {
        for _ in 0..OPS_PER_ITER {
            v ^= v.wrapping_mul(0x9e3779b97f4a7c15);
        }
    }

    std::hint::black_box(v);
    let elapsed_ms = performance.now() - start;

    BenchResult {
        elapsed_ms,
        and_gates: n as u64 * OPS_PER_ITER as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyval_vs_zig() {
        use crate::gf128_zig::gf128_mul_zig;

        // Test several values
        let test_values: [(u128, u128); 5] = [
            (0x123456789abcdef0fedcba9876543210, 0x123456789abcdef0fedcba9876543210),
            (0x00000000000000000000000000000001, 0x00000000000000000000000000000002),
            (0xffffffffffffffffffffffffffffffff, 0xffffffffffffffffffffffffffffffff),
            (0x80000000000000000000000000000000, 0x00000000000000000000000000000001),
            (0xdeadbeefcafebabe1234567890abcdef, 0xfedcba0987654321babecafeefbeadde),
        ];

        for (x, y) in test_values {
            let polyval_result = gf128_mul_polyval(x, y);
            let zig_result = gf128_mul_zig(x, y);
            assert_eq!(
                polyval_result, zig_result,
                "Mismatch for x={:#034x}, y={:#034x}\npolyval: {:#034x}\nzig:     {:#034x}",
                x, y, polyval_result, zig_result
            );
        }
    }

    #[test]
    fn test_squaring_chain_consistency() {
        use crate::gf128_zig::gf128_mul_zig;

        let mut polyval_a = 0x123456789abcdef0fedcba9876543210_u128;
        let mut zig_a = polyval_a;

        // Run 100 iterations of squaring
        for i in 0..100 {
            polyval_a = gf128_mul_polyval(polyval_a, polyval_a);
            zig_a = gf128_mul_zig(zig_a, zig_a);
            assert_eq!(
                polyval_a, zig_a,
                "Squaring chain diverged at iteration {}",
                i
            );
        }
    }
}
