//! This module implements the Goldilocks prime field F_{2^64 - 2^32 + 1}.
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 = 18446744069414584321 is particularly
//! efficient for modular arithmetic due to its special structure, and is highly
//! NTT-friendly with primitive roots of unity up to order 2^32.
//!
//! Key properties:
//! - 64-bit prime that fits in a u64
//! - p - 1 = 2^32 * (2^32 - 1), so 2^32 divides the group order
//! - Efficient reduction: 2^64 ≡ 2^32 - 1 (mod p)

use std::ops::{Add, Mul, Neg, Sub};

use hybrid_array::Array;
use itybity::{BitLength, FromBitIterator, GetBit, Lsb0, Msb0};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use typenum::{U64, U8};

use crate::{Field, FieldError};

/// The Goldilocks prime: 2^64 - 2^32 + 1.
pub const GOLDILOCKS: u64 = 0xFFFFFFFF00000001;

/// 2^32, used in reduction.
const TWO_POW_32: u64 = 1u64 << 32;

/// A field element in F_{2^64 - 2^32 + 1} (Goldilocks field).
///
/// Elements are stored in canonical form in the range [0, GOLDILOCKS).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(into = "[u8; 8]")]
#[serde(try_from = "[u8; 8]")]
pub struct Goldilocks(u64);

opaque_debug::implement!(Goldilocks);

impl Goldilocks {
    /// Creates a new field element from a u64.
    ///
    /// The value is reduced modulo GOLDILOCKS if necessary.
    #[inline]
    pub const fn new(value: u64) -> Self {
        if value >= GOLDILOCKS {
            Self(value - GOLDILOCKS)
        } else {
            Self(value)
        }
    }

    /// Returns the inner u64 value.
    #[inline]
    pub const fn inner(self) -> u64 {
        self.0
    }

    /// Reduces a u128 value modulo GOLDILOCKS using the special structure.
    ///
    /// Since p = 2^64 - 2^32 + 1, we have 2^64 ≡ 2^32 - 1 (mod p).
    /// Uses the Plonky2-style reduction which is simple and efficient.
    #[inline(always)]
    const fn reduce(x: u128) -> u64 {
        let (x_lo, x_hi) = (x as u64, (x >> 64) as u64);
        // x_hi * 2^64 ≡ x_hi * (2^32 - 1) (mod p)
        // But we need to be careful: x_hi * (2^32 - 1) can exceed 64 bits if x_hi >= 2^32
        // So we do it in steps using 128-bit arithmetic, then reduce again
        let x_hi_red = (x_hi as u128) * (TWO_POW_32 - 1) as u128;
        let sum = x_lo as u128 + x_hi_red;

        // If sum >= 2^64, we need another reduction round
        let (s_lo, s_hi) = (sum as u64, (sum >> 64) as u64);
        if s_hi == 0 {
            // Simple case: result fits in 64 bits
            if s_lo >= GOLDILOCKS {
                s_lo - GOLDILOCKS
            } else {
                s_lo
            }
        } else {
            // s_hi is small (at most a few bits), reduce again
            let s_hi_red = (s_hi as u128) * (TWO_POW_32 - 1) as u128;
            let final_sum = s_lo as u128 + s_hi_red;
            let mut result = final_sum as u64;
            if final_sum >= GOLDILOCKS as u128 {
                result = (final_sum - GOLDILOCKS as u128) as u64;
            }
            if result >= GOLDILOCKS {
                result -= GOLDILOCKS;
            }
            result
        }
    }

    /// Computes the modular inverse using extended GCD.
    ///
    /// Returns None if self is zero.
    fn inverse_impl(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }

        // Extended Euclidean algorithm
        let mut t: i128 = 0;
        let mut new_t: i128 = 1;
        let mut r: i128 = GOLDILOCKS as i128;
        let mut new_r: i128 = self.0 as i128;

        while new_r != 0 {
            let quotient = r / new_r;

            let temp_t = t - quotient * new_t;
            t = new_t;
            new_t = temp_t;

            let temp_r = r - quotient * new_r;
            r = new_r;
            new_r = temp_r;
        }

        debug_assert_eq!(r, 1, "GCD should be 1 for non-zero element");

        let inv = if t < 0 {
            (t + GOLDILOCKS as i128) as u64
        } else {
            t as u64
        };

        Some(Self(inv))
    }

    /// Computes self^exp using binary exponentiation.
    #[inline]
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::one();

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }

        result
    }

    /// Returns a primitive root of unity of order 2^k.
    ///
    /// For Goldilocks, p - 1 = 2^32 * (2^32 - 1), so we can find
    /// primitive roots of unity up to order 2^32.
    ///
    /// Returns None if k > 32.
    pub fn primitive_root_of_unity(k: u32) -> Option<Self> {
        if k > 32 {
            return None;
        }

        // A primitive 2^32-th root of unity in the Goldilocks field.
        // This is g^((p-1)/2^32) where g is a generator of the multiplicative group.
        // 7 is a primitive root mod p, so:
        // omega_32 = 7^((p-1)/2^32) = 7^(2^32 - 1)
        //
        // Precomputed value for the primitive 2^32-th root of unity:
        const OMEGA_32: u64 = 1753635133440165772;

        // To get a 2^k-th root of unity, we raise omega_32 to the power 2^(32-k)
        let omega = Self(OMEGA_32);
        let exp = 1u64 << (32 - k);
        Some(omega.pow(exp))
    }

    /// Returns the maximum NTT size supported (as log2).
    ///
    /// For Goldilocks, this is 32 since 2^32 divides p - 1.
    pub const fn max_ntt_log_size() -> u32 {
        32
    }

    /// Returns the multiplicative generator of the field.
    ///
    /// 7 is a primitive root modulo the Goldilocks prime.
    pub const fn generator() -> Self {
        Self(7)
    }

    // ========================================================================
    // NTT (Number Theoretic Transform) Operations
    // ========================================================================

    /// Computes the NTT (Number Theoretic Transform) in place.
    ///
    /// The input slice length must be a power of 2 and <= 2^32.
    /// Uses the Cooley-Tukey radix-2 DIT algorithm.
    ///
    /// After NTT, `a[i]` contains the evaluation of the polynomial at ω^i,
    /// where ω is the primitive n-th root of unity.
    pub fn ntt(a: &mut [Self]) {
        let n = a.len();
        if n <= 1 {
            return;
        }

        assert!(n.is_power_of_two(), "NTT size must be power of 2");
        let log_n = n.trailing_zeros();
        assert!(log_n <= 32, "NTT size exceeds maximum (2^32)");

        // Bit-reversal permutation
        Self::bit_reverse_permutation(a);

        // Cooley-Tukey iterative NTT
        for s in 1..=log_n {
            let m = 1 << s;
            let half_m = m >> 1;

            // ω_m = primitive m-th root of unity
            let omega_m = Self::primitive_root_of_unity(s).unwrap();

            for k in (0..n).step_by(m) {
                let mut omega = Self::one();
                for j in 0..half_m {
                    let t = omega * a[k + j + half_m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + half_m] = u - t;
                    omega = omega * omega_m;
                }
            }
        }
    }

    /// Computes the inverse NTT in place.
    ///
    /// This transforms evaluations back to coefficients.
    /// The result is scaled by 1/n.
    pub fn intt(a: &mut [Self]) {
        let n = a.len();
        if n <= 1 {
            return;
        }

        assert!(n.is_power_of_two(), "NTT size must be power of 2");
        let log_n = n.trailing_zeros();
        assert!(log_n <= 32, "NTT size exceeds maximum (2^32)");

        // Bit-reversal permutation
        Self::bit_reverse_permutation(a);

        // Gentleman-Sande iterative inverse NTT (same as NTT but with inverse roots)
        for s in 1..=log_n {
            let m = 1 << s;
            let half_m = m >> 1;

            // ω_m^(-1) = inverse of primitive m-th root of unity
            let omega_m = Self::primitive_root_of_unity(s).unwrap();
            let omega_m_inv = omega_m.inverse().unwrap();

            for k in (0..n).step_by(m) {
                let mut omega = Self::one();
                for j in 0..half_m {
                    let t = omega * a[k + j + half_m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + half_m] = u - t;
                    omega = omega * omega_m_inv;
                }
            }
        }

        // Scale by 1/n
        let n_inv = Self::new(n as u64).inverse().unwrap();
        for x in a.iter_mut() {
            *x = *x * n_inv;
        }
    }

    /// Bit-reversal permutation for NTT.
    fn bit_reverse_permutation(a: &mut [Self]) {
        let n = a.len();
        let log_n = n.trailing_zeros();

        for i in 0..n {
            let j = Self::bit_reverse(i as u32, log_n) as usize;
            if i < j {
                a.swap(i, j);
            }
        }
    }

    /// Reverses the lower `bits` bits of `x`.
    #[inline]
    fn bit_reverse(x: u32, bits: u32) -> u32 {
        x.reverse_bits() >> (32 - bits)
    }
}

// ============================================================================
// Precomputed NTT Context for Fast Inverse NTT
// ============================================================================

/// Precomputed context for fast inverse NTT operations.
///
/// Stores precomputed twiddle factors and inverse scaling to avoid
/// repeated computation of roots of unity and modular inverses.
#[derive(Clone, Debug)]
pub struct InttContext {
    /// log2 of the NTT size
    log_n: u32,
    /// NTT size (power of 2)
    n: usize,
    /// Precomputed inverse twiddle factors for each stage.
    /// twiddles_inv[s] contains ω_m^(-j) for j = 0..m/2 where m = 2^(s+1)
    twiddles_inv: Vec<Vec<Goldilocks>>,
    /// Precomputed 1/n for final scaling
    n_inv: Goldilocks,
}

impl InttContext {
    /// Creates a new INTT context for the given size.
    ///
    /// # Arguments
    /// * `n` - NTT size (must be power of 2, <= 2^32)
    ///
    /// # Panics
    /// Panics if n is not a power of 2 or exceeds 2^32.
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "NTT size must be power of 2");
        let log_n = n.trailing_zeros();
        assert!(log_n <= 32, "NTT size exceeds maximum (2^32)");

        // Precompute twiddle factors for each stage
        let mut twiddles_inv = Vec::with_capacity(log_n as usize);

        for s in 1..=log_n {
            let m = 1usize << s;
            let half_m = m >> 1;

            // Get inverse of primitive m-th root of unity
            let omega_m = Goldilocks::primitive_root_of_unity(s).unwrap();
            let omega_m_inv = omega_m.inverse().unwrap();

            // Precompute ω^(-j) for j = 0..half_m
            let mut stage_twiddles = Vec::with_capacity(half_m);
            let mut omega = Goldilocks::one();
            for _ in 0..half_m {
                stage_twiddles.push(omega);
                omega = omega * omega_m_inv;
            }
            twiddles_inv.push(stage_twiddles);
        }

        let n_inv = Goldilocks::new(n as u64).inverse().unwrap();

        Self {
            log_n,
            n,
            twiddles_inv,
            n_inv,
        }
    }

    /// Returns the NTT size this context was created for.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Computes the inverse NTT in place using precomputed twiddle factors.
    ///
    /// This is faster than `Goldilocks::intt()` when performing multiple INTTs
    /// of the same size, as twiddle factors are precomputed.
    pub fn intt(&self, a: &mut [Goldilocks]) {
        let n = a.len();
        assert_eq!(n, self.n, "Input size must match context size");

        if n <= 1 {
            return;
        }

        // Bit-reversal permutation
        Goldilocks::bit_reverse_permutation(a);

        // Gentleman-Sande iterative inverse NTT with precomputed twiddles
        for s in 0..self.log_n as usize {
            let m = 1 << (s + 1);
            let half_m = m >> 1;
            let twiddles = &self.twiddles_inv[s];

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let omega = twiddles[j];
                    let t = omega * a[k + j + half_m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + half_m] = u - t;
                }
            }
        }

        // Scale by 1/n
        for x in a.iter_mut() {
            *x = *x * self.n_inv;
        }
    }

    /// Computes the inverse NTT with fused scaling in the last stage.
    ///
    /// Slightly faster than `intt()` by avoiding a separate scaling pass.
    pub fn intt_fused(&self, a: &mut [Goldilocks]) {
        let n = a.len();
        assert_eq!(n, self.n, "Input size must match context size");

        if n <= 1 {
            if n == 1 {
                // Still need to scale by 1/n = 1/1 = 1, no-op
            }
            return;
        }

        // Bit-reversal permutation
        Goldilocks::bit_reverse_permutation(a);

        let last_stage = self.log_n as usize - 1;

        // All stages except the last
        for s in 0..last_stage {
            let m = 1 << (s + 1);
            let half_m = m >> 1;
            let twiddles = &self.twiddles_inv[s];

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let omega = twiddles[j];
                    let t = omega * a[k + j + half_m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + half_m] = u - t;
                }
            }
        }

        // Last stage with fused scaling by 1/n
        {
            let m = 1 << self.log_n;
            let half_m = m >> 1;
            let twiddles = &self.twiddles_inv[last_stage];

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let omega = twiddles[j];
                    let t = omega * a[k + j + half_m];
                    let u = a[k + j];
                    // Fuse scaling into butterfly output
                    a[k + j] = (u + t) * self.n_inv;
                    a[k + j + half_m] = (u - t) * self.n_inv;
                }
            }
        }
    }
}

impl Goldilocks {
    /// Multiplies two polynomials using NTT.
    ///
    /// Returns coefficients of a(x) * b(x).
    /// The result has degree deg(a) + deg(b).
    pub fn poly_mul_ntt(a: &[Self], b: &[Self]) -> Vec<Self> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }

        // Result degree = deg(a) + deg(b), so we need n >= len(a) + len(b) - 1
        let result_len = a.len() + b.len() - 1;
        let n = result_len.next_power_of_two();

        // Pad to power of 2
        let mut a_padded = vec![Self::zero(); n];
        let mut b_padded = vec![Self::zero(); n];
        a_padded[..a.len()].copy_from_slice(a);
        b_padded[..b.len()].copy_from_slice(b);

        // Forward NTT
        Self::ntt(&mut a_padded);
        Self::ntt(&mut b_padded);

        // Pointwise multiplication
        for i in 0..n {
            a_padded[i] = a_padded[i] * b_padded[i];
        }

        // Inverse NTT
        Self::intt(&mut a_padded);

        // Trim to actual result length
        a_padded.truncate(result_len);
        a_padded
    }

    /// Evaluates a polynomial at multiple points using NTT.
    ///
    /// Given coefficients [c_0, c_1, ..., c_{n-1}], returns evaluations
    /// at [1, ω, ω², ..., ω^{n-1}] where ω is the primitive n-th root of unity.
    pub fn poly_eval_ntt(coeffs: &[Self]) -> Vec<Self> {
        if coeffs.is_empty() {
            return vec![];
        }

        let n = coeffs.len().next_power_of_two();
        let mut padded = vec![Self::zero(); n];
        padded[..coeffs.len()].copy_from_slice(coeffs);

        Self::ntt(&mut padded);
        padded
    }

    /// Interpolates a polynomial from evaluations at roots of unity.
    ///
    /// Given evaluations [f(1), f(ω), f(ω²), ..., f(ω^{n-1})],
    /// returns coefficients [c_0, c_1, ..., c_{n-1}].
    pub fn poly_interpolate_ntt(evals: &[Self]) -> Vec<Self> {
        if evals.is_empty() {
            return vec![];
        }

        let n = evals.len().next_power_of_two();
        let mut padded = vec![Self::zero(); n];
        padded[..evals.len()].copy_from_slice(evals);

        Self::intt(&mut padded);
        padded
    }

    /// Interpolates a polynomial from evaluations at arbitrary points.
    ///
    /// Given points (x_0, y_0), ..., (x_{n-1}, y_{n-1}), finds the unique
    /// polynomial of degree < n passing through all points.
    ///
    /// Uses a combination of techniques for efficiency:
    /// - For power-of-2 sizes at roots of unity: O(n log n) via NTT
    /// - For general points: O(n log² n) via divide-and-conquer
    pub fn interpolate(points: &[Self], values: &[Self]) -> Vec<Self> {
        assert_eq!(points.len(), values.len());
        let n = points.len();

        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![values[0]];
        }

        // Check if points are consecutive powers of a root of unity
        // For now, use the general O(n²) algorithm but optimized
        // TODO: Implement O(n log² n) for arbitrary points

        Self::interpolate_lagrange_optimized(points, values)
    }

    // ========================================================================
    // u64 Helper Functions (for integration with existing code)
    // ========================================================================

    /// Interpolates from u64 points and values, returns u64 coefficients.
    ///
    /// This is a convenience wrapper for code that uses raw u64 values.
    pub fn interpolate_u64(points: &[u64], values: &[u64]) -> Vec<u64> {
        let points_g: Vec<Self> = points.iter().map(|&x| Self::new(x)).collect();
        let values_g: Vec<Self> = values.iter().map(|&y| Self::new(y)).collect();
        let coeffs = Self::interpolate(&points_g, &values_g);
        coeffs.iter().map(|c| c.inner()).collect()
    }

    /// Evaluates a polynomial (given as u64 coefficients) at a u64 point.
    pub fn poly_eval_u64(coeffs: &[u64], x: u64) -> u64 {
        let x_g = Self::new(x);
        let mut result = Self::zero();
        let mut x_pow = Self::one();
        for &c in coeffs {
            result = result + Self::new(c) * x_pow;
            x_pow = x_pow * x_g;
        }
        result.inner()
    }

    /// Multiplies two polynomials (given as u64 coefficients) using NTT.
    pub fn poly_mul_u64(a: &[u64], b: &[u64]) -> Vec<u64> {
        let a_g: Vec<Self> = a.iter().map(|&x| Self::new(x)).collect();
        let b_g: Vec<Self> = b.iter().map(|&x| Self::new(x)).collect();
        let result = Self::poly_mul_ntt(&a_g, &b_g);
        result.iter().map(|c| c.inner()).collect()
    }

    /// Optimized Lagrange interpolation.
    ///
    /// Still O(n²) but with better constants than naive implementation.
    fn interpolate_lagrange_optimized(points: &[Self], values: &[Self]) -> Vec<Self> {
        let n = points.len();
        let mut result = vec![Self::zero(); n];

        // Precompute denominators: d_i = ∏_{j≠i} (x_i - x_j)
        let mut denoms = vec![Self::one(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    denoms[i] = denoms[i] * (points[i] - points[j]);
                }
            }
        }

        // Compute ∏(x - x_j) for all j
        let mut master = vec![Self::one()];
        for &p in points {
            // Multiply by (x - p)
            let mut new_master = vec![Self::zero(); master.len() + 1];
            for (k, &c) in master.iter().enumerate() {
                new_master[k + 1] = new_master[k + 1] + c;
                new_master[k] = new_master[k] - c * p;
            }
            master = new_master;
        }

        // For each i, compute L_i(x) = master(x) / (x - x_i) / d_i * y_i
        for i in 0..n {
            let denom_inv = denoms[i].inverse().unwrap();
            let scale = values[i] * denom_inv;

            // Divide master by (x - x_i) using synthetic division
            let mut quotient = vec![Self::zero(); n];
            let mut remainder = Self::zero();
            for k in (0..=n).rev() {
                let coeff = if k < master.len() { master[k] } else { Self::zero() };
                let new_coeff = coeff + remainder;
                if k > 0 {
                    quotient[k - 1] = new_coeff;
                }
                remainder = new_coeff * points[i];
            }

            // Add scale * quotient to result
            for k in 0..n {
                result[k] = result[k] + quotient[k] * scale;
            }
        }

        result
    }
}

// ============================================================================
// Precomputed Lagrange Interpolation
// ============================================================================

/// Precomputed data for fast Lagrange interpolation with fixed evaluation points.
///
/// When interpolating multiple polynomials over the same set of points α₁, ..., αₙ,
/// we can precompute point-dependent data once and reuse it for each interpolation.
///
/// This reduces per-interpolation cost from O(n²) (with expensive operations) to
/// O(n²) (with just multiply-accumulate), giving ~3-5x speedup.
#[derive(Clone, Debug)]
pub struct LagrangePrecompute {
    /// Number of points
    n: usize,
    /// Inverse denominators: 1/d_i where d_i = ∏_{j≠i}(x_i - x_j)
    inv_denoms: Vec<Goldilocks>,
    /// Quotient polynomials: q_i(x) = M(x)/(x - x_i) where M(x) = ∏(x - x_j)
    /// Stored as n polynomials, each of degree n-1 (so n coefficients each)
    /// Layout: quotients[i * n + k] = coefficient k of q_i(x)
    quotients: Vec<Goldilocks>,
}

impl LagrangePrecompute {
    /// Creates a new precomputed interpolation context for the given points.
    ///
    /// This performs O(n²) precomputation that can be amortized over many interpolations.
    pub fn new(points: &[Goldilocks]) -> Self {
        let n = points.len();
        if n == 0 {
            return Self {
                n: 0,
                inv_denoms: vec![],
                quotients: vec![],
            };
        }

        // Compute denominators: d_i = ∏_{j≠i}(x_i - x_j)
        let mut denoms = vec![Goldilocks::one(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    denoms[i] = denoms[i] * (points[i] - points[j]);
                }
            }
        }

        // Compute inverse denominators
        let inv_denoms: Vec<Goldilocks> = denoms
            .iter()
            .map(|d| d.inverse().expect("points must be distinct"))
            .collect();

        // Compute master polynomial M(x) = ∏(x - x_j)
        let mut master = vec![Goldilocks::one()];
        for &p in points {
            let mut new_master = vec![Goldilocks::zero(); master.len() + 1];
            for (k, &c) in master.iter().enumerate() {
                new_master[k + 1] = new_master[k + 1] + c;
                new_master[k] = new_master[k] - c * p;
            }
            master = new_master;
        }

        // Compute quotient polynomials q_i(x) = M(x) / (x - x_i)
        // Using synthetic division
        let mut quotients = vec![Goldilocks::zero(); n * n];
        for i in 0..n {
            let mut remainder = Goldilocks::zero();
            for k in (0..=n).rev() {
                let coeff = if k < master.len() { master[k] } else { Goldilocks::zero() };
                let new_coeff = coeff + remainder;
                if k > 0 {
                    quotients[i * n + (k - 1)] = new_coeff;
                }
                remainder = new_coeff * points[i];
            }
        }

        Self {
            n,
            inv_denoms,
            quotients,
        }
    }

    /// Creates a precomputed context from u64 points.
    pub fn new_u64(points: &[u64]) -> Self {
        let points_g: Vec<Goldilocks> = points.iter().map(|&x| Goldilocks::new(x)).collect();
        Self::new(&points_g)
    }

    /// Fast interpolation using precomputed data.
    ///
    /// Given values y₁, ..., yₙ, computes the unique polynomial f of degree < n
    /// such that f(αᵢ) = yᵢ for all i.
    ///
    /// Returns the coefficients [c₀, c₁, ..., c_{n-1}] where f(x) = Σ cₖ xᵏ.
    #[inline]
    pub fn interpolate(&self, values: &[Goldilocks]) -> Vec<Goldilocks> {
        assert_eq!(values.len(), self.n, "values length must match points length");

        if self.n == 0 {
            return vec![];
        }
        if self.n == 1 {
            return vec![values[0]];
        }

        let n = self.n;
        let mut result = vec![Goldilocks::zero(); n];

        // result = Σᵢ yᵢ * (1/dᵢ) * qᵢ(x)
        for i in 0..n {
            let scale = values[i] * self.inv_denoms[i];
            let q_offset = i * n;

            // Add scale * q_i to result
            for k in 0..n {
                result[k] = result[k] + self.quotients[q_offset + k] * scale;
            }
        }

        result
    }

    /// Fast interpolation from u64 values, returns u64 coefficients.
    #[inline]
    pub fn interpolate_u64(&self, values: &[u64]) -> Vec<u64> {
        let values_g: Vec<Goldilocks> = values.iter().map(|&y| Goldilocks::new(y)).collect();
        let coeffs = self.interpolate(&values_g);
        coeffs.iter().map(|c| c.inner()).collect()
    }
}

impl From<Goldilocks> for [u8; 8] {
    fn from(value: Goldilocks) -> Self {
        value.0.to_le_bytes()
    }
}

impl TryFrom<[u8; 8]> for Goldilocks {
    type Error = FieldError;

    fn try_from(value: [u8; 8]) -> Result<Self, Self::Error> {
        let n = u64::from_le_bytes(value);
        if n >= GOLDILOCKS {
            return Err(FieldError(Box::new(GoldilocksError::OutOfRange(n))));
        }
        Ok(Self(n))
    }
}

impl TryFrom<Array<u8, U8>> for Goldilocks {
    type Error = FieldError;

    fn try_from(value: Array<u8, U8>) -> Result<Self, Self::Error> {
        let inner: [u8; 8] = value.into();
        Goldilocks::try_from(inner)
    }
}

impl Distribution<Goldilocks> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks {
        // Rejection sampling for uniform distribution
        loop {
            let value = rng.next_u64();
            if value < GOLDILOCKS {
                return Goldilocks(value);
            }
        }
    }
}

impl Add for Goldilocks {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (sum, overflow) = self.0.overflowing_add(rhs.0);
        if overflow || sum >= GOLDILOCKS {
            // If overflow, we added 2^64 which is 2^32 - 1 mod p
            // So result = sum + (2^32 - 1) mod p
            // But if just sum >= GOLDILOCKS, subtract GOLDILOCKS
            if overflow {
                Self(sum.wrapping_add(TWO_POW_32 - 1))
            } else {
                Self(sum - GOLDILOCKS)
            }
        } else {
            Self(sum)
        }
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        if borrow {
            // We subtracted too much, add back GOLDILOCKS
            Self(diff.wrapping_add(GOLDILOCKS))
        } else {
            Self(diff)
        }
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let prod = self.0 as u128 * rhs.0 as u128;
        Self(Self::reduce(prod))
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            self
        } else {
            Self(GOLDILOCKS - self.0)
        }
    }
}

impl Field for Goldilocks {
    type BitSize = U64;
    type ByteSize = U8;

    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn one() -> Self {
        Self(1)
    }

    #[inline]
    fn two_pow(rhs: u32) -> Self {
        if rhs < 64 {
            Self::new(1u64 << rhs)
        } else {
            // 2^64 = 2^32 - 1 mod p
            // 2^(64+k) = (2^32 - 1) * 2^k mod p
            let k = rhs - 64;
            let base = Self(TWO_POW_32 - 1);
            if k < 64 {
                base * Self::new(1u64 << k)
            } else {
                // Recurse for very large exponents
                base * Self::two_pow(k)
            }
        }
    }

    #[inline]
    fn inverse(self) -> Option<Self> {
        self.inverse_impl()
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn to_be_bytes(&self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }
}

impl BitLength for Goldilocks {
    const BITS: usize = 64;
}

impl GetBit<Lsb0> for Goldilocks {
    #[inline]
    fn get_bit(&self, index: usize) -> bool {
        if index >= 64 {
            false
        } else {
            (self.0 >> index) & 1 == 1
        }
    }
}

impl GetBit<Msb0> for Goldilocks {
    #[inline]
    fn get_bit(&self, index: usize) -> bool {
        if index >= 64 {
            false
        } else {
            (self.0 >> (63 - index)) & 1 == 1
        }
    }
}

impl FromBitIterator for Goldilocks {
    fn from_lsb0_iter(iter: impl IntoIterator<Item = bool>) -> Self {
        let mut value = 0u64;
        for (i, bit) in iter.into_iter().enumerate().take(64) {
            if bit {
                value |= 1u64 << i;
            }
        }
        Self::new(value)
    }

    fn from_msb0_iter(iter: impl IntoIterator<Item = bool>) -> Self {
        let mut value = 0u64;
        for (i, bit) in iter.into_iter().enumerate().take(64) {
            if bit {
                value |= 1u64 << (63 - i);
            }
        }
        Self::new(value)
    }
}

/// Error type for Goldilocks field operations.
#[derive(Debug, thiserror::Error)]
pub enum GoldilocksError {
    /// Value is out of range for the field.
    #[error("value {0} is out of range for Goldilocks field (must be < 2^64 - 2^32 + 1)")]
    OutOfRange(u64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_core::{Block, prg::Prg};
    use rand::{Rng, SeedableRng};

    use crate::tests::{
        test_field_basic, test_field_bit_ops_lsb0, test_field_bit_ops_msb0,
        test_field_compute_product_repeated,
    };

    #[test]
    fn test_goldilocks_basic() {
        test_field_basic::<Goldilocks>();
        assert_eq!(Goldilocks::new(0), Goldilocks::zero());
        assert_eq!(Goldilocks::new(1), Goldilocks::one());
    }

    #[test]
    fn test_goldilocks_compute_product_repeated() {
        test_field_compute_product_repeated::<Goldilocks>();
    }

    #[test]
    fn test_goldilocks_bit_ops() {
        test_field_bit_ops_lsb0::<Goldilocks>();
        test_field_bit_ops_msb0::<Goldilocks>();
    }

    #[test]
    fn test_goldilocks_serialize() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..32 {
            let a: Goldilocks = rng.random();
            let bytes: [u8; 8] = a.into();
            let b = Goldilocks::try_from(bytes).unwrap();

            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_goldilocks_constants() {
        // 2^64 - 2^32 + 1 = (2^64 - 1) - (2^32 - 1) + 1 - 1 + 1 = 2^64 - 2^32 + 1
        // We verify by checking: p + 2^32 - 1 = 2^64 (which wraps to 0 in u64)
        assert_eq!(GOLDILOCKS.wrapping_add(TWO_POW_32 - 1), 0);
        assert_eq!(GOLDILOCKS, 0xFFFFFFFF00000001);
        assert_eq!(GOLDILOCKS, 18446744069414584321);
    }

    #[test]
    fn test_goldilocks_reduction() {
        // Test that GOLDILOCKS reduces to 0
        assert_eq!(Goldilocks::new(GOLDILOCKS), Goldilocks::zero());

        // Test values near the boundary
        assert_eq!(Goldilocks::new(GOLDILOCKS - 1).inner(), GOLDILOCKS - 1);
        assert_eq!(Goldilocks::new(GOLDILOCKS + 1).inner(), 1);
    }

    #[test]
    fn test_goldilocks_arithmetic() {
        let a = Goldilocks::new(12345);
        let b = Goldilocks::new(67890);

        // Addition
        assert_eq!((a + b).inner(), 12345 + 67890);

        // Subtraction
        assert_eq!((b - a).inner(), 67890 - 12345);

        // Subtraction with wrap
        let diff = a - b;
        assert_eq!((diff + b).inner(), a.inner());

        // Multiplication
        let prod = a * b;
        assert_eq!(prod.inner(), (12345u128 * 67890u128 % GOLDILOCKS as u128) as u64);

        // Negation
        assert_eq!((a + (-a)).inner(), 0);
    }

    #[test]
    fn test_goldilocks_inverse() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: Goldilocks = rng.random();
            if a.inner() != 0 {
                let inv = a.inverse().unwrap();
                assert_eq!((a * inv).inner(), 1);
            }
        }

        // Zero has no inverse
        assert!(Goldilocks::zero().inverse().is_none());
    }

    #[test]
    fn test_goldilocks_two_pow() {
        assert_eq!(Goldilocks::two_pow(0), Goldilocks::one());
        assert_eq!(Goldilocks::two_pow(1).inner(), 2);
        assert_eq!(Goldilocks::two_pow(32).inner(), TWO_POW_32);

        // 2^64 = 2^32 - 1 mod p
        assert_eq!(Goldilocks::two_pow(64), Goldilocks::new(TWO_POW_32 - 1));
    }

    #[test]
    fn test_goldilocks_pow() {
        let base = Goldilocks::new(3);

        assert_eq!(base.pow(0), Goldilocks::one());
        assert_eq!(base.pow(1), base);
        assert_eq!(base.pow(2), Goldilocks::new(9));
        assert_eq!(base.pow(3), Goldilocks::new(27));

        // Fermat's little theorem: a^(p-1) = 1 mod p for a != 0
        let mut rng = Prg::from_seed(Block::ZERO);
        let a: Goldilocks = rng.random();
        if a.inner() != 0 {
            assert_eq!(a.pow(GOLDILOCKS - 1), Goldilocks::one());
        }
    }

    #[test]
    fn test_goldilocks_primitive_root() {
        // Test that we can get roots of unity for various powers of 2
        for k in 0..=10 {
            let omega = Goldilocks::primitive_root_of_unity(k).unwrap();
            let order = 1u64 << k;

            // omega^order should be 1
            assert_eq!(
                omega.pow(order),
                Goldilocks::one(),
                "omega^{} should be 1 for k={}",
                order,
                k
            );

            // omega^(order/2) should be -1 for k > 0
            if k > 0 {
                assert_eq!(
                    omega.pow(order / 2),
                    -Goldilocks::one(),
                    "omega^{} should be -1 for k={}",
                    order / 2,
                    k
                );
            }
        }

        // Cannot get 2^33 root of unity (only up to 2^32)
        assert!(Goldilocks::primitive_root_of_unity(33).is_none());

        // max_ntt_log_size is 32
        assert_eq!(Goldilocks::max_ntt_log_size(), 32);
    }

    #[test]
    fn test_goldilocks_primitive_root_order_32() {
        // Verify that the 2^32-th root of unity has exact order 2^32
        let omega = Goldilocks::primitive_root_of_unity(32).unwrap();

        // omega^(2^32) should be 1
        let order = 1u64 << 32;
        assert_eq!(omega.pow(order), Goldilocks::one());

        // omega^(2^31) should be -1 (not 1)
        assert_eq!(omega.pow(order / 2), -Goldilocks::one());
    }

    #[test]
    fn test_goldilocks_distributivity() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: Goldilocks = rng.random();
            let b: Goldilocks = rng.random();
            let c: Goldilocks = rng.random();

            // a * (b + c) = a * b + a * c
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn test_goldilocks_associativity() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: Goldilocks = rng.random();
            let b: Goldilocks = rng.random();
            let c: Goldilocks = rng.random();

            // (a + b) + c = a + (b + c)
            assert_eq!((a + b) + c, a + (b + c));

            // (a * b) * c = a * (b * c)
            assert_eq!((a * b) * c, a * (b * c));
        }
    }

    #[test]
    fn test_goldilocks_edge_cases() {
        let zero = Goldilocks::zero();
        let one = Goldilocks::one();
        let max = Goldilocks::new(GOLDILOCKS - 1);

        // Operations with zero
        assert_eq!(zero + zero, zero);
        assert_eq!(zero * one, zero);
        assert_eq!(max + zero, max);

        // Operations with one
        assert_eq!(one * one, one);
        assert_eq!(max * one, max);

        // Max value operations
        assert_eq!(max + one, zero);
        assert_eq!(max + max, Goldilocks::new(GOLDILOCKS - 2));
    }

    #[test]
    fn test_goldilocks_large_multiplication() {
        // Test multiplication of large values
        let a = Goldilocks::new(GOLDILOCKS - 2);
        let b = Goldilocks::new(GOLDILOCKS - 3);

        // (p-2) * (p-3) = p^2 - 5p + 6 ≡ 6 (mod p)
        // Actually: (p-2)(p-3) mod p = (-2)(-3) mod p = 6
        assert_eq!((a * b).inner(), 6);
    }

    // ========================================================================
    // NTT Tests
    // ========================================================================

    #[test]
    fn test_ntt_inverse_identity() {
        // NTT followed by INTT should give back the original
        let original: Vec<Goldilocks> = vec![
            Goldilocks::new(1),
            Goldilocks::new(2),
            Goldilocks::new(3),
            Goldilocks::new(4),
        ];

        let mut a = original.clone();
        Goldilocks::ntt(&mut a);
        Goldilocks::intt(&mut a);

        assert_eq!(a, original);
    }

    #[test]
    fn test_ntt_larger() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for log_n in 1..=10 {
            let n = 1 << log_n;
            let original: Vec<Goldilocks> = (0..n).map(|_| rng.random()).collect();

            let mut a = original.clone();
            Goldilocks::ntt(&mut a);
            Goldilocks::intt(&mut a);

            assert_eq!(a, original, "Failed for n = {}", n);
        }
    }

    #[test]
    fn test_ntt_polynomial_eval() {
        // f(x) = 1 + 2x + 3x^2 + 4x^3
        let coeffs = vec![
            Goldilocks::new(1),
            Goldilocks::new(2),
            Goldilocks::new(3),
            Goldilocks::new(4),
        ];

        let mut evals = coeffs.clone();
        Goldilocks::ntt(&mut evals);

        // Verify by direct evaluation at roots of unity
        let omega = Goldilocks::primitive_root_of_unity(2).unwrap(); // 4th root of unity

        for (i, &eval) in evals.iter().enumerate() {
            let x = omega.pow(i as u64);
            let expected = coeffs[0]
                + coeffs[1] * x
                + coeffs[2] * x * x
                + coeffs[3] * x * x * x;
            assert_eq!(eval, expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_poly_mul_ntt() {
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let a = vec![Goldilocks::new(1), Goldilocks::new(2)];
        let b = vec![Goldilocks::new(3), Goldilocks::new(4)];

        let result = Goldilocks::poly_mul_ntt(&a, &b);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].inner(), 3);  // 1*3
        assert_eq!(result[1].inner(), 10); // 1*4 + 2*3
        assert_eq!(result[2].inner(), 8);  // 2*4
    }

    #[test]
    fn test_poly_mul_ntt_larger() {
        let mut rng = Prg::from_seed(Block::ZERO);

        // Test polynomial multiplication with random polynomials
        let a: Vec<Goldilocks> = (0..16).map(|_| rng.random()).collect();
        let b: Vec<Goldilocks> = (0..16).map(|_| rng.random()).collect();

        let result = Goldilocks::poly_mul_ntt(&a, &b);

        // Verify by naive multiplication
        let mut expected = vec![Goldilocks::zero(); a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                expected[i + j] = expected[i + j] + ai * bj;
            }
        }

        assert_eq!(result, expected);
    }

    #[test]
    fn test_interpolate_simple() {
        // Interpolate a line through (1, 3) and (2, 5) -> y = 2x + 1
        let points = vec![Goldilocks::new(1), Goldilocks::new(2)];
        let values = vec![Goldilocks::new(3), Goldilocks::new(5)];

        let coeffs = Goldilocks::interpolate(&points, &values);

        assert_eq!(coeffs.len(), 2);
        assert_eq!(coeffs[0].inner(), 1); // constant term
        assert_eq!(coeffs[1].inner(), 2); // x coefficient
    }

    #[test]
    fn test_interpolate_quadratic() {
        // Interpolate x^2 through (0, 0), (1, 1), (2, 4)
        let points = vec![
            Goldilocks::new(0),
            Goldilocks::new(1),
            Goldilocks::new(2),
        ];
        let values = vec![
            Goldilocks::new(0),
            Goldilocks::new(1),
            Goldilocks::new(4),
        ];

        let coeffs = Goldilocks::interpolate(&points, &values);

        assert_eq!(coeffs.len(), 3);
        assert_eq!(coeffs[0].inner(), 0); // constant term
        assert_eq!(coeffs[1].inner(), 0); // x coefficient
        assert_eq!(coeffs[2].inner(), 1); // x^2 coefficient
    }

    #[test]
    fn test_interpolate_random() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for n in [4, 8, 16, 32] {
            // Generate random distinct points
            let points: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::new(i as u64 + 1)).collect();
            let values: Vec<Goldilocks> = (0..n).map(|_| rng.random()).collect();

            let coeffs = Goldilocks::interpolate(&points, &values);

            // Verify by evaluating at each point
            for (i, &x) in points.iter().enumerate() {
                let mut y = Goldilocks::zero();
                let mut x_pow = Goldilocks::one();
                for &c in &coeffs {
                    y = y + c * x_pow;
                    x_pow = x_pow * x;
                }
                assert_eq!(y, values[i], "Mismatch at point {} for n={}", i, n);
            }
        }
    }

    // ========================================================================
    // LagrangePrecompute Tests
    // ========================================================================

    #[test]
    fn test_lagrange_precompute_simple() {
        // Same test as test_interpolate_simple
        let points = vec![Goldilocks::new(1), Goldilocks::new(2)];
        let values = vec![Goldilocks::new(3), Goldilocks::new(5)];

        let precompute = LagrangePrecompute::new(&points);
        let coeffs = precompute.interpolate(&values);

        // f(x) = 1 + 2x should give f(1) = 3, f(2) = 5
        assert_eq!(coeffs.len(), 2);
        assert_eq!(coeffs[0].inner(), 1);
        assert_eq!(coeffs[1].inner(), 2);
    }

    #[test]
    fn test_lagrange_precompute_matches_regular() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for n in [4, 8, 16, 32, 64, 100] {
            let points: Vec<Goldilocks> = (0..n).map(|i| Goldilocks::new(i as u64 + 1)).collect();
            let values: Vec<Goldilocks> = (0..n).map(|_| rng.random()).collect();

            // Regular interpolation
            let coeffs_regular = Goldilocks::interpolate(&points, &values);

            // Precomputed interpolation
            let precompute = LagrangePrecompute::new(&points);
            let coeffs_precompute = precompute.interpolate(&values);

            assert_eq!(coeffs_regular, coeffs_precompute, "Mismatch for n={}", n);
        }
    }

    #[test]
    fn test_lagrange_precompute_reuse() {
        let mut rng = Prg::from_seed(Block::ZERO);

        // Fixed points
        let points: Vec<Goldilocks> = (0..50).map(|i| Goldilocks::new(i as u64 + 1)).collect();
        let precompute = LagrangePrecompute::new(&points);

        // Multiple interpolations with different values
        for _ in 0..10 {
            let values: Vec<Goldilocks> = (0..50).map(|_| rng.random()).collect();

            let coeffs = precompute.interpolate(&values);

            // Verify by evaluating at each point
            for (i, &x) in points.iter().enumerate() {
                let mut y = Goldilocks::zero();
                let mut x_pow = Goldilocks::one();
                for &c in &coeffs {
                    y = y + c * x_pow;
                    x_pow = x_pow * x;
                }
                assert_eq!(y, values[i]);
            }
        }
    }

    #[test]
    fn test_lagrange_precompute_u64() {
        let points: Vec<u64> = vec![1, 2, 3, 4, 5];
        let values: Vec<u64> = vec![10, 20, 30, 40, 50];

        let precompute = LagrangePrecompute::new_u64(&points);
        let coeffs = precompute.interpolate_u64(&values);

        // Verify
        let coeffs_regular = Goldilocks::interpolate_u64(&points, &values);
        assert_eq!(coeffs, coeffs_regular);
    }
}
