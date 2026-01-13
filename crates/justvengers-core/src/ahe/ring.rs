//! Ring polynomial arithmetic for R_q = Z_q[X]/(X^n + 1).
//!
//! This module provides polynomial operations in the cyclotomic ring used by BGV.
//! The ring is the quotient Z_q[X]/(X^n + 1) where n is a power of 2.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::params::BgvParams;

/// A polynomial in the ring R_q = Z_q[X]/(X^n + 1).
///
/// Coefficients are stored in coefficient form, with index i corresponding
/// to the coefficient of X^i.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RingPoly {
    /// Coefficients in Z_q, length exactly n.
    coeffs: Vec<u64>,
    /// The modulus q.
    q: u64,
    /// Primitive 2n-th root of unity for NTT (optional).
    omega: Option<u64>,
}

/// Barrett reduction for modulus q.
/// For a product a*b where a,b < q, computes (a*b) mod q without division.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct BarrettReducer {
    q: u64,
    q_128: u128,
    /// μ = floor(2^(2*k) / q) where k = 64, stored as 128-bit
    /// For 64-bit q, we use μ = floor(2^128 / q)
    mu_lo: u64,
    mu_hi: u64,
}

impl BarrettReducer {
    /// Creates a new Barrett reducer for modulus q.
    pub fn new(q: u64) -> Self {
        // Compute μ = floor(2^128 / q)
        // This is a 128-bit value, but we can compute it as:
        // 2^128 / q = (2^128 - 1) / q + adjustment
        //
        // For simplicity, we compute this using 128-bit division once
        // during precomputation (which is fast since it's only done once)
        let q_128 = q as u128;

        // μ = floor(2^128 / q)
        // We can't represent 2^128 directly, so use: floor((2^128 - 1) / q) + 1 if exact
        // Or approximate: we know 2^128 = q * floor(2^128/q) + (2^128 mod q)
        // Let's compute floor(2^128 / q) = floor((2^64 * 2^64) / q)
        //
        // Using: 2^128 = (2^64)^2, and q fits in 64 bits
        // floor(2^128 / q) = floor(2^64 * 2^64 / q)
        //                  = floor(2^64 * (2^64 / q + (2^64 mod q)/q))
        //                  = 2^64 * floor(2^64 / q) + floor(2^64 * ((2^64 mod q)/q))
        //
        // Simpler: compute directly using 128-bit arithmetic
        // u128::MAX / q gives us floor((2^128 - 1) / q)
        let mu = u128::MAX / q_128;
        // This is slightly less than floor(2^128 / q), but close enough
        // The difference is at most 1, which we handle with correction steps

        Self {
            q,
            q_128,
            mu_lo: mu as u64,
            mu_hi: (mu >> 64) as u64,
        }
    }

    /// Reduces a 128-bit value modulo q using Barrett reduction.
    #[inline(always)]
    pub fn reduce(&self, a: u128) -> u64 {
        // Barrett reduction: q_hat = floor(a * μ / 2^128)
        // Then r = a - q_hat * q, with corrections if needed

        // For small a (< 2^64), use 64-bit modulo (single div instruction)
        if a < (1u128 << 64) {
            return (a as u64) % self.q;
        }

        // Compute floor(a * μ / 2^128)
        // a * μ is up to 256 bits, we need the top 128 bits (shifted right by 128)
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;

        // μ = mu_hi * 2^64 + mu_lo
        // a * μ = (a_hi * 2^64 + a_lo) * (mu_hi * 2^64 + mu_lo)
        //       = a_hi * mu_hi * 2^128 + (a_hi * mu_lo + a_lo * mu_hi) * 2^64 + a_lo * mu_lo
        //
        // We need floor(a * μ / 2^128) = a_hi * mu_hi + floor((a_hi * mu_lo + a_lo * mu_hi + a_lo * mu_lo / 2^64) / 2^64)

        let t0 = (a_lo as u128) * (self.mu_lo as u128); // 128 bits
        let t1 = (a_lo as u128) * (self.mu_hi as u128); // 128 bits
        let t2 = (a_hi as u128) * (self.mu_lo as u128); // 128 bits
        let t3 = (a_hi as u128) * (self.mu_hi as u128); // 128 bits

        // Sum the middle terms
        let mid = t1 + t2 + (t0 >> 64);
        let q_hat = t3 + (mid >> 64);

        // r = a - q_hat * q
        let r = a.wrapping_sub(q_hat.wrapping_mul(self.q_128));

        // r should be in [0, 3q) typically, correct if needed
        let mut r = r;
        if r >= self.q_128 { r -= self.q_128; }
        if r >= self.q_128 { r -= self.q_128; }
        if r >= self.q_128 { r -= self.q_128; }
        r as u64
    }
}

impl RingPoly {
    /// Creates a new ring polynomial with given coefficients.
    ///
    /// Coefficients are reduced modulo q.
    /// Note: This constructor doesn't include omega, so NTT won't be used.
    /// Prefer using `from_params` when BgvParams is available.
    pub fn new(coeffs: Vec<u64>, q: u64) -> Self {
        let mut poly = Self { coeffs, q, omega: None };
        poly.reduce();
        poly
    }

    /// Creates a new ring polynomial with NTT support.
    pub fn new_with_omega(coeffs: Vec<u64>, q: u64, omega: u64) -> Self {
        let mut poly = Self { coeffs, q, omega: Some(omega) };
        poly.reduce();
        poly
    }

    /// Creates the zero polynomial.
    pub fn zero(params: &BgvParams) -> Self {
        Self {
            coeffs: vec![0; params.n],
            q: params.q,
            omega: if params.omega > 0 { Some(params.omega) } else { None },
        }
    }

    /// Creates a polynomial with all coefficients equal to c.
    pub fn constant(c: u64, params: &BgvParams) -> Self {
        let mut coeffs = vec![0; params.n];
        coeffs[0] = c % params.q;
        Self {
            coeffs,
            q: params.q,
            omega: if params.omega > 0 { Some(params.omega) } else { None },
        }
    }

    /// Creates a polynomial from a slice, padding or truncating to length n.
    pub fn from_slice(slice: &[u64], params: &BgvParams) -> Self {
        let mut coeffs = vec![0; params.n];
        let len = slice.len().min(params.n);
        coeffs[..len].copy_from_slice(&slice[..len]);
        let mut poly = Self {
            coeffs,
            q: params.q,
            omega: if params.omega > 0 { Some(params.omega) } else { None },
        };
        poly.reduce();
        poly
    }

    /// Returns the coefficients.
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Returns the modulus.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// Returns the ring dimension.
    pub fn dimension(&self) -> usize {
        self.coeffs.len()
    }

    /// Reduces all coefficients modulo q.
    fn reduce(&mut self) {
        for c in &mut self.coeffs {
            *c %= self.q;
        }
    }

    /// Adds two polynomials coefficient-wise.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        // For addition, a + b < 2q, so we just need a conditional subtract
        let q = self.q;
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                if sum >= q { sum - q } else { sum }
            })
            .collect();

        Self { coeffs, q, omega: self.omega }
    }

    /// Subtracts two polynomials coefficient-wise.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        // Since a, b are already reduced to [0, q), we have:
        // a - b in [-(q-1), q-1]
        // If a >= b: result is a - b (already in [0, q))
        // If a < b: result is q - (b - a) = q + a - b (in [1, q))
        let q = self.q;
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| {
                if a >= b {
                    a - b
                } else {
                    q - (b - a)
                }
            })
            .collect();

        Self { coeffs, q, omega: self.omega }
    }

    /// Negates all coefficients.
    pub fn neg(&self) -> Self {
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| if c == 0 { 0 } else { self.q - c })
            .collect();

        Self { coeffs, q: self.q, omega: self.omega }
    }

    /// Multiplies by a scalar.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let s = scalar % self.q;
        let reducer = BarrettReducer::new(self.q);
        self.scalar_mul_with_reducer(s, &reducer)
    }

    /// Multiplies by a scalar using a precomputed Barrett reducer.
    ///
    /// This is faster when performing many scalar multiplications with the same modulus,
    /// as the reducer computation (which involves a 128-bit division) is done once.
    #[inline]
    pub fn scalar_mul_with_reducer(&self, scalar: u64, reducer: &BarrettReducer) -> Self {
        let s = scalar % self.q;
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| reducer.reduce((c as u128) * (s as u128)))
            .collect();

        Self { coeffs, q: self.q, omega: self.omega }
    }

    /// Multiplies two polynomials in the ring R_q = Z_q[X]/(X^n + 1).
    ///
    /// Uses NTT (O(n log n)) for large polynomials when omega is available,
    /// otherwise schoolbook multiplication (O(n²)).
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        let n = self.coeffs.len();

        // Use NTT only for large polynomials (n >= 2048) where it's faster
        // For smaller n, schoolbook with Barrett reduction is competitive
        if n >= 2048 {
            if let Some(omega) = self.omega {
                return self.mul_ntt_internal(other, omega);
            }
        }

        // Use schoolbook for smaller polynomials or when NTT unavailable
        self.mul_schoolbook(other)
    }

    /// Schoolbook multiplication (O(n²)).
    fn mul_schoolbook(&self, other: &Self) -> Self {
        let n = self.coeffs.len();
        let q = self.q;
        let reducer = BarrettReducer::new(q);

        // Use u64 result with modular reduction to avoid i128 overflow.
        // For large q and n, accumulating in i128 can overflow (e.g., q ≈ 2^60, n = 4096).
        let mut result = vec![0u64; n];

        // Schoolbook multiplication with immediate reduction
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                let prod = reducer.reduce((a as u128) * (b as u128));
                let idx = i + j;

                if idx < n {
                    // result[idx] = (result[idx] + prod) mod q
                    result[idx] = reducer.reduce((result[idx] as u128) + (prod as u128));
                } else {
                    // X^n ≡ -1, so X^(n+k) ≡ -X^k
                    // result[idx-n] = (result[idx-n] - prod) mod q
                    let target = idx - n;
                    if result[target] >= prod {
                        result[target] -= prod;
                    } else {
                        // result - prod + q to handle underflow
                        result[target] = q - (prod - result[target]);
                    }
                }
            }
        }

        Self { coeffs: result, q, omega: self.omega }
    }

    /// NTT-based multiplication for negacyclic convolution (O(n log n)).
    ///
    /// For R_q = Z_q[X]/(X^n + 1), we use the "twisted" NTT approach:
    /// 1. Pre-multiply by powers of psi (where psi = omega, a primitive 2n-th root)
    /// 2. Standard NTT using omega^2 (a primitive n-th root)
    /// 3. Pointwise multiplication
    /// 4. Inverse NTT
    /// 5. Post-multiply by powers of psi^(-1)
    fn mul_ntt_internal(&self, other: &Self, psi: u64) -> Self {
        let n = self.coeffs.len();
        let q = self.q;
        let reducer = BarrettReducer::new(q);

        // Precompute psi powers and omega (psi^2)
        let omega = reducer.reduce((psi as u128) * (psi as u128)); // omega = psi^2 is n-th root
        let psi_inv = Self::mod_inverse(psi, q);

        // Pre-multiply by psi^j (twist for negacyclic)
        let mut a_twisted: Vec<u64> = Vec::with_capacity(n);
        let mut b_twisted: Vec<u64> = Vec::with_capacity(n);
        let mut psi_j = 1u64;
        for j in 0..n {
            a_twisted.push(reducer.reduce((self.coeffs[j] as u128) * (psi_j as u128)));
            b_twisted.push(reducer.reduce((other.coeffs[j] as u128) * (psi_j as u128)));
            psi_j = reducer.reduce((psi_j as u128) * (psi as u128));
        }

        // Forward NTT using omega (n-th root of unity)
        Self::ntt_forward(&mut a_twisted, omega, q, &reducer);
        Self::ntt_forward(&mut b_twisted, omega, q, &reducer);

        // Pointwise multiplication
        let mut c_ntt: Vec<u64> = a_twisted
            .iter()
            .zip(b_twisted.iter())
            .map(|(&a, &b)| reducer.reduce((a as u128) * (b as u128)))
            .collect();

        // Inverse NTT
        let omega_inv = Self::mod_inverse(omega, q);
        Self::ntt_inverse(&mut c_ntt, omega_inv, q, &reducer);

        // Scale by 1/n and post-multiply by psi^(-j) (untwist)
        let n_inv = Self::mod_inverse(n as u64, q);
        let mut psi_inv_j = 1u64;
        let coeffs: Vec<u64> = c_ntt
            .iter()
            .map(|&c| {
                let scaled = reducer.reduce((c as u128) * (n_inv as u128));
                let result = reducer.reduce((scaled as u128) * (psi_inv_j as u128));
                psi_inv_j = reducer.reduce((psi_inv_j as u128) * (psi_inv as u128));
                result
            })
            .collect();

        Self { coeffs, q, omega: self.omega }
    }

    /// Multiplies two polynomials using NTT (faster for large n).
    ///
    /// Requires that q ≡ 1 (mod 2n) for NTT to work.
    pub fn mul_ntt(&self, other: &Self, omega: u64) -> Self {
        self.mul_ntt_internal(other, omega)
    }

    /// Forward NTT (Cooley-Tukey) with Barrett reduction.
    fn ntt_forward(data: &mut [u64], omega: u64, q: u64, reducer: &BarrettReducer) {
        let n = data.len();
        let log_n = n.trailing_zeros();

        // Bit-reversal permutation
        Self::bit_reverse_permutation(data);

        // Cooley-Tukey butterflies
        for s in 0..log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;

            // omega_m = omega^(n/m)
            let exp = n / m;
            let omega_m = Self::mod_pow_barrett(omega, exp as u64, reducer);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let t = reducer.reduce((w as u128) * (data[k + j + half_m] as u128));
                    let u = data[k + j];

                    data[k + j] = if u + t >= q { u + t - q } else { u + t };
                    data[k + j + half_m] = if u >= t { u - t } else { q + u - t };

                    w = reducer.reduce((w as u128) * (omega_m as u128));
                }
            }
        }
    }

    /// Inverse NTT.
    fn ntt_inverse(data: &mut [u64], omega_inv: u64, q: u64, reducer: &BarrettReducer) {
        Self::ntt_forward(data, omega_inv, q, reducer);
    }

    /// Modular exponentiation using Barrett reduction.
    fn mod_pow_barrett(mut base: u64, mut exp: u64, reducer: &BarrettReducer) -> u64 {
        let mut result = 1u64;

        while exp > 0 {
            if exp & 1 == 1 {
                result = reducer.reduce((result as u128) * (base as u128));
            }
            exp >>= 1;
            base = reducer.reduce((base as u128) * (base as u128));
        }

        result
    }

    /// Bit-reversal permutation.
    fn bit_reverse_permutation(data: &mut [u64]) {
        let n = data.len();
        let log_n = n.trailing_zeros();

        for i in 0..n {
            let j = Self::bit_reverse(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }
    }

    /// Reverses bits of a number.
    fn bit_reverse(mut x: usize, bits: u32) -> usize {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Modular exponentiation.
    pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % modulus as u128) as u64;
            }
            exp >>= 1;
            base = ((base as u128 * base as u128) % modulus as u128) as u64;
        }

        result
    }

    /// Modular inverse using extended GCD.
    pub fn mod_inverse(a: u64, modulus: u64) -> u64 {
        let mut t: i128 = 0;
        let mut new_t: i128 = 1;
        let mut r: i128 = modulus as i128;
        let mut new_r: i128 = a as i128;

        while new_r != 0 {
            let quotient = r / new_r;

            let temp = t - quotient * new_t;
            t = new_t;
            new_t = temp;

            let temp = r - quotient * new_r;
            r = new_r;
            new_r = temp;
        }

        if t < 0 {
            (t + modulus as i128) as u64
        } else {
            t as u64
        }
    }

    /// Finds a primitive 2n-th root of unity modulo q.
    ///
    /// Requires q ≡ 1 (mod 2n).
    pub fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
        // q - 1 must be divisible by 2n
        let order = 2 * n as u64;
        if (q - 1) % order != 0 {
            return None;
        }

        // Find a generator of Z_q^* and raise to power (q-1)/(2n)
        let exp = (q - 1) / order;

        // Try small primes as potential generators
        for g in 2..100 {
            let omega = Self::mod_pow(g, exp, q);

            // Verify it's a primitive 2n-th root
            // omega^n should be -1 (q-1), omega^(2n) should be 1
            let omega_n = Self::mod_pow(omega, n as u64, q);
            if omega_n == q - 1 {
                return Some(omega);
            }
        }

        None
    }
}

// Trait implementations for ergonomic usage

impl Add for RingPoly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        RingPoly::add(&self, &rhs)
    }
}

impl<'a> Add<&'a RingPoly> for RingPoly {
    type Output = Self;

    fn add(self, rhs: &'a RingPoly) -> Self::Output {
        RingPoly::add(&self, rhs)
    }
}

impl AddAssign for RingPoly {
    fn add_assign(&mut self, rhs: Self) {
        *self = RingPoly::add(self, &rhs);
    }
}

impl Sub for RingPoly {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        RingPoly::sub(&self, &rhs)
    }
}

impl<'a> Sub<&'a RingPoly> for RingPoly {
    type Output = Self;

    fn sub(self, rhs: &'a RingPoly) -> Self::Output {
        RingPoly::sub(&self, rhs)
    }
}

impl SubAssign for RingPoly {
    fn sub_assign(&mut self, rhs: Self) {
        *self = RingPoly::sub(self, &rhs);
    }
}

impl Neg for RingPoly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        RingPoly::neg(&self)
    }
}

impl Mul for RingPoly {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        RingPoly::mul(&self, &rhs)
    }
}

impl<'a> Mul<&'a RingPoly> for RingPoly {
    type Output = Self;

    fn mul(self, rhs: &'a RingPoly) -> Self::Output {
        RingPoly::mul(&self, rhs)
    }
}

impl MulAssign for RingPoly {
    fn mul_assign(&mut self, rhs: Self) {
        *self = RingPoly::mul(self, &rhs);
    }
}

#[cfg(test)]
mod ring_tests {
    use super::*;
    use crate::ahe::params::ParamSet;

    fn test_params() -> BgvParams {
        ParamSet::Toy.params()
    }

    #[test]
    fn test_ring_zero() {
        let params = test_params();
        let zero = RingPoly::zero(&params);

        assert_eq!(zero.dimension(), params.n);
        assert!(zero.coeffs().iter().all(|&c| c == 0));
    }

    #[test]
    fn test_ring_constant() {
        let params = test_params();
        let c = RingPoly::constant(42, &params);

        assert_eq!(c.coeffs()[0], 42);
        assert!(c.coeffs()[1..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_ring_add() {
        let params = test_params();
        let a = RingPoly::from_slice(&[1, 2, 3], &params);
        let b = RingPoly::from_slice(&[4, 5, 6], &params);

        let c = a + b;

        assert_eq!(c.coeffs()[0], 5);
        assert_eq!(c.coeffs()[1], 7);
        assert_eq!(c.coeffs()[2], 9);
    }

    #[test]
    fn test_ring_sub() {
        let params = test_params();
        let a = RingPoly::from_slice(&[10, 20, 30], &params);
        let b = RingPoly::from_slice(&[1, 2, 3], &params);

        let c = a - b;

        assert_eq!(c.coeffs()[0], 9);
        assert_eq!(c.coeffs()[1], 18);
        assert_eq!(c.coeffs()[2], 27);
    }

    #[test]
    fn test_ring_neg() {
        let params = test_params();
        let a = RingPoly::from_slice(&[1, 2, 3], &params);
        let neg_a = -a.clone();

        let sum = a + neg_a;
        assert!(sum.coeffs().iter().all(|&c| c == 0));
    }

    #[test]
    fn test_ring_scalar_mul() {
        let params = test_params();
        let a = RingPoly::from_slice(&[1, 2, 3], &params);
        let b = a.scalar_mul(5);

        assert_eq!(b.coeffs()[0], 5);
        assert_eq!(b.coeffs()[1], 10);
        assert_eq!(b.coeffs()[2], 15);
    }

    #[test]
    fn test_ring_mul_simple() {
        // Test (1 + x) * (1 + x) = 1 + 2x + x^2 in Z_q[X]/(X^n + 1)
        let params = test_params();
        let a = RingPoly::from_slice(&[1, 1], &params);
        let b = a.clone();

        let c = a * b;

        assert_eq!(c.coeffs()[0], 1);
        assert_eq!(c.coeffs()[1], 2);
        assert_eq!(c.coeffs()[2], 1);
    }

    #[test]
    fn test_ring_mul_wraparound() {
        // Test that X^n ≡ -1 (mod X^n + 1)
        // Create polynomial X^(n-1), multiply by X to get X^n ≡ -1
        let params = test_params();
        let n = params.n;

        // a = X^(n-1)
        let mut a_coeffs = vec![0u64; n];
        a_coeffs[n - 1] = 1;
        let a = RingPoly::new(a_coeffs, params.q);

        // b = X
        let mut b_coeffs = vec![0u64; n];
        b_coeffs[1] = 1;
        let b = RingPoly::new(b_coeffs, params.q);

        // a * b = X^n ≡ -1 = q - 1
        let c = a * b;

        assert_eq!(c.coeffs()[0], params.q - 1); // -1 mod q
        assert!(c.coeffs()[1..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(RingPoly::mod_pow(2, 10, 1000), 24); // 2^10 = 1024 ≡ 24 (mod 1000)
        assert_eq!(RingPoly::mod_pow(3, 5, 100), 43); // 3^5 = 243 ≡ 43 (mod 100)
    }

    #[test]
    fn test_mod_inverse() {
        let q = 1073741789u64; // Prime from Toy params
        let a = 12345u64;
        let a_inv = RingPoly::mod_inverse(a, q);

        let prod = ((a as u128 * a_inv as u128) % q as u128) as u64;
        assert_eq!(prod, 1);
    }
}
