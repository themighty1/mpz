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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingPoly {
    /// Coefficients in Z_q, length exactly n.
    coeffs: Vec<u64>,
    /// The modulus q.
    q: u64,
}

impl RingPoly {
    /// Creates a new ring polynomial with given coefficients.
    ///
    /// Coefficients are reduced modulo q.
    pub fn new(coeffs: Vec<u64>, q: u64) -> Self {
        let mut poly = Self { coeffs, q };
        poly.reduce();
        poly
    }

    /// Creates the zero polynomial.
    pub fn zero(params: &BgvParams) -> Self {
        Self {
            coeffs: vec![0; params.n],
            q: params.q,
        }
    }

    /// Creates a polynomial with all coefficients equal to c.
    pub fn constant(c: u64, params: &BgvParams) -> Self {
        let mut coeffs = vec![0; params.n];
        coeffs[0] = c % params.q;
        Self {
            coeffs,
            q: params.q,
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

        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| {
                let sum = (a as u128) + (b as u128);
                (sum % self.q as u128) as u64
            })
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Subtracts two polynomials coefficient-wise.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| {
                if a >= b {
                    a - b
                } else {
                    self.q - (b - a) % self.q
                }
            })
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Negates all coefficients.
    pub fn neg(&self) -> Self {
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| if c == 0 { 0 } else { self.q - c })
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Multiplies by a scalar.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let s = scalar % self.q;
        let coeffs: Vec<u64> = self
            .coeffs
            .iter()
            .map(|&c| ((c as u128 * s as u128) % self.q as u128) as u64)
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Multiplies two polynomials in the ring R_q = Z_q[X]/(X^n + 1).
    ///
    /// Uses schoolbook multiplication with reduction mod X^n + 1.
    /// For X^n ≡ -1 (mod X^n + 1), so X^(n+i) ≡ -X^i.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        let n = self.coeffs.len();
        let mut result = vec![0i128; n];

        // Schoolbook multiplication
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                let prod = (a as i128) * (b as i128);
                let idx = i + j;

                if idx < n {
                    result[idx] += prod;
                } else {
                    // X^n ≡ -1, so X^(n+k) ≡ -X^k
                    result[idx - n] -= prod;
                }
            }
        }

        // Reduce to [0, q)
        let coeffs: Vec<u64> = result
            .iter()
            .map(|&c| {
                let c_mod = c.rem_euclid(self.q as i128);
                c_mod as u64
            })
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Multiplies two polynomials using NTT (faster for large n).
    ///
    /// Requires that q ≡ 1 (mod 2n) for NTT to work.
    pub fn mul_ntt(&self, other: &Self, omega: u64) -> Self {
        assert_eq!(self.q, other.q, "moduli must match");
        assert_eq!(self.coeffs.len(), other.coeffs.len(), "dimensions must match");

        let n = self.coeffs.len();

        // Convert to NTT domain
        let mut a_ntt = self.coeffs.clone();
        let mut b_ntt = other.coeffs.clone();

        Self::ntt_forward(&mut a_ntt, omega, self.q);
        Self::ntt_forward(&mut b_ntt, omega, self.q);

        // Pointwise multiplication
        let mut c_ntt: Vec<u64> = a_ntt
            .iter()
            .zip(b_ntt.iter())
            .map(|(&a, &b)| ((a as u128 * b as u128) % self.q as u128) as u64)
            .collect();

        // Convert back
        let omega_inv = Self::mod_inverse(omega, self.q);
        Self::ntt_inverse(&mut c_ntt, omega_inv, self.q);

        // Scale by 1/n
        let n_inv = Self::mod_inverse(n as u64, self.q);
        let coeffs: Vec<u64> = c_ntt
            .iter()
            .map(|&c| ((c as u128 * n_inv as u128) % self.q as u128) as u64)
            .collect();

        Self { coeffs, q: self.q }
    }

    /// Forward NTT (Cooley-Tukey).
    fn ntt_forward(data: &mut [u64], omega: u64, q: u64) {
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
            let omega_m = Self::mod_pow(omega, exp as u64, q);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let t = ((w as u128 * data[k + j + half_m] as u128) % q as u128) as u64;
                    let u = data[k + j];

                    data[k + j] = if u + t >= q { u + t - q } else { u + t };
                    data[k + j + half_m] = if u >= t { u - t } else { q + u - t };

                    w = ((w as u128 * omega_m as u128) % q as u128) as u64;
                }
            }
        }
    }

    /// Inverse NTT.
    fn ntt_inverse(data: &mut [u64], omega_inv: u64, q: u64) {
        Self::ntt_forward(data, omega_inv, q);
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
