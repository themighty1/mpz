//! Polynomial arithmetic for Justvengers.
//!
//! This module provides polynomials over finite fields with support for:
//! - Lagrange interpolation
//! - Evaluation at arbitrary points
//! - Vanishing polynomial generation and checking
//! - Basic arithmetic operations

use mpz_fields::Field;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A polynomial over a finite field F.
///
/// Represented in coefficient form where `coeffs[i]` is the coefficient of x^i.
/// The zero polynomial has an empty coefficient vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<F: Field> {
    /// Coefficients in ascending order of degree: coeffs[i] = coefficient of x^i
    coeffs: Vec<F>,
}

impl<F: Field> Poly<F> {
    /// Creates the zero polynomial.
    #[inline]
    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    /// Creates a constant polynomial.
    #[inline]
    pub fn constant(c: F) -> Self {
        if c == F::zero() {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// Creates the polynomial f(x) = x.
    #[inline]
    pub fn x() -> Self {
        Self {
            coeffs: vec![F::zero(), F::one()],
        }
    }

    /// Creates a polynomial from coefficients (ascending order).
    ///
    /// `coeffs[i]` is the coefficient of x^i.
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Returns the coefficients of the polynomial.
    #[inline]
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Returns the degree of the polynomial.
    ///
    /// Returns `None` for the zero polynomial.
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Returns true if this is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Returns the leading coefficient, or None for zero polynomial.
    #[inline]
    pub fn leading_coeff(&self) -> Option<F> {
        self.coeffs.last().copied()
    }

    /// Evaluates the polynomial at point `x` using Horner's method.
    pub fn eval(&self, x: F) -> F {
        if self.coeffs.is_empty() {
            return F::zero();
        }

        // Horner's method: a_n*x^n + ... + a_0 = ((...(a_n*x + a_{n-1})*x + ...)*x + a_0
        let mut result = F::zero();
        for &coeff in self.coeffs.iter().rev() {
            result = result * x + coeff;
        }
        result
    }

    /// Evaluates the polynomial at multiple points.
    pub fn eval_many(&self, points: &[F]) -> Vec<F> {
        points.iter().map(|&x| self.eval(x)).collect()
    }

    /// Performs Lagrange interpolation.
    ///
    /// Given points `(α_1, y_1), ..., (α_d, y_d)`, returns the unique polynomial
    /// of degree at most d-1 such that f(α_i) = y_i for all i.
    ///
    /// # Panics
    /// Panics if `alphas` and `ys` have different lengths, or if alphas are not distinct.
    pub fn interpolate(alphas: &[F], ys: &[F]) -> Self {
        assert_eq!(
            alphas.len(),
            ys.len(),
            "interpolation requires equal number of points and values"
        );

        if alphas.is_empty() {
            return Self::zero();
        }

        let n = alphas.len();

        // Lagrange interpolation:
        // L(x) = Σ_i y_i * Π_{j≠i} (x - α_j) / (α_i - α_j)

        let mut result = Self::zero();

        for i in 0..n {
            // Compute the i-th Lagrange basis polynomial:
            // L_i(x) = Π_{j≠i} (x - α_j) / (α_i - α_j)
            let mut basis = Self::constant(F::one());
            let mut denom = F::one();

            for j in 0..n {
                if i != j {
                    // Numerator: multiply by (x - α_j)
                    let factor = Self::from_coeffs(vec![-alphas[j], F::one()]);
                    basis = basis * factor;

                    // Denominator: multiply by (α_i - α_j)
                    denom = denom * (alphas[i] - alphas[j]);
                }
            }

            // L_i(x) = basis / denom
            let denom_inv = denom.inverse().expect("interpolation points must be distinct");
            basis = basis.scalar_mul(denom_inv);

            // result += y_i * L_i(x)
            result = result + basis.scalar_mul(ys[i]);
        }

        result
    }

    /// Computes the vanishing polynomial for the given points.
    ///
    /// Returns Z(x) = Π_i (x - α_i), which vanishes at all given points.
    pub fn vanishing(alphas: &[F]) -> Self {
        if alphas.is_empty() {
            return Self::constant(F::one());
        }

        let mut result = Self::from_coeffs(vec![-alphas[0], F::one()]);

        for &alpha in &alphas[1..] {
            let factor = Self::from_coeffs(vec![-alpha, F::one()]);
            result = result * factor;
        }

        result
    }

    /// Returns a random vanishing polynomial of a given degree for the specified points.
    ///
    /// The polynomial vanishes at all points in `alphas` and has the specified degree.
    /// This is done by generating a random polynomial of degree (degree - |alphas|)
    /// and multiplying by the vanishing polynomial.
    pub fn random_vanishing<R: rand::Rng>(alphas: &[F], degree: usize, rng: &mut R) -> Self {
        let z = Self::vanishing(alphas);
        let z_deg = z.degree().unwrap_or(0);

        if degree < z_deg {
            panic!(
                "requested degree {} is less than vanishing polynomial degree {}",
                degree, z_deg
            );
        }

        // Generate random polynomial of degree (degree - z_deg)
        let extra_deg = degree - z_deg;
        let mut random_coeffs = Vec::with_capacity(extra_deg + 1);
        for _ in 0..=extra_deg {
            random_coeffs.push(F::rand(rng));
        }
        let random_poly = Self::from_coeffs(random_coeffs);

        z * random_poly
    }

    /// Checks if the polynomial vanishes at all given points.
    pub fn is_vanishing_at(&self, alphas: &[F]) -> bool {
        alphas.iter().all(|&alpha| self.eval(alpha) == F::zero())
    }

    /// Multiplies the polynomial by a scalar.
    pub fn scalar_mul(mut self, scalar: F) -> Self {
        if scalar == F::zero() {
            return Self::zero();
        }
        for coeff in &mut self.coeffs {
            *coeff = *coeff * scalar;
        }
        self
    }
}

// NTT-based multiplication for fields that support it
impl<F: crate::ntt::NttField> Poly<F> {
    /// Multiplies two polynomials using NTT for O(n log n) complexity.
    ///
    /// This is much faster than the naive O(n²) multiplication for large
    /// polynomials. The crossover point is typically around degree 32-64.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The polynomial to multiply with.
    ///
    /// # Returns
    ///
    /// The product polynomial.
    ///
    /// # Panics
    ///
    /// Panics if the result degree would exceed the NTT size limit (2^32 for Goldilocks).
    pub fn mul_ntt(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let result_len = self.coeffs.len() + rhs.coeffs.len() - 1;

        // Find smallest power of 2 >= result_len
        let ntt_size = crate::ntt::next_power_of_two(result_len);
        let log_size = crate::ntt::log2(ntt_size);

        // Create NTT instance
        let ntt = crate::ntt::Ntt::new(log_size)
            .expect("NTT size should be within field limits");

        // Pad coefficients to NTT size
        let mut a = self.coeffs.clone();
        a.resize(ntt_size, F::zero());

        let mut b = rhs.coeffs.clone();
        b.resize(ntt_size, F::zero());

        // Forward NTT
        ntt.forward(&mut a).expect("length should match");
        ntt.forward(&mut b).expect("length should match");

        // Pointwise multiplication
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai = *ai * *bi;
        }

        // Inverse NTT
        ntt.inverse(&mut a).expect("length should match");

        // Trim to actual result length
        a.truncate(result_len);

        Self::from_coeffs(a)
    }

    /// Multiplies two polynomials, automatically choosing the best algorithm.
    ///
    /// Uses NTT for large polynomials (degree > threshold) and naive
    /// multiplication for small ones.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The polynomial to multiply with.
    /// * `threshold` - Minimum degree to use NTT (default: 64).
    ///
    /// # Returns
    ///
    /// The product polynomial.
    pub fn mul_auto(&self, rhs: &Self, threshold: usize) -> Self {
        let max_degree = self
            .degree()
            .unwrap_or(0)
            .max(rhs.degree().unwrap_or(0));

        if max_degree >= threshold {
            self.mul_ntt(rhs)
        } else {
            self.clone() * rhs.clone()
        }
    }
}

impl<F: Field> Poly<F> {
    /// Removes trailing zero coefficients.
    fn normalize(&mut self) {
        while let Some(&c) = self.coeffs.last() {
            if c == F::zero() {
                self.coeffs.pop();
            } else {
                break;
            }
        }
    }
}

impl<F: Field> Default for Poly<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> Add for Poly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let max_len = self.coeffs.len().max(rhs.coeffs.len());
        let mut coeffs = vec![F::zero(); max_len];

        for (i, &c) in self.coeffs.iter().enumerate() {
            coeffs[i] = coeffs[i] + c;
        }
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            coeffs[i] = coeffs[i] + c;
        }

        Self::from_coeffs(coeffs)
    }
}

impl<F: Field> AddAssign for Poly<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field> Sub for Poly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<F: Field> SubAssign for Poly<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field> Neg for Poly<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_coeffs(self.coeffs.into_iter().map(|c| -c).collect())
    }
}

impl<F: Field> Mul for Poly<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let result_len = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut coeffs = vec![F::zero(); result_len];

        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j] + a * b;
            }
        }

        Self::from_coeffs(coeffs)
    }
}

impl<F: Field> MulAssign for Poly<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_fields::gf2_128::Gf2_128;

    type F = Gf2_128;
    type P = Poly<F>;

    fn f(v: u128) -> F {
        Gf2_128::new(v)
    }

    #[test]
    fn test_poly_zero() {
        let p = P::zero();
        assert!(p.is_zero());
        assert_eq!(p.degree(), None);
        assert_eq!(p.eval(f(5)), f(0));
    }

    #[test]
    fn test_poly_constant() {
        let p = P::constant(f(7));
        assert!(!p.is_zero());
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.eval(f(0)), f(7));
        assert_eq!(p.eval(f(100)), f(7));
    }

    #[test]
    fn test_poly_x() {
        let p = P::x();
        assert_eq!(p.degree(), Some(1));
        assert_eq!(p.eval(f(0)), f(0));
        assert_eq!(p.eval(f(5)), f(5));
        assert_eq!(p.eval(f(123)), f(123));
    }

    #[test]
    fn test_poly_from_coeffs() {
        // p(x) = 1 + 2x + 3x^2
        let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
        assert_eq!(p.degree(), Some(2));

        // p(0) = 1
        assert_eq!(p.eval(f(0)), f(1));

        // p(1) = 1 + 2 + 3 = 1 XOR 2 XOR 3 = 0 (in GF(2^128))
        assert_eq!(p.eval(f(1)), f(1) + f(2) + f(3));
    }

    #[test]
    fn test_poly_add() {
        let p1 = P::from_coeffs(vec![f(1), f(2)]);
        let p2 = P::from_coeffs(vec![f(3), f(0), f(5)]);
        let sum = p1 + p2;

        // (1 + 2x) + (3 + 5x^2) = (1+3) + 2x + 5x^2
        assert_eq!(sum.coeffs(), &[f(1) + f(3), f(2), f(5)]);
    }

    #[test]
    fn test_poly_mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        // In GF(2^128): 1 + 0x + x^2 = 1 + x^2
        let p = P::from_coeffs(vec![f(1), f(1)]);
        let prod = p.clone() * p;

        assert_eq!(prod.degree(), Some(2));
        // In GF(2^128), 1+1 = 0
        assert_eq!(prod.coeffs()[1], f(0));
    }

    #[test]
    fn test_poly_scalar_mul() {
        let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
        let scaled = p.scalar_mul(f(5));

        assert_eq!(
            scaled.coeffs(),
            &[f(1) * f(5), f(2) * f(5), f(3) * f(5)]
        );
    }

    #[test]
    fn test_interpolation_single_point() {
        let alphas = vec![f(5)];
        let ys = vec![f(42)];

        let p = P::interpolate(&alphas, &ys);

        // Should be constant polynomial
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.eval(f(5)), f(42));
    }

    #[test]
    fn test_interpolation_two_points() {
        // Line through (1, 3) and (2, 7)
        let alphas = vec![f(1), f(2)];
        let ys = vec![f(3), f(7)];

        let p = P::interpolate(&alphas, &ys);

        assert!(p.degree().unwrap() <= 1);
        assert_eq!(p.eval(f(1)), f(3));
        assert_eq!(p.eval(f(2)), f(7));
    }

    #[test]
    fn test_interpolation_many_points() {
        let alphas: Vec<F> = (1u128..=5).map(f).collect();
        let ys: Vec<F> = vec![f(10), f(20), f(30), f(40), f(50)];

        let p = P::interpolate(&alphas, &ys);

        // Should interpolate all points
        for (i, &alpha) in alphas.iter().enumerate() {
            assert_eq!(p.eval(alpha), ys[i]);
        }

        // Degree should be at most n-1
        assert!(p.degree().unwrap() <= 4);
    }

    #[test]
    fn test_vanishing_polynomial() {
        let alphas: Vec<F> = vec![f(1), f(2), f(3)];
        let z = P::vanishing(&alphas);

        // Z should vanish at all alphas
        for &alpha in &alphas {
            assert_eq!(z.eval(alpha), f(0));
        }

        // Degree should be |alphas|
        assert_eq!(z.degree(), Some(3));
    }

    #[test]
    fn test_is_vanishing_at() {
        let alphas: Vec<F> = vec![f(1), f(2), f(3)];
        let z = P::vanishing(&alphas);

        assert!(z.is_vanishing_at(&alphas));

        // Non-vanishing polynomial
        let p = P::constant(f(1));
        assert!(!p.is_vanishing_at(&alphas));
    }

    #[test]
    fn test_random_vanishing() {
        use mpz_core::{Block, prg::Prg};
        use rand::SeedableRng;

        let mut rng = Prg::from_seed(Block::ZERO);
        let alphas: Vec<F> = vec![f(1), f(2), f(3)];

        let p = P::random_vanishing(&alphas, 5, &mut rng);

        // Should vanish at all points
        assert!(p.is_vanishing_at(&alphas));

        // Should have correct degree
        assert_eq!(p.degree(), Some(5));
    }

    #[test]
    fn test_interpolation_constant() {
        // All y values the same should give constant polynomial
        let alphas: Vec<F> = vec![f(1), f(2), f(3)];
        let ys: Vec<F> = vec![f(42), f(42), f(42)];

        let p = P::interpolate(&alphas, &ys);

        // Should be constant 42
        assert_eq!(p.eval(f(0)), f(42));
        assert_eq!(p.eval(f(100)), f(42));
    }

    // ==================== Additional exhaustive tests ====================

    mod arithmetic_properties {
        use super::*;

        #[test]
        fn test_add_commutativity() {
            let p1 = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let p2 = P::from_coeffs(vec![f(4), f(5)]);

            assert_eq!(p1.clone() + p2.clone(), p2 + p1);
        }

        #[test]
        fn test_add_associativity() {
            let p1 = P::from_coeffs(vec![f(1), f(2)]);
            let p2 = P::from_coeffs(vec![f(3), f(4)]);
            let p3 = P::from_coeffs(vec![f(5), f(6), f(7)]);

            let lhs = (p1.clone() + p2.clone()) + p3.clone();
            let rhs = p1 + (p2 + p3);
            assert_eq!(lhs, rhs);
        }

        #[test]
        fn test_add_identity() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let zero = P::zero();

            assert_eq!(p.clone() + zero.clone(), p.clone());
            assert_eq!(zero + p.clone(), p);
        }

        #[test]
        fn test_mul_commutativity() {
            let p1 = P::from_coeffs(vec![f(1), f(2)]);
            let p2 = P::from_coeffs(vec![f(3), f(4), f(5)]);

            assert_eq!(p1.clone() * p2.clone(), p2 * p1);
        }

        #[test]
        fn test_mul_associativity() {
            let p1 = P::from_coeffs(vec![f(1), f(2)]);
            let p2 = P::from_coeffs(vec![f(3), f(4)]);
            let p3 = P::from_coeffs(vec![f(5), f(6)]);

            let lhs = (p1.clone() * p2.clone()) * p3.clone();
            let rhs = p1 * (p2 * p3);
            assert_eq!(lhs, rhs);
        }

        #[test]
        fn test_mul_identity() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let one = P::constant(F::one());

            assert_eq!(p.clone() * one.clone(), p.clone());
            assert_eq!(one * p.clone(), p);
        }

        #[test]
        fn test_mul_zero() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let zero = P::zero();

            assert_eq!(p.clone() * zero.clone(), P::zero());
            assert_eq!(zero * p, P::zero());
        }

        #[test]
        fn test_distributivity() {
            let p1 = P::from_coeffs(vec![f(1), f(2)]);
            let p2 = P::from_coeffs(vec![f(3), f(4)]);
            let p3 = P::from_coeffs(vec![f(5), f(6)]);

            // p1 * (p2 + p3) = p1*p2 + p1*p3
            let lhs = p1.clone() * (p2.clone() + p3.clone());
            let rhs = (p1.clone() * p2) + (p1 * p3);
            assert_eq!(lhs, rhs);
        }

        #[test]
        fn test_subtraction() {
            let p1 = P::from_coeffs(vec![f(5), f(7), f(9)]);
            let p2 = P::from_coeffs(vec![f(2), f(3)]);

            let diff = p1.clone() - p2.clone();
            // Verify: diff + p2 = p1
            assert_eq!(diff + p2, p1);
        }

        #[test]
        fn test_negation() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let neg_p = -p.clone();

            // p + (-p) = 0
            assert_eq!(p + neg_p, P::zero());
        }

        #[test]
        fn test_sub_self_is_zero() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            assert_eq!(p.clone() - p, P::zero());
        }

        #[test]
        fn test_scalar_mul_zero() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            assert_eq!(p.scalar_mul(F::zero()), P::zero());
        }

        #[test]
        fn test_scalar_mul_one() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            assert_eq!(p.clone().scalar_mul(F::one()), p);
        }
    }

    mod normalization_tests {
        use super::*;

        #[test]
        fn test_trailing_zeros_removed() {
            let p = P::from_coeffs(vec![f(1), f(2), f(0), f(0), f(0)]);
            assert_eq!(p.degree(), Some(1));
            assert_eq!(p.coeffs().len(), 2);
        }

        #[test]
        fn test_all_zeros_becomes_zero_poly() {
            let p = P::from_coeffs(vec![f(0), f(0), f(0)]);
            assert!(p.is_zero());
            assert_eq!(p.degree(), None);
        }

        #[test]
        fn test_add_cancellation_normalizes() {
            // p1 + p2 where leading terms cancel
            let p1 = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let p2 = P::from_coeffs(vec![f(0), f(0), f(3)]); // In GF(2^128), 3+3=0

            let sum = p1 + p2;
            assert_eq!(sum.degree(), Some(1)); // x^2 term cancelled
        }
    }

    mod evaluation_tests {
        use super::*;

        #[test]
        fn test_eval_many_consistency() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3), f(4)]);
            let points: Vec<F> = (0u128..10).map(f).collect();

            let results = p.eval_many(&points);

            for (i, &pt) in points.iter().enumerate() {
                assert_eq!(results[i], p.eval(pt));
            }
        }

        #[test]
        fn test_eval_higher_degree() {
            // p(x) = 1 + x + x^2 + x^3 + x^4
            let p = P::from_coeffs(vec![f(1), f(1), f(1), f(1), f(1)]);

            // p(0) = 1
            assert_eq!(p.eval(f(0)), f(1));

            // p(1) = 1+1+1+1+1 = 1 (in GF(2^128), odd number of 1s)
            assert_eq!(p.eval(f(1)), f(1));
        }

        #[test]
        fn test_eval_zero_poly_everywhere() {
            let p = P::zero();
            for i in 0u128..100 {
                assert_eq!(p.eval(f(i)), f(0));
            }
        }
    }

    mod interpolation_tests {
        use super::*;

        #[test]
        fn test_interpolation_empty() {
            let alphas: Vec<F> = vec![];
            let ys: Vec<F> = vec![];

            let p = P::interpolate(&alphas, &ys);
            assert!(p.is_zero());
        }

        #[test]
        #[should_panic(expected = "interpolation requires equal number")]
        fn test_interpolation_mismatched_lengths() {
            let alphas = vec![f(1), f(2)];
            let ys = vec![f(10)];
            let _ = P::interpolate(&alphas, &ys);
        }

        #[test]
        #[should_panic(expected = "interpolation points must be distinct")]
        fn test_interpolation_duplicate_points() {
            let alphas = vec![f(1), f(1)]; // duplicate!
            let ys = vec![f(10), f(20)];
            let _ = P::interpolate(&alphas, &ys);
        }

        #[test]
        fn test_interpolation_degree_bound() {
            // n points -> degree at most n-1
            for n in 1usize..=8 {
                let alphas: Vec<F> = (1..=n as u128).map(f).collect();
                let ys: Vec<F> = (100..100 + n as u128).map(f).collect();

                let p = P::interpolate(&alphas, &ys);

                if let Some(deg) = p.degree() {
                    assert!(deg <= n - 1, "degree {} > n-1 = {} for n={}", deg, n - 1, n);
                }
            }
        }

        #[test]
        fn test_interpolation_uniqueness() {
            // Interpolate twice, should get same polynomial
            let alphas: Vec<F> = vec![f(1), f(2), f(3), f(4)];
            let ys: Vec<F> = vec![f(10), f(25), f(33), f(47)];

            let p1 = P::interpolate(&alphas, &ys);
            let p2 = P::interpolate(&alphas, &ys);

            assert_eq!(p1, p2);
        }

        #[test]
        fn test_interpolation_roundtrip() {
            // Create polynomial, evaluate at points, interpolate back
            let original = P::from_coeffs(vec![f(5), f(3), f(7), f(2)]);
            let alphas: Vec<F> = vec![f(1), f(2), f(3), f(4)];
            let ys: Vec<F> = alphas.iter().map(|&a| original.eval(a)).collect();

            let recovered = P::interpolate(&alphas, &ys);

            // Should recover original (or equivalent)
            for &a in &alphas {
                assert_eq!(original.eval(a), recovered.eval(a));
            }
        }
    }

    mod vanishing_tests {
        use super::*;
        use mpz_core::{Block, prg::Prg};
        use rand::SeedableRng;

        #[test]
        fn test_vanishing_empty() {
            let alphas: Vec<F> = vec![];
            let z = P::vanishing(&alphas);

            // Empty vanishing polynomial is constant 1
            assert_eq!(z, P::constant(F::one()));
        }

        #[test]
        fn test_vanishing_single() {
            let alphas = vec![f(5)];
            let z = P::vanishing(&alphas);

            // Z(x) = x - 5
            assert_eq!(z.degree(), Some(1));
            assert_eq!(z.eval(f(5)), f(0));
            assert_ne!(z.eval(f(0)), f(0));
        }

        #[test]
        fn test_vanishing_degree_equals_num_points() {
            for n in 1usize..=10 {
                let alphas: Vec<F> = (1..=n as u128).map(f).collect();
                let z = P::vanishing(&alphas);
                assert_eq!(z.degree(), Some(n));
            }
        }

        #[test]
        fn test_vanishing_leading_coeff_is_one() {
            let alphas: Vec<F> = vec![f(1), f(2), f(3), f(4), f(5)];
            let z = P::vanishing(&alphas);

            assert_eq!(z.leading_coeff(), Some(F::one()));
        }

        #[test]
        fn test_vanishing_only_at_roots() {
            let alphas: Vec<F> = vec![f(10), f(20), f(30)];
            let z = P::vanishing(&alphas);

            // Should vanish at roots
            for &a in &alphas {
                assert_eq!(z.eval(a), f(0));
            }

            // Should NOT vanish at other points (with high probability)
            for i in 0u128..100 {
                if !alphas.contains(&f(i)) {
                    // In a large field, unlikely to accidentally be zero
                    // (not a hard guarantee, but good sanity check)
                }
            }
            // At least verify at one specific non-root
            assert_ne!(z.eval(f(15)), f(0));
        }

        #[test]
        fn test_random_vanishing_different_each_time() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let alphas: Vec<F> = vec![f(1), f(2), f(3)];

            let p1 = P::random_vanishing(&alphas, 6, &mut rng);
            let p2 = P::random_vanishing(&alphas, 6, &mut rng);

            // Both should vanish at alphas
            assert!(p1.is_vanishing_at(&alphas));
            assert!(p2.is_vanishing_at(&alphas));

            // But should be different polynomials (with overwhelming probability)
            assert_ne!(p1, p2);
        }

        #[test]
        #[should_panic(expected = "requested degree")]
        fn test_random_vanishing_degree_too_small() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let alphas: Vec<F> = vec![f(1), f(2), f(3)]; // vanishing has degree 3
            let _ = P::random_vanishing(&alphas, 2, &mut rng); // requested degree < 3
        }

        #[test]
        fn test_is_vanishing_zero_poly() {
            let zero = P::zero();
            let alphas: Vec<F> = vec![f(1), f(2), f(3)];

            // Zero polynomial vanishes everywhere
            assert!(zero.is_vanishing_at(&alphas));
        }

        #[test]
        fn test_is_vanishing_empty_points() {
            let p = P::from_coeffs(vec![f(1), f(2), f(3)]);
            let alphas: Vec<F> = vec![];

            // Any polynomial "vanishes" at empty set of points
            assert!(p.is_vanishing_at(&alphas));
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_constant_zero_is_zero_poly() {
            let p = P::constant(F::zero());
            assert!(p.is_zero());
        }

        #[test]
        fn test_leading_coeff_zero_poly() {
            let p = P::zero();
            assert_eq!(p.leading_coeff(), None);
        }

        #[test]
        fn test_mul_degree_sum() {
            // deg(p*q) = deg(p) + deg(q) when neither is zero
            let p1 = P::from_coeffs(vec![f(1), f(2), f(3)]); // deg 2
            let p2 = P::from_coeffs(vec![f(4), f(5)]);       // deg 1

            let prod = p1 * p2;
            assert_eq!(prod.degree(), Some(3)); // 2 + 1
        }

        #[test]
        fn test_default_is_zero() {
            let p: P = Default::default();
            assert!(p.is_zero());
        }
    }

    mod ntt_tests {
        use mpz_fields::goldilocks::Goldilocks;

        type G = Goldilocks;
        type PG = super::Poly<G>;

        fn g(v: u64) -> G {
            Goldilocks::new(v)
        }

        #[test]
        fn test_mul_ntt_simple() {
            // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
            let p1 = PG::from_coeffs(vec![g(1), g(2)]);
            let p2 = PG::from_coeffs(vec![g(3), g(4)]);

            let naive = p1.clone() * p2.clone();
            let ntt = p1.mul_ntt(&p2);

            assert_eq!(naive, ntt);
            assert_eq!(ntt.coeffs()[0], g(3));
            assert_eq!(ntt.coeffs()[1], g(10));
            assert_eq!(ntt.coeffs()[2], g(8));
        }

        #[test]
        fn test_mul_ntt_larger() {
            // (1 + x + x^2) * (1 + x + x^2) = 1 + 2x + 3x^2 + 2x^3 + x^4
            let p = PG::from_coeffs(vec![g(1), g(1), g(1)]);

            let naive = p.clone() * p.clone();
            let ntt = p.mul_ntt(&p);

            assert_eq!(naive, ntt);
        }

        #[test]
        fn test_mul_ntt_zero() {
            let zero = PG::zero();
            let p = PG::from_coeffs(vec![g(1), g(2), g(3)]);

            assert_eq!(zero.mul_ntt(&p), PG::zero());
            assert_eq!(p.mul_ntt(&zero), PG::zero());
            assert_eq!(zero.mul_ntt(&zero), PG::zero());
        }

        #[test]
        fn test_mul_ntt_random() {
            use mpz_core::{Block, prg::Prg};
            use mpz_fields::UniformRand;
            use rand::SeedableRng;

            let mut rng = Prg::from_seed(Block::ZERO);

            // Generate random polynomials of degree 63
            let a_coeffs: Vec<G> = (0..64).map(|_| G::rand(&mut rng)).collect();
            let b_coeffs: Vec<G> = (0..64).map(|_| G::rand(&mut rng)).collect();

            let p1 = PG::from_coeffs(a_coeffs);
            let p2 = PG::from_coeffs(b_coeffs);

            let naive = p1.clone() * p2.clone();
            let ntt = p1.mul_ntt(&p2);

            assert_eq!(naive, ntt);
        }

        #[test]
        fn test_mul_auto_uses_naive_for_small() {
            let p1 = PG::from_coeffs(vec![g(1), g(2), g(3)]);
            let p2 = PG::from_coeffs(vec![g(4), g(5)]);

            let naive = p1.clone() * p2.clone();
            let auto = p1.mul_auto(&p2, 64); // threshold 64, so use naive

            assert_eq!(naive, auto);
        }

        #[test]
        fn test_mul_auto_uses_ntt_for_large() {
            use mpz_core::{Block, prg::Prg};
            use mpz_fields::UniformRand;
            use rand::SeedableRng;

            let mut rng = Prg::from_seed(Block::ZERO);

            let a_coeffs: Vec<G> = (0..128).map(|_| G::rand(&mut rng)).collect();
            let b_coeffs: Vec<G> = (0..128).map(|_| G::rand(&mut rng)).collect();

            let p1 = PG::from_coeffs(a_coeffs);
            let p2 = PG::from_coeffs(b_coeffs);

            let naive = p1.clone() * p2.clone();
            let auto = p1.mul_auto(&p2, 64); // threshold 64, so use NTT

            assert_eq!(naive, auto);
        }
    }
}
