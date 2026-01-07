//! Number Theoretic Transform (NTT) for efficient polynomial multiplication.
//!
//! NTT is the finite field analogue of the Fast Fourier Transform, enabling
//! O(n log n) polynomial multiplication instead of O(n²).
//!
//! # Requirements
//!
//! The field must have primitive roots of unity of order 2^k where k is the
//! log2 of the transform size. The Goldilocks field supports up to 2^32.
//!
//! # Usage
//!
//! ```ignore
//! use mpz_justvengers_core::ntt::Ntt;
//! use mpz_fields::goldilocks::Goldilocks;
//!
//! // Create NTT instance for size 8
//! let ntt = Ntt::<Goldilocks>::new(3).unwrap(); // 2^3 = 8
//!
//! // Transform coefficients to evaluation form
//! let mut coeffs = vec![Goldilocks::new(1), Goldilocks::new(2), ...];
//! ntt.forward(&mut coeffs);
//!
//! // Transform back to coefficient form
//! ntt.inverse(&mut coeffs);
//! ```

use mpz_fields::Field;
use thiserror::Error;

/// Trait for fields that support NTT operations.
pub trait NttField: Field {
    /// Returns a primitive root of unity of order 2^k.
    ///
    /// Returns None if k exceeds the maximum supported.
    fn primitive_root_of_unity(k: u32) -> Option<Self>;

    /// Returns the maximum log2 of supported NTT size.
    fn max_ntt_log_size() -> u32;
}

// Implement NttField for Goldilocks
impl NttField for mpz_fields::goldilocks::Goldilocks {
    fn primitive_root_of_unity(k: u32) -> Option<Self> {
        mpz_fields::goldilocks::Goldilocks::primitive_root_of_unity(k)
    }

    fn max_ntt_log_size() -> u32 {
        mpz_fields::goldilocks::Goldilocks::max_ntt_log_size()
    }
}

/// Errors that can occur during NTT operations.
#[derive(Debug, Error)]
pub enum NttError {
    /// The requested NTT size exceeds the field's maximum.
    #[error("NTT size 2^{0} exceeds maximum supported 2^{1}")]
    SizeTooLarge(u32, u32),

    /// Input length is not a power of two.
    #[error("input length {0} is not a power of two")]
    NotPowerOfTwo(usize),

    /// Input length doesn't match NTT size.
    #[error("input length {0} doesn't match NTT size {1}")]
    LengthMismatch(usize, usize),
}

/// Number Theoretic Transform for a specific size.
///
/// Precomputes twiddle factors for efficient repeated transforms.
#[derive(Clone)]
pub struct Ntt<F: NttField> {
    /// Log2 of the transform size.
    log_size: u32,
    /// Transform size (2^log_size).
    size: usize,
    /// Precomputed twiddle factors for forward transform.
    /// twiddles[i] = omega^(bit_reverse(i))
    twiddles: Vec<F>,
    /// Precomputed twiddle factors for inverse transform.
    inv_twiddles: Vec<F>,
    /// 1/n for inverse transform normalization.
    inv_size: F,
}

impl<F: NttField> Ntt<F> {
    /// Creates a new NTT instance for transforms of size 2^log_size.
    ///
    /// # Errors
    ///
    /// Returns an error if log_size exceeds the field's maximum NTT size.
    pub fn new(log_size: u32) -> Result<Self, NttError> {
        let max_log = F::max_ntt_log_size();
        if log_size > max_log {
            return Err(NttError::SizeTooLarge(log_size, max_log));
        }

        let size = 1usize << log_size;

        // Get primitive root of unity of order 2^log_size
        let omega = F::primitive_root_of_unity(log_size)
            .expect("primitive root should exist for valid log_size");
        let omega_inv = omega
            .inverse()
            .expect("root of unity should have inverse");

        // Precompute twiddle factors
        let twiddles = Self::compute_twiddles(omega, size);
        let inv_twiddles = Self::compute_twiddles(omega_inv, size);

        // Compute 1/n
        let n = Self::field_from_usize(size);
        let inv_size = n.inverse().expect("size should be invertible");

        Ok(Self {
            log_size,
            size,
            twiddles,
            inv_twiddles,
            inv_size,
        })
    }

    /// Converts a usize to field element.
    fn field_from_usize(n: usize) -> F {
        let mut result = F::zero();
        let one = F::one();
        for _ in 0..n {
            result = result + one;
        }
        result
    }

    /// Computes twiddle factors in bit-reversed order.
    fn compute_twiddles(omega: F, size: usize) -> Vec<F> {
        let mut twiddles = Vec::with_capacity(size);
        let mut current = F::one();

        for _ in 0..size {
            twiddles.push(current);
            current = current * omega;
        }

        twiddles
    }

    /// Returns the transform size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns log2 of the transform size.
    pub fn log_size(&self) -> u32 {
        self.log_size
    }

    /// Performs the forward NTT in-place (Cooley-Tukey).
    ///
    /// Transforms coefficients to evaluation form.
    ///
    /// # Errors
    ///
    /// Returns an error if the input length doesn't match the NTT size.
    pub fn forward(&self, data: &mut [F]) -> Result<(), NttError> {
        if data.len() != self.size {
            return Err(NttError::LengthMismatch(data.len(), self.size));
        }

        // Bit-reversal permutation
        Self::bit_reverse_permutation(data, self.log_size);

        // Cooley-Tukey butterfly
        self.cooley_tukey(data, &self.twiddles);

        Ok(())
    }

    /// Performs the inverse NTT in-place (Cooley-Tukey).
    ///
    /// Transforms evaluation form back to coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error if the input length doesn't match the NTT size.
    pub fn inverse(&self, data: &mut [F]) -> Result<(), NttError> {
        if data.len() != self.size {
            return Err(NttError::LengthMismatch(data.len(), self.size));
        }

        // Bit-reversal permutation
        Self::bit_reverse_permutation(data, self.log_size);

        // Cooley-Tukey butterfly with inverse twiddles
        self.cooley_tukey(data, &self.inv_twiddles);

        // Normalize by 1/n
        for x in data.iter_mut() {
            *x = *x * self.inv_size;
        }

        Ok(())
    }

    /// Performs bit-reversal permutation in-place.
    fn bit_reverse_permutation(data: &mut [F], log_n: u32) {
        let n = data.len();
        for i in 0..n {
            let j = Self::bit_reverse(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }
    }

    /// Reverses the bits of a number.
    #[inline]
    fn bit_reverse(mut x: usize, bits: u32) -> usize {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Cooley-Tukey iterative NTT.
    fn cooley_tukey(&self, data: &mut [F], twiddles: &[F]) {
        let n = data.len();
        let mut m = 1;

        for _ in 0..self.log_size {
            let half_m = m;
            m <<= 1;

            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    // Twiddle factor: omega^(j * n / m)
                    let twiddle_idx = j * (n / m);
                    let t = twiddles[twiddle_idx] * data[k + j + half_m];
                    let u = data[k + j];

                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;
                }
            }
        }
    }

    /// Multiplies two polynomials using NTT.
    ///
    /// Both polynomials must have length at most `size/2` to avoid wraparound.
    /// The result is placed in the first polynomial's buffer, padded to `size`.
    ///
    /// # Arguments
    ///
    /// * `a` - First polynomial coefficients (will be modified)
    /// * `b` - Second polynomial coefficients
    ///
    /// # Panics
    ///
    /// Panics if either polynomial has more than `size/2` coefficients.
    pub fn multiply(&self, a: &mut Vec<F>, b: &[F]) {
        assert!(
            a.len() <= self.size / 2,
            "polynomial a too large for NTT multiplication"
        );
        assert!(
            b.len() <= self.size / 2,
            "polynomial b too large for NTT multiplication"
        );

        // Pad to NTT size
        a.resize(self.size, F::zero());
        let mut b_padded = b.to_vec();
        b_padded.resize(self.size, F::zero());

        // Forward NTT
        self.forward(a).expect("length should match");
        self.forward(&mut b_padded).expect("length should match");

        // Pointwise multiplication
        for (ai, bi) in a.iter_mut().zip(b_padded.iter()) {
            *ai = *ai * *bi;
        }

        // Inverse NTT
        self.inverse(a).expect("length should match");
    }
}

/// Finds the smallest power of 2 >= n.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Returns log2 of a power of 2.
///
/// # Panics
///
/// Panics if n is not a power of 2.
pub fn log2(n: usize) -> u32 {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    n.trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_fields::goldilocks::Goldilocks;

    #[test]
    fn test_ntt_roundtrip() {
        let ntt = Ntt::<Goldilocks>::new(4).unwrap(); // Size 16

        let original: Vec<Goldilocks> = (0..16).map(|i| Goldilocks::new(i as u64)).collect();
        let mut data = original.clone();

        ntt.forward(&mut data).unwrap();
        ntt.inverse(&mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_roundtrip_random() {
        use mpz_core::{Block, prg::Prg};
        use mpz_fields::UniformRand;
        use rand::SeedableRng;

        let ntt = Ntt::<Goldilocks>::new(8).unwrap(); // Size 256

        let mut rng = Prg::from_seed(Block::ZERO);
        let original: Vec<Goldilocks> =
            (0..256).map(|_| Goldilocks::rand(&mut rng)).collect();
        let mut data = original.clone();

        ntt.forward(&mut data).unwrap();
        ntt.inverse(&mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_convolution() {
        // Test that NTT correctly computes polynomial multiplication
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let ntt = Ntt::<Goldilocks>::new(2).unwrap(); // Size 4

        let mut a = vec![Goldilocks::new(1), Goldilocks::new(2)];
        let b = vec![Goldilocks::new(3), Goldilocks::new(4)];

        ntt.multiply(&mut a, &b);

        assert_eq!(a[0], Goldilocks::new(3)); // constant term
        assert_eq!(a[1], Goldilocks::new(10)); // x coefficient
        assert_eq!(a[2], Goldilocks::new(8)); // x^2 coefficient
        assert_eq!(a[3], Goldilocks::zero()); // x^3 coefficient
    }

    #[test]
    fn test_ntt_multiply_larger() {
        // (1 + x + x^2) * (1 + x + x^2) = 1 + 2x + 3x^2 + 2x^3 + x^4
        let ntt = Ntt::<Goldilocks>::new(3).unwrap(); // Size 8

        let mut a = vec![
            Goldilocks::new(1),
            Goldilocks::new(1),
            Goldilocks::new(1),
        ];
        let b = vec![
            Goldilocks::new(1),
            Goldilocks::new(1),
            Goldilocks::new(1),
        ];

        ntt.multiply(&mut a, &b);

        assert_eq!(a[0], Goldilocks::new(1)); // 1
        assert_eq!(a[1], Goldilocks::new(2)); // 2x
        assert_eq!(a[2], Goldilocks::new(3)); // 3x^2
        assert_eq!(a[3], Goldilocks::new(2)); // 2x^3
        assert_eq!(a[4], Goldilocks::new(1)); // x^4
        assert_eq!(a[5], Goldilocks::zero());
        assert_eq!(a[6], Goldilocks::zero());
        assert_eq!(a[7], Goldilocks::zero());
    }

    #[test]
    fn test_ntt_multiply_random() {
        use mpz_core::{Block, prg::Prg};
        use mpz_fields::UniformRand;
        use rand::SeedableRng;

        let mut rng = Prg::from_seed(Block::ZERO);

        // Generate random polynomials of degree 15
        let a_coeffs: Vec<Goldilocks> =
            (0..16).map(|_| Goldilocks::rand(&mut rng)).collect();
        let b_coeffs: Vec<Goldilocks> =
            (0..16).map(|_| Goldilocks::rand(&mut rng)).collect();

        // Compute product using NTT
        let ntt = Ntt::<Goldilocks>::new(6).unwrap(); // Size 64 (need 32 for degree 31 product)
        let mut a_ntt = a_coeffs.clone();
        ntt.multiply(&mut a_ntt, &b_coeffs);

        // Compute product using naive O(n^2) method
        let mut naive_result = vec![Goldilocks::zero(); 32];
        for (i, &ai) in a_coeffs.iter().enumerate() {
            for (j, &bj) in b_coeffs.iter().enumerate() {
                naive_result[i + j] = naive_result[i + j] + ai * bj;
            }
        }

        // Compare (NTT result is padded to 64)
        for i in 0..32 {
            assert_eq!(
                a_ntt[i], naive_result[i],
                "mismatch at coefficient {}",
                i
            );
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b000, 3), 0b000);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b001, 3), 0b100);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b010, 3), 0b010);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b011, 3), 0b110);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b100, 3), 0b001);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b101, 3), 0b101);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b110, 3), 0b011);
        assert_eq!(Ntt::<Goldilocks>::bit_reverse(0b111, 3), 0b111);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(17), 32);
    }

    #[test]
    fn test_log2() {
        assert_eq!(log2(1), 0);
        assert_eq!(log2(2), 1);
        assert_eq!(log2(4), 2);
        assert_eq!(log2(8), 3);
        assert_eq!(log2(256), 8);
    }

    #[test]
    fn test_ntt_size_too_large() {
        let result = Ntt::<Goldilocks>::new(33);
        assert!(result.is_err());
    }

    #[test]
    fn test_ntt_length_mismatch() {
        let ntt = Ntt::<Goldilocks>::new(3).unwrap(); // Size 8
        let mut data = vec![Goldilocks::zero(); 4];

        assert!(ntt.forward(&mut data).is_err());
        assert!(ntt.inverse(&mut data).is_err());
    }

    // ==================== Additional comprehensive tests ====================

    mod stress_tests {
        use super::*;
        use mpz_core::{Block, prg::Prg};
        use mpz_fields::UniformRand;
        use rand::SeedableRng;

        #[test]
        fn test_ntt_large_transform() {
            // Test with size 2^12 = 4096
            let ntt = Ntt::<Goldilocks>::new(12).unwrap();

            let mut rng = Prg::from_seed(Block::ZERO);
            let original: Vec<Goldilocks> =
                (0..4096).map(|_| Goldilocks::rand(&mut rng)).collect();
            let mut data = original.clone();

            ntt.forward(&mut data).unwrap();
            ntt.inverse(&mut data).unwrap();

            assert_eq!(data, original);
        }

        #[test]
        fn test_ntt_multiply_many_random() {
            use rand::Rng;

            // Test multiplication correctness with many random pairs
            let mut rng = Prg::from_seed(Block::new([42u8; 16]));

            for _ in 0..10 {
                let deg_a = (rng.random::<u32>() % 31) as usize + 1;
                let deg_b = (rng.random::<u32>() % 31) as usize + 1;

                let a_coeffs: Vec<Goldilocks> =
                    (0..=deg_a).map(|_| Goldilocks::rand(&mut rng)).collect();
                let b_coeffs: Vec<Goldilocks> =
                    (0..=deg_b).map(|_| Goldilocks::rand(&mut rng)).collect();

                // NTT multiplication
                let ntt = Ntt::<Goldilocks>::new(7).unwrap(); // Size 128
                let mut a_ntt = a_coeffs.clone();
                ntt.multiply(&mut a_ntt, &b_coeffs);

                // Naive multiplication
                let mut naive = vec![Goldilocks::zero(); deg_a + deg_b + 1];
                for (i, &ai) in a_coeffs.iter().enumerate() {
                    for (j, &bj) in b_coeffs.iter().enumerate() {
                        naive[i + j] = naive[i + j] + ai * bj;
                    }
                }

                // Compare
                for i in 0..naive.len() {
                    assert_eq!(a_ntt[i], naive[i], "mismatch at coeff {}", i);
                }
            }
        }

        #[test]
        fn test_ntt_identity_element() {
            // Multiplying by 1 (constant polynomial) should give same result
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let mut rng = Prg::from_seed(Block::ZERO);
            let a_coeffs: Vec<Goldilocks> =
                (0..8).map(|_| Goldilocks::rand(&mut rng)).collect();

            let one = vec![Goldilocks::one()];
            let mut result = a_coeffs.clone();
            ntt.multiply(&mut result, &one);

            // First 8 coefficients should match
            for i in 0..8 {
                assert_eq!(result[i], a_coeffs[i]);
            }
        }

        #[test]
        fn test_ntt_zero_element() {
            // Multiplying by 0 should give zero
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let a_coeffs: Vec<Goldilocks> = (0..8).map(|i| Goldilocks::new(i as u64 + 1)).collect();
            let zero = vec![Goldilocks::zero()];

            let mut result = a_coeffs.clone();
            ntt.multiply(&mut result, &zero);

            for coeff in result {
                assert_eq!(coeff, Goldilocks::zero());
            }
        }
    }

    mod linearity_tests {
        use super::*;
        use mpz_core::{Block, prg::Prg};
        use mpz_fields::UniformRand;
        use rand::SeedableRng;

        #[test]
        fn test_ntt_is_linear() {
            // NTT(a + b) = NTT(a) + NTT(b)
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let mut rng = Prg::from_seed(Block::ZERO);
            let a: Vec<Goldilocks> = (0..16).map(|_| Goldilocks::rand(&mut rng)).collect();
            let b: Vec<Goldilocks> = (0..16).map(|_| Goldilocks::rand(&mut rng)).collect();

            // a + b
            let sum: Vec<Goldilocks> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

            let mut ntt_a = a.clone();
            let mut ntt_b = b.clone();
            let mut ntt_sum = sum.clone();

            ntt.forward(&mut ntt_a).unwrap();
            ntt.forward(&mut ntt_b).unwrap();
            ntt.forward(&mut ntt_sum).unwrap();

            // NTT(a) + NTT(b)
            let sum_of_ntt: Vec<Goldilocks> =
                ntt_a.iter().zip(ntt_b.iter()).map(|(&x, &y)| x + y).collect();

            assert_eq!(ntt_sum, sum_of_ntt);
        }

        #[test]
        fn test_ntt_scalar_multiplication() {
            // NTT(c * a) = c * NTT(a)
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let mut rng = Prg::from_seed(Block::ZERO);
            let a: Vec<Goldilocks> = (0..16).map(|_| Goldilocks::rand(&mut rng)).collect();
            let c = Goldilocks::new(12345);

            // c * a
            let scaled: Vec<Goldilocks> = a.iter().map(|&x| c * x).collect();

            let mut ntt_a = a.clone();
            let mut ntt_scaled = scaled.clone();

            ntt.forward(&mut ntt_a).unwrap();
            ntt.forward(&mut ntt_scaled).unwrap();

            // c * NTT(a)
            let scaled_ntt: Vec<Goldilocks> = ntt_a.iter().map(|&x| c * x).collect();

            assert_eq!(ntt_scaled, scaled_ntt);
        }
    }

    mod edge_cases_tests {
        use super::*;

        #[test]
        fn test_ntt_size_one() {
            let ntt = Ntt::<Goldilocks>::new(0).unwrap(); // Size 1

            let mut data = vec![Goldilocks::new(42)];
            let original = data.clone();

            ntt.forward(&mut data).unwrap();
            ntt.inverse(&mut data).unwrap();

            assert_eq!(data, original);
        }

        #[test]
        fn test_ntt_size_two() {
            let ntt = Ntt::<Goldilocks>::new(1).unwrap(); // Size 2

            let original = vec![Goldilocks::new(3), Goldilocks::new(7)];
            let mut data = original.clone();

            ntt.forward(&mut data).unwrap();
            ntt.inverse(&mut data).unwrap();

            assert_eq!(data, original);
        }

        #[test]
        fn test_ntt_all_zeros() {
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let mut data = vec![Goldilocks::zero(); 16];

            ntt.forward(&mut data).unwrap();
            // NTT of all zeros should be all zeros
            assert!(data.iter().all(|&x| x == Goldilocks::zero()));

            ntt.inverse(&mut data).unwrap();
            assert!(data.iter().all(|&x| x == Goldilocks::zero()));
        }

        #[test]
        fn test_ntt_all_ones() {
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let original = vec![Goldilocks::one(); 16];
            let mut data = original.clone();

            ntt.forward(&mut data).unwrap();
            ntt.inverse(&mut data).unwrap();

            assert_eq!(data, original);
        }

        #[test]
        fn test_ntt_max_supported_size() {
            // Create NTT at max size for Goldilocks (2^32)
            // This is too large to actually run, so just verify we can create smaller ones
            let result = Ntt::<Goldilocks>::new(16); // Size 2^16
            assert!(result.is_ok());
        }

        #[test]
        fn test_multiply_degree_zero() {
            // (5) * (3) = 15
            let ntt = Ntt::<Goldilocks>::new(2).unwrap();

            let mut a = vec![Goldilocks::new(5)];
            let b = vec![Goldilocks::new(3)];

            ntt.multiply(&mut a, &b);

            assert_eq!(a[0], Goldilocks::new(15));
        }
    }

    mod convolution_properties {
        use super::*;

        #[test]
        fn test_multiply_commutativity() {
            // a * b = b * a
            let ntt = Ntt::<Goldilocks>::new(4).unwrap();

            let a = vec![Goldilocks::new(1), Goldilocks::new(2), Goldilocks::new(3)];
            let b = vec![Goldilocks::new(4), Goldilocks::new(5)];

            let mut result_ab = a.clone();
            ntt.multiply(&mut result_ab, &b);

            let mut result_ba = b.clone();
            ntt.multiply(&mut result_ba, &a);

            // Compare coefficients up to the product degree
            for i in 0..5 {
                assert_eq!(result_ab[i], result_ba[i], "mismatch at {}", i);
            }
        }

        #[test]
        fn test_multiply_associativity() {
            // (a * b) * c = a * (b * c)
            let ntt = Ntt::<Goldilocks>::new(5).unwrap(); // Size 32

            let a = vec![Goldilocks::new(1), Goldilocks::new(2)];
            let b = vec![Goldilocks::new(3), Goldilocks::new(4)];
            let c = vec![Goldilocks::new(5), Goldilocks::new(6)];

            // (a * b) * c
            let mut ab = a.clone();
            ntt.multiply(&mut ab, &b);
            let mut abc_left: Vec<_> = ab.iter().take(4).cloned().collect();
            let ntt2 = Ntt::<Goldilocks>::new(4).unwrap();
            ntt2.multiply(&mut abc_left, &c);

            // a * (b * c)
            let mut bc = b.clone();
            ntt.multiply(&mut bc, &c);
            let mut abc_right = a.clone();
            ntt2.multiply(&mut abc_right, &bc.iter().take(4).cloned().collect::<Vec<_>>());

            // Compare first few coefficients
            for i in 0..5 {
                assert_eq!(abc_left[i], abc_right[i], "mismatch at {}", i);
            }
        }
    }
}
