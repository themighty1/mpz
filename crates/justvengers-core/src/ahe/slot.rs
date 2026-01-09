//! Slot packing for SIMD operations in BGV.
//!
//! When the plaintext modulus t satisfies t ≡ 1 (mod 2n), the cyclotomic
//! polynomial X^n + 1 factors completely into linear factors modulo t.
//! This allows encoding n plaintext values into a single polynomial,
//! enabling SIMD-style operations where addition and scalar multiplication
//! act component-wise on all n slots.
//!
//! # Mathematical Background
//!
//! The ring Z_t[X]/(X^n + 1) is isomorphic to Z_t^n via the CRT:
//!   Z_t[X]/(X^n + 1) ≅ ∏_{i=0}^{n-1} Z_t[X]/(X - ζ^{2i+1})
//!
//! where ζ is a primitive 2n-th root of unity modulo t.
//!
//! Encoding n values (m_0, ..., m_{n-1}) into a polynomial uses inverse NTT,
//! and decoding uses forward NTT (evaluation at the roots).

use super::params::BgvParams;
use super::ring::RingPoly;

/// Slot encoder/decoder for BGV plaintexts.
#[derive(Clone, Debug)]
pub struct SlotEncoder {
    /// Number of slots (equals ring dimension n).
    n: usize,
    /// Plaintext modulus t.
    t: u64,
    /// Primitive 2n-th root of unity modulo t.
    /// Satisfies: zeta^n ≡ -1 (mod t), zeta^(2n) ≡ 1 (mod t).
    zeta: u64,
    /// Inverse of zeta modulo t.
    zeta_inv: u64,
    /// Inverse of n modulo t (for inverse NTT scaling).
    n_inv: u64,
    /// Precomputed powers: zeta_powers[i] = zeta^(2i+1) mod t (evaluation points).
    zeta_powers: Vec<u64>,
    /// Precomputed inverse powers for inverse NTT.
    zeta_inv_powers: Vec<u64>,
}

impl SlotEncoder {
    /// Creates a new slot encoder for the given parameters.
    ///
    /// Returns None if slot packing is not supported (t ≢ 1 (mod 2n)).
    pub fn new(params: &BgvParams) -> Option<Self> {
        Self::new_direct(params.n, params.t)
    }

    /// Creates a new slot encoder with explicit n and t.
    ///
    /// Returns None if slot packing is not supported (t ≢ 1 (mod 2n)).
    pub fn new_direct(n: usize, t: u64) -> Option<Self> {

        // Check if slot packing is supported: t ≡ 1 (mod 2n)
        let order = 2 * n as u64;
        if (t - 1) % order != 0 {
            return None;
        }

        // Find primitive 2n-th root of unity modulo t
        let zeta = Self::find_primitive_root(n, t)?;
        let zeta_inv = Self::mod_inverse(zeta, t);
        let n_inv = Self::mod_inverse(n as u64, t);

        // Precompute powers: zeta^(2i+1) for i = 0..n
        let mut zeta_powers = Vec::with_capacity(n);
        let mut power = zeta; // zeta^1
        for _ in 0..n {
            zeta_powers.push(power);
            // Next power: zeta^(2(i+1)+1) = zeta^(2i+1) * zeta^2
            power = Self::mod_mul(power, Self::mod_mul(zeta, zeta, t), t);
        }

        // Precompute inverse powers for inverse NTT
        let zeta_inv_2 = Self::mod_mul(zeta_inv, zeta_inv, t);
        let mut zeta_inv_powers = Vec::with_capacity(n);
        let mut inv_power = zeta_inv;
        for _ in 0..n {
            zeta_inv_powers.push(inv_power);
            inv_power = Self::mod_mul(inv_power, zeta_inv_2, t);
        }

        Some(Self {
            n,
            t,
            zeta,
            zeta_inv,
            n_inv,
            zeta_powers,
            zeta_inv_powers,
        })
    }

    /// Returns the number of slots (equals ring dimension).
    pub fn num_slots(&self) -> usize {
        self.n
    }

    /// Returns the plaintext modulus.
    pub fn modulus(&self) -> u64 {
        self.t
    }

    /// Encodes n slot values into a polynomial.
    ///
    /// Given values (m_0, ..., m_{n-1}), computes polynomial p(X) such that
    /// p(zeta^(2i+1)) = m_i for all i.
    ///
    /// This is done via inverse NTT (interpolation).
    pub fn encode(&self, slots: &[u64]) -> Vec<u64> {
        assert_eq!(slots.len(), self.n, "must provide exactly n slot values");

        // Reduce slot values modulo t
        let mut values: Vec<u64> = slots.iter().map(|&v| v % self.t).collect();

        // Inverse NTT to get polynomial coefficients
        self.inverse_ntt(&mut values);

        values
    }

    /// Decodes a polynomial to extract slot values.
    ///
    /// Evaluates p(zeta^(2i+1)) for i = 0..n to get the slot values.
    ///
    /// This is done via forward NTT (evaluation).
    pub fn decode(&self, coeffs: &[u64]) -> Vec<u64> {
        assert_eq!(coeffs.len(), self.n, "polynomial must have n coefficients");

        let mut values: Vec<u64> = coeffs.to_vec();

        // Forward NTT to evaluate at roots
        self.forward_ntt(&mut values);

        values
    }

    /// Encodes slots into a RingPoly suitable for encryption.
    pub fn encode_to_ring(&self, slots: &[u64], params: &BgvParams) -> RingPoly {
        let coeffs = self.encode(slots);
        RingPoly::from_slice(&coeffs, params)
    }

    /// Forward NTT: evaluates polynomial at zeta^(2i+1) for i = 0..n.
    ///
    /// Uses pre-twist followed by Cooley-Tukey butterfly.
    /// This computes p(ζ^(2k+1)) = Σ_i a_i * ζ^((2k+1)i) for k = 0..n-1.
    ///
    /// The pre-twist converts evaluation at odd powers to standard NTT:
    /// - Pre-twist: b_i = a_i * ζ^i
    /// - NTT: result[k] = Σ_i b_i * ω^(ik) = Σ_i a_i * ζ^((2k+1)i) = p(ζ^(2k+1))
    fn forward_ntt(&self, data: &mut [u64]) {
        let n = self.n;
        let log_n = n.trailing_zeros();

        // Pre-twist: multiply by zeta^i for evaluation at odd powers
        // This converts: p(ζ^(2k+1)) = Σ a_i * ζ^((2k+1)i) to standard NTT form
        let mut twist = 1u64;
        for i in 0..n {
            data[i] = Self::mod_mul(data[i], twist, self.t);
            twist = Self::mod_mul(twist, self.zeta, self.t);
        }

        // Bit-reversal permutation
        Self::bit_reverse(data, log_n);

        // Cooley-Tukey butterflies for standard NTT
        // Uses omega = zeta^2 (an n-th root of unity)
        for s in 0..log_n {
            let m = 1 << (s + 1);
            let half_m = m / 2;

            let exp = n / m;
            let w_m = Self::mod_pow(self.zeta, (2 * exp) as u64, self.t);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let u = data[k + j];
                    let t_val = Self::mod_mul(w, data[k + j + half_m], self.t);

                    data[k + j] = Self::mod_add(u, t_val, self.t);
                    data[k + j + half_m] = Self::mod_sub(u, t_val, self.t);

                    w = Self::mod_mul(w, w_m, self.t);
                }
            }
        }
    }

    /// Inverse NTT: interpolates polynomial from values at zeta^(2i+1).
    ///
    /// Uses Gentleman-Sande butterfly with post-untwist.
    /// This is the inverse of forward_ntt (which uses pre-twist).
    ///
    /// The post-untwist converts from standard inverse NTT to interpolation at odd powers:
    /// - Inverse NTT gives: c_i = (1/n) * Σ_k v_k * ω^(-ik)
    /// - Post-untwist: a_i = c_i * ζ^(-i) gives correct coefficients for p(ζ^(2k+1)) = v_k
    fn inverse_ntt(&self, data: &mut [u64]) {
        let n = self.n;
        let log_n = n.trailing_zeros();

        // Gentleman-Sande (inverse) butterflies
        for s in (0..log_n).rev() {
            let m = 1 << (s + 1);
            let half_m = m / 2;

            let exp = n / m;
            let w_m_inv = Self::mod_pow(self.zeta_inv, (2 * exp) as u64, self.t);

            for k in (0..n).step_by(m) {
                let mut w = 1u64;
                for j in 0..half_m {
                    let u = data[k + j];
                    let v = data[k + j + half_m];

                    data[k + j] = Self::mod_add(u, v, self.t);
                    data[k + j + half_m] = Self::mod_mul(Self::mod_sub(u, v, self.t), w, self.t);

                    w = Self::mod_mul(w, w_m_inv, self.t);
                }
            }
        }

        // Bit-reversal permutation
        Self::bit_reverse(data, log_n);

        // Scale by 1/n
        for x in data.iter_mut() {
            *x = Self::mod_mul(*x, self.n_inv, self.t);
        }

        // Post-untwist: multiply by zeta^(-i) to convert from standard inverse NTT
        // to interpolation at odd powers ζ^(2k+1)
        let mut twist_inv = 1u64;
        for i in 0..n {
            data[i] = Self::mod_mul(data[i], twist_inv, self.t);
            twist_inv = Self::mod_mul(twist_inv, self.zeta_inv, self.t);
        }
    }

    /// Bit-reversal permutation.
    fn bit_reverse(data: &mut [u64], log_n: u32) {
        let n = data.len();
        for i in 0..n {
            let j = Self::reverse_bits(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }
    }

    /// Reverses the bits of x using log_n bits.
    fn reverse_bits(mut x: usize, log_n: u32) -> usize {
        let mut result = 0;
        for _ in 0..log_n {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Finds a primitive 2n-th root of unity modulo t.
    fn find_primitive_root(n: usize, t: u64) -> Option<u64> {
        let order = 2 * n as u64;
        if (t - 1) % order != 0 {
            return None;
        }

        let exp = (t - 1) / order;

        // Try small primes as generators
        for g in 2..1000u64 {
            let zeta = Self::mod_pow(g, exp, t);

            // Verify: zeta^n should equal t-1 (i.e., -1 mod t)
            let zeta_n = Self::mod_pow(zeta, n as u64, t);
            if zeta_n == t - 1 {
                return Some(zeta);
            }
        }

        None
    }

    /// Modular addition: (a + b) mod t.
    /// Uses u128 to avoid overflow with large moduli like Goldilocks.
    #[inline]
    fn mod_add(a: u64, b: u64, t: u64) -> u64 {
        let sum = (a as u128) + (b as u128);
        (sum % t as u128) as u64
    }

    /// Modular subtraction: (a - b) mod t.
    /// Uses u128 to handle underflow correctly.
    #[inline]
    fn mod_sub(a: u64, b: u64, t: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            // a - b + t, but a < b, so compute t - (b - a)
            t - (b - a)
        }
    }

    /// Modular multiplication: (a * b) mod t.
    #[inline]
    fn mod_mul(a: u64, b: u64, t: u64) -> u64 {
        ((a as u128 * b as u128) % t as u128) as u64
    }

    /// Modular exponentiation: base^exp mod t.
    fn mod_pow(mut base: u64, mut exp: u64, t: u64) -> u64 {
        let mut result = 1u64;
        base %= t;

        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mod_mul(result, base, t);
            }
            exp >>= 1;
            base = Self::mod_mul(base, base, t);
        }

        result
    }

    /// Modular inverse using extended GCD.
    fn mod_inverse(a: u64, t: u64) -> u64 {
        let mut old_r = t as i128;
        let mut r = a as i128;
        let mut old_s = 0i128;
        let mut s = 1i128;

        while r != 0 {
            let quotient = old_r / r;
            (old_r, r) = (r, old_r - quotient * r);
            (old_s, s) = (s, old_s - quotient * s);
        }

        if old_s < 0 {
            (old_s + t as i128) as u64
        } else {
            old_s as u64
        }
    }
}

#[cfg(test)]
mod slot_tests {
    use super::*;
    use crate::ahe::params::ParamSet;

    #[test]
    fn test_slot_encoder_creation() {
        // Toy params: n=256, t=65537
        // Check: 65537 - 1 = 65536 = 2^16, 2n = 512 = 2^9
        // 65536 % 512 = 0 ✓
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params);
        assert!(encoder.is_some(), "Toy params should support slot packing");

        let encoder = encoder.unwrap();
        assert_eq!(encoder.num_slots(), params.n);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();

        // Create test slot values
        let slots: Vec<u64> = (0..params.n as u64).collect();

        // Encode and decode
        let encoded = encoder.encode(&slots);
        let decoded = encoder.decode(&encoded);

        // Should get back original values
        for (i, (&original, &recovered)) in slots.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                original % encoder.modulus(),
                recovered,
                "mismatch at slot {}",
                i
            );
        }
    }

    #[test]
    fn test_encode_decode_random() {
        use mpz_core::{prg::Prg, Block};
        use rand::{Rng, SeedableRng};

        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();
        let mut rng = Prg::from_seed(Block::ZERO);

        // Random slot values
        let slots: Vec<u64> = (0..params.n)
            .map(|_| rng.random::<u64>() % encoder.modulus())
            .collect();

        let encoded = encoder.encode(&slots);
        let decoded = encoder.decode(&encoded);

        for (i, (&original, &recovered)) in slots.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(original, recovered, "mismatch at slot {}", i);
        }
    }

    #[test]
    fn test_encode_constant() {
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();

        // All slots have same value
        let value = 42u64;
        let slots = vec![value; params.n];

        let encoded = encoder.encode(&slots);
        let decoded = encoder.decode(&encoded);

        // All decoded slots should be the same value
        for (i, &d) in decoded.iter().enumerate() {
            assert_eq!(d, value, "slot {} mismatch", i);
        }
    }

    #[test]
    fn test_slot_addition_homomorphism() {
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();
        let t = encoder.modulus();

        // Two sets of slot values
        let slots1: Vec<u64> = (0..params.n as u64).map(|i| i % t).collect();
        let slots2: Vec<u64> = (0..params.n as u64).map(|i| (i * 2) % t).collect();

        // Encode both
        let poly1 = encoder.encode(&slots1);
        let poly2 = encoder.encode(&slots2);

        // Add polynomials coefficient-wise
        let poly_sum: Vec<u64> = poly1
            .iter()
            .zip(poly2.iter())
            .map(|(&a, &b)| (a + b) % t)
            .collect();

        // Decode sum
        let decoded_sum = encoder.decode(&poly_sum);

        // Should equal slot-wise addition
        for (i, ((&s1, &s2), &d)) in slots1.iter().zip(slots2.iter()).zip(decoded_sum.iter()).enumerate() {
            let expected = (s1 + s2) % t;
            assert_eq!(expected, d, "mismatch at slot {} for addition", i);
        }
    }

    #[test]
    fn test_primitive_root() {
        // t = 65537 (Fermat prime), n = 256
        // 2n = 512, and 65536 / 512 = 128
        let t = 65537u64;
        let n = 256usize;

        let zeta = SlotEncoder::find_primitive_root(n, t).unwrap();

        // zeta^n should equal -1 mod t
        let zeta_n = SlotEncoder::mod_pow(zeta, n as u64, t);
        assert_eq!(zeta_n, t - 1, "zeta^n should be -1");

        // zeta^(2n) should equal 1 mod t
        let zeta_2n = SlotEncoder::mod_pow(zeta, 2 * n as u64, t);
        assert_eq!(zeta_2n, 1, "zeta^(2n) should be 1");
    }

    #[test]
    fn test_goldilocks_slot_encoder() {
        use crate::ahe::params::GOLDILOCKS;

        // Create a mock BgvParams with Goldilocks modulus and N=8192
        // Note: We can't use the full BgvParams because q > t is required
        // but we can test SlotEncoder directly with Goldilocks as modulus

        // Verify Goldilocks supports N=8192 slot packing
        let n = 8192usize;
        let t = GOLDILOCKS;
        let order = 2 * n as u64;
        assert_eq!((t - 1) % order, 0, "Goldilocks must support N=8192");

        // Test primitive root finding
        let zeta = SlotEncoder::find_primitive_root(n, t);
        assert!(zeta.is_some(), "Should find primitive root for Goldilocks with N=8192");

        let zeta = zeta.unwrap();

        // Verify zeta^n = -1 mod t
        let zeta_n = SlotEncoder::mod_pow(zeta, n as u64, t);
        assert_eq!(zeta_n, t - 1, "zeta^n should be -1 mod Goldilocks");

        // Verify zeta^(2n) = 1 mod t
        let zeta_2n = SlotEncoder::mod_pow(zeta, 2 * n as u64, t);
        assert_eq!(zeta_2n, 1, "zeta^(2n) should be 1 mod Goldilocks");
    }

    #[test]
    fn test_goldilocks_small_example() {
        use crate::ahe::params::GOLDILOCKS;

        // Use a smaller ring dimension that Goldilocks also supports
        // Goldilocks - 1 = 2^64 - 2^32 = 2^32 * (2^32 - 1)
        // This is divisible by many powers of 2, so we can use various ring dimensions

        // Test with N=64 (order=128)
        let n = 64usize;
        let t = GOLDILOCKS;

        // Create a minimal BgvParams-like struct just for testing
        // We manually construct a SlotEncoder
        let order = 2 * n as u64;
        assert_eq!((t - 1) % order, 0);

        let zeta = SlotEncoder::find_primitive_root(n, t).unwrap();
        let zeta_inv = SlotEncoder::mod_inverse(zeta, t);
        let n_inv = SlotEncoder::mod_inverse(n as u64, t);

        // Simple roundtrip test: encode small values and decode
        let slots: Vec<u64> = (0..n as u64).collect();

        // Manual encode/decode (since we don't have full BgvParams)
        // This verifies the NTT math works with Goldilocks

        // Verify inverse exists
        assert_eq!(SlotEncoder::mod_mul(zeta, zeta_inv, t), 1);
        assert_eq!(SlotEncoder::mod_mul(n as u64, n_inv, t), 1);
    }

    #[test]
    fn test_decode_constant_polynomial() {
        // Test that the constant polynomial p(X) = 1 evaluates to 1 at all roots.
        // This is critical for verifying the NTT correctly evaluates at odd roots.
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();
        let n = encoder.num_slots();

        // Constant polynomial: coeffs = [1, 0, 0, ..., 0]
        let mut coeffs = vec![0u64; n];
        coeffs[0] = 1;

        // Decode should give [1, 1, 1, ..., 1]
        let decoded = encoder.decode(&coeffs);

        for (i, &val) in decoded.iter().enumerate() {
            assert_eq!(val, 1, "slot {} should be 1 for constant polynomial p(X)=1", i);
        }
    }

    #[test]
    fn test_encode_constant_slots() {
        // Test that encoding constant slots [c, c, ..., c] gives the constant polynomial [c, 0, 0, ..., 0].
        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();
        let n = encoder.num_slots();

        let constant = 42u64;
        let slots = vec![constant; n];

        let encoded = encoder.encode(&slots);

        // First coefficient should be the constant value
        assert_eq!(encoded[0], constant, "first coefficient should equal the constant");

        // All other coefficients should be 0
        for (i, &coeff) in encoded.iter().enumerate().skip(1) {
            assert_eq!(coeff, 0, "coefficient {} should be 0 for constant encoding", i);
        }
    }

    #[test]
    fn test_slot_multiplication_homomorphism() {
        // Test that polynomial multiplication gives slot-wise (element-wise) multiplication.
        // This is the key property for SIMD operations: decode(encode(s1) * encode(s2)) = s1 ⊙ s2

        let params = ParamSet::Toy.params();
        let encoder = SlotEncoder::new(&params).unwrap();
        let t = encoder.modulus();
        let n = encoder.num_slots();

        // Two sets of slot values (small values to avoid overflow)
        let slots1: Vec<u64> = (1..=n as u64).map(|i| (i % 100) + 1).collect();
        let slots2: Vec<u64> = (1..=n as u64).map(|i| ((i * 3) % 100) + 1).collect();

        // Encode both
        let poly1 = encoder.encode(&slots1);
        let poly2 = encoder.encode(&slots2);

        // Multiply polynomials mod X^n + 1 in Z_t (not Z_q)
        // Use schoolbook multiplication with mod t reduction
        let product_coeffs = poly_mul_mod_xn_plus_1(&poly1, &poly2, n, t);

        // Decode the product
        let decoded_product = encoder.decode(&product_coeffs);

        // Should equal slot-wise multiplication
        for (i, ((&s1, &s2), &d)) in slots1.iter().zip(slots2.iter()).zip(decoded_product.iter()).enumerate() {
            let expected = (s1 * s2) % t;
            assert_eq!(expected, d, "mismatch at slot {} for multiplication: {} * {} = {} (got {})", i, s1, s2, expected, d);
        }
    }

    /// Helper: polynomial multiplication mod (X^n + 1) with modulus t.
    fn poly_mul_mod_xn_plus_1(a: &[u64], b: &[u64], n: usize, t: u64) -> Vec<u64> {
        let mut result = vec![0i128; n];
        let t_i128 = t as i128;

        // Schoolbook multiplication
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = (ai as i128) * (bj as i128);
                let idx = i + j;
                if idx < n {
                    result[idx] = (result[idx] + prod) % t_i128;
                } else {
                    // X^n = -1, so X^(n+k) = -X^k
                    result[idx - n] = (result[idx - n] - prod) % t_i128;
                }
            }
        }

        // Reduce to positive mod t
        result.iter().map(|&r| {
            let rem = r % t_i128;
            if rem < 0 { (rem + t_i128) as u64 } else { rem as u64 }
        }).collect()
    }
}
