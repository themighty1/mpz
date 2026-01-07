//! This module implements the Mersenne prime field F_{2^61-1}.
//!
//! The Mersenne prime M61 = 2^61 - 1 = 2305843009213693951 is particularly
//! efficient for modular arithmetic because reduction can be done with
//! simple bit operations: `x mod p = (x & p) + (x >> 61)`.
//!
//! This field is NTT-friendly with primitive roots of unity for power-of-2
//! sizes up to 2^60.

use std::ops::{Add, Mul, Neg, Sub};

use hybrid_array::Array;
use itybity::{BitLength, FromBitIterator, GetBit, Lsb0, Msb0};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use typenum::{U61, U8};

use crate::{Field, FieldError};

/// The Mersenne prime 2^61 - 1.
pub const M61: u64 = (1u64 << 61) - 1;

/// A field element in F_{2^61-1}.
///
/// Elements are stored in canonical form in the range [0, M61).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(into = "[u8; 8]")]
#[serde(try_from = "[u8; 8]")]
pub struct M61Field(u64);

opaque_debug::implement!(M61Field);

impl M61Field {
    /// Creates a new field element from a u64.
    ///
    /// The value is reduced modulo M61 if necessary.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(Self::reduce(value as u128))
    }

    /// Returns the inner u64 value.
    #[inline]
    pub const fn inner(self) -> u64 {
        self.0
    }

    /// Reduces a u128 value modulo M61 using the Mersenne prime property.
    ///
    /// For a Mersenne prime p = 2^k - 1, we have:
    /// x mod p = (x & p) + (x >> k) (with possible final reduction)
    #[inline]
    const fn reduce(x: u128) -> u64 {
        // First reduction: split into low 61 bits and high bits
        let low = (x as u64) & M61;
        let high = (x >> 61) as u64;
        let sum = low + high;

        // Second reduction if needed (sum could be up to ~2*M61)
        let low2 = sum & M61;
        let high2 = sum >> 61;
        let result = low2 + high2;

        // Final check: if result == M61, reduce to 0
        if result >= M61 {
            result - M61
        } else {
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
        let mut r: i128 = M61 as i128;
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

        // r should be 1 (gcd), t is the inverse
        debug_assert_eq!(r, 1, "GCD should be 1 for non-zero element");

        // Convert negative result to positive
        let inv = if t < 0 { t + M61 as i128 } else { t };

        Some(Self(inv as u64))
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
    /// For M61, the multiplicative group has order M61 - 1 = 2^61 - 2 = 2 * (2^60 - 1).
    /// Since 2^60 - 1 is odd, the only power of 2 dividing the group order is 2^1.
    /// This means M61 only supports 2nd roots of unity (1 and -1).
    ///
    /// Returns None if k > 1.
    pub fn primitive_root_of_unity(k: u32) -> Option<Self> {
        // M61 - 1 = 2 * (2^60 - 1), where 2^60 - 1 is odd
        // So only 2^0 = 1 and 2^1 = 2 divide the group order
        match k {
            0 => Some(Self::one()),
            1 => Some(Self(M61 - 1)), // -1 mod M61
            _ => None,
        }
    }

    /// Returns the maximum NTT size supported (as a power of 2).
    ///
    /// For M61, this is only 2^1 = 2 since M61 - 1 = 2 * (odd).
    /// M61 is NOT NTT-friendly for larger transforms.
    pub const fn max_ntt_log_size() -> u32 {
        1
    }
}

impl From<M61Field> for [u8; 8] {
    fn from(value: M61Field) -> Self {
        value.0.to_le_bytes()
    }
}

impl TryFrom<[u8; 8]> for M61Field {
    type Error = FieldError;

    fn try_from(value: [u8; 8]) -> Result<Self, Self::Error> {
        let n = u64::from_le_bytes(value);
        if n >= M61 {
            return Err(FieldError(Box::new(M61Error::OutOfRange(n))));
        }
        Ok(Self(n))
    }
}

impl TryFrom<Array<u8, U8>> for M61Field {
    type Error = FieldError;

    fn try_from(value: Array<u8, U8>) -> Result<Self, Self::Error> {
        let inner: [u8; 8] = value.into();
        M61Field::try_from(inner)
    }
}

impl Distribution<M61Field> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> M61Field {
        // Sample uniformly from [0, M61)
        // Use rejection sampling to avoid bias
        loop {
            let value = rng.next_u64() & M61; // Mask to 61 bits
            if value < M61 {
                return M61Field(value);
            }
        }
    }
}

impl Add for M61Field {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 as u128 + rhs.0 as u128;
        Self(Self::reduce(sum))
    }
}

impl Sub for M61Field {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        // Add M61 to avoid underflow, then reduce
        let diff = self.0 as u128 + M61 as u128 - rhs.0 as u128;
        Self(Self::reduce(diff))
    }
}

impl Mul for M61Field {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let prod = self.0 as u128 * rhs.0 as u128;
        Self(Self::reduce(prod))
    }
}

impl Neg for M61Field {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            self
        } else {
            Self(M61 - self.0)
        }
    }
}

impl Field for M61Field {
    type BitSize = U61;
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
        if rhs >= 61 {
            // 2^61 = 1 mod M61, so 2^k = 2^(k mod 61)
            let exp = rhs % 61;
            Self(1u64 << exp)
        } else {
            Self(1u64 << rhs)
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

impl BitLength for M61Field {
    const BITS: usize = 61;
}

impl GetBit<Lsb0> for M61Field {
    #[inline]
    fn get_bit(&self, index: usize) -> bool {
        if index >= 61 {
            false
        } else {
            (self.0 >> index) & 1 == 1
        }
    }
}

impl GetBit<Msb0> for M61Field {
    #[inline]
    fn get_bit(&self, index: usize) -> bool {
        if index >= 61 {
            false
        } else {
            (self.0 >> (60 - index)) & 1 == 1
        }
    }
}

impl FromBitIterator for M61Field {
    fn from_lsb0_iter(iter: impl IntoIterator<Item = bool>) -> Self {
        let mut value = 0u64;
        for (i, bit) in iter.into_iter().enumerate().take(61) {
            if bit {
                value |= 1u64 << i;
            }
        }
        Self::new(value)
    }

    fn from_msb0_iter(iter: impl IntoIterator<Item = bool>) -> Self {
        let mut value = 0u64;
        for (i, bit) in iter.into_iter().enumerate().take(61) {
            if bit {
                value |= 1u64 << (60 - i);
            }
        }
        Self::new(value)
    }
}

/// Error type for M61 field operations.
#[derive(Debug, thiserror::Error)]
pub enum M61Error {
    /// Value is out of range for the field.
    #[error("value {0} is out of range for M61 field (must be < 2^61-1)")]
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
    fn test_m61_basic() {
        test_field_basic::<M61Field>();
        assert_eq!(M61Field::new(0), M61Field::zero());
        assert_eq!(M61Field::new(1), M61Field::one());
    }

    #[test]
    fn test_m61_compute_product_repeated() {
        test_field_compute_product_repeated::<M61Field>();
    }

    #[test]
    fn test_m61_bit_ops() {
        test_field_bit_ops_lsb0::<M61Field>();
        test_field_bit_ops_msb0::<M61Field>();
    }

    #[test]
    fn test_m61_serialize() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..32 {
            let a: M61Field = rng.random();
            let bytes: [u8; 8] = a.into();
            let b = M61Field::try_from(bytes).unwrap();

            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_m61_constants() {
        assert_eq!(M61, (1u64 << 61) - 1);
        assert_eq!(M61, 2305843009213693951);
    }

    #[test]
    fn test_m61_reduction() {
        // Test that M61 reduces to 0
        assert_eq!(M61Field::new(M61), M61Field::zero());

        // Test that M61 + 1 reduces to 1
        assert_eq!(M61Field::new(M61 + 1), M61Field::one());

        // Test large values
        let large = M61Field::new(u64::MAX);
        assert!(large.inner() < M61);
    }

    #[test]
    fn test_m61_arithmetic() {
        let a = M61Field::new(12345);
        let b = M61Field::new(67890);

        // Addition
        assert_eq!((a + b).inner(), 12345 + 67890);

        // Subtraction
        assert_eq!((b - a).inner(), 67890 - 12345);

        // Subtraction with wrap
        let diff = a - b;
        assert_eq!((diff + b).inner(), a.inner());

        // Multiplication
        assert_eq!((a * b).inner(), (12345u128 * 67890) as u64 % M61);

        // Negation
        assert_eq!((a + (-a)).inner(), 0);
    }

    #[test]
    fn test_m61_inverse() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: M61Field = rng.random();
            if a.inner() != 0 {
                let inv = a.inverse().unwrap();
                assert_eq!((a * inv).inner(), 1);
            }
        }

        // Zero has no inverse
        assert!(M61Field::zero().inverse().is_none());
    }

    #[test]
    fn test_m61_two_pow() {
        assert_eq!(M61Field::two_pow(0), M61Field::one());
        assert_eq!(M61Field::two_pow(1).inner(), 2);
        assert_eq!(M61Field::two_pow(10).inner(), 1024);

        // 2^61 = 1 mod M61 (since M61 = 2^61 - 1)
        assert_eq!(M61Field::two_pow(61), M61Field::one());

        // 2^62 = 2 mod M61
        assert_eq!(M61Field::two_pow(62), M61Field::new(2));
    }

    #[test]
    fn test_m61_pow() {
        let base = M61Field::new(3);

        assert_eq!(base.pow(0), M61Field::one());
        assert_eq!(base.pow(1), base);
        assert_eq!(base.pow(2), M61Field::new(9));
        assert_eq!(base.pow(3), M61Field::new(27));

        // Fermat's little theorem: a^(p-1) = 1 mod p for a != 0
        let mut rng = Prg::from_seed(Block::ZERO);
        let a: M61Field = rng.random();
        if a.inner() != 0 {
            assert_eq!(a.pow(M61 - 1), M61Field::one());
        }
    }

    #[test]
    fn test_m61_primitive_root() {
        // M61 is NOT NTT-friendly: M61 - 1 = 2 * (2^60 - 1) where 2^60 - 1 is odd
        // So only 2^0 and 2^1 roots of unity exist

        // 2^0 = 1 root of unity is 1
        let omega0 = M61Field::primitive_root_of_unity(0).unwrap();
        assert_eq!(omega0, M61Field::one());

        // 2^1 = 2 root of unity is -1
        let omega1 = M61Field::primitive_root_of_unity(1).unwrap();
        assert_eq!(omega1, -M61Field::one());
        assert_eq!(omega1.pow(2), M61Field::one()); // (-1)^2 = 1

        // No higher roots of unity exist
        assert!(M61Field::primitive_root_of_unity(2).is_none());
        assert!(M61Field::primitive_root_of_unity(10).is_none());
        assert!(M61Field::primitive_root_of_unity(60).is_none());

        // max_ntt_log_size is 1
        assert_eq!(M61Field::max_ntt_log_size(), 1);
    }

    #[test]
    fn test_m61_distributivity() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: M61Field = rng.random();
            let b: M61Field = rng.random();
            let c: M61Field = rng.random();

            // a * (b + c) = a * b + a * c
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn test_m61_associativity() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: M61Field = rng.random();
            let b: M61Field = rng.random();
            let c: M61Field = rng.random();

            // (a + b) + c = a + (b + c)
            assert_eq!((a + b) + c, a + (b + c));

            // (a * b) * c = a * (b * c)
            assert_eq!((a * b) * c, a * (b * c));
        }
    }

    #[test]
    fn test_m61_commutativity() {
        let mut rng = Prg::from_seed(Block::ZERO);

        for _ in 0..100 {
            let a: M61Field = rng.random();
            let b: M61Field = rng.random();

            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
        }
    }

    #[test]
    fn test_m61_edge_cases() {
        let zero = M61Field::zero();
        let one = M61Field::one();
        let max = M61Field::new(M61 - 1);

        // Operations with zero
        assert_eq!(zero + zero, zero);
        assert_eq!(zero * one, zero);
        assert_eq!(max + zero, max);

        // Operations with one
        assert_eq!(one * one, one);
        assert_eq!(max * one, max);

        // Max value operations
        assert_eq!(max + one, zero);
        assert_eq!(max + max, M61Field::new(M61 - 2));
    }
}
