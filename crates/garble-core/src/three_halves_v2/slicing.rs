//! Wire Label Slicing for Three Halves Garbling
//!
//! This module implements the "slicing" technique from the Three Halves paper.
//! Wire labels are split into left and right halves (κ/2 bits each), allowing
//! the evaluator to compute each half using potentially different linear combinations.
//!
//! # Paper Reference
//!
//! Section 3.1 (Page 8):
//! > "We slice a wire label W into two halves W_L and W_R, each of length κ/2."
//!
//! Section 5 (Page 11):
//! > "The slicing technique means that 'half' of each wire label (i.e., κ/2 bits)
//! > can be computed from a different linear combination."
//!
//! # Layout
//!
//! A 128-bit Block is split as follows:
//! ```text
//! Block (128 bits):  [byte0, byte1, ..., byte7, byte8, ..., byte15]
//!                    [======= left =======][======= right ========]
//!                         (64 bits)              (64 bits)
//! ```
//!
//! The left half occupies bytes 0-7, the right half occupies bytes 8-15.
//! This matches the little-endian layout used throughout mpz.

use mpz_core::Block;
use std::ops::{BitXor, BitXorAssign};

/// A wire label split into left and right halves.
///
/// Each half is κ/2 = 64 bits, stored as `[u8; 8]`.
///
/// # Paper Reference
///
/// The paper uses notation like `A_L` and `A_R` for left and right halves
/// of a wire label `A`. This struct represents that split form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlicedLabel {
    /// Left half of the wire label (κ/2 bits)
    ///
    /// In evaluation equations, this appears in even rows (0, 2, 4, 6).
    pub left: [u8; 8],

    /// Right half of the wire label (κ/2 bits)
    ///
    /// In evaluation equations, this appears in odd rows (1, 3, 5, 7).
    pub right: [u8; 8],
}

impl SlicedLabel {
    /// Create a new SlicedLabel from left and right halves.
    #[inline]
    pub const fn new(left: [u8; 8], right: [u8; 8]) -> Self {
        Self { left, right }
    }

    /// Create a zero-valued SlicedLabel.
    pub const ZERO: Self = Self {
        left: [0u8; 8],
        right: [0u8; 8],
    };

    /// Split a 128-bit Block into left and right halves.
    ///
    /// # Layout
    ///
    /// ```text
    /// Block bytes:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ///               [======= left ========][========== right ===========]
    /// ```
    #[inline]
    pub fn from_block(block: Block) -> Self {
        let bytes: [u8; 16] = block.into();
        let mut left = [0u8; 8];
        let mut right = [0u8; 8];

        left.copy_from_slice(&bytes[0..8]);
        right.copy_from_slice(&bytes[8..16]);

        Self { left, right }
    }

    /// Recombine left and right halves into a 128-bit Block.
    #[inline]
    pub fn to_block(&self) -> Block {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&self.left);
        bytes[8..16].copy_from_slice(&self.right);
        Block::new(bytes)
    }

    /// Get the left half as a u64 (little-endian).
    #[inline]
    pub fn left_u64(&self) -> u64 {
        u64::from_le_bytes(self.left)
    }

    /// Get the right half as a u64 (little-endian).
    #[inline]
    pub fn right_u64(&self) -> u64 {
        u64::from_le_bytes(self.right)
    }

    /// Create from two u64 values (little-endian).
    #[inline]
    pub fn from_u64(left: u64, right: u64) -> Self {
        Self {
            left: left.to_le_bytes(),
            right: right.to_le_bytes(),
        }
    }

    /// XOR two sliced labels component-wise.
    ///
    /// This is equivalent to splitting both labels, XORing, then recombining:
    /// `(A ⊕ B).split() == A.split() ⊕ B.split()`
    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        Self {
            left: xor_arrays(&self.left, &other.left),
            right: xor_arrays(&self.right, &other.right),
        }
    }

    /// XOR this label with another in place.
    #[inline]
    pub fn xor_assign(&mut self, other: &Self) {
        xor_arrays_assign(&mut self.left, &other.left);
        xor_arrays_assign(&mut self.right, &other.right);
    }

    /// Get a specific half by index.
    ///
    /// - `half = 0` returns left
    /// - `half = 1` returns right
    ///
    /// # Panics
    ///
    /// Panics if `half > 1`.
    #[inline]
    pub fn half(&self, half: usize) -> [u8; 8] {
        match half {
            0 => self.left,
            1 => self.right,
            _ => panic!("half must be 0 or 1"),
        }
    }

    /// Get a mutable reference to a specific half by index.
    #[inline]
    pub fn half_mut(&mut self, half: usize) -> &mut [u8; 8] {
        match half {
            0 => &mut self.left,
            1 => &mut self.right,
            _ => panic!("half must be 0 or 1"),
        }
    }
}

/// XOR two 8-byte arrays.
#[inline]
fn xor_arrays(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    let mut result = [0u8; 8];
    for i in 0..8 {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// XOR-assign a 8-byte array into another.
#[inline]
fn xor_arrays_assign(a: &mut [u8; 8], b: &[u8; 8]) {
    for i in 0..8 {
        a[i] ^= b[i];
    }
}

// Implement standard XOR operators for ergonomics

impl BitXor for SlicedLabel {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(&rhs)
    }
}

impl BitXor<&SlicedLabel> for SlicedLabel {
    type Output = SlicedLabel;

    #[inline]
    fn bitxor(self, rhs: &SlicedLabel) -> Self::Output {
        self.xor(rhs)
    }
}

impl BitXor<SlicedLabel> for &SlicedLabel {
    type Output = SlicedLabel;

    #[inline]
    fn bitxor(self, rhs: SlicedLabel) -> Self::Output {
        self.xor(&rhs)
    }
}

impl BitXor<&SlicedLabel> for &SlicedLabel {
    type Output = SlicedLabel;

    #[inline]
    fn bitxor(self, rhs: &SlicedLabel) -> Self::Output {
        self.xor(rhs)
    }
}

impl BitXorAssign for SlicedLabel {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.xor_assign(&rhs);
    }
}

impl BitXorAssign<&SlicedLabel> for SlicedLabel {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &SlicedLabel) {
        self.xor_assign(rhs);
    }
}

impl From<Block> for SlicedLabel {
    #[inline]
    fn from(block: Block) -> Self {
        Self::from_block(block)
    }
}

impl From<SlicedLabel> for Block {
    #[inline]
    fn from(sliced: SlicedLabel) -> Self {
        sliced.to_block()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    /// Test 1: Round-trip conversion
    ///
    /// Splitting and recombining should give back the original block.
    #[test]
    fn test_roundtrip() {
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        // Test with random blocks
        for _ in 0..100 {
            let block = Block::random(&mut rng);
            let sliced = SlicedLabel::from_block(block);
            let recovered = sliced.to_block();
            assert_eq!(block, recovered, "Round-trip failed");
        }

        // Test with zero block
        let zero_sliced = SlicedLabel::from_block(Block::ZERO);
        assert_eq!(zero_sliced.left, [0u8; 8]);
        assert_eq!(zero_sliced.right, [0u8; 8]);
        assert_eq!(zero_sliced.to_block(), Block::ZERO);

        // Test with all-ones block
        let ones_sliced = SlicedLabel::from_block(Block::ONES);
        assert_eq!(ones_sliced.left, [0xffu8; 8]);
        assert_eq!(ones_sliced.right, [0xffu8; 8]);
        assert_eq!(ones_sliced.to_block(), Block::ONES);
    }

    /// Test 2: XOR consistency
    ///
    /// XORing blocks and then splitting should equal splitting then XORing:
    /// `(A ⊕ B).split() == A.split() ⊕ B.split()`
    #[test]
    fn test_xor_consistency() {
        let mut rng = ChaCha12Rng::seed_from_u64(123);

        for _ in 0..100 {
            let a = Block::random(&mut rng);
            let b = Block::random(&mut rng);

            // Method 1: XOR blocks, then split
            let xor_then_split = SlicedLabel::from_block(a ^ b);

            // Method 2: Split blocks, then XOR
            let a_sliced = SlicedLabel::from_block(a);
            let b_sliced = SlicedLabel::from_block(b);
            let split_then_xor = a_sliced ^ b_sliced;

            assert_eq!(
                xor_then_split, split_then_xor,
                "XOR consistency failed: (A⊕B).split() != A.split()⊕B.split()"
            );
        }
    }

    /// Test 3: XOR properties (associativity, commutativity, identity, self-inverse)
    #[test]
    fn test_xor_properties() {
        let mut rng = ChaCha12Rng::seed_from_u64(456);

        let a = SlicedLabel::from_block(Block::random(&mut rng));
        let b = SlicedLabel::from_block(Block::random(&mut rng));
        let c = SlicedLabel::from_block(Block::random(&mut rng));

        // Commutativity: A ⊕ B = B ⊕ A
        assert_eq!(a ^ b, b ^ a, "XOR should be commutative");

        // Associativity: (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)
        assert_eq!((a ^ b) ^ c, a ^ (b ^ c), "XOR should be associative");

        // Identity: A ⊕ 0 = A
        assert_eq!(a ^ SlicedLabel::ZERO, a, "Zero should be identity");

        // Self-inverse: A ⊕ A = 0
        assert_eq!(a ^ a, SlicedLabel::ZERO, "A ⊕ A should be zero");
    }

    /// Test 4: u64 conversion round-trip
    #[test]
    fn test_u64_conversion() {
        let mut rng = ChaCha12Rng::seed_from_u64(789);

        for _ in 0..100 {
            let left: u64 = rng.random();
            let right: u64 = rng.random();

            let sliced = SlicedLabel::from_u64(left, right);
            assert_eq!(sliced.left_u64(), left);
            assert_eq!(sliced.right_u64(), right);
        }
    }

    /// Test 5: Half indexing
    #[test]
    fn test_half_indexing() {
        let sliced = SlicedLabel::new([1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]);

        assert_eq!(sliced.half(0), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(sliced.half(1), [9, 10, 11, 12, 13, 14, 15, 16]);
    }

    /// Test 6: Verify layout matches Block's sigma function
    ///
    /// Block::sigma treats the first 8 bytes as x0 (left) and last 8 as x1 (right).
    /// Our slicing should match this convention.
    #[test]
    fn test_layout_matches_sigma() {
        // Create a block where left and right halves are different
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        bytes[8..16].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        let block = Block::new(bytes);

        let sliced = SlicedLabel::from_block(block);

        // Verify left is bytes 0-7 and right is bytes 8-15
        assert_eq!(sliced.left, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(sliced.right, [9, 10, 11, 12, 13, 14, 15, 16]);

        // Cross-check with direct u64 interpretation from bytes
        let block_bytes = block.to_bytes();
        let left_u64 = u64::from_le_bytes(block_bytes[0..8].try_into().unwrap());
        let right_u64 = u64::from_le_bytes(block_bytes[8..16].try_into().unwrap());

        assert_eq!(sliced.left_u64(), left_u64, "Left should match first u64");
        assert_eq!(
            sliced.right_u64(),
            right_u64,
            "Right should match second u64"
        );
    }

    /// Test 7: From/Into trait implementations
    #[test]
    fn test_from_into_traits() {
        let mut rng = ChaCha12Rng::seed_from_u64(999);
        let block = Block::random(&mut rng);

        // Test From<Block> for SlicedLabel
        let sliced: SlicedLabel = block.into();
        assert_eq!(sliced, SlicedLabel::from_block(block));

        // Test From<SlicedLabel> for Block
        let recovered: Block = sliced.into();
        assert_eq!(recovered, block);
    }

    /// Test 8: XOR-assign operations
    #[test]
    fn test_xor_assign() {
        let mut rng = ChaCha12Rng::seed_from_u64(111);

        let a = SlicedLabel::from_block(Block::random(&mut rng));
        let b = SlicedLabel::from_block(Block::random(&mut rng));

        // Test owned XOR-assign
        let mut c = a;
        c ^= b;
        assert_eq!(c, a ^ b);

        // Test reference XOR-assign
        let mut d = a;
        d ^= &b;
        assert_eq!(d, a ^ b);
    }
}
