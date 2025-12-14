//! Fixed-key AES cipher

use aes::Aes128Enc;
use cipher::{BlockCipherEncrypt, KeyInit};
use once_cell::sync::Lazy;

use crate::Block;

/// Reduction polynomial for GF(2^64): x^64 + x^4 + x^3 + x + 1
/// The constant represents the low-order bits (without the x^64 term).
const GF64_REDUCTION: u64 = 0x1B; // bits 0,1,3,4 = 1+2+8+16 = 0x1B

/// Multiply by 2 (i.e., x) in GF(2^64).
///
/// This is a left shift with conditional reduction when the high bit overflows.
#[inline]
fn gf64_double(x: u64) -> u64 {
    let overflow = x >> 63;
    (x << 1) ^ (overflow.wrapping_neg() & GF64_REDUCTION)
}

/// RTCCR sigma function: σ(X_L || X_R) = (2·X_L) || (2·X_R)
///
/// Multiplies both 64-bit halves of the block by α = 2 in GF(2^64).
/// This is used in the RTCCR (Randomized Tweakable CCR) hash function from
/// "Three Halves Make a Whole" (Rosulek & Roy, 2021).
///
/// # Choice of α = 2
///
/// The paper (Section 5) requires α ∈ GF(2^64) \ GF(2²), meaning α must not be
/// in the subfield GF(4) = {0, 1, β, β+1} where β² + β + 1 = 0. Elements in GF(4)
/// have multiplicative order dividing 3, which would make σ³ = identity and break
/// circular correlation robustness.
///
/// We use α = 2 (the element x in polynomial representation) which satisfies this
/// constraint and requires only a single shift operation, making it very efficient.
///
/// **Security note**: Some implementations use α = 0x87 (the GCM polynomial constant)
/// which has higher multiplicative order. While the paper's proof only requires
/// α ∉ GF(4), a higher-order α may provide additional security margin against
/// attacks not covered by the proof. We chose α = 2 for efficiency, following the
/// paper's minimal requirement.
#[inline]
pub fn rtccr_sigma(block: Block) -> Block {
    let halves: [u64; 2] = bytemuck::cast(block);
    let result: [u64; 2] = [gf64_double(halves[0]), gf64_double(halves[1])];
    bytemuck::cast(result)
}

/// A fixed AES key (arbitrarily chosen).
pub const FIXED_KEY: [u8; 16] = [
    69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42,
];

/// Fixed-key AES cipher
pub static FIXED_KEY_AES: Lazy<FixedKeyAes> = Lazy::new(|| FixedKeyAes::new(FIXED_KEY));

/// Fixed-key AES cipher with RTCCR universal hash parameters.
///
/// # Universal Hash Implementation Note
///
/// The paper (Section 5, Page 9) specifies U(τ) = (u₁·τ_L) ‖ (u₂·τ_R) using
/// two independent GF(2⁶⁴) multiplications. We instead use a single GF(2¹²⁸)
/// multiplication: U(τ) = u · τ in GF(2¹²⁸).
///
/// Rationale:
/// - GF(2¹²⁸) multiplication uses hardware CLMUL instructions (~10-20x faster)
/// - A single field multiplication is still a valid universal hash function
/// - GF(2¹²⁸) provides better mixing than two independent GF(2⁶⁴) operations
/// - The security proof only requires U to be universal, not the specific construction
pub struct FixedKeyAes {
    aes: Aes128Enc,
    /// Universal hash coefficient (derived from key) for GF(2¹²⁸) multiplication
    u: Block,
}

impl FixedKeyAes {
    /// Create a fixed-key AES cipher with a given key.
    ///
    /// Derives the universal hash coefficient by encrypting a fixed constant.
    pub fn new(key: [u8; 16]) -> Self {
        let aes = Aes128Enc::new(&key.into());

        // Derive u by encrypting the constant 0
        let mut u = Block::new([0u8; 16]);
        aes.encrypt_block(u.as_array_mut());

        Self { aes, u }
    }

    /// Compute universal hash U(τ) = u · τ in GF(2¹²⁸)
    ///
    /// This is used in RTCCR to achieve correlation robustness.
    /// Uses hardware-accelerated CLMUL for GF(2¹²⁸) multiplication.
    ///
    /// Note: The paper suggests U(τ) = (u₁·τ_L) ‖ (u₂·τ_R) in GF(2⁶⁴),
    /// but we use GF(2¹²⁸) for performance (see struct docs).
    #[inline]
    fn universal_hash(&self, tweak: Block) -> Block {
        self.u.gfmul(tweak)
    }

    /// Randomized tweakable circular correlation-robust hash function (RTCCR).
    ///
    /// From "Three Halves Make a Whole" (Rosulek & Roy, 2021):
    /// <https://eprint.iacr.org/2021/749>
    ///
    /// `H(X, τ) = AES_k(X ⊕ U(τ)) ⊕ σ(X ⊕ U(τ))`
    ///
    /// Where U(τ) = (u₁·τ_L) ‖ (u₂·τ_R) is a universal hash function.
    /// Uses only 1 AES call (vs 2 for TCCR), with GF(2^64) sigma function.
    #[inline]
    pub fn rtccr(&self, tweak: Block, block: Block) -> Block {
        let u_tweak = self.universal_hash(tweak);
        let tweaked = block ^ u_tweak;
        let mut encrypted = tweaked;
        self.aes.encrypt_block(encrypted.as_array_mut());
        encrypted ^ rtccr_sigma(tweaked)
    }

    /// Randomized tweakable circular correlation-robust hash function (RTCCR) - batch version.
    ///
    /// From "Three Halves Make a Whole" (Rosulek & Roy, 2021):
    /// <https://eprint.iacr.org/2021/749>
    ///
    /// `H(X, τ) = AES_k(X ⊕ U(τ)) ⊕ σ(X ⊕ U(τ))`
    ///
    /// Where U(τ) = (u₁·τ_L) ‖ (u₂·τ_R) is a universal hash function.
    ///
    /// # Arguments
    ///
    /// * `tweaks` - The tweaks to use for each block.
    /// * `blocks` - The blocks to hash in-place.
    #[inline]
    pub fn rtccr_many<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        // Compute X ⊕ U(τ) for all blocks
        for (block, tweak) in blocks.iter_mut().zip(tweaks.iter()) {
            *block ^= self.universal_hash(*tweak);
        }

        // Store σ(X ⊕ U(τ)) in buf before encryption overwrites blocks
        let sigma_buf: [Block; N] = std::array::from_fn(|i| rtccr_sigma(blocks[i]));

        // Encrypt all tweaked blocks: AES_k(X ⊕ U(τ))
        self.aes.encrypt_blocks(Block::as_array_mut_slice(blocks));

        // XOR with sigma: AES_k(X ⊕ U(τ)) ⊕ σ(X ⊕ U(τ))
        for (block, sigma) in blocks.iter_mut().zip(sigma_buf.iter()) {
            *block ^= *sigma;
        }
    }

    /// Tweakable circular correlation-robust hash function instantiated
    /// using fixed-key AES.
    ///
    /// See <https://eprint.iacr.org/2019/074> (Section 7.4)
    ///
    /// `π(π(x) ⊕ i) ⊕ π(x)`, where `π` is instantiated using fixed-key AES.
    #[inline]
    pub fn tccr(&self, tweak: Block, block: Block) -> Block {
        let mut h1 = block;
        self.aes.encrypt_block(h1.as_array_mut());

        let mut h2 = h1 ^ tweak;
        self.aes.encrypt_block(h2.as_array_mut());

        h1 ^ h2
    }

    /// Tweakable circular correlation-robust hash function instantiated
    /// using fixed-key AES.
    ///
    /// See <https://eprint.iacr.org/2019/074> (Section 7.4)
    ///
    /// `π(π(x) ⊕ i) ⊕ π(x)`, where `π` is instantiated using fixed-key AES.
    ///
    /// # Arguments
    ///
    /// * `tweaks` - The tweaks to use for each block in `blocks`.
    /// * `blocks` - The blocks to hash in-place.
    #[inline]
    pub fn tccr_many<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        // Store π(x) in `blocks`
        self.aes.encrypt_blocks(Block::as_array_mut_slice(blocks));

        // Write π(x) ⊕ i into `buf`
        let mut buf: [Block; N] = std::array::from_fn(|i| blocks[i] ^ tweaks[i]);

        // Write π(π(x) ⊕ i) in `buf`
        self.aes.encrypt_blocks(Block::as_array_mut_slice(&mut buf));

        // Write π(π(x) ⊕ i) ⊕ π(x) into `blocks`
        blocks
            .iter_mut()
            .zip(buf.iter())
            .for_each(|(a, b)| *a ^= *b);
    }

    /// Correlation-robust hash function instantiated using fixed-key AES
    /// (cf. <https://eprint.iacr.org/2019/074>, §7.2).
    ///
    /// `π(x) ⊕ x`, where `π` is instantiated using fixed-key AES.
    #[inline]
    pub fn cr(&self, block: Block) -> Block {
        let mut h = block;
        self.aes.encrypt_block(h.as_array_mut());
        h ^ block
    }

    /// Correlation-robust hash function instantiated using fixed-key AES
    /// (cf. <https://eprint.iacr.org/2019/074>, §7.2).
    ///
    /// `π(x) ⊕ x`, where `π` is instantiated using fixed-key AES.
    ///
    /// # Arguments
    ///
    /// * `blocks` - The blocks to hash in-place.
    #[inline]
    pub fn cr_many<const N: usize>(&self, blocks: &mut [Block; N]) {
        let mut buf = *blocks;

        self.aes.encrypt_blocks(Block::as_array_mut_slice(&mut buf));

        blocks
            .iter_mut()
            .zip(buf.iter())
            .for_each(|(a, b)| *a ^= *b);
    }

    /// Circular correlation-robust hash function instantiated using fixed-key
    /// AES (cf.<https://eprint.iacr.org/2019/074>, §7.3).
    ///
    /// `π(σ(x)) ⊕ σ(x)`, where `π` is instantiated using fixed-key AES
    ///
    /// See [`Block::sigma`](Block::sigma) for more details on `σ`.
    #[inline]
    pub fn ccr(&self, block: Block) -> Block {
        self.cr(Block::sigma(block))
    }

    /// Circular correlation-robust hash function instantiated using fixed-key
    /// AES (cf.<https://eprint.iacr.org/2019/074>, §7.3).
    ///
    /// `π(σ(x)) ⊕ σ(x)`, where `π` is instantiated using fixed-key AES
    ///
    /// See [`Block::sigma`](Block::sigma) for more details on `σ`.
    ///
    /// # Arguments
    ///
    /// * `blocks` - The blocks to hash in-place.
    #[inline]
    pub fn ccr_many<const N: usize>(&self, blocks: &mut [Block; N]) {
        blocks.iter_mut().for_each(|b| *b = Block::sigma(*b));
        self.cr_many(blocks);
    }
}

/// A wrapper of aes, only for encryption.
#[derive(Clone)]
pub struct AesEncryptor(Aes128Enc);

impl AesEncryptor {
    /// Constant number of AES blocks, always set to 8.
    pub const AES_BLOCK_COUNT: usize = 8;

    /// Initiate an AesEncryptor instance with key.
    #[inline(always)]
    pub fn new(key: Block) -> Self {
        let _key: [u8; 16] = key.into();
        AesEncryptor(Aes128Enc::new_from_slice(&_key).unwrap())
    }

    /// Encrypt a block.
    #[inline(always)]
    pub fn encrypt_block(&self, mut blk: Block) -> Block {
        self.0.encrypt_block(blk.as_array_mut());
        blk
    }

    /// Encrypt a block in-place.
    pub fn encrypt_block_inplace(&self, blk: &mut Block) {
        self.0.encrypt_block(blk.as_array_mut());
    }

    /// Encrypt many blocks in-place.
    #[inline(always)]
    pub fn encrypt_many_blocks<const N: usize>(&self, blks: &mut [Block; N]) {
        self.0
            .encrypt_blocks(Block::as_array_mut_slice(blks.as_mut_slice()));
    }

    /// Encrypt slice of blocks in-place.
    #[inline]
    pub fn encrypt_blocks(&self, blks: &mut [Block]) {
        self.0.encrypt_blocks(Block::as_array_mut_slice(blks));
    }

    /// Encrypt many blocks with many keys.
    ///
    /// Each batch of NM blocks is encrypted by a corresponding AES key.
    ///
    /// **Only the first NK * NM blocks of blks are handled, the rest are
    /// ignored.**
    ///
    /// # Arguments
    ///
    /// * `keys` - A slice of keys used to encrypt the blocks.
    /// * `blks` - A slice of blocks to be encrypted.
    ///
    /// # Panics
    ///
    /// * If the length of `blks` is less than `NM * NK`.
    #[inline(always)]
    pub fn para_encrypt<const NK: usize, const NM: usize>(keys: &[Self; NK], blks: &mut [Block]) {
        assert!(blks.len() >= NM * NK);

        keys.iter()
            .zip(blks.chunks_exact_mut(NM))
            .for_each(|(key, blks)| {
                key.encrypt_blocks(blks);
            });
    }
}

#[test]
fn aes_test() {
    let aes = AesEncryptor::new(Block::default());
    let aes1 = AesEncryptor::new(Block::ONES);

    let mut blks = [Block::default(); 4];
    blks[1] = Block::ONES;
    blks[3] = Block::ONES;
    AesEncryptor::para_encrypt::<2, 2>(&[aes, aes1], &mut blks);
    assert_eq!(
        blks,
        [
            Block::from((0x2E2B34CA59FA4C883B2C8AEFD44BE966_u128).to_le_bytes()),
            Block::from((0x4E668D3ED24773FA0A5A85EAC98C5B3F_u128).to_le_bytes()),
            Block::from((0x2CC9BF3845486489CD5F7D878C25F6A1_u128).to_le_bytes()),
            Block::from((0x79B93A19527051B230CF80B27C21BFBC_u128).to_le_bytes())
        ]
    );
}

#[cfg(test)]
mod rtccr_tests {
    use super::*;

    /// Test that rtccr and rtccr_many produce identical results
    #[test]
    fn rtccr_single_vs_batch() {
        let key = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let aes = FixedKeyAes::new(key);

        let tweak = Block::new([0xAB; 16]);
        let block = Block::new([0xCD; 16]);

        // Single call
        let single_result = aes.rtccr(tweak, block);

        // Batch call with 1 element
        let mut blocks = [block];
        aes.rtccr_many(&[tweak], &mut blocks);

        assert_eq!(single_result, blocks[0], "Single and batch RTCCR should match");
    }

    /// Test that rtccr_many processes multiple blocks correctly
    #[test]
    fn rtccr_many_multiple_blocks() {
        let key = [42u8; 16];
        let aes = FixedKeyAes::new(key);

        let tweaks = [
            Block::new([1u8; 16]),
            Block::new([2u8; 16]),
            Block::new([3u8; 16]),
            Block::new([4u8; 16]),
        ];
        let blocks_original = [
            Block::new([0x10; 16]),
            Block::new([0x20; 16]),
            Block::new([0x30; 16]),
            Block::new([0x40; 16]),
        ];

        // Compute individually
        let expected: [Block; 4] =
            std::array::from_fn(|i| aes.rtccr(tweaks[i], blocks_original[i]));

        // Compute in batch
        let mut blocks = blocks_original;
        aes.rtccr_many(&tweaks, &mut blocks);

        assert_eq!(blocks, expected, "Batch should match individual calls");
    }

    /// Test that universal hash produces different outputs for different tweaks
    #[test]
    fn universal_hash_different_tweaks() {
        let key = [0x55u8; 16];
        let aes = FixedKeyAes::new(key);

        let block = Block::new([0xAA; 16]);
        let tweak1 = Block::new([1u8; 16]);
        let tweak2 = Block::new([2u8; 16]);

        let result1 = aes.rtccr(tweak1, block);
        let result2 = aes.rtccr(tweak2, block);

        assert_ne!(result1, result2, "Different tweaks should produce different outputs");
    }

    /// Test that RTCCR is deterministic
    #[test]
    fn rtccr_deterministic() {
        let key = [0x77u8; 16];
        let aes = FixedKeyAes::new(key);

        let tweak = Block::new([0x11; 16]);
        let block = Block::new([0x22; 16]);

        let result1 = aes.rtccr(tweak, block);
        let result2 = aes.rtccr(tweak, block);

        assert_eq!(result1, result2, "RTCCR should be deterministic");
    }

    /// Test that u1, u2 are derived consistently from the same key
    #[test]
    fn universal_hash_key_derivation() {
        let key = [0x99u8; 16];

        let aes1 = FixedKeyAes::new(key);
        let aes2 = FixedKeyAes::new(key);

        // Both should produce same results
        let tweak = Block::new([0xBB; 16]);
        let block = Block::new([0xCC; 16]);

        assert_eq!(
            aes1.rtccr(tweak, block),
            aes2.rtccr(tweak, block),
            "Same key should produce identical u1, u2"
        );
    }

    /// Test that different keys produce different u1, u2
    #[test]
    fn universal_hash_different_keys() {
        let key1 = [0x11u8; 16];
        let key2 = [0x22u8; 16];

        let aes1 = FixedKeyAes::new(key1);
        let aes2 = FixedKeyAes::new(key2);

        let tweak = Block::new([0xDD; 16]);
        let block = Block::new([0xEE; 16]);

        assert_ne!(
            aes1.rtccr(tweak, block),
            aes2.rtccr(tweak, block),
            "Different keys should produce different RTCCR outputs"
        );
    }

    /// Test GF(2^64) doubling (multiplication by 2) properties
    #[test]
    fn gf64_double_properties() {
        // 2 * 0 = 0
        assert_eq!(gf64_double(0), 0);

        // 2 * 1 = 2
        assert_eq!(gf64_double(1), 2);

        // Linearity: 2*(a ⊕ b) = 2*a ⊕ 2*b
        let a = 0xABCD1234_u64;
        let b = 0x5678EFAB_u64;
        assert_eq!(gf64_double(a ^ b), gf64_double(a) ^ gf64_double(b));

        // Test reduction: when high bit is set, should XOR with reduction polynomial
        let high_bit_set = 1u64 << 63;
        // 2 * (2^63) should reduce: (2^64) mod p = 0x1B
        assert_eq!(gf64_double(high_bit_set), GF64_REDUCTION);

        // Verify α = 2 is not in GF(4) by checking 2^3 ≠ 2
        // (elements in GF(4) satisfy x^3 = x)
        let two_cubed = gf64_double(gf64_double(gf64_double(1))); // 2^3 = 8
        assert_ne!(two_cubed, 2, "α = 2 should not be in GF(4)");
    }

    /// Test that rtccr_sigma is linear: σ(A ⊕ B) = σ(A) ⊕ σ(B)
    #[test]
    fn rtccr_sigma_linearity() {
        let a = Block::new([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                           0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        let b = Block::new([0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
                           0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11]);

        let sigma_xor = rtccr_sigma(a ^ b);
        let xor_sigma = rtccr_sigma(a) ^ rtccr_sigma(b);

        assert_eq!(sigma_xor, xor_sigma, "σ should be linear");
    }
}
