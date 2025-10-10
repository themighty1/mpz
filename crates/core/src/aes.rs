//! Fixed-key AES cipher

use aes::Aes128Enc;
use cipher::{BlockCipherEncrypt, KeyInit};
use once_cell::sync::Lazy;

use crate::Block;

/// A fixed AES key (arbitrarily chosen).
pub const FIXED_KEY: [u8; 16] = [
    69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42, 69, 42,
];

/// Fixed-key AES cipher
pub static FIXED_KEY_AES: Lazy<FixedKeyAes> = Lazy::new(|| FixedKeyAes {
    aes: Aes128Enc::new_from_slice(&FIXED_KEY).unwrap(),
    aes1: Aes128Enc::new_from_slice(&FIXED_KEY).unwrap(),
    hasher: blake3::Hasher::new(),
});

/// Fixed-key AES cipher
pub struct FixedKeyAes {
    aes: Aes128Enc,
}
use blake3::{self, hazmat::HasherExt};

impl FixedKeyAes {
    /// Create a fixed-key AES cipher with a given key.
    pub fn new(key: [u8; 16]) -> Self {
        Self {
            aes: Aes128Enc::new(&key.into()),
        }
    }

    /// Multi-instance TCCR hash. See https://eprint.iacr.org/2019/1168
    /// (Section 4.2)
    ///
    /// E(i, σ(x)) ⊕ σ(x), where E is AES modelled as an ideal cipher.
    #[inline]
    pub fn tccr_many_mi<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        // TODO: since we know that we only use 2 distinct tweaks in garbling, we
        // one need 2 AES instances.

        let aes0 = Aes128Enc::new(&tweaks[0].to_bytes().into());
        let aes1 = Aes128Enc::new(&tweaks[1].to_bytes().into());

        // sigmas for blocks with even indices 0, 2, etc
        let mut sigmas_even = blocks
            .iter_mut()
            .step_by(2)
            .map(|b| Block::sigma(*b))
            .collect::<Vec<_>>();

        // sigmas for blocks with even indices 1, 3 etc
        let mut sigmas_odd = blocks
            .iter_mut()
            .skip(1)
            .step_by(2)
            .map(|b| Block::sigma(*b))
            .collect::<Vec<_>>();

        let h_even: Vec<Block> = sigmas_even.clone();
        let h_odd: Vec<Block> = sigmas_odd.clone();

        // Encrypt potentially multiple messages in one call.
        aes0.encrypt_blocks(Block::as_array_mut_slice(&mut sigmas_even));
        aes1.encrypt_blocks(Block::as_array_mut_slice(&mut sigmas_odd));

        blocks
            .iter_mut()
            .step_by(2)
            .zip(sigmas_even.iter())
            .zip(h_even.iter())
            .for_each(|((block, sigma), h)| *block = sigma ^ h);

        blocks
            .iter_mut()
            .skip(1)
            .step_by(2)
            .zip(sigmas_odd.iter())
            .zip(h_odd.iter())
            .for_each(|((block, sigma), h)| *block = sigma ^ h);
    }

    /// Use blake3 to hash.
    #[inline]
    pub fn hash_many_blake<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        let mut input = [0u8; 32];
        blocks.iter_mut().zip(tweaks).for_each(|(block, tweak)| {
            input[0..16].copy_from_slice(block.as_bytes());
            input[16..32].copy_from_slice(tweak.as_bytes());

            let hash = blake3::hash(&input);
            *block = Block::new(hash.as_bytes()[0..16].try_into().unwrap());
        });
    }

    /// Tweakable circular correlation-robust hash function instantiated
    /// using fixed-key AES.
    ///
    /// See <https://eprint.iacr.org/2019/074> (Section 7.4)
    ///
    /// `π(π(x) ⊕ i) ⊕ π(x)`, where `π` is instantiated using fixed-key AES.
    #[inline]
    pub fn tccr(&self, tweak: Block, block: Block) -> Block {
        // let mut buf = [0u8; 32];
        // buf[0..16].copy_from_slice(tweak.as_bytes());
        // buf[16..32].copy_from_slice(block.as_bytes());
        // let hash = blake3::hash(&buf);
        // Block::new(hash.as_bytes()[0..16].try_into().unwrap())

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
