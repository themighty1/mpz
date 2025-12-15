//! Tweakable Circular Correlation-Robust (TCCR) hash functions.

use crate::{Block, aes::FIXED_KEY_AES};
use std::fmt::Debug;

/// Trait for TCCR hash implementations.
pub trait TccrHash: Send + Sync + Debug {
    /// Compute TCCR hash of a single block with a tweak.
    fn tccr(&self, tweak: Block, block: Block) -> Block;
    /// Compute TCCR hash of multiple blocks with corresponding tweaks.
    fn tccr_many<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]);
}

/// Blake3-based TCCR implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Blake3Tccr;

impl TccrHash for Blake3Tccr {
    fn tccr(&self, tweak: Block, block: Block) -> Block {
        let mut input = [0u8; 32];
        input[0..16].copy_from_slice(block.as_bytes());
        input[16..32].copy_from_slice(tweak.as_bytes());
        let hash = blake3::hash(&input);
        Block::new(hash.as_bytes()[0..16].try_into().unwrap())
    }

    fn tccr_many<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        let mut input = [0u8; 32];
        blocks.iter_mut().zip(tweaks).for_each(|(block, tweak)| {
            input[0..16].copy_from_slice(block.as_bytes());
            input[16..32].copy_from_slice(tweak.as_bytes());

            let hash = blake3::hash(&input);
            *block = Block::new(hash.as_bytes()[0..16].try_into().unwrap());
        });
    }
}

/// Enum wrapper for TCCR implementations, enabling runtime selection without
/// dynamic dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub enum TccrImpl {
    /// AES-based TCCR (default).
    #[default]
    Aes,
    /// Blake3-based TCCR.
    Blake3,
}

impl TccrImpl {
    /// Create a `TccrImpl` from a string identifier.
    ///
    /// # Arguments
    /// * `s` - "blake3" for Blake3, anything else defaults to AES.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "blake3" => Self::Blake3,
            _ => Self::Aes,
        }
    }
}

impl TccrHash for TccrImpl {
    #[inline]
    fn tccr(&self, tweak: Block, block: Block) -> Block {
        match self {
            Self::Aes => FIXED_KEY_AES.tccr(tweak, block),
            Self::Blake3 => Blake3Tccr.tccr(tweak, block),
        }
    }

    #[inline]
    fn tccr_many<const N: usize>(&self, tweaks: &[Block; N], blocks: &mut [Block; N]) {
        match self {
            Self::Aes => FIXED_KEY_AES.tccr_many(tweaks, blocks),
            Self::Blake3 => Blake3Tccr.tccr_many(tweaks, blocks),
        }
    }
}
