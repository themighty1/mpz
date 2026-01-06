//! Block VOLE (Vector Oblivious Linear Evaluation) functionality.
//!
//! This module implements the Block VOLE functionality F_{bVOLE} from
//! "Authenticated Garbling from Simple Correlations" (Figure 7).
//!
//! Block VOLE is a collection of VOLE instances where party A uses the same
//! inputs across all VOLE calls. This enables efficient batched computation.
//!
//! # Variants
//! - **Block Subfield VOLE**: F = F_2, E = F_{2^ρ} (A's input is bits)
//! - **Block VOLE**: F = F_{2^ρ}, E = F_{2^ρ} (A's input is field elements)
//!
//! # Protocol
//! 1. B chooses β_1, ..., β_k ∈ E and sends to F_{bVOLE}
//! 2. F_{bVOLE} chooses vectors b_1, ..., b_k ∈ E^n and sends to A
//! 3. A chooses vector a ∈ F^n and sends to F_{bVOLE}
//! 4. For i = 1,...,k: F_{bVOLE} computes v_i = a·β_i + b_i and sends to B
//!
//! The key property is that A uses the SAME vector `a` for ALL k VOLE instances.

use mpz_core::Block;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Error types for Block VOLE operations.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum BlockVoleError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid k parameter: {0}")]
    InvalidK(usize),
}

/// Output for the sender (Party A) in Block VOLE.
///
/// A receives random vectors b_1, ..., b_k that will be used as masks.
#[derive(Debug, Clone)]
pub struct BlockVoleSenderOutput {
    /// Random vectors b_i ∈ F_{2^ρ}^n for i = 1..k
    /// Stored as b[i][j] where i is the VOLE instance and j is the vector index
    pub b: Vec<Vec<Block>>,
}

/// Output for the receiver (Party B) in Block VOLE.
///
/// B receives v_i = a·β_i + b_i for each VOLE instance.
#[derive(Debug, Clone)]
pub struct BlockVoleReceiverOutput {
    /// Computed vectors v_i = a·β_i + b_i for i = 1..k
    pub v: Vec<Vec<Block>>,
}

/// Ideal Block VOLE functionality for testing.
///
/// This is an ideal/trusted implementation that both parties would
/// interact with in the ideal world. In practice, this would be
/// replaced with a real protocol based on OT extension.
#[derive(Debug)]
pub struct IdealBlockVole {
    rng: ChaCha12Rng,
}

impl IdealBlockVole {
    /// Create a new ideal Block VOLE instance.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// Execute Block Subfield VOLE where A's input is bits (F = F_2).
    ///
    /// # Arguments
    /// * `a` - A's input vector of bits (length n)
    /// * `betas` - B's correlation values β_1, ..., β_k ∈ F_{2^ρ}
    ///
    /// # Returns
    /// * `(BlockVoleSenderOutput, BlockVoleReceiverOutput)` - outputs for A and B
    ///
    /// # Computation
    /// For each i in 1..k:
    /// - b_i is random in F_{2^ρ}^n
    /// - v_i[j] = a[j] * β_i + b_i[j] (multiplication is scalar: 0 or β_i)
    pub fn subfield_vole(
        &mut self,
        a: &[bool],
        betas: &[Block],
    ) -> Result<(BlockVoleSenderOutput, BlockVoleReceiverOutput), BlockVoleError> {
        let n = a.len();
        let k = betas.len();

        if k == 0 {
            return Err(BlockVoleError::InvalidK(k));
        }

        // Generate random b vectors for A
        let mut b: Vec<Vec<Block>> = Vec::with_capacity(k);
        for _ in 0..k {
            let bi: Vec<Block> = (0..n).map(|_| self.rng.random()).collect();
            b.push(bi);
        }

        // Compute v vectors for B: v_i[j] = a[j] * β_i + b_i[j]
        let mut v: Vec<Vec<Block>> = Vec::with_capacity(k);
        for i in 0..k {
            let vi: Vec<Block> = (0..n)
                .map(|j| {
                    if a[j] {
                        // a[j] = 1, so v_i[j] = β_i + b_i[j]
                        betas[i] ^ b[i][j]
                    } else {
                        // a[j] = 0, so v_i[j] = b_i[j]
                        b[i][j]
                    }
                })
                .collect();
            v.push(vi);
        }

        Ok((BlockVoleSenderOutput { b }, BlockVoleReceiverOutput { v }))
    }

    /// Execute Block VOLE where A's input is field elements (F = F_{2^ρ}).
    ///
    /// # Arguments
    /// * `a` - A's input vector in F_{2^ρ}^n
    /// * `betas` - B's correlation values β_1, ..., β_k ∈ F_{2^ρ}
    ///
    /// # Returns
    /// * `(BlockVoleSenderOutput, BlockVoleReceiverOutput)` - outputs for A and B
    ///
    /// # Computation
    /// For each i in 1..k:
    /// - b_i is random in F_{2^ρ}^n
    /// - v_i[j] = a[j] * β_i + b_i[j] (field multiplication in F_{2^ρ})
    pub fn block_vole(
        &mut self,
        a: &[Block],
        betas: &[Block],
    ) -> Result<(BlockVoleSenderOutput, BlockVoleReceiverOutput), BlockVoleError> {
        let n = a.len();
        let k = betas.len();

        if k == 0 {
            return Err(BlockVoleError::InvalidK(k));
        }

        // Generate random b vectors for A
        let mut b: Vec<Vec<Block>> = Vec::with_capacity(k);
        for _ in 0..k {
            let bi: Vec<Block> = (0..n).map(|_| self.rng.random()).collect();
            b.push(bi);
        }

        // Compute v vectors for B: v_i[j] = a[j] * β_i + b_i[j]
        // Using field multiplication in GF(2^128) (via gfmul which does clmul + reduction)
        let mut v: Vec<Vec<Block>> = Vec::with_capacity(k);
        for i in 0..k {
            let vi: Vec<Block> = (0..n)
                .map(|j| {
                    // v_i[j] = a[j] * β_i + b_i[j]
                    // Field multiplication followed by XOR (addition in GF(2^128))
                    a[j].gfmul(betas[i]) ^ b[i][j]
                })
                .collect();
            v.push(vi);
        }

        Ok((BlockVoleSenderOutput { b }, BlockVoleReceiverOutput { v }))
    }
}

/// Subfield VOLE output for a single instance.
///
/// In subfield VOLE (F = F_2, E = F_{2^ρ}):
/// - A holds: key k ∈ F_{2^ρ}
/// - B holds: choice bit c ∈ F_2 and MAC m = c·Δ + k
#[derive(Debug, Clone)]
pub struct SubfieldVoleOutput {
    /// A's random key
    pub key: Block,
    /// B's choice bit
    pub choice: bool,
    /// B's MAC: m = choice * delta + key
    pub mac: Block,
}

/// Ideal Subfield VOLE for generating authenticated bits.
///
/// This generates correlations of the form:
/// - A holds: random key k
/// - B holds: (choice c, MAC m) where m = c·Δ + k
#[derive(Debug)]
pub struct IdealSubfieldVole {
    rng: ChaCha12Rng,
    /// B's global correlation Δ
    delta: Block,
}

impl IdealSubfieldVole {
    /// Create a new ideal subfield VOLE with given delta.
    pub fn new(seed: u64, delta: Block) -> Self {
        Self {
            rng: ChaCha12Rng::seed_from_u64(seed),
            delta,
        }
    }

    /// Generate `count` subfield VOLE correlations.
    ///
    /// Returns vectors of (key, choice, mac) tuples where:
    /// - mac = choice * delta + key
    pub fn generate(&mut self, count: usize) -> Vec<SubfieldVoleOutput> {
        (0..count)
            .map(|_| {
                let key: Block = self.rng.random();
                let choice: bool = self.rng.random();
                let mac = if choice {
                    key ^ self.delta
                } else {
                    key
                };
                SubfieldVoleOutput { key, choice, mac }
            })
            .collect()
    }

    /// Generate subfield VOLE with specific choices (for testing).
    pub fn generate_with_choices(&mut self, choices: &[bool]) -> Vec<SubfieldVoleOutput> {
        choices
            .iter()
            .map(|&choice| {
                let key: Block = self.rng.random();
                let mac = if choice {
                    key ^ self.delta
                } else {
                    key
                };
                SubfieldVoleOutput { key, choice, mac }
            })
            .collect()
    }

    /// Get the delta value.
    pub fn delta(&self) -> Block {
        self.delta
    }
}

/// Extended VOLE that can be used to extend base VOLEs.
///
/// Given L base VOLEs, this can be extended to n VOLEs
/// using the public matrix MH.
#[derive(Debug)]
pub struct ExtendedVole {
    /// Base VOLE outputs (length L)
    base: Vec<SubfieldVoleOutput>,
}

impl ExtendedVole {
    /// Create from base VOLE outputs.
    pub fn new(base: Vec<SubfieldVoleOutput>) -> Self {
        Self { base }
    }

    /// Extend using matrix MH to get n outputs.
    ///
    /// For each output wire i:
    /// - key'[i] = XOR of base keys where MH[i][j] = 1
    /// - choice'[i] = XOR of base choices where MH[i][j] = 1
    /// - mac'[i] = XOR of base MACs where MH[i][j] = 1
    pub fn extend(&self, matrix: &crate::fcp::BinaryMatrix) -> Vec<SubfieldVoleOutput> {
        let n = matrix.rows;
        let l = matrix.cols;

        assert_eq!(
            self.base.len(),
            l,
            "Base VOLE length must match matrix columns"
        );

        (0..n)
            .map(|i| {
                let mut key = Block::ZERO;
                let mut choice = false;
                let mut mac = Block::ZERO;

                for j in 0..l {
                    if matrix.get(i, j) {
                        key = key ^ self.base[j].key;
                        choice ^= self.base[j].choice;
                        mac = mac ^ self.base[j].mac;
                    }
                }

                SubfieldVoleOutput { key, choice, mac }
            })
            .collect()
    }

    /// Get the base outputs.
    pub fn base(&self) -> &[SubfieldVoleOutput] {
        &self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_subfield_vole() {
        let delta = Block::from([0xABu8; 16]);
        let mut vole = IdealSubfieldVole::new(42, delta);

        let outputs = vole.generate(100);

        for out in &outputs {
            // Verify: mac = choice * delta + key
            let expected_mac = if out.choice {
                out.key ^ delta
            } else {
                out.key
            };
            assert_eq!(out.mac, expected_mac);
        }
    }

    #[test]
    fn test_ideal_block_subfield_vole() {
        let mut vole = IdealBlockVole::new(42);

        // A's input: 10 bits
        let a = vec![true, false, true, true, false, false, true, false, true, false];

        // B's correlations: 3 different beta values
        let betas = vec![
            Block::from([1u8; 16]),
            Block::from([2u8; 16]),
            Block::from([3u8; 16]),
        ];

        let (sender_out, receiver_out) = vole.subfield_vole(&a, &betas).unwrap();

        // Verify dimensions
        assert_eq!(sender_out.b.len(), 3);
        assert_eq!(receiver_out.v.len(), 3);
        for i in 0..3 {
            assert_eq!(sender_out.b[i].len(), 10);
            assert_eq!(receiver_out.v[i].len(), 10);
        }

        // Verify correlation: v_i[j] = a[j] * β_i + b_i[j]
        for i in 0..3 {
            for j in 0..10 {
                let expected = if a[j] {
                    betas[i] ^ sender_out.b[i][j]
                } else {
                    sender_out.b[i][j]
                };
                assert_eq!(receiver_out.v[i][j], expected);
            }
        }
    }

    #[test]
    fn test_ideal_block_vole() {
        let mut vole = IdealBlockVole::new(42);

        // A's input: 5 field elements
        let a: Vec<Block> = (0..5).map(|i| Block::from([i as u8; 16])).collect();

        // B's correlations: 2 different beta values
        let betas = vec![Block::from([10u8; 16]), Block::from([20u8; 16])];

        let (sender_out, receiver_out) = vole.block_vole(&a, &betas).unwrap();

        // Verify dimensions
        assert_eq!(sender_out.b.len(), 2);
        assert_eq!(receiver_out.v.len(), 2);

        // Verify correlation: v_i[j] = a[j] * β_i + b_i[j]
        for i in 0..2 {
            for j in 0..5 {
                let expected = a[j].gfmul(betas[i]) ^ sender_out.b[i][j];
                assert_eq!(receiver_out.v[i][j], expected);
            }
        }
    }

    #[test]
    fn test_extended_vole() {
        use crate::fcp::BinaryMatrix;

        // Create base VOLE with L=5 outputs
        let delta = Block::from([0xFFu8; 16]);
        let mut base_vole = IdealSubfieldVole::new(42, delta);
        let base = base_vole.generate(5);

        // Create a 10x5 matrix (extend from 5 to 10)
        let matrix = BinaryMatrix::from_seed(10, 5, 123);

        // Extend
        let extended = ExtendedVole::new(base.clone());
        let outputs = extended.extend(&matrix);

        assert_eq!(outputs.len(), 10);

        // Verify each extended output maintains the VOLE correlation
        for (i, out) in outputs.iter().enumerate() {
            // Manually compute what the output should be
            let mut expected_key = Block::ZERO;
            let mut expected_choice = false;
            for j in 0..5 {
                if matrix.get(i, j) {
                    expected_key = expected_key ^ base[j].key;
                    expected_choice ^= base[j].choice;
                }
            }
            let expected_mac = if expected_choice {
                expected_key ^ delta
            } else {
                expected_key
            };

            assert_eq!(out.key, expected_key);
            assert_eq!(out.choice, expected_choice);
            assert_eq!(out.mac, expected_mac);
        }
    }

    #[test]
    fn test_block_vole_same_a_across_instances() {
        // The key property of block VOLE: A uses the SAME input vector
        // across all k VOLE instances, while B has different betas.

        let mut vole = IdealBlockVole::new(42);

        let a = vec![true, true, false, true, false];
        let betas = vec![
            Block::from([1u8; 16]),
            Block::from([2u8; 16]),
            Block::from([3u8; 16]),
            Block::from([4u8; 16]),
        ];

        let (sender_out, receiver_out) = vole.subfield_vole(&a, &betas).unwrap();

        // For each position j, the difference between v values
        // should reflect the beta differences when a[j] = 1
        for j in 0..5 {
            if a[j] {
                // When a[j] = 1:
                // v_i[j] = β_i + b_i[j]
                // v_k[j] = β_k + b_k[j]
                // So v_i[j] XOR v_k[j] = β_i XOR β_k XOR b_i[j] XOR b_k[j]

                // The correlations are maintained
                for i in 0..4 {
                    let expected = betas[i] ^ sender_out.b[i][j];
                    assert_eq!(receiver_out.v[i][j], expected);
                }
            } else {
                // When a[j] = 0:
                // v_i[j] = b_i[j]
                for i in 0..4 {
                    assert_eq!(receiver_out.v[i][j], sender_out.b[i][j]);
                }
            }
        }
    }
}
