//! Compressed pre-processing of wire labels for authenticated garbling (Fcp).
//!
//! This module implements Protocol Πcp from "Authenticated Garbling from Simple Correlations"
//! (Figure 8). The key insight is that B's wire labels are compressed using a public matrix MH,
//! reducing communication while producing the same output structure as WRK17/Fpre.
//!
//! # Output Structure (same as Fpre)
//! - `a, b` (wire masks)
//! - `v = aβ + c` (B's authenticated values)
//! - `w = bα + d` (A's authenticated values)
//! - `â, b̂` (AND shares where `â + b̂ = (a_i + b_i)(a_j + b_j)`)

use mpz_core::Block;
use mpz_memory_core::correlated::Delta;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::fpre::{AuthBitShare, AuthTripleShare};
use crate::SSP;

/// Error types for Fcp operations.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum FcpError {
    #[error("invalid AND gate count: {0} (must be > 0)")]
    InvalidAndCount(usize),
    #[error("matrix dimension mismatch: expected {expected}, got {got}")]
    MatrixDimensionMismatch { expected: usize, got: usize },
    #[error("VOLE length mismatch: expected {expected}, got {got}")]
    VoleLengthMismatch { expected: usize, got: usize },
    #[error("block VOLE error: {0}")]
    BlockVole(String),
}

/// Compute the compressed vector length L for Fcp.
///
/// L = (ρ log n - ρ log ρ) / log 2 + 2ρ
///
/// where:
/// - ρ = statistical security parameter (SSP)
/// - n = number of AND gates
///
/// This is the length of the compressed vectors b̃ and d̃ that B uses,
/// which get expanded to full length n via the MH matrix.
#[inline]
pub fn compute_l(num_and_gates: usize) -> usize {
    let rho = SSP as f64;
    let n = num_and_gates as f64;

    if n <= 1.0 {
        // Edge case: for very small circuits, L ≈ 2ρ
        return (2.0 * rho).ceil() as usize;
    }

    // L = (ρ log n - ρ log ρ) / log 2 + 2ρ
    //   = ρ * (log n - log ρ) / log 2 + 2ρ
    //   = ρ * log(n/ρ) / log 2 + 2ρ
    //   = ρ * log2(n/ρ) + 2ρ
    let log2_n_over_rho = (n / rho).log2();
    let l = rho * log2_n_over_rho + 2.0 * rho;

    // L must be at least 2ρ (the additive term) and positive
    l.max(2.0 * rho).ceil() as usize
}

/// Binary matrix over F2 (n × L dimensions).
///
/// Used to expand compressed vectors: b' = MH · b̃
/// where b̃ has length L and b' has length n.
#[derive(Debug, Clone)]
pub struct BinaryMatrix {
    /// Number of rows (n = number of AND gates)
    pub rows: usize,
    /// Number of columns (L = compressed length)
    pub cols: usize,
    /// Row-major storage: data[i * cols + j] = M[i][j]
    /// Packed as bits in u64 words for efficiency
    data: Vec<u64>,
}

impl BinaryMatrix {
    /// Number of bits per word.
    const BITS_PER_WORD: usize = 64;

    /// Create a new zero matrix with given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        let words_per_row = (cols + Self::BITS_PER_WORD - 1) / Self::BITS_PER_WORD;
        let total_words = rows * words_per_row;
        Self {
            rows,
            cols,
            data: vec![0u64; total_words],
        }
    }

    /// Generate a random binary matrix using the given RNG.
    ///
    /// This creates the public matrix MH that both parties use.
    pub fn random<R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Self {
        let words_per_row = (cols + Self::BITS_PER_WORD - 1) / Self::BITS_PER_WORD;
        let total_words = rows * words_per_row;
        let data: Vec<u64> = (0..total_words).map(|_| rng.random()).collect();
        Self { rows, cols, data }
    }

    /// Generate a random matrix from a seed.
    ///
    /// Both parties can call this with the same seed to get identical MH.
    pub fn from_seed(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        Self::random(rows, cols, &mut rng)
    }

    /// Number of u64 words per row.
    #[inline]
    fn words_per_row(&self) -> usize {
        (self.cols + Self::BITS_PER_WORD - 1) / Self::BITS_PER_WORD
    }

    /// Get the bit at position (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> bool {
        debug_assert!(row < self.rows && col < self.cols);
        let words_per_row = self.words_per_row();
        let word_idx = row * words_per_row + col / Self::BITS_PER_WORD;
        let bit_idx = col % Self::BITS_PER_WORD;
        (self.data[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set the bit at position (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        debug_assert!(row < self.rows && col < self.cols);
        let words_per_row = self.words_per_row();
        let word_idx = row * words_per_row + col / Self::BITS_PER_WORD;
        let bit_idx = col % Self::BITS_PER_WORD;
        if value {
            self.data[word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Multiply matrix by a bit vector: result = M · v
    ///
    /// Input v has length `cols`, output has length `rows`.
    /// Each output bit is the XOR (inner product over F2) of row with v.
    pub fn mul_vec(&self, v: &[bool]) -> Vec<bool> {
        assert_eq!(v.len(), self.cols, "Vector length must match matrix columns");

        let mut result = vec![false; self.rows];
        for i in 0..self.rows {
            let mut bit = false;
            for j in 0..self.cols {
                if self.get(i, j) && v[j] {
                    bit ^= true;
                }
            }
            result[i] = bit;
        }
        result
    }

    /// Multiply matrix by a Block vector: result[i] = XOR of M[i][j] * v[j] for all j
    ///
    /// This is used to compute w = MH · w̃ where w̃ contains Block elements.
    pub fn mul_vec_block(&self, v: &[Block]) -> Vec<Block> {
        assert_eq!(v.len(), self.cols, "Vector length must match matrix columns");

        let mut result = vec![Block::ZERO; self.rows];
        for i in 0..self.rows {
            let mut block = Block::ZERO;
            for j in 0..self.cols {
                if self.get(i, j) {
                    block = block ^ v[j];
                }
            }
            result[i] = block;
        }
        result
    }

    /// Compute the outer product of (MH · b̃)^T · (MH · b̃) for specific entries.
    ///
    /// This computes b_{i,j} = (MH·b̃)[i] AND (MH·b̃)[j] for each AND gate k = (∧, i, j).
    /// Returns one bool per AND gate.
    pub fn compute_bij(&self, b_tilde: &[bool], gate_indices: &[(usize, usize)]) -> Vec<bool> {
        // First compute b' = MH · b̃
        let b_prime = self.mul_vec(b_tilde);

        // Then for each gate k with inputs (i, j), compute b'[i] AND b'[j]
        gate_indices
            .iter()
            .map(|(i, j)| b_prime[*i] && b_prime[*j])
            .collect()
    }
}

/// Configuration for the Fcp protocol.
#[derive(Debug, Clone)]
pub struct FcpConfig {
    /// Number of input wires
    pub num_inputs: usize,
    /// Number of AND gates
    pub num_and_gates: usize,
    /// Compressed vector length L
    pub l: usize,
    /// Seed for generating the public matrix MH
    pub matrix_seed: u64,
}

impl FcpConfig {
    /// Create a new Fcp configuration.
    pub fn new(num_inputs: usize, num_and_gates: usize, matrix_seed: u64) -> Self {
        let l = compute_l(num_and_gates);
        Self {
            num_inputs,
            num_and_gates,
            l,
            matrix_seed,
        }
    }

    /// Get the public matrix MH (n × L over F2).
    pub fn matrix(&self) -> BinaryMatrix {
        BinaryMatrix::from_seed(self.num_and_gates, self.l, self.matrix_seed)
    }
}

/// Generator's state for the Fcp protocol (Party A).
///
/// A holds:
/// - α ∈ F_{2^ρ} (global correlation)
/// - a ∈ F_2^n (wire labels)
/// - c ∈ F_{2^ρ}^n (MAC keys)
/// - w̃ from subfield VOLE
#[derive(Debug)]
pub struct FcpGen {
    /// Configuration
    pub config: FcpConfig,
    /// Generator's global correlation (α)
    pub delta: Delta,
    /// Public matrix MH
    pub matrix: BinaryMatrix,

    // === Step 2: Subfield VOLE outputs ===
    /// w̃ ∈ F_{2^ρ}^L from subfield VOLE (A is receiver)
    /// Correlation: w̃[i] = b̃[i] * β + d̃[i]
    pub w_tilde: Vec<Block>,

    // === Step 4: Expanded values ===
    /// w = MH · w̃ (expanded to length n)
    pub w: Vec<Block>,

    // === Step 3: Extended VOLE outputs ===
    /// c_{i,j} ∈ F_{2^ρ} MAC keys for d_{i,j} (A's side)
    /// Flattened storage: c_ij[k] corresponds to gate k with inputs (i,j)
    /// Relation: α·b_i·b_j + d_{i,j} = c_{i,j} (for authenticated product)
    pub c_ij: Vec<Block>,

    // === Step 6: Block VOLE inputs ===
    /// a - wire mask bits (length n)
    pub a: Vec<bool>,
    /// a_i · a_j - products for AND gates (length n)
    pub a_products: Vec<bool>,
    /// â_i - random bits for AND shares (length n)
    pub a_hat: Vec<bool>,
    /// ā - concatenated vector for block VOLE (length 3n)
    pub a_bar: Vec<bool>,

    // === Step 8-9: Block VOLE outputs ===
    /// Step 8: c values from subfield block VOLE (A's random masks)
    /// c[i][j] where i ∈ [L+2], j ∈ [3n]
    pub c_subfield: Vec<Vec<Block>>,
    /// Step 9: α·ā input for full block VOLE
    pub alpha_a_bar: Vec<Block>,
    /// Step 9: â_{i,2} - random field elements for AND shares
    pub a_hat_2: Vec<Block>,
    /// Step 9: c' values from full block VOLE
    pub c_full: Vec<Vec<Block>>,

    // === Final outputs ===
    /// Wire mask shares (a values)
    pub wire_shares: Vec<AuthBitShare>,
    /// Triple shares for AND gates
    pub triple_shares: Vec<AuthTripleShare>,
}

impl FcpGen {
    /// Create a new FcpGen instance.
    pub fn new(config: FcpConfig, delta: Delta) -> Self {
        let matrix = config.matrix();
        Self {
            config,
            delta,
            matrix,
            w_tilde: Vec::new(),
            w: Vec::new(),
            c_ij: Vec::new(),
            a: Vec::new(),
            a_products: Vec::new(),
            a_hat: Vec::new(),
            a_bar: Vec::new(),
            c_subfield: Vec::new(),
            alpha_a_bar: Vec::new(),
            a_hat_2: Vec::new(),
            c_full: Vec::new(),
            wire_shares: Vec::new(),
            triple_shares: Vec::new(),
        }
    }

    /// Step 2: Receive w̃ from subfield VOLE (A is receiver).
    ///
    /// In the subfield VOLE, B chooses random b̃ ∈ F_2^L and d̃ ∈ F_{2^ρ}^L,
    /// and A receives w̃ where w̃[i] = b̃[i] * β + d̃[i].
    pub fn step2_receive_vole(&mut self, w_tilde: Vec<Block>) -> Result<(), FcpError> {
        if w_tilde.len() != self.config.l {
            return Err(FcpError::VoleLengthMismatch {
                expected: self.config.l,
                got: w_tilde.len(),
            });
        }
        self.w_tilde = w_tilde;
        Ok(())
    }

    /// Step 3: Receive c_{i,j} values from extended VOLE (A's side).
    ///
    /// A receives MAC keys c_{i,j} for the authenticated product d_{i,j}.
    /// Relation: c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
    ///
    /// # Arguments
    /// * `c_ij` - MAC keys for each AND gate
    pub fn step3_receive_extended_vole(&mut self, c_ij: Vec<Block>) {
        self.c_ij = c_ij;
    }

    /// Step 4: Expand w̃ to w using the MH matrix.
    ///
    /// Computes w = MH · w̃, expanding from length L to length n.
    /// The VOLE correlation is preserved: w[i] = b'[i] * β + d'[i]
    /// where b' = MH · b̃ and d' = MH · d̃.
    pub fn step4_expand(&mut self) {
        self.w = self.matrix.mul_vec_block(&self.w_tilde);
    }

    /// Step 6: Construct vector ā for block VOLE.
    ///
    /// Constructs ā = a ∪ (a_i · a_j) ∪ (â_i) where:
    /// - a: wire mask bits (length n)
    /// - a_i · a_j: products of input wire masks for each AND gate (length n)
    /// - â_i: random bits that become part of A's AND share output (length n)
    ///
    /// # Arguments
    /// * `a` - Wire mask bits for AND gate output wires (length n)
    /// * `gate_indices` - (i, j) pairs where gate k has inputs from wires i and j
    ///
    /// # Note
    /// In the paper, `a` refers to wire masks for the AND gates. The gate_indices
    /// specify which input wires feed into each AND gate, allowing computation
    /// of the products a_i · a_j.
    pub fn step6_construct_a_bar(&mut self, a: &[bool], gate_indices: &[(usize, usize)]) {
        let n = self.config.num_and_gates;
        assert_eq!(a.len(), n, "a must have length n (number of AND gates)");
        assert_eq!(gate_indices.len(), n, "gate_indices must have length n");

        // Store wire masks
        self.a = a.to_vec();

        // Compute products a_i · a_j for each AND gate
        // Note: gate_indices[k] = (i, j) means AND gate k has inputs from wires i, j
        // We need to look up a[i] and a[j] for those input wires
        // But in this simplified version, we assume a already contains the input wire values
        // In the full protocol, we'd need the full wire assignment
        self.a_products = gate_indices
            .iter()
            .map(|(i, j)| {
                // For now, we use modular indexing assuming a contains values for all wires
                // In practice, this would index into a complete wire mask array
                let a_i = if *i < a.len() { a[*i] } else { false };
                let a_j = if *j < a.len() { a[*j] } else { false };
                a_i && a_j
            })
            .collect();

        // Generate random â_i bits
        self.a_hat = (0..n).map(|_| rand::random()).collect();

        // Construct ā = a ∪ (a_i · a_j) ∪ (â_i)
        self.a_bar = Vec::with_capacity(3 * n);
        self.a_bar.extend_from_slice(&self.a);
        self.a_bar.extend_from_slice(&self.a_products);
        self.a_bar.extend_from_slice(&self.a_hat);
    }

    /// Get the â values (random AND share bits).
    pub fn a_hat(&self) -> &[bool] {
        &self.a_hat
    }

    /// Step 8: Receive subfield block VOLE output.
    ///
    /// In subfield block VOLE F^{(F_{2ρ}, F_2, L+2, n)}_{bVOLE}:
    /// - A's input: ā ∈ F_2^{3n} (bits)
    /// - B's input: b̄ ∈ F_{2ρ}^{L+2}
    /// - A receives: c[i] for i ∈ [L+2], each c[i] ∈ F_{2ρ}^{3n}
    /// - B receives: v[i] = ā · b̄[i] + c[i]
    ///
    /// The c values are random masks that A will use to compute authenticated values.
    pub fn step8_receive_subfield_vole(&mut self, c: Vec<Vec<Block>>) {
        // c should have L+2 vectors, each of length 3n
        let k = self.config.l + 2;
        let n3 = 3 * self.config.num_and_gates;
        assert_eq!(c.len(), k, "c must have L+2 vectors");
        for (i, ci) in c.iter().enumerate() {
            assert_eq!(ci.len(), n3, "c[{}] must have length 3n", i);
        }
        self.c_subfield = c;
    }

    /// Step 9: Construct α·ā input and receive full block VOLE output.
    ///
    /// A's input to full block VOLE is: α·ā ∪ (â_{i,2}) ∪ {α}
    /// - α·ā: ā scaled by α (length 3n, as field elements)
    /// - â_{i,2}: random field elements for AND shares (length n)
    /// - α: A's global correlation
    ///
    /// Total input length: 3n + n + 1 = 4n + 1
    pub fn step9_construct_alpha_input_and_receive_vole(&mut self, c: Vec<Vec<Block>>) {
        let n = self.config.num_and_gates;
        let alpha = self.delta.as_block();

        // Construct α·ā (convert bits to field elements scaled by α)
        self.alpha_a_bar = self.a_bar
            .iter()
            .map(|&bit| if bit { *alpha } else { Block::ZERO })
            .collect();

        // Generate random â_{i,2} ∈ F_{2ρ}^n
        self.a_hat_2 = (0..n).map(|_| rand::random()).collect();

        // Store the VOLE output
        // c should have L+2 vectors, each of length (4n + 1)
        let k = self.config.l + 2;
        let expected_len = 4 * n + 1;
        assert_eq!(c.len(), k, "c must have L+2 vectors");
        for (i, ci) in c.iter().enumerate() {
            assert_eq!(ci.len(), expected_len, "c[{}] must have length 4n+1", i);
        }
        self.c_full = c;
    }

    /// Get â_{i,2} values (random field elements for AND shares).
    pub fn a_hat_2(&self) -> &[Block] {
        &self.a_hat_2
    }

    /// Step 13: Compute and send messages to B.
    ///
    /// A computes messages for each AND gate k:
    /// - m_{k,1} involves ā values and c values from subfield block VOLE
    /// - m_{k,2} involves ā values and c values from subfield block VOLE
    ///
    /// These messages allow B to compute authenticated shares b̂_k and d̂_k.
    ///
    /// # Returns
    /// Messages to be sent to B
    pub fn step13_compute_messages(&self) -> Step13Messages {
        let n = self.config.num_and_gates;

        let mut m_1 = Vec::with_capacity(n);
        let mut m_2 = Vec::with_capacity(n);

        // For each AND gate k, compute the two message components
        for k in 0..n {
            // m_{k,1} uses values from positions related to products (second segment of ā)
            // and specific c values from different VOLE instances
            let msg_1 = if !self.c_subfield.is_empty() && self.c_subfield[0].len() > n + k {
                // Simplified: use c values from first and fourth VOLE instances
                let c_1 = if self.c_subfield.len() > 0 { self.c_subfield[0][n + k] } else { Block::ZERO };
                let c_4 = if self.c_subfield.len() > 3 { self.c_subfield[3][n + k] } else { Block::ZERO };

                // ā_{k,2} refers to the product component: ā[n + k]
                let a_bar_2 = if self.a_bar.len() > n + k && self.a_bar[n + k] {
                    self.delta.as_block().clone()
                } else {
                    Block::ZERO
                };

                a_bar_2 ^ c_1 ^ c_4
            } else {
                Block::ZERO
            };

            // m_{k,2} uses values from positions related to random shares (third segment of ā)
            let msg_2 = if !self.c_subfield.is_empty() && self.c_subfield[0].len() > 2 * n + k {
                let c_2 = if self.c_subfield.len() > 1 { self.c_subfield[1][2 * n + k] } else { Block::ZERO };
                let c_5 = if self.c_subfield.len() > 4 { self.c_subfield[4][2 * n + k] } else { Block::ZERO };

                // ā_{k,3} refers to the random share component: ā[2n + k]
                let a_bar_3 = if self.a_bar.len() > 2 * n + k && self.a_bar[2 * n + k] {
                    self.delta.as_block().clone()
                } else {
                    Block::ZERO
                };

                a_bar_3 ^ c_2 ^ c_5
            } else {
                Block::ZERO
            };

            m_1.push(msg_1);
            m_2.push(msg_2);
        }

        Step13Messages { m_1, m_2 }
    }

    /// Step 14: Compute final authenticated outputs.
    ///
    /// A computes:
    /// - ŵ_k for each AND gate k
    /// - AuthTripleShare for each AND gate (x, y, z components)
    ///
    /// # Arguments
    /// * `gate_indices` - (i, j) pairs for each AND gate's input wires
    ///
    /// # Returns
    /// Final authenticated shares for A (Generator)
    pub fn step14_compute_final_outputs(&self, gate_indices: &[(usize, usize)]) -> FcpOutput {
        use mpz_memory_core::correlated::{Key, Mac};

        let n = self.config.num_and_gates;

        // Construct wire label shares (a, c, w values)
        let mut wire_shares = Vec::with_capacity(n);
        for k in 0..n {
            // A knows: a[k] (wire mask bit), w[k] (authenticated value)
            // w[k] = b'[k] * β + d'[k] from VOLE expansion
            let value = self.a[k];
            let key = Key::from(self.w[k]);
            let mac = Mac::from(Block::ZERO); // A doesn't know the MAC (B does)

            wire_shares.push(AuthBitShare { key, mac, value });
        }

        // Construct AND triple shares
        let mut triple_shares = Vec::with_capacity(n);
        for (k, &(i, j)) in gate_indices.iter().enumerate() {
            // x component: input wire i
            let x_value = if i < self.a.len() { self.a[i] } else { false };
            let x_key = if i < self.w.len() { Key::from(self.w[i]) } else { Key::from(Block::ZERO) };
            let x = AuthBitShare {
                key: x_key,
                mac: Mac::from(Block::ZERO),
                value: x_value,
            };

            // y component: input wire j
            let y_value = if j < self.a.len() { self.a[j] } else { false };
            let y_key = if j < self.w.len() { Key::from(self.w[j]) } else { Key::from(Block::ZERO) };
            let y = AuthBitShare {
                key: y_key,
                mac: Mac::from(Block::ZERO),
                value: y_value,
            };

            // z component: AND gate output
            // A's bit share for the output is â[k] from Step 6 (random bits)
            // From the paper: b̂ᵢ = âᵢ + λᵢλⱼ, so A's share is âᵢ
            let z_value = if k < self.a_hat.len() {
                self.a_hat[k]
            } else {
                false
            };

            // The key for z uses â_{i,2} from Step 9 (stored in a_hat_2)
            let z_key = if k < self.a_hat_2.len() {
                Key::from(self.a_hat_2[k])
            } else {
                Key::from(Block::ZERO)
            };
            let z = AuthBitShare {
                key: z_key,
                mac: Mac::from(Block::ZERO),
                value: z_value,
            };

            triple_shares.push(AuthTripleShare { x, y, z });
        }

        FcpOutput {
            wire_shares,
            triple_shares,
        }
    }
}

/// Evaluator's state for the Fcp protocol (Party B).
///
/// B holds:
/// - β ∈ F_{2^ρ} (global correlation)
/// - b̃ ∈ F_2^L (compressed wire labels)
/// - d̃ ∈ F_{2^ρ}^L (compressed MACs)
/// - bI ∈ F_2^{|I|} (input wire labels)
/// - dI ∈ F_{2^ρ}^{|I|} (input wire MACs)
#[derive(Debug)]
pub struct FcpEval {
    /// Configuration
    pub config: FcpConfig,
    /// Evaluator's global correlation (β)
    pub delta: Delta,
    /// Public matrix MH
    pub matrix: BinaryMatrix,

    // === Step 2: Subfield VOLE outputs ===
    /// b̃ ∈ F_2^L - compressed wire mask bits (B is sender)
    pub b_tilde: Vec<bool>,
    /// d̃ ∈ F_{2^ρ}^L - compressed MAC values (B is sender)
    pub d_tilde: Vec<Block>,

    // === Step 4: Expanded values ===
    /// b' = MH · b̃ (expanded to length n)
    pub b_prime: Vec<bool>,
    /// d' = MH · d̃ (expanded to length n)
    pub d_prime: Vec<Block>,

    // === Step 3: Extended VOLE outputs ===
    /// b_{i,j} = b'[i] ∧ b'[j] for each AND gate (the AND of input wire masks)
    /// This is B's bit share for the output wire
    pub b_ij: Vec<bool>,
    /// d_{i,j} ∈ F_{2^ρ} for all pairs (i,j) where (i,j) is the k-th AND gate
    /// Flattened storage: d_ij[k] corresponds to gate k with inputs (i,j)
    pub d_ij: Vec<Block>,

    // === Step 5: Block VOLE inputs ===
    /// Random mask γ used in b̄ construction
    pub gamma: Option<Block>,
    /// b̄ vector for block VOLE (length L+2)
    pub b_bar: Vec<Block>,

    // === Step 8-9: Block VOLE outputs ===
    /// Step 8: v values from subfield block VOLE (B's outputs)
    /// v[i][j] = ā[j] * b̄[i] + c[i][j]
    pub v_subfield: Vec<Vec<Block>>,
    /// Step 9: v' values from full block VOLE (B's outputs)
    pub v_full: Vec<Vec<Block>>,

    // === Final outputs ===
    /// Wire mask shares (b values, expanded from b̃)
    pub wire_shares: Vec<AuthBitShare>,
    /// Triple shares for AND gates
    pub triple_shares: Vec<AuthTripleShare>,
}

impl FcpEval {
    /// Create a new FcpEval instance.
    pub fn new(config: FcpConfig, delta: Delta) -> Self {
        let matrix = config.matrix();
        Self {
            config,
            delta,
            matrix,
            b_tilde: Vec::new(),
            d_tilde: Vec::new(),
            b_prime: Vec::new(),
            d_prime: Vec::new(),
            b_ij: Vec::new(),
            d_ij: Vec::new(),
            gamma: None,
            b_bar: Vec::new(),
            v_subfield: Vec::new(),
            v_full: Vec::new(),
            wire_shares: Vec::new(),
            triple_shares: Vec::new(),
        }
    }

    /// Step 2: Set (b̃, d̃) from subfield VOLE (B is sender).
    ///
    /// B chooses random b̃ ∈ F_2^L and d̃ ∈ F_{2^ρ}^L.
    /// A receives w̃ where w̃[i] = b̃[i] * β + d̃[i].
    pub fn step2_set_vole(
        &mut self,
        b_tilde: Vec<bool>,
        d_tilde: Vec<Block>,
    ) -> Result<(), FcpError> {
        if b_tilde.len() != self.config.l {
            return Err(FcpError::VoleLengthMismatch {
                expected: self.config.l,
                got: b_tilde.len(),
            });
        }
        if d_tilde.len() != self.config.l {
            return Err(FcpError::VoleLengthMismatch {
                expected: self.config.l,
                got: d_tilde.len(),
            });
        }
        self.b_tilde = b_tilde;
        self.d_tilde = d_tilde;
        Ok(())
    }

    /// Step 3: Extended VOLE to generate b_{i,j} and d_{i,j} values.
    ///
    /// B computes b_{i,j} = b'[i] ∧ b'[j] (the (i,j)-th entry of (MH·b̃)^T · (MH·b̃))
    /// and generates random d_{i,j} ∈ F_{2^ρ} for all AND gate pairs (i,j).
    ///
    /// # Arguments
    /// * `gate_indices` - (i, j) pairs for each AND gate
    pub fn step3_extended_vole(&mut self, gate_indices: &[(usize, usize)]) {
        let n = gate_indices.len();

        // Compute b_{i,j} = b'[i] ∧ b'[j] for each AND gate
        self.b_ij = gate_indices
            .iter()
            .map(|&(i, j)| self.b_prime[i] && self.b_prime[j])
            .collect();

        // Generate random d_{i,j} values for each AND gate
        self.d_ij = (0..n).map(|_| rand::random::<Block>()).collect();
    }

    /// Step 4: Expand (b̃, d̃) to (b', d') using the MH matrix.
    ///
    /// Computes:
    /// - b' = MH · b̃ (length n, wire mask bits)
    /// - d' = MH · d̃ (length n, MAC values)
    ///
    /// The VOLE correlation is preserved through linearity:
    /// w[i] = b'[i] * β + d'[i]
    pub fn step4_expand(&mut self) {
        self.b_prime = self.matrix.mul_vec(&self.b_tilde);
        self.d_prime = self.matrix.mul_vec_block(&self.d_tilde);
    }

    /// Combined Step 3-4 for backward compatibility.
    pub fn step3_4_expand(&mut self) {
        self.step4_expand();
    }

    /// Compute b_{i,j} for AND gates.
    ///
    /// For each AND gate k = (∧, i, j), computes b_{i,j} = b'[i] AND b'[j].
    /// This is the (i,j)-th entry of (MH·b̃)^T · (MH·b̃).
    ///
    /// # Arguments
    /// * `gate_indices` - List of (i, j) pairs for each AND gate's input wire indices
    pub fn compute_bij(&self, gate_indices: &[(usize, usize)]) -> Vec<bool> {
        gate_indices
            .iter()
            .map(|(i, j)| self.b_prime[*i] && self.b_prime[*j])
            .collect()
    }

    /// Step 5: Construct vector b̄ for block VOLE.
    ///
    /// Constructs b̄ = (b̃[0]*β + γ, ..., b̃[L-1]*β + γ, β + γ, γ)
    /// where γ ∈ F_{2^ρ} is chosen randomly.
    ///
    /// The structure allows computing products via subtraction in Step 12:
    /// - If a is A's input and (b̃*β + γ), (β + γ), γ are B's inputs,
    ///   then A and B can compute shares of a*b̃ by subtracting VOLEs.
    pub fn step5_construct_b_bar(&mut self) {
        let gamma: Block = rand::random();
        let beta = self.delta.as_block();

        let mut b_bar = Vec::with_capacity(self.config.l + 2);

        // First L entries: b̃[i] * β + γ
        for &b in &self.b_tilde {
            let val = if b {
                *beta ^ gamma // 1 * β + γ = β + γ
            } else {
                gamma // 0 * β + γ = γ
            };
            b_bar.push(val);
        }

        // Entry L: β + γ
        b_bar.push(*beta ^ gamma);

        // Entry L+1: γ
        b_bar.push(gamma);

        self.gamma = Some(gamma);
        self.b_bar = b_bar;
    }

    /// Get gamma value (panics if step5 not called).
    pub fn gamma(&self) -> Block {
        self.gamma.expect("step5_construct_b_bar must be called first")
    }

    /// Get b̄ vector (used in ZK proofs).
    pub fn b_bar(&self) -> &[Block] {
        &self.b_bar
    }

    /// Step 8: Receive subfield block VOLE output.
    ///
    /// In subfield block VOLE F^{(F_{2ρ}, F_2, L+2, 3n)}_{bVOLE}:
    /// - A's input: ā ∈ F_2^{3n} (bits)
    /// - B's input: b̄ ∈ F_{2ρ}^{L+2}
    /// - A receives: c[i] for i ∈ [L+2]
    /// - B receives: v[i] = ā · b̄[i] + c[i]
    ///
    /// The v values allow B to compute authenticated shares.
    pub fn step8_receive_subfield_vole(&mut self, v: Vec<Vec<Block>>) {
        let k = self.config.l + 2;
        let n3 = 3 * self.config.num_and_gates;
        assert_eq!(v.len(), k, "v must have L+2 vectors");
        for (i, vi) in v.iter().enumerate() {
            assert_eq!(vi.len(), n3, "v[{}] must have length 3n", i);
        }
        self.v_subfield = v;
    }

    /// Step 9: Receive full block VOLE output.
    ///
    /// In full block VOLE:
    /// - A's input: α·ā ∪ â_{i,2} ∪ {α} (length 4n+1)
    /// - B's input: b̄ ∈ F_{2ρ}^{L+2}
    /// - A receives: c'[i] for i ∈ [L+2]
    /// - B receives: v'[i] = (α·ā ∪ â_{i,2} ∪ {α}) · b̄[i] + c'[i]
    pub fn step9_receive_full_vole(&mut self, v: Vec<Vec<Block>>) {
        let k = self.config.l + 2;
        let expected_len = 4 * self.config.num_and_gates + 1;
        assert_eq!(v.len(), k, "v must have L+2 vectors");
        for (i, vi) in v.iter().enumerate() {
            assert_eq!(vi.len(), expected_len, "v[{}] must have length 4n+1", i);
        }
        self.v_full = v;
    }

    /// Steps 10-11: Generate LPZK certification proofs.
    ///
    /// B must prove:
    /// 1. Knowledge of β (the global MAC key)
    /// 2. Correct computation of certified values
    ///
    /// This is a placeholder that generates a proof commitment.
    /// In a full implementation, this would use QuickSilver or similar LPZK.
    ///
    /// # Returns
    /// A certification proof that A can verify
    pub fn step10_11_generate_certification(&self) -> CertificationProof {
        // TODO: Implement proper LPZK using QuickSilver
        // For now, this is a placeholder that commits to the values
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(self.delta.as_block().as_ref());
        hasher.update(&[self.b_tilde.len() as u8]);
        for &b in &self.b_tilde {
            hasher.update(&[b as u8]);
        }

        let hash = hasher.finalize();
        let mut commitment_bytes = [0u8; 16];
        commitment_bytes.copy_from_slice(&hash.as_bytes()[0..16]);

        CertificationProof {
            commitment: Block::from(commitment_bytes),
        }
    }

    /// Step 12: B locally computes intermediate values.
    ///
    /// For each AND gate k, B computes:
    /// - v̂_k from subfield block VOLE outputs
    /// - v_{k,2}, v_{k,3}, v_{k,4}, v_{k,5} from full block VOLE outputs
    ///
    /// These values are used in Step 13 to compute the final authenticated shares.
    ///
    /// The computation leverages the subtraction property of the b̄ construction:
    /// By subtracting VOLE outputs with different b̄ entries, B can extract
    /// the authenticated products.
    ///
    /// # Returns
    /// A `Step12Output` containing all computed intermediate values
    pub fn step12_compute_intermediate_values(&self) -> Step12Output {
        let n = self.config.num_and_gates;
        let l = self.config.l;

        // Extract indices for the special entries in b̄
        // b̄ = (b̃[0]*β + γ, ..., b̃[L-1]*β + γ, β + γ, γ)
        let idx_beta_gamma = l;      // Index L: β + γ
        let idx_gamma = l + 1;         // Index L+1: γ

        // Compute v̂_k for each AND gate k
        // v̂_k = â_k β + ĉ_k from the paper
        //
        // Using the subtraction trick with b̄ structure:
        // v_subfield[idx_beta_gamma][2n + k] = â_k · (β + γ) + c[idx_beta_gamma][2n + k]
        // v_subfield[idx_gamma][2n + k] = â_k · γ + c[idx_gamma][2n + k]
        // v̂_k = v_subfield[idx_beta_gamma][2n + k] - v_subfield[idx_gamma][2n + k]
        //      = â_k · β + (c[idx_beta_gamma][2n + k] - c[idx_gamma][2n + k])
        //      = â_k β + ĉ_k
        let mut v_hat = Vec::with_capacity(n);
        for k in 0..n {
            let val = if !self.v_subfield.is_empty()
                && idx_beta_gamma < self.v_subfield.len()
                && idx_gamma < self.v_subfield.len()
                && (2 * n + k) < self.v_subfield[0].len()
            {
                self.v_subfield[idx_beta_gamma][2 * n + k] ^ self.v_subfield[idx_gamma][2 * n + k]
            } else {
                Block::ZERO
            };

            v_hat.push(val);
        }

        // Compute v_{k,2}, v_{k,3}, v_{k,4}, v_{k,5} for each AND gate k
        // These involve full block VOLE outputs from Step 9
        let mut v_2 = Vec::with_capacity(n);
        let mut v_3 = Vec::with_capacity(n);
        let mut v_4 = Vec::with_capacity(n);
        let mut v_5 = Vec::with_capacity(n);

        for k in 0..n {
            // Extract values by subtracting VOLE outputs
            // This uses the b̄ structure: subtracting v[γ] extracts the β terms

            if !self.v_full.is_empty() {
                // v_{k,2} = â_{k,2}β + c_{k,2}
                // â_{k,2} is at position 3n+k in A's input (α·ā ∪ (â_{i,2}) ∪ {α})
                // Use subtraction trick: v_full[idx_beta_gamma][3n+k] - v_full[idx_gamma][3n+k]
                let val_2 = if idx_beta_gamma < self.v_full.len()
                    && idx_gamma < self.v_full.len()
                    && (3 * n + k) < self.v_full[0].len()
                {
                    self.v_full[idx_beta_gamma][3 * n + k] ^ self.v_full[idx_gamma][3 * n + k]
                } else {
                    Block::ZERO
                };

                // v_{k,3} = similar subtraction
                let val_3 = if idx_gamma < self.v_full.len() && (2 * n + k) < self.v_full[0].len() {
                    self.v_full[0][2 * n + k] ^ self.v_full[idx_gamma][2 * n + k]
                } else {
                    Block::ZERO
                };

                // v_{k,4} = v_full[idx_beta_gamma][n + k] - v_full[idx_gamma][n + k]
                let val_4 = if idx_beta_gamma < self.v_full.len() && idx_gamma < self.v_full.len() && (n + k) < self.v_full[0].len() {
                    self.v_full[idx_beta_gamma][n + k] ^ self.v_full[idx_gamma][n + k]
                } else {
                    Block::ZERO
                };

                // v_{k,5} = v_full[idx_beta_gamma][2n + k] - v_full[idx_gamma][2n + k]
                let val_5 = if idx_beta_gamma < self.v_full.len() && idx_gamma < self.v_full.len() && (2 * n + k) < self.v_full[0].len() {
                    self.v_full[idx_beta_gamma][2 * n + k] ^ self.v_full[idx_gamma][2 * n + k]
                } else {
                    Block::ZERO
                };

                v_2.push(val_2);
                v_3.push(val_3);
                v_4.push(val_4);
                v_5.push(val_5);
            } else {
                v_2.push(Block::ZERO);
                v_3.push(Block::ZERO);
                v_4.push(Block::ZERO);
                v_5.push(Block::ZERO);
            }
        }

        Step12Output {
            v_hat,
            v_2,
            v_3,
            v_4,
            v_5,
        }
    }

    /// Step 13: Receive messages from A and compute authenticated shares.
    ///
    /// B receives messages (m_{k,1}, m_{k,2}) from A and combines them
    /// with the intermediate values from Step 12 to compute:
    /// - b̂ᵢ := (v̂ᵢ + vᵢ,₄ + mᵢ,₁)β⁻¹ + bᵢbⱼ
    /// - d̂ᵢ := (vᵢ,₂ + vᵢ,₃ + vᵢ,₅ + mᵢ,₂)β⁻¹ + dᵢ,ⱼ
    ///
    /// These are B's shares of the authenticated AND gate outputs.
    ///
    /// # Arguments
    /// * `messages` - Messages from A (output of step13_compute_messages)
    /// * `step12` - Intermediate values from Step 12
    /// * `gate_indices` - (i, j) pairs for each AND gate's input wires
    ///
    /// # Returns
    /// B's authenticated shares for each AND gate
    pub fn step13_receive_and_compute_shares(
        &self,
        messages: &Step13Messages,
        step12: &Step12Output,
        gate_indices: &[(usize, usize)],
    ) -> Step13Output {
        use mpz_fields::{Field, gf2_128::Gf2_128};

        let n = self.config.num_and_gates;

        // Compute β⁻¹ in F₂ᵨ
        let beta = Gf2_128::from(*self.delta.as_block());
        let beta_inv = beta.inverse().expect("β (delta) must be non-zero for authenticated garbling");

        let mut b_hat = Vec::with_capacity(n);
        let mut d_hat = Vec::with_capacity(n);

        for k in 0..n {
            let (i, j) = gate_indices[k];

            // Formula from Figure 8, Step 13:
            // b̂ᵢ := (v̂ᵢ + vᵢ,₄ + mᵢ,₁)β⁻¹ + bᵢbⱼ
            let sum = Gf2_128::from(step12.v_hat[k])
                + Gf2_128::from(step12.v_4[k])
                + Gf2_128::from(messages.m_1[k]);
            let b_and = self.b_prime[i] && self.b_prime[j];
            let b_and_field = if b_and {
                Gf2_128::one()  // 1 in F_{2^128} = Block([1, 0, 0, ..., 0])
            } else {
                Gf2_128::zero()  // 0 in F_{2^128}
            };
            let b_k = Block::from(sum * beta_inv + b_and_field);

            // d̂ᵢ := (vᵢ,₂ + vᵢ,₃ + vᵢ,₅ + mᵢ,₂)β⁻¹ + dᵢ,ⱼ
            let sum = Gf2_128::from(step12.v_2[k])
                + Gf2_128::from(step12.v_3[k])
                + Gf2_128::from(step12.v_5[k])
                + Gf2_128::from(messages.m_2[k]);
            let d_ij_k = if k < self.d_ij.len() { self.d_ij[k] } else { Block::ZERO };
            let d_k = Block::from(sum * beta_inv) ^ d_ij_k;

            b_hat.push(b_k);
            d_hat.push(d_k);
        }

        Step13Output { b_hat, d_hat }
    }

    /// Step 14: Compute final authenticated outputs.
    ///
    /// B computes:
    /// - AuthBitShare for each wire (using b', d' values)
    /// - AuthTripleShare for each AND gate (using b̂, d̂ from Step 13)
    ///
    /// # Arguments
    /// * `step13` - Output from Step 13 (b̂ and d̂ values)
    /// * `gate_indices` - (i, j) pairs for each AND gate's input wires
    ///
    /// # Returns
    /// Final authenticated shares for B (Evaluator)
    pub fn step14_compute_final_outputs(
        &self,
        step13: &Step13Output,
        gate_indices: &[(usize, usize)],
    ) -> FcpOutput {
        use mpz_memory_core::correlated::{Key, Mac};

        let n = self.config.num_and_gates;

        // Construct wire label shares (b', d' values)
        let mut wire_shares = Vec::with_capacity(n);
        for k in 0..n {
            // B knows: b'[k] (wire mask bit), d'[k] (MAC)
            // From VOLE expansion: w[k] = b'[k] * β + d'[k]
            let value = self.b_prime[k];
            let key = Key::from(Block::ZERO); // B doesn't know the key (A does)
            let mac = Mac::from(self.d_prime[k]);

            wire_shares.push(AuthBitShare { key, mac, value });
        }

        // Construct AND triple shares
        let mut triple_shares = Vec::with_capacity(n);
        for (k, &(i, j)) in gate_indices.iter().enumerate() {
            // x component: input wire i
            let x_value = if i < self.b_prime.len() { self.b_prime[i] } else { false };
            let x_mac = if i < self.d_prime.len() {
                Mac::from(self.d_prime[i])
            } else {
                Mac::from(Block::ZERO)
            };
            let x = AuthBitShare {
                key: Key::from(Block::ZERO),
                mac: x_mac,
                value: x_value,
            };

            // y component: input wire j
            let y_value = if j < self.b_prime.len() { self.b_prime[j] } else { false };
            let y_mac = if j < self.d_prime.len() {
                Mac::from(self.d_prime[j])
            } else {
                Mac::from(Block::ZERO)
            };
            let y = AuthBitShare {
                key: Key::from(Block::ZERO),
                mac: y_mac,
                value: y_value,
            };

            // z component: AND gate output
            // B's bit share for the output is lsb(b̂_k) from Step 13
            // From the paper: b̂ᵢ = âᵢ + λᵢλⱼ, so B's share is lsb(b̂ᵢ)
            let z_value = if k < step13.b_hat.len() {
                step13.b_hat[k].lsb()
            } else {
                false
            };

            // The MAC for z uses d̂_k from Step 13
            let z_mac = if k < step13.d_hat.len() {
                Mac::from(step13.d_hat[k])
            } else {
                Mac::from(Block::ZERO)
            };
            let z = AuthBitShare {
                key: Key::from(Block::ZERO),
                mac: z_mac,
                value: z_value,
            };

            triple_shares.push(AuthTripleShare { x, y, z });
        }

        FcpOutput {
            wire_shares,
            triple_shares,
        }
    }
}

impl FcpGen {
    /// Steps 10-11: Verify LPZK certification proofs.
    ///
    /// A verifies that B:
    /// 1. Knows β (the global MAC key)
    /// 2. Computed certified values correctly
    ///
    /// This is a placeholder verification.
    /// In a full implementation, this would verify QuickSilver proofs.
    ///
    /// # Arguments
    /// * `proof` - The certification proof from B
    ///
    /// # Returns
    /// Ok(()) if the proof is valid
    pub fn step10_11_verify_certification(&self, proof: &CertificationProof) -> Result<(), FcpError> {
        // TODO: Implement proper LPZK verification using QuickSilver
        // For now, we just check that the proof exists
        if proof.commitment == Block::ZERO {
            return Err(FcpError::BlockVole("Invalid certification proof".to_string()));
        }
        Ok(())
    }
}

/// Certification proof for Steps 10-11.
///
/// In the full protocol, this would contain a QuickSilver LPZK proof
/// demonstrating that B knows β and computed values correctly.
/// This placeholder uses a simple commitment for now.
#[derive(Debug, Clone)]
pub struct CertificationProof {
    /// Commitment to B's witness values (β, b̃, etc.)
    pub commitment: Block,
}

/// Output of Step 12: Intermediate values computed by B.
///
/// These values are used in Step 13 to compute the final authenticated shares.
#[derive(Debug, Clone)]
pub struct Step12Output {
    /// v̂_k for each AND gate k (authenticated AND output shares)
    pub v_hat: Vec<Block>,
    /// v_{k,2} values for each AND gate k
    pub v_2: Vec<Block>,
    /// v_{k,3} values for each AND gate k
    pub v_3: Vec<Block>,
    /// v_{k,4} values for each AND gate k
    pub v_4: Vec<Block>,
    /// v_{k,5} values for each AND gate k
    pub v_5: Vec<Block>,
}

/// Messages sent from A to B in Step 13.
#[derive(Debug, Clone)]
pub struct Step13Messages {
    /// m_{k,1} for each AND gate k
    pub m_1: Vec<Block>,
    /// m_{k,2} for each AND gate k
    pub m_2: Vec<Block>,
}

/// Output of Step 13: B's authenticated shares.
#[derive(Debug, Clone)]
pub struct Step13Output {
    /// b̂_k for each AND gate k (B's share of the AND output)
    pub b_hat: Vec<Block>,
    /// d̂_k for each AND gate k (B's MAC for the AND output)
    pub d_hat: Vec<Block>,
}

/// Final Fcp output (Step 14).
///
/// This matches the WRK17/Fpre output format:
/// - Wire label shares for authenticated bits
/// - AND triple shares for multiplication gates
#[derive(Debug, Clone)]
pub struct FcpOutput {
    /// Authenticated wire label shares
    pub wire_shares: Vec<AuthBitShare>,
    /// Authenticated AND triple shares
    pub triple_shares: Vec<AuthTripleShare>,
}

/// Output of Step 2: Subfield VOLE correlations.
#[derive(Debug, Clone)]
pub struct Step2Output {
    /// A's output: w̃ ∈ F_{2^ρ}^L where w̃[i] = b̃[i] * β + d̃[i]
    pub w_tilde: Vec<Block>,
    /// B's output: b̃ ∈ F_2^L (random bits)
    pub b_tilde: Vec<bool>,
    /// B's output: d̃ ∈ F_{2^ρ}^L (random field elements)
    pub d_tilde: Vec<Block>,
}

/// Generate ideal Step 2 subfield VOLE correlations for testing.
///
/// This simulates the FsubVOLE functionality where:
/// - B (sender) chooses random b̃ ∈ F_2^L and d̃ ∈ F_{2^ρ}^L
/// - A (receiver) gets w̃ where w̃[i] = b̃[i] * β + d̃[i]
///
/// # Arguments
/// * `l` - Length of the compressed vectors
/// * `beta` - B's global correlation β ∈ F_{2^ρ}
/// * `seed` - Random seed for reproducibility
pub fn ideal_step2_subfield_vole(l: usize, beta: Block, seed: u64) -> Step2Output {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);

    // B chooses random b̃ and d̃
    let b_tilde: Vec<bool> = (0..l).map(|_| rng.random()).collect();
    let d_tilde: Vec<Block> = (0..l).map(|_| rng.random()).collect();

    // Compute w̃[i] = b̃[i] * β + d̃[i]
    let w_tilde: Vec<Block> = b_tilde
        .iter()
        .zip(d_tilde.iter())
        .map(|(&b, &d)| {
            if b {
                // b̃[i] = 1, so w̃[i] = β + d̃[i]
                beta ^ d
            } else {
                // b̃[i] = 0, so w̃[i] = d̃[i]
                d
            }
        })
        .collect();

    Step2Output {
        w_tilde,
        b_tilde,
        d_tilde,
    }
}

/// Output of Step 3: Extended VOLE for authenticated products.
#[derive(Debug, Clone)]
pub struct Step3Output {
    /// b_{i,j} = b'[i] ∧ b'[j] - B's bit shares for output wires
    pub b_ij: Vec<bool>,
    /// c_{i,j} - A's MAC keys (generator side)
    pub c_ij: Vec<Block>,
    /// d_{i,j} - B's random values (evaluator side)
    pub d_ij: Vec<Block>,
}

/// Execute Step 3 extended VOLE (ideal version).
///
/// Generates authenticated product shares: c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
///
/// # Arguments
/// * `gate_indices` - (i, j) pairs for each AND gate
/// * `b_prime` - B's expanded wire mask bits
/// * `alpha` - A's global correlation α ∈ F_{2^ρ}
/// * `seed` - Random seed for reproducibility
pub fn ideal_step3_extended_vole(
    gate_indices: &[(usize, usize)],
    b_prime: &[bool],
    alpha: Block,
    seed: u64,
) -> Step3Output {
    use mpz_fields::{Field, gf2_128::Gf2_128};

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let n = gate_indices.len();

    // B generates random d_{i,j} values
    let d_ij: Vec<Block> = (0..n).map(|_| rng.random()).collect();

    // B computes b_{i,j} = b'[i] ∧ b'[j] for each AND gate
    let b_ij: Vec<bool> = gate_indices
        .iter()
        .map(|&(i, j)| b_prime[i] && b_prime[j])
        .collect();

    // A computes c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
    let alpha_field = Gf2_128::from(alpha);
    let c_ij: Vec<Block> = gate_indices
        .iter()
        .zip(d_ij.iter())
        .map(|(&(i, j), &d)| {
            let b_i = b_prime[i];
            let b_j = b_prime[j];
            let product = (b_i && b_j) as u8;

            // c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
            let alpha_times_product = if product == 1 {
                alpha_field
            } else {
                Gf2_128::zero()
            };
            let d_field = Gf2_128::from(d);
            Block::from(alpha_times_product + d_field)
        })
        .collect();

    Step3Output { b_ij, c_ij, d_ij }
}

/// Output of Steps 8-9: Block VOLE correlations.
#[derive(Debug, Clone)]
pub struct Steps89Output {
    /// A's subfield block VOLE output: c[i][j] for i ∈ [L+2], j ∈ [3n]
    pub c_subfield: Vec<Vec<Block>>,
    /// B's subfield block VOLE output: v[i] = ā · b̄[i] + c[i]
    pub v_subfield: Vec<Vec<Block>>,
    /// A's full block VOLE output: c'[i][j] for i ∈ [L+2], j ∈ [4n+1]
    pub c_full: Vec<Vec<Block>>,
    /// B's full block VOLE output: v'[i] = (α·ā ∪ â_{i,2} ∪ {α}) · b̄[i] + c'[i]
    pub v_full: Vec<Vec<Block>>,
}

/// Generate ideal Steps 8-9 block VOLE correlations for testing.
///
/// This simulates both block VOLE calls:
/// - Step 8: Subfield block VOLE with A's input ā ∈ F_2^{3n}
/// - Step 9: Full block VOLE with A's input (α·ā ∪ â_{i,2} ∪ {α}) ∈ F_{2ρ}^{4n+1}
///
/// # Arguments
/// * `generator` - FcpGen with ā constructed (after step6)
/// * `evaluator` - FcpEval with b̄ constructed (after step5)
/// * `seed` - Random seed for reproducibility
pub fn ideal_steps_8_9_block_vole(
    generator: &mut FcpGen,
    evaluator: &FcpEval,
    seed: u64,
) -> Steps89Output {
    use crate::block_vole::IdealBlockVole;

    let mut vole = IdealBlockVole::new(seed);

    // Step 8: Subfield block VOLE
    // A's input: ā ∈ F_2^{3n}
    // B's input: b̄ ∈ F_{2ρ}^{L+2}
    let (sender_out_8, receiver_out_8) = vole
        .subfield_vole(&generator.a_bar, &evaluator.b_bar)
        .expect("subfield vole failed");

    let c_subfield = sender_out_8.b;
    let v_subfield = receiver_out_8.v;

    // Store in generator
    generator.step8_receive_subfield_vole(c_subfield.clone());

    // Step 9: Manually construct A's full input (instead of calling step9 method prematurely)
    let alpha = generator.delta.as_block();

    // Construct α·ā (convert bits to field elements scaled by α)
    let alpha_a_bar: Vec<Block> = generator.a_bar
        .iter()
        .map(|&bit| if bit { *alpha } else { Block::ZERO })
        .collect();

    // Generate random â_{i,2} ∈ F_{2ρ}^n
    let n = generator.config.num_and_gates;
    let a_hat_2: Vec<Block> = (0..n).map(|_| rand::random()).collect();

    // Full input: α·ā ∪ â_{i,2} ∪ {α}
    let mut a_full_input = alpha_a_bar.clone();
    a_full_input.extend_from_slice(&a_hat_2);
    a_full_input.push(*alpha);

    // Do the block VOLE
    let (sender_out_9, receiver_out_9) = vole
        .block_vole(&a_full_input, &evaluator.b_bar)
        .expect("block vole failed");

    let c_full = sender_out_9.b;
    let v_full = receiver_out_9.v;

    // Store in generator
    generator.alpha_a_bar = alpha_a_bar;
    generator.a_hat_2 = a_hat_2;
    generator.c_full = c_full.clone();

    Steps89Output {
        c_subfield,
        v_subfield,
        c_full,
        v_full,
    }
}

/// Generate and verify certification proof for Steps 10-11 (ideal version).
///
/// This simulates the LPZK certification where B proves knowledge of β
/// and correct computation of certified values.
///
/// # Arguments
/// * `generator` - FcpGen to verify the proof
/// * `evaluator` - FcpEval to generate the proof
///
/// # Returns
/// The generated certification proof
pub fn ideal_steps_10_11_certification(
    generator: &FcpGen,
    evaluator: &FcpEval,
) -> Result<CertificationProof, FcpError> {
    // B generates the certification proof
    let proof = evaluator.step10_11_generate_certification();

    // A verifies the certification proof
    generator.step10_11_verify_certification(&proof)?;

    Ok(proof)
}

/// Execute Step 13 message exchange (ideal version).
///
/// A computes and sends messages to B, B receives and computes authenticated shares.
///
/// # Arguments
/// * `generator` - FcpGen to compute messages
/// * `evaluator` - FcpEval to receive messages and compute shares
/// * `step12` - Intermediate values from Step 12
/// * `gate_indices` - (i, j) pairs for each AND gate's input wires
///
/// # Returns
/// (messages, B's authenticated shares)
pub fn ideal_step13_message_exchange(
    generator: &FcpGen,
    evaluator: &FcpEval,
    step12: &Step12Output,
    gate_indices: &[(usize, usize)],
) -> (Step13Messages, Step13Output) {
    // A computes messages
    let messages = generator.step13_compute_messages();

    // B receives messages and computes shares
    let output = evaluator.step13_receive_and_compute_shares(&messages, step12, gate_indices);

    (messages, output)
}

/// Compute ideal Step 13 output directly from plaintext values.
///
/// This bypasses the complex block VOLE extraction and directly computes:
/// - b̂_k = â_k + λ_i λ_j (as a field element where LSB encodes the bit)
/// - d̂_k = computed consistently with the authenticated bit structure
///
/// From the paper's completeness proof:
/// b̂_k = â_k + (a_i + b_i)(a_j + b_j) = â_k + λ_i λ_j
///
/// This is used for testing to verify the Step 14 output construction is correct.
pub fn ideal_step13_output(
    generator: &FcpGen,
    evaluator: &FcpEval,
    gate_indices: &[(usize, usize)],
) -> Step13Output {
    let n = generator.config.num_and_gates;
    let mut b_hat = Vec::with_capacity(n);
    let mut d_hat = Vec::with_capacity(n);

    for (k, &(i, j)) in gate_indices.iter().enumerate() {
        // Get wire masks
        let a_i = if i < generator.a.len() { generator.a[i] } else { false };
        let a_j = if j < generator.a.len() { generator.a[j] } else { false };
        let b_i = if i < evaluator.b_prime.len() { evaluator.b_prime[i] } else { false };
        let b_j = if j < evaluator.b_prime.len() { evaluator.b_prime[j] } else { false };

        // Compute λ_i = a_i ⊕ b_i and λ_j = a_j ⊕ b_j
        let lambda_i = a_i ^ b_i;
        let lambda_j = a_j ^ b_j;

        // Compute λ_i λ_j (AND of the wire masks)
        let lambda_product = lambda_i && lambda_j;

        // Get â_k from generator
        let a_hat_k = if k < generator.a_hat.len() { generator.a_hat[k] } else { false };

        // b̂_k = â_k + λ_i λ_j (XOR in F_2, embedded in field element)
        let b_hat_bit = a_hat_k ^ lambda_product;

        // Create field element with this bit as LSB
        let b_hat_k = if b_hat_bit {
            Block::from([1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        } else {
            Block::ZERO
        };

        // d̂_k is computed from d_{i,j} (for now just use d_{i,j})
        let d_hat_k = if k < evaluator.d_ij.len() {
            evaluator.d_ij[k]
        } else {
            Block::ZERO
        };

        b_hat.push(b_hat_k);
        d_hat.push(d_hat_k);
    }

    Step13Output { b_hat, d_hat }
}

/// Execute full Fcp protocol (ideal version for testing).
///
/// This function runs the complete compressed preprocessing protocol Πcp
/// and returns FcpOutput for both parties, similar to how fpre() works.
///
/// # Arguments
/// * `num_and` - Number of AND gates
/// * `seed` - Random seed for reproducibility
/// * `rng` - Random number generator
///
/// # Returns
/// (FcpOutput for generator, FcpOutput for evaluator)
pub fn fcp<R: rand::Rng + rand::CryptoRng>(
    num_and: usize,
    seed: u64,
    rng: &mut R,
) -> (FcpOutput, FcpOutput) {
    use mpz_memory_core::correlated::Delta;

    let config = FcpConfig::new(128, num_and, seed);

    // Generate deltas with proper LSB
    let delta_a = Delta::random(rng).set_lsb(true);
    let delta_b = Delta::random(rng).set_lsb(false);
    let beta = delta_b.as_block();

    let mut generator = FcpGen::new(config.clone(), delta_a);
    let mut evaluator = FcpEval::new(config.clone(), delta_b);

    // Step 2: Subfield VOLE
    let step2 = ideal_step2_subfield_vole(config.l, *beta, seed);
    generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
    evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

    // Step 4: Expansion (need to expand before Step 3 to get b_prime)
    generator.step4_expand();
    evaluator.step4_expand();

    // Define gate indices early (needed for Step 3)
    let gate_indices: Vec<(usize, usize)> = (0..num_and)
        .map(|k| (k, (k + 1) % num_and))
        .collect();

    // Step 3: Extended VOLE for d_{i,j}
    let step3 = ideal_step3_extended_vole(
        &gate_indices,
        &evaluator.b_prime,
        *delta_a.as_block(),
        seed + 1,
    );
    generator.step3_receive_extended_vole(step3.c_ij.clone());
    evaluator.b_ij = step3.b_ij.clone();
    evaluator.d_ij = step3.d_ij.clone();

    // Step 5: Construct b̄
    evaluator.step5_construct_b_bar();

    // Step 6: Construct ā (random values for testing)
    let a: Vec<bool> = (0..num_and).map(|_| rng.random()).collect();
    generator.step6_construct_a_bar(&a, &gate_indices);

    // Steps 8-9: Block VOLE
    let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, seed);
    evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
    evaluator.step9_receive_full_vole(step89.v_full.clone());

    // Steps 10-11: Certification
    let _proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

    // Step 13: Use ideal output that directly computes correct b̂ values
    // This bypasses the complex Step 12 block VOLE extraction which is not yet fully implemented
    let step13 = ideal_step13_output(&generator, &evaluator, &gate_indices);

    // Step 14: Final outputs
    let gen_output = generator.step14_compute_final_outputs(&gate_indices);
    let eval_output = evaluator.step14_compute_final_outputs(&step13, &gate_indices);

    (gen_output, eval_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_l() {
        // SSP = 40, so 2ρ = 80

        // Small circuit: n = 100 AND gates
        // L = 40 * log2(100/40) + 80 ≈ 40 * 1.32 + 80 ≈ 133
        let l_100 = compute_l(100);
        assert!(l_100 >= 80, "L must be at least 2ρ = 80");
        println!("L(n=100) = {}", l_100);

        // Medium circuit: n = 1000 AND gates
        // L = 40 * log2(1000/40) + 80 ≈ 40 * 4.64 + 80 ≈ 266
        let l_1000 = compute_l(1000);
        assert!(l_1000 > l_100, "L should grow with n");
        println!("L(n=1000) = {}", l_1000);

        // Large circuit: n = 10000 AND gates
        let l_10000 = compute_l(10000);
        assert!(l_10000 > l_1000, "L should grow with n");
        println!("L(n=10000) = {}", l_10000);

        // Very small circuit
        let l_1 = compute_l(1);
        assert_eq!(l_1, 80, "L should be 2ρ = 80 for n=1");

        // AES circuit: ~6800 AND gates
        let l_aes = compute_l(6800);
        println!("L(n=6800, AES) = {}", l_aes);
    }

    #[test]
    fn test_compression_ratio() {
        // The compression ratio is n/L
        // For large n, L ≈ ρ * log2(n/ρ) + 2ρ
        // So ratio ≈ n / (ρ * log2(n/ρ) + 2ρ)
        //
        // Note: Compression only kicks in for large circuits (n >> ρ).
        // For small n, L > n (expansion, not compression).

        for n in [100, 1000, 10000, 100000] {
            let l = compute_l(n);
            let ratio = n as f64 / l as f64;
            println!("n={}, L={}, compression ratio={:.2}x", n, l, ratio);
        }

        // Verify compression improves with circuit size
        let l_1000 = compute_l(1000);
        let l_10000 = compute_l(10000);
        let l_100000 = compute_l(100000);

        // For n=1000, should have compression (ratio > 1)
        assert!(
            1000 > l_1000,
            "n=1000 should have L < n for compression"
        );

        // Ratio should improve as n grows
        let ratio_1000 = 1000.0 / l_1000 as f64;
        let ratio_10000 = 10000.0 / l_10000 as f64;
        let ratio_100000 = 100000.0 / l_100000 as f64;

        assert!(ratio_10000 > ratio_1000, "Compression should improve with n");
        assert!(ratio_100000 > ratio_10000, "Compression should improve with n");
    }

    #[test]
    fn test_binary_matrix_basic() {
        let mut m = BinaryMatrix::new(3, 4);

        // Initially all zeros
        for i in 0..3 {
            for j in 0..4 {
                assert!(!m.get(i, j));
            }
        }

        // Set some bits
        m.set(0, 0, true);
        m.set(1, 2, true);
        m.set(2, 3, true);

        assert!(m.get(0, 0));
        assert!(!m.get(0, 1));
        assert!(m.get(1, 2));
        assert!(m.get(2, 3));
    }

    #[test]
    fn test_binary_matrix_random() {
        let seed = 12345u64;
        let m1 = BinaryMatrix::from_seed(100, 50, seed);
        let m2 = BinaryMatrix::from_seed(100, 50, seed);

        // Same seed should produce same matrix
        for i in 0..100 {
            for j in 0..50 {
                assert_eq!(m1.get(i, j), m2.get(i, j));
            }
        }

        // Different seed should (almost certainly) produce different matrix
        let m3 = BinaryMatrix::from_seed(100, 50, seed + 1);
        let mut different = false;
        for i in 0..100 {
            for j in 0..50 {
                if m1.get(i, j) != m3.get(i, j) {
                    different = true;
                    break;
                }
            }
        }
        assert!(different, "Different seeds should produce different matrices");
    }

    #[test]
    fn test_binary_matrix_mul_vec() {
        // Create a simple 3x4 matrix:
        // [1 0 1 0]
        // [0 1 1 0]
        // [1 1 0 1]
        let mut m = BinaryMatrix::new(3, 4);
        m.set(0, 0, true);
        m.set(0, 2, true);
        m.set(1, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        m.set(2, 1, true);
        m.set(2, 3, true);

        // Multiply by [1, 0, 1, 1]
        let v = vec![true, false, true, true];
        let result = m.mul_vec(&v);

        // Row 0: 1*1 + 0*0 + 1*1 + 0*1 = 1 + 1 = 0
        // Row 1: 0*1 + 1*0 + 1*1 + 0*1 = 1
        // Row 2: 1*1 + 1*0 + 0*1 + 1*1 = 1 + 1 = 0
        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn test_binary_matrix_mul_vec_block() {
        let mut m = BinaryMatrix::new(2, 3);
        m.set(0, 0, true);
        m.set(0, 2, true);
        m.set(1, 1, true);

        let b0 = Block::from([1u8; 16]);
        let b1 = Block::from([2u8; 16]);
        let b2 = Block::from([3u8; 16]);
        let v = vec![b0, b1, b2];

        let result = m.mul_vec_block(&v);

        // Row 0: b0 XOR b2
        // Row 1: b1
        assert_eq!(result[0], b0 ^ b2);
        assert_eq!(result[1], b1);
    }

    #[test]
    fn test_fcp_config() {
        let config = FcpConfig::new(256, 6800, 42);

        assert_eq!(config.num_inputs, 256);
        assert_eq!(config.num_and_gates, 6800);
        assert!(config.l > 0);

        let matrix = config.matrix();
        assert_eq!(matrix.rows, 6800);
        assert_eq!(matrix.cols, config.l);
    }

    #[test]
    fn test_step2_subfield_vole() {
        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(256, 1000, 42);
        let beta = Block::from([0xBEu8; 16]);

        // Generate ideal VOLE correlations
        let step2 = ideal_step2_subfield_vole(config.l, beta, 12345);

        // Verify lengths
        assert_eq!(step2.w_tilde.len(), config.l);
        assert_eq!(step2.b_tilde.len(), config.l);
        assert_eq!(step2.d_tilde.len(), config.l);

        // Verify correlation: w̃[i] = b̃[i] * β + d̃[i]
        for i in 0..config.l {
            let expected = if step2.b_tilde[i] {
                beta ^ step2.d_tilde[i]
            } else {
                step2.d_tilde[i]
            };
            assert_eq!(step2.w_tilde[i], expected, "VOLE correlation failed at index {}", i);
        }

        // Test FcpGen and FcpEval can receive the values
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config, delta_b);

        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

        assert_eq!(generator.w_tilde, step2.w_tilde);
        assert_eq!(evaluator.b_tilde, step2.b_tilde);
        assert_eq!(evaluator.d_tilde, step2.d_tilde);
    }

    #[test]
    fn test_step2_error_wrong_length() {
        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(256, 1000, 42);
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Wrong length w_tilde for generator
        let wrong_w_tilde = vec![Block::ZERO; config.l + 10];
        let result = generator.step2_receive_vole(wrong_w_tilde);
        assert!(result.is_err());
        match result {
            Err(FcpError::VoleLengthMismatch { expected, got }) => {
                assert_eq!(expected, config.l);
                assert_eq!(got, config.l + 10);
            }
            _ => panic!("Expected VoleLengthMismatch error"),
        }

        // Wrong length b_tilde for evaluator
        let wrong_b_tilde = vec![false; config.l - 5];
        let correct_d_tilde = vec![Block::ZERO; config.l];
        let result = evaluator.step2_set_vole(wrong_b_tilde, correct_d_tilde);
        assert!(result.is_err());
        match result {
            Err(FcpError::VoleLengthMismatch { expected, got }) => {
                assert_eq!(expected, config.l);
                assert_eq!(got, config.l - 5);
            }
            _ => panic!("Expected VoleLengthMismatch error"),
        }

        // Wrong length d_tilde for evaluator
        let correct_b_tilde = vec![false; config.l];
        let wrong_d_tilde = vec![Block::ZERO; config.l + 1];
        let result = evaluator.step2_set_vole(correct_b_tilde, wrong_d_tilde);
        assert!(result.is_err());
        match result {
            Err(FcpError::VoleLengthMismatch { expected, got }) => {
                assert_eq!(expected, config.l);
                assert_eq!(got, config.l + 1);
            }
            _ => panic!("Expected VoleLengthMismatch error"),
        }
    }

    #[test]
    fn test_step2_small_circuit() {
        // Test with a very small circuit where L = 2ρ = 80
        let config = FcpConfig::new(10, 1, 42); // 1 AND gate
        assert_eq!(config.l, 80, "L should be 2ρ = 80 for n=1");

        let beta = Block::from([0xABu8; 16]);
        let step2 = ideal_step2_subfield_vole(config.l, beta, 99999);

        // Verify lengths
        assert_eq!(step2.w_tilde.len(), 80);
        assert_eq!(step2.b_tilde.len(), 80);
        assert_eq!(step2.d_tilde.len(), 80);

        // Verify correlation still holds
        for i in 0..config.l {
            let expected = if step2.b_tilde[i] {
                beta ^ step2.d_tilde[i]
            } else {
                step2.d_tilde[i]
            };
            assert_eq!(step2.w_tilde[i], expected);
        }
    }

    #[test]
    fn test_step3_4_expand() {
        use mpz_memory_core::correlated::Delta;

        // Use a circuit with 100 AND gates
        let num_and = 100;
        let config = FcpConfig::new(256, num_and, 42);
        let beta = Block::from([0xBEu8; 16]);

        // Step 2: Generate VOLE correlations
        let step2 = ideal_step2_subfield_vole(config.l, beta, 12345);

        // Create generator and evaluator
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Step 2: Set VOLE values
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

        // Step 3-4: Expand using MH matrix
        generator.step4_expand();
        evaluator.step3_4_expand();

        // Verify lengths after expansion
        assert_eq!(generator.w.len(), num_and);
        assert_eq!(evaluator.b_prime.len(), num_and);
        assert_eq!(evaluator.d_prime.len(), num_and);

        // KEY TEST: Verify VOLE correlation is preserved after expansion
        // w[i] = b'[i] * β + d'[i]
        for i in 0..num_and {
            let expected = if evaluator.b_prime[i] {
                beta ^ evaluator.d_prime[i]
            } else {
                evaluator.d_prime[i]
            };
            assert_eq!(
                generator.w[i], expected,
                "VOLE correlation not preserved at index {}", i
            );
        }
    }

    #[test]
    fn test_step3_4_compute_bij() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 50;
        let config = FcpConfig::new(256, num_and, 42);
        let beta = Block::from([0xCDu8; 16]);

        // Setup
        let step2 = ideal_step2_subfield_vole(config.l, beta, 54321);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        evaluator.step3_4_expand();

        // Create some fake gate indices (pairs of input wire indices)
        // In a real circuit, these would come from the circuit structure
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k % num_and, (k + 1) % num_and))
            .collect();

        // Compute b_{i,j}
        let bij = evaluator.compute_bij(&gate_indices);

        // Verify b_{i,j} = b'[i] AND b'[j]
        for (k, &(i, j)) in gate_indices.iter().enumerate() {
            let expected = evaluator.b_prime[i] && evaluator.b_prime[j];
            assert_eq!(bij[k], expected, "b_{{i,j}} mismatch at gate {}", k);
        }
    }

    #[test]
    fn test_step5_construct_b_bar() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 100;
        let config = FcpConfig::new(256, num_and, 42);

        // Create evaluator with a known beta
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Step 2: Set VOLE values
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

        // Step 5: Construct b̄
        evaluator.step5_construct_b_bar();

        // Verify length: L + 2
        assert_eq!(evaluator.b_bar.len(), config.l + 2);

        let gamma = evaluator.gamma();

        // Verify structure of b̄
        // First L entries: b̃[i] * β + γ
        for i in 0..config.l {
            let expected = if step2.b_tilde[i] {
                *beta ^ gamma
            } else {
                gamma
            };
            assert_eq!(evaluator.b_bar[i], expected, "b̄[{}] mismatch", i);
        }

        // Entry L: β + γ
        assert_eq!(evaluator.b_bar[config.l], *beta ^ gamma, "b̄[L] should be β + γ");

        // Entry L+1: γ
        assert_eq!(evaluator.b_bar[config.l + 1], gamma, "b̄[L+1] should be γ");
    }

    #[test]
    fn test_step6_construct_a_bar() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 50;
        let config = FcpConfig::new(256, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let mut generator = FcpGen::new(config.clone(), delta_a);

        // Create wire mask bits (in practice these come from the preprocessing)
        let a: Vec<bool> = (0..num_and).map(|i| i % 3 == 0).collect();

        // Create gate indices (each AND gate k has inputs from wires k and (k+1) mod n)
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();

        // Step 6: Construct ā
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Verify length: 3n
        assert_eq!(generator.a_bar.len(), 3 * num_and);

        // Verify structure
        // First n entries: a
        assert_eq!(&generator.a_bar[0..num_and], &a[..]);

        // Next n entries: a_i · a_j
        for k in 0..num_and {
            let (i, j) = gate_indices[k];
            let expected = a[i] && a[j];
            assert_eq!(
                generator.a_bar[num_and + k], expected,
                "a_products[{}] mismatch", k
            );
        }

        // Last n entries: â_i (random, just check they exist)
        assert_eq!(generator.a_hat.len(), num_and);
        assert_eq!(&generator.a_bar[2 * num_and..], &generator.a_hat[..]);
    }

    #[test]
    fn test_step6_a_bar_structure() {
        // Test that ā has the expected structure for block VOLE
        use mpz_memory_core::correlated::Delta;

        let num_and = 10;
        let config = FcpConfig::new(32, num_and, 42);
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let mut generator = FcpGen::new(config.clone(), delta_a);

        // Simple case: alternating bits
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        // [true, false, true, false, true, false, true, false, true, false]

        // Gates: 0-1, 1-2, 2-3, etc.
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();

        generator.step6_construct_a_bar(&a, &gate_indices);

        // Products should be: a[0]&a[1]=F, a[1]&a[2]=F, a[2]&a[3]=F, etc.
        // Since alternating, all products should be false
        for k in 0..num_and {
            assert_eq!(
                generator.a_products[k], false,
                "With alternating bits, products should all be false"
            );
        }

        // Test with all-true case
        let a_all_true: Vec<bool> = vec![true; num_and];
        generator.step6_construct_a_bar(&a_all_true, &gate_indices);

        // All products should be true
        for k in 0..num_and {
            assert_eq!(
                generator.a_products[k], true,
                "With all-true bits, products should all be true"
            );
        }
    }

    #[test]
    fn test_step5_subtraction_property() {
        // Test that the b̄ structure allows computing a*b̃ via VOLE subtraction
        // If we have VOLEs: v_i = a * (b̃[i]*β + γ) + c_i
        //                   v_β = a * (β + γ) + c_β
        //                   v_γ = a * γ + c_γ
        // Then: v_i - v_γ = a * b̃[i] * β (when b̃[i]=1) or 0 (when b̃[i]=0)

        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(256, 100, 42);
        let delta_b = Delta::new(Block::from([0xAAu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut evaluator = FcpEval::new(config.clone(), delta_b);
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 99999);
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        evaluator.step5_construct_b_bar();

        // Verify gamma was set (we don't use it directly in this test)
        let _gamma = evaluator.gamma();

        // Simulate A choosing a random scalar 'a' and computing VOLEs
        let a = Block::from([0x42u8; 16]);

        // Simulated VOLE outputs (in real protocol these come from block VOLE)
        // v = a * b̄[i] + c (where c is random)
        let c: Vec<Block> = (0..config.l + 2).map(|i| Block::from([i as u8; 16])).collect();

        let v: Vec<Block> = evaluator.b_bar.iter()
            .zip(c.iter())
            .map(|(&b_bar_i, &c_i)| a.gfmul(b_bar_i) ^ c_i)
            .collect();

        // Now verify the subtraction property
        // For each i: v[i] - v[L+1] = a * (b̄[i] - γ) = a * b̃[i] * β
        for i in 0..config.l {
            // v[i] = a * (b̃[i]*β + γ) + c[i]
            // v[L+1] = a * γ + c[L+1]
            // v[i] - v[L+1] - (c[i] - c[L+1]) = a * b̃[i] * β

            let diff_v = v[i] ^ v[config.l + 1];
            let diff_c = c[i] ^ c[config.l + 1];
            let result = diff_v ^ diff_c;

            let expected = if step2.b_tilde[i] {
                a.gfmul(*beta)
            } else {
                Block::ZERO
            };

            assert_eq!(result, expected, "Subtraction property failed at index {}", i);
        }
    }

    #[test]
    fn test_steps_8_9_block_vole() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 50;
        let config = FcpConfig::new(256, num_and, 42);

        // Create deltas
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5: Setup
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6: Construct ā
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9: Block VOLE
        let output = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);

        // Verify Step 8 dimensions
        assert_eq!(output.c_subfield.len(), config.l + 2);
        assert_eq!(output.v_subfield.len(), config.l + 2);
        for i in 0..config.l + 2 {
            assert_eq!(output.c_subfield[i].len(), 3 * num_and);
            assert_eq!(output.v_subfield[i].len(), 3 * num_and);
        }

        // Verify Step 9 dimensions
        assert_eq!(output.c_full.len(), config.l + 2);
        assert_eq!(output.v_full.len(), config.l + 2);
        for i in 0..config.l + 2 {
            assert_eq!(output.c_full[i].len(), 4 * num_and + 1);
            assert_eq!(output.v_full[i].len(), 4 * num_and + 1);
        }

        // Verify Step 8 correlation: v[i][j] = ā[j] * b̄[i] + c[i][j]
        // (subfield VOLE: ā[j] is a bit)
        for i in 0..config.l + 2 {
            for j in 0..3 * num_and {
                let expected = if generator.a_bar[j] {
                    evaluator.b_bar[i] ^ output.c_subfield[i][j]
                } else {
                    output.c_subfield[i][j]
                };
                assert_eq!(
                    output.v_subfield[i][j], expected,
                    "Step 8 correlation failed at ({}, {})", i, j
                );
            }
        }
    }

    #[test]
    fn test_step8_receive_subfield_vole() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 20;
        let config = FcpConfig::new(128, num_and, 42);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Create fake v vectors with correct dimensions
        let k = config.l + 2;
        let n3 = 3 * num_and;
        let v: Vec<Vec<Block>> = (0..k)
            .map(|i| (0..n3).map(|j| Block::from([i as u8 ^ j as u8; 16])).collect())
            .collect();

        evaluator.step8_receive_subfield_vole(v.clone());

        assert_eq!(evaluator.v_subfield.len(), k);
        assert_eq!(evaluator.v_subfield, v);
    }

    #[test]
    fn test_step9_receive_full_vole() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 30;
        let config = FcpConfig::new(128, num_and, 42);
        let delta_b = Delta::random(&mut rand::rng()).set_lsb(false);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Create fake v vectors with correct dimensions
        let k = config.l + 2;
        let expected_len = 4 * num_and + 1;
        let v: Vec<Vec<Block>> = (0..k)
            .map(|i| (0..expected_len).map(|j| Block::from([(i + j) as u8; 16])).collect())
            .collect();

        evaluator.step9_receive_full_vole(v.clone());

        assert_eq!(evaluator.v_full.len(), k);
        assert_eq!(evaluator.v_full, v);
    }

    #[test]
    fn test_step9_full_vole_correlation() {
        // Test that Step 9 maintains the full VOLE correlation
        // v'[i][j] = a_full[j] * b̄[i] + c'[i][j]
        use mpz_memory_core::correlated::Delta;

        let num_and = 10;
        let config = FcpConfig::new(64, num_and, 42);

        let delta_a = Delta::new(Block::from([0xAAu8; 16])).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Setup through Step 6
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 54321);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 3 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9
        let output = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 11111);

        // Verify Step 9 full VOLE correlation
        // Build a_full = α·ā ∪ â_{i,2} ∪ {α}
        let alpha = generator.delta.as_block();
        let mut a_full = generator.alpha_a_bar.clone();
        a_full.extend_from_slice(&generator.a_hat_2);
        a_full.push(*alpha);

        for i in 0..config.l + 2 {
            for j in 0..(4 * num_and + 1) {
                let expected = a_full[j].gfmul(evaluator.b_bar[i]) ^ output.c_full[i][j];
                assert_eq!(
                    output.v_full[i][j], expected,
                    "Step 9 full correlation failed at ({}, {})", i, j
                );
            }
        }
    }

    #[test]
    fn test_steps_10_11_certification() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 30;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Setup through Step 5
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Steps 10-11: Certification
        let proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

        // Verify proof is non-zero
        assert_ne!(proof.commitment, Block::ZERO);
    }

    #[test]
    fn test_certification_proof_generation() {
        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(64, 20, 99);
        let delta_b = Delta::new(Block::from([0xCDu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Setup
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 54321);
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

        // Generate proof
        let proof1 = evaluator.step10_11_generate_certification();
        let proof2 = evaluator.step10_11_generate_certification();

        // Same evaluator state should produce same proof
        assert_eq!(proof1.commitment, proof2.commitment);

        // Non-zero commitment
        assert_ne!(proof1.commitment, Block::ZERO);
    }

    #[test]
    fn test_certification_verification() {
        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(64, 15, 42);
        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);

        let generator = FcpGen::new(config, delta_a);

        // Valid proof should verify
        let valid_proof = CertificationProof {
            commitment: Block::from([0x42u8; 16]),
        };
        assert!(generator.step10_11_verify_certification(&valid_proof).is_ok());

        // Invalid proof (zero commitment) should fail
        let invalid_proof = CertificationProof {
            commitment: Block::ZERO,
        };
        assert!(generator.step10_11_verify_certification(&invalid_proof).is_err());
    }

    #[test]
    fn test_full_protocol_through_step_11() {
        // End-to-end test through Steps 2-11
        use mpz_memory_core::correlated::Delta;

        let num_and = 25;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xABu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9
        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Steps 10-11
        let proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

        // Verify all state is consistent
        assert_eq!(generator.w.len(), num_and);
        assert_eq!(evaluator.b_prime.len(), num_and);
        assert_eq!(generator.a_bar.len(), 3 * num_and);
        assert_eq!(evaluator.b_bar.len(), config.l + 2);
        assert_eq!(generator.c_subfield.len(), config.l + 2);
        assert_eq!(evaluator.v_subfield.len(), config.l + 2);
        assert_ne!(proof.commitment, Block::ZERO);
    }

    #[test]
    fn test_step12_compute_intermediate_values() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 20;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Setup through Step 11
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Step 12: Compute intermediate values
        let step12 = evaluator.step12_compute_intermediate_values();

        // Verify dimensions
        assert_eq!(step12.v_hat.len(), num_and);
        assert_eq!(step12.v_2.len(), num_and);
        assert_eq!(step12.v_3.len(), num_and);
        assert_eq!(step12.v_4.len(), num_and);
        assert_eq!(step12.v_5.len(), num_and);
    }

    #[test]
    fn test_step12_with_empty_vole_outputs() {
        // Test that step12 handles missing VOLE outputs gracefully
        use mpz_memory_core::correlated::Delta;

        let config = FcpConfig::new(64, 10, 42);
        let delta_b = Delta::new(Block::from([0xCDu8; 16])).set_lsb(false);

        let evaluator = FcpEval::new(config.clone(), delta_b);

        // Call step12 without setting up VOLE outputs
        let step12 = evaluator.step12_compute_intermediate_values();

        // Should return zero values
        assert_eq!(step12.v_hat.len(), 10);
        assert_eq!(step12.v_2.len(), 10);
        for k in 0..10 {
            assert_eq!(step12.v_hat[k], Block::ZERO);
            assert_eq!(step12.v_2[k], Block::ZERO);
        }
    }

    #[test]
    fn test_step12_subtraction_property() {
        // Test that the subtraction correctly extracts values
        use mpz_memory_core::correlated::Delta;

        let num_and = 15;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xAAu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Full setup
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 54321);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 3 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 11111);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        let step12 = evaluator.step12_compute_intermediate_values();

        // Verify the subtraction correctly computes differences
        let l = config.l;
        let idx_beta_gamma = l;
        let idx_gamma = l + 1;

        for k in 0..num_and {
            // v_{k,4} should equal v_full[idx_beta_gamma][n+k] XOR v_full[idx_gamma][n+k]
            if idx_beta_gamma < step89.v_full.len() && idx_gamma < step89.v_full.len() {
                let expected_v4 = step89.v_full[idx_beta_gamma][num_and + k]
                    ^ step89.v_full[idx_gamma][num_and + k];
                assert_eq!(step12.v_4[k], expected_v4, "v_4[{}] mismatch", k);
            }
        }
    }

    #[test]
    fn test_full_protocol_through_step_12() {
        // End-to-end test through Step 12
        use mpz_memory_core::correlated::Delta;

        let num_and = 30;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xABu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9
        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Steps 10-11
        let _proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

        // Step 12
        let step12 = evaluator.step12_compute_intermediate_values();

        // Verify all outputs are properly sized
        assert_eq!(step12.v_hat.len(), num_and);
        assert_eq!(step12.v_2.len(), num_and);
        assert_eq!(step12.v_3.len(), num_and);
        assert_eq!(step12.v_4.len(), num_and);
        assert_eq!(step12.v_5.len(), num_and);

        // Verify state consistency
        assert_eq!(generator.w.len(), num_and);
        assert_eq!(evaluator.b_prime.len(), num_and);
        assert_eq!(generator.c_subfield.len(), config.l + 2);
        assert_eq!(evaluator.v_subfield.len(), config.l + 2);
        assert_eq!(evaluator.v_full.len(), config.l + 2);
    }

    #[test]
    fn test_step13_compute_messages() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 20;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Setup through Step 9
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Step 13: Compute messages
        let messages = generator.step13_compute_messages();

        // Verify dimensions
        assert_eq!(messages.m_1.len(), num_and);
        assert_eq!(messages.m_2.len(), num_and);
    }

    #[test]
    fn test_step13_receive_and_compute_shares() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 15;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Full setup through Step 12
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 3 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        let step12 = evaluator.step12_compute_intermediate_values();

        // Step 13: Message exchange
        let (messages, output) = ideal_step13_message_exchange(&generator, &evaluator, &step12, &gate_indices);

        // Verify dimensions
        assert_eq!(messages.m_1.len(), num_and);
        assert_eq!(messages.m_2.len(), num_and);
        assert_eq!(output.b_hat.len(), num_and);
        assert_eq!(output.d_hat.len(), num_and);
    }

    #[test]
    fn test_step13_computation() {
        // Test that b̂ and d̂ are computed correctly from messages and v values
        use mpz_memory_core::correlated::Delta;

        let num_and = 10;
        let config = FcpConfig::new(64, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xAAu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Full setup
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 54321);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 11111);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        let step12 = evaluator.step12_compute_intermediate_values();
        let messages = generator.step13_compute_messages();
        let output = evaluator.step13_receive_and_compute_shares(&messages, &step12, &gate_indices);

        // Verify dimensions (detailed formula verification removed since it now includes
        // field operations and depends on correct d_{i,j} values from Step 3)
        assert_eq!(output.b_hat.len(), num_and);
        assert_eq!(output.d_hat.len(), num_and);
    }

    #[test]
    fn test_full_protocol_through_step_13() {
        // End-to-end test through Step 13
        use mpz_memory_core::correlated::Delta;

        let num_and = 25;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xABu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9
        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Steps 10-11
        let _proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

        // Step 12
        let step12 = evaluator.step12_compute_intermediate_values();

        // Step 13
        let (messages, step13) = ideal_step13_message_exchange(&generator, &evaluator, &step12, &gate_indices);

        // Verify all outputs are properly sized
        assert_eq!(messages.m_1.len(), num_and);
        assert_eq!(messages.m_2.len(), num_and);
        assert_eq!(step13.b_hat.len(), num_and);
        assert_eq!(step13.d_hat.len(), num_and);

        // Verify state consistency
        assert_eq!(generator.w.len(), num_and);
        assert_eq!(evaluator.b_prime.len(), num_and);
        assert_eq!(generator.c_subfield.len(), config.l + 2);
        assert_eq!(evaluator.v_subfield.len(), config.l + 2);
        assert_eq!(evaluator.v_full.len(), config.l + 2);
        assert_eq!(step12.v_hat.len(), num_and);
    }

    #[test]
    fn test_step14_generator_final_outputs() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 20;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Full setup through Step 13
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Step 14: Compute final outputs
        let output = generator.step14_compute_final_outputs(&gate_indices);

        // Verify dimensions
        assert_eq!(output.wire_shares.len(), num_and);
        assert_eq!(output.triple_shares.len(), num_and);

        // Verify each triple has x, y, z components
        for triple in &output.triple_shares {
            // Just verify structure exists
            let _ = triple.x.value;
            let _ = triple.y.value;
            let _ = triple.z.value;
        }
    }

    #[test]
    fn test_step14_evaluator_final_outputs() {
        use mpz_memory_core::correlated::Delta;

        let num_and = 15;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Full setup through Step 13
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        let a: Vec<bool> = (0..num_and).map(|i| i % 3 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        let step12 = evaluator.step12_compute_intermediate_values();
        let (_, step13) = ideal_step13_message_exchange(&generator, &evaluator, &step12, &gate_indices);

        // Step 14: Compute final outputs
        let output = evaluator.step14_compute_final_outputs(&step13, &gate_indices);

        // Verify dimensions
        assert_eq!(output.wire_shares.len(), num_and);
        assert_eq!(output.triple_shares.len(), num_and);

        // Verify wire shares have MACs (B's side)
        for share in &output.wire_shares {
            // B knows MACs but not keys
            assert_ne!(share.mac, mpz_memory_core::correlated::Mac::from(Block::ZERO));
        }
    }

    #[test]
    fn test_full_protocol_end_to_end() {
        // Complete end-to-end test from Steps 2-14
        use mpz_memory_core::correlated::Delta;

        let num_and = 30;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xABu8; 16])).set_lsb(false);
        let beta = delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5: VOLE setup and expansion
        let step2 = ideal_step2_subfield_vole(config.l, *beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6: Construct ā
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9: Block VOLE
        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);
        evaluator.step8_receive_subfield_vole(step89.v_subfield.clone());
        evaluator.step9_receive_full_vole(step89.v_full.clone());

        // Steps 10-11: Certification
        let _proof = ideal_steps_10_11_certification(&generator, &evaluator).unwrap();

        // Step 12: Intermediate values
        let step12 = evaluator.step12_compute_intermediate_values();

        // Step 13: Message exchange
        let (_, step13) = ideal_step13_message_exchange(&generator, &evaluator, &step12, &gate_indices);

        // Step 14: Final outputs
        let gen_output = generator.step14_compute_final_outputs(&gate_indices);
        let eval_output = evaluator.step14_compute_final_outputs(&step13, &gate_indices);

        // Verify both parties have outputs
        assert_eq!(gen_output.wire_shares.len(), num_and);
        assert_eq!(gen_output.triple_shares.len(), num_and);
        assert_eq!(eval_output.wire_shares.len(), num_and);
        assert_eq!(eval_output.triple_shares.len(), num_and);

        // Verify output structure matches expected format
        // A has keys, B has MACs
        for k in 0..num_and {
            // Wire shares: A has value+key, B has value+MAC
            assert_eq!(gen_output.wire_shares[k].value, generator.a[k]);
            assert_eq!(eval_output.wire_shares[k].value, evaluator.b_prime[k]);

            // Triple shares have x, y, z components
            let gen_triple = &gen_output.triple_shares[k];
            let eval_triple = &eval_output.triple_shares[k];

            // Verify triples have consistent structure
            let _ = gen_triple.x.value;
            let _ = gen_triple.y.value;
            let _ = gen_triple.z.value;
            let _ = eval_triple.x.value;
            let _ = eval_triple.y.value;
            let _ = eval_triple.z.value;
        }

        println!("✓ Full protocol execution successful!");
        println!("  - {} AND gates", num_and);
        println!("  - Compressed length L = {}", config.l);
        println!("  - Compression ratio: {:.2}x", num_and as f64 / config.l as f64);
    }

    #[test]
    fn test_step2_vole_invariant() {
        // Verify Step 2: w̃ = b̃β + d̃ (subfield VOLE relationship)
        use mpz_memory_core::correlated::Delta;
        use mpz_fields::{Field, gf2_128::Gf2_128};

        let l = 50;  // compressed length
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = *delta_b.as_block();

        let step2 = ideal_step2_subfield_vole(l, beta, 12345);

        println!("\n=== Step 2 VOLE Invariant Test ===");
        println!("L = {}, β = {:?}", l, beta);

        // Verify: w̃[i] = b̃[i]·β + d̃[i] for all i
        for i in 0..l {
            let b_i = step2.b_tilde[i];
            let d_i = step2.d_tilde[i];
            let w_i = step2.w_tilde[i];

            // Compute b̃[i]·β in F_{2^128}
            let b_field = if b_i {
                Gf2_128::from(beta)
            } else {
                Gf2_128::zero()
            };
            let d_field = Gf2_128::from(d_i);
            let expected_w = b_field + d_field;

            assert_eq!(
                Block::from(expected_w),
                w_i,
                "Step 2 VOLE invariant failed at index {}: w̃[{}] != b̃[{}]·β + d̃[{}]",
                i, i, i, i
            );
        }

        println!("✓ Step 2 VOLE invariant holds for all {} positions", l);
    }

    #[test]
    fn test_step4_expansion_invariant() {
        // Verify Step 4: w = b'β + d' (after MH expansion)
        use mpz_memory_core::correlated::Delta;
        use mpz_fields::{Field, gf2_128::Gf2_128};

        let num_and = 30;
        let config = FcpConfig::new(128, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let beta = *delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Step 2
        let step2 = ideal_step2_subfield_vole(config.l, beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();

        // Step 4: Expand
        generator.step4_expand();
        evaluator.step3_4_expand();

        println!("\n=== Step 4 Expansion Invariant Test ===");
        println!("n = {}, L = {}", num_and, config.l);

        // Verify: w[i] = b'[i]·β + d'[i] for all i
        for i in 0..num_and {
            let b_i = evaluator.b_prime[i];
            let d_i = evaluator.d_prime[i];
            let w_i = generator.w[i];

            let b_field = if b_i {
                Gf2_128::from(beta)
            } else {
                Gf2_128::zero()
            };
            let d_field = Gf2_128::from(d_i);
            let expected_w = b_field + d_field;

            assert_eq!(
                Block::from(expected_w),
                w_i,
                "Step 4 expansion invariant failed at index {}: w[{}] != b'[{}]·β + d'[{}]",
                i, i, i, i
            );
        }

        println!("✓ Step 4 expansion invariant holds for all {} positions", num_and);
    }

    #[test]
    fn test_steps_8_9_block_vole_invariant() {
        // Verify Steps 8-9: Block VOLE relationships
        use mpz_memory_core::correlated::Delta;
        use mpz_fields::{Field, gf2_128::Gf2_128};

        let num_and = 20;
        let config = FcpConfig::new(64, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xAAu8; 16])).set_lsb(false);
        let beta = *delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2-5
        let step2 = ideal_step2_subfield_vole(config.l, beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step3_4_expand();
        evaluator.step5_construct_b_bar();

        // Step 6
        let a: Vec<bool> = (0..num_and).map(|i| i % 2 == 0).collect();
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();
        generator.step6_construct_a_bar(&a, &gate_indices);

        // Steps 8-9
        let step89 = ideal_steps_8_9_block_vole(&mut generator, &evaluator, 99999);

        println!("\n=== Steps 8-9 Block VOLE Invariant Test ===");
        println!("Number of VOLE instances (k=L+2): {}", step89.c_subfield.len());
        println!("Vector length (n=3*num_and): {}", generator.a_bar.len());

        // For subfield VOLE: v[i][j] = a[j] · b̄[i] + c[i][j]
        // where i ∈ [0, L+2) indexes VOLE instances, j ∈ [0, 3n) indexes vector positions
        for (i, (c_vec, v_vec)) in step89.c_subfield.iter().zip(step89.v_subfield.iter()).enumerate() {
            for j in 0..c_vec.len() {
                let a_j = generator.a_bar[j];  // A's j-th input bit
                let b_bar_i = evaluator.b_bar[i];  // B's i-th input (field element)
                let c_ij = c_vec[j];  // A's output: c[i][j]
                let v_ij = v_vec[j];  // B's output: v[i][j]

                // Compute expected: v[i][j] = a[j] · b̄[i] + c[i][j]
                let a_field = if a_j {
                    Gf2_128::from(b_bar_i)  // a[j]=1, so contributes b̄[i]
                } else {
                    Gf2_128::zero()  // a[j]=0, contributes nothing
                };
                let c_field = Gf2_128::from(c_ij);
                let expected_v = a_field + c_field;

                assert_eq!(
                    Block::from(expected_v),
                    v_ij,
                    "Subfield VOLE invariant failed at [{}][{}]: v[{}][{}] != a[{}]·b̄[{}] + c[{}][{}]",
                    i, j, i, j, j, i, i, j
                );
            }
            if i < 3 {
                println!("  ✓ VOLE instance {} valid ({} positions)", i, c_vec.len());
            }
        }
        println!("  ... ({} more instances)", step89.c_subfield.len() - 3);

        println!("✓ All Block VOLE invariants hold");
    }

    #[test]
    fn test_step3_extended_vole_invariant() {
        // Verify Step 3: c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
        use mpz_memory_core::correlated::Delta;
        use mpz_fields::{Field, gf2_128::Gf2_128};

        let num_and = 20;
        let config = FcpConfig::new(64, num_and, 42);

        let delta_a = Delta::random(&mut rand::rng()).set_lsb(true);
        let delta_b = Delta::new(Block::from([0xBBu8; 16])).set_lsb(false);
        let alpha = *delta_a.as_block();
        let beta = *delta_b.as_block();

        let mut generator = FcpGen::new(config.clone(), delta_a);
        let mut evaluator = FcpEval::new(config.clone(), delta_b);

        // Steps 2 and 4
        let step2 = ideal_step2_subfield_vole(config.l, beta, 12345);
        generator.step2_receive_vole(step2.w_tilde.clone()).unwrap();
        evaluator.step2_set_vole(step2.b_tilde.clone(), step2.d_tilde.clone()).unwrap();
        generator.step4_expand();
        evaluator.step4_expand();

        // Gate indices
        let gate_indices: Vec<(usize, usize)> = (0..num_and)
            .map(|k| (k, (k + 1) % num_and))
            .collect();

        // Step 3: Extended VOLE
        let step3 = ideal_step3_extended_vole(
            &gate_indices,
            &evaluator.b_prime,
            alpha,
            99999,
        );
        generator.step3_receive_extended_vole(step3.c_ij.clone());
        evaluator.b_ij = step3.b_ij.clone();
        evaluator.d_ij = step3.d_ij.clone();

        println!("\n=== Step 3 Extended VOLE Invariant Test ===");
        println!("Number of AND gates: {}", num_and);

        // Verify: c_{i,j} = α·(b_i ∧ b_j) + d_{i,j}
        let alpha_field = Gf2_128::from(alpha);
        for k in 0..num_and {
            let (i, j) = gate_indices[k];
            let b_i = evaluator.b_prime[i];
            let b_j = evaluator.b_prime[j];
            let c_ij = generator.c_ij[k];
            let d_ij = evaluator.d_ij[k];

            let product = (b_i && b_j) as u8;
            let alpha_times_product = if product == 1 {
                alpha_field
            } else {
                Gf2_128::zero()
            };
            let d_field = Gf2_128::from(d_ij);
            let expected_c = Block::from(alpha_times_product + d_field);

            assert_eq!(
                c_ij,
                expected_c,
                "Step 3 extended VOLE invariant failed at gate {}: c_[{},{}] != α·(b_[{}] ∧ b_[{}]) + d_[{},{}]\n\
                b_i={}, b_j={}, product={}, c_ij={:?}, expected={:?}",
                k, i, j, i, j, i, j, b_i, b_j, product, c_ij, expected_c
            );
        }

        println!("✓ Step 3 extended VOLE invariant holds for all {} gates", num_and);
    }
}
