//! Garbler for Three Halves Scheme
//!
//! This module implements the garbling function for circuits using the
//! Three Halves technique from Rosulek & Roy 2021.

use core::fmt;
use std::ops::Range;

use mpz_circuits::{Circuit, Gate};
use mpz_core::{
    Block,
    aes::{FIXED_KEY_AES, FixedKeyAes},
};
use mpz_memory_core::correlated::{Delta, Key, Mac};
use rand::{CryptoRng, Rng};

/// Pre-generated random bits for efficient consumption during garbling.
///
/// Instead of calling RNG for each bit, we pre-generate all needed random bits
/// as u64 words and extract bits sequentially. This reduces RNG calls by ~64x.
struct RandomBitSource {
    /// Pre-generated random words
    data: Vec<u64>,
    /// Current bit index
    bit_idx: usize,
}

impl RandomBitSource {
    /// Create a new source with pre-generated random bits.
    ///
    /// # Arguments
    /// * `num_bits` - Total number of random bits needed
    /// * `rng` - Random number generator to use for generation
    fn new<R: Rng>(num_bits: usize, rng: &mut R) -> Self {
        let num_u64s = (num_bits + 63) / 64;
        let data: Vec<u64> = (0..num_u64s).map(|_| rng.random()).collect();
        Self { data, bit_idx: 0 }
    }

    /// Get the next random bit.
    #[inline]
    fn next_bit(&mut self) -> bool {
        let word_idx = self.bit_idx / 64;
        let bit_pos = self.bit_idx % 64;
        self.bit_idx += 1;
        (self.data[word_idx] >> bit_pos) & 1 == 1
    }

    /// Get the next two random bits (for control matrix sampling).
    #[inline]
    fn next_two_bits(&mut self) -> [bool; 2] {
        [self.next_bit(), self.next_bit()]
    }
}

use super::{control::sample_r_odd, slicing::SlicedLabel};

use crate::DEFAULT_BATCH_SIZE;

/// Errors that can occur during garbled circuit generation.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum GarblerError {
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("garbler not finished")]
    NotFinished,
}

/// Gate ciphertexts for a Three Halves AND gate.
///
/// Contains 3 ciphertexts of κ/2 bits each = 1.5κ bits total.
/// This is smaller than half-gates which uses 2κ bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreeHalvesGate {
    /// Gate ciphertext G₀ (κ/2 = 64 bits)
    pub g0: [u8; 8],
    /// Gate ciphertext G₁ (κ/2 = 64 bits)
    pub g1: [u8; 8],
    /// Gate ciphertext G₂ (κ/2 = 64 bits)
    pub g2: [u8; 8],
}

impl ThreeHalvesGate {
    /// Create a new gate from three κ/2-bit ciphertexts.
    pub fn new(g0: [u8; 8], g1: [u8; 8], g2: [u8; 8]) -> Self {
        Self { g0, g1, g2 }
    }

    /// Total size in bits: 3 × 64 = 192 bits = 1.5κ
    pub const SIZE_BITS: usize = 192;

    /// Total size in bytes: 24 bytes
    pub const SIZE_BYTES: usize = 24;
}

/// Control bits for evaluator (compressed form).
///
/// The r_bar is a 4×2 matrix where each row r_bar[ij] contains the coefficients
/// [c₁, c₂] for input position (i,j). The evaluator expands this to a 2×4
/// marginal using: R_ij = c₁·S₁ ⊕ c₂·S₂
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlBits {
    /// Compressed representation r̄: 4 entries of 2 coefficients each
    pub r_bar: [[bool; 2]; 4],
}

impl ControlBits {
    /// Create new control bits from the compressed r_bar representation.
    pub fn new(r_bar: [[bool; 2]; 4]) -> Self {
        Self { r_bar }
    }

    /// Total size in bytes for transmission (8 bits).
    pub const SIZE_BYTES: usize = 8;
}

/// Encrypted gate for Three Halves scheme.
///
/// Contains both the gate ciphertexts and control bits needed for evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncryptedGate {
    /// Gate ciphertexts (1.5κ bits)
    pub gate: ThreeHalvesGate,
    /// Control bits for evaluator
    pub control_bits: ControlBits,
}

impl EncryptedGate {
    /// Create a new encrypted gate.
    pub fn new(gate: ThreeHalvesGate, control_bits: ControlBits) -> Self {
        Self { gate, control_bits }
    }
}

/// Output of the garbler.
#[derive(Debug)]
pub struct GarblerOutput {
    /// Input label pairs for each input wire: (label_for_false,
    /// label_for_true). Used for OT to give the evaluator the correct label
    /// based on their input bit.
    pub inputs: Vec<(Mac, Mac)>,
    /// Output label pairs for each output wire: (label_for_false,
    /// label_for_true). The evaluator can decode by matching their MAC
    /// against these labels.
    pub outputs: Vec<(Mac, Mac)>,
}

/// Garbler for Three Halves scheme.
#[derive(Debug, Default)]
pub struct Garbler {
    /// Buffer for the 0-bit labels.
    /// TODO: is it correct to call them keys???
    buffer: Vec<Block>,
}

impl Garbler {
    /// Returns an iterator over the encrypted gates of a circuit.
    ///
    /// # Arguments
    ///
    /// * `circ` - The circuit to garble.
    /// * `delta` - The delta value to use for garbling.
    /// * `inputs` - The input labels to the circuit.
    /// * `rng` - Random number generator for control matrix randomization.
    pub fn generate<'a, R: Rng + CryptoRng>(
        &'a mut self,
        circ: &'a Circuit,
        delta: Delta,
        inputs: &[Key],
        rng: &mut R,
    ) -> Result<EncryptedGateIter<'a, std::slice::Iter<'a, Gate>>, GarblerError> {
        if inputs.len() != circ.inputs().len() {
            return Err(GarblerError::InputLength {
                expected: circ.inputs().len(),
                actual: inputs.len(),
            });
        }

        // Expand the buffer to fit the circuit
        if circ.feed_count() > self.buffer.len() {
            self.buffer.resize(circ.feed_count(), Default::default());
        }

        // Pre-generate all random bits needed:
        // - 1 bit per input wire (for permute bits)
        // - 2 bits per AND gate (for control matrix randomization)
        let num_random_bits = inputs.len() + 2 * circ.and_count();
        let mut random_bits = RandomBitSource::new(num_random_bits, rng);

        // Initialize permute bits for all wires
        let mut permute_bits = vec![false; circ.feed_count()];

        // For input wires: keys must have LSB = 0
        // Permute bits are randomly generated from pre-generated source
        for (i, key) in inputs.iter().enumerate() {
            let label = *key.as_block();
            debug_assert!(
                !label.lsb(),
                "Three Halves requires input keys with LSB = 0"
            );
            self.buffer[i] = label;
            permute_bits[i] = random_bits.next_bit();
        }

        Ok(EncryptedGateIter::new(
            delta,
            circ.gates().iter(),
            &mut self.buffer,
            permute_bits,
            circ.and_count(),
            0..inputs.len(),
            circ.outputs(),
            random_bits,
        ))
    }

    /// Returns an iterator over batched encrypted gates of a circuit.
    ///
    /// # Arguments
    ///
    /// * `circ` - The circuit to garble.
    /// * `delta` - The delta value to use for garbling.
    /// * `inputs` - The input labels to the circuit.
    /// * `rng` - Random number generator for control matrix randomization.
    pub fn generate_batched<'a, R: Rng + CryptoRng>(
        &'a mut self,
        circ: &'a Circuit,
        delta: Delta,
        inputs: &[Key],
        rng: &mut R,
    ) -> Result<EncryptedGateBatchIter<'a, std::slice::Iter<'a, Gate>>, GarblerError> {
        self.generate(circ, delta, inputs, rng)
            .map(EncryptedGateBatchIter)
    }
}

/// Iterator over encrypted gates of a garbled circuit.
pub struct EncryptedGateIter<'a, I> {
    /// Cipher to use to encrypt the gates.
    cipher: &'static FixedKeyAes,
    /// Global offset.
    delta: Delta,
    /// Buffer for the 0-color-bit labels (always have LSB = 0).
    labels: &'a mut [Block],
    /// Buffer for the point-and-permute bits (tracked separately from labels).
    permute_bits: Vec<bool>,
    /// Iterator over the gates.
    gates: I,
    /// Current gate id.
    gid: usize,
    /// Number of AND gates generated.
    counter: usize,
    /// Number of AND gates in the circuit.
    and_count: usize,
    /// Range of the inputs in the buffer.
    inputs: Range<usize>,
    /// Range of the outputs in the buffer.
    outputs: Range<usize>,
    /// Whether the entire circuit has been garbled.
    complete: bool,
    /// Pre-generated random bits for control matrix randomization.
    random_bits: RandomBitSource,
}

impl<I> fmt::Debug for EncryptedGateIter<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EncryptedGateIter {{ .. }}")
    }
}

impl<'a, I> EncryptedGateIter<'a, I>
where
    I: Iterator<Item = &'a Gate>,
{
    fn new(
        delta: Delta,
        gates: I,
        labels: &'a mut [Block],
        permute_bits: Vec<bool>,
        and_count: usize,
        inputs: Range<usize>,
        outputs: Range<usize>,
        random_bits: RandomBitSource,
    ) -> Self {
        Self {
            cipher: &(*FIXED_KEY_AES),
            delta,
            gates,
            labels,
            permute_bits,
            gid: 1,
            counter: 0,
            and_count,
            inputs,
            outputs,
            complete: false,
            random_bits,
        }
    }

    /// Returns `true` if the garbler has more encrypted gates to generate.
    #[inline]
    pub fn has_gates(&self) -> bool {
        self.counter != self.and_count
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(mut self) -> Result<GarblerOutput, GarblerError> {
        if self.has_gates() {
            return Err(GarblerError::NotFinished);
        }

        // Finish computing any "free" gates.
        if !self.complete {
            assert_eq!(self.next(), None);
        }

        let delta_block = *self.delta.as_block();

        // Helper to compute label pairs for a wire
        let compute_pair = |labels: &[Block], permute_bits: &[bool], i: usize| {
            let w_k = labels[i]; // LSB = 0
            let w_k_delta = w_k ^ delta_block; // LSB = 1
            let pi_k = permute_bits[i];

            // W_k represents value π_k, W_k ⊕ Δ represents value 1-π_k
            if pi_k {
                // π_k = 1: W_k = true label, W_k ⊕ Δ = false label
                (w_k_delta.into(), w_k.into())
            } else {
                // π_k = 0: W_k = false label, W_k ⊕ Δ = true label
                (w_k.into(), w_k_delta.into())
            }
        };

        // Return both labels for each input wire, ordered by semantic value.
        let input_pairs: Vec<(Mac, Mac)> = self
            .inputs
            .clone()
            .map(|i| compute_pair(self.labels, &self.permute_bits, i))
            .collect();

        // Return both labels for each output wire, ordered by semantic value.
        let output_pairs: Vec<(Mac, Mac)> = self
            .outputs
            .clone()
            .map(|i| compute_pair(self.labels, &self.permute_bits, i))
            .collect();

        Ok(GarblerOutput {
            inputs: input_pairs,
            outputs: output_pairs,
        })
    }
}

impl<'a, I> Iterator for EncryptedGateIter<'a, I>
where
    I: Iterator<Item = &'a Gate>,
{
    type Item = EncryptedGate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(gate) = self.gates.next() {
            match gate {
                Gate::Xor {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    // Free XOR: output label = XOR of input labels
                    let x_0 = self.labels[node_x.id()];
                    let y_0 = self.labels[node_y.id()];
                    self.labels[node_z.id()] = x_0 ^ y_0;
                    // Permute bit of output = XOR of input permute bits
                    self.permute_bits[node_z.id()] =
                        self.permute_bits[node_x.id()] ^ self.permute_bits[node_y.id()];
                }
                Gate::And {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x_0 = self.labels[node_x.id()];
                    let y_0 = self.labels[node_y.id()];
                    let pi_a = self.permute_bits[node_x.id()];
                    let pi_b = self.permute_bits[node_y.id()];

                    // Get pre-generated random bits for this AND gate
                    let rand_bits = self.random_bits.next_two_bits();

                    let (c, pi_c, encrypted_gate) = and_gate(
                        self.cipher,
                        &x_0,
                        &y_0,
                        pi_a,
                        pi_b,
                        &self.delta,
                        self.gid,
                        rand_bits,
                    );
                    // c already has LSB = 0 (adjusted in and_gate)
                    self.labels[node_z.id()] = c;
                    self.permute_bits[node_z.id()] = pi_c;

                    self.gid += 1;
                    self.counter += 1;

                    // If we have generated all AND gates, compute remaining free gates.
                    if !self.has_gates() {
                        assert!(self.next().is_none());
                        self.complete = true;
                    }

                    return Some(encrypted_gate);
                }
                Gate::Inv {
                    x: node_x,
                    z: node_z,
                } => {
                    // INV: label stays the same (we're tracking the 0-color-bit label)
                    // but the permute bit flips
                    let x_0 = self.labels[node_x.id()];
                    self.labels[node_z.id()] = x_0;
                    self.permute_bits[node_z.id()] = !self.permute_bits[node_x.id()];
                }
                Gate::Id {
                    x: node_x,
                    z: node_z,
                } => {
                    let x_0 = self.labels[node_x.id()];
                    self.labels[node_z.id()] = x_0;
                    self.permute_bits[node_z.id()] = self.permute_bits[node_x.id()];
                }
            }
        }

        None
    }
}

/// A batch of encrypted gates.
#[derive(Debug)]
pub struct EncryptedGateBatch<const N: usize = DEFAULT_BATCH_SIZE>([EncryptedGate; N]);

impl<const N: usize> EncryptedGateBatch<N> {
    /// Creates a new batch of encrypted gates.
    pub fn new(batch: [EncryptedGate; N]) -> Self {
        Self(batch)
    }

    /// Returns the inner array.
    pub fn into_array(self) -> [EncryptedGate; N] {
        self.0
    }
}

/// Iterator returned by [`Garbler::generate_batched`].
#[derive(Debug)]
pub struct EncryptedGateBatchIter<'a, I: Iterator, const N: usize = DEFAULT_BATCH_SIZE>(
    EncryptedGateIter<'a, I>,
);

impl<'a, I, const N: usize> EncryptedGateBatchIter<'a, I, N>
where
    I: Iterator<Item = &'a Gate>,
{
    /// Returns `true` if the garbler has more encrypted gates to generate.
    pub fn has_gates(&self) -> bool {
        self.0.has_gates()
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(self) -> Result<GarblerOutput, GarblerError> {
        self.0.finish()
    }
}

impl<'a, I, const N: usize> Iterator for EncryptedGateBatchIter<'a, I, N>
where
    I: Iterator<Item = &'a Gate>,
{
    type Item = EncryptedGateBatch<N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_gates() {
            return None;
        }

        let mut batch = [EncryptedGate::default(); N];
        let mut i = 0;
        for gate in self.0.by_ref() {
            batch[i] = gate;
            i += 1;

            if i == N {
                break;
            }
        }

        Some(EncryptedGateBatch::new(batch))
    }
}

// ============================================================================
// Single gate garbling (internal)
// ============================================================================

/// Garble a single AND gate using the Three Halves scheme.
///
/// # Arguments
/// * `cipher` - The fixed-key AES cipher
/// * `w_a` - The 0-color-bit label for input A (has LSB = 0)
/// * `w_b` - The 0-color-bit label for input B (has LSB = 0)
/// * `pi_a` - The permute bit for input A
/// * `pi_b` - The permute bit for input B
/// * `delta` - The global offset
/// * `gid` - The gate ID
/// * `rand_bits` - Pre-generated random bits for control matrix
///
/// # Returns
/// * `(W_c, pi_c, gate)` where W_c has LSB = 0 and pi_c is the output permute
///   bit
fn and_gate(
    cipher: &FixedKeyAes,
    w_a: &Block,
    w_b: &Block,
    pi_a: bool,
    pi_b: bool,
    delta: &Delta,
    gid: usize,
    rand_bits: [bool; 2],
) -> (Block, bool, EncryptedGate) {
    let delta_block = *delta.as_block();

    //

    // Compute the 6 hash values using 0-color-bit labels
    // The paper uses A_0, A_1 = A_0 ⊕ Δ, etc.
    let hashes = compute_hashes(cipher, *w_a, *w_b, delta_block, gid);

    // Slice the input labels
    let a0_sliced = SlicedLabel::from_block(*w_a);
    let b0_sliced = SlicedLabel::from_block(*w_b);
    let delta_sliced = SlicedLabel::from_block(delta_block);

    // Sample the randomized control matrix R for AND gate (ODD mode)
    //
    // The permute bits define the relationship between color bits and logical values:
    // - Color bit i corresponds to logical value (i ⊕ π_a) for input A
    // - Color bit j corresponds to logical value (j ⊕ π_b) for input B
    //
    // Compute the R matrix index for both sample_r_odd lookup and apply_r_to_inputs.
    // Index = (pi_a << 3) | (pi_b << 2) | (r0 << 1) | r1
    let r_index = (pi_a as usize) << 3
        | (pi_b as usize) << 2
        | (rand_bits[0] as usize) << 1
        | rand_bits[1] as usize;
    let r_bar = sample_r_odd(pi_a, pi_b, rand_bits);

    // Compute M · H⃗ (hash contribution)
    let m_times_h = apply_m_to_hashes(&hashes);

    // Compute R · [A₀; B₀; Δ] (input contribution)
    // Uses precomputed branchless operations instead of 48-branch matrix multiply
    let r_times_input = apply_r_to_inputs(r_index, &a0_sliced, &b0_sliced, &delta_sliced);

    // Compute RHS = M·H ⊕ R·input
    let mut rhs = [[0u8; 8]; 8];
    for i in 0..8 {
        rhs[i] = m_times_h[i];
        xor_assign_8(&mut rhs[i], &r_times_input[i]);
    }

    // Solve for [C; G⃗]
    let output = solve_for_output(&rhs, &delta_sliced, pi_a, pi_b);

    // Extract output label C from the linear algebra
    let c = SlicedLabel::new(output[0], output[1]).to_block();

    // Per the paper (Section 5.3):
    // π_c := lsb(C)
    // W_c := C ⊕ π_c·Δ
    // This ensures W_c has LSB = 0
    let pi_c = c.lsb();
    let w_c = if pi_c { c ^ delta_block } else { c };

    let gate = ThreeHalvesGate::new(output[2], output[3], output[4]);
    let control_bits = ControlBits::new(r_bar);

    (w_c, pi_c, EncryptedGate::new(gate, control_bits))
}

/// Compute the 6 hash values needed for garbling.
fn compute_hashes(
    cipher: &FixedKeyAes,
    a0: Block,
    b0: Block,
    delta: Block,
    gid: usize,
) -> [SlicedLabel; 6] {
    let a1 = a0 ^ delta;
    let b1 = b0 ^ delta;

    let tweak = Block::new((gid as u128).to_be_bytes());
    let mut blocks = [a0, a1, b0, b1, a0 ^ b0, a0 ^ b1];
    cipher.rtccr_many(&[tweak; 6], &mut blocks);

    [
        SlicedLabel::from_block(blocks[0]),
        SlicedLabel::from_block(blocks[1]),
        SlicedLabel::from_block(blocks[2]),
        SlicedLabel::from_block(blocks[3]),
        SlicedLabel::from_block(blocks[4]),
        SlicedLabel::from_block(blocks[5]),
    ]
}

// ============================================================================
// Precomputed M Matrix Application
// ============================================================================
//
// # Why This Optimization Exists
//
// Similar to `apply_r_to_inputs`, the naive `apply_m_to_hashes` function has
// 48 conditional branches (8 rows × 6 columns). While M is a fixed matrix
// (unlike R which has 16 variants), we can still eliminate branches using
// precomputed bitmasks.
//
// # The M Matrix (from Paper Page 10, Table 2)
//
// M specifies which hash values to XOR for each evaluation equation:
//
// ```text
// M = [ 1 0 0 0 1 0 ]  <- (0,0) left:  H(A₀) ⊕ H(A₀⊕B₀)
//     [ 0 0 1 0 1 0 ]  <- (0,0) right: H(B₀) ⊕ H(A₀⊕B₀)
//     [ 1 0 0 0 0 1 ]  <- (0,1) left:  H(A₀) ⊕ H(A₀⊕B₁)
//     [ 0 0 0 1 0 1 ]  <- (0,1) right: H(B₁) ⊕ H(A₀⊕B₁)
//     [ 0 1 0 0 0 1 ]  <- (1,0) left:  H(A₁) ⊕ H(A₀⊕B₁)
//     [ 0 0 1 0 0 1 ]  <- (1,0) right: H(B₀) ⊕ H(A₀⊕B₁)
//     [ 0 1 0 0 1 0 ]  <- (1,1) left:  H(A₁) ⊕ H(A₀⊕B₀)
//     [ 0 0 0 1 1 0 ]  <- (1,1) right: H(B₁) ⊕ H(A₀⊕B₀)
// ```
//
// Columns: [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]
//
// # Bitmask Encoding
//
// Each u8 bitmask has bits [0..5] indicating which columns to XOR:
//   bit 0 = H(A₀)
//   bit 1 = H(A₁)
//   bit 2 = H(B₀)
//   bit 3 = H(B₁)
//   bit 4 = H(A₀⊕B₀)
//   bit 5 = H(A₀⊕B₁)

/// Precomputed column bitmasks for the M matrix.
///
/// Since M is a fixed constant matrix, we only need one set of 8 bitmasks
/// (one per row), unlike R which has 16 variants.
///
/// Generated from the M matrix in matrices.rs at compile time.
const M_COLUMN_MASKS: [u8; 8] = {
    // Convert M matrix rows to bitmasks
    // M[row][col] == 1 means bit `col` is set in the mask
    //
    // M = [[1,0,0,0,1,0], [0,0,1,0,1,0], [1,0,0,0,0,1], [0,0,0,1,0,1],
    //      [0,1,0,0,0,1], [0,0,1,0,0,1], [0,1,0,0,1,0], [0,0,0,1,1,0]]
    [
        0b_010001, // Row 0: cols 0,4 → bits 0,4 = 1 + 16 = 17 = 0x11
        0b_010100, // Row 1: cols 2,4 → bits 2,4 = 4 + 16 = 20 = 0x14
        0b_100001, // Row 2: cols 0,5 → bits 0,5 = 1 + 32 = 33 = 0x21
        0b_101000, // Row 3: cols 3,5 → bits 3,5 = 8 + 32 = 40 = 0x28
        0b_100010, // Row 4: cols 1,5 → bits 1,5 = 2 + 32 = 34 = 0x22
        0b_100100, // Row 5: cols 2,5 → bits 2,5 = 4 + 32 = 36 = 0x24
        0b_010010, // Row 6: cols 1,4 → bits 1,4 = 2 + 16 = 18 = 0x12
        0b_011000, // Row 7: cols 3,4 → bits 3,4 = 8 + 16 = 24 = 0x18
    ]
};

/// Apply matrix M to hash vector using precomputed branchless operations.
///
/// # How It Works
///
/// Instead of:
/// ```ignore
/// for row in 0..8 {
///     for col in 0..6 {
///         if M[row][col] == 1 { result[row] ^= hashes[col].left; }  // 48 branches!
///     }
/// }
/// ```
///
/// We use branchless masking with precomputed bitmasks:
/// ```ignore
/// for row in 0..8 {
///     let m = M_COLUMN_MASKS[row];
///     result[row] = (hashes[0] & expand(m, 0))
///                 ^ (hashes[1] & expand(m, 1))
///                 ^ ... ;
/// }
/// ```
///
/// The `expand(mask, bit)` converts a single bit to a full u64 mask:
/// - 0 → 0x0000000000000000
/// - 1 → 0xFFFFFFFFFFFFFFFF
///
/// # Arguments
///
/// * `hashes` - The 6 RTCCR hash values [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]
///
/// # Returns
///
/// 8×8 byte array representing M × H⃗, where each row is a 64-bit value.
#[inline]
fn apply_m_to_hashes(hashes: &[SlicedLabel; 6]) -> [[u8; 8]; 8] {
    // Pack hash left-halves as u64 for efficient XOR operations
    let inputs: [u64; 6] = [
        u64::from_le_bytes(hashes[0].left),
        u64::from_le_bytes(hashes[1].left),
        u64::from_le_bytes(hashes[2].left),
        u64::from_le_bytes(hashes[3].left),
        u64::from_le_bytes(hashes[4].left),
        u64::from_le_bytes(hashes[5].left),
    ];

    let mut result = [[0u8; 8]; 8];

    for row in 0..8 {
        let m = M_COLUMN_MASKS[row] as u64;

        // Branchless expansion: convert each bit to a full u64 mask
        // ((m >> bit) & 1) is 0 or 1
        // .wrapping_neg() converts: 0 → 0, 1 → 0xFFFFFFFFFFFFFFFF
        let row_result = (inputs[0] & ((m >> 0) & 1).wrapping_neg())
            ^ (inputs[1] & ((m >> 1) & 1).wrapping_neg())
            ^ (inputs[2] & ((m >> 2) & 1).wrapping_neg())
            ^ (inputs[3] & ((m >> 3) & 1).wrapping_neg())
            ^ (inputs[4] & ((m >> 4) & 1).wrapping_neg())
            ^ (inputs[5] & ((m >> 5) & 1).wrapping_neg());

        result[row] = row_result.to_le_bytes();
    }

    result
}

// ============================================================================
// Precomputed R Matrix Application
// ============================================================================
//
// # Why This Optimization Exists
//
// The naive `apply_r_to_inputs` function performs an 8×6 matrix-vector multiply
// where each entry is a conditional XOR:
//
//   for row in 0..8:
//       for col in 0..6:
//           if R[row][col]:
//               result[row] ^= inputs[col]
//
// This results in 48 conditional branches per AND gate. With millions of gates,
// branch mispredictions become a significant bottleneck (~33% of garbling time
// in profiling).
//
// # The Optimization
//
// The control matrix R comes from `sample_r_odd`, which returns one of only
// **16 possible matrices** (4 permute bit combinations × 4 random bit combinations).
// Instead of runtime conditionals, we:
//
// 1. Precompute a bitmask for each row of each R variant, indicating which
//    columns (inputs) to XOR together
// 2. At runtime, use branchless masking: `result ^= input & mask`
//
// This eliminates all branches and reduces the operation to pure XOR/AND.
//
// # Bitmask Format
//
// Each u8 bitmask has bits [0..5] corresponding to the 6 input columns:
//   bit 0 = A₀_L (a0.left)
//   bit 1 = A₀_R (a0.right)
//   bit 2 = B₀_L (b0.left)
//   bit 3 = B₀_R (b0.right)
//   bit 4 = Δ_L  (delta.left)
//   bit 5 = Δ_R  (delta.right)
//
// # Table Index
//
// Index = (pi_a << 3) | (pi_b << 2) | (r0 << 1) | r1
// Same indexing as SAMPLE_R_ODD_TABLE in control.rs.

/// Precomputed column bitmasks for each R matrix variant.
///
/// `R_COLUMN_MASKS[variant][row]` is a u8 where bit `col` indicates whether
/// R[row][col] is true (i.e., whether to XOR input[col] into result[row]).
///
/// Generated from SAMPLE_R_ODD_TABLE - each entry is the same R matrix but
/// encoded as bitmasks for branchless application.
const R_COLUMN_MASKS: [[u8; 8]; 16] = {
    // Helper to convert a bool row [c0,c1,c2,c3,c4,c5] to bitmask
    const fn row_to_mask(row: [bool; 6]) -> u8 {
        (row[0] as u8)
            | ((row[1] as u8) << 1)
            | ((row[2] as u8) << 2)
            | ((row[3] as u8) << 3)
            | ((row[4] as u8) << 4)
            | ((row[5] as u8) << 5)
    }

    // Helper to convert full 8×6 R matrix to 8 bitmasks
    const fn matrix_to_masks(r: [[bool; 6]; 8]) -> [u8; 8] {
        [
            row_to_mask(r[0]),
            row_to_mask(r[1]),
            row_to_mask(r[2]),
            row_to_mask(r[3]),
            row_to_mask(r[4]),
            row_to_mask(r[5]),
            row_to_mask(r[6]),
            row_to_mask(r[7]),
        ]
    }

    const F: bool = false;
    const T: bool = true;

    // These are the exact same R matrices as SAMPLE_R_ODD_TABLE in control.rs,
    // just converted to bitmask form at compile time.
    [
        // Index 0: pi_a=F, pi_b=F, r0=F, r1=F
        matrix_to_masks([
            [F, F, T, F, F, F], [F, T, F, F, F, F], [T, F, T, T, T, T], [F, T, T, T, T, T],
            [T, T, T, F, T, T], [T, T, F, T, T, T], [F, T, T, T, T, F], [T, T, T, F, F, T],
        ]),
        // Index 1: pi_a=F, pi_b=F, r0=F, r1=T
        matrix_to_masks([
            [T, F, T, T, F, F], [F, F, T, T, F, F], [F, F, T, F, T, F], [F, F, F, F, F, F],
            [F, T, T, T, F, T], [T, F, T, F, T, F], [T, T, T, F, F, T], [T, F, F, T, T, T],
        ]),
        // Index 2: pi_a=F, pi_b=F, r0=T, r1=F
        matrix_to_masks([
            [T, T, F, F, F, F], [T, T, F, T, F, F], [F, T, F, T, F, T], [T, T, T, F, T, F],
            [F, F, F, F, F, F], [F, T, F, F, F, T], [T, F, F, T, T, T], [F, T, T, T, T, F],
        ]),
        // Index 3: pi_a=F, pi_b=F, r0=T, r1=T
        matrix_to_masks([
            [F, T, F, T, F, F], [T, F, T, F, F, F], [T, T, F, F, F, F], [T, F, F, T, F, T],
            [T, F, F, T, T, F], [F, F, T, T, F, F], [F, F, F, F, F, F], [F, F, F, F, F, F],
        ]),
        // Index 4: pi_a=F, pi_b=T, r0=F, r1=F
        matrix_to_masks([
            [F, F, T, F, F, F], [F, T, F, F, F, F], [F, T, F, T, F, T], [T, T, T, F, T, F],
            [T, F, F, T, T, F], [F, F, T, T, F, F], [T, T, T, F, F, T], [T, F, F, T, T, T],
        ]),
        // Index 5: pi_a=F, pi_b=T, r0=F, r1=T
        matrix_to_masks([
            [T, F, T, T, F, F], [F, F, T, T, F, F], [T, T, F, F, F, F], [T, F, F, T, F, T],
            [F, F, F, F, F, F], [F, T, F, F, F, T], [F, T, T, T, T, F], [T, T, T, F, F, T],
        ]),
        // Index 6: pi_a=F, pi_b=T, r0=T, r1=F
        matrix_to_masks([
            [T, T, F, F, F, F], [T, T, F, T, F, F], [T, F, T, T, T, T], [F, T, T, T, T, T],
            [F, T, T, T, F, T], [T, F, T, F, T, F], [F, F, F, F, F, F], [F, F, F, F, F, F],
        ]),
        // Index 7: pi_a=F, pi_b=T, r0=T, r1=T
        matrix_to_masks([
            [F, T, F, T, F, F], [T, F, T, F, F, F], [F, F, T, F, T, F], [F, F, F, F, F, F],
            [T, T, T, F, T, T], [T, T, F, T, T, T], [T, F, F, T, T, T], [F, T, T, T, T, F],
        ]),
        // Index 8: pi_a=T, pi_b=F, r0=F, r1=F
        matrix_to_masks([
            [F, F, T, F, F, F], [F, T, F, F, F, F], [T, T, F, F, F, F], [T, F, F, T, F, T],
            [F, T, T, T, F, T], [T, F, T, F, T, F], [T, F, F, T, T, T], [F, T, T, T, T, F],
        ]),
        // Index 9: pi_a=T, pi_b=F, r0=F, r1=T
        matrix_to_masks([
            [T, F, T, T, F, F], [F, F, T, T, F, F], [F, T, F, T, F, T], [T, T, T, F, T, F],
            [T, T, T, F, T, T], [T, T, F, T, T, T], [F, F, F, F, F, F], [F, F, F, F, F, F],
        ]),
        // Index 10: pi_a=T, pi_b=F, r0=T, r1=F
        matrix_to_masks([
            [T, T, F, F, F, F], [T, T, F, T, F, F], [F, F, T, F, T, F], [F, F, F, F, F, F],
            [T, F, F, T, T, F], [F, F, T, T, F, F], [F, T, T, T, T, F], [T, T, T, F, F, T],
        ]),
        // Index 11: pi_a=T, pi_b=F, r0=T, r1=T
        matrix_to_masks([
            [F, T, F, T, F, F], [T, F, T, F, F, F], [T, F, T, T, T, T], [F, T, T, T, T, T],
            [F, F, F, F, F, F], [F, T, F, F, F, T], [T, T, T, F, F, T], [T, F, F, T, T, T],
        ]),
        // Index 12: pi_a=T, pi_b=T, r0=F, r1=F
        matrix_to_masks([
            [F, F, T, F, F, F], [F, T, F, F, F, F], [F, F, T, F, T, F], [F, F, F, F, F, F],
            [F, F, F, F, F, F], [F, T, F, F, F, T], [F, F, F, F, F, F], [F, F, F, F, F, F],
        ]),
        // Index 13: pi_a=T, pi_b=T, r0=F, r1=T
        matrix_to_masks([
            [T, F, T, T, F, F], [F, F, T, T, F, F], [T, F, T, T, T, T], [F, T, T, T, T, T],
            [T, F, F, T, T, F], [F, F, T, T, F, F], [T, F, F, T, T, T], [F, T, T, T, T, F],
        ]),
        // Index 14: pi_a=T, pi_b=T, r0=T, r1=F
        matrix_to_masks([
            [T, T, F, F, F, F], [T, T, F, T, F, F], [T, T, F, F, F, F], [T, F, F, T, F, T],
            [T, T, T, F, T, T], [T, T, F, T, T, T], [T, T, T, F, F, T], [T, F, F, T, T, T],
        ]),
        // Index 15: pi_a=T, pi_b=T, r0=T, r1=T
        matrix_to_masks([
            [F, T, F, T, F, F], [T, F, T, F, F, F], [F, T, F, T, F, T], [T, T, T, F, T, F],
            [F, T, T, T, F, T], [T, F, T, F, T, F], [F, T, T, T, T, F], [T, T, T, F, F, T],
        ]),
    ]
};

/// Apply control matrix R to input labels using precomputed branchless operations.
///
/// # How It Works
///
/// Instead of:
/// ```ignore
/// for row in 0..8 {
///     for col in 0..6 {
///         if R[row][col] { result[row] ^= inputs[col]; }  // 48 branches!
///     }
/// }
/// ```
///
/// We use:
/// ```ignore
/// for row in 0..8 {
///     let mask = R_COLUMN_MASKS[variant][row];
///     // Branchless: XOR each input masked by whether its bit is set
///     result[row] = (inputs[0] & expand(mask, 0))
///                 ^ (inputs[1] & expand(mask, 1))
///                 ^ ... ;
/// }
/// ```
///
/// The `expand(mask, bit)` function converts a single bit to a full u64 mask:
/// - If bit is 0: returns 0x0000000000000000
/// - If bit is 1: returns 0xFFFFFFFFFFFFFFFF
///
/// This is done with `((mask >> bit) & 1).wrapping_neg()` which is branchless.
///
/// # Arguments
///
/// * `r_index` - Index into R_COLUMN_MASKS (same as SAMPLE_R_ODD_TABLE index)
/// * `a0` - Input wire A's 0-label, sliced into left/right halves
/// * `b0` - Input wire B's 0-label, sliced into left/right halves
/// * `delta` - Global offset Δ, sliced into left/right halves
///
/// # Returns
///
/// 8×8 byte array representing R × [A₀; B₀; Δ], where each row is a 64-bit value
/// stored as [u8; 8].
#[inline]
fn apply_r_to_inputs(
    r_index: usize,
    a0: &SlicedLabel,
    b0: &SlicedLabel,
    delta: &SlicedLabel,
) -> [[u8; 8]; 8] {
    // Pack inputs as u64 for efficient XOR operations (instead of byte-by-byte)
    let inputs: [u64; 6] = [
        u64::from_le_bytes(a0.left),
        u64::from_le_bytes(a0.right),
        u64::from_le_bytes(b0.left),
        u64::from_le_bytes(b0.right),
        u64::from_le_bytes(delta.left),
        u64::from_le_bytes(delta.right),
    ];

    let masks = &R_COLUMN_MASKS[r_index];
    let mut result = [[0u8; 8]; 8];

    for row in 0..8 {
        let m = masks[row] as u64;

        // Branchless expansion: convert each bit to a full u64 mask
        // ((m >> bit) & 1) is 0 or 1
        // .wrapping_neg() converts: 0 → 0, 1 → 0xFFFFFFFFFFFFFFFF
        let row_result = (inputs[0] & ((m >> 0) & 1).wrapping_neg())
            ^ (inputs[1] & ((m >> 1) & 1).wrapping_neg())
            ^ (inputs[2] & ((m >> 2) & 1).wrapping_neg())
            ^ (inputs[3] & ((m >> 3) & 1).wrapping_neg())
            ^ (inputs[4] & ((m >> 4) & 1).wrapping_neg())
            ^ (inputs[5] & ((m >> 5) & 1).wrapping_neg());

        result[row] = row_result.to_le_bytes();
    }

    result
}

/// Solve for [C_L, C_R, G₀, G₁, G₂] from RHS.
///
/// For AND gates, the identity block (true output) is at position (!pi_a, !pi_b).
fn solve_for_output(
    rhs: &[[u8; 8]; 8],
    delta: &SlicedLabel,
    pi_a: bool,
    pi_b: bool,
) -> [[u8; 8]; 5] {
    let mut rhs_adjusted = *rhs;

    // Adjust for truth table: identity block is at position (!pi_a, !pi_b)
    let true_i = !pi_a as usize;
    let true_j = !pi_b as usize;
    let true_ij = (true_i << 1) | true_j;
    let row_l = 2 * true_ij;
    let row_r = 2 * true_ij + 1;
    xor_assign_8(&mut rhs_adjusted[row_l], &delta.left);
    xor_assign_8(&mut rhs_adjusted[row_r], &delta.right);

    let mut result = [[0u8; 8]; 5];

    // C_L = RHS[0]
    result[0] = rhs_adjusted[0];
    // C_R = RHS[1]
    result[1] = rhs_adjusted[1];
    // G₀ = RHS[0] ⊕ RHS[1] ⊕ RHS[4] ⊕ RHS[5]
    for k in 0..8 {
        result[2][k] =
            rhs_adjusted[0][k] ^ rhs_adjusted[1][k] ^ rhs_adjusted[4][k] ^ rhs_adjusted[5][k];
    }
    // G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
    for k in 0..8 {
        result[3][k] =
            rhs_adjusted[0][k] ^ rhs_adjusted[1][k] ^ rhs_adjusted[2][k] ^ rhs_adjusted[3][k];
    }
    // G₂ = RHS[4] ⊕ RHS[6]
    for k in 0..8 {
        result[4][k] = rhs_adjusted[4][k] ^ rhs_adjusted[6][k];
    }

    result
}

/// XOR-assign two 8-byte arrays.
#[inline]
pub(crate) fn xor_assign_8(a: &mut [u8; 8], b: &[u8; 8]) {
    for i in 0..8 {
        a[i] ^= b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::three_halves::matrices::M;
    use mpz_circuits::circuits::xor;
    use mpz_core::aes::FIXED_KEY_AES;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    /// Naive implementation of apply_r_to_inputs for testing
    fn apply_r_to_inputs_naive(
        r: &[[bool; 6]; 8],
        a0: &SlicedLabel,
        b0: &SlicedLabel,
        delta: &SlicedLabel,
    ) -> [[u8; 8]; 8] {
        let inputs: [[u8; 8]; 6] = [
            a0.left,
            a0.right,
            b0.left,
            b0.right,
            delta.left,
            delta.right,
        ];

        let mut result = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..6 {
                if r[row][col] {
                    xor_assign_8(&mut result[row], &inputs[col]);
                }
            }
        }
        result
    }

    /// Verify R_COLUMN_MASKS matches the original sample_r_odd matrices
    #[test]
    fn test_r_column_masks_correctness() {
        // Test all 16 variants with random inputs
        let mut rng = ChaCha12Rng::seed_from_u64(12345);
        let a0 = SlicedLabel::from_block(Block::random(&mut rng));
        let b0 = SlicedLabel::from_block(Block::random(&mut rng));
        let delta = SlicedLabel::from_block(Block::random(&mut rng));

        for pi_a in [false, true] {
            for pi_b in [false, true] {
                for r0 in [false, true] {
                    for r1 in [false, true] {
                        let r_index = (pi_a as usize) << 3
                            | (pi_b as usize) << 2
                            | (r0 as usize) << 1
                            | r1 as usize;

                        let (r_matrix, _) = crate::three_halves::control::sample_r_odd_with_r(pi_a, pi_b, [r0, r1]);

                        let naive_result = apply_r_to_inputs_naive(&r_matrix, &a0, &b0, &delta);
                        let fast_result = apply_r_to_inputs(r_index, &a0, &b0, &delta);

                        assert_eq!(
                            naive_result, fast_result,
                            "Mismatch for index {} (pi_a={}, pi_b={}, r0={}, r1={})",
                            r_index, pi_a, pi_b, r0, r1
                        );
                    }
                }
            }
        }
    }

    /// Naive implementation of apply_m_to_hashes for testing
    fn apply_m_to_hashes_naive(hashes: &[SlicedLabel; 6]) -> [[u8; 8]; 8] {
        let mut result = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..6 {
                if M[row][col] == 1 {
                    xor_assign_8(&mut result[row], &hashes[col].left);
                }
            }
        }
        result
    }

    /// Verify M_COLUMN_MASKS matches the original M matrix
    #[test]
    fn test_m_column_masks_correctness() {
        let mut rng = ChaCha12Rng::seed_from_u64(54321);

        // Generate random hashes
        let hashes: [SlicedLabel; 6] = [
            SlicedLabel::from_block(Block::random(&mut rng)),
            SlicedLabel::from_block(Block::random(&mut rng)),
            SlicedLabel::from_block(Block::random(&mut rng)),
            SlicedLabel::from_block(Block::random(&mut rng)),
            SlicedLabel::from_block(Block::random(&mut rng)),
            SlicedLabel::from_block(Block::random(&mut rng)),
        ];

        let naive_result = apply_m_to_hashes_naive(&hashes);
        let fast_result = apply_m_to_hashes(&hashes);

        assert_eq!(naive_result, fast_result, "M matrix application mismatch");
    }

    #[test]
    fn test_garbling_deterministic() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut a0 = Block::random(&mut rng);
        let mut b0 = Block::random(&mut rng);
        let mut delta = Block::random(&mut rng);
        a0.set_lsb(false);
        b0.set_lsb(false);
        delta.set_lsb(true);

        // Use the same random bits for both calls
        let rand_bits = [true, false];

        // Test with permute bits both false (since labels have LSB=0)
        let (z1, pi_z1, gate1) = and_gate(
            cipher,
            &a0,
            &b0,
            false,
            false,
            &Delta::new(delta),
            1,
            rand_bits,
        );
        let (z2, pi_z2, gate2) = and_gate(
            cipher,
            &a0,
            &b0,
            false,
            false,
            &Delta::new(delta),
            1,
            rand_bits,
        );

        assert_eq!(z1, z2);
        assert_eq!(pi_z1, pi_z2);
        assert_eq!(gate1, gate2);
    }

    #[test]
    fn test_garble_xor_circuit() {
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let circ = xor(8);

        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        let delta = Delta::new(delta);

        // Three Halves requires input keys to have LSB = 0
        let input_keys: Vec<Key> = (0..circ.inputs().len())
            .map(|_| {
                let mut block: Block = rng.random();
                block.set_lsb(false);
                block.into()
            })
            .collect();

        let mut gb = Garbler::default();
        let iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

        // XOR circuit has no AND gates
        assert!(!iter.has_gates());

        let output = iter.finish().unwrap();
        assert_eq!(output.outputs.len(), circ.outputs().len());
    }
}
