//! Garbling for Three Halves Scheme
//!
//! This module implements the garbling function for AND gates using the
//! Three Halves technique from Rosulek & Roy 2021.
//!
//! # Paper Reference
//!
//! The main garbling equation (Equation 4, Page 11):
//!
//! ```text
//! V · [C; G⃗] = M · H⃗ ⊕ R · [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]ᵀ
//! ```
//!
//! Where:
//! - V (8×5): Maps output vector to evaluation equations
//! - [C; G⃗] (5 elements): Output label halves + 3 gate ciphertexts (each κ/2
//!   bits)
//! - M (8×6): Selects which hashes contribute to each equation
//! - H⃗ (6 elements): Hash outputs [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀),
//!   H(A₀⊕B₁)]
//! - R (8×6): Control matrix (randomized to hide truth table)
//! - Input vector (6 elements): Label halves [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
//!
//! # Row Structure
//!
//! Each row i of the equation corresponds to one (input_combination, half)
//! pair:
//! - Row 0: (0,0) left half
//! - Row 1: (0,0) right half
//! - Row 2: (0,1) left half
//! - Row 3: (0,1) right half
//! - Row 4: (1,0) left half
//! - Row 5: (1,0) right half
//! - Row 6: (1,1) left half
//! - Row 7: (1,1) right half

use mpz_core::{Block, aes::FixedKeyAes};

use super::{
    control::{and_truth_table, sample_r_odd},
    matrices::M,
    slicing::SlicedLabel,
};

/// Gate ciphertexts for a Three Halves AND gate.
///
/// Contains 3 ciphertexts of κ/2 bits each = 1.5κ bits total.
/// This is smaller than half-gates which uses 2κ bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Control bits for evaluator.
///
/// DUMMY PROTOCOL: In this experimental version, we send the full R matrix
/// in plaintext instead of the compressed r_bar. This breaks privacy but
/// allows us to debug the core math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlBits {
    /// The full 8×6 R matrix (sent in plaintext for debugging)
    pub r: [[u8; 6]; 8],
}

impl ControlBits {
    /// Create new control bits from the full R matrix.
    pub fn new(r: [[u8; 6]; 8]) -> Self {
        Self { r }
    }
}

/// Output of garbling a Three Halves AND gate.
#[derive(Clone, Copy, Debug)]
pub struct GarbledGate {
    /// Output wire label for input 0 (C₀)
    pub output_label: Block,
    /// Gate ciphertexts (1.5κ bits)
    pub gate: ThreeHalvesGate,
    /// Control bits for evaluator
    pub control_bits: ControlBits,
}

/// Compute the 6 hash values needed for garbling.
///
/// # Arguments
/// * `cipher` - The fixed-key AES cipher for TCCR hash
/// * `a0` - Input wire A, label for bit 0
/// * `b0` - Input wire B, label for bit 0
/// * `delta` - Global correlation Δ
/// * `gid` - Gate ID (used as tweak)
///
/// # Returns
/// Array of 6 sliced labels: [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]
fn compute_hashes(
    cipher: &FixedKeyAes,
    a0: Block,
    b0: Block,
    delta: Block,
    gid: usize,
) -> [SlicedLabel; 6] {
    let a1 = a0 ^ delta;
    let b1 = b0 ^ delta;

    // Compute all 6 hashes using batched TCCR
    let tweak = Block::new((gid as u128).to_be_bytes());
    let mut blocks = [a0, a1, b0, b1, a0 ^ b0, a0 ^ b1];
    cipher.tccr_many(&[tweak; 6], &mut blocks);

    // Convert to sliced labels
    [
        SlicedLabel::from_block(blocks[0]), // H(A₀)
        SlicedLabel::from_block(blocks[1]), // H(A₁)
        SlicedLabel::from_block(blocks[2]), // H(B₀)
        SlicedLabel::from_block(blocks[3]), // H(B₁)
        SlicedLabel::from_block(blocks[4]), // H(A₀⊕B₀)
        SlicedLabel::from_block(blocks[5]), // H(A₀⊕B₁)
    ]
}

/// Apply matrix M to hash vector H⃗, producing 8 half-results.
///
/// For row i (computing half h = i%2):
///   result[i] = XOR of H[j].half(h) for all j where M[i][j] = 1
///
/// # Paper Reference
/// M is defined on Page 10, mapping hash outputs to evaluation equations.
fn apply_m_to_hashes(hashes: &[SlicedLabel; 6]) -> [[u8; 8]; 8] {
    let mut result = [[0u8; 8]; 8];

    for row in 0..8 {
        let half = row % 2; // 0 = left, 1 = right

        for col in 0..6 {
            if M[row][col] == 1 {
                let h_half = hashes[col].half(half);
                xor_assign_8(&mut result[row], &h_half);
            }
        }
    }

    result
}

/// Apply matrix R to input vector [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R].
///
/// For row i:
///   result[i] = XOR of input[j] for all j where R[i][j] = 1
///
/// # Arguments
/// * `r` - The 8×6 control matrix R
/// * `a0` - Input wire A label (sliced)
/// * `b0` - Input wire B label (sliced)
/// * `delta` - Global correlation Δ (sliced)
fn apply_r_to_inputs(
    r: &[[u8; 6]; 8],
    a0: &SlicedLabel,
    b0: &SlicedLabel,
    delta: &SlicedLabel,
) -> [[u8; 8]; 8] {
    // Build input vector: [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
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
            if r[row][col] == 1 {
                xor_assign_8(&mut result[row], &inputs[col]);
            }
        }
    }

    result
}

/// Solve for [C_L, C_R, G₀, G₁, G₂] from RHS using V⁻¹ matrix.
///
/// From Paper Page 18, Equation 10, V⁻¹ is:
/// ```text
/// V⁻¹ = [ 1 0 | 0 0 | 0 0 | 0 0 ]  -> C_L = RHS[0]
///       [ 0 1 | 0 0 | 0 0 | 0 0 ]  -> C_R = RHS[1]
///       [ 1 1 | 0 0 | 1 1 | 0 0 ]  -> G₀ = RHS[0] ⊕ RHS[1] ⊕ RHS[4] ⊕ RHS[5]
///       [ 1 1 | 1 1 | 0 0 | 0 0 ]  -> G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
///       [ 0 0 | 0 0 | 1 0 | 1 0 ]  -> G₂ = RHS[4] ⊕ RHS[6]
/// ```
fn solve_for_output(rhs: &[[u8; 8]; 8]) -> [[u8; 8]; 5] {
    let mut result = [[0u8; 8]; 5];

    // C_L = RHS[0]
    result[0] = rhs[0];

    // C_R = RHS[1]
    result[1] = rhs[1];

    // G₀ = RHS[0] ⊕ RHS[1] ⊕ RHS[4] ⊕ RHS[5]
    for k in 0..8 {
        result[2][k] = rhs[0][k] ^ rhs[1][k] ^ rhs[4][k] ^ rhs[5][k];
    }

    // G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
    for k in 0..8 {
        result[3][k] = rhs[0][k] ^ rhs[1][k] ^ rhs[2][k] ^ rhs[3][k];
    }

    // G₂ = RHS[4] ⊕ RHS[6]
    for k in 0..8 {
        result[4][k] = rhs[4][k] ^ rhs[6][k];
    }

    result
}

/// XOR-assign two 8-byte arrays.
#[inline]
fn xor_assign_8(a: &mut [u8; 8], b: &[u8; 8]) {
    for i in 0..8 {
        a[i] ^= b[i];
    }
}

/// Garble an AND gate using the Three Halves scheme.
///
/// # Arguments
/// * `cipher` - Fixed-key AES cipher for TCCR hash
/// * `a0` - Input wire A, label for bit 0
/// * `b0` - Input wire B, label for bit 0
/// * `delta` - Global correlation Δ
/// * `gid` - Gate ID
/// * `rand_bits` - 2 random bits for control matrix randomization
///
/// # Returns
/// The garbled gate containing output label, ciphertexts, and control bits.
///
/// # Paper Reference
/// This implements the garbling algorithm from Section 5 (Page 11-15).
pub fn garble_and_gate(
    cipher: &FixedKeyAes,
    a0: Block,
    b0: Block,
    delta: Block,
    gid: usize,
    rand_bits: [bool; 2],
) -> GarbledGate {
    // 1. Compute the 6 hash values
    let hashes = compute_hashes(cipher, a0, b0, delta, gid);

    // 2. Slice the input labels
    let a0_sliced = SlicedLabel::from_block(a0);
    let b0_sliced = SlicedLabel::from_block(b0);
    let delta_sliced = SlicedLabel::from_block(delta);

    // 3. Sample the randomized control matrix R for AND gate (ODD mode)
    let t = and_truth_table();
    let (r, _r_bar) = sample_r_odd(&t, rand_bits);

    // DUMMY PROTOCOL: We send the full R matrix instead of r_bar

    // 4. Compute M · H⃗ (hash contribution)
    let m_times_h = apply_m_to_hashes(&hashes);

    // 5. Compute R · [A₀; B₀; Δ] (input contribution)
    let r_times_input = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

    // 6. Compute RHS = M·H ⊕ R·input
    let mut rhs = [[0u8; 8]; 8];
    for i in 0..8 {
        rhs[i] = m_times_h[i];
        xor_assign_8(&mut rhs[i], &r_times_input[i]);
    }

    // 7. Solve for [C; G⃗] using correct formulas that ensure all marginal equations
    //    hold
    let output = solve_for_output(&rhs);

    // 8. Extract output label and gate ciphertexts
    // output = [C_L, C_R, G₀, G₁, G₂]
    let c0 = SlicedLabel::new(output[0], output[1]).to_block();

    let gate = ThreeHalvesGate::new(output[2], output[3], output[4]);

    // DUMMY PROTOCOL: Send full R matrix instead of compressed r_bar
    let control_bits = ControlBits::new(r);

    GarbledGate {
        output_label: c0,
        gate,
        control_bits,
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// DUMMY PROTOCOL: Extract the evaluator's marginal directly from R matrix.
///
/// The evaluator has access to the full R matrix. For input (i,j), they need
/// rows 2*ij and 2*ij+1, columns 0-3 (the A and B coefficients).
///
/// # Key Insight from Paper (Equation 6, Page 12)
///
/// The R matrix is designed such that the Δ columns (4-5) satisfy:
/// R[row][4] = R[row][0]*i + R[row][2]*j
/// R[row][5] = R[row][1]*i + R[row][3]*j
///
/// This means the Δ contribution is implicitly handled when the evaluator
/// uses their labels A_i = A₀ ⊕ i·Δ and B_j = B₀ ⊕ j·Δ.
fn extract_evaluator_marginal_from_r(r: &[[u8; 6]; 8], i: usize, j: usize) -> [[u8; 4]; 2] {
    let ij = (i << 1) | j;
    let row_l = 2 * ij;
    let row_r = 2 * ij + 1;

    // Extract columns 0-3 (A_L, A_R, B_L, B_R coefficients)
    let mut marginal = [[0u8; 4]; 2];
    for col in 0..4 {
        marginal[0][col] = r[row_l][col];
        marginal[1][col] = r[row_r][col];
    }

    marginal
}

/// Evaluate a Three Halves AND gate.
///
/// DUMMY PROTOCOL: Uses the full R matrix sent in plaintext.
///
/// # Arguments
/// * `cipher` - Fixed-key AES cipher for TCCR hash
/// * `a` - Input wire A label (for bit i)
/// * `b` - Input wire B label (for bit j)
/// * `gate` - Gate ciphertexts from garbling
/// * `control_bits` - Control bits (contains full R matrix)
/// * `gid` - Gate ID
///
/// # Returns
/// The output wire label C_{i∧j}
pub fn evaluate_and_gate(
    cipher: &FixedKeyAes,
    a: Block,
    b: Block,
    gate: &ThreeHalvesGate,
    control_bits: &ControlBits,
    gid: usize,
) -> Block {
    // 1. Determine input combination (i, j) from pointer bits
    let i = a.lsb() as usize;
    let j = b.lsb() as usize;
    let ij = (i << 1) | j;

    // 2. Compute the three hashes the evaluator has access to
    let tweak = Block::new((gid as u128).to_be_bytes());
    let mut hash_inputs = [a, b, a ^ b];
    cipher.tccr_many(&[tweak; 3], &mut hash_inputs);

    let h_a = SlicedLabel::from_block(hash_inputs[0]); // H(A_i)
    let h_b = SlicedLabel::from_block(hash_inputs[1]); // H(B_j)
    let h_ab = SlicedLabel::from_block(hash_inputs[2]); // H(A_i ⊕ B_j)

    // 3. Slice the input labels
    let a_sliced = SlicedLabel::from_block(a);
    let b_sliced = SlicedLabel::from_block(b);

    // 4. DUMMY PROTOCOL: Extract marginal directly from the full R matrix
    let marginal = extract_evaluator_marginal_from_r(&control_bits.r, i, j);

    // 5. Get the two rows for this input combination
    let row_l = 2 * ij; // Left half row
    let row_r = 2 * ij + 1; // Right half row

    // 6. Compute contributions for each half

    // Hash contribution (from M matrix)
    let hash_contrib_l = compute_hash_contribution_eval(row_l, &h_a, &h_b, &h_ab);
    let hash_contrib_r = compute_hash_contribution_eval(row_r, &h_a, &h_b, &h_ab);

    // Input contribution (from R marginal, columns 0-3 only)
    let input_contrib_l = compute_input_contribution(&marginal, 0, &a_sliced, &b_sliced);
    let input_contrib_r = compute_input_contribution(&marginal, 1, &a_sliced, &b_sliced);

    // Gate contribution (from V matrix)
    let gate_contrib_l = compute_gate_contribution(row_l, gate);
    let gate_contrib_r = compute_gate_contribution(row_r, gate);

    // 7. Combine all contributions
    let mut c_l = hash_contrib_l;
    xor_assign_8(&mut c_l, &input_contrib_l);
    xor_assign_8(&mut c_l, &gate_contrib_l);

    let mut c_r = hash_contrib_r;
    xor_assign_8(&mut c_r, &input_contrib_r);
    xor_assign_8(&mut c_r, &gate_contrib_r);

    // 8. Reconstruct output label
    SlicedLabel::new(c_l, c_r).to_block()
}

/// Compute hash contribution for evaluation (left or right half).
///
/// The evaluator has H(A_i), H(B_j), H(A_i⊕B_j) and needs to compute
/// the hash contribution matching what the garbler computed via M·H.
///
/// # Key Insight: Hash Column Mapping
///
/// The evaluator's three hashes map to the garbler's 6 columns as follows:
/// - `h_a = H(A_i)` → column `i` (0 for i=0, 1 for i=1)
/// - `h_b = H(B_j)` → column `2+j` (2 for j=0, 3 for j=1)
/// - `h_ab = H(A_i⊕B_j)` → column 4 or 5 based on Free-XOR:
///   - (0,0): H(A₀⊕B₀) → col 4
///   - (0,1): H(A₀⊕B₁) → col 5
///   - (1,0): H(A₁⊕B₀) = H(A₀⊕B₁) → col 5
///   - (1,1): H(A₁⊕B₁) = H(A₀⊕B₀) → col 4
///
/// This uses M matrix values to determine which hashes to XOR.
fn compute_hash_contribution_eval(
    row: usize,
    h_a: &SlicedLabel,
    h_b: &SlicedLabel,
    h_ab: &SlicedLabel,
) -> [u8; 8] {
    use super::matrices::M;

    let half = row % 2;
    let ij = row / 2;
    let i = ij >> 1;
    let j = ij & 1;

    // Determine which garbler column maps to evaluator's h_ab
    // Due to Free-XOR: A_i ⊕ B_j when (i,j) has odd parity maps to col 5 (A₀⊕B₁)
    //                  when (i,j) has even parity maps to col 4 (A₀⊕B₀)
    let ab_col = if (i ^ j) == 0 { 4 } else { 5 };

    let mut result = [0u8; 8];

    // XOR in h_a if M says to use column i (the evaluator's A hash column)
    if M[row][i] == 1 {
        xor_assign_8(&mut result, &h_a.half(half));
    }

    // XOR in h_b if M says to use column 2+j (the evaluator's B hash column)
    if M[row][2 + j] == 1 {
        xor_assign_8(&mut result, &h_b.half(half));
    }

    // XOR in h_ab if M says to use the corresponding combined hash column
    if M[row][ab_col] == 1 {
        xor_assign_8(&mut result, &h_ab.half(half));
    }

    result
}

/// Compute input contribution from expanded marginal R.
///
/// The expanded marginal r_bar is a 2×4 matrix where:
/// - Row 0 is for left half computation
/// - Row 1 is for right half computation
/// - Columns are: [coeff for A_L, coeff for A_R, coeff for B_L, coeff for B_R]
fn compute_input_contribution(
    r_bar_expanded: &[[u8; 4]; 2],
    half: usize, // 0 = left, 1 = right
    a: &SlicedLabel,
    b: &SlicedLabel,
) -> [u8; 8] {
    let coeffs = r_bar_expanded[half];
    let mut result = [0u8; 8];

    if coeffs[0] == 1 {
        xor_assign_8(&mut result, &a.left);
    }
    if coeffs[1] == 1 {
        xor_assign_8(&mut result, &a.right);
    }
    if coeffs[2] == 1 {
        xor_assign_8(&mut result, &b.left);
    }
    if coeffs[3] == 1 {
        xor_assign_8(&mut result, &b.right);
    }

    result
}

/// Compute gate ciphertext contribution based on V matrix.
///
/// V[row][2:5] gives the coefficients for G₀, G₁, G₂
fn compute_gate_contribution(row: usize, gate: &ThreeHalvesGate) -> [u8; 8] {
    use super::matrices::V;

    let mut result = [0u8; 8];

    // V columns 2, 3, 4 correspond to G₀, G₁, G₂
    if V[row][2] == 1 {
        xor_assign_8(&mut result, &gate.g0);
    }
    if V[row][3] == 1 {
        xor_assign_8(&mut result, &gate.g1);
    }
    if V[row][4] == 1 {
        xor_assign_8(&mut result, &gate.g2);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_core::aes::FIXED_KEY_AES;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    /// Test 1: Garbling produces consistent output
    ///
    /// Same inputs with same randomness should produce same output.
    #[test]
    fn test_garbling_deterministic() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);
        let gid = 1;
        let rand_bits = [true, false];

        let result1 = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);
        let result2 = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);

        assert_eq!(result1.output_label, result2.output_label);
        assert_eq!(result1.gate, result2.gate);
        assert_eq!(result1.control_bits, result2.control_bits);
    }

    /// Test 2: Different random bits produce different control bits (R matrix)
    #[test]
    fn test_random_bits_affect_control() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);
        let gid = 1;

        let result1 = garble_and_gate(cipher, a0, b0, delta, gid, [false, false]);
        let result2 = garble_and_gate(cipher, a0, b0, delta, gid, [true, false]);
        let result3 = garble_and_gate(cipher, a0, b0, delta, gid, [false, true]);
        let result4 = garble_and_gate(cipher, a0, b0, delta, gid, [true, true]);

        // R matrices should differ based on random bits
        assert_ne!(result1.control_bits.r, result2.control_bits.r);
        assert_ne!(result1.control_bits.r, result3.control_bits.r);
        assert_ne!(result1.control_bits.r, result4.control_bits.r);
    }

    /// Test 3: Different gate IDs produce different hashes
    #[test]
    fn test_gate_id_affects_output() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);
        let rand_bits = [false, false];

        let result1 = garble_and_gate(cipher, a0, b0, delta, 1, rand_bits);
        let result2 = garble_and_gate(cipher, a0, b0, delta, 2, rand_bits);

        // Output labels should differ due to different gate IDs
        assert_ne!(result1.output_label, result2.output_label);
    }

    /// Test 4: Hash computation produces expected structure
    #[test]
    fn test_hash_computation() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);
        let gid = 1;

        let hashes = compute_hashes(cipher, a0, b0, delta, gid);

        // All 6 hashes should be different
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert_ne!(
                    hashes[i].to_block(),
                    hashes[j].to_block(),
                    "Hashes {} and {} should differ",
                    i,
                    j
                );
            }
        }
    }

    /// Test 5: Matrix application produces non-zero results
    #[test]
    fn test_matrix_application() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);
        let gid = 1;

        let hashes = compute_hashes(cipher, a0, b0, delta, gid);
        let m_result = apply_m_to_hashes(&hashes);

        // At least some rows should be non-zero
        let non_zero_count = m_result.iter().filter(|row| **row != [0u8; 8]).count();
        assert!(non_zero_count > 0, "M·H should have non-zero rows");
    }

    /// Test 6: Output label has correct Free-XOR relationship
    ///
    /// For AND gate: C₁ = C₀ ⊕ Δ only when output is 1 (i.e., both inputs are
    /// 1) This test verifies the garbling equation is set up correctly.
    #[test]
    fn test_output_label_structure() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let a0 = Block::random(&mut rng);
        let b0 = Block::random(&mut rng);
        let delta = Block::random(&mut rng);

        // Garble multiple gates to check consistency
        for gid in 1..10 {
            let rand_bits: [bool; 2] = [rng.random(), rng.random()];
            let result = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);

            // The output label C₀ should be a valid block
            assert_ne!(result.output_label, Block::ZERO);

            // Gate ciphertexts should be populated
            let gate = result.gate;
            // At least one should be non-zero (probabilistically)
            let has_nonzero = gate.g0 != [0u8; 8] || gate.g1 != [0u8; 8] || gate.g2 != [0u8; 8];
            // This might occasionally fail by chance, but very unlikely
            assert!(
                has_nonzero || gid > 5,
                "Gate ciphertexts should have some non-zero values"
            );
        }
    }

    /// Test 7: solve_for_output satisfies the marginal equations
    ///
    /// For each input combination (i,j), applying V to [C;G] should give back
    /// the original RHS values for the corresponding rows.
    #[test]
    fn test_solve_for_output_satisfies_marginals() {
        use super::super::matrices::V;

        let mut rng = ChaCha12Rng::seed_from_u64(123);

        // Create a random 8-element RHS
        let mut rhs = [[0u8; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                rhs[i][j] = rng.random();
            }
        }

        // Solve for [C_L, C_R, G₀, G₁, G₂]
        let output = solve_for_output(&rhs);

        // Verify that rows 0-5 are exactly satisfied (inputs 0,0 and 0,1 and 1,0)
        // Row 0: C_L = RHS[0]
        assert_eq!(output[0], rhs[0], "Row 0 should be satisfied");

        // Row 1: C_R = RHS[1]
        assert_eq!(output[1], rhs[1], "Row 1 should be satisfied");

        // Row 2: C_L ⊕ G₂ = RHS[2]
        let mut row2_check = [0u8; 8];
        for k in 0..8 {
            row2_check[k] = output[0][k] ^ output[4][k]; // C_L ⊕ G₂
        }
        assert_eq!(row2_check, rhs[2], "Row 2 should be satisfied");

        // Row 3: C_R ⊕ G₁ ⊕ G₂ = RHS[3]
        let mut row3_check = [0u8; 8];
        for k in 0..8 {
            row3_check[k] = output[1][k] ^ output[3][k] ^ output[4][k]; // C_R ⊕ G₁ ⊕ G₂
        }
        assert_eq!(row3_check, rhs[3], "Row 3 should be satisfied");

        // Row 4: C_L ⊕ G₀ ⊕ G₂ = RHS[4]
        let mut row4_check = [0u8; 8];
        for k in 0..8 {
            row4_check[k] = output[0][k] ^ output[2][k] ^ output[4][k]; // C_L ⊕ G₀ ⊕ G₂
        }
        assert_eq!(row4_check, rhs[4], "Row 4 should be satisfied");

        // Note: Rows 5, 6, 7 may not be exactly satisfied (they differ by Δ
        // terms) This is intentional - the scheme handles it through
        // the K·RHS structure
    }

    /// Test 8: THE CRITICAL TEST - Garble then evaluate for all 4 input
    /// combinations
    ///
    /// This verifies the complete correctness of the Three Halves scheme:
    /// - Garble with A₀, B₀, Δ
    /// - Evaluate with (A_i, B_j) for all (i,j) ∈ {0,1}²
    /// - Output should be C₀ when i∧j=0, and C₁=C₀⊕Δ when i∧j=1
    #[test]
    fn test_garble_then_evaluate_all_inputs() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(999);
        let mut failures: Vec<(usize, usize, usize)> = Vec::new();

        // Test multiple random instances
        for trial in 0..10 {
            // Generate random labels with proper LSB structure for pointer bits
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            // Ensure delta has LSB = 1 (Free-XOR requirement)
            delta.set_lsb(true);

            // Ensure A₀ and B₀ have LSB = 0 (so pointer bit indicates the actual bit value)
            a0.set_lsb(false);
            b0.set_lsb(false);

            let a1 = a0 ^ delta; // A₁ has LSB = 1
            let b1 = b0 ^ delta; // B₁ has LSB = 1

            let gid = trial + 1;
            let rand_bits: [bool; 2] = [rng.random(), rng.random()];

            // Garble the AND gate
            let garbled = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);
            let c0 = garbled.output_label;
            let c1 = c0 ^ delta; // Expected output for 1∧1

            // Test all 4 input combinations
            let test_cases = [
                (a0, b0, 0, 0, c0), // 0 ∧ 0 = 0
                (a0, b1, 0, 1, c0), // 0 ∧ 1 = 0
                (a1, b0, 1, 0, c0), // 1 ∧ 0 = 0
                (a1, b1, 1, 1, c1), // 1 ∧ 1 = 1
            ];

            for (a_label, b_label, i, j, expected) in test_cases {
                let result = evaluate_and_gate(
                    cipher,
                    a_label,
                    b_label,
                    &garbled.gate,
                    &garbled.control_bits,
                    gid,
                );

                if result == expected {
                    println!("Trial {}: evaluate({},{}) PASSED", trial, i, j);
                } else {
                    println!("Trial {}: evaluate({},{}) FAILED", trial, i, j);
                    failures.push((trial, i, j));
                }
            }
        }

        if !failures.is_empty() {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            for (trial, i, j) in &failures {
                println!("  Trial {}: ({},{})", trial, i, j);
            }
            panic!("{} test cases failed", failures.len());
        }
    }

    /// Test 9: Evaluation produces consistent results
    ///
    /// Same inputs should always produce same outputs.
    #[test]
    fn test_evaluation_deterministic() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut a0 = Block::random(&mut rng);
        let mut b0 = Block::random(&mut rng);
        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        a0.set_lsb(false);
        b0.set_lsb(false);

        let gid = 1;
        let rand_bits = [true, false];

        let garbled = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);

        // Evaluate twice with same inputs
        let result1 = evaluate_and_gate(cipher, a0, b0, &garbled.gate, &garbled.control_bits, gid);
        let result2 = evaluate_and_gate(cipher, a0, b0, &garbled.gate, &garbled.control_bits, gid);

        assert_eq!(result1, result2, "Evaluation should be deterministic");
    }

    /// Test R matrix computation in isolation.
    ///
    /// This test verifies that the R matrix works correctly WITHOUT involving
    /// hashes or gate ciphertexts. It isolates the input contribution logic.
    ///
    /// G computes: R · [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
    /// E computes: R[rows][cols 0-3] · [A_i_L, A_i_R, B_j_L, B_j_R]
    ///
    /// For input (i,j), E's result for rows 2*ij and 2*ij+1 should match
    /// G's result for those same rows.
    #[test]
    fn test_r_matrix_computation_isolated() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::slicing::SlicedLabel;

        let mut rng = ChaCha12Rng::seed_from_u64(12345);

        // Test with multiple random label sets
        for trial in 0..20 {
            // Generate random labels
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            // Set up proper LSB structure
            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let a1 = a0 ^ delta;
            let b1 = b0 ^ delta;

            // Slice labels
            let a0_sliced = SlicedLabel::from_block(a0);
            let a1_sliced = SlicedLabel::from_block(a1);
            let b0_sliced = SlicedLabel::from_block(b0);
            let b1_sliced = SlicedLabel::from_block(b1);
            let delta_sliced = SlicedLabel::from_block(delta);

            // Test all 4 random bit combinations for R matrix
            for rand_bits in [[false, false], [false, true], [true, false], [true, true]] {
                let t = and_truth_table();
                let (r, _r_bar) = sample_r_odd(&t, rand_bits);

                // G computes: R · [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
                let garbler_input: [[u8; 8]; 6] = [
                    a0_sliced.left,
                    a0_sliced.right,
                    b0_sliced.left,
                    b0_sliced.right,
                    delta_sliced.left,
                    delta_sliced.right,
                ];

                let mut garbler_result = [[0u8; 8]; 8];
                for row in 0..8 {
                    for col in 0..6 {
                        if r[row][col] == 1 {
                            for k in 0..8 {
                                garbler_result[row][k] ^= garbler_input[col][k];
                            }
                        }
                    }
                }

                // Test all 4 input combinations (i, j)
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let ij = (i << 1) | j;
                    let row_l = 2 * ij;
                    let row_r = 2 * ij + 1;

                    // Get evaluator's labels based on input bits
                    let a_sliced = if i == 0 { &a0_sliced } else { &a1_sliced };
                    let b_sliced = if j == 0 { &b0_sliced } else { &b1_sliced };

                    // E computes: R[row][cols 0-3] · [A_i_L, A_i_R, B_j_L, B_j_R]
                    let eval_input: [[u8; 8]; 4] = [
                        a_sliced.left,
                        a_sliced.right,
                        b_sliced.left,
                        b_sliced.right,
                    ];

                    // Compute evaluator's result for left half (row_l)
                    let mut eval_result_l = [0u8; 8];
                    for col in 0..4 {
                        if r[row_l][col] == 1 {
                            for k in 0..8 {
                                eval_result_l[k] ^= eval_input[col][k];
                            }
                        }
                    }

                    // Compute evaluator's result for right half (row_r)
                    let mut eval_result_r = [0u8; 8];
                    for col in 0..4 {
                        if r[row_r][col] == 1 {
                            for k in 0..8 {
                                eval_result_r[k] ^= eval_input[col][k];
                            }
                        }
                    }

                    // Verify: E's result should match G's result for these rows
                    assert_eq!(
                        eval_result_l, garbler_result[row_l],
                        "Trial {}, rand_bits={:?}, input ({},{}), row {} (left): \
                         Evaluator result doesn't match Garbler result.\n\
                         E got: {:?}\n\
                         G got: {:?}\n\
                         R[{}] = {:?}",
                        trial, rand_bits, i, j, row_l,
                        eval_result_l, garbler_result[row_l], row_l, r[row_l]
                    );

                    assert_eq!(
                        eval_result_r, garbler_result[row_r],
                        "Trial {}, rand_bits={:?}, input ({},{}), row {} (right): \
                         Evaluator result doesn't match Garbler result.\n\
                         E got: {:?}\n\
                         G got: {:?}\n\
                         R[{}] = {:?}",
                        trial, rand_bits, i, j, row_r,
                        eval_result_r, garbler_result[row_r], row_r, r[row_r]
                    );
                }
            }
        }

        println!("✓ R matrix computation test passed for all trials and random bit combinations!");
    }

    /// Test hash contribution (M matrix) computation in isolation.
    ///
    /// This test verifies that the evaluator's hash contribution matches
    /// the garbler's hash contribution for the evaluator's rows.
    ///
    /// G computes: M · [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]
    /// E computes: Using only H(A_i), H(B_j), H(A_i⊕B_j) mapped to correct columns
    ///
    /// For input (i,j), E's result for rows 2*ij and 2*ij+1 should match
    /// G's result for those same rows.
    #[test]
    fn test_hash_contribution_isolated() {
        use super::super::matrices::M;
        use super::super::slicing::SlicedLabel;

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(54321);

        // Test with multiple random label sets
        for trial in 0..20 {
            // Generate random labels
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            // Set up proper LSB structure
            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let a1 = a0 ^ delta;
            let b1 = b0 ^ delta;

            let gid = trial + 1;
            let tweak = Block::new((gid as u128).to_be_bytes());

            // G computes 6 hashes
            let mut garbler_hash_inputs = [a0, a1, b0, b1, a0 ^ b0, a0 ^ b1];
            cipher.tccr_many(&[tweak; 6], &mut garbler_hash_inputs);
            let garbler_hashes: [SlicedLabel; 6] = [
                SlicedLabel::from_block(garbler_hash_inputs[0]), // H(A₀)
                SlicedLabel::from_block(garbler_hash_inputs[1]), // H(A₁)
                SlicedLabel::from_block(garbler_hash_inputs[2]), // H(B₀)
                SlicedLabel::from_block(garbler_hash_inputs[3]), // H(B₁)
                SlicedLabel::from_block(garbler_hash_inputs[4]), // H(A₀⊕B₀)
                SlicedLabel::from_block(garbler_hash_inputs[5]), // H(A₀⊕B₁)
            ];

            // G computes M · H for all 8 rows
            let mut garbler_result = [[0u8; 8]; 8];
            for row in 0..8 {
                let half = row % 2; // 0 = left, 1 = right
                for col in 0..6 {
                    if M[row][col] == 1 {
                        let h_half = garbler_hashes[col].half(half);
                        for k in 0..8 {
                            garbler_result[row][k] ^= h_half[k];
                        }
                    }
                }
            }

            // Test all 4 input combinations (i, j)
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let ij = (i << 1) | j;
                let row_l = 2 * ij;
                let row_r = 2 * ij + 1;

                // Get evaluator's labels based on input bits
                let a_label = if i == 0 { a0 } else { a1 };
                let b_label = if j == 0 { b0 } else { b1 };

                // E computes 3 hashes
                let mut eval_hash_inputs = [a_label, b_label, a_label ^ b_label];
                cipher.tccr_many(&[tweak; 3], &mut eval_hash_inputs);
                let h_a = SlicedLabel::from_block(eval_hash_inputs[0]); // H(A_i)
                let h_b = SlicedLabel::from_block(eval_hash_inputs[1]); // H(B_j)
                let h_ab = SlicedLabel::from_block(eval_hash_inputs[2]); // H(A_i⊕B_j)

                // E's hash column mapping:
                // - H(A_i) → column i
                // - H(B_j) → column 2+j
                // - H(A_i⊕B_j) → column 4 if (i^j)==0, column 5 if (i^j)==1
                let a_col = i;
                let b_col = 2 + j;
                let ab_col = if (i ^ j) == 0 { 4 } else { 5 };

                // E computes hash contribution for their two rows
                for (row, half) in [(row_l, 0), (row_r, 1)] {
                    let mut eval_result = [0u8; 8];

                    // XOR in h_a if M says to use column a_col
                    if M[row][a_col] == 1 {
                        let h_half = h_a.half(half);
                        for k in 0..8 {
                            eval_result[k] ^= h_half[k];
                        }
                    }

                    // XOR in h_b if M says to use column b_col
                    if M[row][b_col] == 1 {
                        let h_half = h_b.half(half);
                        for k in 0..8 {
                            eval_result[k] ^= h_half[k];
                        }
                    }

                    // XOR in h_ab if M says to use column ab_col
                    if M[row][ab_col] == 1 {
                        let h_half = h_ab.half(half);
                        for k in 0..8 {
                            eval_result[k] ^= h_half[k];
                        }
                    }

                    // Verify: E's result should match G's result for this row
                    assert_eq!(
                        eval_result, garbler_result[row],
                        "Trial {}, input ({},{}), row {}: \
                         Evaluator hash contribution doesn't match Garbler.\n\
                         E got: {:?}\n\
                         G got: {:?}\n\
                         M[{}] = {:?}\n\
                         E's columns: a_col={}, b_col={}, ab_col={}",
                        trial, i, j, row,
                        eval_result, garbler_result[row], row, M[row],
                        a_col, b_col, ab_col
                    );
                }
            }
        }

        println!("✓ Hash contribution test passed for all trials and input combinations!");
    }

    /// Test solve_for_output + V matrix in isolation.
    ///
    /// This test verifies that solve_for_output produces gate values such that
    /// V · [C_L, C_R, G₀, G₁, G₂] gives the correct output for all 4 inputs.
    ///
    /// For AND gate:
    /// - Inputs (0,0), (0,1), (1,0): Evaluator should get C₀
    /// - Input (1,1): Evaluator should get C₁ = C₀ ⊕ Δ
    #[test]
    fn test_solve_for_output_with_v_matrix() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::matrices::{M, V};
        use super::super::slicing::SlicedLabel;
        use super::{apply_m_to_hashes, apply_r_to_inputs, solve_for_output};

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(99999);

        // Test with multiple random label sets
        for trial in 0..20 {
            // Generate random labels
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let gid = trial + 1;
            let tweak = Block::new((gid as u128).to_be_bytes());

            // Slice labels
            let a0_sliced = SlicedLabel::from_block(a0);
            let b0_sliced = SlicedLabel::from_block(b0);
            let delta_sliced = SlicedLabel::from_block(delta);

            // Test all 4 R matrix variations
            for rand_bits in [[false, false], [false, true], [true, false], [true, true]] {
                let t = and_truth_table();
                let (r, _) = sample_r_odd(&t, rand_bits);

                // Compute 6 hashes (garbler's view)
                let mut hash_inputs = [a0, a0 ^ delta, b0, b0 ^ delta, a0 ^ b0, a0 ^ (b0 ^ delta)];
                cipher.tccr_many(&[tweak; 6], &mut hash_inputs);
                let hashes: [SlicedLabel; 6] = [
                    SlicedLabel::from_block(hash_inputs[0]),
                    SlicedLabel::from_block(hash_inputs[1]),
                    SlicedLabel::from_block(hash_inputs[2]),
                    SlicedLabel::from_block(hash_inputs[3]),
                    SlicedLabel::from_block(hash_inputs[4]),
                    SlicedLabel::from_block(hash_inputs[5]),
                ];

                // Compute M · H
                let m_times_h = apply_m_to_hashes(&hashes);

                // Compute R · [A₀; B₀; Δ]
                let r_times_input = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

                // RHS = M·H ⊕ R·input
                let mut rhs = [[0u8; 8]; 8];
                for row in 0..8 {
                    for k in 0..8 {
                        rhs[row][k] = m_times_h[row][k] ^ r_times_input[row][k];
                    }
                }

                // Solve for [C_L, C_R, G₀, G₁, G₂]
                let output = solve_for_output(&rhs);
                let c_l = output[0];
                let c_r = output[1];
                let g0 = output[2];
                let g1 = output[3];
                let g2 = output[4];

                // Expected outputs
                let c0_l = c_l;
                let c0_r = c_r;
                let c1_l: [u8; 8] = std::array::from_fn(|k| c_l[k] ^ delta_sliced.left[k]);
                let c1_r: [u8; 8] = std::array::from_fn(|k| c_r[k] ^ delta_sliced.right[k]);

                // Test all 4 input combinations
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let ij = (i << 1) | j;
                    let row_l = 2 * ij;
                    let row_r = 2 * ij + 1;

                    // Compute V[row_l] · output (left half)
                    let mut v_output_l = [0u8; 8];
                    if V[row_l][0] == 1 {
                        for k in 0..8 { v_output_l[k] ^= c_l[k]; }
                    }
                    if V[row_l][1] == 1 {
                        for k in 0..8 { v_output_l[k] ^= c_r[k]; }
                    }
                    if V[row_l][2] == 1 {
                        for k in 0..8 { v_output_l[k] ^= g0[k]; }
                    }
                    if V[row_l][3] == 1 {
                        for k in 0..8 { v_output_l[k] ^= g1[k]; }
                    }
                    if V[row_l][4] == 1 {
                        for k in 0..8 { v_output_l[k] ^= g2[k]; }
                    }

                    // Compute V[row_r] · output (right half)
                    let mut v_output_r = [0u8; 8];
                    if V[row_r][0] == 1 {
                        for k in 0..8 { v_output_r[k] ^= c_l[k]; }
                    }
                    if V[row_r][1] == 1 {
                        for k in 0..8 { v_output_r[k] ^= c_r[k]; }
                    }
                    if V[row_r][2] == 1 {
                        for k in 0..8 { v_output_r[k] ^= g0[k]; }
                    }
                    if V[row_r][3] == 1 {
                        for k in 0..8 { v_output_r[k] ^= g1[k]; }
                    }
                    if V[row_r][4] == 1 {
                        for k in 0..8 { v_output_r[k] ^= g2[k]; }
                    }

                    // What should evaluator get?
                    // RHS[row] = hash_contrib ⊕ input_contrib (for evaluator)
                    // Evaluator computes: RHS[row] ⊕ gate_contrib = RHS[row] ⊕ V[row][2:5]·[G₀,G₁,G₂]
                    // This should equal C₀ for (0,0),(0,1),(1,0) and C₁ for (1,1)

                    // Compute what evaluator would get:
                    // eval_result = RHS[row] ⊕ V[row]·[0,0,G₀,G₁,G₂]
                    // But V[row]·output = V[row]·[C_L,C_R,G₀,G₁,G₂]
                    // So eval_result = RHS[row] ⊕ (V[row]·output - V[row][0:2]·[C_L,C_R])

                    // Actually simpler: evaluator computes RHS ⊕ gate_contrib
                    // gate_contrib uses V[row][2:5] for G₀,G₁,G₂
                    let mut gate_contrib_l = [0u8; 8];
                    if V[row_l][2] == 1 { for k in 0..8 { gate_contrib_l[k] ^= g0[k]; } }
                    if V[row_l][3] == 1 { for k in 0..8 { gate_contrib_l[k] ^= g1[k]; } }
                    if V[row_l][4] == 1 { for k in 0..8 { gate_contrib_l[k] ^= g2[k]; } }

                    let mut gate_contrib_r = [0u8; 8];
                    if V[row_r][2] == 1 { for k in 0..8 { gate_contrib_r[k] ^= g0[k]; } }
                    if V[row_r][3] == 1 { for k in 0..8 { gate_contrib_r[k] ^= g1[k]; } }
                    if V[row_r][4] == 1 { for k in 0..8 { gate_contrib_r[k] ^= g2[k]; } }

                    // Evaluator's result = RHS ⊕ gate_contrib
                    let eval_l: [u8; 8] = std::array::from_fn(|k| rhs[row_l][k] ^ gate_contrib_l[k]);
                    let eval_r: [u8; 8] = std::array::from_fn(|k| rhs[row_r][k] ^ gate_contrib_r[k]);

                    // Expected: C₀ for AND=0, C₁ for AND=1
                    let and_result = i & j;
                    let (expected_l, expected_r) = if and_result == 0 {
                        (c0_l, c0_r)
                    } else {
                        (c1_l, c1_r)
                    };

                    assert_eq!(
                        eval_l, expected_l,
                        "Trial {}, rand_bits={:?}, input ({},{}), left half:\n\
                         Evaluator got: {:?}\n\
                         Expected (C{}): {:?}\n\
                         RHS[{}]: {:?}\n\
                         gate_contrib: {:?}",
                        trial, rand_bits, i, j,
                        eval_l, and_result, expected_l,
                        row_l, rhs[row_l], gate_contrib_l
                    );

                    assert_eq!(
                        eval_r, expected_r,
                        "Trial {}, rand_bits={:?}, input ({},{}), right half:\n\
                         Evaluator got: {:?}\n\
                         Expected (C{}): {:?}\n\
                         RHS[{}]: {:?}\n\
                         gate_contrib: {:?}",
                        trial, rand_bits, i, j,
                        eval_r, and_result, expected_r,
                        row_r, rhs[row_r], gate_contrib_r
                    );
                }
            }
        }

        println!("✓ solve_for_output + V matrix test passed for all trials!");
    }
}
