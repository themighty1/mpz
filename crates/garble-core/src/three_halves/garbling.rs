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

/// Compressed control bits for evaluator.
///
/// For ODD mode (AND gate), we need 2 bits per marginal × 4 marginals,
/// but due to the constraint structure, this compresses to 5 bits total.
///
/// Paper Section 5.3 (Page 15):
/// > "The garbler can compress the 8 bits of r̄ into 5 bits"
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlBits {
    /// The 4 compressed marginal views (2 bits each)
    /// r_bar[ij] for ij ∈ {00, 01, 10, 11}
    pub r_bar: [[u8; 2]; 4],
}

impl ControlBits {
    /// Create new control bits from compressed marginal views.
    pub fn new(r_bar: [[u8; 2]; 4]) -> Self {
        Self { r_bar }
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

/// Solve for [C_L, C_R, G₀, G₁, G₂] from RHS.
///
/// # Critical Insight (Paper Section 5)
///
/// The standard V⁻¹ assumes K·RHS = 0, but for AND gates K·RHS = [Δ_L, Δ_R,
/// Δ_L⊕Δ_R] ≠ 0. This means V⁻¹ gives incorrect G₂.
///
/// The correct formulas that ensure ALL marginal equations are satisfied:
/// - C_L = RHS[0]
/// - C_R = RHS[1]
/// - G₂ = RHS[0] ⊕ RHS[2]  (ensures row 2: C_L ⊕ G₂ = RHS[2])
/// - G₀ = RHS[2] ⊕ RHS[4]  (ensures row 4: C_L ⊕ G₀ ⊕ G₂ = RHS[4])
/// - G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3] (ensures row 3: C_R ⊕ G₁ ⊕ G₂ =
///   RHS[3])
///
/// With these formulas:
/// - Rows 0,1 (input 0,0): Satisfied by construction
/// - Rows 2,3 (input 0,1): Satisfied by construction
/// - Rows 4,5 (input 1,0): Row 4 by construction, Row 5 follows from K
///   constraints
/// - Rows 6,7 (input 1,1): These give C ⊕ Δ (the correct output for AND = 1)
fn solve_for_output(rhs: &[[u8; 8]; 8]) -> [[u8; 8]; 5] {
    let mut result = [[0u8; 8]; 5];

    // C_L = RHS[0]
    result[0] = rhs[0];

    // C_R = RHS[1]
    result[1] = rhs[1];

    // G₀ = RHS[2] ⊕ RHS[4]
    for k in 0..8 {
        result[2][k] = rhs[2][k] ^ rhs[4][k];
    }

    // G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
    for k in 0..8 {
        result[3][k] = rhs[0][k] ^ rhs[1][k] ^ rhs[2][k] ^ rhs[3][k];
    }

    // G₂ = RHS[0] ⊕ RHS[2]
    for k in 0..8 {
        result[4][k] = rhs[0][k] ^ rhs[2][k];
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
    let (r, r_bar) = sample_r_odd(&t, rand_bits);

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

    let control_bits = ControlBits::new(r_bar);

    GarbledGate {
        output_label: c0,
        gate,
        control_bits,
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Extract the effective R marginal for an evaluator with input (i, j).
///
/// The control bits encode R$ marginals (without R_P). For ODD mode (AND gate),
/// we need to add the R_P marginal since parity is public.
///
/// # Key Insight from Paper (Equation 6, Page 12)
///
/// The R matrix is designed such that the Δ columns (4-5) satisfy:
/// R[row][4] = R[row][0]*i + R[row][2]*j
/// R[row][5] = R[row][1]*i + R[row][3]*j
///
/// This means extracting columns 0-3 directly gives the correct marginal.
fn extract_evaluator_marginal(i: usize, j: usize, r_bar_ij: &[u8; 2]) -> [[u8; 4]; 2] {
    use super::control::expand_marginal;

    // r_bar_ij already encodes the full marginal (including parity),
    // so just expand and return.
    expand_marginal(r_bar_ij)

    // Claudes faulty code below
    // use super::control::{R_P, expand_marginal, extract_marginal};

    // // Step 1: Expand r_bar_ij to get the R$ marginal (2×4)
    // let r_dollar_marginal = expand_marginal(r_bar_ij);

    // // Step 2: Extract R_P's marginal (columns 0-3 only)
    // let r_p_marginal = extract_marginal(&R_P, i, j);

    // // Step 3: Combined marginal = R$ marginal ⊕ R_P marginal
    // let mut marginal = [[0u8; 4]; 2];
    // for row in 0..2 {
    //     for col in 0..4 {
    //         marginal[row][col] = r_dollar_marginal[row][col] ^
    // r_p_marginal[row][col];     }
    // }

    // marginal
}

/// Evaluate a Three Halves AND gate.
///
/// # Arguments
/// * `cipher` - Fixed-key AES cipher for TCCR hash
/// * `a` - Input wire A label (for bit i)
/// * `b` - Input wire B label (for bit j)
/// * `gate` - Gate ciphertexts from garbling
/// * `control_bits` - Control bits from garbling
/// * `gid` - Gate ID
///
/// # Returns
/// The output wire label C_{i∧j}
///
/// # Paper Reference
/// This implements the evaluation algorithm from Section 5.2 (Page 13-14).
///
/// The evaluator:
/// 1. Determines which input combination (i,j) they have via pointer bits
/// 2. Extracts the marginal view R̄_{ij} from control bits
/// 3. Computes the appropriate linear combination of hashes and inputs
/// 4. Recovers the output label using gate ciphertexts
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
    let ij = (i << 1) | j; // Index into control_bits.r_bar

    // 2. Get the marginal view for this input combination
    let r_bar_ij = control_bits.r_bar[ij];

    // 3. Compute the two hashes needed for this input combination From the M matrix
    //    structure:
    //    - (0,0): H(A_i), H(A_i⊕B_j) = H(A₀), H(A₀⊕B₀)
    //    - (0,1): H(A_i), H(A_i⊕B_j) = H(A₀), H(A₀⊕B₁)
    //    - (1,0): H(A_i), H(A_i⊕B_j) = H(A₁), H(A₁⊕B₀)
    //    - (1,1): H(A_i), H(A_i⊕B_j) = H(A₁), H(A₁⊕B₁)
    //    But evaluator only has A_i, B_j, so computes H(A_i), H(B_j), H(A_i⊕B_j)
    let tweak = Block::new((gid as u128).to_be_bytes());
    let mut hash_inputs = [a, b, a ^ b];
    cipher.tccr_many(&[tweak; 3], &mut hash_inputs);

    let h_a = SlicedLabel::from_block(hash_inputs[0]); // H(A_i)
    let h_b = SlicedLabel::from_block(hash_inputs[1]); // H(B_j)
    let h_ab = SlicedLabel::from_block(hash_inputs[2]); // H(A_i ⊕ B_j)

    // 4. Slice the input labels
    let a_sliced = SlicedLabel::from_block(a);
    let b_sliced = SlicedLabel::from_block(b);

    // 5. Expand the marginal control bits and compute the effective R marginal for
    //    the evaluator's input combination (i, j).
    //
    //    The r_bar encodes R$ (without R_P) as a 2-bit value per marginal.
    //    For ODD mode (AND gate), we also add R_P since parity is public.
    //
    //    CRITICAL: The marginal extraction must account for the Δ contribution!
    //    When the evaluator has input (i,j), their labels are:
    //      A_i = A₀ ⊕ i·Δ
    //      B_j = B₀ ⊕ j·Δ
    //
    //    The garbler computed R · [A₀; B₀; Δ]. For the evaluator to get the
    //    same result using [A_i; B_j], the marginal coefficients must be adjusted:
    //      - A coefficients: R[0:2] ⊕ i·R[4:6]
    //      - B coefficients: R[2:4] ⊕ j·R[4:6]
    //
    //    (Paper Equation 6, Page 12)
    let r_bar_expanded = extract_evaluator_marginal(i, j, &r_bar_ij);

    // 6. Compute the evaluation for left and right halves
    //
    //    From paper Section 5.2, the evaluator computes:
    //    For the left half (row 2*ij):
    //      result_L = hash_contribution_L ⊕ input_contribution_L ⊕
    // gate_contribution_L    For the right half (row 2*ij+1):
    //      result_R = hash_contribution_R ⊕ input_contribution_R ⊕
    // gate_contribution_R
    //
    //    The V matrix tells us how gate ciphertexts contribute.
    //    The M matrix tells us which hashes contribute (but evaluator only has 3).
    //    The R marginal tells us the input contribution.

    // Get the two rows for this input combination
    let row_l = 2 * ij; // Left half row
    let row_r = 2 * ij + 1; // Right half row

    // Hash contribution (from M matrix)
    // For evaluator: they compute using their available hashes
    let hash_contrib_l = compute_hash_contribution_eval(row_l, &h_a, &h_b, &h_ab);
    let hash_contrib_r = compute_hash_contribution_eval(row_r, &h_a, &h_b, &h_ab);

    // Input contribution (from R marginal)
    // The expanded marginal r_bar_expanded is a 2×4 matrix
    // Columns are: A_L, A_R, B_L, B_R (relative to evaluator's labels)
    let input_contrib_l = compute_input_contribution(&r_bar_expanded, 0, &a_sliced, &b_sliced);
    let input_contrib_r = compute_input_contribution(&r_bar_expanded, 1, &a_sliced, &b_sliced);

    // Gate contribution (from V matrix)
    // V[row][2:5] gives coefficients for G₀, G₁, G₂
    let gate_contrib_l = compute_gate_contribution(row_l, gate);
    let gate_contrib_r = compute_gate_contribution(row_r, gate);

    // Combine all contributions
    let mut c_l = hash_contrib_l;
    xor_assign_8(&mut c_l, &input_contrib_l);
    xor_assign_8(&mut c_l, &gate_contrib_l);

    let mut c_r = hash_contrib_r;
    xor_assign_8(&mut c_r, &input_contrib_r);
    xor_assign_8(&mut c_r, &gate_contrib_r);

    // 7. Reconstruct output label
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

    /// Test 2: Different random bits produce different control bits
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

        // Control bits should differ
        assert_ne!(result1.control_bits.r_bar, result2.control_bits.r_bar);
        assert_ne!(result1.control_bits.r_bar, result3.control_bits.r_bar);
        assert_ne!(result1.control_bits.r_bar, result4.control_bits.r_bar);
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

    /// Debug test to trace through evaluation
    #[test]
    fn test_debug_evaluation() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(999);

        // Generate random labels with proper LSB structure
        let mut a0 = Block::random(&mut rng);
        let mut b0 = Block::random(&mut rng);
        let mut delta = Block::random(&mut rng);

        delta.set_lsb(true);
        a0.set_lsb(false);
        b0.set_lsb(false);

        let a1 = a0 ^ delta;
        let b1 = b0 ^ delta;

        let gid = 1;
        let rand_bits = [false, false]; // Deterministic for debugging

        // Garble the AND gate
        let garbled = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);
        let c0 = garbled.output_label;

        println!("\n=== GARBLING ===");
        println!("A0 = {:?}", a0);
        println!("B0 = {:?}", b0);
        println!("Delta = {:?}", delta);
        println!("C0 (output) = {:?}", c0);
        println!(
            "Gate: G0={:?}, G1={:?}, G2={:?}",
            garbled.gate.g0, garbled.gate.g1, garbled.gate.g2
        );
        println!("Control bits r_bar: {:?}", garbled.control_bits.r_bar);

        // Test (0,0) first - this should work
        println!("\n=== EVALUATE (0,0) ===");
        let result_00 =
            evaluate_and_gate(cipher, a0, b0, &garbled.gate, &garbled.control_bits, gid);
        println!("Expected: {:?}", c0);
        println!("Got:      {:?}", result_00);
        println!("Match: {}", result_00 == c0);

        // Test (0,1) - this fails
        println!("\n=== EVALUATE (0,1) ===");
        let result_01 =
            evaluate_and_gate(cipher, a0, b1, &garbled.gate, &garbled.control_bits, gid);
        println!("Expected: {:?}", c0);
        println!("Got:      {:?}", result_01);
        println!("Match: {}", result_01 == c0);

        // Detailed trace for (0,1)
        println!("\n=== DETAILED TRACE (0,1) ===");
        let i = 0usize;
        let j = 1usize;
        let ij = (i << 1) | j;
        println!("i={}, j={}, ij={}", i, j, ij);

        let r_bar_ij = garbled.control_bits.r_bar[ij];
        println!("r_bar[{}] = {:?}", ij, r_bar_ij);

        // Compute hashes
        let tweak = Block::new((gid as u128).to_be_bytes());
        let mut hash_inputs = [a0, b1, a0 ^ b1];
        cipher.tccr_many(&[tweak; 3], &mut hash_inputs);
        let h_a = SlicedLabel::from_block(hash_inputs[0]);
        let h_b = SlicedLabel::from_block(hash_inputs[1]);
        let h_ab = SlicedLabel::from_block(hash_inputs[2]);
        println!("H(A0) = {:?}", h_a);
        println!("H(B1) = {:?}", h_b);
        println!("H(A0^B1) = {:?}", h_ab);

        // Slice input labels
        let a_sliced = SlicedLabel::from_block(a0);
        let b_sliced = SlicedLabel::from_block(b1);
        println!("A0 sliced = {:?}", a_sliced);
        println!("B1 sliced = {:?}", b_sliced);

        // Extract marginal
        use crate::three_halves::control::{R_P, expand_marginal, extract_marginal};
        let r_dollar_marginal = expand_marginal(&r_bar_ij);
        let r_p_marginal = extract_marginal(&R_P, i, j);
        println!("R$ marginal = {:?}", r_dollar_marginal);
        println!("R_P marginal = {:?}", r_p_marginal);

        let mut total_marginal = [[0u8; 4]; 2];
        for row in 0..2 {
            for col in 0..4 {
                total_marginal[row][col] = r_dollar_marginal[row][col] ^ r_p_marginal[row][col];
            }
        }
        println!("Total marginal = {:?}", total_marginal);

        // Now manually compute what the garbler computed for rows 2,3
        println!("\n=== COMPARE WITH GARBLER ===");

        // Garbler's hash contribution for row 2
        let garbler_hashes = compute_hashes(cipher, a0, b0, delta, gid);
        println!("Garbler H(A0) = {:?}", garbler_hashes[0]);
        println!("Garbler H(A0^B1) = {:?}", garbler_hashes[5]);

        // Check if hashes match
        println!("\nHash comparison:");
        println!(
            "Evaluator H(A0) == Garbler H(A0): {}",
            h_a == garbler_hashes[0]
        );
        println!(
            "Evaluator H(A0^B1) == Garbler H(A0^B1): {}",
            h_ab == garbler_hashes[5]
        );

        // Manually compute hash contribution for row 2 (left) and row 3 (right)
        println!("\n=== HASH CONTRIBUTION ===");
        // Row 2: H(A0).L ⊕ H(A0^B1).L
        let mut hash_left = [0u8; 8];
        for k in 0..8 {
            hash_left[k] = h_a.left[k] ^ h_ab.left[k];
        }
        println!("Hash contrib left (row 2): {:?}", hash_left);

        // Row 3: H(B1).R ⊕ H(A0^B1).R
        let mut hash_right = [0u8; 8];
        for k in 0..8 {
            hash_right[k] = h_b.right[k] ^ h_ab.right[k];
        }
        println!("Hash contrib right (row 3): {:?}", hash_right);

        // Manually compute input contribution
        println!("\n=== INPUT CONTRIBUTION ===");
        // Row 2 (left): marginal[0] applied to [A0_L, A0_R, B1_L, B1_R]
        let mut input_left = [0u8; 8];
        if total_marginal[0][0] == 1 {
            for k in 0..8 {
                input_left[k] ^= a_sliced.left[k];
            }
        }
        if total_marginal[0][1] == 1 {
            for k in 0..8 {
                input_left[k] ^= a_sliced.right[k];
            }
        }
        if total_marginal[0][2] == 1 {
            for k in 0..8 {
                input_left[k] ^= b_sliced.left[k];
            }
        }
        if total_marginal[0][3] == 1 {
            for k in 0..8 {
                input_left[k] ^= b_sliced.right[k];
            }
        }
        println!("Input contrib left: {:?}", input_left);
        println!("  marginal[0] = {:?}", total_marginal[0]);
        println!("  A0.L={:?}, A0.R={:?}", a_sliced.left, a_sliced.right);
        println!("  B1.L={:?}, B1.R={:?}", b_sliced.left, b_sliced.right);

        // Row 3 (right): marginal[1] applied to [A0_L, A0_R, B1_L, B1_R]
        let mut input_right = [0u8; 8];
        if total_marginal[1][0] == 1 {
            for k in 0..8 {
                input_right[k] ^= a_sliced.left[k];
            }
        }
        if total_marginal[1][1] == 1 {
            for k in 0..8 {
                input_right[k] ^= a_sliced.right[k];
            }
        }
        if total_marginal[1][2] == 1 {
            for k in 0..8 {
                input_right[k] ^= b_sliced.left[k];
            }
        }
        if total_marginal[1][3] == 1 {
            for k in 0..8 {
                input_right[k] ^= b_sliced.right[k];
            }
        }
        println!("Input contrib right: {:?}", input_right);
        println!("  marginal[1] = {:?}", total_marginal[1]);

        // Manually compute gate contribution
        println!("\n=== GATE CONTRIBUTION ===");
        use crate::three_halves::matrices::V;
        // Row 2: V[2] = [1, 0, 0, 0, 1] -> G₂
        let mut gate_left = [0u8; 8];
        if V[2][2] == 1 {
            for k in 0..8 {
                gate_left[k] ^= garbled.gate.g0[k];
            }
        }
        if V[2][3] == 1 {
            for k in 0..8 {
                gate_left[k] ^= garbled.gate.g1[k];
            }
        }
        if V[2][4] == 1 {
            for k in 0..8 {
                gate_left[k] ^= garbled.gate.g2[k];
            }
        }
        println!("Gate contrib left (row 2): {:?}", gate_left);
        println!("  V[2] = {:?}", V[2]);

        // Row 3: V[3] = [0, 1, 0, 1, 1] -> G₁ ⊕ G₂
        let mut gate_right = [0u8; 8];
        if V[3][2] == 1 {
            for k in 0..8 {
                gate_right[k] ^= garbled.gate.g0[k];
            }
        }
        if V[3][3] == 1 {
            for k in 0..8 {
                gate_right[k] ^= garbled.gate.g1[k];
            }
        }
        if V[3][4] == 1 {
            for k in 0..8 {
                gate_right[k] ^= garbled.gate.g2[k];
            }
        }
        println!("Gate contrib right (row 3): {:?}", gate_right);
        println!("  V[3] = {:?}", V[3]);

        // Combine all contributions
        println!("\n=== COMBINED ===");
        let mut c_l_computed = [0u8; 8];
        let mut c_r_computed = [0u8; 8];
        for k in 0..8 {
            c_l_computed[k] = hash_left[k] ^ input_left[k] ^ gate_left[k];
            c_r_computed[k] = hash_right[k] ^ input_right[k] ^ gate_right[k];
        }
        let computed = SlicedLabel::new(c_l_computed, c_r_computed);
        println!("Computed C_L: {:?}", c_l_computed);
        println!("Computed C_R: {:?}", c_r_computed);
        println!("Computed Block: {:?}", computed.to_block());

        let expected = SlicedLabel::from_block(c0);
        println!("\nExpected C_L: {:?}", expected.left);
        println!("Expected C_R: {:?}", expected.right);

        // Also show what the garbler computed for RHS[2] and RHS[3]
        println!("\n=== GARBLER'S RHS for rows 2,3 ===");
        // We need to compute what the garbler got before applying V^-1
        // RHS = M*H ⊕ R*input

        // First, M*H for rows 2 and 3
        let mut m_h_2 = [0u8; 8];
        let mut m_h_3 = [0u8; 8];
        use crate::three_halves::matrices::M;
        // M[2] = [1, 0, 0, 0, 0, 1] -> H(A0).L ⊕ H(A0⊕B1).L for left half
        for col in 0..6 {
            if M[2][col] == 1 {
                let h_half = garbler_hashes[col].half(0); // left half for row 2
                for k in 0..8 {
                    m_h_2[k] ^= h_half[k];
                }
            }
        }
        // M[3] = [0, 0, 0, 1, 0, 1] -> H(B1).R ⊕ H(A0⊕B1).R for right half
        for col in 0..6 {
            if M[3][col] == 1 {
                let h_half = garbler_hashes[col].half(1); // right half for row 3
                for k in 0..8 {
                    m_h_3[k] ^= h_half[k];
                }
            }
        }
        println!("Garbler M*H[2] (left): {:?}", m_h_2);
        println!("Garbler M*H[3] (right): {:?}", m_h_3);

        // Now R*input for rows 2 and 3
        // R[2] = [1, 0, 1, 1, 1, 1] (computed above)
        // R[3] = [0, 1, 1, 1, 1, 1]
        let a0_sliced = SlicedLabel::from_block(a0);
        let b0_sliced = SlicedLabel::from_block(b0);
        let delta_sliced = SlicedLabel::from_block(delta);
        let inputs_garbler: [[u8; 8]; 6] = [
            a0_sliced.left,
            a0_sliced.right,
            b0_sliced.left,
            b0_sliced.right,
            delta_sliced.left,
            delta_sliced.right,
        ];

        // Full R for rows 2,3
        use crate::three_halves::control::{R_A, R_B, R_P as R_P_MAT};
        let mut r_row_2 = [0u8; 6];
        let mut r_row_3 = [0u8; 6];
        for col in 0..6 {
            r_row_2[col] = R_A[2][col] ^ R_B[2][col] ^ R_P_MAT[2][col];
            r_row_3[col] = R_A[3][col] ^ R_B[3][col] ^ R_P_MAT[3][col];
        }
        println!("R[2] = {:?}", r_row_2);
        println!("R[3] = {:?}", r_row_3);

        let mut r_input_2 = [0u8; 8];
        let mut r_input_3 = [0u8; 8];
        for col in 0..6 {
            if r_row_2[col] == 1 {
                for k in 0..8 {
                    r_input_2[k] ^= inputs_garbler[col][k];
                }
            }
            if r_row_3[col] == 1 {
                for k in 0..8 {
                    r_input_3[k] ^= inputs_garbler[col][k];
                }
            }
        }
        println!("Garbler R*input[2]: {:?}", r_input_2);
        println!("Garbler R*input[3]: {:?}", r_input_3);

        // RHS = M*H ⊕ R*input
        let mut rhs_2 = [0u8; 8];
        let mut rhs_3 = [0u8; 8];
        for k in 0..8 {
            rhs_2[k] = m_h_2[k] ^ r_input_2[k];
            rhs_3[k] = m_h_3[k] ^ r_input_3[k];
        }
        println!("Garbler RHS[2]: {:?}", rhs_2);
        println!("Garbler RHS[3]: {:?}", rhs_3);

        // The evaluator should get the same RHS values!
        println!(
            "\nEvaluator RHS[2] (hash_left ^ input_left): {:?}",
            hash_left
                .iter()
                .zip(input_left.iter())
                .map(|(a, b)| a ^ b)
                .collect::<Vec<_>>()
        );
        println!(
            "Evaluator RHS[3] (hash_right ^ input_right): {:?}",
            hash_right
                .iter()
                .zip(input_right.iter())
                .map(|(a, b)| a ^ b)
                .collect::<Vec<_>>()
        );

        // Now test (1,0) - this is the one that's failing
        println!("\n\n=== TESTING INPUT (1,0) ===");
        let a1 = a0 ^ delta;
        let result_10 =
            evaluate_and_gate(cipher, a1, b0, &garbled.gate, &garbled.control_bits, gid);
        println!("Expected: {:?}", c0);
        println!("Got:      {:?}", result_10);
        println!("Match: {}", result_10 == c0);

        // Debug (1,0)
        let i_10 = 1usize;
        let j_10 = 0usize;
        let ij_10 = (i_10 << 1) | j_10;
        println!("i={}, j={}, ij={}", i_10, j_10, ij_10);

        let r_bar_10 = garbled.control_bits.r_bar[ij_10];
        println!("r_bar[{}] = {:?}", ij_10, r_bar_10);

        // Compute hashes for (1,0)
        let mut hash_inputs_10 = [a1, b0, a1 ^ b0];
        cipher.tccr_many(&[tweak; 3], &mut hash_inputs_10);
        let h_a_10 = SlicedLabel::from_block(hash_inputs_10[0]); // H(A₁)
        let h_b_10 = SlicedLabel::from_block(hash_inputs_10[1]); // H(B₀)
        let h_ab_10 = SlicedLabel::from_block(hash_inputs_10[2]); // H(A₁⊕B₀) = H(A₀⊕B₁)

        // Hash contribution for row 5 (right half)
        // M[5] = [0, 0, 1, 0, 0, 1] -> H(B₀) and H(A₀⊕B₁)
        let mut hash_right_10 = [0u8; 8];
        for k in 0..8 {
            hash_right_10[k] = h_b_10.right[k] ^ h_ab_10.right[k];
        }
        println!("Hash contrib right (row 5): {:?}", hash_right_10);

        // Marginal for (1,0)
        let r_dollar_10 = expand_marginal(&r_bar_10);
        let r_p_10 = extract_marginal(&R_P, i_10, j_10);
        println!("R$ marginal for (1,0) = {:?}", r_dollar_10);
        println!("R_P marginal for (1,0) = {:?}", r_p_10);

        let mut total_marginal_10 = [[0u8; 4]; 2];
        for row in 0..2 {
            for col in 0..4 {
                total_marginal_10[row][col] = r_dollar_10[row][col] ^ r_p_10[row][col];
            }
        }
        println!("Total marginal for (1,0) = {:?}", total_marginal_10);

        // Input contribution for row 5
        let a1_sliced = SlicedLabel::from_block(a1);
        let b0_sliced_10 = SlicedLabel::from_block(b0);
        let mut input_right_10 = [0u8; 8];
        if total_marginal_10[1][0] == 1 {
            for k in 0..8 {
                input_right_10[k] ^= a1_sliced.left[k];
            }
        }
        if total_marginal_10[1][1] == 1 {
            for k in 0..8 {
                input_right_10[k] ^= a1_sliced.right[k];
            }
        }
        if total_marginal_10[1][2] == 1 {
            for k in 0..8 {
                input_right_10[k] ^= b0_sliced_10.left[k];
            }
        }
        if total_marginal_10[1][3] == 1 {
            for k in 0..8 {
                input_right_10[k] ^= b0_sliced_10.right[k];
            }
        }
        println!("Input contrib right (row 5): {:?}", input_right_10);

        // Gate contribution for row 5
        // V[5] = [0, 1, 0, 0, 1] -> G₂
        println!("Gate contrib right (row 5) = G₂ = {:?}", garbled.gate.g2);

        // Garbler's RHS[5]
        let mut m_h_5 = [0u8; 8];
        for col in 0..6 {
            if M[5][col] == 1 {
                let h_half = garbler_hashes[col].half(1);
                for k in 0..8 {
                    m_h_5[k] ^= h_half[k];
                }
            }
        }
        let mut r_row_5 = [0u8; 6];
        for col in 0..6 {
            r_row_5[col] = R_A[5][col] ^ R_B[5][col] ^ R_P_MAT[5][col];
        }
        // Add R$ contribution (need rand_bits from garbling - but we know rand_bits was
        // [false, false]) Actually the test uses random rand_bits, so let me
        // skip this for now

        // Check the constraint: RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[5] should be 0
        let mut rhs_0_garbler = [0u8; 8];
        let mut rhs_1_garbler = [0u8; 8];
        for col in 0..6 {
            if M[0][col] == 1 {
                for k in 0..8 {
                    rhs_0_garbler[k] ^= garbler_hashes[col].half(0)[k];
                }
            }
            if M[1][col] == 1 {
                for k in 0..8 {
                    rhs_1_garbler[k] ^= garbler_hashes[col].half(1)[k];
                }
            }
        }
        println!("\nChecking constraint RHS[0]⊕RHS[1]⊕RHS[2]⊕RHS[5] = 0:");
        println!("This constraint involves input contributions which depend on rand_bits...");

        // Compare evaluator's computation
        let eval_rhs_5: Vec<u8> = hash_right_10
            .iter()
            .zip(input_right_10.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        println!("Evaluator RHS[5] = {:?}", eval_rhs_5);

        let eval_c_r: Vec<u8> = eval_rhs_5
            .iter()
            .zip(garbled.gate.g2.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        println!("Evaluator C_R = RHS[5] ⊕ G₂ = {:?}", eval_c_r);

        let expected_c0_sliced = SlicedLabel::from_block(c0);
        println!("Expected C_R = {:?}", expected_c0_sliced.right);
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

                assert_eq!(
                    result, expected,
                    "Trial {}: evaluate({},{}) failed. Expected {:?}, got {:?}",
                    trial, i, j, expected, result
                );
            }
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

    /// Focused debug test for (1,0) mismatch
    #[test]
    fn test_focused_10_debug() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(999);

        let mut a0 = Block::random(&mut rng);
        let mut b0 = Block::random(&mut rng);
        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        a0.set_lsb(false);
        b0.set_lsb(false);

        let a1 = a0 ^ delta;
        let gid = 1;
        let rand_bits = [false, false];

        let garbled = garble_and_gate(cipher, a0, b0, delta, gid, rand_bits);

        // Now manually trace evaluate for (1,0)
        let i = 1usize;
        let j = 0usize;
        let ij = (i << 1) | j;

        // Get marginal
        let r_bar_ij = garbled.control_bits.r_bar[ij];
        let marginal = extract_evaluator_marginal(i, j, &r_bar_ij);
        println!("Marginal for (1,0): {:?}", marginal);

        // Compute evaluator hashes
        let tweak = Block::new((gid as u128).to_be_bytes());
        let mut hash_inputs = [a1, b0, a1 ^ b0];
        cipher.tccr_many(&[tweak; 3], &mut hash_inputs);
        let h_a = SlicedLabel::from_block(hash_inputs[0]);
        let h_b = SlicedLabel::from_block(hash_inputs[1]);
        let h_ab = SlicedLabel::from_block(hash_inputs[2]);

        // Row 5 (right half)
        let row = 5;
        let hash_contrib = compute_hash_contribution_eval(row, &h_a, &h_b, &h_ab);
        println!("Hash contrib (row 5): {:?}", hash_contrib);

        // Input contribution
        let a1_sliced = SlicedLabel::from_block(a1);
        let b0_sliced = SlicedLabel::from_block(b0);
        let input_contrib = compute_input_contribution(&marginal, 1, &a1_sliced, &b0_sliced);
        println!("Input contrib (row 5): {:?}", input_contrib);

        // What evaluate_and_gate actually produces
        let result = evaluate_and_gate(cipher, a1, b0, &garbled.gate, &garbled.control_bits, gid);
        let result_sliced = SlicedLabel::from_block(result);
        let expected_sliced = SlicedLabel::from_block(garbled.output_label);

        println!("Result right: {:?}", result_sliced.right);
        println!("Expected right: {:?}", expected_sliced.right);

        // Check the gate contribution
        let gate_contrib = compute_gate_contribution(row, &garbled.gate);
        println!("Gate contrib (row 5): {:?}", gate_contrib);

        // RHS and C_R calculation
        let mut rhs = [0u8; 8];
        for k in 0..8 {
            rhs[k] = hash_contrib[k] ^ input_contrib[k];
        }
        println!("RHS[5] (hash^input): {:?}", rhs);

        let mut c_r = [0u8; 8];
        for k in 0..8 {
            c_r[k] = rhs[k] ^ gate_contrib[k];
        }
        println!("Computed C_R (RHS^gate): {:?}", c_r);
    }
}
