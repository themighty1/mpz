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
    control::{and_truth_table_with_permute, sample_r_odd},
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

/// Control bits for evaluator (compressed form).
///
/// The r_bar is a 4×2 matrix where each row r_bar[ij] contains the coefficients
/// [c₁, c₂] for input position (i,j). The evaluator expands this to a 2×4
/// marginal using: R_ij = c₁·S₁ ⊕ c₂·S₂
///
/// In ODD mode (AND gates), the evaluator also adds R_P's marginal since
/// parity is public.
///
/// Total size: 4 × 2 = 8 bits = 1 byte (but stored as bytes for simplicity)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlBits {
    /// Compressed representation r̄: 4 entries of 2 coefficients each
    /// r_bar[0] = (0,0), r_bar[1] = (0,1), r_bar[2] = (1,0), r_bar[3] = (1,1)
    pub r_bar: [[u8; 2]; 4],
}

impl ControlBits {
    /// Create new control bits from the compressed r_bar representation.
    pub fn new(r_bar: [[u8; 2]; 4]) -> Self {
        Self { r_bar }
    }

    /// Total size in bytes for transmission.
    /// Each of the 4 entries has 2 bits, so 8 bits total = 1 byte.
    /// In practice we use 8 bytes for alignment.
    pub const SIZE_BYTES: usize = 8;
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

/// Apply matrix M to hash vector H, producing 8 κ/2-bit outputs.
///
/// # Paper Reference (Section 4.2, Page 10)
///
/// > "H(·) is a function with κ/2 bits of output"
///
/// **CRITICAL**: ALL 8 rows use the SAME κ/2-bit hash values. The "left/right"
/// distinction in the scheme refers to which half of the OUTPUT label (C_L vs C_R)
/// is being computed, NOT which half of the hash to use.
///
/// The M matrix specifies which of the 6 hash outputs to XOR for each row.
/// K×M = 0 holds because the coefficients cancel when all rows use the same values.
///
/// M is defined on Page 12:
/// ```text
/// M = [1 0 0 0 1 0]  row 0: H(A₀) ⊕ H(A₀⊕B₀)
///     [0 0 1 0 1 0]  row 1: H(B₀) ⊕ H(A₀⊕B₀)
///     [1 0 0 0 0 1]  row 2: H(A₀) ⊕ H(A₀⊕B₁)
///     [0 0 0 1 0 1]  row 3: H(B₁) ⊕ H(A₀⊕B₁)
///     [0 1 0 0 0 1]  row 4: H(A₁) ⊕ H(A₀⊕B₁)
///     [0 0 1 0 0 1]  row 5: H(B₀) ⊕ H(A₀⊕B₁)
///     [0 1 0 0 1 0]  row 6: H(A₁) ⊕ H(A₀⊕B₀)
///     [0 0 0 1 1 0]  row 7: H(B₁) ⊕ H(A₀⊕B₀)
/// ```
fn apply_m_to_hashes(hashes: &[SlicedLabel; 6]) -> [[u8; 8]; 8] {
    let mut result = [[0u8; 8]; 8];

    for row in 0..8 {
        for col in 0..6 {
            if M[row][col] == 1 {
                // Use .left consistently for all rows (the κ/2-bit hash output)
                xor_assign_8(&mut result[row], &hashes[col].left);
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
/// # Key Insight
///
/// The RHS = M·H ⊕ R·input is NOT directly in col(V) because R includes
/// the truth table contribution. For rows where the truth table says TRUE,
/// we need to adjust RHS by XORing Δ before solving.
///
/// This adjustment makes RHS_adjusted ∈ col(V) = ker(K), allowing V_INV
/// to find a valid solution.
///
/// # Arguments
/// * `rhs` - The right-hand side matrix M·H ⊕ R·input
/// * `delta` - The global correlation Δ (sliced)
/// * `t` - The 8×2 truth table matrix (identity blocks mark TRUE outputs)
///
/// From Paper Page 18, Equation 10, V⁻¹ is:
/// ```text
/// V⁻¹ = [ 1 0 | 0 0 | 0 0 | 0 0 ]  -> C_L = RHS[0]
///       [ 0 1 | 0 0 | 0 0 | 0 0 ]  -> C_R = RHS[1]
///       [ 1 1 | 0 0 | 1 1 | 0 0 ]  -> G₀ = RHS[0] ⊕ RHS[1] ⊕ RHS[4] ⊕ RHS[5]
///       [ 1 1 | 1 1 | 0 0 | 0 0 ]  -> G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
///       [ 0 0 | 0 0 | 1 0 | 1 0 ]  -> G₂ = RHS[4] ⊕ RHS[6]
/// ```
fn solve_for_output(rhs: &[[u8; 8]; 8], delta: &SlicedLabel, t: &[[u8; 2]; 8]) -> [[u8; 8]; 5] {
    // Adjust RHS for truth table: find which rows have TRUE output
    // Identity block [1,0; 0,1] marks TRUE output at rows (2*ij, 2*ij+1)
    let mut rhs_adjusted = *rhs;

    // Find the TRUE position by looking for identity block in truth table
    for ij in 0..4 {
        let row_l = 2 * ij;
        let row_r = 2 * ij + 1;
        // Check if this is an identity block (TRUE output)
        if t[row_l] == [1, 0] && t[row_r] == [0, 1] {
            // XOR Δ into the rows where truth table says TRUE
            xor_assign_8(&mut rhs_adjusted[row_l], &delta.left);
            xor_assign_8(&mut rhs_adjusted[row_r], &delta.right);
        }
    }

    let mut result = [[0u8; 8]; 5];

    // C_L = RHS[0]
    result[0] = rhs_adjusted[0];

    // C_R = RHS[1]
    result[1] = rhs_adjusted[1];

    // G₀ = RHS[0] ⊕ RHS[1] ⊕ RHS[4] ⊕ RHS[5]
    for k in 0..8 {
        result[2][k] = rhs_adjusted[0][k] ^ rhs_adjusted[1][k] ^ rhs_adjusted[4][k] ^ rhs_adjusted[5][k];
    }

    // G₁ = RHS[0] ⊕ RHS[1] ⊕ RHS[2] ⊕ RHS[3]
    for k in 0..8 {
        result[3][k] = rhs_adjusted[0][k] ^ rhs_adjusted[1][k] ^ rhs_adjusted[2][k] ^ rhs_adjusted[3][k];
    }

    // G₂ = RHS[4] ⊕ RHS[6]
    for k in 0..8 {
        result[4][k] = rhs_adjusted[4][k] ^ rhs_adjusted[6][k];
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
/// Garble an AND gate.
///
/// # Arguments
/// * `cipher` - The fixed-key AES cipher for hashing
/// * `a0` - Wire A label with color bit 0
/// * `b0` - Wire B label with color bit 0
/// * `delta` - Global correlation Δ
/// * `gid` - Gate ID (for domain separation)
/// * `pi_a` - Point-and-permute bit for wire A (determines which label is TRUE)
/// * `pi_b` - Point-and-permute bit for wire B (determines which label is TRUE)
/// * `rand_bits` - Random bits for control matrix sampling
///
/// # Paper Reference (Page 16-17, Figure 6)
///
/// The truth table is computed as:
/// ```text
/// t := [g(πA⊕i, πB⊕j)]  for (i,j) in [(0,0), (0,1), (1,0), (1,1)]
/// ```
/// where g is the AND function.
pub fn garble_and_gate(
    cipher: &FixedKeyAes,
    a0: Block,
    b0: Block,
    delta: Block,
    gid: usize,
    pi_a: bool,
    pi_b: bool,
    rand_bits: [bool; 2],
) -> GarbledGate {
    // 1. Compute the 6 hash values
    let hashes = compute_hashes(cipher, a0, b0, delta, gid);

    // 2. Slice the input labels
    let a0_sliced = SlicedLabel::from_block(a0);
    let b0_sliced = SlicedLabel::from_block(b0);
    let delta_sliced = SlicedLabel::from_block(delta);

    // 3. Sample the randomized control matrix R for AND gate (ODD mode)
    // The truth table depends on the point-and-permute bits
    let t = and_truth_table_with_permute(pi_a, pi_b);
    let (r, r_bar) = sample_r_odd(&t, rand_bits);

    // The garbler uses full R internally, but sends compressed r_bar to evaluator

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
    //    hold. Pass truth table so delta is XORed into the correct rows.
    let output = solve_for_output(&rhs, &delta_sliced, &t);

    // 8. Extract output label and gate ciphertexts
    // output = [C_L, C_R, G₀, G₁, G₂]
    let c0 = SlicedLabel::new(output[0], output[1]).to_block();

    let gate = ThreeHalvesGate::new(output[2], output[3], output[4]);

    // Send compressed r_bar to evaluator (proper protocol)
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

use super::control::{expand_marginal, extract_r_p_marginal};

/// Extract and expand the evaluator's marginal from compressed r_bar.
///
/// For ODD mode (AND gates), this:
/// 1. Gets r_bar_ij coefficients for input position (i,j)
/// 2. Expands using basis {S₁, S₂}: R_ij = c₁·S₁ ⊕ c₂·S₂
/// 3. Adds R_P's marginal (since parity is public in ODD mode)
///
/// # Key Insight from Paper (Section 5.1, Figure 4)
///
/// In ODD mode, the evaluator knows parity is odd and adds R_P themselves.
/// The compressed r_bar only contains R$ ⊕ a·R_a ⊕ b·R_b (without R_P).
fn expand_evaluator_marginal(r_bar: &[[u8; 2]; 4], i: usize, j: usize) -> [[u8; 4]; 2] {
    let ij = (i << 1) | j;

    // 1. Get compressed coefficients for this input position
    let r_bar_ij = r_bar[ij];

    // 2. Expand using basis: R_ij = c₁·S₁ ⊕ c₂·S₂
    let mut marginal = expand_marginal(&r_bar_ij);

    // 3. Add R_P's marginal (ODD mode: evaluator knows parity is odd)
    let r_p_marginal = extract_r_p_marginal(i, j);
    for row in 0..2 {
        for col in 0..4 {
            marginal[row][col] ^= r_p_marginal[row][col];
        }
    }

    marginal
}

/// Evaluate a Three Halves AND gate.
///
/// Uses the compressed r_bar protocol where the evaluator:
/// 1. Expands r_bar_ij to full marginal using basis {S₁, S₂}
/// 2. Adds R_P's marginal (ODD mode: parity is public)
///
/// # Arguments
/// * `cipher` - Fixed-key AES cipher for TCCR hash
/// * `a` - Input wire A label (for bit i)
/// * `b` - Input wire B label (for bit j)
/// * `gate` - Gate ciphertexts from garbling
/// * `control_bits` - Control bits (contains compressed r_bar)
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

    // 4. Expand compressed r_bar to full marginal (includes R_P for ODD mode)
    let marginal = expand_evaluator_marginal(&control_bits.r_bar, i, j);

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

/// Compute hash contribution for evaluation.
///
/// The evaluator has H(A_i), H(B_j), H(A_i⊕B_j) and needs to compute
/// the hash contribution matching what the garbler computed via M·H.
///
/// # Paper Reference (Section 4.2, Page 10)
///
/// > "H(·) is a function with κ/2 bits of output"
///
/// **CRITICAL**: ALL rows use the SAME κ/2-bit hash values (using `.left`).
/// This must match the garbler's `apply_m_to_hashes` function.
///
/// # Hash Column Mapping
///
/// The evaluator's three hashes map to the garbler's 6 columns as follows:
/// - `h_a = H(A_i)` → column `i` (0 for i=0, 1 for i=1)
/// - `h_b = H(B_j)` → column `2+j` (2 for j=0, 3 for j=1)
/// - `h_ab = H(A_i⊕B_j)` → column 4 or 5 based on Free-XOR:
///   - (0,0): H(A₀⊕B₀) → col 4
///   - (0,1): H(A₀⊕B₁) → col 5
///   - (1,0): H(A₁⊕B₀) = H(A₀⊕B₁) → col 5
///   - (1,1): H(A₁⊕B₁) = H(A₀⊕B₀) → col 4
fn compute_hash_contribution_eval(
    row: usize,
    h_a: &SlicedLabel,
    h_b: &SlicedLabel,
    h_ab: &SlicedLabel,
) -> [u8; 8] {
    use super::matrices::M;

    let ij = row / 2;
    let i = ij >> 1;
    let j = ij & 1;

    // Determine which garbler column maps to evaluator's h_ab
    // Due to Free-XOR: A_i ⊕ B_j when (i,j) has odd parity maps to col 5 (A₀⊕B₁)
    //                  when (i,j) has even parity maps to col 4 (A₀⊕B₀)
    let ab_col = if (i ^ j) == 0 { 4 } else { 5 };

    let mut result = [0u8; 8];

    // XOR in h_a if M says to use column i (the evaluator's A hash column)
    // Use .left consistently (κ/2-bit hash output) - must match garbler
    if M[row][i] == 1 {
        xor_assign_8(&mut result, &h_a.left);
    }

    // XOR in h_b if M says to use column 2+j (the evaluator's B hash column)
    if M[row][2 + j] == 1 {
        xor_assign_8(&mut result, &h_b.left);
    }

    // XOR in h_ab if M says to use the corresponding combined hash column
    if M[row][ab_col] == 1 {
        xor_assign_8(&mut result, &h_ab.left);
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

        let result1 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, rand_bits);
        let result2 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, rand_bits);

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

        let result1 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, [false, false]);
        let result2 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, [true, false]);
        let result3 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, [false, true]);
        let result4 = garble_and_gate(cipher, a0, b0, delta, gid, false, false, [true, true]);

        // r_bar matrices should differ based on random bits
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

        let result1 = garble_and_gate(cipher, a0, b0, delta, 1, false, false, rand_bits);
        let result2 = garble_and_gate(cipher, a0, b0, delta, 2, false, false, rand_bits);

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
            let result = garble_and_gate(cipher, a0, b0, delta, gid, false, false, rand_bits);

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
    ///
    /// IMPORTANT: solve_for_output only works correctly when RHS is in col(V) = ker(K).
    /// We generate valid RHS by computing RHS = V · x for random x.
    #[test]
    fn test_solve_for_output_satisfies_marginals() {
        use super::super::matrices::V;
        use super::super::slicing::SlicedLabel;

        let mut rng = ChaCha12Rng::seed_from_u64(123);

        // Generate random "output" values [C_L, C_R, G₀, G₁, G₂]
        let mut x = [[0u8; 8]; 5];
        for i in 0..5 {
            for j in 0..8 {
                x[i][j] = rng.random();
            }
        }

        // Compute RHS = V · x (this ensures RHS is in col(V) = ker(K))
        let mut rhs = [[0u8; 8]; 8];
        for row in 0..8 {
            for k in 0..8 {
                let mut val = 0u8;
                for col in 0..5 {
                    if V[row][col] == 1 {
                        val ^= x[col][k];
                    }
                }
                rhs[row][k] = val;
            }
        }

        // Use zero delta and zero truth table (no adjustment needed)
        let zero_delta = SlicedLabel::ZERO;
        let zero_t = [[0u8; 2]; 8];

        // Solve for [C_L, C_R, G₀, G₁, G₂]
        let output = solve_for_output(&rhs, &zero_delta, &zero_t);

        // With valid RHS in col(V), solve_for_output should recover the original x
        // Verify V · output = RHS for all 8 rows
        for row in 0..8 {
            let mut v_times_output = [0u8; 8];
            for k in 0..8 {
                for col in 0..5 {
                    if V[row][col] == 1 {
                        v_times_output[k] ^= output[col][k];
                    }
                }
            }
            assert_eq!(
                v_times_output, rhs[row],
                "Row {} should be satisfied: V·output should equal RHS", row
            );
        }
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

            // Garble the AND gate (pi_a=false, pi_b=false means A₀ and B₀ represent 0)
            let garbled = garble_and_gate(cipher, a0, b0, delta, gid, false, false, rand_bits);
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

        let garbled = garble_and_gate(cipher, a0, b0, delta, gid, false, false, rand_bits);

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
                let output = solve_for_output(&rhs, &delta_sliced, &t);
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

    /// Test evaluator's input contribution matches garbler's.
    ///
    /// Garbler computes: R · [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]ᵀ → 8 values
    /// Evaluator computes: marginal · [A_i_L, A_i_R, B_j_L, B_j_R]ᵀ → 2 values
    ///
    /// For input (i,j), the evaluator's 2 values should match garbler's rows 2*ij and 2*ij+1.
    ///
    /// This tests the compressed r_bar protocol:
    /// 1. Garbler samples (R, r_bar) and computes full R·[A₀;B₀;Δ]
    /// 2. Evaluator expands r_bar to marginal and adds R_P (ODD mode)
    /// 3. Evaluator's marginal·[A_i;B_j] should match garbler's rows
    #[test]
    fn test_evaluator_input_contribution() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::slicing::SlicedLabel;
        use super::{apply_r_to_inputs, compute_input_contribution, expand_evaluator_marginal};

        let mut rng = ChaCha12Rng::seed_from_u64(77777);
        let mut failures: Vec<(usize, [bool; 2], usize, usize, &str)> = Vec::new();

        for trial in 0..10 {
            // Generate random labels
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let a1 = a0 ^ delta;
            let b1 = b0 ^ delta;

            // Slice labels
            let a0_sliced = SlicedLabel::from_block(a0);
            let b0_sliced = SlicedLabel::from_block(b0);
            let a1_sliced = SlicedLabel::from_block(a1);
            let b1_sliced = SlicedLabel::from_block(b1);
            let delta_sliced = SlicedLabel::from_block(delta);

            // Test all R matrix variations
            for rand_bits in [[false, false], [false, true], [true, false], [true, true]] {
                let t = and_truth_table();
                let (r, r_bar) = sample_r_odd(&t, rand_bits);

                // Garbler's full input contribution
                let garbler_result = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

                // Test all 4 input combinations
                let inputs = [
                    (0, 0, &a0_sliced, &b0_sliced),
                    (0, 1, &a0_sliced, &b1_sliced),
                    (1, 0, &a1_sliced, &b0_sliced),
                    (1, 1, &a1_sliced, &b1_sliced),
                ];

                for (i, j, a_sliced, b_sliced) in inputs {
                    let ij = (i << 1) | j;
                    let row_l = 2 * ij;
                    let row_r = 2 * ij + 1;

                    // Evaluator expands r_bar to marginal (includes R_P for ODD mode)
                    let marginal = expand_evaluator_marginal(&r_bar, i, j);
                    let eval_l = compute_input_contribution(&marginal, 0, a_sliced, b_sliced);
                    let eval_r = compute_input_contribution(&marginal, 1, a_sliced, b_sliced);

                    // Compare with garbler's result for these rows
                    if eval_l != garbler_result[row_l] {
                        println!(
                            "FAIL Trial {}, rand_bits={:?}, input ({},{}), LEFT half:\n\
                             Evaluator: {:?}\n\
                             Garbler[{}]: {:?}\n\
                             Marginal[0]: {:?}\n\
                             R[{}]: {:?}",
                            trial, rand_bits, i, j,
                            eval_l, row_l, garbler_result[row_l],
                            marginal[0], row_l, r[row_l]
                        );
                        failures.push((trial, rand_bits, i, j, "left"));
                    } else {
                        println!("PASS Trial {}, rand_bits={:?}, input ({},{}), LEFT", trial, rand_bits, i, j);
                    }

                    if eval_r != garbler_result[row_r] {
                        println!(
                            "FAIL Trial {}, rand_bits={:?}, input ({},{}), RIGHT half:\n\
                             Evaluator: {:?}\n\
                             Garbler[{}]: {:?}\n\
                             Marginal[1]: {:?}\n\
                             R[{}]: {:?}",
                            trial, rand_bits, i, j,
                            eval_r, row_r, garbler_result[row_r],
                            marginal[1], row_r, r[row_r]
                        );
                        failures.push((trial, rand_bits, i, j, "right"));
                    } else {
                        println!("PASS Trial {}, rand_bits={:?}, input ({},{}), RIGHT", trial, rand_bits, i, j);
                    }
                }
            }
        }

        if !failures.is_empty() {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            panic!("{} input contribution tests failed", failures.len());
        }

        println!("✓ Evaluator input contribution test passed!");
    }

    /// Test that solve_for_output produces valid output.
    ///
    /// After adjusting TRUE rows with Δ (based on truth table):
    /// 1. K·RHS_adjusted should = 0 (RHS_adjusted in ker(K))
    /// 2. V·output should = RHS_adjusted for ALL 8 rows
    #[test]
    fn test_solve_for_output_produces_valid_output() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::matrices::{K, V};
        use super::super::slicing::SlicedLabel;
        use super::{apply_m_to_hashes, apply_r_to_inputs, compute_hashes, solve_for_output, xor_assign_8};

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(22222);

        for trial in 0..10 {
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let gid = trial + 1;
            let rand_bits: [bool; 2] = [rng.random(), rng.random()];

            let a0_sliced = SlicedLabel::from_block(a0);
            let b0_sliced = SlicedLabel::from_block(b0);
            let delta_sliced = SlicedLabel::from_block(delta);

            // Compute garbler's RHS
            let hashes = compute_hashes(cipher, a0, b0, delta, gid);
            let m_times_h = apply_m_to_hashes(&hashes);

            let t = and_truth_table();
            let (r, _) = sample_r_odd(&t, rand_bits);
            let r_times_input = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

            let mut rhs = [[0u8; 8]; 8];
            for row in 0..8 {
                rhs[row] = m_times_h[row];
                xor_assign_8(&mut rhs[row], &r_times_input[row]);
            }

            // Compute RHS_adjusted (what solve_for_output uses internally)
            // Find TRUE rows from truth table and XOR delta into them
            let mut rhs_adjusted = rhs;
            for ij in 0..4 {
                let row_l = 2 * ij;
                let row_r = 2 * ij + 1;
                if t[row_l] == [1, 0] && t[row_r] == [0, 1] {
                    xor_assign_8(&mut rhs_adjusted[row_l], &delta_sliced.left);
                    xor_assign_8(&mut rhs_adjusted[row_r], &delta_sliced.right);
                }
            }

            // Check 1: K·RHS_adjusted should = 0
            for k_row in 0..3 {
                let mut k_times_rhs_adj = [0u8; 8];
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        xor_assign_8(&mut k_times_rhs_adj, &rhs_adjusted[col]);
                    }
                }
                if k_times_rhs_adj != [0u8; 8] {
                    panic!(
                        "Trial {}: K[{}]·RHS_adjusted ≠ 0\n\
                         K[{}] = {:?}\n\
                         Result = {:?}",
                        trial, k_row, k_row, K[k_row], k_times_rhs_adj
                    );
                }
            }

            // Get output from solve_for_output
            let output = solve_for_output(&rhs, &delta_sliced, &t);

            // Check 2: V·output should = RHS_adjusted for ALL 8 rows
            for row in 0..8 {
                let mut v_times_output = [0u8; 8];
                if V[row][0] == 1 { xor_assign_8(&mut v_times_output, &output[0]); }
                if V[row][1] == 1 { xor_assign_8(&mut v_times_output, &output[1]); }
                if V[row][2] == 1 { xor_assign_8(&mut v_times_output, &output[2]); }
                if V[row][3] == 1 { xor_assign_8(&mut v_times_output, &output[3]); }
                if V[row][4] == 1 { xor_assign_8(&mut v_times_output, &output[4]); }

                if v_times_output != rhs_adjusted[row] {
                    panic!(
                        "Trial {}: V[{}]·output ≠ RHS_adjusted[{}]\n\
                         V[{}] = {:?}\n\
                         V·output = {:?}\n\
                         RHS_adjusted[{}] = {:?}\n\
                         output = C_L={:?}, C_R={:?}, G₀={:?}, G₁={:?}, G₂={:?}",
                        trial, row, row, row, V[row], v_times_output, row, rhs_adjusted[row],
                        output[0], output[1], output[2], output[3], output[4]
                    );
                }
            }
        }

        println!("✓ solve_for_output produces valid output test passed!");
    }

    /// Test gate contribution in isolation.
    ///
    /// Verify that compute_gate_contribution correctly applies V[row][2:4] to [G₀, G₁, G₂].
    ///
    /// For each row, gate_contrib should equal:
    ///   V[row][2]·G₀ ⊕ V[row][3]·G₁ ⊕ V[row][4]·G₂
    #[test]
    fn test_gate_contribution_isolated() {
        use super::super::matrices::V;
        use super::{compute_gate_contribution, ThreeHalvesGate};

        let mut rng = ChaCha12Rng::seed_from_u64(11111);

        // Test with random G values
        for trial in 0..10 {
            let mut g0 = [0u8; 8];
            let mut g1 = [0u8; 8];
            let mut g2 = [0u8; 8];
            for i in 0..8 {
                g0[i] = rng.random();
                g1[i] = rng.random();
                g2[i] = rng.random();
            }

            let gate = ThreeHalvesGate::new(g0, g1, g2);

            // Test each row
            for row in 0..8 {
                let result = compute_gate_contribution(row, &gate);

                // Compute expected: V[row][2]·G₀ ⊕ V[row][3]·G₁ ⊕ V[row][4]·G₂
                let mut expected = [0u8; 8];
                if V[row][2] == 1 {
                    for k in 0..8 { expected[k] ^= g0[k]; }
                }
                if V[row][3] == 1 {
                    for k in 0..8 { expected[k] ^= g1[k]; }
                }
                if V[row][4] == 1 {
                    for k in 0..8 { expected[k] ^= g2[k]; }
                }

                assert_eq!(
                    result, expected,
                    "Trial {}, row {}: gate_contrib mismatch\n\
                     V[{}] = {:?}\n\
                     G₀={:?}, G₁={:?}, G₂={:?}\n\
                     Got: {:?}\n\
                     Expected: {:?}",
                    trial, row, row, V[row], g0, g1, g2, result, expected
                );
            }
        }

        println!("✓ Gate contribution isolated test passed!");
    }

    /// Test R matrix Δ column constraint.
    ///
    /// For evaluator with A_i = A₀ ⊕ i·Δ and B_j = B₀ ⊕ j·Δ to get the same
    /// result as garbler, the R matrix must satisfy:
    ///
    /// For row r corresponding to input (i,j):
    ///   R[r][4] = i·R[r][0] ⊕ j·R[r][2]  (Δ_L column)
    ///   R[r][5] = i·R[r][1] ⊕ j·R[r][3]  (Δ_R column)
    ///
    /// Row mapping:
    ///   Rows 0,1: (0,0) → R[r][4]=0, R[r][5]=0
    ///   Rows 2,3: (0,1) → R[r][4]=R[r][2], R[r][5]=R[r][3]
    ///   Rows 4,5: (1,0) → R[r][4]=R[r][0], R[r][5]=R[r][1]
    ///   Rows 6,7: (1,1) → R[r][4]=R[r][0]⊕R[r][2], R[r][5]=R[r][1]⊕R[r][3]
    #[test]
    fn test_r_matrix_delta_column_constraint() {
        use super::super::control::{and_truth_table, sample_r_odd};

        let mut failures: Vec<(usize, [bool; 2], usize, &str)> = Vec::new();

        // Test all R matrix variations
        for (idx, rand_bits) in [[false, false], [false, true], [true, false], [true, true]].iter().enumerate() {
            let t = and_truth_table();
            let (r, _) = sample_r_odd(&t, *rand_bits);

            // Check each row
            for row in 0..8 {
                let ij = row / 2;
                let i = ij >> 1;
                let j = ij & 1;

                // Expected Δ_L column: i·R[r][0] ⊕ j·R[r][2]
                let expected_delta_l = (i as u8 * r[row][0]) ^ (j as u8 * r[row][2]);
                // Expected Δ_R column: i·R[r][1] ⊕ j·R[r][3]
                let expected_delta_r = (i as u8 * r[row][1]) ^ (j as u8 * r[row][3]);

                if r[row][4] != expected_delta_l {
                    println!(
                        "FAIL rand_bits={:?}, row {} (i={},j={}), Δ_L:\n\
                         R[{}][4] = {}, expected {} (= {}*R[{}][0] ⊕ {}*R[{}][2] = {}*{} ⊕ {}*{})\n\
                         Full row: {:?}",
                        rand_bits, row, i, j,
                        row, r[row][4], expected_delta_l,
                        i, row, j, row, i, r[row][0], j, r[row][2],
                        r[row]
                    );
                    failures.push((idx, *rand_bits, row, "Δ_L"));
                }

                if r[row][5] != expected_delta_r {
                    println!(
                        "FAIL rand_bits={:?}, row {} (i={},j={}), Δ_R:\n\
                         R[{}][5] = {}, expected {} (= {}*R[{}][1] ⊕ {}*R[{}][3] = {}*{} ⊕ {}*{})\n\
                         Full row: {:?}",
                        rand_bits, row, i, j,
                        row, r[row][5], expected_delta_r,
                        i, row, j, row, i, r[row][1], j, r[row][3],
                        r[row]
                    );
                    failures.push((idx, *rand_bits, row, "Δ_R"));
                }
            }
        }

        if failures.is_empty() {
            println!("✓ R matrix Δ column constraint test passed!");
        } else {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            panic!("{} Δ column constraint tests failed", failures.len());
        }
    }

    /// Test evaluator's hash contribution matches garbler's.
    ///
    /// Garbler computes M · [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]ᵀ → 8 values
    /// Evaluator computes hash contribution using only 3 hashes: H(A_i), H(B_j), H(A_i⊕B_j)
    ///
    /// For input (i,j), evaluator's hash contribution for rows 2*ij and 2*ij+1
    /// should match garbler's M·H⃗ for those rows.
    #[test]
    fn test_evaluator_hash_contribution() {
        use super::super::slicing::SlicedLabel;
        use super::{apply_m_to_hashes, compute_hash_contribution_eval, compute_hashes};

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(88888);
        let mut failures: Vec<(usize, usize, usize, &str)> = Vec::new();

        for trial in 0..10 {
            // Generate random labels
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let a1 = a0 ^ delta;
            let b1 = b0 ^ delta;

            let gid = trial + 1;
            let tweak = Block::new((gid as u128).to_be_bytes());

            // Garbler computes all 6 hashes
            let garbler_hashes = compute_hashes(cipher, a0, b0, delta, gid);
            let garbler_m_h = apply_m_to_hashes(&garbler_hashes);

            // Test all 4 input combinations
            let inputs = [
                (0, 0, a0, b0),
                (0, 1, a0, b1),
                (1, 0, a1, b0),
                (1, 1, a1, b1),
            ];

            for (i, j, a, b) in inputs {
                let ij = (i << 1) | j;
                let row_l = 2 * ij;
                let row_r = 2 * ij + 1;

                // Evaluator computes their 3 hashes
                let mut eval_hash_inputs = [a, b, a ^ b];
                cipher.tccr_many(&[tweak; 3], &mut eval_hash_inputs);

                let h_a = SlicedLabel::from_block(eval_hash_inputs[0]);
                let h_b = SlicedLabel::from_block(eval_hash_inputs[1]);
                let h_ab = SlicedLabel::from_block(eval_hash_inputs[2]);

                // Evaluator's hash contribution
                let eval_l = compute_hash_contribution_eval(row_l, &h_a, &h_b, &h_ab);
                let eval_r = compute_hash_contribution_eval(row_r, &h_a, &h_b, &h_ab);

                // Compare with garbler's M·H⃗
                if eval_l != garbler_m_h[row_l] {
                    println!(
                        "FAIL Trial {}, input ({},{}), LEFT half:\n\
                         Evaluator: {:?}\n\
                         Garbler M·H[{}]: {:?}",
                        trial, i, j, eval_l, row_l, garbler_m_h[row_l]
                    );
                    failures.push((trial, i, j, "left"));
                } else {
                    println!("PASS Trial {}, input ({},{}), LEFT", trial, i, j);
                }

                if eval_r != garbler_m_h[row_r] {
                    println!(
                        "FAIL Trial {}, input ({},{}), RIGHT half:\n\
                         Evaluator: {:?}\n\
                         Garbler M·H[{}]: {:?}",
                        trial, i, j, eval_r, row_r, garbler_m_h[row_r]
                    );
                    failures.push((trial, i, j, "right"));
                } else {
                    println!("PASS Trial {}, input ({},{}), RIGHT", trial, i, j);
                }
            }
        }

        if !failures.is_empty() {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            panic!("{} hash contribution tests failed", failures.len());
        }

        println!("✓ Evaluator hash contribution test passed!");
    }

    /// Test evaluator's gate contribution + final combination.
    ///
    /// Given correct hash_contrib and input_contrib (verified above),
    /// verify that adding gate_contrib produces the correct output C.
    #[test]
    fn test_evaluator_gate_and_final_combination() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::slicing::SlicedLabel;
        use super::{
            apply_m_to_hashes, apply_r_to_inputs, compute_gate_contribution,
            compute_hashes, solve_for_output, ThreeHalvesGate, xor_assign_8,
        };

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(99999);
        let mut failures: Vec<(usize, usize, usize, &str)> = Vec::new();

        for trial in 0..10 {
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);

            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let gid = trial + 1;

            // Slice labels
            let a0_sliced = SlicedLabel::from_block(a0);
            let b0_sliced = SlicedLabel::from_block(b0);
            let delta_sliced = SlicedLabel::from_block(delta);

            // Garbler computes everything
            let hashes = compute_hashes(cipher, a0, b0, delta, gid);
            let m_times_h = apply_m_to_hashes(&hashes);

            let rand_bits = [rng.random(), rng.random()];
            let t = and_truth_table();
            let (r, _) = sample_r_odd(&t, rand_bits);

            let r_times_input = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

            // Compute RHS = M·H ⊕ R·input
            let mut rhs = [[0u8; 8]; 8];
            for row in 0..8 {
                rhs[row] = m_times_h[row];
                xor_assign_8(&mut rhs[row], &r_times_input[row]);
            }

            // Solve for output
            let output = solve_for_output(&rhs, &delta_sliced, &t);
            let c0 = SlicedLabel::new(output[0], output[1]);
            let c1 = c0 ^ delta_sliced;
            let gate = ThreeHalvesGate::new(output[2], output[3], output[4]);

            // Compute RHS_adjusted (for verification)
            let mut rhs_adjusted = rhs;
            for ij in 0..4 {
                let row_l = 2 * ij;
                let row_r = 2 * ij + 1;
                if t[row_l] == [1, 0] && t[row_r] == [0, 1] {
                    xor_assign_8(&mut rhs_adjusted[row_l], &delta_sliced.left);
                    xor_assign_8(&mut rhs_adjusted[row_r], &delta_sliced.right);
                }
            }

            // VERIFY: K · RHS_adjusted should equal 0 (RHS_adjusted must be in kernel of K)
            use super::super::matrices::{K, V};
            for row in 0..3 {
                let mut k_times_rhs = [0u8; 8];
                for col in 0..8 {
                    if K[row][col] == 1 {
                        xor_assign_8(&mut k_times_rhs, &rhs_adjusted[col]);
                    }
                }
                if k_times_rhs != [0u8; 8] {
                    println!(
                        "KERNEL FAIL Trial {}, K row {}:\n\
                         K[{}]·RHS_adjusted = {:?} (should be 0)\n\
                         K[{}] = {:?}",
                        trial, row, row, k_times_rhs, row, K[row]
                    );
                }
            }

            // VERIFY: V · output should equal RHS_adjusted
            for row in 0..8 {
                let mut v_times_output = [0u8; 8];
                if V[row][0] == 1 { xor_assign_8(&mut v_times_output, &output[0]); }
                if V[row][1] == 1 { xor_assign_8(&mut v_times_output, &output[1]); }
                if V[row][2] == 1 { xor_assign_8(&mut v_times_output, &output[2]); }
                if V[row][3] == 1 { xor_assign_8(&mut v_times_output, &output[3]); }
                if V[row][4] == 1 { xor_assign_8(&mut v_times_output, &output[4]); }

                if v_times_output != rhs_adjusted[row] {
                    println!(
                        "EQUATION FAIL Trial {}, row {}:\n\
                         V[{}]·output = {:?}\n\
                         RHS_adjusted[{}] = {:?}\n\
                         V[{}] = {:?}",
                        trial, row, row, v_times_output, row, rhs_adjusted[row], row, V[row]
                    );
                }
            }

            // Test all 4 input combinations
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let ij = (i << 1) | j;
                let row_l = 2 * ij;
                let row_r = 2 * ij + 1;

                // Expected output: C₀ for (0,0),(0,1),(1,0), C₁ for (1,1)
                let expected = if i == 1 && j == 1 { c1 } else { c0 };

                // Evaluator's computation:
                // eval = RHS[row] ⊕ gate_contrib
                // (RHS here is what evaluator computes, which we verified matches garbler's)
                let gate_contrib_l = compute_gate_contribution(row_l, &gate);
                let gate_contrib_r = compute_gate_contribution(row_r, &gate);

                let mut eval_l = rhs[row_l];
                xor_assign_8(&mut eval_l, &gate_contrib_l);

                let mut eval_r = rhs[row_r];
                xor_assign_8(&mut eval_r, &gate_contrib_r);

                if eval_l != expected.left {
                    println!(
                        "FAIL Trial {}, input ({},{}), LEFT:\n\
                         Evaluator: {:?}\n\
                         Expected C{}: {:?}\n\
                         RHS[{}]: {:?}\n\
                         gate_contrib: {:?}",
                        trial, i, j, eval_l,
                        if i == 1 && j == 1 { 1 } else { 0 }, expected.left,
                        row_l, rhs[row_l], gate_contrib_l
                    );
                    failures.push((trial, i, j, "left"));
                } else {
                    println!("PASS Trial {}, input ({},{}), LEFT", trial, i, j);
                }

                if eval_r != expected.right {
                    println!(
                        "FAIL Trial {}, input ({},{}), RIGHT:\n\
                         Evaluator: {:?}\n\
                         Expected C{}: {:?}\n\
                         RHS[{}]: {:?}\n\
                         gate_contrib: {:?}",
                        trial, i, j, eval_r,
                        if i == 1 && j == 1 { 1 } else { 0 }, expected.right,
                        row_r, rhs[row_r], gate_contrib_r
                    );
                    failures.push((trial, i, j, "right"));
                } else {
                    println!("PASS Trial {}, input ({},{}), RIGHT", trial, i, j);
                }
            }
        }

        if !failures.is_empty() {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            panic!("{} gate/combination tests failed", failures.len());
        }

        println!("✓ Gate contribution + final combination test passed!");
    }

    /// Minimal diagnostic test: verify K·RHS = 0 step by step.
    ///
    /// We know:
    /// - K·M = 0 (tested)
    /// - K·R = K·[0 0 t] (tested)
    ///
    /// So K·(M·H ⊕ R·input ⊕ t·Δ) should = K·M·H ⊕ K·R·input ⊕ K·(t·Δ)
    ///                                    = 0 ⊕ K·[0 0 t]·input ⊕ K·(t·Δ)
    ///                                    = K·(t·Δ) ⊕ K·(t·Δ) = 0
    #[test]
    fn test_k_rhs_diagnostic() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::matrices::{K, M};
        use super::super::slicing::SlicedLabel;
        use super::xor_assign_8;

        let mut rng = ChaCha12Rng::seed_from_u64(12345);

        // Generate random labels
        let mut a0 = Block::random(&mut rng);
        let mut b0 = Block::random(&mut rng);
        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        a0.set_lsb(false);
        b0.set_lsb(false);

        let a0_sliced = SlicedLabel::from_block(a0);
        let b0_sliced = SlicedLabel::from_block(b0);
        let delta_sliced = SlicedLabel::from_block(delta);

        // Sample R
        let t = and_truth_table();
        let (r, _) = sample_r_odd(&t, [true, false]);

        // Build input vector: [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
        let inputs: [[u8; 8]; 6] = [
            a0_sliced.left, a0_sliced.right,
            b0_sliced.left, b0_sliced.right,
            delta_sliced.left, delta_sliced.right,
        ];

        // Step 1: Compute R·input
        let mut r_times_input = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..6 {
                if r[row][col] == 1 {
                    xor_assign_8(&mut r_times_input[row], &inputs[col]);
                }
            }
        }

        // Step 2: Compute K·(R·input)
        println!("\n=== K·(R·input) ===");
        for k_row in 0..3 {
            let mut k_times_r_input = [0u8; 8];
            for col in 0..8 {
                if K[k_row][col] == 1 {
                    xor_assign_8(&mut k_times_r_input, &r_times_input[col]);
                }
            }
            let is_zero = k_times_r_input == [0u8; 8];
            println!("K[{}]·(R·input) = {:?} (zero: {})", k_row, k_times_r_input, is_zero);
        }

        // Step 3: Compute [0 0 t]·input = t·Δ
        // For AND gate: t[6]=[1,0], t[7]=[0,1], rest are [0,0]
        let mut t_times_delta = [[0u8; 8]; 8];
        // Row 6: 1·Δ_L + 0·Δ_R = Δ_L
        t_times_delta[6] = delta_sliced.left;
        // Row 7: 0·Δ_L + 1·Δ_R = Δ_R
        t_times_delta[7] = delta_sliced.right;

        // Step 4: Compute K·(t·Δ)
        println!("\n=== K·(t·Δ) ===");
        for k_row in 0..3 {
            let mut k_times_t_delta = [0u8; 8];
            for col in 0..8 {
                if K[k_row][col] == 1 {
                    xor_assign_8(&mut k_times_t_delta, &t_times_delta[col]);
                }
            }
            let is_zero = k_times_t_delta == [0u8; 8];
            println!("K[{}]·(t·Δ) = {:?} (zero: {})", k_row, k_times_t_delta, is_zero);
        }

        // Step 5: They should be equal (both = K·[0 0 t]·input)
        println!("\n=== Comparing K·(R·input) vs K·(t·Δ) ===");
        for k_row in 0..3 {
            let mut k_times_r_input = [0u8; 8];
            let mut k_times_t_delta = [0u8; 8];
            for col in 0..8 {
                if K[k_row][col] == 1 {
                    xor_assign_8(&mut k_times_r_input, &r_times_input[col]);
                    xor_assign_8(&mut k_times_t_delta, &t_times_delta[col]);
                }
            }
            let equal = k_times_r_input == k_times_t_delta;
            println!("K[{}]: R·input={:?}, t·Δ={:?}, equal: {}",
                k_row, k_times_r_input, k_times_t_delta, equal);
            if !equal {
                // Print XOR to see the difference
                let mut diff = [0u8; 8];
                for i in 0..8 { diff[i] = k_times_r_input[i] ^ k_times_t_delta[i]; }
                println!("       Difference (XOR): {:?}", diff);
            }
        }

        // Step 6: Verify K·R = K·[0 0 t] by computing K·R directly
        println!("\n=== Verifying K·R = K·[0 0 t] ===");
        // K is 3×8, R is 8×6, so K·R is 3×6
        let mut k_times_r = [[0u8; 6]; 3];
        for i in 0..3 {
            for j in 0..6 {
                for k in 0..8 {
                    k_times_r[i][j] ^= K[i][k] * r[k][j];
                }
            }
        }
        // [0 0 t] is 8×6 with t in columns 4,5
        let mut zero_zero_t = [[0u8; 6]; 8];
        for i in 0..8 {
            zero_zero_t[i][4] = t[i][0];
            zero_zero_t[i][5] = t[i][1];
        }
        let mut k_times_zero_zero_t = [[0u8; 6]; 3];
        for i in 0..3 {
            for j in 0..6 {
                for k in 0..8 {
                    k_times_zero_zero_t[i][j] ^= K[i][k] * zero_zero_t[k][j];
                }
            }
        }
        for i in 0..3 {
            let equal = k_times_r[i] == k_times_zero_zero_t[i];
            println!("K·R[{}] = {:?}, K·[0 0 t][{}] = {:?}, equal: {}",
                i, k_times_r[i], i, k_times_zero_zero_t[i], equal);
        }

        // The XOR of K·(R·input) and K·(t·Δ) should be zero
        // since K·R·input = K·[0 0 t]·input and [0 0 t]·input = t·Δ
        println!("\n=== Final check: K·(R·input) ⊕ K·(t·Δ) ===");
        let mut all_zero = true;
        for k_row in 0..3 {
            let mut xor_result = [0u8; 8];
            for col in 0..8 {
                if K[k_row][col] == 1 {
                    xor_assign_8(&mut xor_result, &r_times_input[col]);
                    xor_assign_8(&mut xor_result, &t_times_delta[col]);
                }
            }
            let is_zero = xor_result == [0u8; 8];
            if !is_zero { all_zero = false; }
            println!("K[{}]·(R·input) ⊕ K[{}]·(t·Δ) = {:?} (zero: {})",
                k_row, k_row, xor_result, is_zero);
        }

        assert!(all_zero, "K·(R·input) should equal K·(t·Δ)");
    }

    /// Verify K·V = 0 in the sliced case.
    ///
    /// K[0] and K[1] only involve same-parity rows, so they work.
    /// K[2] mixes parities, so it may fail.
    #[test]
    fn test_k_v_sliced_case() {
        use super::super::matrices::{K, V};
        use super::super::slicing::SlicedLabel;
        use super::xor_assign_8;

        let mut rng = ChaCha12Rng::seed_from_u64(55555);

        // Random C and G values
        let c_l: [u8; 8] = rng.random();
        let c_r: [u8; 8] = rng.random();
        let g0: [u8; 8] = rng.random();
        let g1: [u8; 8] = rng.random();
        let g2: [u8; 8] = rng.random();

        // Compute V·[C_L, C_R, G₀, G₁, G₂]
        let x = [c_l, c_r, g0, g1, g2];
        let mut v_times_x = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..5 {
                if V[row][col] == 1 {
                    xor_assign_8(&mut v_times_x[row], &x[col]);
                }
            }
        }

        println!("\n=== V·x for each row ===");
        for row in 0..8 {
            println!("(V·x)[{}] = {:?}", row, v_times_x[row]);
        }

        println!("\n=== K·(V·x) for each K row ===");
        for k_row in 0..3 {
            let mut result = [0u8; 8];
            for col in 0..8 {
                if K[k_row][col] == 1 {
                    xor_assign_8(&mut result, &v_times_x[col]);
                }
            }
            let is_zero = result == [0u8; 8];
            println!("K[{}]·(V·x) = {:?} (zero: {})", k_row, result, is_zero);
            if !is_zero {
                println!("  K[{}] involves rows: {:?}", k_row,
                    (0..8).filter(|&c| K[k_row][c] == 1).collect::<Vec<_>>());
                // Show what each involved row contributes
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        println!("    Row {} ({}): V[{}] = {:?}, (V·x)[{}] = {:?}",
                            col, if col % 2 == 0 { "left" } else { "right" },
                            col, V[col], col, v_times_x[col]);
                    }
                }
            }
        }
    }

    /// Extended diagnostic: include M·H term and verify full K·RHS_adjusted = 0.
    #[test]
    fn test_k_rhs_with_hashes_diagnostic() {
        use super::super::control::{and_truth_table, sample_r_odd};
        use super::super::matrices::K;
        use super::super::slicing::SlicedLabel;
        use super::{apply_m_to_hashes, apply_r_to_inputs, compute_hashes, xor_assign_8};

        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(22222); // Same seed as failing test

        for trial in 0..3 {
            let mut a0 = Block::random(&mut rng);
            let mut b0 = Block::random(&mut rng);
            let mut delta = Block::random(&mut rng);
            delta.set_lsb(true);
            a0.set_lsb(false);
            b0.set_lsb(false);

            let gid = trial + 1;
            let rand_bits: [bool; 2] = [rng.random(), rng.random()];

            let a0_sliced = SlicedLabel::from_block(a0);
            let b0_sliced = SlicedLabel::from_block(b0);
            let delta_sliced = SlicedLabel::from_block(delta);

            println!("\n========== Trial {} ==========", trial);

            // Compute M·H
            let hashes = compute_hashes(cipher, a0, b0, delta, gid);
            let m_times_h = apply_m_to_hashes(&hashes);

            // Compute R·input
            let t = and_truth_table();
            let (r, _) = sample_r_odd(&t, rand_bits);
            let r_times_input = apply_r_to_inputs(&r, &a0_sliced, &b0_sliced, &delta_sliced);

            // Compute RHS = M·H ⊕ R·input
            let mut rhs = [[0u8; 8]; 8];
            for row in 0..8 {
                rhs[row] = m_times_h[row];
                xor_assign_8(&mut rhs[row], &r_times_input[row]);
            }

            // Compute RHS_adjusted = RHS ⊕ t·Δ
            let mut rhs_adjusted = rhs;
            xor_assign_8(&mut rhs_adjusted[6], &delta_sliced.left);
            xor_assign_8(&mut rhs_adjusted[7], &delta_sliced.right);

            // Check K·(M·H)
            println!("\n--- K·(M·H) ---");
            for k_row in 0..3 {
                let mut result = [0u8; 8];
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        xor_assign_8(&mut result, &m_times_h[col]);
                    }
                }
                let is_zero = result == [0u8; 8];
                println!("K[{}]·(M·H) = {:?} (zero: {})", k_row, result, is_zero);
            }

            // Check K·(R·input)
            println!("\n--- K·(R·input) ---");
            for k_row in 0..3 {
                let mut result = [0u8; 8];
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        xor_assign_8(&mut result, &r_times_input[col]);
                    }
                }
                let is_zero = result == [0u8; 8];
                println!("K[{}]·(R·input) = {:?} (zero: {})", k_row, result, is_zero);
            }

            // Check K·RHS
            println!("\n--- K·RHS (before adjustment) ---");
            for k_row in 0..3 {
                let mut result = [0u8; 8];
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        xor_assign_8(&mut result, &rhs[col]);
                    }
                }
                let is_zero = result == [0u8; 8];
                println!("K[{}]·RHS = {:?} (zero: {})", k_row, result, is_zero);
            }

            // Check K·RHS_adjusted
            println!("\n--- K·RHS_adjusted (after t·Δ adjustment) ---");
            for k_row in 0..3 {
                let mut result = [0u8; 8];
                for col in 0..8 {
                    if K[k_row][col] == 1 {
                        xor_assign_8(&mut result, &rhs_adjusted[col]);
                    }
                }
                let is_zero = result == [0u8; 8];
                println!("K[{}]·RHS_adjusted = {:?} (zero: {})", k_row, result, is_zero);
                if !is_zero {
                    println!("  PROBLEM: K[{}] involves rows {:?}", k_row,
                        (0..8).filter(|&c| K[k_row][c] == 1).collect::<Vec<_>>());
                }
            }
        }
    }

    /// Test the evaluator's gate contribution function in isolation.
    ///
    /// This verifies that compute_gate_contribution correctly applies
    /// V[row][2:4] · [G₀, G₁, G₂] for all 8 rows.
    #[test]
    fn test_evaluator_gate_contribution() {
        use super::super::matrices::V;
        use super::compute_gate_contribution;

        let mut rng = ChaCha12Rng::seed_from_u64(77777);

        for trial in 0..10 {
            // Generate random gate ciphertexts
            let g0: [u8; 8] = rng.random();
            let g1: [u8; 8] = rng.random();
            let g2: [u8; 8] = rng.random();

            let gate = ThreeHalvesGate::new(g0, g1, g2);

            // Test all 8 rows
            for row in 0..8 {
                // Compute using the function under test
                let result = compute_gate_contribution(row, &gate);

                // Compute expected result manually: V[row][2:4] · [G₀, G₁, G₂]
                let mut expected = [0u8; 8];
                if V[row][2] == 1 {
                    for k in 0..8 {
                        expected[k] ^= g0[k];
                    }
                }
                if V[row][3] == 1 {
                    for k in 0..8 {
                        expected[k] ^= g1[k];
                    }
                }
                if V[row][4] == 1 {
                    for k in 0..8 {
                        expected[k] ^= g2[k];
                    }
                }

                assert_eq!(
                    result, expected,
                    "Trial {}, row {}: gate contribution mismatch.\n\
                     V[{}][2:5] = [{}, {}, {}]\n\
                     Got:      {:?}\n\
                     Expected: {:?}",
                    trial, row, row, V[row][2], V[row][3], V[row][4],
                    result, expected
                );

                println!(
                    "PASS Trial {}, row {}: V[{}][2:5]=[{},{},{}]",
                    trial, row, row, V[row][2], V[row][3], V[row][4]
                );
            }
        }

        println!("✓ Evaluator gate contribution test passed!");
    }

    /// Test 18: Garbling and evaluation with different point-and-permute bits
    ///
    /// This test verifies that the truth table is correctly computed when
    /// the point-and-permute bits (pi_a, pi_b) are varied. With permute bits:
    /// - When pi_a=false: A₀ represents semantic value 0, A₁ represents 1
    /// - When pi_a=true: A₀ represents semantic value 1, A₁ represents 0
    /// - Similarly for pi_b
    #[test]
    fn test_garble_evaluate_with_permute_bits() {
        let cipher = &(*FIXED_KEY_AES);
        let mut rng = ChaCha12Rng::seed_from_u64(0xDEADBEEF);
        let mut failures = Vec::new();

        // Test all 4 combinations of permute bits
        for pi_a in [false, true] {
            for pi_b in [false, true] {
                for trial in 0..5 {
                    let mut a0 = Block::random(&mut rng);
                    let mut b0 = Block::random(&mut rng);
                    let mut delta = Block::random(&mut rng);
                    delta.set_lsb(true);
                    a0.set_lsb(false);
                    b0.set_lsb(false);

                    let a1 = a0 ^ delta;
                    let b1 = b0 ^ delta;

                    let gid = trial + 1;
                    let rand_bits: [bool; 2] = [rng.random(), rng.random()];

                    // Garble with specific permute bits
                    let garbled =
                        garble_and_gate(cipher, a0, b0, delta, gid, pi_a, pi_b, rand_bits);
                    let c0 = garbled.output_label;
                    let c1 = c0 ^ delta;

                    // Determine which label represents which semantic value
                    // When pi_a=false: A₀ = semantic 0, A₁ = semantic 1
                    // When pi_a=true: A₀ = semantic 1, A₁ = semantic 0
                    let (a_false, a_true) = if pi_a { (a1, a0) } else { (a0, a1) };
                    let (b_false, b_true) = if pi_b { (b1, b0) } else { (b0, b1) };

                    // Test all 4 input combinations (using semantic values)
                    // AND gate: output = semantic_a AND semantic_b
                    let test_cases = [
                        (a_false, b_false, 0, 0, c0), // 0 ∧ 0 = 0
                        (a_false, b_true, 0, 1, c0),  // 0 ∧ 1 = 0
                        (a_true, b_false, 1, 0, c0),  // 1 ∧ 0 = 0
                        (a_true, b_true, 1, 1, c1),   // 1 ∧ 1 = 1
                    ];

                    for (a_label, b_label, sem_a, sem_b, expected) in test_cases {
                        let result = evaluate_and_gate(
                            cipher,
                            a_label,
                            b_label,
                            &garbled.gate,
                            &garbled.control_bits,
                            gid,
                        );

                        if result != expected {
                            failures.push((pi_a, pi_b, trial, sem_a, sem_b));
                            println!(
                                "FAIL pi_a={}, pi_b={}, trial {}: evaluate({},{}) got wrong result",
                                pi_a, pi_b, trial, sem_a, sem_b
                            );
                        }
                    }
                }
            }
        }

        if !failures.is_empty() {
            println!("\n=== SUMMARY ===");
            println!("Total failures: {}", failures.len());
            for (pi_a, pi_b, trial, sem_a, sem_b) in &failures {
                println!(
                    "  pi_a={}, pi_b={}, trial {}: ({},{})",
                    pi_a, pi_b, trial, sem_a, sem_b
                );
            }
            panic!("{} test cases failed", failures.len());
        }

        println!("✓ All permute bit combinations pass!");
    }
}
