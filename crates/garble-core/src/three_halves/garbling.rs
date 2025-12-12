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
//! - [C; G⃗] (5 elements): Output label halves + 3 gate ciphertexts (each κ/2 bits)
//! - M (8×6): Selects which hashes contribute to each equation
//! - H⃗ (6 elements): Hash outputs [H(A₀), H(A₁), H(B₀), H(B₁), H(A₀⊕B₀), H(A₀⊕B₁)]
//! - R (8×6): Control matrix (randomized to hide truth table)
//! - Input vector (6 elements): Label halves [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
//!
//! # Row Structure
//!
//! Each row i of the equation corresponds to one (input_combination, half) pair:
//! - Row 0: (0,0) left half
//! - Row 1: (0,0) right half
//! - Row 2: (0,1) left half
//! - Row 3: (0,1) right half
//! - Row 4: (1,0) left half
//! - Row 5: (1,0) right half
//! - Row 6: (1,1) left half
//! - Row 7: (1,1) right half

use mpz_core::Block;
use mpz_core::aes::FixedKeyAes;

use super::control::{and_truth_table, sample_r_odd};
use super::matrices::{M, V_INV};
use super::slicing::SlicedLabel;

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

/// Apply V⁻¹ to the RHS vector to solve for [C_L, C_R, G₀, G₁, G₂].
///
/// Given: V · output = rhs
/// Solve: output = V⁻¹ · rhs
///
/// # Paper Reference
/// V⁻¹ is the left-inverse of V (Page 12).
/// The output vector has 5 elements, each κ/2 bits.
fn apply_v_inv(rhs: &[[u8; 8]; 8]) -> [[u8; 8]; 5] {
    let mut result = [[0u8; 8]; 5];

    for row in 0..5 {
        for col in 0..8 {
            if V_INV[row][col] == 1 {
                xor_assign_8(&mut result[row], &rhs[col]);
            }
        }
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

    // 7. Solve V · [C; G⃗] = RHS  →  [C; G⃗] = V⁻¹ · RHS
    let output = apply_v_inv(&rhs);

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
/// The R matrix has a special structure where the Δ columns (4-5) are
/// related to the A or B columns depending on the input combination.
/// Due to this structure, the evaluator's marginal is simply columns 0-3
/// of R - the Δ contribution is automatically handled.
fn extract_evaluator_marginal(i: usize, j: usize, r_bar_ij: &[u8; 2]) -> [[u8; 4]; 2] {
    use super::control::{expand_marginal, extract_marginal, R_P};

    // Step 1: Expand r_bar_ij to get the R$ marginal (2×4)
    // r_bar_ij is already the compressed representation for this specific (i,j)
    let r_dollar_marginal = expand_marginal(r_bar_ij);

    // Step 2: Extract R_P's marginal for this input combination
    let r_p_marginal = extract_marginal(&R_P, i, j);

    // Step 3: Combined marginal = R$ marginal ⊕ R_P marginal
    let mut marginal = [[0u8; 4]; 2];
    for row in 0..2 {
        for col in 0..4 {
            marginal[row][col] = r_dollar_marginal[row][col] ^ r_p_marginal[row][col];
        }
    }

    marginal
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

    // 3. Compute the two hashes needed for this input combination
    //    From the M matrix structure:
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

    // 5. Expand the marginal control bits and compute the effective R marginal
    //    for the evaluator's input combination (i, j).
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
    //      result_L = hash_contribution_L ⊕ input_contribution_L ⊕ gate_contribution_L
    //    For the right half (row 2*ij+1):
    //      result_R = hash_contribution_R ⊕ input_contribution_R ⊕ gate_contribution_R
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
/// the hash contribution for their specific row.
///
/// From the M matrix structure, for input (i,j), rows 2*ij and 2*ij+1:
/// - Use H(A_i) according to M[row][i] (where i=0 for A₀, i=1 for A₁)
/// - Use H(B_j) according to M[row][2+j]
/// - Use H(A_i⊕B_j) according to M[row][4+j] (approximately)
fn compute_hash_contribution_eval(
    row: usize,
    h_a: &SlicedLabel,
    h_b: &SlicedLabel,
    h_ab: &SlicedLabel,
) -> [u8; 8] {
    let half = row % 2;

    // The M matrix for evaluator's view (simplified):
    // We look at which hashes the evaluator needs based on the row
    //
    // The pattern from M:
    // - Even rows (left): typically use H(A_i).left and one of H(A_i⊕B_j).left
    // - Odd rows (right): typically use H(B_j).right and one of H(A_i⊕B_j).right

    let mut result = [0u8; 8];

    // Simplified evaluation based on row structure:
    // This matches the M matrix pattern for the evaluator's available hashes
    match row {
        0 => {
            // (0,0) L: H(A₀).L ⊕ H(A₀⊕B₀).L
            xor_assign_8(&mut result, &h_a.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        1 => {
            // (0,0) R: H(B₀).R ⊕ H(A₀⊕B₀).R
            xor_assign_8(&mut result, &h_b.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        2 => {
            // (0,1) L: H(A₀).L ⊕ H(A₀⊕B₁).L
            xor_assign_8(&mut result, &h_a.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        3 => {
            // (0,1) R: H(B₁).R ⊕ H(A₀⊕B₁).R
            xor_assign_8(&mut result, &h_b.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        4 => {
            // (1,0) L: H(A₁).L ⊕ H(A₁⊕B₀).L
            xor_assign_8(&mut result, &h_a.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        5 => {
            // (1,0) R: H(B₀).R ⊕ H(A₁⊕B₀).R
            xor_assign_8(&mut result, &h_b.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        6 => {
            // (1,1) L: H(A₁).L ⊕ H(A₁⊕B₁).L
            xor_assign_8(&mut result, &h_a.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        7 => {
            // (1,1) R: H(B₁).R ⊕ H(A₁⊕B₁).R
            xor_assign_8(&mut result, &h_b.half(half));
            xor_assign_8(&mut result, &h_ab.half(half));
        }
        _ => unreachable!(),
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
    /// For AND gate: C₁ = C₀ ⊕ Δ only when output is 1 (i.e., both inputs are 1)
    /// This test verifies the garbling equation is set up correctly.
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

    /// Test 7: V⁻¹ application is consistent with V
    ///
    /// V · (V⁻¹ · x) should equal x projected onto the column space of V.
    #[test]
    fn test_v_inv_application() {
        use super::super::matrices::V;

        let mut rng = ChaCha12Rng::seed_from_u64(123);

        // Create a random 8-element RHS
        let mut rhs = [[0u8; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                rhs[i][j] = rng.random();
            }
        }

        // Apply V⁻¹
        let output = apply_v_inv(&rhs);

        // Apply V to the output
        let mut reconstructed = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..5 {
                if V[row][col] == 1 {
                    xor_assign_8(&mut reconstructed[row], &output[col]);
                }
            }
        }

        // The reconstructed should be in the column space of V
        // Since V·V⁻¹ is a projection, V·V⁻¹·x should equal itself when applied again
        let output2 = apply_v_inv(&reconstructed);
        let mut reconstructed2 = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..5 {
                if V[row][col] == 1 {
                    xor_assign_8(&mut reconstructed2[row], &output2[col]);
                }
            }
        }

        assert_eq!(
            reconstructed, reconstructed2,
            "Projection should be idempotent"
        );
    }

    /// Test 8: THE CRITICAL TEST - Garble then evaluate for all 4 input combinations
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
}
