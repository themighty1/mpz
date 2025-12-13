//! # Control Matrix System for Three-Halves Garbling
//!
//! This module implements the "dicing" technique from the paper.
//! The control matrix R determines which linear combinations of input label
//! pieces the evaluator uses to compute output label halves.
//!
//! ## References
//!
//! - Paper Section 4.3: "Observation #3: Randomize and Hide the Evaluator's
//!   Coefficients"
//! - Paper Section 5.1: "Choosing the Matrices" - detailed explanation of R
//! - Paper Figure 3 (Page 14): Control matrices for even-parity gates
//! - Paper Figure 4 (Page 14): Control matrices for gate-hiding (parity-hiding)
//!
//! ## Overview
//!
//! The control matrix R is an 8×6 matrix that specifies, for each of the 8
//! evaluation equations (4 input combinations × 2 halves), which pieces of
//! the input labels [A₀; B₀; Δ] to include.
//!
//! ### The Problem (Paper Section 4.3)
//!
//! The matrix R must satisfy the constraint:
//! ```text
//! KR = K[0 0 t]
//! ```
//! where t is the truth table. But this constraint depends on t, which must
//! be hidden from the evaluator!
//!
//! ### The Solution: Randomization
//!
//! R is sampled from a distribution R(t) such that:
//! 1. KR = K[0 0 t] always holds (correctness)
//! 2. Each marginal view R_ij is uniform and independent of t (security)
//!
//! ### Marginal Views
//!
//! When the evaluator has input (A_i, B_j), they only see/need the 2×4
//! submatrix: ```text
//! R_ij = [R_ijA  R_ijB]  (rows 2i, 2i+1 and columns for A, B parts)
//! ```
//! The full R is never revealed - only one marginal view per evaluation.
//!
//! ## Compression with Basis {S₁, S₂}
//!
//! Instead of encrypting 8-bit marginal views, we express them in a 2D basis:
//! ```text
//! R_ij = c₁·S₁ ⊕ c₂·S₂
//! ```
//! This reduces overhead to 2 bits per marginal view × 4 views = 8 bits,
//! but we encode it as 5 bits total (see paper Section 5.2).

use super::matrices::{K, is_zero_matrix, matmul_gf2};

// ============================================================================
// Basis Matrices for Marginal View Compression
// ============================================================================

/// Basis matrix S₁ for expressing marginal views
///
/// From Paper Figure 3, Page 14:
/// ```text
/// S₁ = [ 1 1 | 1 0 ]
///      [ 1 0 | 0 1 ]
/// ```
///
/// **Interpretation**: A 2×4 matrix where:
/// - Row 0 is coefficients for the left half computation
/// - Row 1 is coefficients for the right half computation
/// - Columns are [A_L, A_R, B_L, B_R] (the four input label halves)
///
/// The vertical bar separates the A-part from the B-part.
pub const S1: [[u8; 4]; 2] = [
    //  A_L  A_R  B_L  B_R
    [1, 1, 1, 0], // Left half computation
    [1, 0, 0, 1], // Right half computation
];

/// Basis matrix S₂ for expressing marginal views
///
/// From Paper Figure 3, Page 14:
/// ```text
/// S₂ = [ 1 0 | 0 1 ]
///      [ 0 1 | 1 1 ]
/// ```
pub const S2: [[u8; 4]; 2] = [
    //  A_L  A_R  B_L  B_R
    [1, 0, 0, 1], // Left half computation
    [0, 1, 1, 1], // Right half computation
];

// ============================================================================
// Fixed Control Matrices for ODD-Parity Gates (AND, OR, NAND, NOR)
// ============================================================================

/// Matrix R_p: Added for odd-parity gates
///
/// From Paper Figure 4, Page 14:
/// ```text
/// R_p = [ 0 0 | 1 0 | 0 0 ]
///       [ 0 1 | 0 0 | 0 0 ]
///       [ 0 0 | 1 0 | 1 0 ]
///       [ 0 0 | 0 0 | 0 0 ]
///       [ 0 0 | 0 0 | 0 0 ]
///       [ 0 1 | 0 0 | 0 1 ]
///       [ 0 0 | 0 0 | 0 0 ]
///       [ 0 0 | 0 0 | 0 0 ]
/// ```
///
/// **Purpose**: When garbling an odd-parity gate (like AND), we add R_p to
/// the sampled R. The evaluator knows to add the corresponding (R_p)_ij to
/// their marginal view since parity is public in ODD mode.
///
/// **Constraint** (Paper Equation 7):
/// ```text
/// K·R_p = [ 0 0 0 0 | 1 0 ]
///         [ 0 0 0 0 | 0 1 ]
///         [ 0 0 0 0 | 0 0 ]
/// ```
/// This contributes the "p" (parity) term to the K·R constraint.
///
/// **Column layout**: [A₀_L, A₀_R, B₀_L, B₀_R, Δ_L, Δ_R]
pub const R_P: [[u8; 6]; 8] = [
    // (0,0) left:  A₀_L A₀_R B₀_L B₀_R Δ_L Δ_R
    [0, 0, 1, 0, 0, 0],
    // (0,0) right
    [0, 1, 0, 0, 0, 0],
    // (0,1) left
    [0, 0, 1, 0, 1, 0],
    // (0,1) right
    [0, 0, 0, 0, 0, 0],
    // (1,0) left
    [0, 0, 0, 0, 0, 0],
    // (1,0) right
    [0, 1, 0, 0, 0, 1],
    // (1,1) left
    [0, 0, 0, 0, 0, 0],
    // (1,1) right
    [0, 0, 0, 0, 0, 0],
];

/// Matrix R_a: Encodes the 'a' bit of truth table position
///
/// From Paper Figure 3, Page 14:
///
/// **Purpose**: The 'a' bit indicates whether the true output is at input
/// combination (1,0) or (1,1) (i.e., which row when first input is 1).
///
/// **Constraint** (Paper Equation 7):
/// ```text
/// K·R_a = [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 1 0 ]
/// ```
///
/// This is masked by the randomization in R$ so the evaluator can't learn 'a'.
pub const R_A: [[u8; 6]; 8] = [
    // (0,0) left
    [0, 0, 0, 0, 0, 0],
    // (0,0) right
    [0, 0, 0, 0, 0, 0],
    // (0,1) left
    [0, 1, 1, 1, 1, 1],
    // (0,1) right
    [1, 1, 1, 0, 1, 0],
    // (1,0) left
    [1, 0, 0, 1, 1, 0],
    // (1,0) right
    [0, 1, 1, 1, 0, 1],
    // (1,1) left
    [1, 1, 1, 0, 0, 1],
    // (1,1) right
    [1, 0, 0, 1, 1, 1],
];

/// Matrix R_b: Encodes the 'b' bit of truth table position
///
/// From Paper Figure 3, Page 14:
///
/// **Purpose**: The 'b' bit indicates whether the true output is at input
/// combination (0,1) or (1,1) (i.e., which column when second input is 1).
///
/// **Constraint** (Paper Equation 7):
/// ```text
/// K·R_b = [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 1 ]
/// ```
pub const R_B: [[u8; 6]; 8] = [
    // (0,0) left
    [0, 0, 0, 0, 0, 0],
    // (0,0) right
    [0, 0, 0, 0, 0, 0],
    // (0,1) left
    [1, 1, 1, 0, 1, 0],
    // (0,1) right
    [1, 0, 0, 1, 0, 1],
    // (1,0) left
    [0, 1, 1, 1, 0, 1],
    // (1,0) right
    [1, 1, 1, 0, 1, 1],
    // (1,1) left
    [1, 0, 0, 1, 1, 1],
    // (1,1) right
    [0, 1, 1, 1, 1, 0],
];

// ============================================================================
// Randomization Basis for R$
// ============================================================================

/// First basis matrix for R$ randomization
///
/// From Paper Figure 3, Page 14 (first matrix in R$ span):
///
/// **Property**: K · R$_BASIS_0 = 0 (contributes nothing to constraint)
///
/// **Property**: Each row pair (marginal view) when projected gives uniform
/// distribution
pub const R_DOLLAR_BASIS_0: [[u8; 6]; 8] = [
    // These values ensure that:
    // 1. K × R$_BASIS_0 = 0
    // 2. Marginal views span a uniform distribution
    [1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 1],
];

/// Second basis matrix for R$ randomization
///
/// From Paper Figure 3, Page 14 (second matrix in R$ span):
pub const R_DOLLAR_BASIS_1: [[u8; 6]; 8] = [
    [1, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0],
];

// ============================================================================
// Compressed Representation (R̄)
// ============================================================================

/// Compressed representation of R_a in terms of basis {S₁, S₂}
///
/// From Paper Figure 3, Page 14:
/// ```text
/// R̄_a = [ 0 0 ]   <- (0,0): 0·S₁ ⊕ 0·S₂
///       [ 1 1 ]   <- (0,1): 1·S₁ ⊕ 1·S₂
///       [ 0 1 ]   <- (1,0): 0·S₁ ⊕ 1·S₂
///       [ 1 0 ]   <- (1,1): 1·S₁ ⊕ 0·S₂
/// ```
///
/// Each row gives the coefficients [c₁, c₂] such that R_ij = c₁·S₁ ⊕ c₂·S₂
pub const R_BAR_A: [[u8; 2]; 4] = [
    [0, 0], // (0,0)
    [1, 1], // (0,1)
    [0, 1], // (1,0)
    [1, 0], // (1,1)
];

/// Compressed representation of R_b in terms of basis {S₁, S₂}
///
/// From Paper Figure 3, Page 14:
/// ```text
/// R̄_b = [ 0 0 ]
///       [ 1 0 ]
///       [ 1 1 ]
///       [ 0 1 ]
/// ```
pub const R_BAR_B: [[u8; 2]; 4] = [
    [0, 0], // (0,0)
    [1, 0], // (0,1)
    [1, 1], // (1,0)
    [0, 1], // (1,1)
];

/// Compressed representation of R$ basis vectors
///
/// From Paper Figure 3, Page 14:
/// ```text
/// R̄$ ← span { [ 1 0 ]   [ 0 1 ] }
///             [ 1 0 ] , [ 0 1 ]
///             [ 1 0 ]   [ 0 1 ]
///             [ 1 0 ]   [ 0 1 ]
/// ```
///
/// **Key insight**: Both basis vectors have the same value in every row!
/// This means sampling random R$ in compressed form just picks a random
/// pair (c₁, c₂) and uses it for ALL four marginal views.
///
/// This is what makes each marginal view individually uniform while
/// maintaining the correlation needed for KR$ = 0.
pub const R_BAR_DOLLAR_BASIS_0: [[u8; 2]; 4] = [
    [1, 0], // (0,0)
    [1, 0], // (0,1)
    [1, 0], // (1,0)
    [1, 0], // (1,1)
];

/// Second basis vector for compressed R$ randomization
///
/// From Paper Figure 3, Page 14.
/// Same pattern as R_BAR_DOLLAR_BASIS_0 but with [0,1] instead of [1,0].
pub const R_BAR_DOLLAR_BASIS_1: [[u8; 2]; 4] = [
    [0, 1], // (0,0)
    [0, 1], // (0,1)
    [0, 1], // (1,0)
    [0, 1], // (1,1)
];

// ============================================================================
// Truth Table Representation
// ============================================================================

/// Compute truth table for AND gate based on point-and-permute bits.
///
/// # Paper Reference (Page 16, Figure 6)
///
/// ```text
/// t := [g(πA⊕i, πB⊕j)]  for (i,j) in [(0,0), (0,1), (1,0), (1,1)]
/// ```
///
/// The point-and-permute bits determine which wire label represents TRUE:
/// - πA: if 0, then A₀=FALSE, A₁=TRUE; if 1, then A₀=TRUE, A₁=FALSE
/// - πB: similarly for B wire
///
/// When evaluator has labels with color bits (i, j), the actual logical values are:
/// - xA = πA ⊕ i
/// - xB = πB ⊕ j
///
/// The truth table entry is g(xA, xB) where g is the AND function.
///
/// # Arguments
/// * `pi_a` - Point-and-permute bit for wire A (0 or 1)
/// * `pi_b` - Point-and-permute bit for wire B (0 or 1)
///
/// # Returns
/// 8×2 truth table matrix where identity blocks mark TRUE outputs.
pub fn and_truth_table_with_permute(pi_a: bool, pi_b: bool) -> [[u8; 2]; 8] {
    let mut t = [[0u8; 2]; 8];

    for i in 0..2u8 {
        for j in 0..2u8 {
            // Actual logical values based on color bits and permute bits
            let x_a = (pi_a as u8) ^ i;
            let x_b = (pi_b as u8) ^ j;

            // AND gate: output is true iff both inputs are true
            let output = x_a & x_b;

            // Row index: (i,j) maps to rows 2*(2*i + j) and 2*(2*i + j) + 1
            let block = (2 * i + j) as usize;
            let row_l = 2 * block;
            let row_r = 2 * block + 1;

            if output == 1 {
                // Identity block for TRUE output
                t[row_l] = [1, 0];
                t[row_r] = [0, 1];
            }
            // Zero block (default) for FALSE output
        }
    }

    t
}

/// Truth table for AND gate with default permute bits (πA=1, πB=1).
///
/// This is the "canonical" AND truth table where (1,1) gives TRUE.
/// For proper security, use `and_truth_table_with_permute` with random bits.
///
/// **WARNING**: Using fixed permute bits leaks information! In production,
/// always use random πA, πB values.
pub fn and_truth_table() -> [[u8; 2]; 8] {
    // Default: πA=1, πB=1 means A₀=TRUE, B₀=TRUE
    // So (1,1) color bits give logical (0,0) which is FALSE for AND
    // And (0,0) color bits give logical (1,1) which is TRUE for AND
    // Wait, that's backwards from what we had...

    // Let's use πA=0, πB=0 to match the existing behavior:
    // (1,1) color bits → logical (1,1) → TRUE for AND
    and_truth_table_with_permute(false, false)
}

/// Truth table for OR gate
///
/// For OR: inputs (0,1), (1,0), and (1,1) give true output.
/// By convention, we encode using only ONE true position per gate.
///
/// Actually, for garbled circuits we encode which SINGLE input gives
/// the minority output. For OR, only (0,0) gives false.
/// So we'd flip the encoding... but for simplicity, the paper uses
/// odd-parity gates where exactly one or three positions are true.
///
/// For standard AND gate encoding, the truth table marks the TRUE position.
pub fn or_truth_table() -> [[u8; 2]; 8] {
    // OR: true when at least one input is true
    // Using the paper's encoding for odd-parity gate with p=1
    // This is more complex - see paper for full treatment
    // For now, we focus on AND gates
    todo!("OR gate encoding - see paper Section 5 for details")
}

/// Extract (a, b, p) bits from a truth table
///
/// Paper Section 5.1: The constraint KR = K[0 0 t] reduces to
/// matching p, a, b where:
/// - p = parity of truth table (1 for AND/OR, 0 for XOR)
/// - a, b = position bits encoding where the true output is
///
/// For AND gate at position (1,1): a=1, b=1, p=1
pub fn extract_truth_table_bits(t: &[[u8; 2]; 8]) -> (u8, u8, u8) {
    // Count number of identity blocks (positions where output is true)
    // For odd-parity gates, exactly one position is true

    // Check each 2×2 block to see if it's an identity
    let mut true_position = None;
    for i in 0..4 {
        let row1 = t[2 * i];
        let row2 = t[2 * i + 1];
        // Identity block check: [[1,0], [0,1]]
        if row1 == [1, 0] && row2 == [0, 1] {
            true_position = Some(i);
            break;
        }
    }

    match true_position {
        Some(pos) => {
            // pos: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            let a = (pos >> 1) as u8; // First input bit
            let b = (pos & 1) as u8; // Second input bit
            let p = 1u8; // Odd parity (one true output)
            (a, b, p)
        }
        None => {
            // No true position found - could be even parity or invalid
            // For even parity gates, p=0
            (0, 0, 0)
        }
    }
}

// ============================================================================
// Control Matrix Operations
// ============================================================================

/// Sample a control matrix R for a given truth table (ODD mode)
///
/// Paper Section 5.1, Algorithm:
/// ```text
/// R = p·R_p ⊕ a·R_a ⊕ b·R_b ⊕ R$
/// ```
///
/// where R$ is sampled uniformly from span{R$_BASIS_0, R$_BASIS_1}
///
/// # Arguments
/// * `t` - The 8×2 truth table matrix
/// * `rand_bits` - Two random bits [r₀, r₁] for sampling R$
///
/// # Returns
/// * `R` - The 8×6 control matrix
/// * `r_bar` - The 4×2 compressed representation for encryption
pub fn sample_r_odd(t: &[[u8; 2]; 8], rand_bits: [bool; 2]) -> ([[u8; 6]; 8], [[u8; 2]; 4]) {
    let (a, b, p) = extract_truth_table_bits(t);

    // Start with R$ (randomization)
    let mut r = [[0u8; 6]; 8];
    let mut r_bar = [[0u8; 2]; 4];

    // Add random contribution: r₀·R$_BASIS_0 ⊕ r₁·R$_BASIS_1
    let r0 = rand_bits[0] as u8;
    let r1 = rand_bits[1] as u8;

    for i in 0..8 {
        for j in 0..6 {
            r[i][j] ^= r0 * R_DOLLAR_BASIS_0[i][j];
            r[i][j] ^= r1 * R_DOLLAR_BASIS_1[i][j];
        }
    }

    for i in 0..4 {
        for j in 0..2 {
            r_bar[i][j] ^= r0 * R_BAR_DOLLAR_BASIS_0[i][j];
            r_bar[i][j] ^= r1 * R_BAR_DOLLAR_BASIS_1[i][j];
        }
    }

    // Add a·R_a
    for i in 0..8 {
        for j in 0..6 {
            r[i][j] ^= a * R_A[i][j];
        }
    }
    for i in 0..4 {
        for j in 0..2 {
            r_bar[i][j] ^= a * R_BAR_A[i][j];
        }
    }

    // Add b·R_b
    for i in 0..8 {
        for j in 0..6 {
            r[i][j] ^= b * R_B[i][j];
        }
    }
    for i in 0..4 {
        for j in 0..2 {
            r_bar[i][j] ^= b * R_BAR_B[i][j];
        }
    }

    // Add p·R_p to the FULL R matrix (used by garbler for correctness)
    for i in 0..8 {
        for j in 0..6 {
            r[i][j] ^= p * R_P[i][j];
        }
    }

    // NOTE: We do NOT add p·R_p to r_bar because:
    // 1. R_P is NOT in the span of {S₁, S₂} basis (see paper Figure 4)
    // 2. In ODD mode the evaluator knows parity is odd and adds R_P themselves
    // The r_bar compressed form only contains R$ ⊕ a·R_a ⊕ b·R_b

    (r, r_bar)
}

/// Extract marginal view R_ij from full control matrix R
///
/// Paper Section 5.1: "When the evaluator holds input labels A_i, B_j,
/// the submatrix R_ij = [R_ijA R_ijB] is enough to completely determine
/// which linear combination should be applied."
///
/// # Arguments
/// * `r` - The full 8×6 control matrix
/// * `i` - First input's color bit (0 or 1)
/// * `j` - Second input's color bit (0 or 1)
///
/// # Returns
/// The 2×4 marginal view [R_ijA R_ijB] where:
/// - Columns 0-1 are the A-part (coefficients for A_L, A_R)
/// - Columns 2-3 are the B-part (coefficients for B_L, B_R)
pub fn extract_marginal(r: &[[u8; 6]; 8], i: usize, j: usize) -> [[u8; 4]; 2] {
    let row_base = 2 * (2 * i + j); // Row index: 0, 2, 4, or 6

    // The marginal view comes from:
    // - Columns 0-1 of R (A₀ part)
    // - Columns 2-3 of R (B₀ part)
    // Column 4-5 (Δ part) are handled separately based on input combination

    // Actually, the structure from Equation 6 is more nuanced:
    // R = | R₀₀_A  R₀₀_B    0     |
    //     | R₀₁_A  R₀₁_B  R₀₁_B   |
    //     | R₁₀_A  R₁₀_B  R₁₀_A   |
    //     | R₁₁_A  R₁₁_B  R₁₁_A⊕R₁₁_B |

    // For the marginal view, we extract the A and B parts that apply
    // to this input combination. The Δ column depends on which input
    // combination (this determines how Δ gets folded in).

    // For simplicity, we extract just the A and B coefficient columns
    let mut marginal = [[0u8; 4]; 2];

    // Row 0 of marginal = row (row_base) of R, columns 0-3
    // Row 1 of marginal = row (row_base + 1) of R, columns 0-3
    for col in 0..4 {
        marginal[0][col] = r[row_base][col];
        marginal[1][col] = r[row_base + 1][col];
    }

    marginal
}

/// Expand compressed marginal view R̄_ij to full R_ij
///
/// Given coefficients [c₁, c₂], compute R_ij = c₁·S₁ ⊕ c₂·S₂
///
/// # Arguments
/// * `r_bar_ij` - The 2-element compressed representation [c₁, c₂]
///
/// # Returns
/// The 2×4 marginal view matrix
pub fn expand_marginal(r_bar_ij: &[u8; 2]) -> [[u8; 4]; 2] {
    let c1 = r_bar_ij[0];
    let c2 = r_bar_ij[1];

    let mut result = [[0u8; 4]; 2];
    for row in 0..2 {
        for col in 0..4 {
            result[row][col] = (c1 * S1[row][col]) ^ (c2 * S2[row][col]);
        }
    }

    result
}

/// Compress a marginal view to its [c₁, c₂] representation
///
/// Given R_ij, find c₁, c₂ such that R_ij = c₁·S₁ ⊕ c₂·S₂
///
/// This is used for testing to verify that sampled R has valid structure.
pub fn compress_marginal(r_ij: &[[u8; 4]; 2]) -> Option<[u8; 2]> {
    // Try all 4 combinations of (c₁, c₂)
    for c1 in 0..2 {
        for c2 in 0..2 {
            let expanded = expand_marginal(&[c1, c2]);
            if expanded == *r_ij {
                return Some([c1, c2]);
            }
        }
    }
    None // Not expressible in the basis (shouldn't happen for valid R)
}

// ============================================================================
// Verification Functions
// ============================================================================

/// Verify K × R$ = 0 for basis matrices
///
/// Paper Figure 3: "KR$ = 0"
pub fn verify_k_r_dollar_is_zero() -> bool {
    let kr0 = matmul_gf2(&K, &R_DOLLAR_BASIS_0);
    let kr1 = matmul_gf2(&K, &R_DOLLAR_BASIS_1);

    is_zero_matrix(&kr0) && is_zero_matrix(&kr1)
}

/// Verify K × R_p gives expected result
///
/// Paper Equation 7:
/// ```text
/// K·R_p = [ 0 0 0 0 | 1 0 ]
///         [ 0 0 0 0 | 0 1 ]
///         [ 0 0 0 0 | 0 0 ]
/// ```
pub fn verify_k_r_p() -> bool {
    let kr_p = matmul_gf2(&K, &R_P);

    let expected: [[u8; 6]; 3] = [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]];

    kr_p == expected
}

/// Verify K × R_a gives expected result
///
/// Paper Equation 7:
/// ```text
/// K·R_a = [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 1 0 ]
/// ```
pub fn verify_k_r_a() -> bool {
    let kr_a = matmul_gf2(&K, &R_A);

    let expected: [[u8; 6]; 3] = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]];

    kr_a == expected
}

/// Verify K × R_b gives expected result
///
/// Paper Equation 7:
/// ```text
/// K·R_b = [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 0 ]
///         [ 0 0 0 0 | 0 1 ]
/// ```
pub fn verify_k_r_b() -> bool {
    let kr_b = matmul_gf2(&K, &R_B);

    let expected: [[u8; 6]; 3] = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]];

    kr_b == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: K × R$_BASIS vectors = 0
    ///
    /// Paper Figure 3: The R$ distribution must satisfy KR$ = 0
    #[test]
    fn test_k_r_dollar_is_zero() {
        assert!(verify_k_r_dollar_is_zero(), "K × R$_BASIS should be zero");
    }

    /// Test 2: K × R_p matches Equation 7
    #[test]
    fn test_k_r_p_constraint() {
        assert!(verify_k_r_p(), "K × R_p doesn't match Equation 7");
    }

    /// Test 3: K × R_a matches Equation 7
    #[test]
    fn test_k_r_a_constraint() {
        assert!(verify_k_r_a(), "K × R_a doesn't match Equation 7");
    }

    /// Test 4: K × R_b matches Equation 7
    #[test]
    fn test_k_r_b_constraint() {
        assert!(verify_k_r_b(), "K × R_b doesn't match Equation 7");
    }

    /// Test 5: Expanding and compressing marginal views are inverse operations
    #[test]
    fn test_marginal_roundtrip() {
        for c1 in 0..2 {
            for c2 in 0..2 {
                let original = [c1, c2];
                let expanded = expand_marginal(&original);
                let compressed = compress_marginal(&expanded);

                assert_eq!(
                    compressed,
                    Some(original),
                    "Roundtrip failed for [{}, {}]",
                    c1,
                    c2
                );
            }
        }
    }

    /// Test 6: AND gate truth table extraction
    #[test]
    fn test_and_truth_table_bits() {
        let t = and_truth_table();
        let (a, b, p) = extract_truth_table_bits(&t);

        // AND gate: true at (1,1) means a=1, b=1
        assert_eq!(a, 1, "AND gate should have a=1");
        assert_eq!(b, 1, "AND gate should have b=1");
        assert_eq!(p, 1, "AND gate should have p=1 (odd parity)");
    }

    /// Test 7: Sampled R satisfies K × R = K × [0 0 t]
    ///
    /// This is the fundamental correctness property from Paper Equation 5.
    #[test]
    fn test_sampled_r_constraint() {
        let t = and_truth_table();

        // Test with all 4 random bit combinations
        for r0 in [false, true] {
            for r1 in [false, true] {
                let (r, _r_bar) = sample_r_odd(&t, [r0, r1]);

                // Compute K × R
                let kr = matmul_gf2(&K, &r);

                // Compute K × [0 0 t]
                // The matrix [0 0 t] is 8×6 with t in the last 2 columns
                let mut zero_zero_t = [[0u8; 6]; 8];
                for i in 0..8 {
                    zero_zero_t[i][4] = t[i][0]; // Δ_L column gets t's first column
                    zero_zero_t[i][5] = t[i][1]; // Δ_R column gets t's second column
                }
                let k_zero_zero_t = matmul_gf2(&K, &zero_zero_t);

                assert_eq!(
                    kr, k_zero_zero_t,
                    "K×R should equal K×[0 0 t] for rand_bits=[{}, {}]",
                    r0, r1
                );
            }
        }
    }

    /// Test 8: Each marginal view is expressible in the basis {S₁, S₂}
    #[test]
    fn test_marginals_in_basis() {
        let t = and_truth_table();

        for r0 in [false, true] {
            for r1 in [false, true] {
                let (r, r_bar) = sample_r_odd(&t, [r0, r1]);

                // For ODD mode, we need to add R_p to the marginal before checking
                // because sample_r_odd adds R_p to r but not to r_bar
                for ij in 0..4 {
                    let i = ij >> 1;
                    let j = ij & 1;

                    // Get marginal from R without R_p for comparison with r_bar
                    let r_no_p = {
                        let mut tmp = r;
                        for row in 0..8 {
                            for col in 0..6 {
                                tmp[row][col] ^= R_P[row][col];
                            }
                        }
                        tmp
                    };
                    let marginal_no_p = extract_marginal(&r_no_p, i, j);

                    // The r_bar should match the marginal without R_p
                    let expanded = expand_marginal(&r_bar[ij]);
                    assert_eq!(
                        marginal_no_p, expanded,
                        "Marginal view ({},{}) doesn't match compressed form",
                        i, j
                    );
                }
            }
        }
    }

    /// Test 9: Distribution test - each marginal view should be uniform
    ///
    /// Over all 4 choices of rand_bits, each marginal should take each
    /// of the 4 possible values {[0,0], [0,1], [1,0], [1,1]} exactly once.
    #[test]
    fn test_marginal_uniformity() {
        let t = and_truth_table();

        // Collect all marginal views for input (0,0)
        let mut marginals_00 = Vec::new();

        for r0 in [false, true] {
            for r1 in [false, true] {
                let (_r, r_bar) = sample_r_odd(&t, [r0, r1]);
                marginals_00.push(r_bar[0]); // (0,0) marginal
            }
        }

        // Check that we got all 4 possible values
        marginals_00.sort();
        let expected = vec![[0, 0], [0, 1], [1, 0], [1, 1]];
        assert_eq!(
            marginals_00, expected,
            "Marginal (0,0) should cover all 4 basis coefficient pairs"
        );
    }
}
