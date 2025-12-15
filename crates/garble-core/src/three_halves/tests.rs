//! Integration tests for the three-halves garbling scheme
//!
//! These tests verify the complete system works together correctly.

use super::control::{
    and_truth_table_with_permute, expand_marginal, extract_marginal, extract_truth_table_bits,
    sample_r_odd_with_r, verify_k_r_a, verify_k_r_b, verify_k_r_dollar_is_zero, verify_k_r_p, R_P,
};
use super::matrices::{
    compute_v_inv_m, matmul_gf2, verify_km_is_zero, verify_kv_is_zero,
    verify_ranks, verify_v_inv_is_left_inverse, K, M, V,
};

/// Helper: convert bool matrix to u8 for GF(2) matrix operations
fn bool_to_u8_matrix<const ROWS: usize, const COLS: usize>(
    m: &[[bool; COLS]; ROWS],
) -> [[u8; COLS]; ROWS] {
    let mut result = [[0u8; COLS]; ROWS];
    for i in 0..ROWS {
        for j in 0..COLS {
            result[i][j] = m[i][j] as u8;
        }
    }
    result
}

/// Master test: Run all matrix verification checks
#[test]
fn test_all_matrix_properties() {
    // From matrices.rs
    assert!(verify_kv_is_zero(), "K × V should be zero");
    assert!(verify_km_is_zero(), "K × M should be zero");
    assert!(verify_v_inv_is_left_inverse(), "V⁻¹ × V should be identity");

    let (rank_v, rank_m) = verify_ranks();
    assert_eq!(rank_v, 5, "V should have rank 5");
    assert_eq!(rank_m, 5, "M should have rank 5");

    // From control.rs
    assert!(verify_k_r_dollar_is_zero(), "K × R$ should be zero");
    assert!(verify_k_r_p(), "K × R_p should match Equation 7");
    assert!(verify_k_r_a(), "K × R_a should match Equation 7");
    assert!(verify_k_r_b(), "K × R_b should match Equation 7");

    println!("✓ All matrix properties verified!");
}

/// Test the complete constraint: K × R = K × [0 0 t] for AND gate
///
/// This is THE key property that makes the scheme work.
/// Paper Equation 5 and surrounding discussion.
#[test]
fn test_complete_constraint_and_gate() {
    // Test with default permute bits (false, false)
    let (pi_a, pi_b) = (false, false);
    let t = and_truth_table_with_permute(pi_a, pi_b);
    let (a, b, p) = extract_truth_table_bits(pi_a, pi_b);

    println!("AND gate: a={}, b={}, p={}", a, b, p);
    assert_eq!((a, b, p), (true, true, true), "AND gate with pi_a=false, pi_b=false should have a=true, b=true");

    // Test all random combinations
    for r0 in [false, true] {
        for r1 in [false, true] {
            let (r_bool, r_bar) = sample_r_odd_with_r(pi_a, pi_b, [r0, r1]);
            let r = bool_to_u8_matrix(&r_bool);

            // Verify K × R = K × [0 0 t]
            let kr = matmul_gf2(&K, &r);

            // Build [0 0 t] matrix
            let mut zero_zero_t = [[0u8; 6]; 8];
            for i in 0..8 {
                zero_zero_t[i][4] = t[i][0];
                zero_zero_t[i][5] = t[i][1];
            }
            let k_zero_zero_t = matmul_gf2(&K, &zero_zero_t);

            assert_eq!(kr, k_zero_zero_t, "Constraint violated for r0={}, r1={}", r0, r1);

            // Verify compressed form matches
            for ij in 0..4 {
                let i = ij >> 1;
                let j = ij & 1;

                // Extract from R (without R_p for comparison)
                let mut r_no_p = r;
                for row in 0..8 {
                    for col in 0..6 {
                        r_no_p[row][col] ^= R_P[row][col];
                    }
                }
                let marginal = extract_marginal(&r_no_p, i, j);
                let expanded = expand_marginal(&r_bar[ij]);

                assert_eq!(
                    marginal, expanded,
                    "Compressed form mismatch at ({},{}) for r0={}, r1={}",
                    i, j, r0, r1
                );
            }
        }
    }

    println!("✓ AND gate constraint verified for all random choices!");
}

/// Print matrices for visual inspection against paper
#[test]
fn print_matrices_for_verification() {
    println!("\n=== Matrix K (3×8) - Paper Page 12 ===");
    println!("Expected:");
    println!("  [1 0 | 1 0 | 1 0 | 1 0]");
    println!("  [0 1 | 0 1 | 0 1 | 0 1]");
    println!("  [0 0 | 0 1 | 1 0 | 1 1]");
    println!("Actual:");
    for row in &K {
        print!("  [");
        for (i, &val) in row.iter().enumerate() {
            if i > 0 && i % 2 == 0 {
                print!(" |");
            }
            print!(" {}", val);
        }
        println!(" ]");
    }

    println!("\n=== Matrix V (8×5) - Paper Page 12 ===");
    println!("Columns: C_L C_R G₀ G₁ G₂");
    for (i, row) in V.iter().enumerate() {
        let input = match i / 2 {
            0 => "(0,0)",
            1 => "(0,1)",
            2 => "(1,0)",
            3 => "(1,1)",
            _ => "???",
        };
        let half = if i % 2 == 0 { "L" } else { "R" };
        println!("  {} {}: {:?}", input, half, row);
    }

    println!("\n=== Matrix M (8×6) - Paper Page 10 ===");
    println!("Columns: H(A₀) H(A₁) H(B₀) H(B₁) H(A₀⊕B₀) H(A₀⊕B₁)");
    for (i, row) in M.iter().enumerate() {
        let input = match i / 2 {
            0 => "(0,0)",
            1 => "(0,1)",
            2 => "(1,0)",
            3 => "(1,1)",
            _ => "???",
        };
        let half = if i % 2 == 0 { "L" } else { "R" };
        println!("  {} {}: {:?}", input, half, row);
    }

    println!("\n=== V⁻¹ × M (5×6) - Paper Equation 12 ===");
    let v_inv_m = compute_v_inv_m();
    println!("Expected:");
    println!("  [1 0 0 0 1 0]");
    println!("  [0 0 1 0 1 0]");
    println!("  [1 1 0 0 0 0]");
    println!("  [0 0 1 1 0 0]");
    println!("  [0 0 0 0 1 1]");
    println!("Actual:");
    for row in &v_inv_m {
        println!("  {:?}", row);
    }
}
