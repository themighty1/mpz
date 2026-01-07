//! Cross-repetition soldering for Justvengers.
//!
//! Implements Section 4.3 "Soldering Repetitions" from the Justvengers paper.
//! This enables values from one repetition to flow into the next repetition
//! as inputs, essential for sequential computation like CPU emulation.
//!
//! # Overview
//!
//! The core constraint is: `IN₁(αⱼ) = O₁(αⱼ₋₁)` for j ∈ {2, ..., R}
//!
//! This means the input of repetition j equals the output of repetition j-1,
//! enabling chained computation across repetitions.
//!
//! # Technique: "Sacrifice" Method
//!
//! To prove this in ZK without revealing the polynomials:
//!
//! 1. P generates masking pair (r₁, r₂) satisfying: `r₁(αⱼ) = r₂(αⱼ₋₁)`
//! 2. V sends challenge φ
//! 3. P reveals masked versions: `f₁(·) = φ·IN(·) + r₁(·)` and `f₂(·) = φ·O(·) + r₂(·)`
//! 4. V verifies: `f₁(αⱼ) = f₂(αⱼ₋₁)` for all j ∈ {2, ..., R}

use rand::Rng;

#[cfg(feature = "ntt")]
use mpz_fields::goldilocks::{Goldilocks, GOLDILOCKS};

/// A constraint linking output of one repetition to input of the next.
///
/// Enforces: `input[target_input_idx] at rep j = output[source_output_idx] at rep j-1`
/// for all j ∈ {start_rep, ..., R}.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolderingConstraint {
    /// Which input wire receives the value (0-indexed into circuit inputs).
    pub target_input_idx: usize,
    /// Which multiplication output provides the value (0-indexed into mult outputs).
    pub source_output_idx: usize,
    /// Starting repetition (constraint applies from start_rep to R).
    /// Must be >= 2 since rep 1 has no predecessor.
    pub start_rep: usize,
}

impl SolderingConstraint {
    /// Creates a new soldering constraint.
    ///
    /// # Arguments
    /// * `target_input_idx` - Input wire that receives the value
    /// * `source_output_idx` - Mult output wire that provides the value
    ///
    /// The constraint starts from repetition 2 (connecting rep 1 output to rep 2 input).
    pub fn new(target_input_idx: usize, source_output_idx: usize) -> Self {
        Self {
            target_input_idx,
            source_output_idx,
            start_rep: 2,
        }
    }

    /// Creates a constraint starting from a specific repetition.
    pub fn from_rep(target_input_idx: usize, source_output_idx: usize, start_rep: usize) -> Self {
        assert!(start_rep >= 2, "Soldering must start from rep 2 or later");
        Self {
            target_input_idx,
            source_output_idx,
            start_rep,
        }
    }
}

/// Commitment message for soldering proof.
#[derive(Clone, Debug)]
pub struct SolderingCommitMessage {
    /// Number of soldering constraints.
    pub num_constraints: usize,
    /// Hash of masking polynomial commitments (simplified).
    /// In full implementation, these would be IT-PAC commitments.
    pub commitment_hashes: Vec<(u64, u64)>,
}

/// Challenge message from verifier for soldering verification.
#[derive(Clone, Debug)]
pub struct SolderingChallengeMessage {
    /// Challenge φ for combining original and masking polynomials.
    pub phi: u64,
}

/// Revealed masked polynomials for soldering verification.
#[derive(Clone, Debug)]
pub struct SolderingRevealMessage {
    /// For each constraint: (f₁ coefficients, f₂ coefficients).
    /// f₁(·) = φ·IN(·) + r₁(·)
    /// f₂(·) = φ·O(·) + r₂(·)
    pub masked_polys: Vec<(Vec<u64>, Vec<u64>)>,
}

/// Generates a masking polynomial pair (r₁, r₂) satisfying the shifted equality constraint.
///
/// The constraint: `r₁(αⱼ) = r₂(αⱼ₋₁)` for j ∈ {start_rep, ..., R}
///
/// # Arguments
/// * `eval_points` - The R evaluation points α₁, ..., αᵣ (roots of unity for Goldilocks)
/// * `start_rep` - First repetition where constraint applies (1-indexed)
/// * `modulus` - Field modulus
/// * `rng` - Random number generator
///
/// # Returns
/// Tuple (r₁_coeffs, r₂_coeffs) for polynomials of degree R-1.
///
/// When using Goldilocks field with NTT feature, uses O(n log n) inverse NTT
/// instead of O(n²) Lagrange interpolation.
pub fn generate_masking_pair<R: Rng>(
    eval_points: &[u64],
    start_rep: usize,
    modulus: u64,
    rng: &mut R,
) -> (Vec<u64>, Vec<u64>) {
    let r = eval_points.len();
    assert!(r >= 2, "Need at least 2 repetitions for soldering");
    assert!(start_rep >= 2 && start_rep <= r, "Invalid start_rep");

    // Number of constrained positions: from start_rep to R
    let num_constrained = r - start_rep + 1;

    // Generate random shared values for constrained positions
    let shared_values: Vec<u64> = (0..num_constrained)
        .map(|_| rng.random::<u64>() % modulus)
        .collect();

    // For r₁: constrained at positions start_rep, start_rep+1, ..., R
    let mut r1_values: Vec<u64> = (0..r).map(|_| rng.random::<u64>() % modulus).collect();
    for (idx, &shared) in shared_values.iter().enumerate() {
        let j = start_rep + idx;
        r1_values[j - 1] = shared;
    }

    // For r₂: constrained at positions start_rep-1, start_rep, ..., R-1
    let mut r2_values: Vec<u64> = (0..r).map(|_| rng.random::<u64>() % modulus).collect();
    for (idx, &shared) in shared_values.iter().enumerate() {
        let j = start_rep + idx;
        r2_values[j - 2] = shared;
    }

    // Use inverse NTT for Goldilocks, otherwise Lagrange interpolation
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS {
        // Inverse NTT: evaluations at roots of unity -> coefficients
        let mut r1_padded: Vec<Goldilocks> = r1_values
            .iter()
            .map(|&v| Goldilocks::new(v))
            .collect();
        Goldilocks::intt(&mut r1_padded);

        let mut r2_padded: Vec<Goldilocks> = r2_values
            .iter()
            .map(|&v| Goldilocks::new(v))
            .collect();
        Goldilocks::intt(&mut r2_padded);

        return (
            r1_padded.iter().map(|g| g.inner()).collect(),
            r2_padded.iter().map(|g| g.inner()).collect(),
        );
    }

    let r1_coeffs = interpolate(eval_points, &r1_values, modulus);
    let r2_coeffs = interpolate(eval_points, &r2_values, modulus);

    (r1_coeffs, r2_coeffs)
}

/// Computes the masked polynomial: f(·) = φ·g(·) + r(·)
///
/// # Arguments
/// * `phi` - Challenge value from verifier
/// * `g_coeffs` - Original polynomial coefficients
/// * `r_coeffs` - Masking polynomial coefficients
/// * `modulus` - Field modulus
///
/// # Returns
/// Coefficients of f(·) = φ·g(·) + r(·)
pub fn compute_masked_poly(
    phi: u64,
    g_coeffs: &[u64],
    r_coeffs: &[u64],
    modulus: u64,
) -> Vec<u64> {
    let max_len = g_coeffs.len().max(r_coeffs.len());
    let mut result = vec![0u64; max_len];

    for (i, &g) in g_coeffs.iter().enumerate() {
        let phi_g = ((phi as u128 * g as u128) % modulus as u128) as u64;
        result[i] = (result[i] + phi_g) % modulus;
    }

    for (i, &r) in r_coeffs.iter().enumerate() {
        result[i] = (result[i] + r) % modulus;
    }

    result
}

/// Verifies the soldering constraint on masked polynomials.
///
/// Checks: f₁(αⱼ) = f₂(αⱼ₋₁) for all j ∈ {start_rep, ..., R}
///
/// # Arguments
/// * `f1_coeffs` - Masked input polynomial coefficients
/// * `f2_coeffs` - Masked output polynomial coefficients
/// * `eval_points` - Evaluation points α₁, ..., αᵣ
/// * `start_rep` - First repetition where constraint applies (1-indexed)
/// * `modulus` - Field modulus
///
/// # Returns
/// True if all constraints are satisfied.
pub fn verify_soldering_constraint(
    f1_coeffs: &[u64],
    f2_coeffs: &[u64],
    eval_points: &[u64],
    start_rep: usize,
    modulus: u64,
) -> bool {
    let r = eval_points.len();

    for j in start_rep..=r {
        // f₁(αⱼ)
        let f1_at_j = evaluate_poly(f1_coeffs, eval_points[j - 1], modulus);
        // f₂(αⱼ₋₁)
        let f2_at_j_minus_1 = evaluate_poly(f2_coeffs, eval_points[j - 2], modulus);

        if f1_at_j != f2_at_j_minus_1 {
            return false;
        }
    }

    true
}

/// Validates that actual witness values satisfy soldering constraints.
///
/// This is used during prover setup to ensure the witness is consistent.
///
/// # Arguments
/// * `inputs_per_rep` - Input values for each repetition
/// * `outputs_per_rep` - Mult output values for each repetition
/// * `constraint` - The soldering constraint to check
///
/// # Returns
/// True if the witness satisfies the constraint.
pub fn validate_witness_soldering(
    inputs_per_rep: &[Vec<u64>],
    outputs_per_rep: &[Vec<u64>],
    constraint: &SolderingConstraint,
) -> bool {
    let r = inputs_per_rep.len();

    for j in constraint.start_rep..=r {
        // Input at rep j (1-indexed, so array index is j-1)
        let input_val = inputs_per_rep[j - 1]
            .get(constraint.target_input_idx)
            .copied()
            .unwrap_or(0);

        // Output at rep j-1
        let output_val = outputs_per_rep[j - 2]
            .get(constraint.source_output_idx)
            .copied()
            .unwrap_or(0);

        if input_val != output_val {
            return false;
        }
    }

    true
}

/// Reference-based validation (avoids cloning witness vectors).
pub fn validate_witness_soldering_ref(
    inputs_per_rep: &[&[u64]],
    outputs_per_rep: &[&[u64]],
    constraint: &SolderingConstraint,
) -> bool {
    let r = inputs_per_rep.len();

    for j in constraint.start_rep..=r {
        // Input at rep j (1-indexed, so array index is j-1)
        let input_val = inputs_per_rep[j - 1]
            .get(constraint.target_input_idx)
            .copied()
            .unwrap_or(0);

        // Output at rep j-1
        let output_val = outputs_per_rep[j - 2]
            .get(constraint.source_output_idx)
            .copied()
            .unwrap_or(0);

        if input_val != output_val {
            return false;
        }
    }

    true
}

/// Lagrange interpolation to get polynomial coefficients.
///
/// Given (x₁, y₁), ..., (xₙ, yₙ), finds polynomial f(·) such that f(xᵢ) = yᵢ.
///
/// When the `ntt` feature is enabled and the modulus is Goldilocks, uses the
/// optimized NTT-based interpolation which is O(n²) for arbitrary points.
/// Otherwise falls back to naive O(n³) Lagrange interpolation.
fn interpolate(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    assert_eq!(points.len(), values.len());

    if points.is_empty() {
        return vec![];
    }

    // Use Goldilocks NTT-optimized interpolation when available
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS {
        return Goldilocks::interpolate_u64(points, values);
    }

    // Fall back to naive Lagrange interpolation for other moduli
    interpolate_naive(points, values, modulus)
}

/// Naive Lagrange interpolation (O(n³) complexity).
///
/// Used as fallback when NTT is not available or modulus is not Goldilocks.
fn interpolate_naive(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
    let n = points.len();
    let mut result = vec![0u64; n];

    for i in 0..n {
        // Compute Lagrange basis polynomial Lᵢ(X)
        let mut basis = vec![1u64];

        // ∏_{j≠i} (X - xⱼ)
        for j in 0..n {
            if i == j {
                continue;
            }

            let mut new_basis = vec![0u64; basis.len() + 1];
            for (k, &coeff) in basis.iter().enumerate() {
                // coeff * X
                new_basis[k + 1] = (new_basis[k + 1] + coeff) % modulus;
                // coeff * (-xⱼ)
                let neg_x = (modulus - points[j]) % modulus;
                new_basis[k] = ((new_basis[k] as u128 + coeff as u128 * neg_x as u128)
                    % modulus as u128) as u64;
            }
            basis = new_basis;
        }

        // Compute ∏_{j≠i} (xᵢ - xⱼ)
        let mut denom = 1u128;
        for j in 0..n {
            if i == j {
                continue;
            }
            let diff = if points[i] >= points[j] {
                points[i] - points[j]
            } else {
                modulus - (points[j] - points[i])
            };
            denom = (denom * diff as u128) % modulus as u128;
        }

        // Modular inverse
        let denom_inv = mod_inverse(denom as u64, modulus);

        // Scale by yᵢ / denom
        let scale = ((values[i] as u128 * denom_inv as u128) % modulus as u128) as u64;

        for (k, &coeff) in basis.iter().enumerate() {
            let term = ((coeff as u128 * scale as u128) % modulus as u128) as u64;
            result[k] = (result[k] + term) % modulus;
        }
    }

    result
}

/// Evaluates a polynomial at a point.
fn evaluate_poly(coeffs: &[u64], x: u64, modulus: u64) -> u64 {
    let mut result = 0u128;
    let mut power = 1u128;
    for &c in coeffs {
        result = (result + c as u128 * power) % modulus as u128;
        power = (power * x as u128) % modulus as u128;
    }
    result as u64
}

/// Modular inverse using extended Euclidean algorithm.
fn mod_inverse(a: u64, m: u64) -> u64 {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }

    if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    }
}

/// Prover state for soldering proofs.
#[derive(Clone, Debug)]
pub struct SolderingProver {
    /// Soldering constraints.
    constraints: Vec<SolderingConstraint>,
    /// Masking polynomial pairs (r₁, r₂) for each constraint.
    masking_pairs: Vec<(Vec<u64>, Vec<u64>)>,
    /// Original input polynomial coefficients for each constrained input.
    input_polys: Vec<Vec<u64>>,
    /// Original output polynomial coefficients for each constrained output.
    output_polys: Vec<Vec<u64>>,
    /// Field modulus.
    modulus: u64,
    /// Evaluation points.
    eval_points: Vec<u64>,
}

impl SolderingProver {
    /// Creates a new soldering prover.
    pub fn new(modulus: u64) -> Self {
        Self {
            constraints: Vec::new(),
            masking_pairs: Vec::new(),
            input_polys: Vec::new(),
            output_polys: Vec::new(),
            modulus,
            eval_points: Vec::new(),
        }
    }

    /// Sets up the soldering prover with constraints and witness polynomials.
    ///
    /// # Arguments
    /// * `constraints` - Soldering constraints to enforce
    /// * `input_polys` - For each constraint, the input polynomial coefficients
    /// * `output_polys` - For each constraint, the output polynomial coefficients
    /// * `eval_points` - Evaluation points α₁, ..., αᵣ
    /// * `rng` - Random number generator
    pub fn setup<R: Rng>(
        &mut self,
        constraints: Vec<SolderingConstraint>,
        input_polys: Vec<Vec<u64>>,
        output_polys: Vec<Vec<u64>>,
        eval_points: Vec<u64>,
        rng: &mut R,
    ) {
        assert_eq!(constraints.len(), input_polys.len());
        assert_eq!(constraints.len(), output_polys.len());

        self.eval_points = eval_points.clone();
        self.constraints = constraints.clone();
        self.input_polys = input_polys;
        self.output_polys = output_polys;

        // Generate masking pairs for each constraint
        self.masking_pairs = constraints
            .iter()
            .map(|c| generate_masking_pair(&eval_points, c.start_rep, self.modulus, rng))
            .collect();
    }

    /// Generates commitment message for masking polynomials.
    pub fn commit(&self) -> SolderingCommitMessage {
        // In real implementation, this would create IT-PAC commitments.
        // For now, we use hashes as placeholders.
        let commitment_hashes: Vec<(u64, u64)> = self
            .masking_pairs
            .iter()
            .map(|(r1, r2)| {
                let h1 = r1.iter().fold(0u64, |acc, &c| (acc.wrapping_add(c)) % self.modulus);
                let h2 = r2.iter().fold(0u64, |acc, &c| (acc.wrapping_add(c)) % self.modulus);
                (h1, h2)
            })
            .collect();

        SolderingCommitMessage {
            num_constraints: self.constraints.len(),
            commitment_hashes,
        }
    }

    /// Reveals masked polynomials after receiving challenge.
    pub fn reveal(&self, phi: u64) -> SolderingRevealMessage {
        let masked_polys: Vec<(Vec<u64>, Vec<u64>)> = self
            .constraints
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let (r1, r2) = &self.masking_pairs[i];
                let in_poly = &self.input_polys[i];
                let out_poly = &self.output_polys[i];

                let f1 = compute_masked_poly(phi, in_poly, r1, self.modulus);
                let f2 = compute_masked_poly(phi, out_poly, r2, self.modulus);

                (f1, f2)
            })
            .collect();

        SolderingRevealMessage { masked_polys }
    }
}

/// Verifier state for soldering proofs.
#[derive(Clone, Debug)]
pub struct SolderingVerifier {
    /// Soldering constraints.
    constraints: Vec<SolderingConstraint>,
    /// Evaluation points.
    eval_points: Vec<u64>,
    /// Challenge φ.
    phi: Option<u64>,
    /// Field modulus.
    modulus: u64,
}

impl SolderingVerifier {
    /// Creates a new soldering verifier.
    pub fn new(modulus: u64) -> Self {
        Self {
            constraints: Vec::new(),
            eval_points: Vec::new(),
            phi: None,
            modulus,
        }
    }

    /// Sets up the verifier with constraints and evaluation points.
    pub fn setup(&mut self, constraints: Vec<SolderingConstraint>, eval_points: Vec<u64>) {
        self.constraints = constraints;
        self.eval_points = eval_points;
    }

    /// Receives commitment and generates challenge.
    pub fn receive_commit<R: Rng>(
        &mut self,
        _commit: SolderingCommitMessage,
        rng: &mut R,
    ) -> SolderingChallengeMessage {
        let phi = rng.random::<u64>() % self.modulus;
        self.phi = Some(phi);
        SolderingChallengeMessage { phi }
    }

    /// Verifies the soldering proof.
    pub fn verify(&self, reveal: &SolderingRevealMessage) -> bool {
        if reveal.masked_polys.len() != self.constraints.len() {
            return false;
        }

        for (i, constraint) in self.constraints.iter().enumerate() {
            let (f1_coeffs, f2_coeffs) = &reveal.masked_polys[i];

            if !verify_soldering_constraint(
                f1_coeffs,
                f2_coeffs,
                &self.eval_points,
                constraint.start_rep,
                self.modulus,
            ) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;

    const TEST_MODULUS: u64 = 65537;

    #[test]
    fn test_soldering_constraint_new() {
        let c = SolderingConstraint::new(0, 0);
        assert_eq!(c.target_input_idx, 0);
        assert_eq!(c.source_output_idx, 0);
        assert_eq!(c.start_rep, 2);
    }

    #[test]
    fn test_interpolation() {
        // f(1) = 3, f(2) = 5, f(3) = 7 => f(X) = 2X + 1
        let points = vec![1, 2, 3];
        let values = vec![3, 5, 7];
        let coeffs = interpolate(&points, &values, TEST_MODULUS);

        assert_eq!(evaluate_poly(&coeffs, 1, TEST_MODULUS), 3);
        assert_eq!(evaluate_poly(&coeffs, 2, TEST_MODULUS), 5);
        assert_eq!(evaluate_poly(&coeffs, 3, TEST_MODULUS), 7);
    }

    #[test]
    fn test_masking_pair_constraint() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];
        let start_rep = 2;

        let (r1, r2) = generate_masking_pair(&eval_points, start_rep, TEST_MODULUS, &mut rng);

        // Check: r₁(αⱼ) = r₂(αⱼ₋₁) for j = 2, 3, 4
        for j in start_rep..=4 {
            let r1_at_j = evaluate_poly(&r1, eval_points[j - 1], TEST_MODULUS);
            let r2_at_j_minus_1 = evaluate_poly(&r2, eval_points[j - 2], TEST_MODULUS);
            assert_eq!(
                r1_at_j, r2_at_j_minus_1,
                "Constraint failed at j={}: r1(α{})={} != r2(α{})={}",
                j, j, r1_at_j, j - 1, r2_at_j_minus_1
            );
        }
    }

    #[test]
    fn test_compute_masked_poly() {
        let phi = 3u64;
        let g_coeffs = vec![1, 2]; // g(X) = 1 + 2X
        let r_coeffs = vec![5, 7]; // r(X) = 5 + 7X

        let f_coeffs = compute_masked_poly(phi, &g_coeffs, &r_coeffs, TEST_MODULUS);

        // f(X) = φ·g(X) + r(X) = 3(1 + 2X) + (5 + 7X) = 8 + 13X
        assert_eq!(f_coeffs[0], 8);
        assert_eq!(f_coeffs[1], 13);
    }

    #[test]
    fn test_verify_soldering_constraint() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];
        let start_rep = 2;

        let (r1, r2) = generate_masking_pair(&eval_points, start_rep, TEST_MODULUS, &mut rng);

        // Verify that r1, r2 satisfy the constraint (trivially, since f1=r1, f2=r2 with phi=0)
        assert!(verify_soldering_constraint(
            &r1,
            &r2,
            &eval_points,
            start_rep,
            TEST_MODULUS
        ));
    }

    #[test]
    fn test_full_soldering_protocol() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];
        let constraint = SolderingConstraint::new(0, 0);

        // Simulate witness values that satisfy soldering:
        // input[0] at rep j = output[0] at rep j-1
        //
        // Let's say output at each rep is: 10, 20, 30, 40
        // Then input must be: (any), 10, 20, 30
        let output_values = vec![10u64, 20, 30, 40];
        let input_values = vec![5u64, 10, 20, 30]; // First can be anything

        // Interpolate to polynomials
        let in_poly = interpolate(&eval_points, &input_values, TEST_MODULUS);
        let out_poly = interpolate(&eval_points, &output_values, TEST_MODULUS);

        // Prover setup
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            vec![constraint.clone()],
            vec![in_poly],
            vec![out_poly],
            eval_points.clone(),
            &mut rng,
        );

        // Prover commits
        let commit_msg = prover.commit();
        assert_eq!(commit_msg.num_constraints, 1);

        // Verifier setup and challenge
        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(vec![constraint], eval_points);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        // Prover reveals
        let reveal_msg = prover.reveal(challenge.phi);

        // Verifier verifies
        assert!(verifier.verify(&reveal_msg));
    }

    #[test]
    fn test_soldering_rejects_invalid_witness() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];
        let constraint = SolderingConstraint::new(0, 0);

        // INVALID witness: input at rep j ≠ output at rep j-1
        let output_values = vec![10u64, 20, 30, 40];
        let input_values = vec![5u64, 99, 99, 99]; // WRONG - should be 10, 20, 30

        let in_poly = interpolate(&eval_points, &input_values, TEST_MODULUS);
        let out_poly = interpolate(&eval_points, &output_values, TEST_MODULUS);

        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            vec![constraint.clone()],
            vec![in_poly],
            vec![out_poly],
            eval_points.clone(),
            &mut rng,
        );

        let commit_msg = prover.commit();

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(vec![constraint], eval_points);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        let reveal_msg = prover.reveal(challenge.phi);

        // Should FAIL verification
        assert!(!verifier.verify(&reveal_msg));
    }

    #[test]
    fn test_validate_witness_soldering() {
        let constraint = SolderingConstraint::new(0, 0);

        // Valid witness
        let inputs = vec![vec![5], vec![10], vec![20], vec![30]];
        let outputs = vec![vec![10], vec![20], vec![30], vec![40]];

        assert!(validate_witness_soldering(&inputs, &outputs, &constraint));

        // Invalid witness
        let bad_inputs = vec![vec![5], vec![99], vec![99], vec![99]];
        assert!(!validate_witness_soldering(&bad_inputs, &outputs, &constraint));
    }

    #[test]
    fn test_multiple_constraints() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3];

        // Two constraints: input 0 from output 0, input 1 from output 1
        let constraints = vec![
            SolderingConstraint::new(0, 0),
            SolderingConstraint::new(1, 1),
        ];

        // Witness for constraint 0: in[0] at j = out[0] at j-1
        let out0_values = vec![10u64, 20, 30];
        let in0_values = vec![5u64, 10, 20];

        // Witness for constraint 1: in[1] at j = out[1] at j-1
        let out1_values = vec![100u64, 200, 300];
        let in1_values = vec![50u64, 100, 200];

        let in0_poly = interpolate(&eval_points, &in0_values, TEST_MODULUS);
        let in1_poly = interpolate(&eval_points, &in1_values, TEST_MODULUS);
        let out0_poly = interpolate(&eval_points, &out0_values, TEST_MODULUS);
        let out1_poly = interpolate(&eval_points, &out1_values, TEST_MODULUS);

        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            constraints.clone(),
            vec![in0_poly, in1_poly],
            vec![out0_poly, out1_poly],
            eval_points.clone(),
            &mut rng,
        );

        let commit_msg = prover.commit();
        assert_eq!(commit_msg.num_constraints, 2);

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(constraints, eval_points);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        let reveal_msg = prover.reveal(challenge.phi);

        assert!(verifier.verify(&reveal_msg));
    }
}
