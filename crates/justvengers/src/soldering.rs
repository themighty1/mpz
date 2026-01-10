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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SolderingCommitMessage {
    /// Number of soldering constraints.
    pub num_constraints: usize,
    /// Hash of masking polynomial commitments (simplified).
    /// In full implementation, these would be IT-PAC commitments.
    pub commitment_hashes: Vec<(u64, u64)>,
}

/// Challenge message from verifier for soldering verification.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SolderingChallengeMessage {
    /// Challenge φ for combining original and masking polynomials.
    pub phi: u64,
    /// Challenge ψ for aggregating multiple constraints into one.
    /// F₁ = Σᵢ ψⁱ × f₁ᵢ, F₂ = Σᵢ ψⁱ × f₂ᵢ
    pub psi: u64,
}

/// Revealed masked polynomials for soldering verification.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SolderingRevealMessage {
    /// For each constraint: (f₁ coefficients, f₂ coefficients).
    /// f₁(·) = φ·IN(·) + r₁(·)
    /// f₂(·) = φ·O(·) + r₂(·)
    ///
    /// Legacy: sends all S constraint pairs = O(S×R)
    pub masked_polys: Vec<(Vec<u64>, Vec<u64>)>,
}

/// Aggregated soldering reveal message - O(R) instead of O(S×R).
///
/// Uses random linear combination to aggregate S constraints into one:
/// F₁ = Σᵢ ψⁱ × f₁ᵢ, F₂ = Σᵢ ψⁱ × f₂ᵢ
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AggregatedSolderingReveal {
    /// Aggregated input polynomial: F₁ = Σᵢ ψⁱ × f₁ᵢ
    pub aggregated_f1: Vec<u64>,
    /// Aggregated output polynomial: F₂ = Σᵢ ψⁱ × f₂ᵢ
    pub aggregated_f2: Vec<u64>,
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
pub fn generate_masking_pair<Rn: Rng>(
    eval_points: &[u64],
    start_rep: usize,
    actual_reps: usize,
    modulus: u64,
    rng: &mut Rn,
) -> (Vec<u64>, Vec<u64>) {
    let n = eval_points.len(); // May be NTT-padded
    assert!(actual_reps >= 2, "Need at least 2 repetitions for soldering");
    assert!(start_rep >= 2 && start_rep <= actual_reps, "Invalid start_rep");
    assert!(actual_reps <= n, "actual_reps must not exceed eval_points length");

    // Number of constrained positions: from start_rep to actual_reps
    let num_constrained = actual_reps - start_rep + 1;

    // Generate random shared values for constrained positions
    let shared_values: Vec<u64> = (0..num_constrained)
        .map(|_| rng.random::<u64>() % modulus)
        .collect();

    // For r₁: constrained at positions start_rep, start_rep+1, ..., actual_reps
    // Array size is n (NTT-padded), but only positions 0..actual_reps are meaningful
    let mut r1_values: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % modulus).collect();
    for (idx, &shared) in shared_values.iter().enumerate() {
        let j = start_rep + idx;
        r1_values[j - 1] = shared;
    }

    // For r₂: constrained at positions start_rep-1, start_rep, ..., actual_reps-1
    let mut r2_values: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % modulus).collect();
    for (idx, &shared) in shared_values.iter().enumerate() {
        let j = start_rep + idx;
        r2_values[j - 2] = shared;
    }

    // Use NTT for Goldilocks (O(n log n)), fallback to Lagrange otherwise
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS && n.is_power_of_two() {
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
        result[i] = ((result[i] as u128 + phi_g as u128) % modulus as u128) as u64;
    }

    for (i, &r) in r_coeffs.iter().enumerate() {
        result[i] = ((result[i] as u128 + r as u128) % modulus as u128) as u64;
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
    verify_soldering_constraint_with_reps(f1_coeffs, f2_coeffs, eval_points, start_rep, r, modulus)
}

/// Verifies soldering constraint with explicit repetition count.
///
/// This is needed for NTT-padded eval_points where the actual repetition count
/// is less than eval_points.len().
///
/// For Goldilocks with NTT, uses O(n log n) NTT evaluation instead of O(n²) Horner.
pub fn verify_soldering_constraint_with_reps(
    f1_coeffs: &[u64],
    f2_coeffs: &[u64],
    eval_points: &[u64],
    start_rep: usize,
    actual_reps: usize,
    modulus: u64,
) -> bool {
    // For Goldilocks with NTT, use NTT to evaluate at all points at once (O(n log n))
    // instead of Horner for each point (O(n²) total)
    #[cfg(feature = "ntt")]
    if modulus == GOLDILOCKS && eval_points.len().is_power_of_two() {
        // Check if eval_points are NTT roots (ω^0, ω^1, ..., ω^(n-1))
        // If so, NTT(coeffs) gives evaluations directly
        let n = eval_points.len();

        // Apply NTT to get evaluations at all roots of unity
        let mut f1_evals: Vec<Goldilocks> = f1_coeffs.iter().map(|&c| Goldilocks::new(c)).collect();
        f1_evals.resize(n, Goldilocks::new(0));
        Goldilocks::ntt(&mut f1_evals);

        let mut f2_evals: Vec<Goldilocks> = f2_coeffs.iter().map(|&c| Goldilocks::new(c)).collect();
        f2_evals.resize(n, Goldilocks::new(0));
        Goldilocks::ntt(&mut f2_evals);

        // Check constraint: f1(ω^(j-1)) = f2(ω^(j-2)) for j in start_rep..=actual_reps
        for j in start_rep..=actual_reps {
            if f1_evals[j - 1] != f2_evals[j - 2] {
                return false;
            }
        }
        return true;
    }

    // Fallback: use Horner evaluation for each point
    for j in start_rep..=actual_reps {
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
pub fn interpolate(points: &[u64], values: &[u64], modulus: u64) -> Vec<u64> {
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
            result[k] = ((result[k] as u128 + term as u128) % modulus as u128) as u64;
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
    /// * `eval_points` - Evaluation points α₁, ..., αᵣ (may be NTT-padded)
    /// * `actual_reps` - Actual repetition count (may be less than eval_points.len() with NTT)
    /// * `rng` - Random number generator
    pub fn setup<Rn: Rng>(
        &mut self,
        constraints: Vec<SolderingConstraint>,
        input_polys: Vec<Vec<u64>>,
        output_polys: Vec<Vec<u64>>,
        eval_points: Vec<u64>,
        actual_reps: usize,
        rng: &mut Rn,
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
            .map(|c| generate_masking_pair(&eval_points, c.start_rep, actual_reps, self.modulus, rng))
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

    /// Reveals aggregated masked polynomials using random linear combination.
    ///
    /// Instead of sending S pairs of polynomials (O(S×R) communication),
    /// aggregates into a single pair using challenge ψ:
    /// - F₁ = Σᵢ ψⁱ × f₁ᵢ
    /// - F₂ = Σᵢ ψⁱ × f₂ᵢ
    ///
    /// This reduces communication from O(S×R) to O(R).
    pub fn reveal_aggregated(&self, phi: u64, psi: u64) -> AggregatedSolderingReveal {
        let n = self.eval_points.len();
        let mut aggregated_f1 = vec![0u64; n];
        let mut aggregated_f2 = vec![0u64; n];

        let mut psi_power = 1u64; // ψ^0 = 1

        for i in 0..self.constraints.len() {
            let (r1, r2) = &self.masking_pairs[i];
            let in_poly = &self.input_polys[i];
            let out_poly = &self.output_polys[i];

            // f₁ᵢ = φ·IN(·) + r₁(·)
            let f1 = compute_masked_poly(phi, in_poly, r1, self.modulus);
            // f₂ᵢ = φ·O(·) + r₂(·)
            let f2 = compute_masked_poly(phi, out_poly, r2, self.modulus);

            // F₁ += ψⁱ × f₁ᵢ, F₂ += ψⁱ × f₂ᵢ
            for (j, &coeff) in f1.iter().enumerate() {
                let term = ((psi_power as u128 * coeff as u128) % self.modulus as u128) as u64;
                aggregated_f1[j] = ((aggregated_f1[j] as u128 + term as u128) % self.modulus as u128) as u64;
            }
            for (j, &coeff) in f2.iter().enumerate() {
                let term = ((psi_power as u128 * coeff as u128) % self.modulus as u128) as u64;
                aggregated_f2[j] = ((aggregated_f2[j] as u128 + term as u128) % self.modulus as u128) as u64;
            }

            // ψⁱ⁺¹ = ψⁱ × ψ
            psi_power = ((psi_power as u128 * psi as u128) % self.modulus as u128) as u64;
        }

        AggregatedSolderingReveal {
            aggregated_f1,
            aggregated_f2,
        }
    }
}

/// Verifier state for soldering proofs.
#[derive(Clone, Debug)]
pub struct SolderingVerifier {
    /// Soldering constraints.
    constraints: Vec<SolderingConstraint>,
    /// Evaluation points (may be NTT-padded).
    eval_points: Vec<u64>,
    /// Actual repetition count (may be less than eval_points.len() with NTT).
    actual_reps: usize,
    /// Challenge φ for masking.
    phi: Option<u64>,
    /// Challenge ψ for aggregation.
    psi: Option<u64>,
    /// Field modulus.
    modulus: u64,
}

impl SolderingVerifier {
    /// Creates a new soldering verifier.
    pub fn new(modulus: u64) -> Self {
        Self {
            constraints: Vec::new(),
            eval_points: Vec::new(),
            actual_reps: 0,
            phi: None,
            psi: None,
            modulus,
        }
    }

    /// Sets up the verifier with constraints and evaluation points.
    ///
    /// # Arguments
    /// * `constraints` - The soldering constraints
    /// * `eval_points` - The evaluation points (may be NTT-padded)
    /// * `actual_reps` - The actual repetition count (for NTT, may be less than eval_points.len())
    pub fn setup(&mut self, constraints: Vec<SolderingConstraint>, eval_points: Vec<u64>, actual_reps: usize) {
        self.constraints = constraints;
        self.actual_reps = actual_reps;
        self.eval_points = eval_points;
    }

    /// Receives commitment and generates challenge.
    pub fn receive_commit<R: Rng>(
        &mut self,
        _commit: SolderingCommitMessage,
        rng: &mut R,
    ) -> SolderingChallengeMessage {
        let phi = rng.random::<u64>() % self.modulus;
        let psi = rng.random::<u64>() % self.modulus;
        self.phi = Some(phi);
        self.psi = Some(psi);
        SolderingChallengeMessage { phi, psi }
    }

    /// Verifies the soldering proof (legacy non-aggregated version).
    pub fn verify(&self, reveal: &SolderingRevealMessage) -> bool {
        if reveal.masked_polys.len() != self.constraints.len() {
            return false;
        }

        for (i, constraint) in self.constraints.iter().enumerate() {
            let (f1_coeffs, f2_coeffs) = &reveal.masked_polys[i];

            // Use actual_reps instead of eval_points.len() for NTT-padded cases
            if !verify_soldering_constraint_with_reps(
                f1_coeffs,
                f2_coeffs,
                &self.eval_points,
                constraint.start_rep,
                self.actual_reps,
                self.modulus,
            ) {
                return false;
            }
        }

        true
    }

    /// Verifies aggregated soldering proof (O(R) communication).
    ///
    /// The prover sends aggregated polynomials:
    /// - F₁ = Σᵢ ψⁱ × f₁ᵢ
    /// - F₂ = Σᵢ ψⁱ × f₂ᵢ
    ///
    /// The verifier checks: F₁(αⱼ) = F₂(αⱼ₋₁) for j in {2, ..., R}
    ///
    /// By linearity, this implies (with high probability over ψ) that
    /// f₁ᵢ(αⱼ) = f₂ᵢ(αⱼ₋₁) for all i and j.
    pub fn verify_aggregated(&self, reveal: &AggregatedSolderingReveal) -> bool {
        // All constraints have the same start_rep (2) for VM-style soldering
        // If different start_reps are needed, we'd need per-constraint aggregation
        let start_rep = self.constraints.first().map(|c| c.start_rep).unwrap_or(2);

        verify_soldering_constraint_with_reps(
            &reveal.aggregated_f1,
            &reveal.aggregated_f2,
            &self.eval_points,
            start_rep,
            self.actual_reps,
            self.modulus,
        )
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
        let actual_reps = eval_points.len();

        let (r1, r2) = generate_masking_pair(&eval_points, start_rep, actual_reps, TEST_MODULUS, &mut rng);

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
        let actual_reps = eval_points.len();

        let (r1, r2) = generate_masking_pair(&eval_points, start_rep, actual_reps, TEST_MODULUS, &mut rng);

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
        let actual_reps = eval_points.len();
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            vec![constraint.clone()],
            vec![in_poly],
            vec![out_poly],
            eval_points.clone(),
            actual_reps,
            &mut rng,
        );

        // Prover commits
        let commit_msg = prover.commit();
        assert_eq!(commit_msg.num_constraints, 1);

        // Verifier setup and challenge
        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(vec![constraint], eval_points, actual_reps);
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

        let actual_reps = eval_points.len();
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            vec![constraint.clone()],
            vec![in_poly],
            vec![out_poly],
            eval_points.clone(),
            actual_reps,
            &mut rng,
        );

        let commit_msg = prover.commit();

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(vec![constraint], eval_points, actual_reps);
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

        let actual_reps = eval_points.len();
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            constraints.clone(),
            vec![in0_poly, in1_poly],
            vec![out0_poly, out1_poly],
            eval_points.clone(),
            actual_reps,
            &mut rng,
        );

        let commit_msg = prover.commit();
        assert_eq!(commit_msg.num_constraints, 2);

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(constraints, eval_points, actual_reps);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        let reveal_msg = prover.reveal(challenge.phi);

        assert!(verifier.verify(&reveal_msg));
    }

    /// Test VM-style soldering where identity multiplications expose state values.
    ///
    /// Simulates a circuit where:
    /// - Inputs: old_state[0..N], new_state[0..N]
    /// - Identity mults at end: mult_output[offset+j] = new_state[j] * 1
    /// - Soldering: new_state[j] at rep i → old_state[j] at rep i+1
    #[test]
    fn test_soldering_with_identity_mults() {
        use crate::Circuit;

        const STATE_SIZE: usize = 4;

        // Create a circuit with identity multiplications for new_state
        fn create_test_circuit() -> Circuit {
            let mut circuit = Circuit::new();

            // Inputs: old_state[0..4], new_state[0..4]
            let mut old_state = Vec::new();
            let mut new_state = Vec::new();
            for _ in 0..STATE_SIZE {
                old_state.push(circuit.add_input());
            }
            for _ in 0..STATE_SIZE {
                new_state.push(circuit.add_input());
            }

            // Add constant 1 for identity mults
            let one = circuit.add_const(1);

            // Identity mults: mult_output[j] = new_state[j] * 1
            for j in 0..STATE_SIZE {
                circuit.add_mul(new_state[j], one);
            }

            circuit
        }

        // Generate inputs where new_state becomes old_state in next rep
        fn generate_chained_inputs(num_reps: usize) -> Vec<Vec<u64>> {
            let mut inputs = Vec::with_capacity(num_reps);
            let mut state = vec![0u64; STATE_SIZE];

            for rep in 0..num_reps {
                // old_state = previous state
                let old_state = state.clone();

                // new_state = transform (e.g., increment each element)
                let new_state: Vec<u64> = old_state.iter()
                    .map(|&v| v + rep as u64 + 1)
                    .collect();

                // Inputs: [old_state..., new_state...]
                let mut rep_inputs = Vec::with_capacity(STATE_SIZE * 2);
                rep_inputs.extend_from_slice(&old_state);
                rep_inputs.extend_from_slice(&new_state);
                inputs.push(rep_inputs);

                // new_state becomes old_state for next rep
                state = new_state;
            }

            inputs
        }

        let mut circuit = create_test_circuit();
        let num_reps = 4;
        let inputs = generate_chained_inputs(num_reps);

        // Evaluate circuit for each rep to get witnesses
        let witnesses: Vec<_> = inputs.iter()
            .map(|rep_inputs| circuit.evaluate(rep_inputs, TEST_MODULUS))
            .collect();

        // Verify identity mults produce new_state values
        for (rep, (witness, rep_inputs)) in witnesses.iter().zip(inputs.iter()).enumerate() {
            for j in 0..STATE_SIZE {
                let new_state_j = rep_inputs[STATE_SIZE + j]; // new_state[j]
                let mult_output_j = witness.mult_outputs[j]; // identity mult output
                assert_eq!(
                    mult_output_j, new_state_j,
                    "Rep {}: mult_output[{}]={} != new_state[{}]={}",
                    rep, j, mult_output_j, j, new_state_j
                );
            }
        }

        // Verify soldering constraint: mult_output[j] at rep i == input[j] at rep i+1
        for j in 0..STATE_SIZE {
            let constraint = SolderingConstraint::new(j, j); // input[j] from output[j]

            let inputs_slices: Vec<&[u64]> = inputs.iter().map(|v| v.as_slice()).collect();
            let outputs_slices: Vec<&[u64]> = witnesses.iter()
                .map(|w| w.mult_outputs.as_slice())
                .collect();

            assert!(
                validate_witness_soldering_ref(&inputs_slices, &outputs_slices, &constraint),
                "Soldering constraint failed for state element {}",
                j
            );
        }
    }

    /// Test that soldering validation correctly rejects mismatched state chains.
    #[test]
    fn test_soldering_rejects_broken_state_chain() {
        use crate::Circuit;

        const STATE_SIZE: usize = 2;

        fn create_test_circuit() -> Circuit {
            let mut circuit = Circuit::new();
            let mut new_state = Vec::new();
            for _ in 0..STATE_SIZE {
                circuit.add_input(); // old_state
            }
            for _ in 0..STATE_SIZE {
                new_state.push(circuit.add_input()); // new_state
            }
            let one = circuit.add_const(1);
            // Identity mults for new_state
            for j in 0..STATE_SIZE {
                circuit.add_mul(new_state[j], one);
            }
            circuit
        }

        let mut circuit = create_test_circuit();

        // Create BROKEN inputs where new_state does NOT become old_state
        let inputs = vec![
            vec![0, 0, 10, 20], // Rep 0: old=[0,0], new=[10,20]
            vec![99, 99, 30, 40], // Rep 1: old=[99,99] WRONG! should be [10,20]
            vec![30, 40, 50, 60], // Rep 2: old=[30,40], new=[50,60]
        ];

        let witnesses: Vec<_> = inputs.iter()
            .map(|rep_inputs| circuit.evaluate(rep_inputs, TEST_MODULUS))
            .collect();

        let constraint = SolderingConstraint::new(0, 0);

        let inputs_slices: Vec<&[u64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let outputs_slices: Vec<&[u64]> = witnesses.iter()
            .map(|w| w.mult_outputs.as_slice())
            .collect();

        // Should FAIL because rep 1's old_state[0]=99 != rep 0's new_state[0]=10
        assert!(
            !validate_witness_soldering_ref(&inputs_slices, &outputs_slices, &constraint),
            "Should reject broken state chain"
        );
    }

    /// Test VM-style circuit with constraint checks AND identity mults.
    /// This mimics the actual VM profile circuit structure.
    #[test]
    fn test_vm_style_circuit_soldering() {
        use crate::Circuit;
        use mpz_fields::goldilocks::GOLDILOCKS;

        const STATE_SIZE: usize = 4;
        const NUM_BRANCHES: usize = 3;
        const MODULUS: u64 = GOLDILOCKS;

        /// Create a VM-style circuit for a specific op.
        fn create_vm_circuit(op_value: u64) -> Circuit {
            let mut circuit = Circuit::new();

            let mut old_state = Vec::new();
            let mut new_state = Vec::new();
            for _ in 0..STATE_SIZE {
                old_state.push(circuit.add_input());
            }
            for _ in 0..STATE_SIZE {
                new_state.push(circuit.add_input());
            }
            let op = circuit.add_input();

            let op_const = circuit.add_const(op_value);
            let neg_one = circuit.add_const(MODULUS - 1);
            let one = circuit.add_const(1);

            // Some constraint check mults (simplified)
            let neg_op = circuit.add_mul(neg_one, op);
            let _check = circuit.add_mul(neg_op, op_const);

            // Identity mults for new_state at the END
            for j in 0..STATE_SIZE {
                circuit.add_mul(new_state[j], one);
            }

            circuit
        }

        /// Generate inputs with per-rep ops, chaining state.
        fn generate_inputs(num_reps: usize) -> (Vec<Vec<u64>>, Vec<usize>) {
            let mut inputs = Vec::with_capacity(num_reps);
            let mut branches = Vec::with_capacity(num_reps);
            let mut state = vec![0u64; STATE_SIZE];

            for j in 0..num_reps {
                let op = j % NUM_BRANCHES;
                branches.push(op);

                let old_state = state.clone();
                let mut new_state = old_state.clone();
                new_state[0] = op as u64; // First element = op
                new_state[STATE_SIZE - 1] = (old_state[STATE_SIZE - 1] + op as u64) % MODULUS;

                let mut rep_inputs = Vec::new();
                rep_inputs.extend_from_slice(&old_state);
                rep_inputs.extend_from_slice(&new_state);
                rep_inputs.push(op as u64);
                inputs.push(rep_inputs);

                state = new_state;
            }

            (inputs, branches)
        }

        let num_reps = 5;
        let (inputs, branches) = generate_inputs(num_reps);

        // Create circuit batch
        let circuits: Vec<Circuit> = (0..NUM_BRANCHES)
            .map(|op| create_vm_circuit(op as u64))
            .collect();

        // Evaluate each rep's circuit with its inputs
        let witnesses: Vec<_> = branches.iter()
            .zip(inputs.iter())
            .map(|(&branch, rep_inputs)| {
                let mut circuit = circuits[branch].clone();
                circuit.evaluate(rep_inputs, MODULUS)
            })
            .collect();

        // Get state_offset (identity mults are at the end)
        let sample = &circuits[0];
        let state_offset = sample.num_mults() - STATE_SIZE;

        // Debug: verify identity mults produce new_state values
        for (rep, (witness, rep_inputs)) in witnesses.iter().zip(inputs.iter()).enumerate() {
            for j in 0..STATE_SIZE {
                let new_state_j = rep_inputs[STATE_SIZE + j];
                let mult_output = witness.mult_outputs[state_offset + j];
                assert_eq!(
                    mult_output, new_state_j,
                    "Rep {}: mult_output[{}]={} != new_state[{}]={}",
                    rep, state_offset + j, mult_output, j, new_state_j
                );
            }
        }

        // Verify all soldering constraints
        let inputs_slices: Vec<&[u64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let outputs_slices: Vec<&[u64]> = witnesses.iter()
            .map(|w| w.mult_outputs.as_slice())
            .collect();

        for j in 0..STATE_SIZE {
            // new(target_input_idx, source_output_idx)
            // target = old_state[j] at input index j
            // source = new_state[j] at mult_output index (state_offset + j)
            let constraint = SolderingConstraint::new(j, state_offset + j);
            assert!(
                validate_witness_soldering_ref(&inputs_slices, &outputs_slices, &constraint),
                "Soldering constraint failed for state element {} (target_input={}, source_output={})",
                j, j, state_offset + j
            );
        }
    }

    /// Test aggregated soldering proof with multiple constraints.
    ///
    /// This tests the O(R) communication optimization that aggregates S constraints
    /// into a single proof using random linear combination.
    #[test]
    fn test_aggregated_soldering_protocol() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];

        // Two constraints: input 0 from output 0, input 1 from output 1
        let constraints = vec![
            SolderingConstraint::new(0, 0),
            SolderingConstraint::new(1, 1),
        ];

        // Valid witness for constraint 0
        let out0_values = vec![10u64, 20, 30, 40];
        let in0_values = vec![5u64, 10, 20, 30];

        // Valid witness for constraint 1
        let out1_values = vec![100u64, 200, 300, 400];
        let in1_values = vec![50u64, 100, 200, 300];

        let in0_poly = interpolate(&eval_points, &in0_values, TEST_MODULUS);
        let in1_poly = interpolate(&eval_points, &in1_values, TEST_MODULUS);
        let out0_poly = interpolate(&eval_points, &out0_values, TEST_MODULUS);
        let out1_poly = interpolate(&eval_points, &out1_values, TEST_MODULUS);

        let actual_reps = eval_points.len();
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            constraints.clone(),
            vec![in0_poly, in1_poly],
            vec![out0_poly, out1_poly],
            eval_points.clone(),
            actual_reps,
            &mut rng,
        );

        let commit_msg = prover.commit();
        assert_eq!(commit_msg.num_constraints, 2);

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(constraints, eval_points, actual_reps);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        // Test aggregated reveal and verify
        let aggregated_reveal = prover.reveal_aggregated(challenge.phi, challenge.psi);

        // Aggregated proof should be much smaller than non-aggregated
        // Non-aggregated: 2 constraints × 2 polys × 4 coeffs = 16 coefficients
        // Aggregated: 2 polys × 4 coeffs = 8 coefficients
        assert_eq!(aggregated_reveal.aggregated_f1.len(), 4);
        assert_eq!(aggregated_reveal.aggregated_f2.len(), 4);

        // Verify aggregated proof passes
        assert!(verifier.verify_aggregated(&aggregated_reveal), "Aggregated soldering verification failed");
    }

    /// Test that aggregated soldering rejects invalid witness.
    #[test]
    fn test_aggregated_soldering_rejects_invalid() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let eval_points: Vec<u64> = vec![1, 2, 3, 4];

        let constraint = SolderingConstraint::new(0, 0);

        // INVALID witness
        let output_values = vec![10u64, 20, 30, 40];
        let input_values = vec![5u64, 99, 99, 99]; // WRONG - should be 10, 20, 30

        let in_poly = interpolate(&eval_points, &input_values, TEST_MODULUS);
        let out_poly = interpolate(&eval_points, &output_values, TEST_MODULUS);

        let actual_reps = eval_points.len();
        let mut prover = SolderingProver::new(TEST_MODULUS);
        prover.setup(
            vec![constraint.clone()],
            vec![in_poly],
            vec![out_poly],
            eval_points.clone(),
            actual_reps,
            &mut rng,
        );

        let commit_msg = prover.commit();

        let mut verifier = SolderingVerifier::new(TEST_MODULUS);
        verifier.setup(vec![constraint], eval_points, actual_reps);
        let challenge = verifier.receive_commit(commit_msg, &mut rng);

        let aggregated_reveal = prover.reveal_aggregated(challenge.phi, challenge.psi);

        // Should FAIL verification
        assert!(!verifier.verify_aggregated(&aggregated_reveal), "Should reject invalid witness");
    }
}
