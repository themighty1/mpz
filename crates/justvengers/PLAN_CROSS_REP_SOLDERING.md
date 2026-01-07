# Implementation Plan: Cross-Repetition Soldering for Justvengers

## Overview

This plan implements **Section 4.3 "Soldering Repetitions"** from the Justvengers paper, allowing values from one repetition's active branch to be used as inputs in the next repetition.

## Goal

Enable proving constraints like:
```
∀j ∈ [R] \ {1}, IN₁(αⱼ) = O₁(αⱼ₋₁)
```
Where the first input of repetition j equals the first multiplication output of repetition j-1.

This enables **sequential computation** across repetitions, essential for:
- CPU instruction emulation (each step depends on previous state)
- Iterative algorithms (loop state carried forward)
- Chained computations (output → input flow)

## Technique: "Sacrifice" Method (from SPDZ)

The paper uses the sacrifice technique to prove polynomial equality at shifted points without revealing the polynomials:

### Mathematical Formulation

Given polynomials `IN(·)` and `O(·)` encoding values across R repetitions:
- `IN(αⱼ)` = input value at repetition j
- `O(αⱼ)` = output value at repetition j

We want to prove: `IN(αⱼ) = O(αⱼ₋₁)` for all j ∈ {2, ..., R}

### Protocol Steps

1. **P generates masking polynomials**:
   - Sample random `r₁(·)`, `r₂(·)` of degree R-1
   - Constrained so that: `r₁(αⱼ) = r₂(αⱼ₋₁)` for all j ∈ {2, ..., R}

2. **P commits to masking polynomials** via IT-PAC:
   - Generate `[r₁(·)]` and `[r₂(·)]`

3. **V sends challenge** `φ ← F`

4. **P reveals masked polynomials**:
   - `f₁(·) = φ·IN(·) + r₁(·)`
   - `f₂(·) = φ·O(·) + r₂(·)`

5. **V verifies**:
   - Check `f₁(αⱼ) = f₂(αⱼ₋₁)` for all j ∈ {2, ..., R}
   - This implies `φ·IN(αⱼ) + r₁(αⱼ) = φ·O(αⱼ₋₁) + r₂(αⱼ₋₁)`
   - Since `r₁(αⱼ) = r₂(αⱼ₋₁)` by construction, this reduces to:
   - `φ·IN(αⱼ) = φ·O(αⱼ₋₁)`, hence `IN(αⱼ) = O(αⱼ₋₁)` ✓

### Why It's Zero-Knowledge

The random masking polynomials `r₁`, `r₂` act as one-time pads. Since they're uniformly random (subject to the constraint), the revealed polynomials `f₁`, `f₂` reveal nothing about `IN` or `O`.

### Soundness

If `IN(αⱼ) ≠ O(αⱼ₋₁)` for some j, then `φ·IN(αⱼ) + r₁(αⱼ) = φ·O(αⱼ₋₁) + r₂(αⱼ₋₁)` only holds when `φ` is a root of a non-zero polynomial, which happens with probability ≤ 1/|F|.

## Implementation Plan

### Phase 1: Data Structures

#### 1.1 Add SolderingConstraint type
```rust
/// A constraint linking output of one repetition to input of the next
pub struct SolderingConstraint {
    /// Which input wire receives the value (0-indexed)
    pub target_input_idx: usize,
    /// Which output wire provides the value (0-indexed mult output)
    pub source_output_idx: usize,
    /// Starting repetition (constraint applies from rep start_rep to R)
    pub start_rep: usize,
}
```

#### 1.2 Extend ProverState
```rust
pub struct ProverState<const R: usize> {
    // ... existing fields ...

    /// Soldering constraints to enforce
    soldering_constraints: Vec<SolderingConstraint>,

    /// Masking polynomials for each constraint (r₁, r₂ pairs)
    masking_polys: Vec<(Vec<u64>, Vec<u64>)>,

    /// Committed masking polynomial IT-PACs
    masking_commitments: Vec<(u64, u64)>, // [r₁(Λ)], [r₂(Λ)]
}
```

#### 1.3 Extend VerifierState
```rust
pub struct VerifierState<const R: usize> {
    // ... existing fields ...

    /// Soldering challenge φ
    soldering_challenge: Option<u64>,

    /// Received masked polynomials for verification
    masked_polys: Vec<(Vec<u64>, Vec<u64>)>,
}
```

### Phase 2: Masking Polynomial Generation

#### 2.1 Create `soldering.rs` module
```rust
/// Generate a random polynomial r(·) of degree R-1 with specified evaluations
/// at shifted points.
///
/// Given target values v₂, v₃, ..., v_R, constructs r(·) such that:
/// r(α₂) = v₂, r(α₃) = v₃, ..., r(α_R) = v_R
/// and r(α₁) is uniformly random.
pub fn generate_shifted_masking_poly<const R: usize>(
    eval_points: &[u64; R],
    target_values: &[u64], // length R-1, for positions 2..R
    modulus: u64,
    rng: &mut impl Rng,
) -> Vec<u64>
```

#### 2.2 Generate constrained masking pair
```rust
/// Generate (r₁, r₂) pair satisfying r₁(αⱼ) = r₂(αⱼ₋₁) for j ∈ {2,...,R}
pub fn generate_masking_pair<const R: usize>(
    eval_points: &[u64; R],
    modulus: u64,
    rng: &mut impl Rng,
) -> (Vec<u64>, Vec<u64>) {
    // 1. Sample R-1 random values for the "shifted" positions
    let shared_values: Vec<u64> = (0..R-1)
        .map(|_| rng.gen::<u64>() % modulus)
        .collect();

    // 2. r₁ must satisfy: r₁(α₂) = v₁, r₁(α₃) = v₂, ..., r₁(α_R) = v_{R-1}
    //    with r₁(α₁) random
    let r1 = generate_shifted_masking_poly::<R>(
        eval_points,
        &shared_values, // values at α₂, α₃, ..., α_R
        modulus,
        rng,
    );

    // 3. r₂ must satisfy: r₂(α₁) = v₁, r₂(α₂) = v₂, ..., r₂(α_{R-1}) = v_{R-1}
    //    with r₂(α_R) random
    let r2 = generate_unshifted_masking_poly::<R>(
        eval_points,
        &shared_values, // values at α₁, α₂, ..., α_{R-1}
        modulus,
        rng,
    );

    (r1, r2)
}
```

### Phase 3: Protocol Integration

#### 3.1 New protocol phase: Soldering Proof

Add between existing phases (after commitment, before/during disclosure):

```rust
impl<const R: usize> ProverState<R> {
    /// Phase 2.5: Generate and commit soldering proofs
    pub fn commit_soldering(
        &mut self,
        constraints: &[SolderingConstraint],
    ) -> Result<SolderingCommitMessage, ProverError> {
        // For each constraint:
        // 1. Generate masking pair (r₁, r₂)
        // 2. Commit via IT-PAC
        // 3. Store for later revelation
    }

    /// Phase 3.5: Reveal masked polynomials after challenge
    pub fn reveal_soldering(
        &mut self,
        challenge_phi: u64,
    ) -> Result<SolderingRevealMessage, ProverError> {
        // For each constraint:
        // 1. Compute f₁(·) = φ·IN(·) + r₁(·)
        // 2. Compute f₂(·) = φ·O(·) + r₂(·)
        // 3. Return coefficients
    }
}

impl<const R: usize> VerifierState<R> {
    /// Receive soldering commitments
    pub fn receive_soldering_commit(
        &mut self,
        msg: SolderingCommitMessage,
    ) -> Result<SolderingChallengeMessage, VerifierError> {
        // Store commitments
        // Generate and return challenge φ
    }

    /// Verify soldering proofs
    pub fn verify_soldering(
        &mut self,
        msg: SolderingRevealMessage,
    ) -> Result<bool, VerifierError> {
        // For each constraint:
        // 1. Evaluate f₁(αⱼ) and f₂(αⱼ₋₁) for j ∈ {2,...,R}
        // 2. Check equality
    }
}
```

#### 3.2 New message types

```rust
/// Commitment to masking polynomials
pub struct SolderingCommitMessage {
    pub num_constraints: usize,
    /// IT-PAC commitments for (r₁, r₂) pairs
    pub masking_commitments: Vec<(u64, u64)>,
}

/// Challenge for soldering verification
pub struct SolderingChallengeMessage {
    pub phi: u64,
}

/// Revealed masked polynomials
pub struct SolderingRevealMessage {
    /// (f₁ coefficients, f₂ coefficients) for each constraint
    pub masked_polys: Vec<(Vec<u64>, Vec<u64>)>,
}
```

### Phase 4: Wire Mapping

#### 4.1 Extended witness wire indexing

```rust
/// Maps logical constraint to actual polynomial indices
pub struct WireMapping {
    /// Input polynomial indices (IN_k for k ∈ [n_in])
    pub input_poly_indices: Vec<usize>,
    /// Output polynomial indices (O_k for k ∈ [n_×])
    pub output_poly_indices: Vec<usize>,
}
```

### Phase 5: Integration into run_protocol

```rust
pub fn run_protocol_with_soldering<const R: usize>(
    circuits: &CircuitBatch,
    active_branch: usize,
    inputs_per_rep: &[Vec<u64>],
    soldering: &[SolderingConstraint],
    modulus: u64,
) -> Result<bool, ProtocolError> {
    // ... existing setup ...

    // Phase 1: Prover commits (existing)
    let commitment = prover.commit(&setup_msg.eval_points)?;

    // Phase 2: Verifier sends χ (existing)
    let chi_msg = verifier.receive_commitment(commitment)?;

    // NEW: Phase 2.5: Soldering commitment
    let soldering_commit = prover.commit_soldering(soldering)?;
    let soldering_challenge = verifier.receive_soldering_commit(soldering_commit)?;

    // Phase 3: Prover discloses (existing)
    let disclosure = prover.disclose(chi, verifier.topology_vectors())?;

    // NEW: Phase 3.5: Soldering reveal
    let soldering_reveal = prover.reveal_soldering(soldering_challenge.phi)?;
    let soldering_ok = verifier.verify_soldering(soldering_reveal)?;

    // ... rest of protocol ...

    Ok(result && soldering_ok)
}
```

### Phase 6: Testing

#### 6.1 Unit tests for masking polynomial generation
```rust
#[test]
fn test_masking_pair_constraint() {
    // Verify r₁(αⱼ) = r₂(αⱼ₋₁) holds
}
```

#### 6.2 Integration test: Simple chain
```rust
#[test]
fn test_soldering_simple_chain() {
    // Circuit: out = in * 2
    // Solder: rep[j].in = rep[j-1].out
    // So: rep[1].out = in₁ * 2
    //     rep[2].in = rep[1].out = in₁ * 2
    //     rep[2].out = (in₁ * 2) * 2 = in₁ * 4
    // etc.
}
```

#### 6.3 Integration test: Multiple constraints
```rust
#[test]
fn test_soldering_multiple_wires() {
    // Multiple outputs feeding into multiple inputs
}
```

#### 6.4 Soundness test
```rust
#[test]
fn test_soldering_rejects_invalid() {
    // Prover provides inputs that don't satisfy soldering
    // Verification should fail
}
```

### Phase 7: Benchmarks

Add to `e2e_bench.rs`:
```rust
fn bench_soldering(c: &mut Criterion) {
    // Benchmark overhead of soldering vs non-soldering
}
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/lib.rs` | Add `soldering` module, extend `run_protocol` |
| `src/soldering.rs` | NEW: Masking poly generation, constraint types |
| `src/prover.rs` | Add soldering phases, state fields |
| `src/verifier.rs` | Add soldering verification, state fields |
| `src/topology.rs` | Wire mapping utilities |
| `tests/e2e_integration.rs` | Soldering integration tests |
| `benches/e2e_bench.rs` | Soldering benchmarks |

## Communication Overhead Analysis

Per soldering constraint:
- **Commitment phase**: 2 IT-PAC commitments = O(1) field elements
- **Reveal phase**: 2 degree-(R-1) polynomials = O(R) field elements

For S soldering constraints: **O(S·R)** additional communication

This is acceptable since:
1. S is typically small (e.g., 1-10 for CPU state)
2. It's additive, not multiplicative with B or C
3. Total remains O(R + B + C + S·R) = O(R·(S+1) + B + C)

## Security Considerations

1. **Masking polynomial must be truly random** - use cryptographic RNG
2. **Challenge φ must be unpredictable** - generated after commitment
3. **Soundness error**: ≤ S/|F| per constraint (negligible for large fields)
4. **ZK**: Random padding ensures revealed polynomials are uniformly distributed

## Next Steps

1. Implement `soldering.rs` with masking polynomial generation
2. Extend `ProverState` with soldering fields and methods
3. Extend `VerifierState` with verification logic
4. Update `run_protocol` to include soldering phases
5. Add comprehensive tests
6. Add benchmarks
