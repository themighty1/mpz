//! End-to-end benchmarks for Justvengers protocol.
//!
//! Benchmarks the full protocol with:
//! - 100 branches
//! - 1000 multiplication gates per branch
//! - 100 repetitions

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mpz_justvengers::{Circuit, CircuitBatch, ExtendedWitness, UniversalHash};
use mpz_justvengers_core::{
    ahe::ParamSet,
    itmac::{GlobalKey, ItMac, ItMacField, VolePool},
    itpac::interpolate,
};

use mpz_core::{prg::Prg, Block};
use rand::{Rng, SeedableRng};
use std::ops::{Add, Mul, Sub};

/// Field element for IT-MAC operations.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Field64(u64);

/// Prime modulus: 2^61 - 1 (Mersenne prime)
const MODULUS: u64 = (1u64 << 61) - 1;

impl Field64 {
    fn from_u64(v: u64) -> Self {
        Self(v % MODULUS)
    }
}

impl Add for Field64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let sum = self.0 as u128 + rhs.0 as u128;
        Self((sum % MODULUS as u128) as u64)
    }
}

impl Sub for Field64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(MODULUS - (rhs.0 - self.0))
        }
    }
}

impl Mul for Field64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as u128 * rhs.0 as u128) % MODULUS as u128;
        Self(prod as u64)
    }
}

impl ItMacField for Field64 {
    fn zero() -> Self {
        Self(0)
    }
    fn one() -> Self {
        Self(1)
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        Self(rng.random_range(0..MODULUS))
    }
    fn neg(self) -> Self {
        if self.0 == 0 {
            Self(0)
        } else {
            Self(MODULUS - self.0)
        }
    }
}

/// Full scale parameters (same as e2e test)
const NUM_BRANCHES: usize = 100;
const NUM_MULT_GATES: usize = 1000;
const NUM_REPETITIONS: usize = 100;

/// Creates a circuit with the specified number of multiplication gates.
fn create_circuit_with_mults(num_mults: usize) -> Circuit {
    let mut circuit = Circuit::new();
    let x = circuit.add_input();
    let y = circuit.add_input();
    let mut prev = circuit.add_mul(x, y);
    for _ in 1..num_mults {
        prev = circuit.add_mul(prev, x);
    }
    circuit
}

/// Creates a batch of circuits.
fn create_circuit_batch(num_branches: usize, num_mults: usize) -> CircuitBatch {
    let circuits: Vec<Circuit> = (0..num_branches)
        .map(|_| create_circuit_with_mults(num_mults))
        .collect();
    CircuitBatch::new(circuits)
}

/// Generates random inputs for R repetitions.
fn generate_random_inputs<R: Rng>(
    num_repetitions: usize,
    num_inputs: usize,
    rng: &mut R,
) -> Vec<Vec<u64>> {
    (0..num_repetitions)
        .map(|_| {
            (0..num_inputs)
                .map(|_| rng.random_range(1..1000u64))
                .collect()
        })
        .collect()
}

/// Generates chained inputs for soldering where each rep uses previous rep's output.
///
/// For a circuit with 2 inputs where out = in0 * in1:
/// - Rep 1: in0 = initial_value, in1 = multiplier
/// - Rep 2: in0 = out[0] from rep 1, in1 = multiplier
/// - Rep 3: in0 = out[0] from rep 2, in1 = multiplier
/// - etc.
///
/// This creates a chain: v, v*m, v*m^2, v*m^3, ...
fn generate_chained_inputs(
    circuit: &Circuit,
    num_repetitions: usize,
    initial_value: u64,
    multiplier: u64,
    modulus: u64,
) -> Vec<Vec<u64>> {
    let mut inputs = Vec::with_capacity(num_repetitions);
    let mut prev_output = initial_value;

    for _ in 0..num_repetitions {
        // First input is chained from previous output (or initial for rep 0)
        // Remaining inputs are the multiplier
        let num_inputs = circuit.num_inputs();
        let mut rep_inputs = vec![prev_output];
        for _ in 1..num_inputs {
            rep_inputs.push(multiplier);
        }
        inputs.push(rep_inputs.clone());

        // Evaluate to get output for next rep
        let mut circuit_clone = circuit.clone();
        let witness = circuit_clone.evaluate(&rep_inputs, modulus);
        if !witness.mult_outputs.is_empty() {
            prev_output = witness.mult_outputs[0];
        }
    }

    inputs
}

/// IT-MAC commitment for a witness value.
struct ItMacCommitment<F: ItMacField> {
    value: F,
    mac: ItMac<F>,
}

impl<F: ItMacField> ItMacCommitment<F> {
    fn new<R: Rng>(global_key: &GlobalKey<F>, value: F, rng: &mut R) -> Self {
        let mac = ItMac::commit(global_key, value, rng);
        Self { value, mac }
    }

    fn verify(&self, global_key: &GlobalKey<F>) -> bool {
        self.mac.verify(global_key)
    }
}

/// Commits to an extended witness using IT-MACs.
fn commit_witness<R: Rng>(
    global_key: &GlobalKey<Field64>,
    witness: &ExtendedWitness,
    rng: &mut R,
) -> Vec<ItMacCommitment<Field64>> {
    witness
        .to_vec()
        .iter()
        .map(|&v| ItMacCommitment::new(global_key, Field64::from_u64(v), rng))
        .collect()
}

/// Verifies all IT-MAC commitments.
fn verify_commitments(
    global_key: &GlobalKey<Field64>,
    commitments: &[ItMacCommitment<Field64>],
) -> bool {
    commitments.iter().all(|c| c.verify(global_key))
}

/// Verifies multiplication constraints.
fn verify_multiplication_constraints(
    global_key: &GlobalKey<Field64>,
    witness: &ExtendedWitness,
    commitments: &[ItMacCommitment<Field64>],
) -> bool {
    let n = witness.inputs.len();
    let m = witness.num_mults();

    for i in 0..m {
        let a_idx = n + i;
        let b_idx = n + m + i;
        let c_idx = n + 2 * m + i;

        let a = commitments[a_idx].value;
        let b = commitments[b_idx].value;
        let c = commitments[c_idx].value;

        if a * b != c {
            return false;
        }

        if !commitments[a_idx].verify(global_key)
            || !commitments[b_idx].verify(global_key)
            || !commitments[c_idx].verify(global_key)
        {
            return false;
        }
    }
    true
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_circuit_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_creation");

    for (branches, mults) in [(10, 100), (50, 500), (100, 1000)] {
        group.throughput(Throughput::Elements((branches * mults) as u64));
        group.bench_with_input(
            BenchmarkId::new("create_batch", format!("{}x{}", branches, mults)),
            &(branches, mults),
            |b, &(branches, mults)| {
                b.iter(|| black_box(create_circuit_batch(branches, mults)));
            },
        );
    }

    group.finish();
}

fn bench_circuit_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_evaluation");

    let mut rng = Prg::from_seed(Block::ZERO);

    for (mults, reps) in [(100, 10), (500, 50), (1000, 100)] {
        let circuit = create_circuit_with_mults(mults);
        let inputs = generate_random_inputs(reps, circuit.num_inputs(), &mut rng);

        group.throughput(Throughput::Elements((mults * reps) as u64));
        group.bench_with_input(
            BenchmarkId::new("evaluate", format!("{}mults_{}reps", mults, reps)),
            &(circuit, inputs),
            |b, (circuit, inputs)| {
                b.iter(|| {
                    for input in inputs {
                        let mut c = circuit.clone();
                        black_box(c.evaluate(input, MODULUS));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_itmac_commitment(c: &mut Criterion) {
    let mut group = c.benchmark_group("itmac_commitment");

    let mut rng = Prg::from_seed(Block::ZERO);
    let global_key = GlobalKey::<Field64>::generate(&mut rng);

    for witness_size in [100, 1000, 3000] {
        let witness_values: Vec<u64> = (0..witness_size)
            .map(|_| rng.random_range(0..MODULUS))
            .collect();

        group.throughput(Throughput::Elements(witness_size as u64));
        group.bench_with_input(
            BenchmarkId::new("commit", witness_size),
            &witness_values,
            |b, values| {
                b.iter(|| {
                    let commitments: Vec<_> = values
                        .iter()
                        .map(|&v| {
                            ItMacCommitment::new(&global_key, Field64::from_u64(v), &mut rng)
                        })
                        .collect();
                    black_box(commitments)
                });
            },
        );
    }

    group.finish();
}

fn bench_topology_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_vectors");

    let mut rng = Prg::from_seed(Block::ZERO);

    for branches in [10, 50, 100] {
        let circuits = create_circuit_batch(branches, 100);
        let chi: u64 = rng.random_range(1..MODULUS);

        group.throughput(Throughput::Elements(branches as u64));
        group.bench_with_input(
            BenchmarkId::new("compute", branches),
            &(circuits, chi),
            |b, (circuits, chi)| {
                b.iter(|| black_box(circuits.topology_vectors(*chi, MODULUS)));
            },
        );
    }

    group.finish();
}

fn bench_universal_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_hash");

    let mut rng = Prg::from_seed(Block::ZERO);

    for branches in [10, 50, 100] {
        let circuits = create_circuit_batch(branches, 100);
        let chi: u64 = rng.random_range(1..MODULUS);
        let rho: u64 = rng.random_range(1..MODULUS);
        let topology_vectors = circuits.topology_vectors(chi, MODULUS);

        group.throughput(Throughput::Elements(branches as u64));
        group.bench_with_input(
            BenchmarkId::new("compute", branches),
            &(topology_vectors, rho),
            |b, (tvs, rho)| {
                b.iter(|| black_box(UniversalHash::compute(tvs, *rho, MODULUS)));
            },
        );
    }

    group.finish();
}

fn bench_vole_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("vole_pool");

    let mut rng = Prg::from_seed(Block::ZERO);
    let global_key = GlobalKey::<Field64>::generate(&mut rng);

    for pool_size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(pool_size as u64));
        group.bench_with_input(
            BenchmarkId::new("generate", pool_size),
            &pool_size,
            |b, &size| {
                b.iter(|| black_box(VolePool::<Field64>::generate(&global_key, size, &mut rng)));
            },
        );
    }

    group.finish();
}

fn bench_itpac_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("itpac_interpolation");

    let mut rng = Prg::from_seed(Block::ZERO);
    let ahe_params = ParamSet::Toy.params();

    for num_points in [10, 50, 100] {
        let eval_points: Vec<u64> = (1..=num_points as u64).collect();
        let values: Vec<u64> = (0..num_points)
            .map(|_| rng.random_range(0..ahe_params.t))
            .collect();

        group.throughput(Throughput::Elements(num_points as u64));
        group.bench_with_input(
            BenchmarkId::new("interpolate", num_points),
            &(eval_points, values),
            |b, (points, vals)| {
                b.iter(|| black_box(interpolate(points, vals, ahe_params.t)));
            },
        );
    }

    group.finish();
}

fn bench_e2e_full_protocol(c: &mut Criterion) {
    use mpz_justvengers::{run_protocol_with_soldering, SolderingConstraint};

    let mut group = c.benchmark_group("e2e_full_protocol");
    group.sample_size(10); // Fewer samples for expensive benchmarks

    let mut rng = Prg::from_seed(Block::ZERO);

    // Full scale parameters
    let num_branches = NUM_BRANCHES;
    let num_mults = NUM_MULT_GATES;
    let num_reps = NUM_REPETITIONS;
    let active_branch = 42;

    // Pre-create circuits (one-time setup)
    let circuits = create_circuit_batch(num_branches, num_mults);
    let circuit = circuits.get(active_branch).unwrap();

    // Generate CHAINED inputs for soldering: each rep uses previous output
    let chained_inputs = generate_chained_inputs(circuit, num_reps, 3, 2, MODULUS);

    // Soldering constraint: input 0 receives output 0 from previous rep
    let soldering_constraint = SolderingConstraint::new(0, 0);

    // Throughput = total AND gates = reps × branches × circuit_size
    let total_and_gates = (num_reps * num_branches * num_mults) as u64;
    group.throughput(Throughput::Elements(total_and_gates));

    // Benchmark individual phases
    group.bench_function("phase1_setup", |b| {
        b.iter(|| {
            let global_key = GlobalKey::<Field64>::generate(&mut rng);
            let chi: u64 = rng.random_range(1..MODULUS);
            let rho: u64 = rng.random_range(1..MODULUS);
            black_box((global_key, chi, rho))
        });
    });

    group.bench_function("phase2_evaluate_chained", |b| {
        let global_key = GlobalKey::<Field64>::generate(&mut rng);

        b.iter(|| {
            let mut witnesses = Vec::with_capacity(num_reps);
            let mut commitments = Vec::with_capacity(num_reps);

            for rep in 0..num_reps {
                let mut c = circuit.clone();
                let witness = c.evaluate(&chained_inputs[rep], MODULUS);
                let commit = commit_witness(&global_key, &witness, &mut rng);
                witnesses.push(witness);
                commitments.push(commit);
            }

            black_box((witnesses, commitments))
        });
    });

    group.bench_function("phase3_topology_vectors", |b| {
        let chi: u64 = rng.random_range(1..MODULUS);

        b.iter(|| black_box(circuits.topology_vectors(chi, MODULUS)));
    });

    group.bench_function("phase4_universal_hash", |b| {
        let chi: u64 = rng.random_range(1..MODULUS);
        let rho: u64 = rng.random_range(1..MODULUS);
        let topology_vectors = circuits.topology_vectors(chi, MODULUS);

        b.iter(|| black_box(UniversalHash::compute(&topology_vectors, rho, MODULUS)));
    });

    group.bench_function("phase5_verification_with_soldering", |b| {
        let global_key = GlobalKey::<Field64>::generate(&mut rng);

        // Pre-compute witnesses and commitments with chained inputs
        let mut witnesses = Vec::with_capacity(num_reps);
        let mut all_commitments = Vec::with_capacity(num_reps);

        for rep in 0..num_reps {
            let mut c = circuit.clone();
            let witness = c.evaluate(&chained_inputs[rep], MODULUS);
            let commit = commit_witness(&global_key, &witness, &mut rng);
            witnesses.push(witness);
            all_commitments.push(commit);
        }

        b.iter(|| {
            // Verify IT-MAC commitments
            for commitments in &all_commitments {
                black_box(verify_commitments(&global_key, commitments));
            }

            // Verify multiplication constraints
            for (witness, commitments) in witnesses.iter().zip(all_commitments.iter()) {
                black_box(verify_multiplication_constraints(
                    &global_key,
                    witness,
                    commitments,
                ));
            }

            // Verify soldering: input[j] = output[j-1] for j >= 2
            for j in 1..num_reps {
                let prev_output = witnesses[j - 1].mult_outputs[0];
                let curr_input = chained_inputs[j][0];
                black_box(prev_output == curr_input);
            }
        });
    });

    // Full end-to-end benchmark WITH SOLDERING
    group.bench_function("full_e2e_with_soldering", |b| {
        b.iter(|| {
            let global_key = GlobalKey::<Field64>::generate(&mut rng);
            let chi: u64 = rng.random_range(1..MODULUS);
            let rho: u64 = rng.random_range(1..MODULUS);

            // Phase 2: Evaluate and commit with chained inputs
            let mut witnesses = Vec::with_capacity(num_reps);
            let mut all_commitments = Vec::with_capacity(num_reps);

            for rep in 0..num_reps {
                let mut c = circuit.clone();
                let witness = c.evaluate(&chained_inputs[rep], MODULUS);
                let commit = commit_witness(&global_key, &witness, &mut rng);
                witnesses.push(witness);
                all_commitments.push(commit);
            }

            // Phase 3: Topology vectors
            let topology_vectors = circuits.topology_vectors(chi, MODULUS);

            // Phase 4: Universal hash
            let _hash = UniversalHash::compute(&topology_vectors, rho, MODULUS);

            // Phase 5: Verification with soldering check
            let mut valid = true;
            for commitments in &all_commitments {
                valid &= verify_commitments(&global_key, commitments);
            }
            for (witness, commitments) in witnesses.iter().zip(all_commitments.iter()) {
                valid &= verify_multiplication_constraints(&global_key, witness, commitments);
            }

            // Verify soldering constraint
            for j in 1..num_reps {
                valid &= witnesses[j - 1].mult_outputs[0] == chained_inputs[j][0];
            }

            black_box(valid)
        });
    });

    // Full protocol using run_protocol_with_soldering
    // 10 repetitions
    let chained_inputs_10 = generate_chained_inputs(circuit, 10, 3, 2, MODULUS);

    group.bench_function("run_protocol_with_soldering_10_reps", |b| {
        b.iter(|| {
            black_box(run_protocol_with_soldering::<10>(
                &circuits,
                active_branch,
                &chained_inputs_10,
                &[soldering_constraint.clone()],
                MODULUS,
            ))
        });
    });

    // 100 repetitions
    group.bench_function("run_protocol_with_soldering_100_reps", |b| {
        b.iter(|| {
            black_box(run_protocol_with_soldering::<100>(
                &circuits,
                active_branch,
                &chained_inputs,
                &[soldering_constraint.clone()],
                MODULUS,
            ))
        });
    });

    // 100K repetitions
    let chained_inputs_100k = generate_chained_inputs(circuit, 100_000, 3, 2, MODULUS);

    group.bench_function("run_protocol_with_soldering_100K_reps", |b| {
        b.iter(|| {
            black_box(run_protocol_with_soldering::<100000>(
                &circuits,
                active_branch,
                &chained_inputs_100k,
                &[soldering_constraint.clone()],
                MODULUS,
            ))
        });
    });

    group.finish();
}

fn bench_e2e_scaling(c: &mut Criterion) {
    use mpz_justvengers::{run_protocol_with_soldering, SolderingConstraint};

    let mut group = c.benchmark_group("e2e_scaling_with_soldering");
    group.sample_size(10);

    let mut rng = Prg::from_seed(Block::ZERO);

    // Soldering constraint: input 0 receives output 0 from previous rep
    let soldering_constraint = SolderingConstraint::new(0, 0);

    // Test different scales with CHAINED inputs (soldering)
    for (branches, mults, reps) in [
        (10, 100, 10),
        (50, 500, 50),
        (100, 1000, 100),
    ] {
        let circuits = create_circuit_batch(branches, mults);
        let circuit = circuits.get(0).unwrap().clone();

        // Generate CHAINED inputs for soldering
        let chained_inputs = generate_chained_inputs(&circuit, reps, 3, 2, MODULUS);

        // Throughput = total AND gates = reps × branches × circuit_size
        let total_and_gates = (reps * branches * mults) as u64;
        group.throughput(Throughput::Elements(total_and_gates));

        // Benchmark with manual soldering verification
        group.bench_with_input(
            BenchmarkId::new("manual_verification", format!("{}b_{}m_{}r", branches, mults, reps)),
            &(circuits.clone(), circuit.clone(), chained_inputs.clone(), reps),
            |b, (circuits, circuit, inputs, reps)| {
                b.iter(|| {
                    let global_key = GlobalKey::<Field64>::generate(&mut rng);
                    let chi: u64 = rng.random_range(1..MODULUS);
                    let rho: u64 = rng.random_range(1..MODULUS);

                    // Evaluate and commit with chained inputs
                    let mut witnesses = Vec::with_capacity(*reps);
                    let mut all_commitments = Vec::with_capacity(*reps);

                    for rep in 0..*reps {
                        let mut c = circuit.clone();
                        let witness = c.evaluate(&inputs[rep], MODULUS);
                        let commit = commit_witness(&global_key, &witness, &mut rng);
                        witnesses.push(witness);
                        all_commitments.push(commit);
                    }

                    // Topology and hash
                    let topology_vectors = circuits.topology_vectors(chi, MODULUS);
                    let _hash = UniversalHash::compute(&topology_vectors, rho, MODULUS);

                    // Verification with soldering
                    let mut valid = true;
                    for commitments in &all_commitments {
                        valid &= verify_commitments(&global_key, commitments);
                    }
                    for (witness, commitments) in witnesses.iter().zip(all_commitments.iter()) {
                        valid &= verify_multiplication_constraints(&global_key, witness, commitments);
                    }

                    // Verify soldering constraint
                    for j in 1..*reps {
                        valid &= witnesses[j - 1].mult_outputs[0] == inputs[j][0];
                    }

                    black_box(valid)
                });
            },
        );
    }

    // Benchmark using run_protocol_with_soldering at different scales
    for reps in [10usize, 50, 100] {
        let circuits = create_circuit_batch(10, 100); // Fixed small circuit for protocol benchmark
        let circuit = circuits.get(0).unwrap();
        let chained_inputs = generate_chained_inputs(circuit, reps, 3, 2, MODULUS);

        group.throughput(Throughput::Elements((reps * 10 * 100) as u64));

        // We need to use different const generic values
        match reps {
            10 => {
                let inputs_10: Vec<Vec<u64>> = chained_inputs.clone();
                group.bench_function(
                    BenchmarkId::new("run_protocol_with_soldering", format!("{}reps", reps)),
                    |b| {
                        b.iter(|| {
                            black_box(run_protocol_with_soldering::<10>(
                                &circuits,
                                0,
                                &inputs_10,
                                &[soldering_constraint.clone()],
                                MODULUS,
                            ))
                        });
                    },
                );
            }
            50 => {
                let inputs_50: Vec<Vec<u64>> = chained_inputs.clone();
                group.bench_function(
                    BenchmarkId::new("run_protocol_with_soldering", format!("{}reps", reps)),
                    |b| {
                        b.iter(|| {
                            black_box(run_protocol_with_soldering::<50>(
                                &circuits,
                                0,
                                &inputs_50,
                                &[soldering_constraint.clone()],
                                MODULUS,
                            ))
                        });
                    },
                );
            }
            100 => {
                let inputs_100: Vec<Vec<u64>> = chained_inputs.clone();
                group.bench_function(
                    BenchmarkId::new("run_protocol_with_soldering", format!("{}reps", reps)),
                    |b| {
                        b.iter(|| {
                            black_box(run_protocol_with_soldering::<100>(
                                &circuits,
                                0,
                                &inputs_100,
                                &[soldering_constraint.clone()],
                                MODULUS,
                            ))
                        });
                    },
                );
            }
            _ => {}
        }
    }

    group.finish();
}

fn bench_soldering(c: &mut Criterion) {
    use mpz_justvengers::soldering::{
        generate_masking_pair, verify_soldering_constraint, compute_masked_poly,
    };

    let mut group = c.benchmark_group("soldering_primitives");

    let mut rng = Prg::from_seed(Block::ZERO);

    // Benchmark masking polynomial generation at different scales
    for num_reps in [4, 10, 50, 100] {
        let eval_points: Vec<u64> = (1..=num_reps as u64).collect();

        group.throughput(Throughput::Elements(num_reps as u64));
        group.bench_with_input(
            BenchmarkId::new("generate_masking_pair", num_reps),
            &eval_points,
            |b, points| {
                b.iter(|| {
                    black_box(generate_masking_pair(points, 2, MODULUS, &mut rng))
                });
            },
        );
    }

    // Benchmark computing masked polynomial (f = φ·g + r)
    for num_reps in [4, 10, 50, 100] {
        let eval_points: Vec<u64> = (1..=num_reps as u64).collect();
        let (r1, _r2) = generate_masking_pair(&eval_points, 2, MODULUS, &mut rng);
        let g_coeffs: Vec<u64> = (0..num_reps).map(|i| i as u64 * 7 % MODULUS).collect();
        let phi: u64 = rng.random_range(1..MODULUS);

        group.throughput(Throughput::Elements(num_reps as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_masked_poly", num_reps),
            &(phi, g_coeffs.clone(), r1.clone()),
            |b, (phi, g, r)| {
                b.iter(|| {
                    black_box(compute_masked_poly(*phi, g, r, MODULUS))
                });
            },
        );
    }

    // Benchmark soldering verification
    for num_reps in [4, 10, 50, 100] {
        let eval_points: Vec<u64> = (1..=num_reps as u64).collect();
        let (r1, r2) = generate_masking_pair(&eval_points, 2, MODULUS, &mut rng);

        group.throughput(Throughput::Elements(num_reps as u64));
        group.bench_with_input(
            BenchmarkId::new("verify_constraint", num_reps),
            &(r1, r2, eval_points),
            |b, (f1, f2, points)| {
                b.iter(|| {
                    black_box(verify_soldering_constraint(f1, f2, points, 2, MODULUS))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_circuit_creation,
    bench_circuit_evaluation,
    bench_itmac_commitment,
    bench_topology_vectors,
    bench_universal_hash,
    bench_vole_pool,
    bench_itpac_interpolation,
    bench_e2e_full_protocol,
    bench_e2e_scaling,
    bench_soldering,
);

criterion_main!(benches);
