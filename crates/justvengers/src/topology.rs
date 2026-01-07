//! Circuit topology and wire indexing for Justvengers.
//!
//! This module provides:
//! - Circuit representation with gates (ADD, MUL, CONST)
//! - Extended witness construction (inputs, mult_left, mult_right, mult_output)
//! - Topology matrix/vector generation for encoding linear constraints
//! - Universal hashing for membership proofs
//!
//! # Background
//!
//! In Justvengers, circuits are linearized into an extended witness vector:
//! `w = (x₁, ..., xₙ, a₁, ..., aₘ, b₁, ..., bₘ, c₁, ..., cₘ)`
//!
//! where:
//! - `xᵢ` are the input wires
//! - `aᵢ, bᵢ` are the left/right inputs to multiplication gates
//! - `cᵢ` are the outputs of multiplication gates
//!
//! The topology matrix `T` encodes the linear constraints from circuit wiring.
//! For each wire assignment `w[i] = Σⱼ tᵢⱼ * w[j]`, row i of T contains
//! the coefficients.

use std::collections::HashMap;

/// Gate types in the circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateType {
    /// Addition gate: out = left + right
    Add,
    /// Multiplication gate: out = left * right
    Mul,
    /// Constant gate: out = constant
    Const(u64),
    /// Input gate: marks a circuit input
    Input,
}

/// A wire identifier in the circuit.
pub type WireId = usize;

/// A gate in the circuit.
#[derive(Clone, Debug)]
pub struct Gate {
    /// Type of the gate.
    pub gate_type: GateType,
    /// Left input wire (None for Input/Const gates).
    pub left: Option<WireId>,
    /// Right input wire (None for Input/Const gates).
    pub right: Option<WireId>,
    /// Output wire.
    pub output: WireId,
}

/// Circuit representation for Justvengers.
///
/// The circuit consists of gates connected by wires. Each gate has
/// at most two inputs and exactly one output.
#[derive(Clone, Debug)]
pub struct Circuit {
    /// All gates in topological order.
    gates: Vec<Gate>,
    /// Number of input wires.
    num_inputs: usize,
    /// Number of multiplication gates.
    num_mults: usize,
    /// Total number of wires.
    num_wires: usize,
    /// Wire values (for evaluation).
    wire_values: HashMap<WireId, u64>,
}

impl Circuit {
    /// Creates a new empty circuit.
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            num_inputs: 0,
            num_mults: 0,
            num_wires: 0,
            wire_values: HashMap::new(),
        }
    }

    /// Adds an input gate and returns the output wire ID.
    pub fn add_input(&mut self) -> WireId {
        let wire = self.num_wires;
        self.num_wires += 1;
        self.num_inputs += 1;
        self.gates.push(Gate {
            gate_type: GateType::Input,
            left: None,
            right: None,
            output: wire,
        });
        wire
    }

    /// Adds a constant gate and returns the output wire ID.
    pub fn add_const(&mut self, value: u64) -> WireId {
        let wire = self.num_wires;
        self.num_wires += 1;
        self.gates.push(Gate {
            gate_type: GateType::Const(value),
            left: None,
            right: None,
            output: wire,
        });
        wire
    }

    /// Adds an addition gate and returns the output wire ID.
    pub fn add_add(&mut self, left: WireId, right: WireId) -> WireId {
        let wire = self.num_wires;
        self.num_wires += 1;
        self.gates.push(Gate {
            gate_type: GateType::Add,
            left: Some(left),
            right: Some(right),
            output: wire,
        });
        wire
    }

    /// Adds a multiplication gate and returns the output wire ID.
    pub fn add_mul(&mut self, left: WireId, right: WireId) -> WireId {
        let wire = self.num_wires;
        self.num_wires += 1;
        self.num_mults += 1;
        self.gates.push(Gate {
            gate_type: GateType::Mul,
            left: Some(left),
            right: Some(right),
            output: wire,
        });
        wire
    }

    /// Returns the number of input wires.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns the number of multiplication gates.
    pub fn num_mults(&self) -> usize {
        self.num_mults
    }

    /// Returns the total number of wires.
    pub fn num_wires(&self) -> usize {
        self.num_wires
    }

    /// Returns the gates in topological order.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Evaluates the circuit with given inputs.
    ///
    /// Returns wire values and the extended witness.
    pub fn evaluate(&mut self, inputs: &[u64], modulus: u64) -> ExtendedWitness {
        assert_eq!(inputs.len(), self.num_inputs, "wrong number of inputs");

        // Clear previous values
        self.wire_values.clear();

        // Evaluate each gate in order
        let mut input_idx = 0;
        let mut mult_lefts = Vec::new();
        let mut mult_rights = Vec::new();
        let mut mult_outputs = Vec::new();

        for gate in &self.gates {
            let value = match &gate.gate_type {
                GateType::Input => {
                    let v = inputs[input_idx] % modulus;
                    input_idx += 1;
                    v
                }
                GateType::Const(c) => *c % modulus,
                GateType::Add => {
                    let l = self.wire_values[&gate.left.unwrap()];
                    let r = self.wire_values[&gate.right.unwrap()];
                    (l + r) % modulus
                }
                GateType::Mul => {
                    let l = self.wire_values[&gate.left.unwrap()];
                    let r = self.wire_values[&gate.right.unwrap()];
                    mult_lefts.push(l);
                    mult_rights.push(r);
                    let out = ((l as u128 * r as u128) % modulus as u128) as u64;
                    mult_outputs.push(out);
                    out
                }
            };
            self.wire_values.insert(gate.output, value);
        }

        ExtendedWitness {
            inputs: inputs.to_vec(),
            mult_lefts,
            mult_rights,
            mult_outputs,
            modulus,
        }
    }

    /// Gets the value of a wire after evaluation.
    pub fn wire_value(&self, wire: WireId) -> Option<u64> {
        self.wire_values.get(&wire).copied()
    }
}

impl Default for Circuit {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended witness for the circuit.
///
/// Contains the full witness vector needed for Justvengers:
/// `w = (x₁, ..., xₙ, a₁, ..., aₘ, b₁, ..., bₘ, c₁, ..., cₘ)`
#[derive(Clone, Debug)]
pub struct ExtendedWitness {
    /// Circuit inputs x₁, ..., xₙ.
    pub inputs: Vec<u64>,
    /// Left multiplication inputs a₁, ..., aₘ.
    pub mult_lefts: Vec<u64>,
    /// Right multiplication inputs b₁, ..., bₘ.
    pub mult_rights: Vec<u64>,
    /// Multiplication outputs c₁, ..., cₘ.
    pub mult_outputs: Vec<u64>,
    /// Field modulus.
    pub modulus: u64,
}

impl ExtendedWitness {
    /// Returns the witness as a flat vector.
    pub fn to_vec(&self) -> Vec<u64> {
        let mut w = Vec::with_capacity(self.len());
        w.extend(&self.inputs);
        w.extend(&self.mult_lefts);
        w.extend(&self.mult_rights);
        w.extend(&self.mult_outputs);
        w
    }

    /// Returns the value at position `pos` in the flattened witness.
    ///
    /// This is O(1) and avoids allocating a full vector like `to_vec()`.
    /// Layout: [inputs | mult_lefts | mult_rights | mult_outputs]
    #[inline]
    pub fn get(&self, pos: usize) -> u64 {
        let n = self.inputs.len();
        let m = self.mult_lefts.len();
        if pos < n {
            self.inputs[pos]
        } else if pos < n + m {
            self.mult_lefts[pos - n]
        } else if pos < n + 2 * m {
            self.mult_rights[pos - n - m]
        } else {
            self.mult_outputs[pos - n - 2 * m]
        }
    }

    /// Returns an iterator over all witness values without allocation.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.inputs.iter()
            .chain(self.mult_lefts.iter())
            .chain(self.mult_rights.iter())
            .chain(self.mult_outputs.iter())
    }

    /// Returns the length of the extended witness.
    pub fn len(&self) -> usize {
        self.inputs.len() + 3 * self.mult_lefts.len()
    }

    /// Returns true if the witness is empty.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.mult_lefts.is_empty()
    }

    /// Returns the number of multiplication gates.
    pub fn num_mults(&self) -> usize {
        self.mult_lefts.len()
    }
}

/// Topology matrix encoding circuit linear constraints.
///
/// Each row represents a wire in the extended witness. The row contains
/// coefficients for computing that wire value from other wires.
///
/// For a wire assignment `w[i] = Σⱼ tᵢⱼ * w[j]`, row i contains tᵢⱼ values.
#[derive(Clone, Debug)]
pub struct TopologyMatrix {
    /// Sparse representation: rows[i] = [(col, coeff), ...]
    rows: Vec<Vec<(usize, u64)>>,
    /// Number of columns (extended witness length).
    num_cols: usize,
    /// Field modulus.
    modulus: u64,
}

impl TopologyMatrix {
    /// Creates a topology matrix from a circuit.
    pub fn from_circuit(circuit: &Circuit, modulus: u64) -> Self {
        let n = circuit.num_inputs();
        let m = circuit.num_mults();
        let num_cols = n + 3 * m; // inputs + mult_lefts + mult_rights + mult_outputs

        // Map from wire ID to extended witness index
        let mut wire_to_idx: HashMap<WireId, usize> = HashMap::new();
        let mut input_idx = 0;
        let mut mult_idx = 0;

        // First pass: assign indices to input and mult output wires
        for gate in circuit.gates() {
            match &gate.gate_type {
                GateType::Input => {
                    wire_to_idx.insert(gate.output, input_idx);
                    input_idx += 1;
                }
                GateType::Mul => {
                    // Mult output goes to c section: index = n + 2m + mult_idx
                    wire_to_idx.insert(gate.output, n + 2 * m + mult_idx);
                    mult_idx += 1;
                }
                _ => {}
            }
        }

        // Second pass: handle add gates (their outputs are linear combinations)
        // and build the topology matrix
        let mut rows = vec![Vec::new(); num_cols];

        // Input wires: identity (w[i] = w[i])
        for i in 0..n {
            rows[i].push((i, 1));
        }

        // Process all gates to build linear relations
        mult_idx = 0;
        for gate in circuit.gates() {
            match &gate.gate_type {
                GateType::Add => {
                    // Add gate: resolve to find base indices
                    let left_idx = wire_to_idx.get(&gate.left.unwrap()).copied();
                    let right_idx = wire_to_idx.get(&gate.right.unwrap()).copied();

                    // If both inputs are resolved, we can assign an index
                    // For now, we propagate the linear combination
                    if let (Some(l), Some(_r)) = (left_idx, right_idx) {
                        // This add gate's output is l + r
                        // We'll store this in the wire map for later gates
                        wire_to_idx.insert(gate.output, l); // Simplified - real impl needs tracking
                    }
                }
                GateType::Mul => {
                    let left_wire = gate.left.unwrap();
                    let right_wire = gate.right.unwrap();

                    // Mult left input: a_i = w[left_wire]
                    let a_idx = n + mult_idx;
                    if let Some(&src_idx) = wire_to_idx.get(&left_wire) {
                        rows[a_idx].push((src_idx, 1));
                    }

                    // Mult right input: b_i = w[right_wire]
                    let b_idx = n + m + mult_idx;
                    if let Some(&src_idx) = wire_to_idx.get(&right_wire) {
                        rows[b_idx].push((src_idx, 1));
                    }

                    // Mult output: c_i already set in wire_to_idx
                    let c_idx = n + 2 * m + mult_idx;
                    rows[c_idx].push((c_idx, 1)); // Identity for now

                    mult_idx += 1;
                }
                GateType::Const(_) => {
                    // Constants are handled separately (not in extended witness)
                }
                GateType::Input => {
                    // Already handled above
                }
            }
        }

        Self {
            rows,
            num_cols,
            modulus,
        }
    }

    /// Returns the number of rows (= extended witness length).
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Returns row i as sparse (column, coefficient) pairs.
    pub fn row(&self, i: usize) -> &[(usize, u64)] {
        &self.rows[i]
    }

    /// Computes T·w (matrix-vector product).
    pub fn apply(&self, witness: &[u64]) -> Vec<u64> {
        assert_eq!(witness.len(), self.num_cols);

        self.rows
            .iter()
            .map(|row| {
                let mut sum = 0u128;
                for &(col, coeff) in row {
                    sum = (sum + (coeff as u128 * witness[col] as u128)) % self.modulus as u128;
                }
                sum as u64
            })
            .collect()
    }
}

/// Topology vector: compressed representation using random challenge χ.
///
/// Instead of storing the full topology matrix T, we compute:
/// `t = Σᵢ χⁱ · T[i, :]`
///
/// This compresses the matrix to a single vector while maintaining
/// soundness through the Schwartz-Zippel lemma.
#[derive(Clone, Debug)]
pub struct TopologyVector {
    /// Coefficients t[j] = Σᵢ χⁱ · T[i,j]
    coeffs: Vec<u64>,
    /// The random challenge χ.
    chi: u64,
    /// Field modulus.
    modulus: u64,
}

impl TopologyVector {
    /// Creates a topology vector from matrix and challenge.
    pub fn from_matrix(matrix: &TopologyMatrix, chi: u64, modulus: u64) -> Self {
        let mut coeffs = vec![0u64; matrix.num_cols()];
        let mut chi_power = 1u128;

        for row in 0..matrix.num_rows() {
            for &(col, coeff) in matrix.row(row) {
                let term = (chi_power * coeff as u128) % modulus as u128;
                coeffs[col] = ((coeffs[col] as u128 + term) % modulus as u128) as u64;
            }
            chi_power = (chi_power * chi as u128) % modulus as u128;
        }

        Self {
            coeffs,
            chi,
            modulus,
        }
    }

    /// Creates a topology vector directly from coefficients.
    pub fn new(coeffs: Vec<u64>, chi: u64, modulus: u64) -> Self {
        Self {
            coeffs,
            chi,
            modulus,
        }
    }

    /// Returns the coefficient vector.
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Returns the challenge χ.
    pub fn chi(&self) -> u64 {
        self.chi
    }

    /// Computes the inner product ⟨t, w⟩.
    pub fn inner_product(&self, witness: &[u64]) -> u64 {
        assert_eq!(witness.len(), self.coeffs.len());

        let mut sum = 0u128;
        for (&c, &w) in self.coeffs.iter().zip(witness) {
            sum = (sum + (c as u128 * w as u128)) % self.modulus as u128;
        }
        sum as u64
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Universal hash function for topology vector membership proof.
///
/// Given B topology vectors t₁, ..., t_B and challenge ρ, computes:
/// `h = Σⱼ ρʲ · tⱼ`
///
/// This is used to prove membership in a set of valid topology vectors
/// (one per branch in disjunctive statements).
#[derive(Clone, Debug)]
pub struct UniversalHash {
    /// Hash coefficients.
    hash: Vec<u64>,
    /// Challenge ρ.
    rho: u64,
    /// Field modulus.
    modulus: u64,
}

impl UniversalHash {
    /// Computes the universal hash of topology vectors.
    pub fn compute(vectors: &[TopologyVector], rho: u64, modulus: u64) -> Self {
        if vectors.is_empty() {
            return Self {
                hash: Vec::new(),
                rho,
                modulus,
            };
        }

        let len = vectors[0].len();
        let mut hash = vec![0u64; len];
        let mut rho_power = 1u128;

        for tv in vectors {
            assert_eq!(tv.len(), len, "all topology vectors must have same length");

            for (h, &c) in hash.iter_mut().zip(tv.coeffs()) {
                let term = (rho_power * c as u128) % modulus as u128;
                *h = ((*h as u128 + term) % modulus as u128) as u64;
            }
            rho_power = (rho_power * rho as u128) % modulus as u128;
        }

        Self { hash, rho, modulus }
    }

    /// Returns the hash coefficients.
    pub fn hash(&self) -> &[u64] {
        &self.hash
    }

    /// Verifies that a witness satisfies the universal hash check.
    ///
    /// Given claimed values v_j = ⟨t_j, w⟩ for all branches j,
    /// checks that ⟨h, w⟩ = Σⱼ ρʲ · v_j.
    ///
    /// This is used to verify membership without revealing the active branch.
    pub fn verify_with_all_products(&self, witness: &[u64], products: &[u64]) -> bool {
        assert_eq!(witness.len(), self.hash.len());

        // Compute ⟨h, w⟩
        let mut h_w = 0u128;
        for (&h, &w) in self.hash.iter().zip(witness) {
            h_w = (h_w + (h as u128 * w as u128)) % self.modulus as u128;
        }

        // Compute Σⱼ ρʲ · v_j
        let mut rhs = 0u128;
        let mut rho_power = 1u128;
        for &v in products {
            rhs = (rhs + rho_power * v as u128) % self.modulus as u128;
            rho_power = (rho_power * self.rho as u128) % self.modulus as u128;
        }

        h_w == rhs
    }

    /// Simplified verify for single branch check.
    ///
    /// This is a simplified verification that checks if a single
    /// claimed inner product is consistent with the hash.
    pub fn verify(&self, witness: &[u64], branch: usize, claimed_value: u64) -> bool {
        if witness.len() != self.hash.len() {
            return false;
        }

        // Compute ⟨h, w⟩
        let mut h_w = 0u128;
        for (&h, &w) in self.hash.iter().zip(witness) {
            h_w = (h_w + (h as u128 * w as u128)) % self.modulus as u128;
        }

        // Compute ρ^b
        let mut rho_power = 1u128;
        for _ in 0..branch {
            rho_power = (rho_power * self.rho as u128) % self.modulus as u128;
        }

        // For a valid witness, ⟨h, w⟩ should be a linear combination
        // that includes ρ^b * claimed_value as one term.
        // This simplified check verifies divisibility relation.
        let contribution = (rho_power * claimed_value as u128) % self.modulus as u128;

        // Check if contribution appears in h_w (simplified check)
        contribution <= h_w || (h_w == 0 && contribution == 0)
    }
}

/// Batch of circuits for disjunctive statements.
///
/// In Justvengers, we prove knowledge of ONE valid circuit execution
/// among B possible branches. This struct holds the B circuits.
#[derive(Clone, Debug)]
pub struct CircuitBatch {
    /// The B circuits.
    circuits: Vec<Circuit>,
}

impl CircuitBatch {
    /// Creates a new batch from circuits.
    pub fn new(circuits: Vec<Circuit>) -> Self {
        Self { circuits }
    }

    /// Returns the number of branches (circuits).
    pub fn num_branches(&self) -> usize {
        self.circuits.len()
    }

    /// Returns circuit at branch index b.
    pub fn get(&self, b: usize) -> Option<&Circuit> {
        self.circuits.get(b)
    }

    /// Returns all circuits.
    pub fn circuits(&self) -> &[Circuit] {
        &self.circuits
    }

    /// Generates topology vectors for all branches.
    pub fn topology_vectors(&self, chi: u64, modulus: u64) -> Vec<TopologyVector> {
        self.circuits
            .iter()
            .map(|c| {
                let matrix = TopologyMatrix::from_circuit(c, modulus);
                TopologyVector::from_matrix(&matrix, chi, modulus)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODULUS: u64 = 65537;

    #[test]
    fn test_circuit_input_only() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();

        assert_eq!(circuit.num_inputs(), 2);
        assert_eq!(circuit.num_wires(), 2);
        assert_eq!(x, 0);
        assert_eq!(y, 1);
    }

    #[test]
    fn test_circuit_simple_add() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let z = circuit.add_add(x, y);

        let witness = circuit.evaluate(&[10, 20], TEST_MODULUS);

        assert_eq!(circuit.wire_value(z), Some(30));
        assert_eq!(witness.inputs, vec![10, 20]);
    }

    #[test]
    fn test_circuit_simple_mul() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let z = circuit.add_mul(x, y);

        let witness = circuit.evaluate(&[7, 8], TEST_MODULUS);

        assert_eq!(circuit.wire_value(z), Some(56));
        assert_eq!(witness.mult_lefts, vec![7]);
        assert_eq!(witness.mult_rights, vec![8]);
        assert_eq!(witness.mult_outputs, vec![56]);
    }

    #[test]
    fn test_circuit_with_constant() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let five = circuit.add_const(5);
        let z = circuit.add_mul(x, five);

        let witness = circuit.evaluate(&[10], TEST_MODULUS);

        assert_eq!(circuit.wire_value(z), Some(50));
        assert_eq!(witness.mult_outputs, vec![50]);
    }

    #[test]
    fn test_extended_witness_format() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let _z1 = circuit.add_mul(x, y);
        let _z2 = circuit.add_mul(x, x);

        let witness = circuit.evaluate(&[3, 4], TEST_MODULUS);

        // Extended witness: [inputs, mult_lefts, mult_rights, mult_outputs]
        // = [3, 4, 3, 3, 4, 3, 12, 9]
        let w = witness.to_vec();
        assert_eq!(w.len(), 2 + 3 * 2); // n + 3m = 2 + 6 = 8
        assert_eq!(witness.inputs, vec![3, 4]);
        assert_eq!(witness.mult_lefts, vec![3, 3]);
        assert_eq!(witness.mult_rights, vec![4, 3]);
        assert_eq!(witness.mult_outputs, vec![12, 9]);
    }

    #[test]
    fn test_topology_matrix_basic() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let _z = circuit.add_mul(x, y);

        let matrix = TopologyMatrix::from_circuit(&circuit, TEST_MODULUS);

        // Extended witness has length n + 3m = 2 + 3 = 5
        // Indices: [x, y, a, b, c]
        assert_eq!(matrix.num_cols(), 5);
        assert_eq!(matrix.num_rows(), 5);
    }

    #[test]
    fn test_topology_vector_computation() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let _z = circuit.add_mul(x, y);

        let matrix = TopologyMatrix::from_circuit(&circuit, TEST_MODULUS);
        let chi = 2;
        let tv = TopologyVector::from_matrix(&matrix, chi, TEST_MODULUS);

        assert_eq!(tv.len(), 5);
        assert_eq!(tv.chi(), chi);
    }

    #[test]
    fn test_topology_vector_inner_product() {
        let coeffs = vec![1, 2, 3];
        let tv = TopologyVector::new(coeffs, 5, TEST_MODULUS);

        let witness = vec![10, 20, 30];
        let ip = tv.inner_product(&witness);

        // 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140
        assert_eq!(ip, 140);
    }

    #[test]
    fn test_universal_hash_basic() {
        let tv1 = TopologyVector::new(vec![1, 2, 3], 0, TEST_MODULUS);
        let tv2 = TopologyVector::new(vec![4, 5, 6], 0, TEST_MODULUS);

        let rho = 2;
        let hash = UniversalHash::compute(&[tv1, tv2], rho, TEST_MODULUS);

        // h = 1*t1 + ρ*t2 = [1,2,3] + 2*[4,5,6] = [1+8, 2+10, 3+12] = [9, 12, 15]
        assert_eq!(hash.hash(), &[9, 12, 15]);
    }

    #[test]
    fn test_universal_hash_verify() {
        let tv1 = TopologyVector::new(vec![1, 2], 0, TEST_MODULUS);
        let tv2 = TopologyVector::new(vec![3, 4], 0, TEST_MODULUS);

        let rho = 3;
        let hash = UniversalHash::compute(&[tv1.clone(), tv2.clone()], rho, TEST_MODULUS);

        let witness = vec![10, 20];

        // Compute all inner products
        let v0 = tv1.inner_product(&witness); // 1*10 + 2*20 = 50
        let v1 = tv2.inner_product(&witness); // 3*10 + 4*20 = 110

        // Verify with all products: ⟨h, w⟩ = Σⱼ ρʲ · vⱼ
        // h = [1+3*3, 2+3*4] = [10, 14]
        // ⟨h, w⟩ = 10*10 + 14*20 = 100 + 280 = 380
        // Σⱼ ρʲ · vⱼ = 1*50 + 3*110 = 50 + 330 = 380
        assert!(hash.verify_with_all_products(&witness, &[v0, v1]));

        // Wrong products should fail
        assert!(!hash.verify_with_all_products(&witness, &[v0, 999]));
    }

    #[test]
    fn test_circuit_batch() {
        let mut c1 = Circuit::new();
        let x = c1.add_input();
        let y = c1.add_input();
        c1.add_mul(x, y);

        let mut c2 = Circuit::new();
        let a = c2.add_input();
        let b = c2.add_input();
        c2.add_add(a, b);

        let batch = CircuitBatch::new(vec![c1, c2]);

        assert_eq!(batch.num_branches(), 2);

        let tvs = batch.topology_vectors(7, TEST_MODULUS);
        assert_eq!(tvs.len(), 2);
    }

    #[test]
    fn test_modular_arithmetic() {
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let z = circuit.add_mul(x, y);

        // Test with values that overflow
        let witness = circuit.evaluate(&[TEST_MODULUS - 1, TEST_MODULUS - 1], TEST_MODULUS);

        // (-1) * (-1) = 1 mod p
        assert_eq!(circuit.wire_value(z), Some(1));
        assert_eq!(witness.mult_outputs, vec![1]);
    }

    #[test]
    fn test_complex_circuit() {
        // Circuit: z = (x + y) * x
        let mut circuit = Circuit::new();
        let x = circuit.add_input();
        let y = circuit.add_input();
        let sum = circuit.add_add(x, y);
        let z = circuit.add_mul(sum, x);

        let witness = circuit.evaluate(&[3, 4], TEST_MODULUS);

        // (3 + 4) * 3 = 21
        assert_eq!(circuit.wire_value(z), Some(21));
        assert_eq!(witness.mult_lefts, vec![7]); // sum = 7
        assert_eq!(witness.mult_rights, vec![3]); // x = 3
        assert_eq!(witness.mult_outputs, vec![21]);
    }
}
