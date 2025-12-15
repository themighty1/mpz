//! Evaluator for Three Halves Scheme
//!
//! This module implements the evaluation function for circuits using the
//! Three Halves technique from Rosulek & Roy 2021.

use core::fmt;
use std::ops::Range;

use mpz_circuits::{Circuit, Gate};
use mpz_core::{
    Block,
    aes::{FIXED_KEY_AES, FixedKeyAes},
};
use mpz_memory_core::correlated::Mac;

use super::{
    control::{expand_marginal, extract_r_p_marginal},
    garbler::{ControlBits, EncryptedGate, EncryptedGateBatch, ThreeHalvesGate, xor_assign_8},
    matrices::{M, V},
    slicing::SlicedLabel,
};

use crate::DEFAULT_BATCH_SIZE;

/// Errors that can occur during garbled circuit evaluation.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum EvaluatorError {
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("evaluator not finished")]
    NotFinished,
}

/// Output of the evaluator.
#[derive(Debug)]
pub struct EvaluatorOutput {
    /// Output MACs of the circuit.
    pub outputs: Vec<Mac>,
}

/// Evaluator for Three Halves scheme.
#[derive(Debug, Default)]
pub struct Evaluator {
    /// Buffer for the active labels.
    buffer: Vec<Block>,
}

impl Evaluator {
    /// Creates a new evaluator with a buffer of the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Returns a consumer over the encrypted gates of a circuit.
    ///
    /// # Arguments
    ///
    /// * `circ` - The circuit to evaluate.
    /// * `inputs` - The input labels to the circuit.
    pub fn evaluate<'a>(
        &'a mut self,
        circ: &'a Circuit,
        inputs: &[Mac],
    ) -> Result<EncryptedGateConsumer<'a, std::slice::Iter<'a, Gate>>, EvaluatorError> {
        if inputs.len() != circ.inputs().len() {
            return Err(EvaluatorError::InputLength {
                expected: circ.inputs().len(),
                actual: inputs.len(),
            });
        }

        // Expand the buffer to fit the circuit
        if circ.feed_count() > self.buffer.len() {
            self.buffer.resize(circ.feed_count(), Default::default());
        }

        self.buffer[..inputs.len()].copy_from_slice(Mac::as_blocks(inputs));

        Ok(EncryptedGateConsumer::new(
            circ.gates().iter(),
            &mut self.buffer,
            circ.and_count(),
            circ.outputs(),
        ))
    }

    /// Returns a consumer over batched encrypted gates of a circuit.
    ///
    /// # Arguments
    ///
    /// * `circ` - The circuit to evaluate.
    /// * `inputs` - The input labels to the circuit.
    pub fn evaluate_batched<'a>(
        &'a mut self,
        circ: &'a Circuit,
        inputs: &[Mac],
    ) -> Result<EncryptedGateBatchConsumer<'a, std::slice::Iter<'a, Gate>>, EvaluatorError> {
        self.evaluate(circ, inputs).map(EncryptedGateBatchConsumer)
    }
}

/// Consumer over the encrypted gates of a circuit.
pub struct EncryptedGateConsumer<'a, I: Iterator> {
    /// Cipher to use to evaluate the gates.
    cipher: &'static FixedKeyAes,
    /// Buffer for the active labels.
    labels: &'a mut [Block],
    /// Iterator over the gates.
    gates: I,
    /// Current gate id.
    gid: usize,
    /// Number of AND gates evaluated.
    counter: usize,
    /// Total number of AND gates in the circuit.
    and_count: usize,
    /// Range of the outputs in the buffer.
    outputs: Range<usize>,
    /// Whether the entire circuit has been evaluated.
    complete: bool,
}

impl<I: Iterator> fmt::Debug for EncryptedGateConsumer<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EncryptedGateConsumer {{ .. }}")
    }
}

impl<'a, I> EncryptedGateConsumer<'a, I>
where
    I: Iterator<Item = &'a Gate>,
{
    fn new(gates: I, labels: &'a mut [Block], and_count: usize, outputs: Range<usize>) -> Self {
        Self {
            cipher: &(*FIXED_KEY_AES),
            gates,
            labels,
            gid: 1,
            counter: 0,
            and_count,
            outputs,
            complete: false,
        }
    }

    /// Returns `true` if the evaluator wants more encrypted gates.
    #[inline]
    pub fn wants_gates(&self) -> bool {
        self.counter != self.and_count
    }

    /// Evaluates the next encrypted gate in the circuit.
    #[inline]
    pub fn next(&mut self, encrypted_gate: EncryptedGate) {
        while let Some(gate) = self.gates.next() {
            match gate {
                Gate::Xor {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x = self.labels[node_x.id()];
                    let y = self.labels[node_y.id()];
                    self.labels[node_z.id()] = x ^ y;
                }
                Gate::And {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x = self.labels[node_x.id()];
                    let y = self.labels[node_y.id()];
                    let z = and_gate(
                        self.cipher,
                        &x,
                        &y,
                        &encrypted_gate.gate,
                        &encrypted_gate.control_bits,
                        self.gid,
                    );
                    self.labels[node_z.id()] = z;

                    self.gid += 1;
                    self.counter += 1;

                    // If we have more AND gates to evaluate, return.
                    if self.wants_gates() {
                        return;
                    }
                }
                Gate::Inv {
                    x: node_x,
                    z: node_z,
                } => {
                    let x = self.labels[node_x.id()];
                    self.labels[node_z.id()] = x;
                }
                Gate::Id {
                    x: node_x,
                    z: node_z,
                } => {
                    let x = self.labels[node_x.id()];
                    self.labels[node_z.id()] = x;
                }
            }
        }

        self.complete = true;
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(mut self) -> Result<EvaluatorOutput, EvaluatorError> {
        if self.wants_gates() {
            return Err(EvaluatorError::NotFinished);
        }

        // If there were 0 AND gates, evaluate the "free" gates now.
        if !self.complete {
            self.next(Default::default());
        }

        Ok(EvaluatorOutput {
            outputs: Mac::from_blocks(self.labels[self.outputs.clone()].to_vec()),
        })
    }
}

/// Consumer returned by [`Evaluator::evaluate_batched`].
#[derive(Debug)]
pub struct EncryptedGateBatchConsumer<'a, I: Iterator, const N: usize = DEFAULT_BATCH_SIZE>(
    EncryptedGateConsumer<'a, I>,
);

impl<'a, I, const N: usize> EncryptedGateBatchConsumer<'a, I, N>
where
    I: Iterator<Item = &'a Gate>,
{
    /// Returns `true` if the evaluator wants more encrypted gates.
    pub fn wants_gates(&self) -> bool {
        self.0.wants_gates()
    }

    /// Evaluates the next batch of gates in the circuit.
    #[inline]
    pub fn next(&mut self, batch: EncryptedGateBatch<N>) {
        for encrypted_gate in batch.into_array() {
            self.0.next(encrypted_gate);
            if !self.0.wants_gates() {
                return;
            }
        }
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(self) -> Result<EvaluatorOutput, EvaluatorError> {
        self.0.finish()
    }
}

// ============================================================================
// Single gate evaluation (internal)
// ============================================================================

/// Evaluate a single AND gate using the Three Halves scheme.
#[inline]
fn and_gate(
    cipher: &FixedKeyAes,
    a: &Block,
    b: &Block,
    gate: &ThreeHalvesGate,
    control_bits: &ControlBits,
    gid: usize,
) -> Block {
    // Determine input combination (i, j) from pointer bits
    let i = a.lsb() as usize;
    let j = b.lsb() as usize;
    let ij = (i << 1) | j;

    // Compute the three hashes the evaluator has access to
    let tweak = Block::new((gid as u128).to_be_bytes());
    let mut hash_inputs = [*a, *b, *a ^ *b];
    cipher.rtccr_many(&[tweak; 3], &mut hash_inputs);

    let h_a = SlicedLabel::from_block(hash_inputs[0]);
    let h_b = SlicedLabel::from_block(hash_inputs[1]);
    let h_ab = SlicedLabel::from_block(hash_inputs[2]);

    // Slice the input labels
    let a_sliced = SlicedLabel::from_block(*a);
    let b_sliced = SlicedLabel::from_block(*b);

    // Expand compressed r_bar to full marginal (includes R_P for ODD mode)
    let marginal = expand_evaluator_marginal(&control_bits.r_bar, i, j);

    // Get the two rows for this input combination
    let row_l = 2 * ij;
    let row_r = 2 * ij + 1;

    // Compute contributions for each half
    let hash_contrib_l = compute_hash_contribution(row_l, &h_a, &h_b, &h_ab);
    let hash_contrib_r = compute_hash_contribution(row_r, &h_a, &h_b, &h_ab);

    let input_contrib_l = compute_input_contribution(&marginal, 0, &a_sliced, &b_sliced);
    let input_contrib_r = compute_input_contribution(&marginal, 1, &a_sliced, &b_sliced);

    let gate_contrib_l = compute_gate_contribution(row_l, gate);
    let gate_contrib_r = compute_gate_contribution(row_r, gate);

    // Combine all contributions
    let mut c_l = hash_contrib_l;
    xor_assign_8(&mut c_l, &input_contrib_l);
    xor_assign_8(&mut c_l, &gate_contrib_l);

    let mut c_r = hash_contrib_r;
    xor_assign_8(&mut c_r, &input_contrib_r);
    xor_assign_8(&mut c_r, &gate_contrib_r);

    SlicedLabel::new(c_l, c_r).to_block()
}

/// Expand evaluator's marginal from compressed r_bar.
fn expand_evaluator_marginal(r_bar: &[[bool; 2]; 4], i: usize, j: usize) -> [[u8; 4]; 2] {
    let ij = (i << 1) | j;
    let r_bar_ij = r_bar[ij];

    let mut marginal = expand_marginal(&r_bar_ij);

    // Add R_P's marginal (ODD mode)
    let r_p_marginal = extract_r_p_marginal(i, j);
    for row in 0..2 {
        for col in 0..4 {
            marginal[row][col] ^= r_p_marginal[row][col];
        }
    }

    marginal
}

/// Compute hash contribution for evaluation.
fn compute_hash_contribution(
    row: usize,
    h_a: &SlicedLabel,
    h_b: &SlicedLabel,
    h_ab: &SlicedLabel,
) -> [u8; 8] {
    let ij = row / 2;
    let i = ij >> 1;
    let j = ij & 1;

    let ab_col = if (i ^ j) == 0 { 4 } else { 5 };

    let mut result = [0u8; 8];

    if M[row][i] == 1 {
        xor_assign_8(&mut result, &h_a.left);
    }
    if M[row][2 + j] == 1 {
        xor_assign_8(&mut result, &h_b.left);
    }
    if M[row][ab_col] == 1 {
        xor_assign_8(&mut result, &h_ab.left);
    }

    result
}

/// Compute input contribution from expanded marginal.
fn compute_input_contribution(
    r_bar_expanded: &[[u8; 4]; 2],
    half: usize,
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

/// Compute gate ciphertext contribution.
fn compute_gate_contribution(row: usize, gate: &ThreeHalvesGate) -> [u8; 8] {
    let mut result = [0u8; 8];

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
    use super::super::garbler::{Garbler, GarblerOutput};
    use itybity::{FromBitIterator, IntoBitIterator, ToBits};
    use mpz_circuits::CircuitBuilder;
    use mpz_core::Block;
    use mpz_memory_core::correlated::{Delta, Key};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    // Test a circuit with chained AND gates: out = (a AND b) AND c
    #[test]
    fn test_chained_and_gates_circuit() {
        // Build the circuit once
        let mut builder = CircuitBuilder::new();
        let a = builder.add_input();
        let b = builder.add_input();
        let c = builder.add_input();
        let ab = builder.add_and_gate(a, b);
        let abc = builder.add_and_gate(ab, c);
        builder.add_output(abc);
        let circ = builder.build().unwrap();

        assert_eq!(circ.and_count(), 2);

        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Test all 8 input combinations with fresh state each time
        for (a_bit, b_bit, c_bit) in [
            (false, false, false),
            (false, false, true),
            (false, true, false),
            (false, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
        ] {
            // Fresh RNG for each test case
            let mut rng = ChaCha12Rng::seed_from_u64(42);
            let delta = Delta::random(&mut rng);
            // Three Halves requires input keys to have LSB = 0
            let input_keys: Vec<Key> = (0..3)
                .map(|_| {
                    let mut block: Block = rng.random();
                    block.set_lsb(false);
                    block.into()
                })
                .collect();

            let expected_output = a_bit && b_bit && c_bit;

            // First, garble the circuit to get input/output pairs
            let mut gb = Garbler::default();
            let mut gb_iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

            // Collect all encrypted gates
            let mut encrypted_gates = Vec::new();
            while let Some(gate) = gb_iter.next() {
                encrypted_gates.push(gate);
            }

            let GarblerOutput { inputs: input_pairs, outputs: output_pairs } = gb_iter.finish().unwrap();

            // Select input MACs from input pairs based on input bits
            let input_macs: Vec<Mac> = vec![
                if a_bit { input_pairs[0].1 } else { input_pairs[0].0 },
                if b_bit { input_pairs[1].1 } else { input_pairs[1].0 },
                if c_bit { input_pairs[2].1 } else { input_pairs[2].0 },
            ];

            // Now evaluate with the correct input MACs
            let mut ev = Evaluator::default();
            let mut ev_consumer = ev.evaluate(&circ, &input_macs).unwrap();

            for gate in encrypted_gates {
                ev_consumer.next(gate);
            }

            let EvaluatorOutput { outputs: output_macs } = ev_consumer.finish().unwrap();

            // Decode by checking which label the MAC matches
            let (false_label, true_label) = &output_pairs[0];
            let expected_mac = if expected_output { true_label } else { false_label };

            if &output_macs[0] == expected_mac {
                passed.push((a_bit, b_bit, c_bit, expected_output));
            } else {
                // Also check what value we actually got
                let actual_output = if &output_macs[0] == true_label {
                    Some(true)
                } else if &output_macs[0] == false_label {
                    Some(false)
                } else {
                    None // MAC doesn't match either label
                };
                failed.push((a_bit, b_bit, c_bit, expected_output, actual_output));
            }
        }

        // Report results
        println!("\n=== Chained AND gates test results ===");
        println!("PASSED ({}):", passed.len());
        for (a, b, c, out) in &passed {
            println!("  ({}, {}, {}) -> {} ✓", a, b, c, out);
        }
        println!("FAILED ({}):", failed.len());
        for (a, b, c, expected, actual) in &failed {
            let actual_str = match actual {
                Some(v) => format!("{}", v),
                None => "NO MATCH".to_string(),
            };
            println!("  ({}, {}, {}) -> expected {}, got {} ✗", a, b, c, expected, actual_str);
        }

        assert!(failed.is_empty(), "{} test cases failed", failed.len());
    }

    // Test a circuit with XOR then AND: out = (a XOR b) AND c
    #[test]
    fn test_xor_then_and_circuit() {
        let mut builder = CircuitBuilder::new();
        let a = builder.add_input();
        let b = builder.add_input();
        let c = builder.add_input();
        let ab_xor = builder.add_xor_gate(a, b);
        let result = builder.add_and_gate(ab_xor, c);
        builder.add_output(result);
        let circ = builder.build().unwrap();

        assert_eq!(circ.and_count(), 1);

        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for (a_bit, b_bit, c_bit) in [
            (false, false, false),
            (false, false, true),
            (false, true, false),
            (false, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
        ] {
            let mut rng = ChaCha12Rng::seed_from_u64(42);
            let delta = Delta::random(&mut rng);
            let input_keys: Vec<Key> = (0..3)
                .map(|_| {
                    let mut block: Block = rng.random();
                    block.set_lsb(false);
                    block.into()
                })
                .collect();

            // (a XOR b) AND c
            let expected_output = (a_bit ^ b_bit) && c_bit;

            // First, garble the circuit to get input/output pairs
            let mut gb = Garbler::default();
            let mut gb_iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

            let mut encrypted_gates = Vec::new();
            while let Some(gate) = gb_iter.next() {
                encrypted_gates.push(gate);
            }

            let GarblerOutput { inputs: input_pairs, outputs: output_pairs } = gb_iter.finish().unwrap();

            // Select input MACs from input pairs
            let input_macs: Vec<Mac> = vec![
                if a_bit { input_pairs[0].1 } else { input_pairs[0].0 },
                if b_bit { input_pairs[1].1 } else { input_pairs[1].0 },
                if c_bit { input_pairs[2].1 } else { input_pairs[2].0 },
            ];

            let mut ev = Evaluator::default();
            let mut ev_consumer = ev.evaluate(&circ, &input_macs).unwrap();

            for gate in encrypted_gates {
                ev_consumer.next(gate);
            }

            let EvaluatorOutput { outputs: output_macs } = ev_consumer.finish().unwrap();

            let (false_label, true_label) = &output_pairs[0];
            let expected_mac = if expected_output { true_label } else { false_label };

            if &output_macs[0] == expected_mac {
                passed.push((a_bit, b_bit, c_bit, expected_output));
            } else {
                let actual_output = if &output_macs[0] == true_label {
                    Some(true)
                } else if &output_macs[0] == false_label {
                    Some(false)
                } else {
                    None
                };
                failed.push((a_bit, b_bit, c_bit, expected_output, actual_output));
            }
        }

        println!("\n=== XOR then AND test results: (a XOR b) AND c ===");
        println!("PASSED ({}):", passed.len());
        for (a, b, c, out) in &passed {
            println!("  ({}, {}, {}) -> {} ✓", a, b, c, out);
        }
        println!("FAILED ({}):", failed.len());
        for (a, b, c, expected, actual) in &failed {
            let actual_str = match actual {
                Some(v) => format!("{}", v),
                None => "NO MATCH".to_string(),
            };
            println!("  ({}, {}, {}) -> expected {}, got {} ✗", a, b, c, expected, actual_str);
        }

        assert!(failed.is_empty(), "{} test cases failed", failed.len());
    }

    // Test 4 chained AND gates to find where failure occurs
    #[test]
    fn test_four_chained_and_gates() {
        let mut builder = CircuitBuilder::new();
        let a = builder.add_input();
        let b = builder.add_input();
        let c = builder.add_input();
        let d = builder.add_input();
        let e = builder.add_input();
        let ab = builder.add_and_gate(a, b);       // gate 1
        let abc = builder.add_and_gate(ab, c);     // gate 2
        let abcd = builder.add_and_gate(abc, d);   // gate 3
        let abcde = builder.add_and_gate(abcd, e); // gate 4
        // Output all intermediate results to see where it breaks
        builder.add_output(ab);
        builder.add_output(abc);
        builder.add_output(abcd);
        builder.add_output(abcde);
        let circ = builder.build().unwrap();

        assert_eq!(circ.and_count(), 4);

        println!("\n=== 4 Chained AND gates: a AND b AND c AND d AND e ===");
        println!("Format: (a,b,c,d,e) -> [ab, abc, abcd, abcde]");
        println!();

        let mut total_passed = 0;
        let mut total_failed = 0;

        for a_bit in [false, true] {
            for b_bit in [false, true] {
                for c_bit in [false, true] {
                    for d_bit in [false, true] {
                        for e_bit in [false, true] {
                            let mut rng = ChaCha12Rng::seed_from_u64(42);
                            let delta = Delta::random(&mut rng);
                            let input_keys: Vec<Key> = (0..5)
                                .map(|_| {
                                    let mut block: Block = rng.random();
                                    block.set_lsb(false);
                                    block.into()
                                })
                                .collect();

                            let expected = [
                                a_bit && b_bit,
                                a_bit && b_bit && c_bit,
                                a_bit && b_bit && c_bit && d_bit,
                                a_bit && b_bit && c_bit && d_bit && e_bit,
                            ];

                            // First, garble the circuit to get input/output pairs
                            let mut gb = Garbler::default();
                            let mut gb_iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

                            // Collect all encrypted gates
                            let mut encrypted_gates = Vec::new();
                            while let Some(gate) = gb_iter.next() {
                                encrypted_gates.push(gate);
                            }

                            let GarblerOutput { inputs: input_pairs, outputs: output_pairs } = gb_iter.finish().unwrap();

                            // Select input MACs from input pairs based on input bits
                            let input_macs: Vec<Mac> = vec![
                                if a_bit { input_pairs[0].1 } else { input_pairs[0].0 },
                                if b_bit { input_pairs[1].1 } else { input_pairs[1].0 },
                                if c_bit { input_pairs[2].1 } else { input_pairs[2].0 },
                                if d_bit { input_pairs[3].1 } else { input_pairs[3].0 },
                                if e_bit { input_pairs[4].1 } else { input_pairs[4].0 },
                            ];

                            // Now evaluate
                            let mut ev = Evaluator::default();
                            let mut ev_consumer = ev.evaluate(&circ, &input_macs).unwrap();

                            for gate in encrypted_gates {
                                ev_consumer.next(gate);
                            }
                            let EvaluatorOutput { outputs: output_macs } = ev_consumer.finish().unwrap();

                            let mut results = Vec::new();
                            let mut all_pass = true;
                            for i in 0..4 {
                                let (false_label, true_label) = &output_pairs[i];
                                let actual = if &output_macs[i] == true_label {
                                    Some(true)
                                } else if &output_macs[i] == false_label {
                                    Some(false)
                                } else {
                                    None
                                };
                                let pass = actual == Some(expected[i]);
                                if !pass {
                                    all_pass = false;
                                }
                                results.push((expected[i], actual, pass));
                            }

                            if all_pass {
                                total_passed += 1;
                            } else {
                                total_failed += 1;
                                print!("({},{},{},{},{}) -> ", a_bit as u8, b_bit as u8, c_bit as u8, d_bit as u8, e_bit as u8);
                                for (i, (exp, act, pass)) in results.iter().enumerate() {
                                    let gate_name = ["ab", "abc", "abcd", "abcde"][i];
                                    let act_str = match act {
                                        Some(v) => format!("{}", *v as u8),
                                        None => "?".to_string(),
                                    };
                                    let mark = if *pass { "✓" } else { "✗" };
                                    print!("{}:{}→{}{} ", gate_name, *exp as u8, act_str, mark);
                                }
                                println!();
                            }
                        }
                    }
                }
            }
        }

        println!();
        println!("Total: {} passed, {} failed", total_passed, total_failed);
        assert_eq!(total_failed, 0, "{} test cases failed", total_failed);
    }

    // Test a circuit with AND then XOR then AND: out = ((a AND b) XOR c) AND d
    #[test]
    fn test_and_xor_and_circuit() {
        let mut builder = CircuitBuilder::new();
        let a = builder.add_input();
        let b = builder.add_input();
        let c = builder.add_input();
        let d = builder.add_input();
        let ab_and = builder.add_and_gate(a, b);
        let abc_xor = builder.add_xor_gate(ab_and, c);
        let result = builder.add_and_gate(abc_xor, d);
        builder.add_output(result);
        let circ = builder.build().unwrap();

        assert_eq!(circ.and_count(), 2);

        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for a_bit in [false, true] {
            for b_bit in [false, true] {
                for c_bit in [false, true] {
                    for d_bit in [false, true] {
                        let mut rng = ChaCha12Rng::seed_from_u64(42);
                        let delta = Delta::random(&mut rng);
                        let input_keys: Vec<Key> = (0..4)
                            .map(|_| {
                                let mut block: Block = rng.random();
                                block.set_lsb(false);
                                block.into()
                            })
                            .collect();

                        // ((a AND b) XOR c) AND d
                        let expected_output = ((a_bit && b_bit) ^ c_bit) && d_bit;

                        // First, garble the circuit to get input/output pairs
                        let mut gb = Garbler::default();
                        let mut gb_iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

                        let mut encrypted_gates = Vec::new();
                        while let Some(gate) = gb_iter.next() {
                            encrypted_gates.push(gate);
                        }

                        let GarblerOutput { inputs: input_pairs, outputs: output_pairs } = gb_iter.finish().unwrap();

                        // Select input MACs from input pairs
                        let input_macs: Vec<Mac> = vec![
                            if a_bit { input_pairs[0].1 } else { input_pairs[0].0 },
                            if b_bit { input_pairs[1].1 } else { input_pairs[1].0 },
                            if c_bit { input_pairs[2].1 } else { input_pairs[2].0 },
                            if d_bit { input_pairs[3].1 } else { input_pairs[3].0 },
                        ];

                        let mut ev = Evaluator::default();
                        let mut ev_consumer = ev.evaluate(&circ, &input_macs).unwrap();

                        for gate in encrypted_gates {
                            ev_consumer.next(gate);
                        }

                        let EvaluatorOutput { outputs: output_macs } = ev_consumer.finish().unwrap();

                        let (false_label, true_label) = &output_pairs[0];
                        let expected_mac = if expected_output { true_label } else { false_label };

                        if &output_macs[0] == expected_mac {
                            passed.push((a_bit, b_bit, c_bit, d_bit, expected_output));
                        } else {
                            let actual_output = if &output_macs[0] == true_label {
                                Some(true)
                            } else if &output_macs[0] == false_label {
                                Some(false)
                            } else {
                                None
                            };
                            failed.push((a_bit, b_bit, c_bit, d_bit, expected_output, actual_output));
                        }
                    }
                }
            }
        }

        println!("\n=== AND-XOR-AND test results: ((a AND b) XOR c) AND d ===");
        println!("PASSED ({}):", passed.len());
        for (a, b, c, d, out) in &passed {
            println!("  ({}, {}, {}, {}) -> {} ✓", a, b, c, d, out);
        }
        println!("FAILED ({}):", failed.len());
        for (a, b, c, d, expected, actual) in &failed {
            let actual_str = match actual {
                Some(v) => format!("{}", v),
                None => "NO MATCH".to_string(),
            };
            println!("  ({}, {}, {}, {}) -> expected {}, got {} ✗", a, b, c, d, expected, actual_str);
        }

        assert!(failed.is_empty(), "{} test cases failed", failed.len());
    }

    // Test a simple single AND gate circuit
    #[test]
    fn test_single_and_gate_circuit() {
        // Build a simple AND gate circuit: output = input[0] AND input[1]
        let mut builder = CircuitBuilder::new();
        let a = builder.add_input();
        let b = builder.add_input();
        let c = builder.add_and_gate(a, b);
        builder.add_output(c);
        let circ = builder.build().unwrap();

        assert_eq!(circ.and_count(), 1);
        assert_eq!(circ.inputs().len(), 2);
        assert_eq!(circ.outputs().len(), 1);

        // Test all 4 input combinations
        for (a_bit, b_bit) in [(false, false), (false, true), (true, false), (true, true)] {
            // Fresh RNG for each test case
            let mut rng = ChaCha12Rng::seed_from_u64(42);
            let delta = Delta::random(&mut rng);

            // Three Halves requires input keys to have LSB = 0
            let input_keys: Vec<Key> = (0..2)
                .map(|_| {
                    let mut block: Block = rng.random();
                    block.set_lsb(false);
                    block.into()
                })
                .collect();

            let expected_output = a_bit && b_bit;

            // First, garble the circuit to get input/output pairs
            let mut gb = Garbler::default();
            let mut gb_iter = gb.generate(&circ, delta, &input_keys, &mut rng).unwrap();

            let mut encrypted_gates = Vec::new();
            while let Some(gate) = gb_iter.next() {
                encrypted_gates.push(gate);
            }

            let GarblerOutput { inputs: input_pairs, outputs: output_pairs } = gb_iter.finish().unwrap();

            // Select input MACs from input pairs based on input bits
            let input_macs: Vec<Mac> = vec![
                if a_bit { input_pairs[0].1 } else { input_pairs[0].0 },
                if b_bit { input_pairs[1].1 } else { input_pairs[1].0 },
            ];

            // Now evaluate with the correct input MACs
            let mut ev = Evaluator::default();
            let mut ev_consumer = ev.evaluate(&circ, &input_macs).unwrap();

            for gate in encrypted_gates {
                ev_consumer.next(gate);
            }

            let EvaluatorOutput { outputs: output_macs } = ev_consumer.finish().unwrap();

            assert_eq!(output_pairs.len(), 1);
            assert_eq!(output_macs.len(), 1);

            // Check that the output MAC matches the expected label
            let (false_label, true_label) = &output_pairs[0];
            let expected_mac = if expected_output { true_label } else { false_label };
            let other_mac = if expected_output { false_label } else { true_label };
            if &output_macs[0] == expected_mac {
                println!("PASSED for inputs ({}, {}), expected output {}", a_bit, b_bit, expected_output);
            } else {
                panic!(
                    "FAILED for inputs ({}, {}), expected output {}\n  actual:   {:?}\n  expected: {:?}\n  other:    {:?}",
                    a_bit, b_bit, expected_output, output_macs[0], expected_mac, other_mac
                );
            }
        }
    }

}
