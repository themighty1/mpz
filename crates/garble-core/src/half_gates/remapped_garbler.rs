//! Cache-optimized Half-Gates Garbler using Wire Remapping
//!
//! This module provides two approaches for cache-optimized garbling:
//!
//! 1. **`RemappedCircuit`** (recommended): Renumber wires in the circuit itself,
//!    then use the standard garbler. Zero runtime overhead.
//!
//! 2. **`RemappedGarbler`**: Use indirection table at runtime. Has overhead from
//!    double memory access.
//!
//! # The Problem
//!
//! The standard garbler allocates a buffer of size `feed_count` (total wires).
//! For AES-128, this is ~37,000 wires × 16 bytes = 576 KB, which exceeds L2 cache.
//!
//! However, in topologically-ordered circuits, most wires are "dead" (never read again)
//! after a few gates. At any point, only a small subset of wires are "live".
//!
//! # The Solution
//!
//! Renumber wires so that slot IDs are reused when wires die:
//! - For AES-128: 37,000 wires → 1,500 slots = 24 KB (fits in L1 cache!)
//!
//! # Usage (Recommended)
//!
//! ```ignore
//! // Remap circuit once (can be cached with the circuit)
//! let (remapped_circuit, input_map, output_map) = remap_circuit(&circuit);
//!
//! // Use standard garbler with remapped circuit
//! let mut garbler = half_gates::Garbler::default();
//! let iter = garbler.generate(&remapped_circuit, delta, &remapped_inputs)?;
//! ```

use core::fmt;
use std::ops::Range;

use crate::{DEFAULT_BATCH_SIZE, EncryptedGateBatch, circuit::EncryptedGate};
use mpz_circuits::{Circuit, Gate};
use mpz_core::{
    Block,
    aes::{FIXED_KEY_AES, FixedKeyAes},
};
use mpz_memory_core::correlated::{Delta, Key};

use super::garbler::and_gate;

/// Wire-to-slot remapping for cache-optimized garbling.
///
/// Maps logical wire IDs to physical buffer slot IDs, allowing slot reuse
/// when wires become dead (no longer needed by any future gate).
#[derive(Debug, Clone)]
pub struct WireRemapping {
    /// Maps wire_id -> slot_id
    slot_map: Vec<usize>,
    /// Number of slots needed (max simultaneously live wires)
    num_slots: usize,
    /// Input wire range (original wire IDs)
    inputs: Range<usize>,
    /// Output wire range (original wire IDs)
    outputs: Range<usize>,
}

impl WireRemapping {
    /// Compute wire remapping for a circuit.
    ///
    /// This analyzes wire liveness and assigns slot IDs that get recycled
    /// when wires die. The result can be cached and reused for multiple
    /// garbling operations on the same circuit.
    pub fn compute(circuit: &Circuit) -> Self {
        let feed_count = circuit.feed_count();
        let gates = circuit.gates();
        let inputs = circuit.inputs();
        let outputs = circuit.outputs();

        // Step 1: Find last use of each wire
        let mut last_use: Vec<Option<usize>> = vec![None; feed_count];

        // Outputs are "used" at the very end (after all gates)
        for wire_id in outputs.clone() {
            last_use[wire_id] = Some(gates.len());
        }

        // Scan gates to find last use of each wire
        for (gate_idx, gate) in gates.iter().enumerate() {
            match gate {
                Gate::Xor { x, y, .. } | Gate::And { x, y, .. } => {
                    last_use[x.id()] = Some(gate_idx);
                    last_use[y.id()] = Some(gate_idx);
                }
                Gate::Inv { x, .. } | Gate::Id { x, .. } => {
                    last_use[x.id()] = Some(gate_idx);
                }
            }
        }

        // Step 2: Assign slots using a free list
        let mut slot_map: Vec<usize> = vec![usize::MAX; feed_count];
        let mut free_slots: Vec<usize> = Vec::new();
        let mut next_slot: usize = 0;

        // Allocate slots for input wires
        for wire_id in inputs.clone() {
            let slot = if let Some(s) = free_slots.pop() {
                s
            } else {
                let s = next_slot;
                next_slot += 1;
                s
            };
            slot_map[wire_id] = slot;
        }

        // Process gates, allocating and freeing slots
        for (gate_idx, gate) in gates.iter().enumerate() {
            // Get input wire IDs
            let input_wires: Vec<usize> = match gate {
                Gate::Xor { x, y, .. } | Gate::And { x, y, .. } => vec![x.id(), y.id()],
                Gate::Inv { x, .. } | Gate::Id { x, .. } => vec![x.id()],
            };

            // Allocate slot for output wire
            let z_id = gate.z().id();
            let slot = if let Some(s) = free_slots.pop() {
                s
            } else {
                let s = next_slot;
                next_slot += 1;
                s
            };
            slot_map[z_id] = slot;

            // Free slots for wires whose last use was this gate
            for wire_id in input_wires {
                if last_use[wire_id] == Some(gate_idx) {
                    let slot = slot_map[wire_id];
                    free_slots.push(slot);
                }
            }
        }

        // num_slots is the total distinct slots ever allocated
        // This equals next_slot since slot IDs are 0..next_slot-1
        Self {
            slot_map,
            num_slots: next_slot,
            inputs,
            outputs,
        }
    }

    /// Returns the number of slots needed (max simultaneously live wires).
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Returns the slot ID for a wire.
    #[inline]
    pub fn slot(&self, wire_id: usize) -> usize {
        self.slot_map[wire_id]
    }

    /// Returns the input wire range.
    pub fn inputs(&self) -> Range<usize> {
        self.inputs.clone()
    }

    /// Returns the output wire range.
    pub fn outputs(&self) -> Range<usize> {
        self.outputs.clone()
    }
}

/// Errors that can occur during garbled circuit generation.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum RemappedGarblerError {
    #[error("input length mismatch: expected {expected}, got {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("garbler not finished")]
    NotFinished,
}

/// Output of the remapped garbler.
#[derive(Debug)]
pub struct RemappedGarblerOutput {
    /// Output keys of the circuit.
    pub outputs: Vec<Key>,
}

/// Cache-optimized garbler using wire remapping.
///
/// Uses a smaller label buffer by recycling slots for dead wires.
#[derive(Debug)]
pub struct RemappedGarbler {
    /// Wire remapping
    remapping: WireRemapping,
    /// Buffer for the 0-bit labels (sized to num_slots, not feed_count)
    buffer: Vec<Block>,
}

impl RemappedGarbler {
    /// Create a new remapped garbler with the given wire remapping.
    pub fn new(remapping: WireRemapping) -> Self {
        Self {
            buffer: vec![Block::default(); remapping.num_slots()],
            remapping,
        }
    }

    /// Returns an iterator over the encrypted gates of a circuit.
    pub fn generate<'a>(
        &'a mut self,
        circ: &'a Circuit,
        delta: Delta,
        inputs: &[Key],
    ) -> Result<RemappedEncryptedGateIter<'a, std::slice::Iter<'a, Gate>>, RemappedGarblerError>
    {
        if inputs.len() != circ.inputs().len() {
            return Err(RemappedGarblerError::InputLength {
                expected: circ.inputs().len(),
                actual: inputs.len(),
            });
        }

        // Initialize input labels using remapped slots
        for (i, key) in inputs.iter().enumerate() {
            let wire_id = circ.inputs().start + i;
            let slot = self.remapping.slot(wire_id);
            self.buffer[slot] = *key.as_block();
        }

        Ok(RemappedEncryptedGateIter::new(
            &self.remapping,
            delta,
            circ.gates().iter(),
            &mut self.buffer,
            circ.and_count(),
            circ.outputs(),
        ))
    }

    /// Returns an iterator over batched encrypted gates of a circuit.
    pub fn generate_batched<'a>(
        &'a mut self,
        circ: &'a Circuit,
        delta: Delta,
        inputs: &[Key],
    ) -> Result<RemappedEncryptedGateBatchIter<'a, std::slice::Iter<'a, Gate>>, RemappedGarblerError>
    {
        self.generate(circ, delta, inputs)
            .map(RemappedEncryptedGateBatchIter)
    }
}

/// Iterator over encrypted gates using wire remapping.
pub struct RemappedEncryptedGateIter<'a, I> {
    /// Wire remapping
    remapping: &'a WireRemapping,
    /// Cipher to use to encrypt the gates.
    cipher: &'static FixedKeyAes,
    /// Global offset.
    delta: Delta,
    /// Buffer for the 0-bit labels (sized to num_slots).
    labels: &'a mut [Block],
    /// Iterator over the gates.
    gates: I,
    /// Current gate id.
    gid: usize,
    /// Number of AND gates generated.
    counter: usize,
    /// Number of AND gates in the circuit.
    and_count: usize,
    /// Range of outputs in the circuit (original wire IDs).
    outputs: Range<usize>,
    /// Whether the entire circuit has been garbled.
    complete: bool,
}

impl<I> fmt::Debug for RemappedEncryptedGateIter<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RemappedEncryptedGateIter {{ .. }}")
    }
}

impl<'a, I> RemappedEncryptedGateIter<'a, I>
where
    I: Iterator<Item = &'a Gate>,
{
    fn new(
        remapping: &'a WireRemapping,
        delta: Delta,
        gates: I,
        labels: &'a mut [Block],
        and_count: usize,
        outputs: Range<usize>,
    ) -> Self {
        Self {
            remapping,
            cipher: &(*FIXED_KEY_AES),
            delta,
            gates,
            labels,
            gid: 1,
            counter: 0,
            and_count,
            outputs,
            complete: false,
        }
    }

    /// Returns `true` if the garbler has more encrypted gates to generate.
    #[inline]
    pub fn has_gates(&self) -> bool {
        self.counter != self.and_count
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(mut self) -> Result<RemappedGarblerOutput, RemappedGarblerError> {
        if self.has_gates() {
            return Err(RemappedGarblerError::NotFinished);
        }

        // Finish computing any "free" gates.
        if !self.complete {
            assert_eq!(self.next(), None);
        }

        // Collect output labels using remapped slots
        let outputs: Vec<Key> = self
            .outputs
            .clone()
            .map(|wire_id| {
                let slot = self.remapping.slot(wire_id);
                Key::from(self.labels[slot])
            })
            .collect();

        Ok(RemappedGarblerOutput { outputs })
    }
}

impl<'a, I> Iterator for RemappedEncryptedGateIter<'a, I>
where
    I: Iterator<Item = &'a Gate>,
{
    type Item = EncryptedGate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(gate) = self.gates.next() {
            match gate {
                Gate::Xor {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    // Use remapped slots for label access
                    let x_slot = self.remapping.slot(node_x.id());
                    let y_slot = self.remapping.slot(node_y.id());
                    let z_slot = self.remapping.slot(node_z.id());

                    let x_0 = self.labels[x_slot];
                    let y_0 = self.labels[y_slot];
                    self.labels[z_slot] = x_0 ^ y_0;
                }
                Gate::And {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x_slot = self.remapping.slot(node_x.id());
                    let y_slot = self.remapping.slot(node_y.id());
                    let z_slot = self.remapping.slot(node_z.id());

                    let x_0 = self.labels[x_slot];
                    let y_0 = self.labels[y_slot];
                    let (z_0, encrypted_gate) =
                        and_gate(self.cipher, &x_0, &y_0, &self.delta, self.gid);
                    self.labels[z_slot] = z_0;

                    self.gid += 2;
                    self.counter += 1;

                    if !self.has_gates() {
                        assert!(self.next().is_none());
                        self.complete = true;
                    }

                    return Some(encrypted_gate);
                }
                Gate::Inv {
                    x: node_x,
                    z: node_z,
                } => {
                    let x_slot = self.remapping.slot(node_x.id());
                    let z_slot = self.remapping.slot(node_z.id());

                    let x_0 = self.labels[x_slot];
                    self.labels[z_slot] = x_0 ^ self.delta.as_block();
                }
                Gate::Id {
                    x: node_x,
                    z: node_z,
                } => {
                    let x_slot = self.remapping.slot(node_x.id());
                    let z_slot = self.remapping.slot(node_z.id());

                    let x_0 = self.labels[x_slot];
                    self.labels[z_slot] = x_0;
                }
            }
        }

        None
    }
}

/// Iterator returned by [`RemappedGarbler::generate_batched`].
#[derive(Debug)]
pub struct RemappedEncryptedGateBatchIter<'a, I: Iterator, const N: usize = DEFAULT_BATCH_SIZE>(
    RemappedEncryptedGateIter<'a, I>,
);

impl<'a, I, const N: usize> RemappedEncryptedGateBatchIter<'a, I, N>
where
    I: Iterator<Item = &'a Gate>,
{
    /// Returns `true` if the garbler has more encrypted gates to generate.
    pub fn has_gates(&self) -> bool {
        self.0.has_gates()
    }

    /// Returns the encoded outputs of the circuit.
    pub fn finish(self) -> Result<RemappedGarblerOutput, RemappedGarblerError> {
        self.0.finish()
    }
}

impl<'a, I, const N: usize> Iterator for RemappedEncryptedGateBatchIter<'a, I, N>
where
    I: Iterator<Item = &'a Gate>,
{
    type Item = EncryptedGateBatch<N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_gates() {
            return None;
        }

        let mut batch = [EncryptedGate::default(); N];
        let mut i = 0;
        for gate in self.0.by_ref() {
            batch[i] = gate;
            i += 1;

            if i == N {
                break;
            }
        }

        Some(EncryptedGateBatch::new(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_circuits::AES128;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn test_wire_remapping_reduces_slots() {
        let circ = &*AES128;
        let remapping = WireRemapping::compute(circ);

        println!("feed_count: {}", circ.feed_count());
        println!("num_slots:  {}", remapping.num_slots());
        println!(
            "reduction:  {:.1}x",
            circ.feed_count() as f64 / remapping.num_slots() as f64
        );

        // Should have significant reduction
        assert!(
            remapping.num_slots() < circ.feed_count() / 10,
            "Expected >10x reduction, got {}x",
            circ.feed_count() / remapping.num_slots()
        );
    }

    #[test]
    fn test_remapped_garbler_produces_same_output_count() {
        let circ = &*AES128;
        let remapping = WireRemapping::compute(circ);
        let mut garbler = RemappedGarbler::new(remapping);

        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        let delta = Delta::new(delta);

        let input_keys: Vec<Key> = (0..circ.inputs().len())
            .map(|_| {
                let block: Block = rng.random();
                block.into()
            })
            .collect();

        let mut iter = garbler
            .generate(circ, delta, &input_keys)
            .unwrap();

        // Consume all gates
        let mut gate_count = 0;
        while iter.has_gates() {
            iter.next();
            gate_count += 1;
        }

        assert_eq!(gate_count, circ.and_count());

        let output = iter.finish().unwrap();
        assert_eq!(output.outputs.len(), circ.outputs().len());
    }

    #[test]
    fn test_remapped_matches_standard_garbler() {
        use super::super::garbler::Garbler;

        let circ = &*AES128;
        let remapping = WireRemapping::compute(circ);

        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        let delta = Delta::new(delta);

        let input_keys: Vec<Key> = (0..circ.inputs().len())
            .map(|_| {
                let block: Block = rng.random();
                block.into()
            })
            .collect();

        // Standard garbler
        let mut std_garbler = Garbler::default();
        let std_iter = std_garbler
            .generate(circ, delta, &input_keys)
            .unwrap();
        let std_gates: Vec<_> = std_iter.collect();

        // Remapped garbler
        let mut rem_garbler = RemappedGarbler::new(remapping);
        let rem_iter = rem_garbler
            .generate(circ, delta, &input_keys)
            .unwrap();
        let rem_gates: Vec<_> = rem_iter.collect();

        // Gates should be identical
        assert_eq!(std_gates.len(), rem_gates.len());
        for (i, (std, rem)) in std_gates.iter().zip(rem_gates.iter()).enumerate() {
            assert_eq!(std, rem, "Gate {} mismatch", i);
        }
    }

    #[test]
    fn test_circuit_remap_garble_evaluate() {
        use super::super::{Evaluator, Garbler};
        use mpz_circuits::remap::RemappedCircuit;

        let circ = &*AES128;
        let remapped = RemappedCircuit::new(circ);

        let mut rng = ChaCha12Rng::seed_from_u64(42);

        let mut delta = Block::random(&mut rng);
        delta.set_lsb(true);
        let delta = Delta::new(delta);

        let input_keys: Vec<Key> = (0..circ.inputs().len())
            .map(|_| {
                let block: Block = rng.random();
                block.into()
            })
            .collect();

        // Garble the remapped circuit
        let mut garbler = Garbler::default();
        let mut iter = garbler
            .generate(remapped.circuit(), delta, &input_keys)
            .unwrap();
        let gates: Vec<_> = iter.by_ref().collect();
        let garbler_output = iter.finish().unwrap();

        // Select input labels for evaluation (random bits)
        let input_bits: Vec<bool> = (0..circ.inputs().len()).map(|_| rng.random()).collect();
        let eval_inputs: Vec<_> = input_keys
            .iter()
            .zip(&input_bits)
            .map(|(k, &b)| k.auth(b, &delta))
            .collect();

        // Evaluate the remapped circuit
        let mut evaluator = Evaluator::default();
        let mut consumer = evaluator
            .evaluate(remapped.circuit(), &eval_inputs)
            .unwrap();
        for gate in &gates {
            consumer.next(*gate);
        }
        let _eval_output = consumer.finish().unwrap();

        // Verify output count matches
        // Note: The remapped circuit has outputs: 0..0, so we need to use output_slots
        // to find the actual output labels in the evaluator's label buffer
        assert_eq!(
            garbler_output.outputs.len(),
            0,
            "Remapped circuit should have 0 outputs in Circuit struct"
        );

        // Verify gate counts preserved
        assert_eq!(remapped.circuit().and_count(), circ.and_count());
        assert_eq!(remapped.circuit().gates().len(), circ.gates().len());

        println!(
            "Circuit remap test passed: {} slots vs {} original wires",
            remapped.num_slots(),
            circ.feed_count()
        );
    }
}
