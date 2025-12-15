//! Wire remapping for cache-optimized circuit execution.
//!
//! This module provides functionality to create a "remapped" circuit where wire IDs
//! are reused when wires become dead (no longer needed). This dramatically reduces
//! the buffer size needed during garbling/evaluation.
//!
//! # Example
//!
//! For AES-128:
//! - Original: 36,919 wires → 576 KB buffer (exceeds L2 cache)
//! - Remapped: 1,494 slots → 24 KB buffer (fits in L1 cache)
//!
//! # Usage
//!
//! ```ignore
//! use mpz_circuits::remap::RemappedCircuit;
//!
//! let remapped = RemappedCircuit::new(&original_circuit);
//!
//! // Use remapped.circuit() with standard garbler
//! // Map inputs: remapped.map_input_index(original_idx) -> remapped_idx
//! // Map outputs: remapped.map_output_index(original_idx) -> remapped_idx
//! ```

use crate::circuit::Circuit;
use crate::components::{Feed, Gate, Node, Sink};

/// A circuit with wire IDs remapped to reuse slots for dead wires.
///
/// This reduces the buffer size needed during execution by reusing
/// wire slots when the previous value is no longer needed.
#[derive(Debug, Clone)]
pub struct RemappedCircuit {
    /// The remapped circuit with compacted wire IDs
    circuit: Circuit,
    /// Maps original input index → remapped input index
    /// (inputs may be reordered to optimize slot assignment)
    input_map: Vec<usize>,
    /// Maps original output index → remapped slot ID
    /// (needed to find output values in the smaller buffer)
    output_slots: Vec<usize>,
}

impl RemappedCircuit {
    /// Create a remapped circuit from an original circuit.
    ///
    /// This analyzes wire liveness and assigns slot IDs that get reused
    /// when wires die. The resulting circuit has a much smaller `feed_count`.
    pub fn new(original: &Circuit) -> Self {
        let feed_count = original.feed_count();
        let gates = original.gates();
        let inputs = original.inputs();
        let outputs = original.outputs();

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
        let mut wire_to_slot: Vec<usize> = vec![usize::MAX; feed_count];
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
            wire_to_slot[wire_id] = slot;
        }

        // Process gates, allocating and freeing slots
        for (gate_idx, gate) in gates.iter().enumerate() {
            // Get input wire IDs for freeing
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
            wire_to_slot[z_id] = slot;

            // Free slots for wires whose last use was this gate
            for wire_id in input_wires {
                if last_use[wire_id] == Some(gate_idx) {
                    let slot = wire_to_slot[wire_id];
                    free_slots.push(slot);
                }
            }
        }

        let num_slots = next_slot;

        // Step 3: Build remapped gates
        let remapped_gates: Vec<Gate> = gates
            .iter()
            .map(|gate| match gate {
                Gate::Xor { x, y, z } => Gate::Xor {
                    x: Node::<Sink>::new(wire_to_slot[x.id()]),
                    y: Node::<Sink>::new(wire_to_slot[y.id()]),
                    z: Node::<Feed>::new(wire_to_slot[z.id()]),
                },
                Gate::And { x, y, z } => Gate::And {
                    x: Node::<Sink>::new(wire_to_slot[x.id()]),
                    y: Node::<Sink>::new(wire_to_slot[y.id()]),
                    z: Node::<Feed>::new(wire_to_slot[z.id()]),
                },
                Gate::Inv { x, z } => Gate::Inv {
                    x: Node::<Sink>::new(wire_to_slot[x.id()]),
                    z: Node::<Feed>::new(wire_to_slot[z.id()]),
                },
                Gate::Id { x, z } => Gate::Id {
                    x: Node::<Sink>::new(wire_to_slot[x.id()]),
                    z: Node::<Feed>::new(wire_to_slot[z.id()]),
                },
            })
            .collect();

        // Step 4: Build input/output mappings
        // Input map: original input index → slot ID
        let input_map: Vec<usize> = inputs.clone().map(|w| wire_to_slot[w]).collect();

        // Output slots: original output index → slot ID
        let output_slots: Vec<usize> = outputs.clone().map(|w| wire_to_slot[w]).collect();

        // Step 5: Build the remapped circuit
        // Inputs are slots 0..input_count (in order of input_map)
        // Outputs are wherever they ended up (tracked in output_slots)
        let remapped_circuit = Circuit {
            inputs: 0..inputs.len(),
            outputs: 0..0, // Outputs are scattered, use output_slots instead
            gates: remapped_gates,
            feed_count: num_slots,
            and_count: original.and_count(),
            xor_count: original.xor_count(),
        };

        Self {
            circuit: remapped_circuit,
            input_map,
            output_slots,
        }
    }

    /// Returns a reference to the remapped circuit.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Returns the number of slots (reduced feed_count).
    pub fn num_slots(&self) -> usize {
        self.circuit.feed_count()
    }

    /// Maps an original input index to the slot ID in the remapped circuit.
    ///
    /// Use this to reorder input keys before passing to the garbler.
    pub fn input_slot(&self, original_input_idx: usize) -> usize {
        self.input_map[original_input_idx]
    }

    /// Returns the slot ID for an original output index.
    ///
    /// Use this to find output values in the labels buffer.
    pub fn output_slot(&self, original_output_idx: usize) -> usize {
        self.output_slots[original_output_idx]
    }

    /// Returns the input slot mapping.
    pub fn input_map(&self) -> &[usize] {
        &self.input_map
    }

    /// Returns the output slot mapping.
    pub fn output_slots(&self) -> &[usize] {
        &self.output_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;

    #[test]
    fn test_remap_reduces_feed_count_blake3() {
        let circ = circuits::blake3::compress();
        let remapped = RemappedCircuit::new(&circ);

        println!("Blake3 compress:");
        println!("  Original feed_count: {}", circ.feed_count());
        println!("  Remapped feed_count: {}", remapped.num_slots());
        println!(
            "  Reduction: {:.1}x",
            circ.feed_count() as f64 / remapped.num_slots() as f64
        );

        // Should have some reduction (blake3 may not be as dramatic as AES)
        assert!(
            remapped.num_slots() < circ.feed_count(),
            "Expected some reduction"
        );
    }

    #[test]
    fn test_remap_preserves_gate_counts() {
        let circ = circuits::blake3::compress();
        let remapped = RemappedCircuit::new(&circ);

        assert_eq!(remapped.circuit().and_count(), circ.and_count());
        assert_eq!(remapped.circuit().xor_count(), circ.xor_count());
        assert_eq!(remapped.circuit().gates().len(), circ.gates().len());
    }

    #[test]
    fn test_remap_input_output_mapping() {
        let circ = circuits::blake3::compress();
        let remapped = RemappedCircuit::new(&circ);

        // Should have same number of inputs and outputs
        assert_eq!(remapped.input_map().len(), circ.inputs().len());
        assert_eq!(remapped.output_slots().len(), circ.outputs().len());

        // All input slots should be valid
        for &slot in remapped.input_map() {
            assert!(slot < remapped.num_slots());
        }

        // All output slots should be valid
        for &slot in remapped.output_slots() {
            assert!(slot < remapped.num_slots());
        }
    }

    #[test]
    fn test_remap_simple_adder() {
        let circ = circuits::adder_u8();
        let remapped = RemappedCircuit::new(&circ);

        println!("Adder u8:");
        println!("  Original feed_count: {}", circ.feed_count());
        println!("  Remapped feed_count: {}", remapped.num_slots());

        // Basic sanity checks
        assert_eq!(remapped.input_map().len(), 16); // 2 x 8 bits
        assert_eq!(remapped.output_slots().len(), 8); // 8 bits output
    }

    #[test]
    fn test_remap_correctness_adder() {
        use crate::evaluate;
        use itybity::{FromBitIterator, ToBits};

        let circ = circuits::adder_u8();
        let remapped = RemappedCircuit::new(&circ);

        // Test several input pairs
        let test_cases: [(u8, u8); 5] = [(0, 0), (1, 1), (42, 69), (255, 1), (128, 128)];

        for (a, b) in test_cases {
            // Evaluate original circuit
            let original_result: u8 = evaluate!(&circ, a, b).unwrap();

            // Evaluate remapped circuit using evaluate_raw and extract outputs via output_slots
            let mut feeds = vec![false; remapped.circuit().feed_count()];
            // Set inputs
            for (i, bit) in a.iter_lsb0().chain(b.iter_lsb0()).enumerate() {
                feeds[i] = bit;
            }
            remapped.circuit().evaluate_raw(&mut feeds).unwrap();

            // Extract outputs using output_slots
            let output_bits: Vec<bool> = remapped
                .output_slots()
                .iter()
                .map(|&slot| feeds[slot])
                .collect();
            let remapped_result: u8 = FromBitIterator::from_lsb0_iter(output_bits.into_iter());

            assert_eq!(
                original_result,
                a.wrapping_add(b),
                "Original circuit failed for {a} + {b}"
            );
            assert_eq!(
                remapped_result,
                a.wrapping_add(b),
                "Remapped circuit failed for {a} + {b}"
            );
        }
    }
}
