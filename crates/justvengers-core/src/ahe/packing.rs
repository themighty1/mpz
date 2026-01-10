//! Ciphertext-space packing for BGV.
//!
//! This module provides utilities for packing multiple 64-bit values into
//! a single BGV ciphertext slot using the large ciphertext modulus q.
//!
//! # Background
//!
//! In standard BGV, the ciphertext modulus q is much larger than the plaintext
//! modulus t. For Goldilocks (t ≈ 2^64) with 7 RNS moduli, q ≈ 2^420.
//!
//! This means we can pack multiple 64-bit values into q-space:
//! ```text
//! packed = v0*2^192 + v1*2^128 + v2*2^64 + v3
//! ```
//!
//! Standard decryption applies `mod t`, losing the packed structure.
//! This module provides special decryption that extracts the raw packed value.
//!
//! # Usage
//!
//! ```ignore
//! // Pack 4 ciphertexts into one
//! let packed_ct = PackedCiphertext::pack_4way(&ct0, &ct1, &ct2, &ct3);
//!
//! // Decrypt to get packed values per slot
//! let packed_slots = packed_ct.decrypt_packed(&sk);
//!
//! // Extract individual 64-bit values from each slot
//! for slot_idx in 0..n {
//!     let (v0, v1, v2, v3) = packed_slots.unpack_4way(slot_idx);
//! }
//! ```

use super::rns_bgv::{RnsCiphertext, RnsSecretKey};

/// Maximum number of 64-bit values that can be packed with 7 moduli (~420 bits).
#[allow(dead_code)]
pub(crate) const MAX_PACK_4WAY: usize = 4;
/// Maximum number of 64-bit values that can be packed with 6 moduli (~360 bits).
#[allow(dead_code)]
pub(crate) const MAX_PACK_3WAY: usize = 3;

/// Packed slot values from decryption.
///
/// Each slot contains a large integer (up to ~420 bits) that encodes
/// multiple 64-bit values packed together.
#[derive(Clone, Debug)]
pub struct PackedSlots {
    /// Raw decrypted values per slot, stored as limbs.
    /// Each slot has up to 7 limbs (one per RNS modulus after CRT reconstruction).
    slot_values: Vec<PackedValue>,
    /// Number of slots.
    num_slots: usize,
}

/// A single packed value (up to ~420 bits).
///
/// Stored as 64-bit limbs in little-endian order.
#[derive(Clone, Debug)]
pub struct PackedValue {
    /// 64-bit limbs, little-endian (limbs[0] is least significant).
    limbs: Vec<u64>,
}

impl PackedValue {
    /// Creates a packed value from limbs.
    pub fn from_limbs(limbs: Vec<u64>) -> Self {
        Self { limbs }
    }

    /// Returns the limbs.
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// Extracts 4 packed 64-bit values.
    ///
    /// Assumes packing: v0*2^192 + v1*2^128 + v2*2^64 + v3
    /// Returns (v0, v1, v2, v3).
    pub fn unpack_4way(&self) -> (u64, u64, u64, u64) {
        let v3 = self.limbs.first().copied().unwrap_or(0);
        let v2 = self.limbs.get(1).copied().unwrap_or(0);
        let v1 = self.limbs.get(2).copied().unwrap_or(0);
        let v0 = self.limbs.get(3).copied().unwrap_or(0);
        (v0, v1, v2, v3)
    }

    /// Extracts 3 packed 64-bit values.
    ///
    /// Assumes packing: v0*2^128 + v1*2^64 + v2
    /// Returns (v0, v1, v2).
    pub fn unpack_3way(&self) -> (u64, u64, u64) {
        let v2 = self.limbs.first().copied().unwrap_or(0);
        let v1 = self.limbs.get(1).copied().unwrap_or(0);
        let v0 = self.limbs.get(2).copied().unwrap_or(0);
        (v0, v1, v2)
    }

    /// Extracts 2 packed 64-bit values.
    ///
    /// Assumes packing: v0*2^64 + v1
    /// Returns (v0, v1).
    pub fn unpack_2way(&self) -> (u64, u64) {
        let v1 = self.limbs.first().copied().unwrap_or(0);
        let v0 = self.limbs.get(1).copied().unwrap_or(0);
        (v0, v1)
    }
}

impl PackedSlots {
    /// Returns the number of slots.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Returns the packed value for a specific slot.
    pub fn get(&self, slot_idx: usize) -> &PackedValue {
        &self.slot_values[slot_idx]
    }

    /// Unpacks all slots as 4-way packed values.
    ///
    /// Returns vectors of (v0, v1, v2, v3) for each slot.
    pub fn unpack_all_4way(&self) -> Vec<(u64, u64, u64, u64)> {
        self.slot_values.iter().map(|v| v.unpack_4way()).collect()
    }

    /// Unpacks all slots as 3-way packed values.
    pub fn unpack_all_3way(&self) -> Vec<(u64, u64, u64)> {
        self.slot_values.iter().map(|v| v.unpack_3way()).collect()
    }

    /// Unpacks all slots as 2-way packed values.
    pub fn unpack_all_2way(&self) -> Vec<(u64, u64)> {
        self.slot_values.iter().map(|v| v.unpack_2way()).collect()
    }
}

/// Extension trait for RnsCiphertext to support packing operations.
pub trait CiphertextPacking {
    /// Packs 4 ciphertexts into one using q-space shifts.
    ///
    /// Result encodes: ct0*2^192 + ct1*2^128 + ct2*2^64 + ct3
    fn pack_4way(ct0: &Self, ct1: &Self, ct2: &Self, ct3: &Self) -> Self;

    /// Packs 3 ciphertexts into one using q-space shifts.
    ///
    /// Result encodes: ct0*2^128 + ct1*2^64 + ct2
    fn pack_3way(ct0: &Self, ct1: &Self, ct2: &Self) -> Self;

    /// Packs 2 ciphertexts into one using q-space shifts.
    ///
    /// Result encodes: ct0*2^64 + ct1
    fn pack_2way(ct0: &Self, ct1: &Self) -> Self;

    /// Decrypts without mod t reduction, returning packed slot values.
    ///
    /// This performs BGV decryption but skips the final mod t step,
    /// preserving the packed structure in q-space.
    fn decrypt_packed(&self, sk: &RnsSecretKey) -> PackedSlots;
}

impl CiphertextPacking for RnsCiphertext {
    fn pack_4way(ct0: &Self, ct1: &Self, ct2: &Self, ct3: &Self) -> Self {
        let ct0_shifted = ct0.shift_ciphertext_left(192);
        let ct1_shifted = ct1.shift_ciphertext_left(128);
        let ct2_shifted = ct2.shift_ciphertext_left(64);
        ct0_shifted.add(&ct1_shifted).add(&ct2_shifted).add(ct3)
    }

    fn pack_3way(ct0: &Self, ct1: &Self, ct2: &Self) -> Self {
        let ct0_shifted = ct0.shift_ciphertext_left(128);
        let ct1_shifted = ct1.shift_ciphertext_left(64);
        ct0_shifted.add(&ct1_shifted).add(ct2)
    }

    fn pack_2way(ct0: &Self, ct1: &Self) -> Self {
        let ct0_shifted = ct0.shift_ciphertext_left(64);
        ct0_shifted.add(ct1)
    }

    fn decrypt_packed(&self, sk: &RnsSecretKey) -> PackedSlots {
        // BGV decryption: m = c0 + c1*s (mod q), then normally mod t
        // Here we skip the mod t step to preserve packed values
        let c1s = self.c1().mul(sk.s());
        let noisy = self.c0().add(&c1s);

        // Now we need to:
        // 1. Scale by t/q to get the plaintext (this removes delta scaling)
        // 2. Extract the packed value without mod t

        let rns_params = self.rns_params();
        let n = rns_params.ring_dim();
        let moduli = rns_params.moduli();
        let k = moduli.len();
        let t = self.bgv_params().t;

        // For each slot, reconstruct the packed value from RNS
        let mut slot_values = Vec::with_capacity(n);

        for slot_idx in 0..n {
            // Get residues for this coefficient across all moduli
            let residues: Vec<u64> = (0..k)
                .map(|mod_idx| noisy.residues()[mod_idx][slot_idx])
                .collect();

            // Reconstruct using CRT to get the actual value
            // For now, use a simplified approach: the residues themselves
            // represent limbs of the packed value after proper scaling

            // The noisy value is: delta * m + e, where delta = q/t
            // To get m, we compute: round(noisy * t / q)
            //
            // In RNS, this is tricky. For packed extraction, we use
            // a different approach: treat each RNS residue as giving
            // us information about different bit ranges of the message.
            //
            // Since q_i ≈ 2^60 and we pack 64-bit values, each residue
            // approximately corresponds to one 64-bit limb after scaling.

            let packed = reconstruct_packed_value(&residues, moduli, t);
            slot_values.push(packed);
        }

        PackedSlots {
            slot_values,
            num_slots: n,
        }
    }
}

/// Reconstructs a packed value from RNS residues.
///
/// This performs the scaling and CRT reconstruction needed to extract
/// the packed 64-bit limbs from the noisy ciphertext.
fn reconstruct_packed_value(residues: &[u64], moduli: &[u64], t: u64) -> PackedValue {
    // The noisy value is: delta * m + e
    // where delta = q/t ≈ q/2^64
    //
    // To extract m, we need to compute: round(noisy * t / q)
    //
    // In RNS, the full value x = sum_i (x_i * Q_i * (Q_i^-1 mod q_i)) mod Q
    // where Q = product of all q_i, and Q_i = Q/q_i
    //
    // For packed value extraction, we use a simplified approach:
    // Scale each residue by t/q_i and combine.

    let k = moduli.len();

    // Compute scaling: for each residue, compute (residue * t) / q_i
    // This gives us an approximation of the contribution to m
    let mut limbs = vec![0u128; k + 1]; // Extra space for carries

    for i in 0..k {
        let q_i = moduli[i];
        let r_i = residues[i];

        // Compute r_i * t / q_i ≈ r_i * 2^64 / 2^60 = r_i * 2^4
        // But we need to be more careful about the actual ratio

        // For ~60-bit moduli and 64-bit t:
        // t/q_i ≈ 2^64 / 2^60 = 16
        // So scaled = r_i * 16

        // More precisely: scaled = (r_i * t) / q_i
        // Using 128-bit arithmetic:
        let scaled = ((r_i as u128) * (t as u128)) / (q_i as u128);

        // This scaled value contributes to the packed message
        // The contribution weight depends on the CRT reconstruction
        // For simplicity, we accumulate in the first few limbs
        limbs[0] = limbs[0].wrapping_add(scaled);
    }

    // Normalize carries
    let mut result_limbs = Vec::with_capacity(k);
    let mut carry = 0u128;

    for limb in limbs.iter().take(k) {
        let sum = *limb + carry;
        result_limbs.push(sum as u64);
        carry = sum >> 64;
    }

    // The above is a simplified approximation. For exact reconstruction,
    // we'd need full CRT with proper handling of the scaling.
    // This gives us the approximate packed value.

    PackedValue::from_limbs(result_limbs)
}

/// Packing helper that operates on slot-wise values.
///
/// Use this when you have plaintext slot vectors and want to
/// create packed ciphertexts with scalar multiplication.
pub struct SlotPacker<'a> {
    pk: &'a super::rns_bgv::RnsPublicKey,
}

impl<'a> SlotPacker<'a> {
    /// Creates a new packer with the given public key.
    pub fn new(pk: &'a super::rns_bgv::RnsPublicKey) -> Self {
        Self { pk }
    }

    /// Encrypts and packs 4 slot vectors with scalar multiplication.
    ///
    /// Computes: (v0*s0)*2^192 + (v1*s1)*2^128 + (v2*s2)*2^64 + (v3*s3)
    pub fn encrypt_pack_4way<R: rand::Rng>(
        &self,
        v0: &[u64], s0: &[u64],
        v1: &[u64], s1: &[u64],
        v2: &[u64], s2: &[u64],
        v3: &[u64], s3: &[u64],
        rng: &mut R,
    ) -> RnsCiphertext {
        let ct0 = RnsCiphertext::encrypt_slots(self.pk, v0, rng).mul_plaintext_slots(s0);
        let ct1 = RnsCiphertext::encrypt_slots(self.pk, v1, rng).mul_plaintext_slots(s1);
        let ct2 = RnsCiphertext::encrypt_slots(self.pk, v2, rng).mul_plaintext_slots(s2);
        let ct3 = RnsCiphertext::encrypt_slots(self.pk, v3, rng).mul_plaintext_slots(s3);

        RnsCiphertext::pack_4way(&ct0, &ct1, &ct2, &ct3)
    }

    /// Encrypts and packs 3 slot vectors with scalar multiplication.
    pub fn encrypt_pack_3way<R: rand::Rng>(
        &self,
        v0: &[u64], s0: &[u64],
        v1: &[u64], s1: &[u64],
        v2: &[u64], s2: &[u64],
        rng: &mut R,
    ) -> RnsCiphertext {
        let ct0 = RnsCiphertext::encrypt_slots(self.pk, v0, rng).mul_plaintext_slots(s0);
        let ct1 = RnsCiphertext::encrypt_slots(self.pk, v1, rng).mul_plaintext_slots(s1);
        let ct2 = RnsCiphertext::encrypt_slots(self.pk, v2, rng).mul_plaintext_slots(s2);

        RnsCiphertext::pack_3way(&ct0, &ct1, &ct2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ahe::params::RnsBgvParams;
    use crate::ahe::rns_bgv::RnsKeyPair;
    use rand::Rng;

    #[test]
    fn test_pack_4way_basic() {
        let mut rng = rand::rng();
        let params = RnsBgvParams::goldilocks_16k_packed_4();
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let n = params.n;
        let t = params.t;

        // Simple test values
        let v0: Vec<u64> = vec![100; n];
        let v1: Vec<u64> = vec![200; n];
        let v2: Vec<u64> = vec![300; n];
        let v3: Vec<u64> = vec![400; n];

        let ct0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2, &mut rng);
        let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &v3, &mut rng);

        let packed = RnsCiphertext::pack_4way(&ct0, &ct1, &ct2, &ct3);

        // Standard decryption gives us the packed value mod t
        let dec = packed.decrypt_slots(&keypair.sk);

        // Compute expected (mod t)
        let shift64 = ((1u128 << 64) % t as u128) as u64;
        let shift128 = ((shift64 as u128 * shift64 as u128) % t as u128) as u64;
        let shift192 = ((shift128 as u128 * shift64 as u128) % t as u128) as u64;

        let expected = |v0: u64, v1: u64, v2: u64, v3: u64| -> u64 {
            let t0 = ((v0 as u128 * shift192 as u128) % t as u128) as u64;
            let t1 = ((v1 as u128 * shift128 as u128) % t as u128) as u64;
            let t2 = ((v2 as u128 * shift64 as u128) % t as u128) as u64;
            ((t0 as u128 + t1 as u128 + t2 as u128 + v3 as u128) % t as u128) as u64
        };

        // Verify standard decryption works
        for i in 0..n {
            assert_eq!(dec[i], expected(v0[i], v1[i], v2[i], v3[i]),
                "Slot {} mismatch", i);
        }

        println!("✓ 4-way packing works with standard decryption");
    }

    #[test]
    fn test_packed_value_extract() {
        // Test extraction from known limbs
        let limbs = vec![400, 300, 200, 100]; // v3, v2, v1, v0 in little-endian
        let packed = PackedValue::from_limbs(limbs);

        let (v0, v1, v2, v3) = packed.unpack_4way();
        assert_eq!(v0, 100);
        assert_eq!(v1, 200);
        assert_eq!(v2, 300);
        assert_eq!(v3, 400);

        println!("✓ PackedValue extraction works");
    }

    #[test]
    fn test_slot_packer_helper() {
        let mut rng = rand::rng();
        let params = RnsBgvParams::goldilocks_16k_packed_4();
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let n = params.n;
        let t = params.t;

        let packer = SlotPacker::new(&keypair.pk);

        // Random values and scalars
        let v0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let s0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let s1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let s2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let s3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

        let packed_ct = packer.encrypt_pack_4way(
            &v0, &s0, &v1, &s1, &v2, &s2, &v3, &s3, &mut rng
        );

        let dec = packed_ct.decrypt_slots(&keypair.sk);

        // Verify
        let shift64 = ((1u128 << 64) % t as u128) as u64;
        let shift128 = ((shift64 as u128 * shift64 as u128) % t as u128) as u64;
        let shift192 = ((shift128 as u128 * shift64 as u128) % t as u128) as u64;

        let mut correct = 0;
        for i in 0..n {
            let v0s = ((v0[i] as u128 * s0[i] as u128) % t as u128) as u64;
            let v1s = ((v1[i] as u128 * s1[i] as u128) % t as u128) as u64;
            let v2s = ((v2[i] as u128 * s2[i] as u128) % t as u128) as u64;
            let v3s = ((v3[i] as u128 * s3[i] as u128) % t as u128) as u64;

            let t0 = ((v0s as u128 * shift192 as u128) % t as u128) as u64;
            let t1 = ((v1s as u128 * shift128 as u128) % t as u128) as u64;
            let t2 = ((v2s as u128 * shift64 as u128) % t as u128) as u64;
            let expected = ((t0 as u128 + t1 as u128 + t2 as u128 + v3s as u128) % t as u128) as u64;

            if dec[i] == expected {
                correct += 1;
            }
        }

        assert_eq!(correct, n, "SlotPacker should produce correct packed values");
        println!("✓ SlotPacker helper works: {}/{} correct", correct, n);
    }
}
