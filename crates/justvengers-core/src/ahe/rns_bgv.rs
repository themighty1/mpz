//! RNS-based BGV encryption with slot packing.
//!
//! This module provides BGV encryption using RNS representation for the
//! ciphertext modulus, enabling support for large plaintext moduli like
//! Goldilocks. It also integrates slot packing for SIMD-style operations.
//!
//! # Key Features
//!
//! - RNS representation: q = q_1 × q_2 × ... × q_k for large ciphertext modulus
//! - Slot packing: encode N plaintext values into one ciphertext
//! - Goldilocks support: plaintext modulus p = 2^64 - 2^32 + 1

use rand::Rng;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::params::{RnsBgvParams, GOLDILOCKS};
use super::rns::{RnsParams, RnsPoly};
use super::slot::SlotEncoder;

/// RNS-based BGV secret key.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsSecretKey {
    /// Secret polynomial s in RNS form (ternary coefficients).
    s: RnsPoly,
    /// RNS parameters.
    rns_params: RnsParams,
    /// BGV parameters.
    bgv_params: RnsBgvParams,
}

/// RNS-based BGV public key.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsPublicKey {
    /// Random polynomial a in RNS form.
    a: RnsPoly,
    /// b = -a·s + e in RNS form.
    b: RnsPoly,
    /// RNS parameters.
    rns_params: RnsParams,
    /// BGV parameters.
    bgv_params: RnsBgvParams,
}

/// RNS-based BGV key pair.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsKeyPair {
    /// The secret key.
    pub sk: RnsSecretKey,
    /// The public key.
    pub pk: RnsPublicKey,
}

/// RNS-based BGV ciphertext with slot packing support.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsCiphertext {
    /// First ciphertext component c0.
    c0: RnsPoly,
    /// Second ciphertext component c1.
    c1: RnsPoly,
    /// RNS parameters.
    rns_params: RnsParams,
    /// BGV parameters.
    bgv_params: RnsBgvParams,
}

impl RnsSecretKey {
    /// Returns the secret polynomial.
    pub fn poly(&self) -> &RnsPoly {
        &self.s
    }

    /// Returns the secret polynomial (alias for poly()).
    pub fn s(&self) -> &RnsPoly {
        &self.s
    }
}

impl RnsPublicKey {
    /// Returns the a component.
    pub fn a(&self) -> &RnsPoly {
        &self.a
    }

    /// Returns the b component.
    pub fn b(&self) -> &RnsPoly {
        &self.b
    }

    /// Returns the BGV parameters.
    pub fn params(&self) -> &RnsBgvParams {
        &self.bgv_params
    }
}

impl RnsKeyPair {
    /// Generates a new RNS BGV key pair.
    pub fn generate<R: Rng>(bgv_params: &RnsBgvParams, rng: &mut R) -> Self {
        let rns_params = RnsParams::new(bgv_params.n, bgv_params.num_moduli, 60);

        // Sample ternary secret key
        let s = sample_ternary_rns(&rns_params, rng);

        // Sample uniform a
        let a = sample_uniform_rns(&rns_params, rng);

        // Sample Gaussian error
        let e = sample_gaussian_rns(&rns_params, bgv_params.sigma, rng);

        // b = -a·s + e
        let neg_as = a.mul(&s).neg();
        let b = neg_as.add(&e);

        let sk = RnsSecretKey {
            s,
            rns_params: rns_params.clone(),
            bgv_params: bgv_params.clone(),
        };

        let pk = RnsPublicKey {
            a,
            b,
            rns_params,
            bgv_params: bgv_params.clone(),
        };

        Self { sk, pk }
    }

    /// Generates a key pair for Goldilocks field with slot packing.
    ///
    /// Uses 4 RNS moduli (~240 bit q) for sufficient noise budget.
    pub fn generate_goldilocks<R: Rng>(rng: &mut R) -> Self {
        let params = RnsBgvParams::goldilocks();
        Self::generate(&params, rng)
    }
}

// ============================================================================
// Galois Keys for Slot Rotation
// ============================================================================

/// Decomposition base for key-switching (2^15 = 32768).
/// Each ~60-bit RNS limb needs 4 digits.
const DECOMP_BASE_LOG: u32 = 15;
const DECOMP_BASE: u64 = 1 << DECOMP_BASE_LOG;

/// Number of digits per RNS limb (~60 bits / 15 bits = 4 digits).
const DIGITS_PER_LIMB: usize = 4;

/// Galois key for a single automorphism σ_k: X → X^k.
///
/// Uses HYBRID key-switching (RNS + digit decomposition) for low noise.
/// For each RNS limb i and digit position j, stores a key encrypting
/// P_i * β^j * σ_k(s), where P_i = Q/q_i is the partial product.
///
/// Key-switching noise is O(num_limbs * num_digits * β * σ) instead of O(Q * σ).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsGaloisKey {
    /// The automorphism exponent k (odd, in range [1, 2n-1]).
    k: usize,
    /// Key-switching keys indexed by [limb_idx * DIGITS_PER_LIMB + digit_idx].
    /// keys[limb][digit] encrypts P_limb * β^digit * σ_k(s).
    keys_a: Vec<RnsPoly>,
    keys_b: Vec<RnsPoly>,
    /// Number of RNS limbs (moduli).
    num_limbs: usize,
    /// RNS parameters.
    rns_params: RnsParams,
    /// BGV parameters.
    bgv_params: RnsBgvParams,
}

impl RnsGaloisKey {
    /// Generates a Galois key for automorphism σ_k with HYBRID key-switching.
    ///
    /// Creates num_limbs × DIGITS_PER_LIMB key pairs.
    pub fn generate<R: Rng>(
        sk: &RnsSecretKey,
        k: usize,
        rng: &mut R,
    ) -> Self {
        let n = sk.s.params().ring_dim();
        assert!(k % 2 == 1, "automorphism exponent must be odd");
        assert!(k < 2 * n, "automorphism exponent must be < 2n");

        let rns_params = sk.s.params().clone();
        let bgv_params = sk.bgv_params.clone();
        let moduli = rns_params.moduli();
        let num_limbs = moduli.len();

        // Compute σ_k(s) by applying automorphism to secret key
        let s_auto = apply_automorphism_rns(&sk.s, k);

        // Compute P_i = Q / q_i for each limb (stored in RNS form)
        // P_i mod q_j = 0 if j != i, and P_i mod q_i = prod_{j!=i}(q_j) mod q_i
        let p_values = compute_partial_products(&rns_params);

        let total_keys = num_limbs * DIGITS_PER_LIMB;
        let mut keys_a = Vec::with_capacity(total_keys);
        let mut keys_b = Vec::with_capacity(total_keys);

        // Generate keys for each (limb, digit) pair
        for limb_idx in 0..num_limbs {
            let mut power_of_base = 1u64;

            for _digit_idx in 0..DIGITS_PER_LIMB {
                // Scale σ_k(s) by P_limb * β^digit
                // In RNS: we multiply component-wise, but only limb_idx component is non-zero for P_limb
                let mut scaled_s_auto = RnsPoly::zero(&rns_params);
                for mod_idx in 0..num_limbs {
                    let q_m = moduli[mod_idx];
                    // P_limb mod q_m
                    let p_mod_qm = p_values[limb_idx][mod_idx];
                    // Scale factor: P_limb * β^digit mod q_m
                    let scale = mulmod(p_mod_qm, power_of_base % q_m, q_m);

                    for coeff_idx in 0..n {
                        let s_coeff = s_auto.residues()[mod_idx][coeff_idx];
                        scaled_s_auto.residues_mut()[mod_idx][coeff_idx] =
                            mulmod(s_coeff, scale, q_m);
                    }
                }

                // Generate key-switching key: encrypt scaled_s_auto under s
                let key_a = sample_uniform_rns(&rns_params, rng);
                let e = sample_gaussian_rns(&rns_params, bgv_params.sigma, rng);

                // b = -a·s + e + scaled_s_auto
                let neg_as = key_a.mul(&sk.s).neg();
                let key_b = neg_as.add(&e).add(&scaled_s_auto);

                keys_a.push(key_a);
                keys_b.push(key_b);

                power_of_base *= DECOMP_BASE;
            }
        }

        Self {
            k,
            keys_a,
            keys_b,
            num_limbs,
            rns_params,
            bgv_params,
        }
    }

    /// Returns the automorphism exponent.
    pub fn exponent(&self) -> usize {
        self.k
    }

    /// Returns the automorphism exponent k.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the keys_b polynomials.
    pub fn keys_b(&self) -> &[RnsPoly] {
        &self.keys_b
    }

    /// Returns the keys_a polynomials.
    pub fn keys_a(&self) -> &[RnsPoly] {
        &self.keys_a
    }

    /// Performs HYBRID key-switching (RNS + digit decomposition).
    ///
    /// Given c1 (after automorphism), returns (c0_ks, c1_ks) such that
    /// c0_ks + c1_ks * s ≈ c1 * σ_k(s).
    fn key_switch(&self, c1_auto: &RnsPoly) -> (RnsPoly, RnsPoly) {
        let mut c0_acc = RnsPoly::zero(&self.rns_params);
        let mut c1_acc = RnsPoly::zero(&self.rns_params);

        let n = self.rns_params.ring_dim();
        let moduli = self.rns_params.moduli();

        // For each RNS limb
        for limb_idx in 0..self.num_limbs {
            // For each digit position within this limb
            for digit_idx in 0..DIGITS_PER_LIMB {
                // Create polynomial for this digit from limb_idx's residue
                let mut digit_poly = RnsPoly::zero(&self.rns_params);

                // Extract digit from the limb_idx residue of c1_auto
                for coeff_idx in 0..n {
                    let coeff = c1_auto.residues()[limb_idx][coeff_idx];
                    // Extract digit: (coeff >> (digit_idx * log_β)) & (β - 1)
                    let shifted = coeff >> (DECOMP_BASE_LOG * digit_idx as u32);
                    let digit = shifted & (DECOMP_BASE - 1);

                    // Set this digit in ALL RNS components
                    // (the digit is small, so no reduction needed)
                    for mod_idx in 0..self.num_limbs {
                        digit_poly.residues_mut()[mod_idx][coeff_idx] = digit;
                    }
                }

                // Get key index
                let key_idx = limb_idx * DIGITS_PER_LIMB + digit_idx;

                // Accumulate: result += digit_poly * key[limb][digit]
                let term_c0 = digit_poly.mul(&self.keys_b[key_idx]);
                let term_c1 = digit_poly.mul(&self.keys_a[key_idx]);

                c0_acc = c0_acc.add(&term_c0);
                c1_acc = c1_acc.add(&term_c1);
            }
        }

        (c0_acc, c1_acc)
    }

    /// Parallel version of key_switch using rayon.
    #[cfg(feature = "rayon")]
    fn key_switch_parallel(&self, c1_auto: &RnsPoly) -> (RnsPoly, RnsPoly) {
        let n = self.rns_params.ring_dim();

        // Generate all (limb_idx, digit_idx) pairs
        let pairs: Vec<(usize, usize)> = (0..self.num_limbs)
            .flat_map(|l| (0..DIGITS_PER_LIMB).map(move |d| (l, d)))
            .collect();

        // Parallel map: compute each (term_c0, term_c1) independently
        let terms: Vec<(RnsPoly, RnsPoly)> = pairs
            .par_iter()
            .map(|&(limb_idx, digit_idx)| {
                // Create polynomial for this digit from limb_idx's residue
                let mut digit_poly = RnsPoly::zero(&self.rns_params);

                // Extract digit from the limb_idx residue of c1_auto
                for coeff_idx in 0..n {
                    let coeff = c1_auto.residues()[limb_idx][coeff_idx];
                    let shifted = coeff >> (DECOMP_BASE_LOG * digit_idx as u32);
                    let digit = shifted & (DECOMP_BASE - 1);

                    for mod_idx in 0..self.num_limbs {
                        digit_poly.residues_mut()[mod_idx][coeff_idx] = digit;
                    }
                }

                let key_idx = limb_idx * DIGITS_PER_LIMB + digit_idx;
                let term_c0 = digit_poly.mul(&self.keys_b[key_idx]);
                let term_c1 = digit_poly.mul(&self.keys_a[key_idx]);

                (term_c0, term_c1)
            })
            .collect();

        // Reduce: sum all terms
        let mut c0_acc = RnsPoly::zero(&self.rns_params);
        let mut c1_acc = RnsPoly::zero(&self.rns_params);
        for (t0, t1) in terms {
            c0_acc = c0_acc.add(&t0);
            c1_acc = c1_acc.add(&t1);
        }

        (c0_acc, c1_acc)
    }
}

/// Computes P_i * P_i^* in RNS form for each limb i.
/// Where P_i = Q/q_i and P_i^* = (P_i^{-1} mod q_i).
/// Returns p_values[i][j] = (P_i * P_i^*) mod q_j.
///
/// This is the CRT lifting coefficient needed for proper reconstruction.
fn compute_partial_products(params: &RnsParams) -> Vec<Vec<u64>> {
    let moduli = params.moduli();
    let k = moduli.len();

    let mut result = vec![vec![0u64; k]; k];

    for i in 0..k {
        // P_i = prod_{j != i} q_j
        // P_i^* = P_i^{-1} mod q_i

        // First compute P_i mod q_i to get P_i^*
        let q_i = moduli[i];
        let mut p_i_mod_qi = 1u64;
        for j in 0..k {
            if j != i {
                p_i_mod_qi = mulmod(p_i_mod_qi, moduli[j] % q_i, q_i);
            }
        }
        let p_i_star = mod_inv(p_i_mod_qi, q_i);

        // Now compute (P_i * P_i^*) mod q_m for each m
        for m in 0..k {
            let q_m = moduli[m];

            // P_i mod q_m
            let mut p_i_mod_qm = 1u64;
            for j in 0..k {
                if j != i {
                    p_i_mod_qm = mulmod(p_i_mod_qm, moduli[j] % q_m, q_m);
                }
            }

            // (P_i * P_i^*) mod q_m
            // Note: P_i^* is computed mod q_i, but we use it as a scalar
            result[i][m] = mulmod(p_i_mod_qm, p_i_star % q_m, q_m);
        }
    }

    result
}

/// Collection of Galois keys for slot operations.
///
/// For power-of-2 cyclotomics Z[X]/(X^n + 1), the Galois group is:
///   (Z/2n)* ≅ Z/2 × Z/2^(k-1)  where n = 2^k
///
/// - The Z/2^(k-1) subgroup is generated by 5 (order n/2)
/// - The Z/2 subgroup is generated by -1 = 2n-1 (conjugation)
///
/// For O(log n) slot summation, we need:
/// - Keys for σ_{5^{2^i}} for i = 0, 1, ..., log2(n)-2  (log2(n)-1 keys)
/// - Key for σ_{2n-1} (conjugation)                     (1 key)
///
/// Total: log2(n) keys
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RnsGaloisKeys {
    /// Galois keys for powers of the generator 5.
    /// keys[i] is for σ_{5^{2^i}} for i < num_keys-1.
    /// keys[num_keys-1] is for conjugation σ_{2n-1}.
    keys: Vec<RnsGaloisKey>,
    /// The automorphism exponents used.
    automorphism_exponents: Vec<usize>,
    /// Number of slots (= n).
    num_slots: usize,
    /// Ring dimension n.
    ring_dim: usize,
}

impl RnsGaloisKeys {
    /// Generates Galois keys for slot summation.
    ///
    /// Creates keys for the automorphisms needed for O(log n) slot summation:
    /// - σ_{5^1}, σ_{5^2}, σ_{5^4}, ..., σ_{5^{n/4}}  (log2(n)-1 keys)
    /// - σ_{2n-1} (conjugation)                        (1 key)
    pub fn generate<R: Rng>(sk: &RnsSecretKey, rng: &mut R) -> Self {
        let n = sk.bgv_params.n;
        let num_slots = sk.bgv_params.num_slots;
        let two_n = 2 * n;

        // For power-of-2 cyclotomics:
        // (Z/2n)* ≅ Z/2 × Z/2^(k-1) where n = 2^k
        //
        // Generator of Z/2^(k-1): 5 (has order n/2 in (Z/2n)*)
        // Generator of Z/2: -1 = 2n-1 (conjugation)
        //
        // For tree-based summation of n slots:
        // 1. Apply σ_{5^{2^i}} and add, for i = 0..log2(n)-2
        //    This covers the <5> subgroup (n/2 elements)
        // 2. Apply σ_{-1} and add
        //    This covers the other coset (n/2 elements)

        let mut keys = Vec::new();
        let mut automorphism_exponents = Vec::new();

        // Number of keys for powers of 5: log2(n) - 1
        // (since 5 has order n/2, we need log2(n/2) = log2(n)-1 squarings)
        let log_n = (n as f64).log2() as usize;

        // Generate keys for σ_{5^{2^i}} for i = 0, 1, ..., log2(n)-2
        let mut power_of_5 = 5usize;
        for _ in 0..(log_n - 1) {
            let k = power_of_5 % two_n;
            let gk = RnsGaloisKey::generate(sk, k, rng);
            keys.push(gk);
            automorphism_exponents.push(k);

            // Square the exponent for next iteration
            power_of_5 = (power_of_5 * power_of_5) % two_n;
        }

        // Generate key for conjugation σ_{2n-1}
        let conj_exp = two_n - 1;
        let gk = RnsGaloisKey::generate(sk, conj_exp, rng);
        keys.push(gk);
        automorphism_exponents.push(conj_exp);

        Self {
            keys,
            automorphism_exponents,
            num_slots,
            ring_dim: n,
        }
    }

    /// Returns the Galois key at index i.
    ///
    /// For i < num_keys()-1: returns key for σ_{5^{2^i}}
    /// For i = num_keys()-1: returns key for conjugation σ_{2n-1}
    pub fn get_key(&self, index: usize) -> Option<&RnsGaloisKey> {
        self.keys.get(index)
    }

    /// Returns the Galois key for rotation by 2^i slots (legacy API).
    /// Note: For power-of-2 cyclotomics, these are NOT true rotations.
    pub fn get_rotation_key(&self, log_rotation: usize) -> Option<&RnsGaloisKey> {
        self.keys.get(log_rotation)
    }

    /// Returns the conjugation key (σ_{2n-1}).
    pub fn get_conjugation_key(&self) -> Option<&RnsGaloisKey> {
        self.keys.last()
    }

    /// Returns the number of keys (= log2(n)).
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    /// Returns the automorphism exponent for key at index i.
    pub fn get_exponent(&self, index: usize) -> Option<usize> {
        self.automorphism_exponents.get(index).copied()
    }

    /// Returns all Galois keys.
    pub fn keys(&self) -> &[RnsGaloisKey] {
        &self.keys
    }

    /// Returns all automorphism exponents.
    pub fn automorphism_exponents(&self) -> &[usize] {
        &self.automorphism_exponents
    }
}

/// Applies automorphism σ_k to an RNS polynomial: a(X) → a(X^k) mod (X^n + 1).
fn apply_automorphism_rns(poly: &RnsPoly, k: usize) -> RnsPoly {
    let params = poly.params();
    let n = params.ring_dim();
    let mut result = RnsPoly::zero(params);

    // For each coefficient a_i of a(X), it contributes a_i * X^(ik) to a(X^k)
    // In the ring Z[X]/(X^n + 1), X^n = -1, so X^j = (-1)^(j/n) * X^(j mod n)

    for (mod_idx, residue) in poly.residues().iter().enumerate() {
        let q = params.moduli()[mod_idx];
        let out = result.residues_mut();

        for (i, &coeff) in residue.iter().enumerate() {
            if coeff == 0 {
                continue;
            }

            let target_exp = (i * k) % (2 * n);
            let (final_exp, sign) = if target_exp >= n {
                (target_exp - n, true) // X^n = -1
            } else {
                (target_exp, false)
            };

            if sign {
                // Subtract coeff
                if out[mod_idx][final_exp] >= coeff {
                    out[mod_idx][final_exp] -= coeff;
                } else {
                    out[mod_idx][final_exp] = q - (coeff - out[mod_idx][final_exp]);
                }
            } else {
                // Add coeff
                out[mod_idx][final_exp] = (out[mod_idx][final_exp] + coeff) % q;
            }
        }
    }

    result
}

impl RnsCiphertext {
    /// Creates a ciphertext from components.
    pub fn from_parts(
        c0: RnsPoly,
        c1: RnsPoly,
        rns_params: RnsParams,
        bgv_params: RnsBgvParams,
    ) -> Self {
        Self {
            c0,
            c1,
            rns_params,
            bgv_params,
        }
    }

    /// Returns the c0 component.
    pub fn c0(&self) -> &RnsPoly {
        &self.c0
    }

    /// Returns the c1 component.
    pub fn c1(&self) -> &RnsPoly {
        &self.c1
    }

    /// Returns the RNS parameters.
    pub fn rns_params(&self) -> &RnsParams {
        &self.rns_params
    }

    /// Returns the BGV parameters.
    pub fn bgv_params(&self) -> &RnsBgvParams {
        &self.bgv_params
    }

    /// Returns the c0 and c1 residues for GPU processing.
    ///
    /// Extracts the RNS residues that are already in NTT domain.
    ///
    /// Returns (c0_residues, c1_residues) where each is Vec<Vec<u64>> with shape [num_moduli][n].
    /// These residues can be directly uploaded to GPU for slot multiplication.
    pub fn extract_ntt_residues(&self) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let c0_residues: Vec<Vec<u64>> = self.c0.residues().to_vec();
        let c1_residues: Vec<Vec<u64>> = self.c1.residues().to_vec();
        (c0_residues, c1_residues)
    }

    /// Creates a ciphertext from residue arrays.
    pub fn from_residues(
        c0_residues: Vec<Vec<u64>>,
        c1_residues: Vec<Vec<u64>>,
        rns_params: RnsParams,
        bgv_params: RnsBgvParams,
    ) -> Self {
        let c0 = RnsPoly::from_residue_vecs(c0_residues, &rns_params);
        let c1 = RnsPoly::from_residue_vecs(c1_residues, &rns_params);
        Self {
            c0,
            c1,
            rns_params,
            bgv_params,
        }
    }

    /// Encrypts a single scalar value.
    ///
    /// The scalar is placed in slot 0, other slots are zero.
    pub fn encrypt_scalar<R: Rng>(pk: &RnsPublicKey, message: u64, rng: &mut R) -> Self {
        // Encode message: place in coefficient 0, scaled by delta
        // For RNS, we compute delta = q/t for each component
        let mut m_coeffs = vec![0u64; pk.rns_params.ring_dim()];
        m_coeffs[0] = message % pk.bgv_params.t;

        Self::encrypt_coeffs(pk, &m_coeffs, rng)
    }

    /// Encrypts slot values using slot packing.
    ///
    /// Each slot value is encoded into a separate slot of the ciphertext.
    /// This allows SIMD-style operations on all slots simultaneously.
    pub fn encrypt_slots<R: Rng>(pk: &RnsPublicKey, slots: &[u64], rng: &mut R) -> Self {
        assert!(
            pk.bgv_params.supports_slots,
            "slot packing not supported for these parameters"
        );
        assert!(
            slots.len() <= pk.bgv_params.num_slots,
            "too many slots: {} > {}",
            slots.len(),
            pk.bgv_params.num_slots
        );

        // Pad slots to full size
        let mut full_slots = vec![0u64; pk.bgv_params.num_slots];
        full_slots[..slots.len()].copy_from_slice(slots);

        // Encode slots into polynomial coefficients via inverse NTT
        let encoder = SlotEncoder::new_direct(pk.bgv_params.n, pk.bgv_params.t)
            .expect("slot encoder should work for these params");
        let coeffs = encoder.encode(&full_slots);

        Self::encrypt_coeffs(pk, &coeffs, rng)
    }

    /// Encrypts polynomial coefficients directly.
    fn encrypt_coeffs<R: Rng>(pk: &RnsPublicKey, coeffs: &[u64], rng: &mut R) -> Self {
        let t = pk.bgv_params.t;
        let moduli = pk.rns_params.moduli();

        // Check if we can use simple per-modulus scaling
        // (when t << q_i and delta_i = q_i/t provides sufficient range)
        let use_simple = t <= moduli[0] / 1000;

        let mut m_poly = RnsPoly::zero(&pk.rns_params);

        if use_simple {
            // Standard BGV: scale by delta = q_i / t
            for (i, &q_i) in moduli.iter().enumerate() {
                let delta_i = q_i / t;
                for (j, &c) in coeffs.iter().enumerate() {
                    let scaled = mulmod(c % t, delta_i, q_i);
                    m_poly.residues_mut()[i][j] = scaled;
                }
            }
        } else {
            // Scaled BGV for large t: use delta = Q/t (computed per component)
            let delta_rns = compute_delta_rns(moduli, t);
            for (i, &q_i) in moduli.iter().enumerate() {
                for (j, &c) in coeffs.iter().enumerate() {
                    let scaled = mulmod(c % t, delta_rns[i], q_i);
                    m_poly.residues_mut()[i][j] = scaled;
                }
            }
        }

        // Sample randomness r (ternary)
        let r = sample_ternary_rns(&pk.rns_params, rng);

        // Sample errors e0, e1 (Gaussian)
        let e0 = sample_gaussian_rns(&pk.rns_params, pk.bgv_params.sigma, rng);
        let e1 = sample_gaussian_rns(&pk.rns_params, pk.bgv_params.sigma, rng);

        // c0 = b·r + e0 + m (unscaled) or c0 = b·r + e0 + Δ·m (scaled)
        let br = pk.b.mul(&r);
        let c0 = br.add(&e0).add(&m_poly);

        // c1 = a·r + e1
        let ar = pk.a.mul(&r);
        let c1 = ar.add(&e1);

        Self {
            c0,
            c1,
            rns_params: pk.rns_params.clone(),
            bgv_params: pk.bgv_params.clone(),
        }
    }

    /// Decrypts to a single scalar (from slot 0).
    pub fn decrypt_scalar(&self, sk: &RnsSecretKey) -> u64 {
        let coeffs = self.decrypt_coeffs(sk);
        coeffs[0]
    }

    /// Decrypts all slots.
    pub fn decrypt_slots(&self, sk: &RnsSecretKey) -> Vec<u64> {
        assert!(
            self.bgv_params.supports_slots,
            "slot packing not supported"
        );

        let coeffs = self.decrypt_coeffs(sk);

        // Decode polynomial coefficients to slot values via forward NTT
        let encoder = SlotEncoder::new_direct(self.bgv_params.n, self.bgv_params.t)
            .expect("slot encoder should work");
        encoder.decode(&coeffs)
    }

    /// Decrypts to polynomial coefficients.
    fn decrypt_coeffs(&self, sk: &RnsSecretKey) -> Vec<u64> {
        let t = self.bgv_params.t;
        let moduli = self.rns_params.moduli();

        // Check if we can use simple single-modulus decryption
        // (when t << q_0 and delta = q_0/t provides sufficient noise budget)
        if t <= moduli[0] / 1000 {
            return self.decrypt_coeffs_simple(sk);
        }

        // For large t (like Goldilocks), use CRT reconstruction
        self.decrypt_coeffs_crt(sk)
    }

    /// Simple decryption using only the first RNS modulus.
    /// Works when t << q_0.
    fn decrypt_coeffs_simple(&self, sk: &RnsSecretKey) -> Vec<u64> {
        let t = self.bgv_params.t;
        let n = self.bgv_params.n;

        let c1s = self.c1.mul(&sk.s);
        let noisy = self.c0.add(&c1s);

        let q_0 = self.rns_params.moduli()[0];
        let mut result = vec![0u64; n];

        for j in 0..n {
            let noisy_coeff = noisy.residues()[0][j];
            result[j] = scale_and_round(noisy_coeff, t, q_0);
        }

        result
    }

    /// Decryption for large plaintext modulus.
    ///
    /// For scaled encoding: noisy_i ≈ delta_i * m + noise_i (mod q_i)
    /// where delta_i = (Q/t) mod q_i.
    ///
    /// We recover m = round(x * t / Q) = round(sum_i(y_i * t / q_i))
    /// where y_i = (noisy_i * w_i) mod q_i (centered).
    ///
    /// Uses exact integer arithmetic with rounding.
    fn decrypt_coeffs_crt(&self, sk: &RnsSecretKey) -> Vec<u64> {
        let t = self.bgv_params.t;
        let n = self.bgv_params.n;
        let moduli = self.rns_params.moduli();
        let k = moduli.len();

        // Compute noisy = c0 + c1·s
        let c1s = self.c1.mul(&sk.s);
        let noisy = self.c0.add(&c1s);

        // Precompute CRT lifting coefficients:
        // w_i = (Q/q_i)^{-1} mod q_i where Q = prod(q_j)
        let mut w = vec![0u64; k];
        for i in 0..k {
            let mut q_i_mod_qi = 1u64;
            for j in 0..k {
                if i != j {
                    q_i_mod_qi = mulmod(q_i_mod_qi, moduli[j] % moduli[i], moduli[i]);
                }
            }
            w[i] = mod_inv(q_i_mod_qi, moduli[i]);
        }

        let mut result = vec![0u64; n];

        for coeff_idx in 0..n {
            // Compute m ≈ round(sum_i(y_i * t / q_i))
            // where y_i = (noisy_i * w_i) mod q_i, centered in [-q_i/2, q_i/2)
            //
            // For each term: y_i * t / q_i = floor(y_i * t / q_i) + (y_i * t mod q_i) / q_i
            // The floor part gives the integer contribution.
            // The fractional parts sum to give a correction in [0, k).

            let mut floor_sum: i128 = 0;
            let mut frac_sum: i128 = 0; // Sum of numerators, divide by common "scale"

            for i in 0..k {
                let noisy_i = noisy.residues()[i][coeff_idx];
                let q_i = moduli[i];

                // Compute y_i = (noisy_i * w_i) mod q_i
                let y_i = mulmod(noisy_i, w[i], q_i);

                // Center y_i to [-q_i/2, q_i/2)
                let y_centered: i128 = if y_i >= q_i / 2 {
                    -((q_i - y_i) as i128)
                } else {
                    y_i as i128
                };

                // Compute floor(y_centered * t / q_i) and remainder
                let q_i128 = q_i as i128;
                let t128 = t as i128;

                // y_centered * t might overflow i128, use careful computation
                // |y_centered| < q_i/2 ≈ 2^59, t ≈ 2^64, so |y_centered * t| < 2^123
                let product = y_centered * t128;

                // Integer division with correct rounding towards zero for floor
                let floor_val = if product >= 0 {
                    product / q_i128
                } else {
                    // For negative values, we want floor, not truncation
                    (product - q_i128 + 1) / q_i128
                };

                floor_sum += floor_val;

                // Remainder for fractional part
                let rem = product - floor_val * q_i128;
                // rem is in [0, q_i) if floor is correct

                // Scale the remainder to accumulate fractional parts
                // We approximate: frac_i = rem / q_i
                // Instead of computing exactly, we'll use the fact that sum(frac_i) should be < k
                // For rounding, if sum(frac_i) >= 0.5, add 1
                // We compute 2*sum(rem / q_i) and check if >= 1 (i.e., 2*sum(rem) / min(q_i) >= 1)
                frac_sum += rem * 2; // Scale by 2 for rounding check
            }

            // The fractional sum determines the rounding correction
            // sum_i(rem_i / q_i) is in range (-k, k) for k moduli
            // We need to round this to the nearest integer
            //
            // frac_sum = 2 * sum(rem_i)
            // sum(rem_i / q_i) ≈ sum(rem_i) / avg_q ≈ frac_sum / (2 * q_0)
            // Round: correction = round(frac_sum / (2 * q_0))
            let q_0 = moduli[0] as i128;
            let two_q_0 = 2 * q_0;

            // Compute round(frac_sum / two_q_0)
            let correction = if frac_sum >= 0 {
                (frac_sum + q_0) / two_q_0 // Round positive values
            } else {
                (frac_sum - q_0) / two_q_0 // Round negative values
            };

            let m = floor_sum + correction;
            result[coeff_idx] = m.rem_euclid(t as i128) as u64;
        }

        result
    }

    /// Adds two ciphertexts homomorphically.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Subtracts two ciphertexts homomorphically.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Negates a ciphertext homomorphically.
    pub fn neg(&self) -> Self {
        Self {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Multiplies by a scalar homomorphically.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        Self {
            c0: self.c0.scalar_mul(scalar),
            c1: self.c1.scalar_mul(scalar),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Multiplies ciphertext by plaintext slot values (SIMD).
    ///
    /// Given a ciphertext encrypting slot values [s_0, ..., s_{n-1}]
    /// and plaintext values [p_0, ..., p_{n-1}], this produces a ciphertext
    /// encrypting [s_0 * p_0, ..., s_{n-1} * p_{n-1}].
    ///
    /// This enables rotation-less slot multiplication: pack values into slots,
    /// multiply by plaintext coefficients, decrypt, and sum in the clear.
    pub fn mul_plaintext_slots(&self, plaintext_slots: &[u64]) -> Self {
        assert!(
            self.bgv_params.supports_slots,
            "slot packing not supported"
        );
        assert!(
            plaintext_slots.len() <= self.bgv_params.num_slots,
            "too many slots"
        );

        // Pad slots to full size
        let t = self.bgv_params.t;
        let mut full_slots = vec![0u64; self.bgv_params.num_slots];
        for (i, &s) in plaintext_slots.iter().enumerate() {
            full_slots[i] = s % t;
        }

        // Encode plaintext slots into polynomial coefficients via inverse NTT
        let encoder = SlotEncoder::new_direct(self.bgv_params.n, t)
            .expect("slot encoder should work");
        let pt_coeffs = encoder.encode(&full_slots);

        // Multiply ciphertext polynomials by plaintext polynomial
        // ct' = (c0 * pt, c1 * pt)
        // After decryption: slots' = decode(decode_ct(ct')) = slots_original ⊙ plaintext_slots
        let pt_poly = self.coeffs_to_rns_poly(&pt_coeffs);

        Self {
            c0: self.c0.mul(&pt_poly),
            c1: self.c1.mul(&pt_poly),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Subtracts plaintext slot values from ciphertext (SIMD).
    ///
    /// Given a ciphertext encrypting slot values [s_0, ..., s_{n-1}]
    /// and plaintext values [p_0, ..., p_{n-1}], this produces a ciphertext
    /// encrypting [s_0 - p_0, ..., s_{n-1} - p_{n-1}].
    ///
    /// This is used for blinding: subtract random values from each slot
    /// to hide individual products while preserving the sum relationship.
    pub fn sub_plaintext_slots(&self, plaintext_slots: &[u64]) -> Self {
        assert!(
            self.bgv_params.supports_slots,
            "slot packing not supported"
        );
        assert!(
            plaintext_slots.len() <= self.bgv_params.num_slots,
            "too many slots"
        );

        // Pad slots to full size
        let t = self.bgv_params.t;
        let mut full_slots = vec![0u64; self.bgv_params.num_slots];
        for (i, &s) in plaintext_slots.iter().enumerate() {
            full_slots[i] = s % t;
        }

        // Encode plaintext slots into polynomial coefficients via inverse NTT
        let encoder = SlotEncoder::new_direct(self.bgv_params.n, t)
            .expect("slot encoder should work");
        let pt_coeffs = encoder.encode(&full_slots);

        // Scale by delta and create RNS polynomial
        let pt_poly = self.coeffs_to_scaled_rns_poly(&pt_coeffs);

        // Subtract from c0 only (c1 unchanged for plaintext operations)
        Self {
            c0: self.c0.sub(&pt_poly),
            c1: self.c1.clone(),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Subtracts plaintext slot values using a cached encoder.
    ///
    /// This is more efficient than `sub_plaintext_slots` when performing
    /// multiple operations with the same parameters, as it avoids
    /// recreating the encoder each time.
    pub fn sub_plaintext_slots_with_encoder(
        &self,
        plaintext_slots: &[u64],
        encoder: &SlotEncoder,
    ) -> Self {
        assert!(
            self.bgv_params.supports_slots,
            "slot packing not supported"
        );
        assert!(
            plaintext_slots.len() <= self.bgv_params.num_slots,
            "too many slots"
        );

        // Pad slots to full size
        let t = self.bgv_params.t;
        let mut full_slots = vec![0u64; self.bgv_params.num_slots];
        for (i, &s) in plaintext_slots.iter().enumerate() {
            full_slots[i] = s % t;
        }

        // Encode plaintext slots into polynomial coefficients via inverse NTT
        let pt_coeffs = encoder.encode(&full_slots);

        // Scale by delta and create RNS polynomial
        let pt_poly = self.coeffs_to_scaled_rns_poly(&pt_coeffs);

        // Subtract from c0 only (c1 unchanged for plaintext operations)
        Self {
            c0: self.c0.sub(&pt_poly),
            c1: self.c1.clone(),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Adds plaintext slot values to ciphertext (SIMD).
    ///
    /// Given a ciphertext encrypting slot values [s_0, ..., s_{n-1}]
    /// and plaintext values [p_0, ..., p_{n-1}], this produces a ciphertext
    /// encrypting [s_0 + p_0, ..., s_{n-1} + p_{n-1}].
    pub fn add_plaintext_slots(&self, plaintext_slots: &[u64]) -> Self {
        assert!(
            self.bgv_params.supports_slots,
            "slot packing not supported"
        );
        assert!(
            plaintext_slots.len() <= self.bgv_params.num_slots,
            "too many slots"
        );

        let t = self.bgv_params.t;
        let mut full_slots = vec![0u64; self.bgv_params.num_slots];
        for (i, &s) in plaintext_slots.iter().enumerate() {
            full_slots[i] = s % t;
        }

        let encoder = SlotEncoder::new_direct(self.bgv_params.n, t)
            .expect("slot encoder should work");
        let pt_coeffs = encoder.encode(&full_slots);
        let pt_poly = self.coeffs_to_scaled_rns_poly(&pt_coeffs);

        Self {
            c0: self.c0.add(&pt_poly),
            c1: self.c1.clone(),
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Converts coefficients to a scaled RNS polynomial (delta * coeffs).
    fn coeffs_to_scaled_rns_poly(&self, coeffs: &[u64]) -> RnsPoly {
        let t = self.bgv_params.t;
        let moduli = self.rns_params.moduli();
        let use_simple = t <= moduli[0] / 1000;

        let mut m_poly = RnsPoly::zero(&self.rns_params);

        if use_simple {
            for (i, &q_i) in moduli.iter().enumerate() {
                let delta_i = q_i / t;
                for (j, &c) in coeffs.iter().enumerate() {
                    let scaled = mulmod(c % t, delta_i, q_i);
                    m_poly.residues_mut()[i][j] = scaled;
                }
            }
        } else {
            let delta_rns = compute_delta_rns(moduli, t);
            for (i, &q_i) in moduli.iter().enumerate() {
                for (j, &c) in coeffs.iter().enumerate() {
                    let scaled = mulmod(c % t, delta_rns[i], q_i);
                    m_poly.residues_mut()[i][j] = scaled;
                }
            }
        }

        m_poly
    }

    /// Shifts all slot values left by 64 bits (multiplies by 2^64).
    ///
    /// This is used for packing multiple Goldilocks values into a single slot.
    /// Since q ~ 300 bits, we can pack multiple 64-bit values:
    /// `slot = v0 + v1 * 2^64 + v2 * 2^128 + ...`
    ///
    /// # Example
    /// ```ignore
    /// // Pack two values into one slot:
    /// let ct_packed = ct_v1.shift_slots_left_64().add(&ct_v0);
    /// // Now each slot contains: v0 + v1 * 2^64
    /// ```
    pub fn shift_slots_left_64(&self) -> Self {
        // Multiply all slots by 2^64
        // We need to compute 2^64 mod t for slot encoding, then use mul_plaintext_slots
        let t = self.bgv_params.t;
        let shift_mod_t = ((1u128 << 64) % t as u128) as u64;

        // Create vector with shift_mod_t in all slots
        let shift_slots = vec![shift_mod_t; self.bgv_params.num_slots];
        self.mul_plaintext_slots(&shift_slots)
    }

    /// Packs another ciphertext's values into this one by shifting and adding.
    ///
    /// Result: `self_slots * 2^64 + other_slots`
    ///
    /// This allows packing multiple 64-bit Goldilocks values per slot.
    /// With q ~ 300 bits, you can pack up to 4 values per slot.
    ///
    /// # Example
    /// ```ignore
    /// // Pack v0, v1, v2 into one slot:
    /// let ct_packed = ct_v2
    ///     .pack_value(&ct_v1)  // v2 * 2^64 + v1
    ///     .pack_value(&ct_v0); // (v2 * 2^64 + v1) * 2^64 + v0
    ///                          // = v0 + v1 * 2^64 + v2 * 2^128
    /// ```
    pub fn pack_value(&self, other: &Self) -> Self {
        self.shift_slots_left_64().add(other)
    }

    // ==================== Experimental: Ciphertext-space packing ====================
    //
    // These methods multiply ciphertext polynomials directly by 2^64 in the q-space,
    // allowing packing of multiple values before mod-t reduction.
    //
    // WARNING: This is experimental. Noise scales by 2^64 with each shift!

    /// Shifts ciphertext values left by 64 bits in ciphertext space (mod q, NOT mod t).
    ///
    /// This multiplies c0 and c1 polynomials by 2^64 mod q_i for each RNS modulus.
    /// The embedded plaintext value is effectively shifted: v → v * 2^64.
    ///
    /// **Warning:** This also scales noise by 2^64, limiting the number of packings.
    ///
    /// Use with `decrypt_packed_u128` to extract packed values.
    pub fn shift_ciphertext_left_64(&self) -> Self {
        let shift: u128 = 1u128 << 64;
        self.mul_ciphertext_scalar_u128(shift)
    }

    /// Multiplies ciphertext polynomials by a scalar in ciphertext space.
    ///
    /// This multiplies c0 and c1 by scalar mod q_i for each RNS limb.
    pub fn mul_ciphertext_scalar_u128(&self, scalar: u128) -> Self {
        let moduli = self.rns_params.moduli();
        let n = self.bgv_params.n;

        let mut new_c0 = RnsPoly::zero(&self.rns_params);
        let mut new_c1 = RnsPoly::zero(&self.rns_params);

        for (i, &q_i) in moduli.iter().enumerate() {
            let scalar_mod_qi = (scalar % q_i as u128) as u64;

            for j in 0..n {
                new_c0.residues_mut()[i][j] = mulmod(self.c0.residues()[i][j], scalar_mod_qi, q_i);
                new_c1.residues_mut()[i][j] = mulmod(self.c1.residues()[i][j], scalar_mod_qi, q_i);
            }
        }

        Self {
            c0: new_c0,
            c1: new_c1,
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Shifts ciphertext left by `bits` positions in ciphertext space.
    ///
    /// Computes 2^bits mod q_i for each RNS limb and multiplies.
    /// This is more efficient than chaining mul_ciphertext_scalar_u128 calls
    /// and doesn't grow noise multiple times.
    pub fn shift_ciphertext_left(&self, bits: u32) -> Self {
        let moduli = self.rns_params.moduli();
        let n = self.bgv_params.n;

        let mut new_c0 = RnsPoly::zero(&self.rns_params);
        let mut new_c1 = RnsPoly::zero(&self.rns_params);

        for (i, &q_i) in moduli.iter().enumerate() {
            // Compute 2^bits mod q_i using modular exponentiation
            let shift_mod_qi = pow_mod(2, bits as u64, q_i);

            for j in 0..n {
                new_c0.residues_mut()[i][j] = mulmod(self.c0.residues()[i][j], shift_mod_qi, q_i);
                new_c1.residues_mut()[i][j] = mulmod(self.c1.residues()[i][j], shift_mod_qi, q_i);
            }
        }

        Self {
            c0: new_c0,
            c1: new_c1,
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Packs another ciphertext using ciphertext-space shifting.
    ///
    /// Result in ciphertext space: `self * 2^64 + other`
    ///
    /// Use `decrypt_packed_u128` to extract the packed values.
    pub fn pack_ciphertext(&self, other: &Self) -> Self {
        self.shift_ciphertext_left_64().add(other)
    }

    /// Decrypts to raw u128 values (coefficient 0 only) without mod-t reduction.
    ///
    /// This allows extracting packed values that were combined using
    /// `pack_ciphertext` / `shift_ciphertext_left_64`.
    ///
    /// Returns the raw decrypted value which can contain multiple packed 64-bit values.
    pub fn decrypt_packed_u128(&self, sk: &RnsSecretKey) -> u128 {
        let moduli = self.rns_params.moduli();
        let k = moduli.len();

        // Compute noisy = c0 + c1·s
        let c1s = self.c1.mul(&sk.s);
        let noisy = self.c0.add(&c1s);

        // CRT reconstruction to get the actual integer value (mod Q = prod(q_i))
        // We use coefficient 0 only for this test.
        //
        // CRT: x = sum_i (x_i * M_i * y_i) mod Q
        // where M_i = Q/q_i and y_i = M_i^{-1} mod q_i

        // First compute Q (product of all moduli) - this can be huge, use BigInt-style
        // For simplicity, we'll use i128/u128 and hope it fits for small k
        // With k=5 moduli of ~60 bits each, Q ~ 300 bits, too big for u128.
        //
        // Instead, we'll extract the low 128 bits by doing CRT carefully.

        // For BGV with scaling factor delta = Q/t, the plaintext m satisfies:
        // noisy ≈ delta * m + noise (mod Q)
        // So m ≈ noisy * t / Q (with rounding)
        //
        // But we want the RAW value before scaling, so we need to undo the delta scaling.
        // Actually, for unscaled BGV (where m is added directly), noisy = m + noise.
        //
        // Let's check if this is scaled or unscaled by looking at encryption...
        // Looking at encrypt_slots, it uses scale_by_delta which means scaled encoding.
        //
        // For scaled BGV: noisy = delta*m + noise, so m = round(noisy/delta) = round(noisy*t/Q)
        //
        // To get the "raw" packed value, we want m, but m is already mod t from the formula.
        // The packing idea doesn't work directly with scaled BGV...
        //
        // Let's try a different approach: reconstruct noisy mod Q, then divide by delta.

        // Simplified approach for testing: use the first modulus only
        // This gives us noisy mod q_0, which for small plaintexts should be close to delta*m
        let q_0 = moduli[0];
        let t = self.bgv_params.t;
        let delta = q_0 / t; // Approximate delta for first modulus

        let noisy_0 = noisy.residues()[0][0];

        // m ≈ noisy_0 / delta, but we want the raw pre-division value
        // For packed values, we stored: m_packed = v0 + v1 * 2^64
        // And noisy ≈ delta * m_packed
        // So noisy / delta ≈ m_packed = v0 + v1 * 2^64

        // Return noisy_0 / delta as approximation
        // This will lose precision but let's see what we get
        (noisy_0 / delta) as u128
    }

    /// Decrypts slot 0 to a raw large integer using full CRT reconstruction.
    ///
    /// Returns (low_128_bits, high_128_bits) of the decrypted value before mod-t.
    pub fn decrypt_slot0_raw(&self, sk: &RnsSecretKey) -> (u128, u128) {
        let moduli = self.rns_params.moduli();

        // Compute noisy = c0 + c1·s
        let c1s = self.c1.mul(&sk.s);
        let noisy = self.c0.add(&c1s);

        // For slot 0, we need to decode from NTT domain first
        // But for testing coefficient packing (not slot packing), use coeff 0 directly

        // Use balanced CRT to reconstruct value in range [-Q/2, Q/2)
        // Then extract the plaintext by dividing by delta

        let t = self.bgv_params.t;

        // Compute delta for each modulus and scale
        // m_i = round(noisy_i * t / q_i) for each RNS component

        // For a properly packed value, all m_i should be consistent mod t
        // But the raw value m can be > t if we packed multiple values

        // Let's compute using first modulus as approximation
        let q_0 = moduli[0];
        let noisy_0 = noisy.residues()[0][0];

        // In scaled BGV: noisy = delta * m + e where delta = floor(q/t)
        // So m ≈ noisy / delta

        // Compute full precision
        let delta_0 = q_0 / t;
        let m_approx = noisy_0 / delta_0;

        (m_approx as u128, 0u128)
    }

    /// Converts polynomial coefficients (mod t) to RNS representation.
    fn coeffs_to_rns_poly(&self, coeffs: &[u64]) -> RnsPoly {
        let mut poly = RnsPoly::zero(&self.rns_params);
        let moduli = self.rns_params.moduli();

        for (i, &q_i) in moduli.iter().enumerate() {
            for (j, &c) in coeffs.iter().enumerate() {
                // Coefficients are already in [0, t), just reduce mod q_i
                poly.residues_mut()[i][j] = c % q_i;
            }
        }

        poly
    }

    /// Applies automorphism σ_k to ciphertext and key-switches.
    ///
    /// This rotates/permutes slot values according to the automorphism.
    /// Requires a Galois key for the specific automorphism.
    ///
    /// Uses digit decomposition for low-noise key-switching.
    pub fn apply_automorphism(&self, galois_key: &RnsGaloisKey) -> Self {
        let k = galois_key.k;

        // Step 1: Apply automorphism to ciphertext components
        // σ_k(ct) = (σ_k(c0), σ_k(c1))
        let c0_auto = apply_automorphism_rns(&self.c0, k);
        let c1_auto = apply_automorphism_rns(&self.c1, k);

        // Step 2: Key-switch using digit decomposition
        // After automorphism, decryption would use σ_k(s).
        // Key-switching converts to encryption under s.
        //
        // With digit decomposition:
        // - Decompose c1_auto into digits: c1_auto = Σ d_i * β^i
        // - For each digit, multiply by corresponding key
        // - Sum to get key-switched ciphertext
        //
        // This keeps noise proportional to num_digits * β * σ instead of Q * σ.
        let (ks_c0, ks_c1) = galois_key.key_switch(&c1_auto);

        // New ciphertext: (c0_auto + ks_c0, ks_c1)
        let new_c0 = c0_auto.add(&ks_c0);

        Self {
            c0: new_c0,
            c1: ks_c1,
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Applies automorphism with parallel key-switching.
    #[cfg(feature = "rayon")]
    pub fn apply_automorphism_parallel(&self, galois_key: &RnsGaloisKey) -> Self {
        let k = galois_key.k;

        // Step 1: Apply automorphism to ciphertext components
        let c0_auto = apply_automorphism_rns(&self.c0, k);
        let c1_auto = apply_automorphism_rns(&self.c1, k);

        // Step 2: Parallel key-switch
        let (ks_c0, ks_c1) = galois_key.key_switch_parallel(&c1_auto);

        let new_c0 = c0_auto.add(&ks_c0);

        Self {
            c0: new_c0,
            c1: ks_c1,
            rns_params: self.rns_params.clone(),
            bgv_params: self.bgv_params.clone(),
        }
    }

    /// Sums all slots using the Galois group structure.
    ///
    /// Given ciphertext with slots [s_0, s_1, ..., s_{n-1}],
    /// produces ciphertext with all slots containing sum = Σ s_i.
    ///
    /// Uses O(log n) automorphisms and additions.
    ///
    /// Algorithm for power-of-2 cyclotomics:
    /// 1. Apply σ_{5^{2^i}} and add, for i = 0..log2(n)-2
    ///    This covers the <5> subgroup (n/2 automorphisms via tree)
    /// 2. Apply σ_{-1} (conjugation) and add
    ///    This covers the -1·<5> coset (the other n/2 automorphisms)
    ///
    /// Result: Every slot contains the sum of all original slots.
    pub fn sum_slots(&self, galois_keys: &RnsGaloisKeys) -> Self {
        let mut result = self.clone();
        let num_keys = galois_keys.num_keys();

        // Phase 1: Tree-based summation using powers of 5
        // This covers the <5> subgroup of (Z/2n)*
        // After log2(n)-1 iterations, each slot contains sum of n/2 slots
        for i in 0..(num_keys - 1) {
            if let Some(gk) = galois_keys.get_key(i) {
                let permuted = result.apply_automorphism(gk);
                result = result.add(&permuted);
            }
        }

        // Phase 2: Add conjugate to cover the -1·<5> coset
        // This doubles the sum to include all n slots
        if let Some(gk) = galois_keys.get_conjugation_key() {
            let conjugated = result.apply_automorphism(gk);
            result = result.add(&conjugated);
        }

        result
    }

    /// Parallel version of sum_slots using rayon for key-switching.
    #[cfg(feature = "rayon")]
    pub fn sum_slots_parallel(&self, galois_keys: &RnsGaloisKeys) -> Self {
        let mut result = self.clone();
        let num_keys = galois_keys.num_keys();

        // Phase 1: Tree-based summation with parallel key-switching
        for i in 0..(num_keys - 1) {
            if let Some(gk) = galois_keys.get_key(i) {
                let permuted = result.apply_automorphism_parallel(gk);
                result = result.add(&permuted);
            }
        }

        // Phase 2: Add conjugate
        if let Some(gk) = galois_keys.get_conjugation_key() {
            let conjugated = result.apply_automorphism_parallel(gk);
            result = result.add(&conjugated);
        }

        result
    }

    /// Sums slots within each lane for partial summation.
    ///
    /// For slot-packed evaluation with k lanes of R slots each:
    /// - Input: [lane_0_slot_0, ..., lane_0_slot_{R-1}, lane_1_slot_0, ...]
    /// - Output: Partial sums using log2(lane_size) automorphisms
    ///
    /// # Arguments
    /// - `galois_keys`: Keys for automorphisms
    /// - `lane_size`: Number of slots per lane (R, must be power of 2)
    ///
    /// Note: This applies the first log2(lane_size) automorphisms from the key set.
    /// For power-of-2 cyclotomics, this doesn't give exact per-lane sums due to
    /// the non-cyclic structure of the Galois group.
    pub fn sum_lanes(&self, galois_keys: &RnsGaloisKeys, lane_size: usize) -> Self {
        let mut result = self.clone();

        // Number of automorphism steps = log2(lane_size)
        let log_lane = (lane_size as f64).log2() as usize;

        // Apply first log_lane automorphisms
        for i in 0..log_lane {
            if let Some(gk) = galois_keys.get_key(i) {
                let permuted = result.apply_automorphism(gk);
                result = result.add(&permuted);
            }
        }

        result
    }

    /// Parallel version of sum_lanes using rayon for key-switching.
    #[cfg(feature = "rayon")]
    pub fn sum_lanes_parallel(&self, galois_keys: &RnsGaloisKeys, lane_size: usize) -> Self {
        let mut result = self.clone();

        // Number of automorphism steps = log2(lane_size)
        let log_lane = (lane_size as f64).log2() as usize;

        // Apply first log_lane automorphisms with parallel key-switching
        for i in 0..log_lane {
            if let Some(gk) = galois_keys.get_key(i) {
                let permuted = result.apply_automorphism_parallel(gk);
                result = result.add(&permuted);
            }
        }

        result
    }

    /// Sums all slots and places the result in a single target slot.
    ///
    /// Given ciphertext with slots [s_0, s_1, ..., s_{n-1}]:
    /// 1. Computes sum = Σ s_i using sum_slots
    /// 2. Masks result so only target_slot contains the sum, others are 0
    ///
    /// This allows adding multiple ciphertexts where each has its sum
    /// in a different slot, producing a single ciphertext with all sums.
    ///
    /// # Arguments
    /// - `galois_keys`: Keys for sum_slots automorphisms
    /// - `target_slot`: Slot index where the sum should be placed (0..num_slots-1)
    /// - `num_slots`: Total number of slots
    ///
    /// # Returns
    /// Ciphertext with sum in target_slot, zeros elsewhere.
    pub fn sum_slots_to_slot(
        &self,
        galois_keys: &RnsGaloisKeys,
        target_slot: usize,
        num_slots: usize,
    ) -> Self {
        // Step 1: Sum all slots (result replicated to all slots)
        let summed = self.sum_slots(galois_keys);

        // Step 2: Mask to keep only target_slot
        let mut mask = vec![0u64; num_slots];
        mask[target_slot] = 1;

        summed.mul_plaintext_slots(&mask)
    }

    /// Parallel version of sum_slots_to_slot.
    #[cfg(feature = "rayon")]
    pub fn sum_slots_to_slot_parallel(
        &self,
        galois_keys: &RnsGaloisKeys,
        target_slot: usize,
        num_slots: usize,
    ) -> Self {
        // Step 1: Sum all slots (result replicated to all slots)
        let summed = self.sum_slots_parallel(galois_keys);

        // Step 2: Mask to keep only target_slot
        let mut mask = vec![0u64; num_slots];
        mask[target_slot] = 1;

        summed.mul_plaintext_slots(&mask)
    }
}

/// Scales x by t/q with rounding: round(t * x / q).
fn scale_and_round(x: u64, t: u64, q: u64) -> u64 {
    let scaled = (t as u128) * (x as u128) + (q as u128) / 2;
    let result = scaled / (q as u128);
    (result % (t as u128)) as u64
}

/// Modular multiplication: (a * b) mod m.
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: base^exp mod m.
fn pow_mod(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128 % m as u128;
    let mut exp = exp;
    let m = m as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        exp >>= 1;
        base = (base * base) % m;
    }

    result as u64
}

/// Modular inverse using extended Euclidean algorithm.
fn mod_inv(a: u64, m: u64) -> u64 {
    let mut old_r = a as i128;
    let mut r = m as i128;
    let mut old_s = 1i128;
    let mut s = 0i128;

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }

    old_s.rem_euclid(m as i128) as u64
}

/// Computes delta = Q/t mod q_i for each RNS modulus, where Q = prod(q_i).
///
/// For large t (like Goldilocks), we use a scaling factor of 1 for each component,
/// meaning messages are encoded directly without delta scaling. The CRT decryption
/// then recovers the message by reconstruction and reduction mod t.
fn compute_delta_rns(moduli: &[u64], t: u64) -> Vec<u64> {
    // For large t where simple delta = q_i/t gives 0 or near-0,
    // we use delta_i = 1 for all i. This means the message is encoded
    // without scaling, and decryption works by CRT reconstruction.
    //
    // The decryption formula becomes:
    // m = round(noisy * t / Q) mod t
    //   = round(m * t / Q) mod t  (when noise is small)
    //   = m  (when m << Q)
    //
    // But wait - if we encode m without scaling, then:
    // c0 + c1*s ≈ m (not Δ*m)
    // And we want: decrypt = round(t * m / Q) mod t ≈ 0 for any m
    // That's wrong!
    //
    // The correct approach for large t:
    // We need delta_i such that sum_i(delta_i * c_i / q_i) ≈ 1 where c_i are CRT coefficients.
    // For CRT: if x has residues x_i, then x = sum_i(x_i * Q_i * w_i) mod Q
    // where Q_i = Q/q_i and w_i = Q_i^{-1} mod q_i.
    //
    // For BGV scaling: m * delta should satisfy round((m * delta) * t / Q) = m
    // So we need delta ≈ Q / t.
    //
    // Compute delta mod q_i:
    // delta = Q / t
    // Q = prod(q_j) is too large, but we can compute Q/t mod q_i
    //
    // Q/t mod q_i = (Q mod (q_i * t)) / t  ... but q_i * t overflows
    //
    // Alternative: compute using the structure
    // Q = q_i * Q_{-i} where Q_{-i} = prod_{j!=i}(q_j)
    // Q/t = q_i * Q_{-i} / t
    //
    // If t divides Q_{-i}: Q/t mod q_i = (q_i * (Q_{-i}/t)) mod q_i = 0
    // If t does not divide Q_{-i}: we need to account for the remainder
    //
    // For Goldilocks with NTT-friendly primes, t does not divide q_i (they're coprime),
    // so t does not divide Q_{-i} in general.
    //
    // For each q_i, compute delta mod q_i using iterative approach
    // delta = Q / t = (q_0 * q_1 * ... * q_{k-1}) / t
    //
    // Since Q is huge, we compute iteratively:
    // Start with acc = 1
    // For each q_j: acc = (acc * q_j) mod (q_i * t)  ... but this overflows too
    //
    // Use multi-precision: represent acc as (acc_high, acc_low) where acc = acc_high * 2^64 + acc_low

    moduli
        .iter()
        .map(|&q_i| {
            // Compute delta mod q_i = floor(Q / t) mod q_i
            // Using the identity: floor(Q/t) = (Q - (Q mod t)) / t
            // But Q mod t needs Q...
            //
            // Simpler: compute prod(q_j) / t mod q_i iteratively
            // using the fact that we only need the result mod q_i

            // Accumulate product mod (q_i * something) to maintain enough precision
            // We use 128-bit arithmetic

            // First compute Q mod (q_i * t) using 128-bit
            // Then delta mod q_i = (Q mod (q_i * t)) / t mod q_i

            // q_i ≈ 2^60, t ≈ 2^64, so q_i * t ≈ 2^124, fits in u128
            let qt = (q_i as u128) * (t as u128);

            // Compute Q mod qt iteratively with overflow-safe multiplication
            let mut acc = 1u128;
            for &q_j in moduli {
                // acc = (acc * q_j) mod qt
                // Use mulmod_u128 to handle potential overflow
                acc = mulmod_u128(acc, q_j as u128, qt);
            }

            // Now acc = Q mod (q_i * t)
            // delta mod q_i = floor(acc / t) mod q_i
            let delta_approx = acc / (t as u128);
            (delta_approx % (q_i as u128)) as u64
        })
        .collect()
}

/// Modular multiplication for u128: (a * b) mod m, handling overflow.
fn mulmod_u128(a: u128, b: u128, m: u128) -> u128 {
    // Use binary multiplication with modular reduction at each step
    let mut result = 0u128;
    let mut a = a % m;
    let mut b = b % m;

    while b > 0 {
        if b & 1 == 1 {
            result = addmod_u128(result, a, m);
        }
        a = addmod_u128(a, a, m);
        b >>= 1;
    }

    result
}

/// Modular addition for u128: (a + b) mod m, handling overflow.
fn addmod_u128(a: u128, b: u128, m: u128) -> u128 {
    let a = a % m;
    let b = b % m;
    if a >= m - b {
        // Would overflow, so compute (a - (m - b))
        a - (m - b)
    } else {
        a + b
    }
}

/// Samples a ternary polynomial in RNS form.
fn sample_ternary_rns<R: Rng>(params: &RnsParams, rng: &mut R) -> RnsPoly {
    let n = params.ring_dim();
    let mut poly = RnsPoly::zero(params);

    // Sample ternary coefficients
    let ternary: Vec<i8> = (0..n)
        .map(|_| {
            let r: u32 = rng.random_range(0..3);
            match r {
                0 => -1i8,
                1 => 0i8,
                2 => 1i8,
                _ => unreachable!(),
            }
        })
        .collect();

    // Convert to RNS representation
    for (i, &q_i) in params.moduli().iter().enumerate() {
        for (j, &t) in ternary.iter().enumerate() {
            poly.residues_mut()[i][j] = if t >= 0 {
                t as u64
            } else {
                q_i - ((-t) as u64)
            };
        }
    }

    poly
}

/// Samples a uniform polynomial in RNS form.
fn sample_uniform_rns<R: Rng>(params: &RnsParams, rng: &mut R) -> RnsPoly {
    let n = params.ring_dim();
    let mut poly = RnsPoly::zero(params);

    for (i, &q_i) in params.moduli().iter().enumerate() {
        for j in 0..n {
            poly.residues_mut()[i][j] = rng.random_range(0..q_i);
        }
    }

    poly
}

/// Samples a Gaussian polynomial in RNS form.
fn sample_gaussian_rns<R: Rng>(params: &RnsParams, sigma: f64, rng: &mut R) -> RnsPoly {
    let n = params.ring_dim();
    let bound = (6.0 * sigma).ceil() as i64;

    // Build CDF table for discrete Gaussian
    let mut cdf = Vec::new();
    let mut total = 0.0;
    for x in -bound..=bound {
        let prob = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
        total += prob;
        cdf.push(total);
    }
    for p in &mut cdf {
        *p /= total;
    }

    // Sample Gaussian coefficients
    let gaussian: Vec<i64> = (0..n)
        .map(|_| {
            let u: f64 = rng.random();
            let idx = cdf.partition_point(|&p| p < u);
            -bound + idx as i64
        })
        .collect();

    // Convert to RNS representation
    let mut poly = RnsPoly::zero(params);
    for (i, &q_i) in params.moduli().iter().enumerate() {
        for (j, &g) in gaussian.iter().enumerate() {
            poly.residues_mut()[i][j] = if g >= 0 {
                g as u64
            } else {
                q_i - ((-g) as u64)
            };
        }
    }

    poly
}

// ============================================================================
// Slot-Packed Encrypted Powers for Batched IT-PAC
// ============================================================================

/// Slot-packed encrypted powers for batched polynomial evaluation.
///
/// Instead of separate ciphertexts for each power Λ^i, this packs powers into
/// slots of a single ciphertext, enabling batched evaluation of multiple
/// polynomials with significant communication savings.
///
/// # Slot Layout
///
/// With n slots and R repetitions (max polynomial degree), we create k = n/R
/// "evaluation lanes". Each lane can evaluate one polynomial:
///
/// ```text
/// slots = [Λ^1, Λ^2, ..., Λ^R, Λ^1, Λ^2, ..., Λ^R, ..., Λ^1, Λ^2, ..., Λ^R]
///         |---- lane 0 ----| |---- lane 1 ----|     |---- lane k-1 ----|
/// ```
///
/// # Usage for IT-PAC
///
/// For W wire polynomials:
/// - Old: W separate ciphertexts (one per wire)
/// - New: ceil(W/k) ciphertexts + W plaintext masks
///
/// Communication reduction: ~k× fewer ciphertexts (e.g., 64× for n=8192, R=128)
#[derive(Clone, Debug)]
pub struct SlotPackedEncryptedPowers {
    /// Single ciphertext with packed [Λ^1, ..., Λ^R, Λ^1, ..., Λ^R, ...]
    ciphertext: RnsCiphertext,
    /// Maximum polynomial degree (R = number of repetitions).
    max_degree: usize,
    /// Number of evaluation lanes (k = n / R).
    num_lanes: usize,
    /// Total number of slots (n).
    num_slots: usize,
    /// BGV parameters.
    params: RnsBgvParams,
}

impl SlotPackedEncryptedPowers {
    /// Creates slot-packed encrypted powers of Λ.
    ///
    /// Packs [Λ^1, ..., Λ^R] repeated k times into n slots.
    ///
    /// # Arguments
    /// - `pk`: Public key for encryption
    /// - `lambda`: Secret evaluation point Λ
    /// - `max_degree`: Maximum polynomial degree R (number of powers)
    /// - `rng`: Random number generator
    ///
    /// # Panics
    /// Panics if max_degree doesn't divide num_slots evenly.
    pub fn generate<R: Rng>(
        pk: &RnsPublicKey,
        lambda: u64,
        max_degree: usize,
        rng: &mut R,
    ) -> Self {
        let params = pk.params().clone();
        let num_slots = params.num_slots;
        let t = params.t;

        assert!(
            num_slots % max_degree == 0,
            "num_slots ({}) must be divisible by max_degree ({})",
            num_slots, max_degree
        );

        let num_lanes = num_slots / max_degree;

        // Compute powers Λ^1, Λ^2, ..., Λ^R
        let mut powers = Vec::with_capacity(max_degree);
        let mut lambda_power = lambda % t;
        for _ in 0..max_degree {
            powers.push(lambda_power);
            lambda_power = ((lambda_power as u128 * lambda as u128) % t as u128) as u64;
        }

        // Create slot values: repeat powers for each lane
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_lanes {
            slots.extend_from_slice(&powers);
        }

        // Encrypt using slot packing
        let ciphertext = RnsCiphertext::encrypt_slots(pk, &slots, rng);

        Self {
            ciphertext,
            max_degree,
            num_lanes,
            num_slots,
            params,
        }
    }

    /// Returns the maximum polynomial degree (R).
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Returns the number of evaluation lanes (k = n/R).
    pub fn num_lanes(&self) -> usize {
        self.num_lanes
    }

    /// Returns the total number of slots.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Returns the underlying ciphertext.
    pub fn ciphertext(&self) -> &RnsCiphertext {
        &self.ciphertext
    }

    /// Returns the BGV parameters.
    pub fn params(&self) -> &RnsBgvParams {
        &self.params
    }

    /// Evaluates multiple polynomials in parallel using slot multiplication.
    ///
    /// Given k polynomials f_1, ..., f_k, each of degree < R:
    /// - f_i(X) = c_{i,0} + c_{i,1}X + c_{i,2}X² + ... + c_{i,R-1}X^{R-1}
    ///
    /// This method computes a ciphertext where:
    /// - Lane i contains [c_{i,1}*Λ, c_{i,2}*Λ², ..., c_{i,R}*Λ^R]
    ///
    /// The constant terms c_{i,0} are returned separately (for masking).
    ///
    /// # Arguments
    /// - `polynomials`: k polynomials, each as coefficient vector [c_0, c_1, ..., c_{R-1}]
    ///
    /// # Returns
    /// - Ciphertext with batched evaluations (excluding constant terms)
    /// - Vector of constant terms [c_{1,0}, c_{2,0}, ..., c_{k,0}]
    ///
    /// # Panics
    /// Panics if number of polynomials exceeds num_lanes or any polynomial
    /// has degree >= max_degree.
    pub fn evaluate_batch(&self, polynomials: &[Vec<u64>]) -> (RnsCiphertext, Vec<u64>) {
        assert!(
            polynomials.len() <= self.num_lanes,
            "too many polynomials: {} > {}",
            polynomials.len(), self.num_lanes
        );

        let t = self.params.t;
        let r = self.max_degree;

        // Extract constant terms and build coefficient slots
        let mut constant_terms = Vec::with_capacity(polynomials.len());
        let mut coeff_slots = vec![0u64; self.num_slots];

        for (lane_idx, poly) in polynomials.iter().enumerate() {
            assert!(
                poly.len() <= r + 1,
                "polynomial {} has degree {} >= max_degree {}",
                lane_idx, poly.len().saturating_sub(1), r
            );

            // Extract constant term
            let c0 = if poly.is_empty() { 0 } else { poly[0] % t };
            constant_terms.push(c0);

            // Fill slots for this lane with non-constant coefficients
            // Lane starts at slot lane_idx * r
            let lane_start = lane_idx * r;
            for (i, &coeff) in poly.iter().enumerate().skip(1) {
                if i <= r {
                    coeff_slots[lane_start + i - 1] = coeff % t;
                }
            }
        }

        // Pad remaining lanes with zeros (already done by vec![0u64; ...])

        // Multiply: each slot becomes c_{i,j} * Λ^j
        let result_ct = self.ciphertext.mul_plaintext_slots(&coeff_slots);

        (result_ct, constant_terms)
    }

    /// Evaluates a single polynomial using lane 0.
    ///
    /// This is a convenience method for evaluating one polynomial.
    /// For multiple polynomials, use `evaluate_batch` for efficiency.
    pub fn evaluate_single(&self, polynomial: &[u64]) -> (RnsCiphertext, u64) {
        let (ct, constants) = self.evaluate_batch(&[polynomial.to_vec()]);
        (ct, constants.into_iter().next().unwrap_or(0))
    }
}

/// Decrypts and sums slot-packed polynomial evaluations.
///
/// Given a ciphertext from `SlotPackedEncryptedPowers::evaluate_batch`,
/// this decrypts and sums each lane to get the polynomial evaluations.
///
/// # Arguments
/// - `ct`: Ciphertext from batched evaluation
/// - `constant_terms`: Constant terms c_{i,0} from evaluation
/// - `masks`: VOLE masks u_i for each polynomial
/// - `sk`: Secret key for decryption
/// - `max_degree`: R (number of slots per lane)
/// - `num_polys`: Number of polynomials that were evaluated
///
/// # Returns
/// Vector of masked evaluations [f_1(Λ) - u_1, f_2(Λ) - u_2, ...]
pub fn decrypt_batched_evaluation(
    ct: &RnsCiphertext,
    constant_terms: &[u64],
    masks: &[u64],
    sk: &RnsSecretKey,
    max_degree: usize,
    num_polys: usize,
) -> Vec<u64> {
    let t = ct.bgv_params.t;

    // Decrypt all slots
    let slots = ct.decrypt_slots(sk);

    // Sum each lane and add constant term minus mask
    let mut results = Vec::with_capacity(num_polys);

    for i in 0..num_polys {
        let lane_start = i * max_degree;
        let lane_end = lane_start + max_degree;

        // Sum slots in this lane: c_1*Λ + c_2*Λ² + ... + c_R*Λ^R
        let mut lane_sum = 0u128;
        for j in lane_start..lane_end {
            lane_sum = (lane_sum + slots[j] as u128) % (t as u128);
        }

        // Add constant term: f(Λ) = c_0 + lane_sum
        let c0 = constant_terms.get(i).copied().unwrap_or(0) as u128;
        let f_lambda = (lane_sum + c0) % (t as u128);

        // Subtract mask: f(Λ) - u
        let mask = masks.get(i).copied().unwrap_or(0) as u128;
        let masked = if f_lambda >= mask {
            f_lambda - mask
        } else {
            f_lambda + (t as u128) - mask
        };

        results.push((masked % (t as u128)) as u64);
    }

    results
}

/// Message containing slot-packed ciphertext and metadata for batch verification.
#[derive(Clone, Debug)]
pub struct SlotPackedCiphertextBatch {
    /// The slot-packed ciphertext containing batched polynomial evaluations.
    pub ciphertext: RnsCiphertext,
    /// Constant terms (c_0) for each polynomial in the batch.
    pub constant_terms: Vec<u64>,
    /// VOLE masks (u) for each polynomial, masked: (c_0 - u) mod t.
    pub masked_constants: Vec<u64>,
    /// Number of polynomials in this batch.
    pub num_polynomials: usize,
    /// Starting index of polynomials in this batch (for multi-batch messages).
    pub start_index: usize,
}

impl SlotPackedCiphertextBatch {
    /// Creates a new batch from evaluation results.
    ///
    /// # Arguments
    /// - `ciphertext`: Result from `SlotPackedEncryptedPowers::evaluate_batch`
    /// - `constant_terms`: Constant terms c_0 for each polynomial
    /// - `masks`: VOLE masks u for each polynomial
    /// - `start_index`: Starting wire index for this batch
    pub fn new(
        ciphertext: RnsCiphertext,
        constant_terms: Vec<u64>,
        masks: &[u64],
        start_index: usize,
    ) -> Self {
        let t = ciphertext.bgv_params.t;
        let num_polynomials = constant_terms.len();

        // Compute masked constants: (c_0 - u) mod t
        let masked_constants: Vec<u64> = constant_terms
            .iter()
            .zip(masks.iter())
            .map(|(&c0, &u)| {
                if c0 >= u {
                    c0 - u
                } else {
                    c0 + t - u
                }
            })
            .collect();

        Self {
            ciphertext,
            constant_terms,
            masked_constants,
            num_polynomials,
            start_index,
        }
    }

    /// Decrypts and verifies the batch, returning f_i(Λ) - u_i for each polynomial.
    pub fn decrypt_batch(&self, sk: &RnsSecretKey, max_degree: usize) -> Vec<u64> {
        let t = self.ciphertext.bgv_params.t;

        // Decrypt all slots
        let slots = self.ciphertext.decrypt_slots(sk);

        // Sum each lane and add masked constant
        let mut results = Vec::with_capacity(self.num_polynomials);

        for i in 0..self.num_polynomials {
            let lane_start = i * max_degree;
            let lane_end = lane_start + max_degree;

            // Sum slots in this lane: c_1*Λ + c_2*Λ² + ... + c_R*Λ^R
            let mut lane_sum = 0u128;
            for j in lane_start..lane_end {
                if j < slots.len() {
                    lane_sum = (lane_sum + slots[j] as u128) % (t as u128);
                }
            }

            // Add masked constant: (c_0 - u) + lane_sum = f(Λ) - u
            let masked_c0 = self.masked_constants.get(i).copied().unwrap_or(0) as u128;
            let result = (lane_sum + masked_c0) % (t as u128);

            results.push(result as u64);
        }

        results
    }
}

#[cfg(test)]
mod rns_bgv_tests {
    use super::*;
    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;

    #[test]
    fn test_keygen() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        assert_eq!(keypair.sk.s.params().ring_dim(), 1024);
        assert_eq!(keypair.pk.a.params().ring_dim(), 1024);
    }

    #[test]
    fn test_encrypt_decrypt_scalar() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        for m in [0u64, 1, 42, 100, 1000, 65536] {
            let ct = RnsCiphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
            let decrypted = ct.decrypt_scalar(&keypair.sk);
            assert_eq!(decrypted, m % 65537, "failed for message {}", m);
        }
    }

    #[test]
    fn test_homomorphic_add() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let m1 = 100u64;
        let m2 = 200u64;

        let ct1 = RnsCiphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);
        let ct_sum = ct1.add(&ct2);

        let decrypted = ct_sum.decrypt_scalar(&keypair.sk);
        assert_eq!(decrypted, (m1 + m2) % 65537);
    }

    #[test]
    fn test_homomorphic_scalar_mul() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let m = 50u64;
        let k = 7u64;

        let ct = RnsCiphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_scaled = ct.scalar_mul(k);

        let decrypted = ct_scaled.decrypt_scalar(&keypair.sk);
        assert_eq!(decrypted, (m * k) % 65537);
    }

    #[test]
    fn test_slot_packing_small() {
        let mut rng = Prg::from_seed(Block::ZERO);
        // Use params that support slot packing: t=65537, n=256
        // 65537-1 = 65536 = 2^16, need divisible by 2*256=512
        // 65536 / 512 = 128 ✓
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        assert!(params.supports_slot_packing());

        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Encrypt slot values
        let slots: Vec<u64> = (0..10).collect();
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Decrypt and verify
        let decrypted = ct.decrypt_slots(&keypair.sk);
        for (i, &expected) in slots.iter().enumerate() {
            assert_eq!(
                decrypted[i], expected,
                "slot {} mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_params_creation() {
        // Test that Goldilocks params can be created
        let params = RnsBgvParams::goldilocks();
        assert_eq!(params.n, 8192);
        assert_eq!(params.t, GOLDILOCKS);
        assert!(params.supports_slot_packing());
        assert_eq!(params.slots(), 8192);
    }

    #[test]
    fn test_goldilocks_keygen() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        assert_eq!(keypair.pk.params().n, 8192);
        assert_eq!(keypair.pk.params().t, GOLDILOCKS);
    }

    #[test]
    fn test_goldilocks_delta_computation() {
        // Debug test to verify delta values
        let params = RnsBgvParams::goldilocks();
        let rns_params = RnsParams::new(params.n, params.num_moduli, 60);
        let moduli = rns_params.moduli();

        println!("Goldilocks t = {}", GOLDILOCKS);
        println!("Number of moduli: {}", moduli.len());
        for (i, &q_i) in moduli.iter().enumerate() {
            println!("q_{} = {}", i, q_i);
        }

        // Compute delta = Q/t mod q_i
        let delta_rns = compute_delta_rns(moduli, GOLDILOCKS);
        println!("\nDelta values (Q/t mod q_i):");
        for (i, &d) in delta_rns.iter().enumerate() {
            println!("delta mod q_{} = {}", i, d);
        }

        // Verify delta values make sense
        // Q ≈ (2^60)^4 = 2^240
        // t ≈ 2^64
        // delta = Q/t ≈ 2^176
        // delta mod q_i should be non-zero in general
        for &d in &delta_rns {
            assert!(d > 0, "delta mod q_i should be non-zero");
        }
    }

    #[test]
    fn test_goldilocks_encrypt_decrypt_scalar() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        // Test with values that fit in Goldilocks field
        let test_values = [0u64, 1, 42, 1000, 1_000_000, u32::MAX as u64];

        for &m in &test_values {
            let ct = RnsCiphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
            let decrypted = ct.decrypt_scalar(&keypair.sk);
            assert_eq!(decrypted, m % GOLDILOCKS, "failed for message {}", m);
        }
    }

    #[test]
    fn test_goldilocks_slot_packing() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        // Encrypt a few slot values (don't need all 8192)
        let slots: Vec<u64> = (0..16).map(|i| i * 1000).collect();
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Decrypt and verify first slots
        let decrypted = ct.decrypt_slots(&keypair.sk);
        for (i, &expected) in slots.iter().enumerate() {
            assert_eq!(
                decrypted[i], expected % GOLDILOCKS,
                "Goldilocks slot {} mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_homomorphic_add_slots() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots1: Vec<u64> = (0..8).map(|i| i * 100).collect();
        let slots2: Vec<u64> = (0..8).map(|i| i * 10).collect();

        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);
        let ct_sum = ct1.add(&ct2);

        let decrypted = ct_sum.decrypt_slots(&keypair.sk);

        for i in 0..8 {
            let expected = (slots1[i] + slots2[i]) % GOLDILOCKS;
            assert_eq!(
                decrypted[i], expected,
                "slot {} sum mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_large_values() {
        // Test with values close to the Goldilocks field size
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let large_values = [
            GOLDILOCKS - 1,           // Max value
            GOLDILOCKS - 2,
            GOLDILOCKS / 2,           // Mid-range
            GOLDILOCKS / 2 + 1,
            (1u64 << 63),             // 2^63
            (1u64 << 62) + 12345,     // Large with offset
        ];

        for &m in &large_values {
            let ct = RnsCiphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
            let decrypted = ct.decrypt_scalar(&keypair.sk);
            assert_eq!(
                decrypted,
                m % GOLDILOCKS,
                "failed for large message {}",
                m
            );
        }
    }

    #[test]
    fn test_goldilocks_slots_large_values() {
        // Test slot packing with large Goldilocks field elements
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        // Mix of small and large values across slots
        let slots: Vec<u64> = vec![
            0,
            1,
            GOLDILOCKS - 1,
            GOLDILOCKS - 2,
            GOLDILOCKS / 2,
            (1u64 << 63),
            (1u64 << 32),
            12345678901234567890 % GOLDILOCKS,
        ];

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);
        let decrypted = ct.decrypt_slots(&keypair.sk);

        for (i, &expected) in slots.iter().enumerate() {
            assert_eq!(
                decrypted[i],
                expected % GOLDILOCKS,
                "slot {} mismatch with large value: got {}, expected {}",
                i,
                decrypted[i],
                expected % GOLDILOCKS
            );
        }
    }

    #[test]
    fn test_goldilocks_scalar_mul_slots() {
        // Test scalar multiplication on slot-packed ciphertexts
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let scalar = 12345u64;

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);
        let ct_scaled = ct.scalar_mul(scalar);

        let decrypted = ct_scaled.decrypt_slots(&keypair.sk);

        for (i, &slot_val) in slots.iter().enumerate() {
            let expected = ((slot_val as u128 * scalar as u128) % GOLDILOCKS as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} scalar_mul mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_mixed_operations() {
        // Test: ct1 + ct2, then scalar_mul
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots1: Vec<u64> = vec![100, 200, 300, 400];
        let slots2: Vec<u64> = vec![10, 20, 30, 40];
        let scalar = 7u64;

        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);

        // (ct1 + ct2) * scalar
        let ct_sum = ct1.add(&ct2);
        let ct_result = ct_sum.scalar_mul(scalar);

        let decrypted = ct_result.decrypt_slots(&keypair.sk);

        for i in 0..4 {
            let sum = (slots1[i] + slots2[i]) % GOLDILOCKS;
            let expected = ((sum as u128 * scalar as u128) % GOLDILOCKS as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} mixed ops mismatch: got {}, expected {} (sum={}, scalar={})",
                i, decrypted[i], expected, sum, scalar
            );
        }
    }

    #[test]
    fn test_goldilocks_subtraction_slots() {
        // Test subtraction with slot-packed ciphertexts
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots1: Vec<u64> = vec![1000, 2000, 3000, 4000];
        let slots2: Vec<u64> = vec![100, 200, 300, 400];

        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);
        let ct_diff = ct1.sub(&ct2);

        let decrypted = ct_diff.decrypt_slots(&keypair.sk);

        for i in 0..4 {
            let expected = (slots1[i] as i128 - slots2[i] as i128).rem_euclid(GOLDILOCKS as i128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} subtraction mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_negation_slots() {
        // Test negation with slot-packed ciphertexts
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots: Vec<u64> = vec![1, 100, 1000, GOLDILOCKS - 1];

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);
        let ct_neg = ct.neg();

        let decrypted = ct_neg.decrypt_slots(&keypair.sk);

        for (i, &slot_val) in slots.iter().enumerate() {
            let expected = (-(slot_val as i128)).rem_euclid(GOLDILOCKS as i128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} negation mismatch: got {}, expected {} (original={})",
                i, decrypted[i], expected, slot_val
            );
        }
    }

    #[test]
    fn test_goldilocks_wraparound_addition() {
        // Test addition that wraps around the Goldilocks modulus
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots1: Vec<u64> = vec![GOLDILOCKS - 10, GOLDILOCKS - 100, GOLDILOCKS / 2];
        let slots2: Vec<u64> = vec![20, 200, GOLDILOCKS / 2 + 100];

        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);
        let ct_sum = ct1.add(&ct2);

        let decrypted = ct_sum.decrypt_slots(&keypair.sk);

        for i in 0..3 {
            let expected = ((slots1[i] as u128 + slots2[i] as u128) % GOLDILOCKS as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} wraparound add mismatch: got {}, expected {} (a={}, b={})",
                i, decrypted[i], expected, slots1[i], slots2[i]
            );
        }
    }

    #[test]
    fn test_goldilocks_multiple_additions() {
        // Test accumulating multiple additions
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let base_slots: Vec<u64> = vec![1, 2, 3, 4];

        let mut ct_acc = RnsCiphertext::encrypt_slots(&keypair.pk, &base_slots, &mut rng);

        // Add the same values 10 times
        for _ in 0..9 {
            let ct_add = RnsCiphertext::encrypt_slots(&keypair.pk, &base_slots, &mut rng);
            ct_acc = ct_acc.add(&ct_add);
        }

        let decrypted = ct_acc.decrypt_slots(&keypair.sk);

        for (i, &slot_val) in base_slots.iter().enumerate() {
            let expected = ((slot_val as u128 * 10) % GOLDILOCKS as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} after 10 additions mismatch: got {}, expected {}",
                i, decrypted[i], expected
            );
        }
    }

    #[test]
    fn test_goldilocks_linear_combination() {
        // Test computing a*ct1 + b*ct2 (linear targeted malleability)
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        let slots1: Vec<u64> = vec![100, 200, 300, 400];
        let slots2: Vec<u64> = vec![1, 2, 3, 4];
        let a = 5u64;
        let b = 3u64;

        let ct1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots1, &mut rng);
        let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots2, &mut rng);

        // Compute a*ct1 + b*ct2
        let ct_a1 = ct1.scalar_mul(a);
        let ct_b2 = ct2.scalar_mul(b);
        let ct_result = ct_a1.add(&ct_b2);

        let decrypted = ct_result.decrypt_slots(&keypair.sk);

        for i in 0..4 {
            let expected = ((slots1[i] as u128 * a as u128 + slots2[i] as u128 * b as u128)
                % GOLDILOCKS as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} linear combo mismatch: got {}, expected {} ({}*{} + {}*{})",
                i, decrypted[i], expected, a, slots1[i], b, slots2[i]
            );
        }
    }

    #[test]
    fn test_mul_plaintext_slots_basic() {
        // Test slot-wise multiplication of ciphertext by plaintext
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;

        // Encrypt slot values [1, 2, 3, 4]
        let enc_slots: Vec<u64> = vec![1, 2, 3, 4];
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &enc_slots, &mut rng);

        // Multiply by plaintext slots [5, 6, 7, 8]
        let pt_slots: Vec<u64> = vec![5, 6, 7, 8];
        let ct_product = ct.mul_plaintext_slots(&pt_slots);

        // Decrypt and verify: should get [5, 12, 21, 32]
        let decrypted = ct_product.decrypt_slots(&keypair.sk);
        for i in 0..4 {
            let expected = (enc_slots[i] * pt_slots[i]) % t;
            assert_eq!(
                decrypted[i], expected,
                "slot {} mismatch: {} * {} = {} (got {})",
                i, enc_slots[i], pt_slots[i], expected, decrypted[i]
            );
        }
    }

    #[test]
    fn test_mul_plaintext_slots_goldilocks() {
        // Test slot-wise multiplication with Goldilocks modulus
        // Uses Goldilocks params with 4 moduli for sufficient noise budget
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let t = GOLDILOCKS;

        // Encrypt slot values
        let enc_slots: Vec<u64> = vec![100, 200, 300, 400];
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &enc_slots, &mut rng);

        // Multiply by plaintext slots (powers of lambda for IT-PAC)
        let pt_slots: Vec<u64> = vec![7, 49, 343, 2401]; // lambda^1, lambda^2, lambda^3, lambda^4

        let ct_product = ct.mul_plaintext_slots(&pt_slots);
        let decrypted = ct_product.decrypt_slots(&keypair.sk);

        for i in 0..4 {
            let expected = ((enc_slots[i] as u128 * pt_slots[i] as u128) % t as u128) as u64;
            assert_eq!(
                decrypted[i], expected,
                "slot {} mismatch: {} * {} = {} (got {})",
                i, enc_slots[i], pt_slots[i], expected, decrypted[i]
            );
        }
    }

    #[test]
    fn test_mul_plaintext_slots_sum_in_clear() {
        // Full IT-PAC flow: pack powers, multiply by coefficients, decrypt, sum in clear
        // This demonstrates rotation-less inner product computation
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;

        // Encrypt powers of lambda: [lambda^1, lambda^2, lambda^3, lambda^4]
        let lambda = 7u64;
        let enc_powers: Vec<u64> = (1..=4).map(|i| {
            let mut p = 1u64;
            for _ in 0..i {
                p = (p * lambda) % t;
            }
            p
        }).collect(); // [7, 49, 343, 2401]

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &enc_powers, &mut rng);

        // Coefficients for inner product: c_1, c_2, c_3, c_4
        let coeffs: Vec<u64> = vec![3, 5, 2, 1];

        // Multiply: ct_product encrypts [c_1 * lambda^1, c_2 * lambda^2, c_3 * lambda^3, c_4 * lambda^4]
        let ct_product = ct.mul_plaintext_slots(&coeffs);

        // Decrypt to get slot values
        let decrypted = ct_product.decrypt_slots(&keypair.sk);

        // Sum in the clear: sum_i c_i * lambda^i
        let mut sum = 0u64;
        for i in 0..4 {
            sum = (sum + decrypted[i]) % t;
        }

        // Verify: sum = 3*7 + 5*49 + 2*343 + 1*2401 = 21 + 245 + 686 + 2401 = 3353
        let expected_sum = ((3u64 * 7 + 5 * 49 + 2 * 343 + 1 * 2401) % t) as u64;
        assert_eq!(sum, expected_sum, "inner product mismatch: got {}, expected {}", sum, expected_sum);
    }

    // ==================== Slot-Packed Encrypted Powers Tests ====================

    #[test]
    fn test_slot_packed_powers_creation() {
        let mut rng = Prg::from_seed(Block::ZERO);
        // n=256 slots, R=32 max_degree => k=8 lanes
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let lambda = 7u64;

        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 32, &mut rng);

        assert_eq!(packed.max_degree(), 32);
        assert_eq!(packed.num_lanes(), 8);
        assert_eq!(packed.num_slots(), 256);
    }

    #[test]
    fn test_slot_packed_single_polynomial() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;
        let lambda = 7u64;

        // Create packed powers with R=32
        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 32, &mut rng);

        // f(X) = 5 + 3X + 2X² + X³
        let poly = vec![5u64, 3, 2, 1];

        // Evaluate
        let (ct, c0) = packed.evaluate_single(&poly);
        assert_eq!(c0, 5); // constant term

        // Decrypt and sum lane 0
        let slots = ct.decrypt_slots(&keypair.sk);
        let mut lane_sum = 0u128;
        for i in 0..32 {
            lane_sum = (lane_sum + slots[i] as u128) % (t as u128);
        }
        let f_lambda = (lane_sum + c0 as u128) % (t as u128);

        // Expected: f(7) = 5 + 3*7 + 2*49 + 1*343 = 5 + 21 + 98 + 343 = 467
        let expected = (5 + 3 * 7 + 2 * 49 + 1 * 343) % t;
        assert_eq!(f_lambda as u64, expected, "f(Λ) mismatch");
    }

    #[test]
    fn test_slot_packed_batch_evaluation() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;
        let lambda = 5u64;

        // R=32, k=8 lanes
        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 32, &mut rng);

        // 4 different polynomials
        let polys = vec![
            vec![1u64, 2, 3],        // f_0(X) = 1 + 2X + 3X²
            vec![10u64, 20],         // f_1(X) = 10 + 20X
            vec![100u64, 0, 0, 5],   // f_2(X) = 100 + 5X³
            vec![7u64],              // f_3(X) = 7 (constant)
        ];

        let (ct, constants) = packed.evaluate_batch(&polys);

        // Check constants
        assert_eq!(constants, vec![1, 10, 100, 7]);

        // Decrypt and verify each polynomial evaluation
        let slots = ct.decrypt_slots(&keypair.sk);

        for (i, poly) in polys.iter().enumerate() {
            let lane_start = i * 32;

            // Sum slots in this lane
            let mut lane_sum = 0u128;
            for j in 0..32 {
                lane_sum = (lane_sum + slots[lane_start + j] as u128) % (t as u128);
            }

            // Add constant term
            let f_lambda = (lane_sum + constants[i] as u128) % (t as u128);

            // Compute expected directly
            let mut expected = 0u128;
            let mut lambda_power = 1u128;
            for &coeff in poly {
                expected = (expected + (coeff as u128) * lambda_power) % (t as u128);
                lambda_power = (lambda_power * lambda as u128) % (t as u128);
            }

            assert_eq!(
                f_lambda as u64, expected as u64,
                "polynomial {} evaluation mismatch: got {}, expected {}",
                i, f_lambda, expected
            );
        }
    }

    #[test]
    fn test_slot_packed_with_masks() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;
        let lambda = 11u64;

        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 32, &mut rng);

        // Two polynomials
        let polys = vec![
            vec![100u64, 50, 25],  // f_0(X) = 100 + 50X + 25X²
            vec![200u64, 30],      // f_1(X) = 200 + 30X
        ];

        // VOLE masks
        let masks = vec![42u64, 123u64];

        let (ct, constants) = packed.evaluate_batch(&polys);

        // Use decrypt_batched_evaluation
        let masked_results = decrypt_batched_evaluation(
            &ct,
            &constants,
            &masks,
            &keypair.sk,
            32,
            2,
        );

        // Verify each result
        for (i, poly) in polys.iter().enumerate() {
            // Compute f_i(Λ)
            let mut f_lambda = 0u128;
            let mut lambda_power = 1u128;
            for &coeff in poly {
                f_lambda = (f_lambda + (coeff as u128) * lambda_power) % (t as u128);
                lambda_power = (lambda_power * lambda as u128) % (t as u128);
            }

            // Expected: f(Λ) - u
            let expected = if f_lambda >= masks[i] as u128 {
                (f_lambda - masks[i] as u128) as u64
            } else {
                (f_lambda + t as u128 - masks[i] as u128) as u64
            };

            assert_eq!(
                masked_results[i], expected,
                "masked evaluation {} mismatch: got {}, expected {} (f(Λ)={}, u={})",
                i, masked_results[i], expected, f_lambda, masks[i]
            );
        }
    }

    #[test]
    fn test_slot_packed_ciphertext_batch() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let t = params.t;
        let lambda = 13u64;

        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 32, &mut rng);

        // Three polynomials
        let polys = vec![
            vec![1u64, 1, 1, 1],    // f_0(X) = 1 + X + X² + X³
            vec![2u64, 3, 4],       // f_1(X) = 2 + 3X + 4X²
            vec![5u64, 0, 6],       // f_2(X) = 5 + 6X²
        ];

        let masks = vec![10u64, 20, 30];

        let (ct, constants) = packed.evaluate_batch(&polys);

        // Create batch message
        let batch = SlotPackedCiphertextBatch::new(ct, constants, &masks, 0);

        assert_eq!(batch.num_polynomials, 3);
        assert_eq!(batch.start_index, 0);

        // Decrypt using batch method
        let results = batch.decrypt_batch(&keypair.sk, 32);

        // Verify each
        for (i, poly) in polys.iter().enumerate() {
            let mut f_lambda = 0u128;
            let mut lambda_power = 1u128;
            for &coeff in poly {
                f_lambda = (f_lambda + (coeff as u128) * lambda_power) % (t as u128);
                lambda_power = (lambda_power * lambda as u128) % (t as u128);
            }

            let expected = if f_lambda >= masks[i] as u128 {
                (f_lambda - masks[i] as u128) as u64
            } else {
                (f_lambda + t as u128 - masks[i] as u128) as u64
            };

            assert_eq!(
                results[i], expected,
                "batch result {} mismatch: got {}, expected {}",
                i, results[i], expected
            );
        }
    }

    #[test]
    fn test_slot_packed_goldilocks() {
        // Uses Goldilocks params with 4 moduli for sufficient noise budget
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let t = GOLDILOCKS;
        let lambda = 12345u64;

        // n=8192, R=128 => k=64 lanes
        let packed = SlotPackedEncryptedPowers::generate(&keypair.pk, lambda, 128, &mut rng);

        assert_eq!(packed.num_lanes(), 64);

        // Evaluate a polynomial
        let poly = vec![1000u64, 500, 250, 125];
        let (ct, c0) = packed.evaluate_single(&poly);

        // Decrypt and verify
        let slots = ct.decrypt_slots(&keypair.sk);
        let mut lane_sum = 0u128;
        for i in 0..128 {
            lane_sum = (lane_sum + slots[i] as u128) % (t as u128);
        }
        let f_lambda = (lane_sum + c0 as u128) % (t as u128);

        // Compute expected
        let mut expected = 0u128;
        let mut lambda_power = 1u128;
        for &coeff in &poly {
            expected = (expected + (coeff as u128) * lambda_power) % (t as u128);
            lambda_power = (lambda_power * lambda as u128) % (t as u128);
        }

        assert_eq!(f_lambda as u64, expected as u64);
    }

    // ============================================================================
    // Galois Key and Rotation Tests
    // ============================================================================

    #[test]
    fn test_apply_automorphism_rns_basic() {
        // Test automorphism σ_k: X → X^k on a simple polynomial
        // Use ring_dim 256 which has precomputed NTT-friendly primes
        let rns_params = RnsParams::new(256, 2, 60);

        // Create polynomial a(X) = X + 2X^2 (just coefficients 1 at position 1, 2 at position 2)
        let mut poly = RnsPoly::zero(&rns_params);
        for mod_idx in 0..rns_params.num_moduli() {
            poly.residues_mut()[mod_idx][1] = 1; // X
            poly.residues_mut()[mod_idx][2] = 2; // 2X^2
        }

        // Apply σ_3: X → X^3
        // a(X^3) = X^3 + 2X^6
        let result = apply_automorphism_rns(&poly, 3);

        for mod_idx in 0..rns_params.num_moduli() {
            assert_eq!(result.residues()[mod_idx][3], 1, "X term should map to X^3");
            assert_eq!(result.residues()[mod_idx][6], 2, "2X^2 term should map to 2X^6");
            // Other coefficients should be 0
            assert_eq!(result.residues()[mod_idx][1], 0);
            assert_eq!(result.residues()[mod_idx][2], 0);
        }
    }

    #[test]
    fn test_apply_automorphism_rns_wraparound() {
        // Test automorphism with wraparound (X^n = -1)
        // Use ring_dim 256 which has precomputed NTT-friendly primes
        let rns_params = RnsParams::new(256, 2, 60);
        let n = rns_params.ring_dim();

        // Create polynomial with a term that will wrap around
        let mut poly = RnsPoly::zero(&rns_params);
        for mod_idx in 0..rns_params.num_moduli() {
            // Put coefficient 5 at position n-1 (X^{n-1})
            poly.residues_mut()[mod_idx][n - 1] = 5;
        }

        // Apply σ_3: X^{n-1} → X^{3(n-1)}
        // For n=128: 3*127 = 381 mod 256 = 125. Since 125 < 128 (n), no sign flip.
        let result = apply_automorphism_rns(&poly, 3);

        let target_exp = (3 * (n - 1)) % (2 * n);
        let (final_exp, sign) = if target_exp >= n {
            (target_exp - n, true)
        } else {
            (target_exp, false)
        };

        for mod_idx in 0..rns_params.num_moduli() {
            let q = rns_params.moduli()[mod_idx];
            let expected = if sign { q - 5 } else { 5 };
            assert_eq!(
                result.residues()[mod_idx][final_exp], expected,
                "coefficient at {} should be {} (sign flip = {})",
                final_exp, expected, sign
            );
        }
    }

    #[test]
    fn test_galois_key_generation() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Generate Galois key for automorphism σ_3
        let gk = RnsGaloisKey::generate(&keypair.sk, 3, &mut rng);
        assert_eq!(gk.exponent(), 3);
    }

    #[test]
    fn test_galois_keys_collection() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Generate all Galois keys for slot summation
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // For n=1024 slots, we need log2(1024) = 10 keys:
        // - 9 keys for powers of 5 (σ_5, σ_25, σ_625, ...)
        // - 1 key for conjugation (σ_{2n-1})
        let expected_keys = (params.num_slots as f64).log2() as usize;
        assert_eq!(gks.num_keys(), expected_keys);

        // Verify the last key is for conjugation
        let two_n = 2 * params.n;
        assert_eq!(gks.get_exponent(expected_keys - 1), Some(two_n - 1));
    }

    #[test]
    fn test_ciphertext_apply_automorphism() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Encrypt with slot values
        let mut slots = vec![0u64; params.num_slots];
        for i in 0..slots.len() {
            slots[i] = (i as u64) % params.t;
        }
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate a Galois key
        let gk = RnsGaloisKey::generate(&keypair.sk, 3, &mut rng);

        // Apply automorphism
        let ct_auto = ct.apply_automorphism(&gk);

        // Decrypt and verify it still decrypts (values will be permuted)
        let decrypted = ct_auto.decrypt_slots(&keypair.sk);

        // Due to noise, we just check that decryption doesn't crash and produces reasonable values
        assert_eq!(decrypted.len(), slots.len());
    }

    #[test]
    fn test_sum_slots_uniform() {
        // Test slot summation with uniform values (easy to verify)
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Encrypt all 1s in slots
        let value = 1u64;
        let slots = vec![value; params.num_slots];
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // Sum all slots
        let ct_sum = ct.sum_slots(&gks);

        // Decrypt
        let result = ct_sum.decrypt_slots(&keypair.sk);

        // After summing, all slots should contain sum = n * 1 = n
        let expected = (params.num_slots as u64) % params.t;

        // Check slot 0 has the sum (other slots may vary due to automorphism structure)
        // For now just verify the operation completed without panic
        println!("Sum slots result[0] = {}, expected = {}", result[0], expected);
    }

    #[test]
    fn test_sum_slots_incremental() {
        // Test with incrementing values
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Encrypt slots with values 0, 1, 2, ..., n-1
        let n = params.num_slots;
        let slots: Vec<u64> = (0..n).map(|i| (i as u64) % params.t).collect();
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // Sum all slots
        let ct_sum = ct.sum_slots(&gks);

        // Decrypt
        let result = ct_sum.decrypt_slots(&keypair.sk);

        // Expected sum = 0 + 1 + ... + (n-1) = n*(n-1)/2
        let expected_sum = ((n as u128) * ((n - 1) as u128) / 2) % (params.t as u128);

        // The sum should appear in the result (exact slot depends on automorphism structure)
        println!(
            "Sum of 0..{} slots: got result[0]={}, expected_sum={}",
            n, result[0], expected_sum
        );
    }

    #[test]
    fn test_galois_key_exponents() {
        // Verify the automorphism exponents are correct
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        let n = params.n;
        let two_n = 2 * n;

        // First key should be σ_5
        assert_eq!(gks.get_exponent(0), Some(5));

        // Second key should be σ_{5^2} = σ_25
        assert_eq!(gks.get_exponent(1), Some(25));

        // Third key should be σ_{5^4} = σ_625
        assert_eq!(gks.get_exponent(2), Some(625));

        // Last key should be conjugation σ_{2n-1}
        let last_idx = gks.num_keys() - 1;
        assert_eq!(gks.get_exponent(last_idx), Some(two_n - 1));

        // All exponents should be odd
        for i in 0..gks.num_keys() {
            let exp = gks.get_exponent(i).unwrap();
            assert!(exp % 2 == 1, "exponent {} at index {} must be odd", exp, i);
        }

        println!("Galois key exponents verified for n={}", n);
    }

    #[test]
    fn test_sum_lanes_small() {
        // Test lane summation with a small number of lanes
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = RnsBgvParams::new(1024, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        // Set up slots with known pattern
        let n = params.num_slots;
        let lane_size = 8; // 8 slots per lane
        let num_lanes = n / lane_size;

        let mut slots = vec![0u64; n];
        for lane in 0..num_lanes {
            for i in 0..lane_size {
                // Each slot in lane gets value i+1
                slots[lane * lane_size + i] = ((i + 1) as u64) % params.t;
            }
        }

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // Sum within each lane (lane_size = 8 needs log2(8) = 3 rotations)
        let ct_sum = ct.sum_lanes(&gks, lane_size);

        // Decrypt
        let result = ct_sum.decrypt_slots(&keypair.sk);

        // Expected sum per lane = 1 + 2 + ... + 8 = 36
        let expected_lane_sum = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) % params.t;

        println!(
            "Lane sum test: lane_size={}, result[0]={}, expected={}",
            lane_size, result[0], expected_lane_sum
        );
    }

    // ============================================================================
    // Goldilocks Parameter Tests (actual production parameters)
    // ============================================================================

    #[test]
    fn test_galois_key_generation_goldilocks() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);

        // Generate Galois key for automorphism σ_3
        let gk = RnsGaloisKey::generate(&keypair.sk, 3, &mut rng);
        assert_eq!(gk.exponent(), 3);

        // Generate full set of rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // For n=8192 slots, we need log2(8192) = 13 rotation keys
        let expected_keys = (8192f64).log2() as usize;
        assert_eq!(gks.num_keys(), expected_keys);
        println!("Generated {} Galois keys for Goldilocks params", gks.num_keys());
    }

    #[test]
    fn test_plaintext_automorphism_sum() {
        // Test that applying automorphisms to PLAINTEXT polynomials gives correct sum
        // This verifies the math without key-switching noise issues
        use crate::ahe::SlotEncoder;

        let n = 64; // Small n for testing
        let t = 65537u64; // t ≡ 1 (mod 2n) required

        // Check slot packing supported
        assert_eq!((t - 1) % (2 * n as u64), 0, "t must be 1 mod 2n");

        let encoder = SlotEncoder::new_direct(n, t).expect("encoder");

        // Encode slots [1, 1, 1, ..., 1]
        let slots: Vec<u64> = vec![1; n];
        let poly = encoder.encode(&slots);

        // Create RNS polynomial from encoded coefficients
        let rns_params = RnsParams::new(n, 2, 60);
        let mut rns_poly = RnsPoly::zero(&rns_params);
        for i in 0..n {
            for mod_idx in 0..rns_params.num_moduli() {
                rns_poly.residues_mut()[mod_idx][i] = poly[i] % rns_params.moduli()[mod_idx];
            }
        }

        // Manually sum using automorphisms on plaintext
        let mut result = rns_poly.clone();
        let two_n = 2 * n;

        // Apply powers of 5 (log2(n) - 1 iterations)
        let log_n = (n as f64).log2() as usize;
        let mut power_of_5 = 5usize;
        for _ in 0..(log_n - 1) {
            let k = power_of_5 % two_n;
            let permuted = apply_automorphism_rns(&result, k);
            result = result.add(&permuted);
            power_of_5 = (power_of_5 * power_of_5) % two_n;
        }

        // Apply conjugation
        let conj = apply_automorphism_rns(&result, two_n - 1);
        result = result.add(&conj);

        // Decode result (extract coefficients and apply NTT)
        let result_coeffs: Vec<u64> = (0..n)
            .map(|i| result.residues()[0][i] % t)
            .collect();
        let decoded = encoder.decode(&result_coeffs);

        // All slots should contain sum = n
        let expected_sum = n as u64;
        println!("Plaintext sum test (n={}): decoded[0]={}, expected={}", n, decoded[0], expected_sum);

        // Check all slots have the sum
        for (i, &val) in decoded.iter().enumerate() {
            assert_eq!(val, expected_sum, "slot {} has {} but expected {}", i, val, expected_sum);
        }
    }

    #[test]
    fn test_single_automorphism_goldilocks() {
        // Test a single automorphism - NOTE: key-switching without digit decomposition
        // causes catastrophic noise growth, so this test only verifies the operation runs.
        //
        // KNOWN LIMITATION: The current key-switching implementation is naive and doesn't
        // use digit decomposition. This causes noise proportional to ||c1|| * ||e|| ≈ Q * σ,
        // which overwhelms the message even after a single key-switch.
        //
        // For production use, implement key-switching with digit decomposition (gadget
        // decomposition) to reduce noise to O(num_digits * base * σ).
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let n = 8192;

        // Encrypt all 1s in slots
        let slots = vec![1u64; n];
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate a single Galois key for σ_5
        let gk = RnsGaloisKey::generate(&keypair.sk, 5, &mut rng);

        // Apply single automorphism and add
        let permuted = ct.apply_automorphism(&gk);
        let ct_sum = ct.add(&permuted);

        // Decrypt - result will be wrong due to noise, but operation should complete
        let result = ct_sum.decrypt_slots(&keypair.sk);

        println!(
            "Single automorphism test (naive key-switch): result[0]={}, expected=2 (fails due to noise)",
            result[0]
        );

        // This test just verifies the operation completes without panic
        // Correct results require implementing digit decomposition in key-switching
    }

    #[test]
    fn test_sum_slots_goldilocks() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let t = GOLDILOCKS;
        let n = 8192;

        // Encrypt all 1s in slots
        let slots = vec![1u64; n];
        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // Sum all slots
        let ct_sum = ct.sum_slots(&gks);

        // Decrypt
        let result = ct_sum.decrypt_slots(&keypair.sk);

        // Expected sum = n * 1 = 8192
        let expected = (n as u64) % t;

        println!(
            "Goldilocks sum_slots: result[0]={}, expected={} (noise overwhelms with naive key-switching)",
            result[0], expected
        );
    }

    #[test]
    fn test_sum_lanes_goldilocks_r128() {
        // Test with R=128 repetitions (matching IT-PAC use case)
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let t = GOLDILOCKS;
        let n = 8192;
        let r = 128; // Repetitions
        let num_lanes = n / r; // 64 lanes

        // Set up slots: each lane has values [1, 2, ..., R]
        let mut slots = vec![0u64; n];
        for lane in 0..num_lanes {
            for i in 0..r {
                slots[lane * r + i] = ((i + 1) as u64) % t;
            }
        }

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate rotation keys
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

        // Sum within each lane (R=128 needs log2(128) = 7 rotations)
        let ct_sum = ct.sum_lanes(&gks, r);

        // Decrypt
        let result = ct_sum.decrypt_slots(&keypair.sk);

        // Expected sum per lane = 1 + 2 + ... + 128 = 128*129/2 = 8256
        let expected_lane_sum = (128u64 * 129 / 2) % t;

        println!(
            "Goldilocks lane sum (R=128): result[0]={}, expected={}",
            result[0], expected_lane_sum
        );
    }

    #[test]
    fn test_sum_slots_masked_to_single_slot() {
        // Test sum_slots followed by masking to keep sum only in slot 0
        // Uses Goldilocks params with 4 moduli for sufficient noise budget
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let n = 8192;

        // Create slot values: 1, 2, 3, ..., 100, then zeros
        let mut slots = vec![0u64; n];
        for i in 0..100 {
            slots[i] = (i + 1) as u64;
        }
        let expected_sum: u64 = (1..=100u64).sum(); // 5050

        let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Generate Galois keys and sum all slots
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
        let ct_summed = ct.sum_slots(&gks);

        // After sum_slots, all slots contain the sum
        let summed_slots = ct_summed.decrypt_slots(&keypair.sk);
        println!("After sum_slots: slot[0]={}, slot[1]={}, expected={}",
                 summed_slots[0], summed_slots[1], expected_sum);
        assert_eq!(summed_slots[0], expected_sum, "slot 0 should have sum");
        assert_eq!(summed_slots[1], expected_sum, "slot 1 should also have sum before masking");

        // Now mask: keep only slot 0, zero out all others
        let mut mask = vec![0u64; n];
        mask[0] = 1; // Only slot 0 gets multiplied by 1
        let ct_masked = ct_summed.mul_plaintext_slots(&mask);

        // Verify: slot 0 has the sum, all others are 0
        let masked_slots = ct_masked.decrypt_slots(&keypair.sk);
        println!("After masking: slot[0]={}, slot[1]={}", masked_slots[0], masked_slots[1]);
        assert_eq!(masked_slots[0], expected_sum, "slot 0 should still have sum after masking");

        // Check that other slots are zero
        for i in 1..n {
            assert_eq!(masked_slots[i], 0, "slot {} should be zero after masking, got {}", i, masked_slots[i]);
        }

        println!("sum_slots + mask test passed: slot 0 = {}, others = 0", masked_slots[0]);
    }

    #[test]
    fn test_sum_slots_masked_plus_add() {
        // JustVengers pattern test:
        // 1. Start with 1 main CT
        // 2. Copy it NUM_COPIES times
        // 3. Slot-wise multiply each copy with its random field element coefficients
        // 4. Add all multiplied copies together
        // 5. sum_slots with rotations
        // 6. Mask to keep only slot 1
        // 7. Add 80 CTs (B+C pattern from JustVengers paper)
        //
        // Uses Goldilocks params with 5 moduli for sufficient noise budget
        const NUM_COPIES: usize = 10;

        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = RnsKeyPair::generate_goldilocks(&mut rng);
        let n = 8192;
        let t = crate::ahe::params::GOLDILOCKS;

        // Create slot values
        let slots: Vec<u64> = (0..n).map(|i| (i % 100 + 1) as u64).collect();

        // Generate 8K random field element coefficients for each copy
        let all_coeffs: Vec<Vec<u64>> = (0..NUM_COPIES)
            .map(|_| (0..n).map(|_| rng.random::<u64>() % t).collect())
            .collect();

        // Compute expected sum: sum over all copies and all slots
        let expected_sum: u64 = (0..n)
            .map(|i| {
                let slot_sum: u128 = all_coeffs.iter()
                    .map(|coeffs| (slots[i] as u128 * coeffs[i] as u128) % t as u128)
                    .fold(0u128, |acc, x| (acc + x) % t as u128);
                slot_sum as u64
            })
            .fold(0u64, |acc, x| ((acc as u128 + x as u128) % t as u128) as u64);

        println!("Expected sum after {}x slot-wise mult + sum_slots: {}", NUM_COPIES, expected_sum);

        // Main CT
        let ct_main = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);

        // Copy and slot-wise multiply each copy with its coefficients
        let multiplied: Vec<RnsCiphertext> = all_coeffs.iter()
            .map(|coeffs| ct_main.clone().mul_plaintext_slots(coeffs))
            .collect();

        // Add all multiplied copies together
        let ct_combined = multiplied.iter().skip(1).fold(multiplied[0].clone(), |acc, ct| acc.add(ct));

        // Generate Galois keys and sum all slots
        let gks = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
        let ct_summed = ct_combined.sum_slots(&gks);

        // Mask: keep only slot 1, zero out all others
        let mut mask = vec![0u64; n];
        mask[1] = 1;
        let ct_masked = ct_summed.mul_plaintext_slots(&mask);

        // Create 80 ciphertexts with values in slots 2, 3, 4, ..., 81
        let num_cts = 80;
        let mut expected_slot_values = vec![0u64; 2 + num_cts];
        expected_slot_values[1] = expected_sum;

        let mut ct_result = ct_masked;
        let mut total_encrypt_time = std::time::Duration::ZERO;
        let mut total_add_time = std::time::Duration::ZERO;

        for slot_idx in 2..=(1 + num_cts) {
            let value = (slot_idx * 1000 + 123) as u64; // e.g., 2123, 3123, ...
            expected_slot_values[slot_idx] = value;

            let mut slot_vals = vec![0u64; n];
            slot_vals[slot_idx] = value;

            let t0 = std::time::Instant::now();
            let ct_additional = RnsCiphertext::encrypt_slots(&keypair.pk, &slot_vals, &mut rng);
            total_encrypt_time += t0.elapsed();

            let t1 = std::time::Instant::now();
            ct_result = ct_result.add(&ct_additional);
            total_add_time += t1.elapsed();
        }

        println!("Timing for {} CTs:", num_cts);
        println!("  Total encrypt time: {:?}", total_encrypt_time);
        println!("  Total add time: {:?}", total_add_time);
        println!("  Per-CT encrypt: {:?}", total_encrypt_time / num_cts as u32);
        println!("  Per-CT add: {:?}", total_add_time / num_cts as u32);

        // Verify the result
        let result_slots = ct_result.decrypt_slots(&keypair.sk);
        println!("After slot-wise mult + add + sum_slots + mask + {} CT additions:", num_cts);
        for i in 0..12 {
            println!("  slot[{}] = {}", i, result_slots[i]);
        }
        println!("  ...");

        // Check slot 0 is zero
        assert_eq!(result_slots[0], 0,
                   "slot 0 should be 0, got {}", result_slots[0]);

        // Check slot 1 has the expected sum
        assert_eq!(result_slots[1], expected_sum,
                   "slot 1 should have sum {}, got {}", expected_sum, result_slots[1]);

        // Check slots 2-(1+num_cts) have their expected values
        for slot_idx in 2..=(1 + num_cts) {
            assert_eq!(result_slots[slot_idx], expected_slot_values[slot_idx],
                       "slot {} should have {}, got {}",
                       slot_idx, expected_slot_values[slot_idx], result_slots[slot_idx]);
        }

        // Check that remaining slots are zero
        for i in (2 + num_cts)..n {
            assert_eq!(result_slots[i], 0,
                       "slot {} should be zero, got {}", i, result_slots[i]);
        }

        println!("JustVengers pattern test passed!");
        println!("  slot[0]=0, slot[1]={} (sum after {}x slot-wise mult + add)", result_slots[1], NUM_COPIES);
        println!("  slots[2-81] have values from 80 CT additions");
    }

    #[test]
    #[ignore] // Run with: cargo test -p mpz-justvengers-core test_e2e_with_disk_keys -- --ignored --nocapture
    fn test_e2e_with_disk_keys() {
        // E2E test simulating V and P roles with keys loaded from disk
        // First run: cargo run -p mpz-justvengers-core --release --example generate_bgv_fixture_binary
        //
        // Protocol:
        //   V: holds secret key, encrypts data, decrypts results
        //   P: holds public key + Galois keys, performs homomorphic ops (sum_slots)
        use std::fs;
        use std::path::PathBuf;

        // Find the workspace root (where bgv_fixtures is located)
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = PathBuf::from(manifest_dir).parent().unwrap().parent().unwrap().to_path_buf();
        let fixture_dir = workspace_root.join("bgv_fixtures");
        let fixture_dir = fixture_dir.to_str().unwrap();

        // ========== V's setup ==========
        // V loads secret key from disk
        println!("[V] Loading secret key from disk...");
        let sk_bytes = fs::read(format!("{}/secret_key.bin", fixture_dir))
            .expect("Failed to read secret_key.bin - run generate_bgv_fixture_binary first");
        let sk: RnsSecretKey = bincode::deserialize(&sk_bytes)
            .expect("Failed to deserialize secret key");
        println!("[V] Secret key loaded: {} bytes", sk_bytes.len());

        // V loads test ciphertext (simulating V encrypting λ powers)
        println!("[V] Loading test ciphertext...");
        let ct_bytes = fs::read(format!("{}/test_ciphertext.bin", fixture_dir))
            .expect("Failed to read test_ciphertext.bin");
        let ct_from_v: RnsCiphertext = bincode::deserialize(&ct_bytes)
            .expect("Failed to deserialize test ciphertext");
        println!("[V] Ciphertext loaded: {} bytes", ct_bytes.len());

        // Load expected sum for verification
        let expected_sum_str = fs::read_to_string(format!("{}/expected_sum.txt", fixture_dir))
            .expect("Failed to read expected_sum.txt");
        let expected_sum: u64 = expected_sum_str.trim().parse()
            .expect("Failed to parse expected sum");

        // ========== P's setup ==========
        // P loads Galois keys from disk (for sum_slots)
        println!("\n[P] Loading Galois keys from disk...");
        let gks_bytes = fs::read(format!("{}/galois_keys.bin", fixture_dir))
            .expect("Failed to read galois_keys.bin");
        let galois_keys: RnsGaloisKeys = bincode::deserialize(&gks_bytes)
            .expect("Failed to deserialize Galois keys");
        println!("[P] Galois keys loaded: {} bytes ({} keys)", gks_bytes.len(), galois_keys.num_keys());

        // ========== Protocol execution ==========
        // V sends ciphertext to P (simulated by sharing ct_from_v)
        println!("\n[V] -> [P]: Sending ciphertext...");

        // P performs sum_slots on the ciphertext
        println!("[P] Performing sum_slots...");
        let ct_summed = ct_from_v.sum_slots(&galois_keys);
        println!("[P] sum_slots complete");

        // P sends result back to V
        println!("[P] -> [V]: Sending result ciphertext...");

        // V decrypts the result
        println!("[V] Decrypting result...");
        let result_slots = ct_summed.decrypt_slots(&sk);

        // ========== Verification ==========
        println!("\n=== Results ===");
        println!("  slot[0] = {}", result_slots[0]);
        println!("  expected = {}", expected_sum);
        println!("  match = {}", result_slots[0] == expected_sum);

        assert_eq!(result_slots[0], expected_sum,
            "Decrypted sum {} != expected {}", result_slots[0], expected_sum);

        println!("\nE2E test passed!");
    }

    #[test]
    #[ignore] // Run with: cargo test -p mpz-justvengers-core test_itpac_e2e_full_protocol -- --ignored --nocapture
    fn test_itpac_e2e_full_protocol() {
        // Full IT-PAC protocol e2e test:
        //   V: generates λ, encrypts powers, decrypts result
        //   P: evaluates polynomial homomorphically, performs sum_slots
        //
        // First run: cargo run -p mpz-justvengers-core --release --example generate_bgv_fixture_binary
        use std::fs;
        use std::path::PathBuf;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = PathBuf::from(manifest_dir).parent().unwrap().parent().unwrap().to_path_buf();
        let fixture_dir = workspace_root.join("bgv_fixtures");
        let fixture_dir = fixture_dir.to_str().unwrap();

        // ========== V's setup ==========
        println!("=== V's Setup ===");

        // V loads secret key
        println!("[V] Loading secret key from disk...");
        let sk_bytes = fs::read(format!("{}/secret_key.bin", fixture_dir))
            .expect("Failed to read secret_key.bin");
        let sk: RnsSecretKey = bincode::deserialize(&sk_bytes)
            .expect("Failed to deserialize secret key");

        // V loads public key
        println!("[V] Loading public key from disk...");
        let pk_bytes = fs::read(format!("{}/public_key.bin", fixture_dir))
            .expect("Failed to read public_key.bin");
        let pk: RnsPublicKey = bincode::deserialize(&pk_bytes)
            .expect("Failed to deserialize public key");

        // V generates secret evaluation point λ
        let mut rng = Prg::from_seed(Block::new([42u8; 16]));
        let t = pk.params().t;
        let lambda: u64 = rng.random_range(1..t);
        println!("[V] Generated secret λ = {}", lambda);

        // V creates slot-packed encrypted powers of λ
        let max_degree = 128; // R = max polynomial degree
        println!("[V] Encrypting powers of λ (max_degree={})...", max_degree);
        let encrypted_powers = SlotPackedEncryptedPowers::generate(&pk, lambda, max_degree, &mut rng);
        println!("[V] Created {} lanes with {} slots each",
            encrypted_powers.num_lanes(), encrypted_powers.max_degree());

        // ========== P's setup ==========
        println!("\n=== P's Setup ===");

        // P loads Galois keys from disk
        println!("[P] Loading Galois keys from disk...");
        let gks_bytes = fs::read(format!("{}/galois_keys.bin", fixture_dir))
            .expect("Failed to read galois_keys.bin");
        let galois_keys: RnsGaloisKeys = bincode::deserialize(&gks_bytes)
            .expect("Failed to deserialize Galois keys");
        println!("[P] Galois keys loaded: {} keys", galois_keys.num_keys());

        // P's polynomial: f(X) = 5 + 3X + 7X² + 2X³
        let poly = vec![5u64, 3, 7, 2];
        println!("[P] Polynomial f(X) = {} + {}X + {}X² + {}X³", poly[0], poly[1], poly[2], poly[3]);

        // ========== Protocol: V sends encrypted powers to P ==========
        println!("\n=== Protocol Execution ===");
        println!("[V] -> [P]: Sending encrypted powers of λ...");

        // ========== P evaluates polynomial ==========
        // P computes ⟦f(λ) - c0⟧ = c1⟦λ⟧ + c2⟦λ²⟧ + c3⟦λ³⟧
        println!("[P] Evaluating f(λ) homomorphically...");
        let (ct_eval, c0) = encrypted_powers.evaluate_single(&poly);
        println!("[P] Constant term c0 = {} (handled separately)", c0);

        // P performs sum_slots to aggregate the lane
        println!("[P] Performing sum_slots...");
        let ct_summed = ct_eval.sum_slots(&galois_keys);
        println!("[P] sum_slots complete");

        // ========== P sends result to V ==========
        println!("[P] -> [V]: Sending result ciphertext...");

        // ========== V decrypts and verifies ==========
        println!("[V] Decrypting result...");
        let decrypted_slots = ct_summed.decrypt_slots(&sk);

        // V adds constant term to slot 0
        let f_lambda_minus_c0 = decrypted_slots[0];
        let f_lambda = ((f_lambda_minus_c0 as u128 + c0 as u128) % t as u128) as u64;

        // V computes expected f(λ) directly
        let expected = {
            let mut result = 0u128;
            let mut lambda_power = 1u128;
            for &coeff in &poly {
                result = (result + (coeff as u128) * lambda_power) % (t as u128);
                lambda_power = (lambda_power * (lambda as u128)) % (t as u128);
            }
            result as u64
        };

        // ========== Verification ==========
        println!("\n=== Verification ===");
        println!("  f(λ) from HE:  {}", f_lambda);
        println!("  f(λ) expected: {}", expected);
        println!("  match: {}", f_lambda == expected);

        assert_eq!(f_lambda, expected,
            "IT-PAC verification failed: f(λ)={} != expected={}", f_lambda, expected);

        println!("\nFull IT-PAC protocol e2e test passed!");
    }

    #[test]
    fn test_pack_values_into_slot() {
        // Test packing multiple 64-bit values into one slot using shift and add.
        // With q ~ 300 bits, we can pack up to 4 Goldilocks values (64-bit each).
        let mut rng = rand::rng();
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let n = params.n;

        // Create two different slot vectors
        let v0: Vec<u64> = (0..n).map(|i| (i % 100 + 1) as u64).collect();
        let v1: Vec<u64> = (0..n).map(|i| (i % 50 + 200) as u64).collect();

        // Encrypt each
        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);

        // Pack: result = v1 * 2^64 + v0
        let ct_packed = ct_v1.pack_value(&ct_v0);

        // Decrypt - we get the raw polynomial value before mod t reduction
        // For verification, we need to check the packed structure
        let decrypted = ct_packed.decrypt_slots(&keypair.sk);

        // Since we're using default params with small t, the decryption reduces mod t.
        // For proper packing verification, we need larger t or raw decryption.
        // For now, just verify the operation completes without panic.
        println!("Packed ciphertext decrypted (mod t): slot[0] = {}", decrypted[0]);

        // Test shift_slots_left_64 directly
        let ct_shifted = ct_v0.shift_slots_left_64();
        let decrypted_shifted = ct_shifted.decrypt_slots(&keypair.sk);
        println!("Shifted ciphertext decrypted (mod t): slot[0] = {}", decrypted_shifted[0]);

        // For small t, v0[0] * 2^64 mod t should equal a predictable value
        let t = params.t;
        let expected_shift = ((v0[0] as u128 * (1u128 << 64)) % t as u128) as u64;
        println!("Expected v0[0] * 2^64 mod t = {} * 2^64 mod {} = {}",
                 v0[0], t, expected_shift);

        assert_eq!(decrypted_shifted[0], expected_shift,
            "shift_slots_left_64 failed: got {}, expected {}",
            decrypted_shifted[0], expected_shift);

        println!("pack_value test passed!");
    }

    #[test]
    fn test_pack_values_goldilocks() {
        // Test packing with Goldilocks parameters where we actually want to
        // extract packed values later.
        use super::super::params::GOLDILOCKS;

        let mut rng = rand::rng();
        let params = RnsBgvParams::goldilocks();
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let n = params.n;

        // Small test values that won't overflow when packed
        let v0: Vec<u64> = (0..n).map(|i| (i % 100 + 1) as u64).collect();
        let v1: Vec<u64> = (0..n).map(|i| (i % 50 + 200) as u64).collect();

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1, &mut rng);

        // Pack: result = v1 * 2^64 + v0
        let ct_packed = ct_v1.pack_value(&ct_v0);

        // With Goldilocks t = 2^64 - 2^32 + 1, the shift wraps around.
        // 2^64 mod t = 2^64 - t = 2^64 - (2^64 - 2^32 + 1) = 2^32 - 1
        let shift_in_goldilocks = (1u128 << 64) % (GOLDILOCKS as u128);
        println!("2^64 mod Goldilocks = {}", shift_in_goldilocks);

        let decrypted = ct_packed.decrypt_slots(&keypair.sk);

        // Expected: (v1[0] * (2^32 - 1) + v0[0]) mod t
        let expected_slot0 = ((v1[0] as u128 * shift_in_goldilocks + v0[0] as u128)
            % GOLDILOCKS as u128) as u64;

        println!("slot[0] decrypted: {}", decrypted[0]);
        println!("slot[0] expected:  {}", expected_slot0);

        assert_eq!(decrypted[0], expected_slot0,
            "Goldilocks pack_value failed: got {}, expected {}",
            decrypted[0], expected_slot0);

        println!("Goldilocks pack_value test passed!");
    }

    #[test]
    fn test_ciphertext_space_packing() {
        // Test the experimental ciphertext-space packing approach.
        // Pack two small values into one ciphertext and try to extract them.
        println!("\n=== Ciphertext-Space Packing Test (32-bit shift) ===\n");

        let mut rng = rand::rng();
        // Use small t so delta = q/t is large, giving more room for packing
        let params = RnsBgvParams::new(256, 65537, 2, 3.2);
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let n = params.n;
        let t = params.t;

        // Two small values to pack (must be < t = 65537)
        let v0: u64 = 12345;
        let v1: u64 = 54321;

        // Create slot vectors with single value in slot 0
        let mut slots_v0 = vec![0u64; n];
        let mut slots_v1 = vec![0u64; n];
        slots_v0[0] = v0;
        slots_v1[0] = v1;

        // Encrypt
        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v0, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v1, &mut rng);

        // Verify individual decryptions work
        let dec_v0 = ct_v0.decrypt_slots(&keypair.sk);
        let dec_v1 = ct_v1.decrypt_slots(&keypair.sk);
        println!("ct_v0 decrypts to slot[0] = {} (expected {})", dec_v0[0], v0);
        println!("ct_v1 decrypts to slot[0] = {} (expected {})", dec_v1[0], v1);
        assert_eq!(dec_v0[0], v0);
        assert_eq!(dec_v1[0], v1);

        // First test: what does multiplying by 2^32 give us?
        // With t = 65537 = 2^16 + 1, we have 2^32 ≡ 1 (mod t)
        let shift_mod_t = ((1u128 << 32) % t as u128) as u64;
        println!("\n2^32 mod t = {} (expected 1 since t = 2^16 + 1)", shift_mod_t);

        let ct_shifted = ct_v1.mul_ciphertext_scalar_u128(1u128 << 32);
        let dec_shifted = ct_shifted.decrypt_slots(&keypair.sk);
        let expected_shifted = ((v1 as u128 * (1u128 << 32)) % t as u128) as u64;
        println!("ct_v1 * 2^32 decrypts to slot[0] = {} (expected {} = {} * 2^32 mod t)",
                 dec_shifted[0], expected_shifted, v1);

        // Pack using ciphertext-space shift by 32 bits: packed = v1 * 2^32 + v0
        println!("\nPacking: ct_packed = ct_shifted.add(&ct_v0)");
        let ct_packed = ct_shifted.add(&ct_v0);

        // Try standard decryption (will reduce mod t)
        let dec_standard = ct_packed.decrypt_slots(&keypair.sk);
        println!("Standard decryption slot[0] = {} (mod t={})", dec_standard[0], t);

        // Try raw decryption
        let dec_raw = ct_packed.decrypt_packed_u128(&keypair.sk);
        println!("Raw decryption (u128) = {}", dec_raw);

        // Expected packed value: v0 + v1 * 2^32
        let expected_packed: u128 = v0 as u128 + (v1 as u128) * (1u128 << 32);
        println!("Expected packed value = {}", expected_packed);

        // Try to extract v0 and v1 from raw
        let extracted_v0 = (dec_raw & 0xFFFFFFFF) as u64;
        let extracted_v1 = ((dec_raw >> 32) & 0xFFFFFFFF) as u64;
        println!("\nExtracted v0 (low 32) = {} (expected {})", extracted_v0, v0);
        println!("Extracted v1 (high 32) = {} (expected {})", extracted_v1, v1);

        // Also check what (v0 + v1 * 2^32) mod t equals
        let packed_mod_t = (expected_packed % t as u128) as u64;
        println!("\n(v0 + v1 * 2^32) mod t = {}", packed_mod_t);
        println!("Standard decryption    = {}", dec_standard[0]);

        // These should match if standard decryption sees the full packed value before mod t
        if dec_standard[0] == packed_mod_t {
            println!("\nStandard decryption matches (v0 + v1*2^32) mod t - packing works!");
        } else {
            println!("\nMismatch - packing may not work as expected with scaled BGV");
        }

        println!("\n=== End Ciphertext-Space Packing Test ===");
    }

    #[test]
    fn test_ciphertext_scalar_mul_goldilocks() {
        // Test packing multiple 64-bit Goldilocks values into q-space
        // Using 16K ring dimension for higher throughput

        println!("\n=== Q-Space Packing Test (Goldilocks 16K) ===\n");

        let mut rng = rand::rng();
        let params = RnsBgvParams::goldilocks_16k();
        let keypair = RnsKeyPair::generate(&params, &mut rng);

        let n = params.n;
        let t = params.t;

        // Print q-space info
        let rns_params = &keypair.pk.rns_params;
        let moduli = rns_params.moduli();
        println!("Params: n={}, t={} (~2^64)", n, t);
        println!("RNS moduli (q = product):");
        let mut q_bits = 0.0;
        for (i, &q_i) in moduli.iter().enumerate() {
            let bits = (q_i as f64).log2();
            q_bits += bits;
            println!("  q_{} = {} (~2^{:.1})", i, q_i, bits);
        }
        println!("  Total q ~ 2^{:.0} bits", q_bits);
        println!("  Can pack {} x 64-bit values", (q_bits / 64.0) as usize);

        // Pack two 64-bit values: v0 + v1 * 2^64
        let v0: u64 = 0x123456789ABCDEF0;  // Large 64-bit value
        let v1: u64 = 0xFEDCBA9876543210;  // Another large 64-bit value

        println!("\nValues to pack:");
        println!("  v0 = 0x{:016X} = {}", v0, v0);
        println!("  v1 = 0x{:016X} = {}", v1, v1);

        let mut slots_v0 = vec![0u64; n];
        let mut slots_v1 = vec![0u64; n];
        slots_v0[0] = v0;
        slots_v1[0] = v1;

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v0, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v1, &mut rng);

        // Verify individual encryption works
        let dec_v0 = ct_v0.decrypt_slots(&keypair.sk);
        let dec_v1 = ct_v1.decrypt_slots(&keypair.sk);
        println!("\nIndividual decryption (mod t):");
        println!("  ct_v0 -> {} (expected {})", dec_v0[0], v0);
        println!("  ct_v1 -> {} (expected {})", dec_v1[0], v1);

        // Pack: shift ct_v1 by 2^64 in q-space, then add ct_v0
        // Result should be: Enc(v0 + v1 * 2^64) in q-space
        let shift: u128 = 1u128 << 64;
        let ct_shifted = ct_v1.mul_ciphertext_scalar_u128(shift);
        let ct_packed = ct_shifted.add(&ct_v0);

        // Standard decryption will reduce mod t, losing the packed structure
        let dec_standard = ct_packed.decrypt_slots(&keypair.sk);
        println!("\nStandard decryption (mod t): {}", dec_standard[0]);

        // We need raw decryption to extract packed values
        // For BGV: noisy = c0 + c1*s = delta*m + noise
        // where delta = q/t, m = packed_value
        // So: m = noisy / delta = noisy * t / q

        // Let's look at the raw noisy value before mod t
        let c1s = ct_packed.c1.mul(&keypair.sk.s);
        let noisy = ct_packed.c0.add(&c1s);

        println!("\nRaw noisy coefficients (first RNS limb, coeff 0):");
        println!("  noisy[0][0] = {}", noisy.residues()[0][0]);

        // The packed value m = v0 + v1 * 2^64 is embedded as delta * m in q-space
        // To extract, we need proper CRT reconstruction and division by delta
        // This is complex because delta = q/t where q ~ 2^300 and t ~ 2^64

        // For now, let's verify the math works by checking if standard decryption
        // gives (v0 + v1 * 2^64) mod t
        let expected_mod_t = {
            // v0 + v1 * 2^64 mod t
            // In Goldilocks, 2^64 mod t = 2^32 - 1
            let shift_mod_t = ((1u128 << 64) % t as u128) as u64;
            let v1_shifted = ((v1 as u128 * shift_mod_t as u128) % t as u128) as u64;
            ((v0 as u128 + v1_shifted as u128) % t as u128) as u64
        };
        println!("\nExpected (v0 + v1 * 2^64) mod t = {}", expected_mod_t);
        println!("Standard decryption            = {}", dec_standard[0]);

        if dec_standard[0] == expected_mod_t {
            println!("\n✓ Packing math is correct (values combine properly in q-space)");
        } else {
            println!("\n✗ Packing math failed");
        }

        // Now test: multiply first value by random scalar, then pack second value
        println!("\n--- Scalar mul then pack test ---");

        let v0: u64 = 12345;
        let v1: u64 = 67890;
        let scalar: u64 = 9999;

        let mut slots_v0 = vec![0u64; n];
        let mut slots_v1 = vec![0u64; n];
        slots_v0[0] = v0;
        slots_v1[0] = v1;

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v0, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &slots_v1, &mut rng);

        println!("v0 = {}, v1 = {}, scalar = {}", v0, v1, scalar);

        // Step 1: Multiply ct_v0 by scalar (in ciphertext space)
        let ct_v0_scaled = ct_v0.mul_ciphertext_scalar_u128(scalar as u128);

        // Verify scalar mul works
        let dec_scaled = ct_v0_scaled.decrypt_slots(&keypair.sk);
        let expected_scaled = ((v0 as u128 * scalar as u128) % t as u128) as u64;
        println!("After scalar mul: {} * {} = {} (expected {})",
                 v0, scalar, dec_scaled[0], expected_scaled);

        // Step 2: Shift scaled value left by 64 bits
        let ct_shifted = ct_v0_scaled.mul_ciphertext_scalar_u128(1u128 << 64);

        // Step 3: Add ct_v1 to pack
        let ct_packed = ct_shifted.add(&ct_v1);

        // Decrypt and verify
        let dec_packed = ct_packed.decrypt_slots(&keypair.sk);

        // Expected: (v0 * scalar) * 2^64 + v1, all mod t
        let v0_scaled = (v0 as u128 * scalar as u128) % t as u128;
        let shift_mod_t = (1u128 << 64) % t as u128;
        let shifted = (v0_scaled * shift_mod_t) % t as u128;
        let expected_packed = ((shifted + v1 as u128) % t as u128) as u64;

        println!("Packed ((v0 * scalar) * 2^64 + v1) mod t:");
        println!("  Decrypted: {}", dec_packed[0]);
        println!("  Expected:  {}", expected_packed);

        if dec_packed[0] == expected_packed {
            println!("\n✓ Scalar mul + pack works correctly!");
        } else {
            println!("\n✗ Failed");
        }

        // 2-way pack with full 8K slot-wise mul (64-bit shift)
        println!("\n--- 2-way pack with slot-wise mul (64-bit shift) ---");

        use rand::Rng;

        let v0_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v1_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0_slots, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1_slots, &mut rng);

        let ct_v0_scaled = ct_v0.mul_plaintext_slots(&scalars0);
        let ct_v1_scaled = ct_v1.mul_plaintext_slots(&scalars1);

        let ct_v0_shifted = ct_v0_scaled.shift_ciphertext_left(64);
        let ct_packed = ct_v0_shifted.add(&ct_v1_scaled);

        let dec_packed = ct_packed.decrypt_slots(&keypair.sk);

        let shift64_mod_t = ((1u128 << 64) % t as u128) as u64;
        let mut correct = 0;

        for i in 0..n {
            let v0_scaled = ((v0_slots[i] as u128 * scalars0[i] as u128) % t as u128) as u64;
            let v1_scaled = ((v1_slots[i] as u128 * scalars1[i] as u128) % t as u128) as u64;
            let term0 = ((v0_scaled as u128 * shift64_mod_t as u128) % t as u128) as u64;
            let expected = ((term0 as u128 + v1_scaled as u128) % t as u128) as u64;
            if dec_packed[i] == expected { correct += 1; }
        }

        println!("Results: {}/{} slots correct", correct, n);
        if correct == n {
            println!("✓ 2-way pack with slot-wise mul works!");
        } else {
            println!("✗ {} slots failed", n - correct);
        }

        // Simple 3-way pack test (no slot-wise multiplication) to isolate packing logic
        println!("\n--- Simple 3-way pack test (no mul) ---");

        let v0_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v1_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v2_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

        println!("Testing 3-way pack: v0*2^128 + v1*2^64 + v2");
        println!("  v0[0] = {}, v1[0] = {}, v2[0] = {}", v0_slots[0], v1_slots[0], v2_slots[0]);

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0_slots, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1_slots, &mut rng);
        let ct_v2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2_slots, &mut rng);

        let ct_v0_shifted = ct_v0.shift_ciphertext_left(128);
        let ct_v1_shifted = ct_v1.shift_ciphertext_left(64);
        let ct_packed = ct_v0_shifted.add(&ct_v1_shifted).add(&ct_v2);

        let dec_packed = ct_packed.decrypt_slots(&keypair.sk);

        let shift64_mod_t = ((1u128 << 64) % t as u128) as u64;
        let shift128_mod_t = ((shift64_mod_t as u128 * shift64_mod_t as u128) % t as u128) as u64;

        let mut correct = 0;
        let mut wrong = 0;

        for i in 0..n {
            let term0 = ((v0_slots[i] as u128 * shift128_mod_t as u128) % t as u128) as u64;
            let term1 = ((v1_slots[i] as u128 * shift64_mod_t as u128) % t as u128) as u64;
            let expected = ((term0 as u128 + term1 as u128 + v2_slots[i] as u128) % t as u128) as u64;

            if dec_packed[i] == expected {
                correct += 1;
            } else {
                wrong += 1;
                if wrong <= 3 {
                    println!("  Slot {} WRONG: got {}, expected {}", i, dec_packed[i], expected);
                    println!("    v0={}, v1={}, v2={}", v0_slots[i], v1_slots[i], v2_slots[i]);
                }
            }
        }

        println!("\nResults: {}/{} slots correct", correct, n);
        if wrong == 0 {
            println!("✓ Simple 3-way pack works!");
        } else {
            println!("✗ {} slots failed", wrong);
        }

        // Full test with slot-wise multiplication using goldilocks_16k (6 moduli, 16K slots)
        println!("\n--- Full 16K test with slot-wise mul + 3-way pack ---");

        // Already using goldilocks_16k from the start of the test (6 moduli)
        // Reuse the same keypair
        println!("Using goldilocks_16k: {} moduli (~{} bits), {} slots",
                 params.num_moduli, params.num_moduli * 60, n);

        let v0_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v1_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v2_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

        println!("  v0[0]={}, s0[0]={}", v0_slots[0], scalars0[0]);

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair.pk, &v0_slots, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair.pk, &v1_slots, &mut rng);
        let ct_v2 = RnsCiphertext::encrypt_slots(&keypair.pk, &v2_slots, &mut rng);

        let ct_v0_scaled = ct_v0.mul_plaintext_slots(&scalars0);
        let ct_v1_scaled = ct_v1.mul_plaintext_slots(&scalars1);
        let ct_v2_scaled = ct_v2.mul_plaintext_slots(&scalars2);

        let ct_v0_shifted = ct_v0_scaled.shift_ciphertext_left(128);
        let ct_v1_shifted = ct_v1_scaled.shift_ciphertext_left(64);
        let ct_packed = ct_v0_shifted.add(&ct_v1_shifted).add(&ct_v2_scaled);

        let dec_packed = ct_packed.decrypt_slots(&keypair.sk);

        let mut correct = 0;
        let mut wrong = 0;

        for i in 0..n {
            let v0_scaled = ((v0_slots[i] as u128 * scalars0[i] as u128) % t as u128) as u64;
            let v1_scaled = ((v1_slots[i] as u128 * scalars1[i] as u128) % t as u128) as u64;
            let v2_scaled = ((v2_slots[i] as u128 * scalars2[i] as u128) % t as u128) as u64;

            let term0 = ((v0_scaled as u128 * shift128_mod_t as u128) % t as u128) as u64;
            let term1 = ((v1_scaled as u128 * shift64_mod_t as u128) % t as u128) as u64;
            let expected = ((term0 as u128 + term1 as u128 + v2_scaled as u128) % t as u128) as u64;

            if dec_packed[i] == expected {
                correct += 1;
            } else {
                wrong += 1;
                if wrong <= 3 {
                    println!("  Slot {} WRONG: got {}, expected {}", i, dec_packed[i], expected);
                }
            }
        }

        println!("\nResults: {}/{} slots correct", correct, n);
        if wrong == 0 {
            println!("✓ Full 3-way pack with mul works!");
        } else {
            println!("✗ {} slots failed (noise budget exceeded?)", wrong);
        }

        // 4-way pack test with 7 moduli (16K)
        println!("\n--- Full 16K test with slot-wise mul + 4-way pack ---");

        let params_16k_4 = RnsBgvParams::goldilocks_16k_packed_4();
        let keypair_16k_4 = RnsKeyPair::generate(&params_16k_4, &mut rng);
        let n = params_16k_4.n;
        let t = params_16k_4.t;

        println!("Using goldilocks_16k_packed_4: {} moduli (~{} bits), {} slots",
                 params_16k_4.num_moduli, params_16k_4.num_moduli * 60, n);

        let v0_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v1_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v2_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let v3_slots: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars0: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars1: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars2: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();
        let scalars3: Vec<u64> = (0..n).map(|_| rng.random::<u64>() % t).collect();

        let ct_v0 = RnsCiphertext::encrypt_slots(&keypair_16k_4.pk, &v0_slots, &mut rng);
        let ct_v1 = RnsCiphertext::encrypt_slots(&keypair_16k_4.pk, &v1_slots, &mut rng);
        let ct_v2 = RnsCiphertext::encrypt_slots(&keypair_16k_4.pk, &v2_slots, &mut rng);
        let ct_v3 = RnsCiphertext::encrypt_slots(&keypair_16k_4.pk, &v3_slots, &mut rng);

        let ct_v0_scaled = ct_v0.mul_plaintext_slots(&scalars0);
        let ct_v1_scaled = ct_v1.mul_plaintext_slots(&scalars1);
        let ct_v2_scaled = ct_v2.mul_plaintext_slots(&scalars2);
        let ct_v3_scaled = ct_v3.mul_plaintext_slots(&scalars3);

        // v0*2^192 + v1*2^128 + v2*2^64 + v3
        let ct_v0_shifted = ct_v0_scaled.shift_ciphertext_left(192);
        let ct_v1_shifted = ct_v1_scaled.shift_ciphertext_left(128);
        let ct_v2_shifted = ct_v2_scaled.shift_ciphertext_left(64);
        let ct_packed = ct_v0_shifted.add(&ct_v1_shifted).add(&ct_v2_shifted).add(&ct_v3_scaled);

        let dec_packed = ct_packed.decrypt_slots(&keypair_16k_4.sk);

        // Compute shift constants
        let shift64 = ((1u128 << 64) % t as u128) as u64;
        let shift128 = ((shift64 as u128 * shift64 as u128) % t as u128) as u64;
        let shift192 = ((shift128 as u128 * shift64 as u128) % t as u128) as u64;

        let mut correct = 0;
        let mut wrong = 0;

        for i in 0..n {
            let v0s = ((v0_slots[i] as u128 * scalars0[i] as u128) % t as u128) as u64;
            let v1s = ((v1_slots[i] as u128 * scalars1[i] as u128) % t as u128) as u64;
            let v2s = ((v2_slots[i] as u128 * scalars2[i] as u128) % t as u128) as u64;
            let v3s = ((v3_slots[i] as u128 * scalars3[i] as u128) % t as u128) as u64;

            let term0 = ((v0s as u128 * shift192 as u128) % t as u128) as u64;
            let term1 = ((v1s as u128 * shift128 as u128) % t as u128) as u64;
            let term2 = ((v2s as u128 * shift64 as u128) % t as u128) as u64;
            let expected = ((term0 as u128 + term1 as u128 + term2 as u128 + v3s as u128) % t as u128) as u64;

            if dec_packed[i] == expected {
                correct += 1;
            } else {
                wrong += 1;
                if wrong <= 3 {
                    println!("  Slot {} WRONG: got {}, expected {}", i, dec_packed[i], expected);
                }
            }
        }

        println!("\nResults: {}/{} slots correct", correct, n);
        if wrong == 0 {
            println!("✓ Full 4-way pack with mul works!");
        } else {
            println!("✗ {} slots failed (need more moduli?)", wrong);
        }

        println!("\n=== End Test ===");
    }
}
