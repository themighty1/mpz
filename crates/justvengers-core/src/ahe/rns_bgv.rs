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

use super::params::{RnsBgvParams, GOLDILOCKS};
use super::rns::{RnsParams, RnsPoly};
use super::slot::SlotEncoder;

/// RNS-based BGV secret key.
#[derive(Clone, Debug)]
pub struct RnsSecretKey {
    /// Secret polynomial s in RNS form (ternary coefficients).
    s: RnsPoly,
    /// RNS parameters.
    rns_params: RnsParams,
    /// BGV parameters.
    bgv_params: RnsBgvParams,
}

/// RNS-based BGV public key.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct RnsKeyPair {
    /// The secret key.
    pub sk: RnsSecretKey,
    /// The public key.
    pub pk: RnsPublicKey,
}

/// RNS-based BGV ciphertext with slot packing support.
#[derive(Clone, Debug)]
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
    pub fn generate_goldilocks<R: Rng>(rng: &mut R) -> Self {
        let params = RnsBgvParams::goldilocks();
        Self::generate(&params, rng)
    }
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
}
