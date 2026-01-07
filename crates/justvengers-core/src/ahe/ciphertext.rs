//! BGV ciphertext and encryption/decryption operations.
//!
//! A BGV ciphertext is a pair (c0, c1) in R_q × R_q that encrypts
//! a message m in R_t.

use std::ops::{Add, Mul, Neg, Sub};

use rand::Rng;

use super::keys::{PublicKey, SecretKey};
use super::params::BgvParams;
use super::ring::RingPoly;
use super::sample::{sample_ternary, DiscreteGaussian};

/// A BGV ciphertext encrypting a message.
///
/// A ciphertext (c0, c1) encrypts message m if:
/// c0 + c1·s ≈ Δ·m (mod q)
///
/// where Δ = ⌊q/t⌋ is the scaling factor.
#[derive(Clone, Debug)]
pub struct Ciphertext {
    /// First ciphertext component.
    c0: RingPoly,
    /// Second ciphertext component.
    c1: RingPoly,
    /// Parameters.
    params: BgvParams,
}

impl Ciphertext {
    /// Creates a ciphertext from components.
    pub fn from_parts(c0: RingPoly, c1: RingPoly, params: BgvParams) -> Self {
        Self { c0, c1, params }
    }

    /// Returns the first component c0.
    pub fn c0(&self) -> &RingPoly {
        &self.c0
    }

    /// Returns the second component c1.
    pub fn c1(&self) -> &RingPoly {
        &self.c1
    }

    /// Returns the parameters.
    pub fn params(&self) -> &BgvParams {
        &self.params
    }

    /// Encrypts a scalar message.
    ///
    /// The message is encoded as a constant polynomial.
    ///
    /// # Algorithm
    ///
    /// 1. Encode message: m_poly = m (constant polynomial)
    /// 2. Sample random r from ternary distribution
    /// 3. Sample errors e0, e1 from discrete Gaussian
    /// 4. c0 = b·r + e0 + Δ·m_poly
    /// 5. c1 = a·r + e1
    pub fn encrypt_scalar<R: Rng>(pk: &PublicKey, message: u64, rng: &mut R) -> Self {
        let params = pk.params();

        // Encode message as constant polynomial scaled by Δ
        let m_scaled = (message % params.t) * params.delta;
        let m_poly = RingPoly::constant(m_scaled, params);

        // Sample randomness
        let r = sample_ternary(params, rng);

        // Sample errors
        let gaussian = DiscreteGaussian::new(params.sigma);
        let e0 = gaussian.sample_poly(params, rng);
        let e1 = gaussian.sample_poly(params, rng);

        // c0 = b·r + e0 + Δ·m
        let c0 = pk.b().clone() * &r + e0 + m_poly;

        // c1 = a·r + e1
        let c1 = pk.a().clone() * &r + e1;

        Self {
            c0,
            c1,
            params: *params,
        }
    }

    /// Encrypts a polynomial message.
    ///
    /// Each coefficient of the message polynomial is independently encrypted.
    pub fn encrypt_poly<R: Rng>(pk: &PublicKey, message: &[u64], rng: &mut R) -> Self {
        let params = pk.params();

        // Encode message polynomial scaled by Δ
        let m_coeffs: Vec<u64> = message
            .iter()
            .map(|&m| (m % params.t) * params.delta)
            .collect();
        let m_poly = RingPoly::from_slice(&m_coeffs, params);

        // Sample randomness
        let r = sample_ternary(params, rng);

        // Sample errors
        let gaussian = DiscreteGaussian::new(params.sigma);
        let e0 = gaussian.sample_poly(params, rng);
        let e1 = gaussian.sample_poly(params, rng);

        // c0 = b·r + e0 + Δ·m
        let c0 = pk.b().clone() * &r + e0 + m_poly;

        // c1 = a·r + e1
        let c1 = pk.a().clone() * &r + e1;

        Self {
            c0,
            c1,
            params: *params,
        }
    }

    /// Decrypts to a scalar message.
    ///
    /// # Algorithm
    ///
    /// 1. Compute noisy = c0 + c1·s (mod q)
    /// 2. Scale: m' = ⌊t·noisy/q⌉ (mod t)
    ///
    /// The rounding removes the noise if |noise| < Δ/2.
    pub fn decrypt_scalar(&self, sk: &SecretKey) -> u64 {
        let params = &self.params;

        // Compute c0 + c1·s
        let noisy = self.c0.clone() + self.c1.clone() * sk.poly();

        // Extract the constant coefficient
        let noisy_coeff = noisy.coeffs()[0];

        // Scale by t/q and round
        // m = round(t * noisy / q) mod t
        Self::scale_and_round(noisy_coeff, params.t, params.q)
    }

    /// Decrypts to a polynomial message.
    pub fn decrypt_poly(&self, sk: &SecretKey) -> Vec<u64> {
        let params = &self.params;

        // Compute c0 + c1·s
        let noisy = self.c0.clone() + self.c1.clone() * sk.poly();

        // Scale each coefficient by t/q and round
        noisy
            .coeffs()
            .iter()
            .map(|&c| Self::scale_and_round(c, params.t, params.q))
            .collect()
    }

    /// Scales x by t/q with rounding.
    ///
    /// Computes: round(t * x / q) mod t
    fn scale_and_round(x: u64, t: u64, q: u64) -> u64 {
        // We need to compute round(t * x / q)
        // = floor((t * x + q/2) / q)
        let scaled = (t as u128) * (x as u128) + (q as u128) / 2;
        let result = scaled / (q as u128);
        (result % (t as u128)) as u64
    }

    /// Adds two ciphertexts homomorphically.
    ///
    /// If ct1 encrypts m1 and ct2 encrypts m2,
    /// then ct1 + ct2 encrypts m1 + m2.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.params.q, other.params.q,
            "ciphertexts must have same parameters"
        );

        Self {
            c0: self.c0.clone() + &other.c0,
            c1: self.c1.clone() + &other.c1,
            params: self.params,
        }
    }

    /// Subtracts two ciphertexts homomorphically.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.params.q, other.params.q,
            "ciphertexts must have same parameters"
        );

        Self {
            c0: self.c0.clone() - &other.c0,
            c1: self.c1.clone() - &other.c1,
            params: self.params,
        }
    }

    /// Negates a ciphertext homomorphically.
    pub fn neg(&self) -> Self {
        Self {
            c0: self.c0.clone().neg(),
            c1: self.c1.clone().neg(),
            params: self.params,
        }
    }

    /// Multiplies a ciphertext by a scalar (plaintext) homomorphically.
    ///
    /// If ct encrypts m, then scalar_mul(ct, k) encrypts k·m.
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        Self {
            c0: self.c0.scalar_mul(scalar),
            c1: self.c1.scalar_mul(scalar),
            params: self.params,
        }
    }

    /// Adds a scalar (plaintext) to a ciphertext homomorphically.
    ///
    /// If ct encrypts m, then add_scalar(ct, k) encrypts m + k.
    pub fn add_scalar(&self, scalar: u64) -> Self {
        let k_scaled = (scalar % self.params.t) * self.params.delta;
        let k_poly = RingPoly::constant(k_scaled, &self.params);

        Self {
            c0: self.c0.clone() + k_poly,
            c1: self.c1.clone(),
            params: self.params,
        }
    }

    /// Computes a·ct + b homomorphically (linear targeted malleability).
    ///
    /// If ct encrypts m, returns a ciphertext encrypting a·m + b.
    pub fn linear(&self, a: u64, b: u64) -> Self {
        self.scalar_mul(a).add_scalar(b)
    }

    /// Re-randomizes the ciphertext (for circuit privacy).
    ///
    /// Returns a fresh encryption of the same message with new randomness.
    pub fn rerandomize<R: Rng>(&self, pk: &PublicKey, rng: &mut R) -> Self {
        let params = &self.params;

        // Encrypt zero
        let r = sample_ternary(params, rng);
        let gaussian = DiscreteGaussian::new(params.sigma);
        let e0 = gaussian.sample_poly(params, rng);
        let e1 = gaussian.sample_poly(params, rng);

        let zero_c0 = pk.b().clone() * &r + e0;
        let zero_c1 = pk.a().clone() * &r + e1;

        // Add encryption of zero to rerandomize
        Self {
            c0: self.c0.clone() + zero_c0,
            c1: self.c1.clone() + zero_c1,
            params: self.params,
        }
    }
}

// Trait implementations for ergonomic usage

impl Add for Ciphertext {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Ciphertext::add(&self, &rhs)
    }
}

impl<'a> Add<&'a Ciphertext> for Ciphertext {
    type Output = Self;

    fn add(self, rhs: &'a Ciphertext) -> Self::Output {
        Ciphertext::add(&self, rhs)
    }
}

impl Sub for Ciphertext {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Ciphertext::sub(&self, &rhs)
    }
}

impl<'a> Sub<&'a Ciphertext> for Ciphertext {
    type Output = Self;

    fn sub(self, rhs: &'a Ciphertext) -> Self::Output {
        Ciphertext::sub(&self, rhs)
    }
}

impl Neg for Ciphertext {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Ciphertext::neg(&self)
    }
}

impl Mul<u64> for Ciphertext {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

#[cfg(test)]
mod ciphertext_tests {
    use super::*;
    use crate::ahe::keys::KeyPair;
    use crate::ahe::params::ParamSet;
    use mpz_core::{Block, prg::Prg};
    use rand::SeedableRng;

    fn setup() -> (BgvParams, KeyPair, Prg) {
        let params = ParamSet::Toy.params();
        let mut rng = Prg::from_seed(Block::ZERO);
        let keypair = KeyPair::generate(&params, &mut rng);
        (params, keypair, rng)
    }

    #[test]
    fn test_encrypt_decrypt_scalar() {
        let (_, keypair, mut rng) = setup();

        // Use small messages to avoid noise issues with toy parameters
        for m in [0, 1, 42, 100, 1000] {
            let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
            let decrypted = ct.decrypt_scalar(&keypair.sk);

            assert_eq!(decrypted, m, "decryption failed for message {}", m);
        }
    }

    #[test]
    fn test_homomorphic_add() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m1 = 100u64;
        let m2 = 200u64;

        let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
        let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

        let ct_sum = ct1 + ct2;
        let decrypted = ct_sum.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (m1 + m2) % t);
    }

    #[test]
    fn test_homomorphic_sub() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m1 = 300u64;
        let m2 = 100u64;

        let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
        let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);

        let ct_diff = ct1 - ct2;
        let decrypted = ct_diff.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (m1 - m2) % t);
    }

    #[test]
    fn test_homomorphic_neg() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m = 100u64;

        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_neg = -ct;
        let decrypted = ct_neg.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (t - m) % t);
    }

    #[test]
    fn test_homomorphic_scalar_mul() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m = 50u64;
        let k = 7u64;

        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_scaled = ct * k;
        let decrypted = ct_scaled.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (m * k) % t);
    }

    #[test]
    fn test_homomorphic_add_scalar() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m = 50u64;
        let k = 30u64;

        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_added = ct.add_scalar(k);
        let decrypted = ct_added.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (m + k) % t);
    }

    #[test]
    fn test_linear_malleability() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m = 10u64;
        let a = 5u64;
        let b = 7u64;

        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_linear = ct.linear(a, b);
        let decrypted = ct_linear.decrypt_scalar(&keypair.sk);

        assert_eq!(decrypted, (a * m + b) % t);
    }

    #[test]
    fn test_rerandomize() {
        let (_, keypair, mut rng) = setup();

        let m = 42u64;

        let ct = Ciphertext::encrypt_scalar(&keypair.pk, m, &mut rng);
        let ct_rerand = ct.rerandomize(&keypair.pk, &mut rng);

        // Ciphertexts should be different
        assert_ne!(ct.c0.coeffs(), ct_rerand.c0.coeffs());

        // But decrypt to same value
        let dec_orig = ct.decrypt_scalar(&keypair.sk);
        let dec_rerand = ct_rerand.decrypt_scalar(&keypair.sk);
        assert_eq!(dec_orig, dec_rerand);
    }

    #[test]
    fn test_multiple_operations() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let m1 = 10u64;
        let m2 = 20u64;
        let m3 = 5u64;

        let ct1 = Ciphertext::encrypt_scalar(&keypair.pk, m1, &mut rng);
        let ct2 = Ciphertext::encrypt_scalar(&keypair.pk, m2, &mut rng);
        let ct3 = Ciphertext::encrypt_scalar(&keypair.pk, m3, &mut rng);

        // (ct1 + ct2) * 3 - ct3
        let result = (ct1 + ct2) * 3 - ct3;
        let decrypted = result.decrypt_scalar(&keypair.sk);

        let expected = ((m1 + m2) * 3 - m3) % t;
        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_encrypt_decrypt_poly() {
        let (_, keypair, mut rng) = setup();
        let t = keypair.pk.params().t;

        let message = vec![1, 2, 3, 4, 5];
        let ct = Ciphertext::encrypt_poly(&keypair.pk, &message, &mut rng);
        let decrypted = ct.decrypt_poly(&keypair.sk);

        // Check first few coefficients
        for (i, (&m, &d)) in message.iter().zip(decrypted.iter()).enumerate() {
            assert_eq!(d, m % t, "mismatch at coefficient {}", i);
        }
    }
}
