//! BGV key generation.
//!
//! Generates secret and public keys for BGV encryption.

use rand::Rng;

use super::params::BgvParams;
use super::ring::RingPoly;
use super::sample::{sample_ternary, sample_uniform, DiscreteGaussian};

/// BGV secret key.
///
/// The secret key is a polynomial with small coefficients (ternary).
#[derive(Clone, Debug)]
pub struct SecretKey {
    /// The secret polynomial s.
    s: RingPoly,
    /// Parameters used to generate this key.
    params: BgvParams,
}

impl SecretKey {
    /// Returns the secret polynomial.
    pub fn poly(&self) -> &RingPoly {
        &self.s
    }

    /// Returns the parameters.
    pub fn params(&self) -> &BgvParams {
        &self.params
    }
}

/// BGV public key.
///
/// The public key is (a, b) where:
/// - a is a uniformly random polynomial
/// - b = -a·s + e for small error e
#[derive(Clone, Debug)]
pub struct PublicKey {
    /// Random polynomial a.
    a: RingPoly,
    /// b = -a·s + e.
    b: RingPoly,
    /// Parameters used to generate this key.
    params: BgvParams,
}

impl PublicKey {
    /// Returns the first component (a).
    pub fn a(&self) -> &RingPoly {
        &self.a
    }

    /// Returns the second component (b = -a·s + e).
    pub fn b(&self) -> &RingPoly {
        &self.b
    }

    /// Returns the parameters.
    pub fn params(&self) -> &BgvParams {
        &self.params
    }
}

/// A BGV key pair (secret key + public key).
#[derive(Clone, Debug)]
pub struct KeyPair {
    /// The secret key.
    pub sk: SecretKey,
    /// The public key.
    pub pk: PublicKey,
}

impl KeyPair {
    /// Generates a new BGV key pair.
    ///
    /// # Algorithm
    ///
    /// 1. Sample secret key s from ternary distribution {-1, 0, 1}
    /// 2. Sample random a uniformly from R_q
    /// 3. Sample error e from discrete Gaussian
    /// 4. Compute b = -a·s + e
    /// 5. Return (sk = s, pk = (a, b))
    pub fn generate<R: Rng>(params: &BgvParams, rng: &mut R) -> Self {
        // Sample secret key from ternary distribution
        let s = sample_ternary(params, rng);

        // Sample random a
        let a = sample_uniform(params, rng);

        // Sample error from discrete Gaussian
        let gaussian = DiscreteGaussian::new(params.sigma);
        let e = gaussian.sample_poly(params, rng);

        // Compute b = -a·s + e
        let neg_a_s = (a.clone() * &s).neg();
        let b = neg_a_s + e;

        let sk = SecretKey {
            s,
            params: *params,
        };

        let pk = PublicKey {
            a,
            b,
            params: *params,
        };

        Self { sk, pk }
    }

    /// Returns a reference to the secret key.
    pub fn secret_key(&self) -> &SecretKey {
        &self.sk
    }

    /// Returns a reference to the public key.
    pub fn public_key(&self) -> &PublicKey {
        &self.pk
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;
    use crate::ahe::params::ParamSet;
    use mpz_core::{Block, prg::Prg};
    use rand::SeedableRng;

    #[test]
    fn test_keygen() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();

        let keypair = KeyPair::generate(&params, &mut rng);

        // Check dimensions
        assert_eq!(keypair.sk.s.dimension(), params.n);
        assert_eq!(keypair.pk.a.dimension(), params.n);
        assert_eq!(keypair.pk.b.dimension(), params.n);

        // Check secret key is ternary
        for &c in keypair.sk.s.coeffs() {
            assert!(
                c == 0 || c == 1 || c == params.q - 1,
                "sk coefficient {} not ternary",
                c
            );
        }
    }

    #[test]
    fn test_keygen_deterministic() {
        let params = ParamSet::Toy.params();

        let mut rng1 = Prg::from_seed(Block::ZERO);
        let mut rng2 = Prg::from_seed(Block::ZERO);

        let kp1 = KeyPair::generate(&params, &mut rng1);
        let kp2 = KeyPair::generate(&params, &mut rng2);

        // Same seed should give same keys
        assert_eq!(kp1.sk.s.coeffs(), kp2.sk.s.coeffs());
        assert_eq!(kp1.pk.a.coeffs(), kp2.pk.a.coeffs());
        assert_eq!(kp1.pk.b.coeffs(), kp2.pk.b.coeffs());
    }

    #[test]
    fn test_keygen_different_seeds() {
        let params = ParamSet::Toy.params();

        let mut rng1 = Prg::from_seed(Block::ZERO);
        let mut rng2 = Prg::from_seed(Block::new([1u8; 16]));

        let kp1 = KeyPair::generate(&params, &mut rng1);
        let kp2 = KeyPair::generate(&params, &mut rng2);

        // Different seeds should give different keys
        assert_ne!(kp1.sk.s.coeffs(), kp2.sk.s.coeffs());
    }
}
