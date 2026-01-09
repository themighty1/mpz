//! CL-based Additively Homomorphic Encryption for arbitrary prime fields.
//!
//! This module wraps the Castagnos-Laguillaumie (CL) encryption scheme,
//! supporting arbitrary prime fields including Goldilocks.

use class_group::primitives::cl_enc::{
    self, CLGroup as CLGroupInner, Ciphertext as CiphertextInner,
    PublicKey as PKInner, SecretKey as SKInner,
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

/// CL group parameters for encryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLGroup {
    inner: CLGroupInner,
}

/// CL secret key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLSecretKey {
    inner: SKInner,
}

/// CL public key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLPublicKey {
    inner: PKInner,
}

/// CL key pair.
#[derive(Clone, Debug)]
pub struct CLKeyPair {
    /// Secret key.
    pub sk: CLSecretKey,
    /// Public key.
    pub pk: CLPublicKey,
}

/// CL ciphertext supporting homomorphic operations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CLCiphertext {
    inner: CiphertextInner,
}

impl CLGroup {
    /// Creates a CL group for Goldilocks with hardcoded 1600-bit parameters.
    ///
    /// This provides 128-bit security and loads instantly (no computation).
    pub fn new_goldilocks_hardcoded() -> Self {
        Self {
            inner: CLGroupInner::goldilocks_hardcoded(),
        }
    }

    /// Creates a CL group for the Goldilocks field (2^64 - 2^32 + 1).
    ///
    /// # Arguments
    /// * `security_bits` - Security parameter (1600 for 128-bit security)
    #[allow(dead_code)]
    pub fn new_goldilocks(security_bits: usize) -> Self {
        let seed = default_seed();
        Self {
            inner: CLGroupInner::new_goldilocks(security_bits, &seed),
        }
    }

    /// Creates a CL group for an arbitrary prime field.
    ///
    /// # Arguments
    /// * `field_modulus` - The prime field order q
    /// * `security_bits` - Security parameter
    pub fn new(field_modulus: u64, security_bits: usize) -> Self {
        let seed = default_seed();
        let q = BigInt::from(field_modulus);
        Self {
            inner: CLGroupInner::new(&q, security_bits, &seed),
        }
    }

    /// Creates a CL group with custom seed for deterministic setup.
    pub fn new_with_seed(field_modulus: u64, security_bits: usize, seed: &[u8]) -> Self {
        let seed_bigint = BigInt::from_signed_bytes_be(seed);
        let q = BigInt::from(field_modulus);
        Self {
            inner: CLGroupInner::new(&q, security_bits, &seed_bigint),
        }
    }

    /// Generates a new key pair for this group.
    pub fn keygen(&self) -> CLKeyPair {
        let (sk_inner, pk_inner) = self.inner.keygen();
        CLKeyPair {
            sk: CLSecretKey { inner: sk_inner },
            pk: CLPublicKey { inner: pk_inner },
        }
    }

    /// Returns the field modulus q.
    pub fn field_modulus(&self) -> &BigInt {
        &self.inner.q
    }
}

impl CLSecretKey {
    /// Decrypts a ciphertext.
    pub fn decrypt(&self, group: &CLGroup, ct: &CLCiphertext) -> BigInt {
        cl_enc::decrypt(&group.inner, &self.inner, &ct.inner)
    }

    /// Decrypts and returns as u64 (for Goldilocks-sized values).
    pub fn decrypt_u64(&self, group: &CLGroup, ct: &CLCiphertext) -> u64 {
        let result = self.decrypt(group, ct);
        // Convert BigInt to u64
        let (_, bytes) = result.to_bytes_be();
        let mut arr = [0u8; 8];
        let start = 8usize.saturating_sub(bytes.len());
        arr[start..].copy_from_slice(&bytes[..bytes.len().min(8)]);
        u64::from_be_bytes(arr)
    }
}

impl CLPublicKey {
    /// Encrypts a BigInt value.
    pub fn encrypt(&self, group: &CLGroup, m: &BigInt) -> (CLCiphertext, CLSecretKey) {
        let (ct, r) = cl_enc::encrypt(&group.inner, &self.inner, m);
        (CLCiphertext { inner: ct }, CLSecretKey { inner: r })
    }

    /// Encrypts a u64 value.
    pub fn encrypt_u64(&self, group: &CLGroup, m: u64) -> CLCiphertext {
        let m_bigint = BigInt::from(m);
        let (ct, _) = self.encrypt(group, &m_bigint);
        ct
    }

    /// Encrypts with predefined randomness.
    pub fn encrypt_with_randomness(
        &self,
        group: &CLGroup,
        m: &BigInt,
        r: &CLSecretKey,
    ) -> CLCiphertext {
        let ct = cl_enc::encrypt_with_randomness(&group.inner, &self.inner, m, &r.inner);
        CLCiphertext { inner: ct }
    }
}

impl CLCiphertext {
    /// Homomorphic addition: Enc(m1) + Enc(m2) = Enc(m1 + m2 mod q).
    pub fn add(&self, other: &CLCiphertext) -> CLCiphertext {
        CLCiphertext {
            inner: cl_enc::eval_sum(&self.inner, &other.inner),
        }
    }

    /// Homomorphic scalar multiplication: c * Enc(m) = Enc(c * m mod q).
    pub fn scalar_mul(&self, scalar: u64) -> CLCiphertext {
        let s = BigInt::from(scalar);
        CLCiphertext {
            inner: cl_enc::eval_scal(&self.inner, &s),
        }
    }

    /// Homomorphic scalar multiplication with BigInt.
    pub fn scalar_mul_bigint(&self, scalar: &BigInt) -> CLCiphertext {
        CLCiphertext {
            inner: cl_enc::eval_scal(&self.inner, scalar),
        }
    }

    /// Homomorphic negation: -Enc(m) = Enc(-m mod q).
    pub fn neg(&self) -> CLCiphertext {
        let neg_one = BigInt::from(-1);
        self.scalar_mul_bigint(&neg_one)
    }

    /// Homomorphic subtraction: Enc(m1) - Enc(m2) = Enc(m1 - m2 mod q).
    pub fn sub(&self, other: &CLCiphertext) -> CLCiphertext {
        self.add(&other.neg())
    }
}

/// Default seed for deterministic group generation.
fn default_seed() -> BigInt {
    "314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848"
        .parse()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goldilocks_encrypt_decrypt() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();

        let m = 12345u64;
        let ct = keypair.pk.encrypt_u64(&group, m);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct);

        assert_eq!(decrypted, m);
    }

    #[test]
    fn test_goldilocks_homomorphic_add() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();

        let m1 = 100u64;
        let m2 = 200u64;

        let ct1 = keypair.pk.encrypt_u64(&group, m1);
        let ct2 = keypair.pk.encrypt_u64(&group, m2);

        let ct_sum = ct1.add(&ct2);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_sum);

        assert_eq!(decrypted, m1 + m2);
    }

    #[test]
    fn test_goldilocks_scalar_mul() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();

        let m = 7u64;
        let scalar = 6u64;

        let ct = keypair.pk.encrypt_u64(&group, m);
        let ct_scaled = ct.scalar_mul(scalar);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_scaled);

        assert_eq!(decrypted, m * scalar);
    }

    #[test]
    fn test_goldilocks_linear_combination() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();

        let m1 = 10u64;
        let m2 = 20u64;
        let a1 = 3u64;
        let a2 = 5u64;

        let ct1 = keypair.pk.encrypt_u64(&group, m1);
        let ct2 = keypair.pk.encrypt_u64(&group, m2);

        // Compute a1*m1 + a2*m2
        let ct_result = ct1.scalar_mul(a1).add(&ct2.scalar_mul(a2));
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_result);

        assert_eq!(decrypted, a1 * m1 + a2 * m2);
    }

    #[test]
    fn test_sum_many_values() {
        let group = CLGroup::new_goldilocks_hardcoded();
        let keypair = group.keygen();

        let values: Vec<u64> = (1..=10).collect();
        let expected_sum: u64 = values.iter().sum();

        let ciphertexts: Vec<CLCiphertext> = values
            .iter()
            .map(|&v| keypair.pk.encrypt_u64(&group, v))
            .collect();

        let ct_sum = ciphertexts
            .iter()
            .skip(1)
            .fold(ciphertexts[0].clone(), |acc, ct| acc.add(ct));

        let decrypted = keypair.sk.decrypt_u64(&group, &ct_sum);
        assert_eq!(decrypted, expected_sum);
    }
}
