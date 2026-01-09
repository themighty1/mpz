//! CL-based Additively Homomorphic Encryption.
//!
//! This module implements the Castagnos-Laguillaumie (CL) encryption scheme based on
//! class groups of imaginary quadratic orders. Unlike BGV which uses lattice-based
//! cryptography, CL encryption provides:
//!
//! - **Linear homomorphism**: Supports addition and scalar multiplication of ciphertexts
//! - **No noise accumulation**: Unlike BGV, CL encryption doesn't have noise that grows
//! - **Deterministic decryption**: Decryption is exact without rounding
//!
//! # Security
//!
//! Security is based on:
//! - Hardness of computing the class group order
//! - Discrete logarithm problem in the class group (with a DL-easy subgroup for plaintexts)
//!
//! # Usage
//!
//! ```ignore
//! use mpz_justvengers_core::ahe::cl_ahe::*;
//!
//! // Setup CL group parameters
//! let group = CLGroup::new(1600); // 128-bit security
//!
//! // Generate keypair
//! let (sk, pk) = group.keygen();
//!
//! // Encrypt values
//! let ct1 = pk.encrypt(&group, 42);
//! let ct2 = pk.encrypt(&group, 17);
//!
//! // Homomorphic addition
//! let ct_sum = ct1.add(&ct2);
//!
//! // Decrypt
//! let result = sk.decrypt(&group, &ct_sum); // result = 59
//! ```

use class_group::primitives::cl_dl_public_setup::{
    self as cl, CLGroup as CLGroupInner, Ciphertext as CLCiphertextInner,
    PK as CLPublicKeyInner, SK as CLSecretKeyInner,
};
use class_group::BinaryQF;
use curv::arithmetic::traits::*;
use curv::elliptic::curves::{secp256_k1::Secp256k1, Scalar};
use curv::BigInt;
use serde::{Deserialize, Serialize};

/// Default security parameter in bits (discriminant size).
/// 1600 bits provides approximately 128-bit security.
pub const DEFAULT_SECURITY_PARAMETER: usize = 1600;

/// CL group parameters for encryption.
///
/// The group is constructed with a DL-easy subgroup of order q (the secp256k1 group order),
/// allowing efficient discrete log computation for plaintexts while keeping the full
/// group structure secure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLGroup {
    inner: CLGroupInner,
}

/// CL secret key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLSecretKey {
    inner: CLSecretKeyInner,
}

/// CL public key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CLPublicKey {
    inner: CLPublicKeyInner,
}

/// CL key pair.
#[derive(Clone, Debug)]
pub struct CLKeyPair {
    /// The secret key.
    pub sk: CLSecretKey,
    /// The public key.
    pub pk: CLPublicKey,
}

/// CL ciphertext.
///
/// Encrypts a scalar value from the secp256k1 scalar field.
/// Supports homomorphic addition and scalar multiplication.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CLCiphertext {
    inner: CLCiphertextInner,
}

impl CLGroup {
    /// Creates a new CL group with the given security parameter.
    ///
    /// # Arguments
    ///
    /// * `security_bits` - Security parameter in bits. Recommended: 1600 for 128-bit security.
    ///
    /// # Note
    ///
    /// Group generation is deterministic given the seed. Using a fixed seed allows
    /// for verifiable group generation.
    pub fn new(security_bits: usize) -> Self {
        let seed = BigInt::from_str_radix(
            "314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848",
            10
        ).unwrap();

        Self::new_from_seed(security_bits, &seed)
    }

    /// Creates a new CL group with the given security parameter and seed.
    ///
    /// The seed determines the group parameters. Using the same seed produces
    /// the same group, allowing for verification.
    pub fn new_from_seed(security_bits: usize, seed: &BigInt) -> Self {
        let inner = CLGroupInner::new_from_setup(&security_bits, seed);
        Self { inner }
    }

    /// Creates a CL group with default security (1600-bit discriminant).
    pub fn default_security() -> Self {
        Self::new(DEFAULT_SECURITY_PARAMETER)
    }

    /// Generates a new key pair for this group.
    pub fn keygen(&self) -> CLKeyPair {
        let (sk_inner, pk_inner) = self.inner.keygen();
        CLKeyPair {
            sk: CLSecretKey { inner: sk_inner },
            pk: CLPublicKey { inner: pk_inner },
        }
    }

    /// Verifies that this group was generated correctly from the given seed.
    pub fn verify_setup(&self, seed: &BigInt) -> bool {
        self.inner.setup_verify(seed).is_ok()
    }

    /// Returns the inner CLGroup for advanced operations.
    pub fn inner(&self) -> &CLGroupInner {
        &self.inner
    }
}

impl CLSecretKey {
    /// Decrypts a ciphertext.
    ///
    /// Returns the plaintext as a scalar in the secp256k1 field.
    pub fn decrypt(&self, group: &CLGroup, ciphertext: &CLCiphertext) -> Scalar<Secp256k1> {
        cl::decrypt(&group.inner, &self.inner, &ciphertext.inner)
    }

    /// Decrypts a ciphertext and returns the result as a u64.
    ///
    /// # Panics
    ///
    /// Panics if the decrypted value doesn't fit in u64.
    pub fn decrypt_u64(&self, group: &CLGroup, ciphertext: &CLCiphertext) -> u64 {
        let scalar = self.decrypt(group, ciphertext);
        let bigint: BigInt = scalar.to_bigint();

        // Try to convert to u64
        bigint.to_str_radix(10)
            .parse::<u64>()
            .expect("decrypted value should fit in u64")
    }

    /// Returns the inner secret key for advanced operations.
    pub fn inner(&self) -> &CLSecretKeyInner {
        &self.inner
    }
}

impl CLPublicKey {
    /// Encrypts a scalar value.
    ///
    /// Returns the ciphertext and the randomness used (for proofs).
    pub fn encrypt(&self, group: &CLGroup, message: &Scalar<Secp256k1>) -> (CLCiphertext, CLSecretKey) {
        let (inner, r) = cl::encrypt(&group.inner, &self.inner, message);
        (CLCiphertext { inner }, CLSecretKey { inner: r })
    }

    /// Encrypts a u64 value.
    pub fn encrypt_u64(&self, group: &CLGroup, message: u64) -> CLCiphertext {
        let scalar = Scalar::<Secp256k1>::from(&BigInt::from(message));
        let (ct, _) = self.encrypt(group, &scalar);
        ct
    }

    /// Encrypts with predefined randomness (for deterministic encryption in protocols).
    pub fn encrypt_with_randomness(
        &self,
        group: &CLGroup,
        message: &Scalar<Secp256k1>,
        randomness: &CLSecretKey,
    ) -> CLCiphertext {
        let inner = cl::encrypt_predefined_randomness(&group.inner, &self.inner, message, &randomness.inner);
        CLCiphertext { inner }
    }

    /// Returns the inner public key for advanced operations.
    pub fn inner(&self) -> &CLPublicKeyInner {
        &self.inner
    }
}

impl CLCiphertext {
    /// Homomorphically adds two ciphertexts.
    ///
    /// If `self` encrypts `m1` and `other` encrypts `m2`, the result encrypts `m1 + m2`.
    pub fn add(&self, other: &CLCiphertext) -> CLCiphertext {
        let inner = cl::eval_sum(&self.inner, &other.inner);
        CLCiphertext { inner }
    }

    /// Homomorphically subtracts two ciphertexts.
    ///
    /// If `self` encrypts `m1` and `other` encrypts `m2`, the result encrypts `m1 - m2`.
    pub fn sub(&self, other: &CLCiphertext) -> CLCiphertext {
        // To subtract, we multiply the other ciphertext by -1 and add
        let neg_one = BigInt::from(-1i64);
        let neg_other = other.scalar_mul_bigint(&neg_one);
        self.add(&neg_other)
    }

    /// Homomorphically multiplies the ciphertext by a scalar.
    ///
    /// If `self` encrypts `m`, the result encrypts `scalar * m`.
    pub fn scalar_mul(&self, scalar: u64) -> CLCiphertext {
        let scalar_bigint = BigInt::from(scalar);
        self.scalar_mul_bigint(&scalar_bigint)
    }

    /// Homomorphically multiplies the ciphertext by a scalar (BigInt version).
    pub fn scalar_mul_bigint(&self, scalar: &BigInt) -> CLCiphertext {
        let inner = cl::eval_scal(&self.inner, scalar);
        CLCiphertext { inner }
    }

    /// Homomorphically negates the ciphertext.
    ///
    /// If `self` encrypts `m`, the result encrypts `-m`.
    pub fn neg(&self) -> CLCiphertext {
        let neg_one = BigInt::from(-1i64);
        self.scalar_mul_bigint(&neg_one)
    }

    /// Returns the inner ciphertext for advanced operations.
    pub fn inner(&self) -> &CLCiphertextInner {
        &self.inner
    }
}

/// Helper to create a scalar from a u64.
pub fn scalar_from_u64(value: u64) -> Scalar<Secp256k1> {
    Scalar::<Secp256k1>::from(&BigInt::from(value))
}

/// Helper to create a scalar from bytes.
pub fn scalar_from_bytes(bytes: &[u8]) -> Scalar<Secp256k1> {
    Scalar::<Secp256k1>::from(&BigInt::from_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (CLGroup, CLKeyPair) {
        let group = CLGroup::new(1600);
        let keypair = group.keygen();
        (group, keypair)
    }

    #[test]
    fn test_encrypt_decrypt() {
        let (group, keypair) = setup();

        let message = 42u64;
        let ct = keypair.pk.encrypt_u64(&group, message);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct);

        assert_eq!(decrypted, message);
    }

    #[test]
    fn test_homomorphic_addition() {
        let (group, keypair) = setup();

        let m1 = 100u64;
        let m2 = 200u64;

        let ct1 = keypair.pk.encrypt_u64(&group, m1);
        let ct2 = keypair.pk.encrypt_u64(&group, m2);

        let ct_sum = ct1.add(&ct2);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_sum);

        assert_eq!(decrypted, m1 + m2);
    }

    #[test]
    fn test_homomorphic_scalar_mul() {
        let (group, keypair) = setup();

        let message = 7u64;
        let scalar = 6u64;

        let ct = keypair.pk.encrypt_u64(&group, message);
        let ct_scaled = ct.scalar_mul(scalar);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_scaled);

        assert_eq!(decrypted, message * scalar);
    }

    #[test]
    fn test_linear_combination() {
        let (group, keypair) = setup();

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
    fn test_sum_many() {
        let (group, keypair) = setup();

        let values: Vec<u64> = (1..=10).collect();
        let expected_sum: u64 = values.iter().sum();

        let ciphertexts: Vec<CLCiphertext> = values
            .iter()
            .map(|&v| keypair.pk.encrypt_u64(&group, v))
            .collect();

        // Sum all ciphertexts
        let ct_sum = ciphertexts
            .iter()
            .skip(1)
            .fold(ciphertexts[0].clone(), |acc, ct| acc.add(ct));

        let decrypted = keypair.sk.decrypt_u64(&group, &ct_sum);

        assert_eq!(decrypted, expected_sum);
    }

    #[test]
    fn test_subtraction() {
        let (group, keypair) = setup();

        let m1 = 100u64;
        let m2 = 30u64;

        let ct1 = keypair.pk.encrypt_u64(&group, m1);
        let ct2 = keypair.pk.encrypt_u64(&group, m2);

        let ct_diff = ct1.sub(&ct2);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_diff);

        assert_eq!(decrypted, m1 - m2);
    }

    #[test]
    fn test_negation() {
        let (group, keypair) = setup();

        let message = 42u64;

        let ct = keypair.pk.encrypt_u64(&group, message);
        let ct_neg = ct.neg();

        // Adding original and negation should give 0
        let ct_sum = ct.add(&ct_neg);
        let decrypted = keypair.sk.decrypt_u64(&group, &ct_sum);

        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_group_verification() {
        let seed = BigInt::from_str_radix(
            "314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848",
            10
        ).unwrap();

        let group = CLGroup::new_from_seed(1600, &seed);
        assert!(group.verify_setup(&seed));
    }
}
