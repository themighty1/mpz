//! Information-Theoretic Message Authentication Codes (IT-MAC).
//!
//! IT-MACs provide commitments in the VOLE-hybrid model, where:
//! - V holds a global key Δ ∈ F
//! - For value x, V samples local key k_x ∈ F and P receives m_x = k_x + x·Δ
//! - The commitment [x]_Δ = ⟨(x, m_x), k_x⟩_Δ is binding and supports linear operations
//!
//! Properties:
//! - Hiding: k_x and Δ are independent of the committed value x
//! - Binding: To forge, P must guess Δ (probability 1/|F|)
//! - Linear Homomorphism: Linear combinations computed without communication

use rand::Rng;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

/// Trait for fields that can be used with IT-MAC.
pub trait ItMacField:
    Copy + Clone + Default + PartialEq + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// The zero element.
    fn zero() -> Self;
    /// The one element.
    fn one() -> Self;
    /// Sample a uniform random element.
    fn random<R: Rng>(rng: &mut R) -> Self;
    /// Compute the additive inverse.
    fn neg(self) -> Self;
}

/// Global key held by the verifier.
///
/// This is the secret Δ used in all IT-MAC computations.
#[derive(Clone, Debug)]
pub struct GlobalKey<F: ItMacField> {
    delta: F,
}

impl<F: ItMacField> GlobalKey<F> {
    /// Generate a new random global key.
    pub fn generate<R: Rng>(rng: &mut R) -> Self {
        Self {
            delta: F::random(rng),
        }
    }

    /// Create from a specific delta value (for testing).
    pub fn from_delta(delta: F) -> Self {
        Self { delta }
    }

    /// Returns the delta value.
    pub fn delta(&self) -> F {
        self.delta
    }
}

/// Prover's share of an IT-MAC commitment.
///
/// Contains the committed value x and its MAC m_x = k_x + x·Δ.
#[derive(Clone, Debug)]
pub struct ProverShare<F: ItMacField> {
    /// The committed value.
    value: F,
    /// The MAC tag: m_x = k_x + x·Δ.
    mac: F,
}

impl<F: ItMacField> ProverShare<F> {
    /// Creates a new prover share.
    pub fn new(value: F, mac: F) -> Self {
        Self { value, mac }
    }

    /// Returns the committed value.
    pub fn value(&self) -> F {
        self.value
    }

    /// Returns the MAC tag.
    pub fn mac(&self) -> F {
        self.mac
    }
}

/// Verifier's share of an IT-MAC commitment.
///
/// Contains the local key k_x.
#[derive(Clone, Debug)]
pub struct VerifierShare<F: ItMacField> {
    /// The local key k_x.
    local_key: F,
}

impl<F: ItMacField> VerifierShare<F> {
    /// Creates a new verifier share.
    pub fn new(local_key: F) -> Self {
        Self { local_key }
    }

    /// Returns the local key.
    pub fn local_key(&self) -> F {
        self.local_key
    }
}

/// An IT-MAC commitment [x]_Δ.
///
/// This is the combined view of both parties' shares for an IT-MAC.
/// In practice, P and V each hold only their respective shares.
#[derive(Clone, Debug)]
pub struct ItMac<F: ItMacField> {
    /// Prover's share (x, m_x).
    prover: ProverShare<F>,
    /// Verifier's share (k_x).
    verifier: VerifierShare<F>,
}

impl<F: ItMacField> ItMac<F> {
    /// Creates a new IT-MAC for value x.
    ///
    /// V samples local key k_x, computes m_x = k_x + x·Δ.
    pub fn commit<R: Rng>(global_key: &GlobalKey<F>, value: F, rng: &mut R) -> Self {
        let local_key = F::random(rng);
        let mac = local_key + value * global_key.delta;

        Self {
            prover: ProverShare::new(value, mac),
            verifier: VerifierShare::new(local_key),
        }
    }

    /// Creates a commitment to a random value (VOLE correlation).
    ///
    /// This generates [u] where u is uniform random.
    pub fn random<R: Rng>(global_key: &GlobalKey<F>, rng: &mut R) -> Self {
        let value = F::random(rng);
        Self::commit(global_key, value, rng)
    }

    /// Creates a commitment to zero.
    ///
    /// For zero, both m_0 = 0 and k_0 = 0 since m = k + 0·Δ = k.
    pub fn commit_zero(_global_key: &GlobalKey<F>) -> Self {
        Self {
            prover: ProverShare::new(F::zero(), F::zero()),
            verifier: VerifierShare::new(F::zero()),
        }
    }

    /// Creates a commitment to a public constant c.
    ///
    /// For constant c, P sets m_c = 0 and V sets k_c = -c·Δ.
    pub fn commit_constant(global_key: &GlobalKey<F>, c: F) -> Self {
        let local_key = c.neg() * global_key.delta;
        Self {
            prover: ProverShare::new(c, F::zero()),
            verifier: VerifierShare::new(local_key),
        }
    }

    /// Returns the committed value.
    pub fn value(&self) -> F {
        self.prover.value
    }

    /// Returns the prover's share.
    pub fn prover_share(&self) -> &ProverShare<F> {
        &self.prover
    }

    /// Returns the verifier's share.
    pub fn verifier_share(&self) -> &VerifierShare<F> {
        &self.verifier
    }

    /// Verifies that the commitment is well-formed.
    ///
    /// Checks: m_x = k_x + x·Δ
    pub fn verify(&self, global_key: &GlobalKey<F>) -> bool {
        let expected_mac = self.verifier.local_key + self.prover.value * global_key.delta;
        self.prover.mac == expected_mac
    }

    /// Opens the commitment to the verifier.
    ///
    /// P sends (x, m_x), V accepts if m_x = k_x + x·Δ.
    pub fn open(&self, global_key: &GlobalKey<F>) -> Option<F> {
        if self.verify(global_key) {
            Some(self.prover.value)
        } else {
            None
        }
    }

    /// Computes [x + y] from [x] and [y].
    ///
    /// Linear homomorphism: no communication needed.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            prover: ProverShare::new(
                self.prover.value + other.prover.value,
                self.prover.mac + other.prover.mac,
            ),
            verifier: VerifierShare::new(self.verifier.local_key + other.verifier.local_key),
        }
    }

    /// Computes [x - y] from [x] and [y].
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            prover: ProverShare::new(
                self.prover.value - other.prover.value,
                self.prover.mac - other.prover.mac,
            ),
            verifier: VerifierShare::new(self.verifier.local_key - other.verifier.local_key),
        }
    }

    /// Computes [c·x] from [x] for public constant c.
    ///
    /// Scalar multiplication: no communication needed.
    pub fn scalar_mul(&self, c: F) -> Self {
        Self {
            prover: ProverShare::new(self.prover.value * c, self.prover.mac * c),
            verifier: VerifierShare::new(self.verifier.local_key * c),
        }
    }

    /// Computes [x + c] from [x] for public constant c.
    ///
    /// Constant addition: no communication needed.
    pub fn add_constant(&self, global_key: &GlobalKey<F>, c: F) -> Self {
        // m_{x+c} = m_x (unchanged)
        // k_{x+c} = k_x - c·Δ
        Self {
            prover: ProverShare::new(self.prover.value + c, self.prover.mac),
            verifier: VerifierShare::new(self.verifier.local_key - c * global_key.delta),
        }
    }

    /// Computes [c₀ + c₁x₁ + ... + cₙxₙ] from [x₁], ..., [xₙ].
    ///
    /// General linear combination: no communication needed.
    pub fn linear_combination(
        global_key: &GlobalKey<F>,
        constant: F,
        terms: &[(F, &Self)],
    ) -> Self {
        let mut result_value = constant;
        let mut result_mac = F::zero();
        let mut result_key = constant.neg() * global_key.delta;

        for (coeff, mac) in terms {
            result_value = result_value + *coeff * mac.prover.value;
            result_mac = result_mac + *coeff * mac.prover.mac;
            result_key = result_key + *coeff * mac.verifier.local_key;
        }

        Self {
            prover: ProverShare::new(result_value, result_mac),
            verifier: VerifierShare::new(result_key),
        }
    }
}

/// A batch of IT-MAC commitments.
///
/// Useful for committing to vectors of values.
#[derive(Clone, Debug)]
pub struct ItMacBatch<F: ItMacField> {
    macs: Vec<ItMac<F>>,
}

impl<F: ItMacField> ItMacBatch<F> {
    /// Creates a new batch from individual IT-MACs.
    pub fn new(macs: Vec<ItMac<F>>) -> Self {
        Self { macs }
    }

    /// Commits to a vector of values.
    pub fn commit<R: Rng>(global_key: &GlobalKey<F>, values: &[F], rng: &mut R) -> Self {
        let macs = values
            .iter()
            .map(|&v| ItMac::commit(global_key, v, rng))
            .collect();
        Self { macs }
    }

    /// Returns the number of commitments.
    pub fn len(&self) -> usize {
        self.macs.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.macs.is_empty()
    }

    /// Returns the commitment at index i.
    pub fn get(&self, i: usize) -> Option<&ItMac<F>> {
        self.macs.get(i)
    }

    /// Returns the underlying vector of IT-MACs.
    pub fn macs(&self) -> &[ItMac<F>] {
        &self.macs
    }

    /// Computes element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.macs.len(), other.macs.len());
        let macs = self
            .macs
            .iter()
            .zip(other.macs.iter())
            .map(|(a, b)| a.add(b))
            .collect();
        Self { macs }
    }

    /// Computes scalar multiplication on all elements.
    pub fn scalar_mul(&self, c: F) -> Self {
        let macs = self.macs.iter().map(|m| m.scalar_mul(c)).collect();
        Self { macs }
    }

    /// Computes inner product ⟨a, b⟩ where both are IT-MAC batches.
    ///
    /// Returns [⟨a, b⟩] using Σ aᵢ·bᵢ (requires multiplication protocol).
    /// Note: This only computes the linear combination assuming we have
    /// IT-MACs of the products.
    pub fn inner_product_with_public(_global_key: &GlobalKey<F>, _coeffs: &[F]) -> Self {
        unimplemented!("Inner product requires multiplication which needs LPZK")
    }

    /// Verifies all commitments in the batch.
    pub fn verify_all(&self, global_key: &GlobalKey<F>) -> bool {
        self.macs.iter().all(|m| m.verify(global_key))
    }
}

/// VOLE correlation: a pool of random IT-MACs.
///
/// In the VOLE-hybrid model, P and V can obtain random IT-MACs efficiently.
/// Each random IT-MAC [u] can be consumed once to commit to a chosen value x
/// by P sending (x - u).
#[derive(Clone, Debug)]
pub struct VolePool<F: ItMacField> {
    pool: Vec<ItMac<F>>,
    index: usize,
    _marker: PhantomData<F>,
}

impl<F: ItMacField> VolePool<F> {
    /// Generates a new VOLE pool with n random IT-MACs.
    pub fn generate<R: Rng>(global_key: &GlobalKey<F>, n: usize, rng: &mut R) -> Self {
        let pool = (0..n).map(|_| ItMac::random(global_key, rng)).collect();
        Self {
            pool,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Returns the number of remaining correlations.
    pub fn remaining(&self) -> usize {
        self.pool.len() - self.index
    }

    /// Consumes one VOLE correlation to commit to value x.
    ///
    /// P sends (x - u) to V, and parties compute [x] := [u] + (x - u).
    pub fn commit(&mut self, global_key: &GlobalKey<F>, value: F) -> Option<(ItMac<F>, F)> {
        if self.index >= self.pool.len() {
            return None;
        }

        let random_mac = &self.pool[self.index];
        self.index += 1;

        // P computes x - u (the "difference" to send)
        let diff = value - random_mac.value();

        // Both parties compute [x] = [u] + (x - u)
        let committed = random_mac.add_constant(global_key, diff);

        Some((committed, diff))
    }

    /// Gets a random IT-MAC without binding to a specific value.
    pub fn get_random(&mut self) -> Option<ItMac<F>> {
        if self.index >= self.pool.len() {
            return None;
        }
        let mac = self.pool[self.index].clone();
        self.index += 1;
        Some(mac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test field: integers mod p.
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    struct TestField(u64);

    const TEST_MODULUS: u64 = 1000000007; // Large prime

    impl TestField {
        fn new(v: u64) -> Self {
            Self(v % TEST_MODULUS)
        }
    }

    impl Add for TestField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self((self.0 + rhs.0) % TEST_MODULUS)
        }
    }

    impl Sub for TestField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self((self.0 + TEST_MODULUS - rhs.0) % TEST_MODULUS)
        }
    }

    impl Mul for TestField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self((self.0 as u128 * rhs.0 as u128 % TEST_MODULUS as u128) as u64)
        }
    }

    impl ItMacField for TestField {
        fn zero() -> Self {
            Self(0)
        }
        fn one() -> Self {
            Self(1)
        }
        fn random<R: Rng>(rng: &mut R) -> Self {
            Self(rng.random_range(0..TEST_MODULUS))
        }
        fn neg(self) -> Self {
            if self.0 == 0 {
                Self(0)
            } else {
                Self(TEST_MODULUS - self.0)
            }
        }
    }

    use mpz_core::{prg::Prg, Block};
    use rand::SeedableRng;

    #[test]
    fn test_commit_and_verify() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let value = TestField::new(42);
        let mac = ItMac::commit(&gk, value, &mut rng);

        assert!(mac.verify(&gk));
        assert_eq!(mac.value(), value);
    }

    #[test]
    fn test_open() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let value = TestField::new(12345);
        let mac = ItMac::commit(&gk, value, &mut rng);

        let opened = mac.open(&gk);
        assert_eq!(opened, Some(value));
    }

    #[test]
    fn test_linear_homomorphism_add() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let x = TestField::new(100);
        let y = TestField::new(200);
        let mac_x = ItMac::commit(&gk, x, &mut rng);
        let mac_y = ItMac::commit(&gk, y, &mut rng);

        let mac_sum = mac_x.add(&mac_y);
        assert!(mac_sum.verify(&gk));
        assert_eq!(mac_sum.value(), x + y);
    }

    #[test]
    fn test_linear_homomorphism_sub() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let x = TestField::new(300);
        let y = TestField::new(100);
        let mac_x = ItMac::commit(&gk, x, &mut rng);
        let mac_y = ItMac::commit(&gk, y, &mut rng);

        let mac_diff = mac_x.sub(&mac_y);
        assert!(mac_diff.verify(&gk));
        assert_eq!(mac_diff.value(), x - y);
    }

    #[test]
    fn test_scalar_multiplication() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let x = TestField::new(7);
        let c = TestField::new(5);
        let mac_x = ItMac::commit(&gk, x, &mut rng);

        let mac_cx = mac_x.scalar_mul(c);
        assert!(mac_cx.verify(&gk));
        assert_eq!(mac_cx.value(), c * x);
    }

    #[test]
    fn test_constant_addition() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let x = TestField::new(10);
        let c = TestField::new(33);
        let mac_x = ItMac::commit(&gk, x, &mut rng);

        let mac_xc = mac_x.add_constant(&gk, c);
        assert!(mac_xc.verify(&gk));
        assert_eq!(mac_xc.value(), x + c);
    }

    #[test]
    fn test_linear_combination() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let x1 = TestField::new(10);
        let x2 = TestField::new(20);
        let x3 = TestField::new(30);
        let c0 = TestField::new(5);
        let c1 = TestField::new(2);
        let c2 = TestField::new(3);
        let c3 = TestField::new(4);

        let mac1 = ItMac::commit(&gk, x1, &mut rng);
        let mac2 = ItMac::commit(&gk, x2, &mut rng);
        let mac3 = ItMac::commit(&gk, x3, &mut rng);

        // [c₀ + c₁x₁ + c₂x₂ + c₃x₃]
        let result = ItMac::linear_combination(&gk, c0, &[(c1, &mac1), (c2, &mac2), (c3, &mac3)]);

        assert!(result.verify(&gk));
        let expected = c0 + c1 * x1 + c2 * x2 + c3 * x3;
        assert_eq!(result.value(), expected);
    }

    #[test]
    fn test_commit_constant() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let c = TestField::new(42);
        let mac = ItMac::commit_constant(&gk, c);

        assert!(mac.verify(&gk));
        assert_eq!(mac.value(), c);
    }

    #[test]
    fn test_commit_zero() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let mac = ItMac::commit_zero(&gk);

        assert!(mac.verify(&gk));
        assert_eq!(mac.value(), TestField::zero());
    }

    #[test]
    fn test_vole_pool() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let mut pool = VolePool::generate(&gk, 10, &mut rng);
        assert_eq!(pool.remaining(), 10);

        let value = TestField::new(123);
        let (mac, _diff) = pool.commit(&gk, value).unwrap();

        assert!(mac.verify(&gk));
        assert_eq!(mac.value(), value);
        assert_eq!(pool.remaining(), 9);
    }

    #[test]
    fn test_batch_operations() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let gk = GlobalKey::<TestField>::generate(&mut rng);

        let values1: Vec<_> = (1..=5).map(|i| TestField::new(i * 10)).collect();
        let values2: Vec<_> = (1..=5).map(|i| TestField::new(i * 5)).collect();

        let batch1 = ItMacBatch::commit(&gk, &values1, &mut rng);
        let batch2 = ItMacBatch::commit(&gk, &values2, &mut rng);

        // Test addition
        let sum_batch = batch1.add(&batch2);
        assert!(sum_batch.verify_all(&gk));

        for (i, mac) in sum_batch.macs().iter().enumerate() {
            assert_eq!(mac.value(), values1[i] + values2[i]);
        }
    }

    // ==================== Additional comprehensive tests ====================

    mod comprehensive_tests {
        use super::*;

        #[test]
        fn test_multiple_linear_combinations() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            // Create several commitments
            let values: Vec<TestField> = (0..10).map(|i| TestField::new(i * 7 + 3)).collect();
            let macs: Vec<ItMac<TestField>> = values
                .iter()
                .map(|&v| ItMac::commit(&gk, v, &mut rng))
                .collect();

            // Linear combination with varying coefficients
            let coeffs: Vec<TestField> = (0..10).map(|i| TestField::new(i + 1)).collect();
            let constant = TestField::new(42);

            let terms: Vec<(TestField, &ItMac<TestField>)> =
                coeffs.iter().cloned().zip(macs.iter()).collect();

            let result = ItMac::linear_combination(&gk, constant, &terms);

            assert!(result.verify(&gk));

            // Compute expected value
            let mut expected = constant;
            for (c, v) in coeffs.iter().zip(values.iter()) {
                expected = expected + *c * *v;
            }
            assert_eq!(result.value(), expected);
        }

        #[test]
        fn test_negation() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let value = TestField::new(100);
            let mac = ItMac::commit(&gk, value, &mut rng);

            // Negate using scalar multiplication by -1
            let neg_one = TestField::new(TEST_MODULUS - 1);
            let mac_neg = mac.scalar_mul(neg_one);

            assert!(mac_neg.verify(&gk));
            assert_eq!(mac_neg.value(), value.neg());
        }

        #[test]
        fn test_self_subtraction_is_zero() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let value = TestField::new(12345);
            let mac = ItMac::commit(&gk, value, &mut rng);

            let mac_zero = mac.sub(&mac);

            assert!(mac_zero.verify(&gk));
            assert_eq!(mac_zero.value(), TestField::zero());
        }

        #[test]
        fn test_add_sub_inverse() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let x = TestField::new(100);
            let y = TestField::new(50);
            let mac_x = ItMac::commit(&gk, x, &mut rng);
            let mac_y = ItMac::commit(&gk, y, &mut rng);

            // (x + y) - y = x
            let sum = mac_x.add(&mac_y);
            let result = sum.sub(&mac_y);

            assert!(result.verify(&gk));
            assert_eq!(result.value(), x);
        }

        #[test]
        fn test_scalar_mul_distributive() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let x = TestField::new(10);
            let y = TestField::new(20);
            let c = TestField::new(5);

            let mac_x = ItMac::commit(&gk, x, &mut rng);
            let mac_y = ItMac::commit(&gk, y, &mut rng);

            // c * (x + y)
            let sum = mac_x.add(&mac_y);
            let left = sum.scalar_mul(c);

            // c*x + c*y
            let right = mac_x.scalar_mul(c).add(&mac_y.scalar_mul(c));

            assert!(left.verify(&gk));
            assert!(right.verify(&gk));
            assert_eq!(left.value(), right.value());
        }

        #[test]
        fn test_constant_add_preserves_mac_structure() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let x = TestField::new(100);
            let c = TestField::new(50);
            let mac_x = ItMac::commit(&gk, x, &mut rng);

            let mac_xc = mac_x.add_constant(&gk, c);

            // Value should be x + c
            assert_eq!(mac_xc.value(), x + c);

            // MAC tag should be unchanged (m_{x+c} = m_x)
            assert_eq!(mac_xc.prover_share().mac(), mac_x.prover_share().mac());

            // Should still verify
            assert!(mac_xc.verify(&gk));
        }

        #[test]
        fn test_forge_detection() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let value = TestField::new(42);
            let mac = ItMac::commit(&gk, value, &mut rng);

            // Try to forge by modifying the value
            let forged_prover = ProverShare::new(TestField::new(999), mac.prover_share().mac());
            let forged = ItMac {
                prover: forged_prover,
                verifier: mac.verifier_share().clone(),
            };

            // Should fail verification
            assert!(!forged.verify(&gk));
        }

        #[test]
        fn test_forge_mac_detection() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let value = TestField::new(42);
            let mac = ItMac::commit(&gk, value, &mut rng);

            // Try to forge by modifying the MAC tag
            let forged_prover =
                ProverShare::new(mac.prover_share().value(), TestField::new(12345));
            let forged = ItMac {
                prover: forged_prover,
                verifier: mac.verifier_share().clone(),
            };

            // Should fail verification
            assert!(!forged.verify(&gk));
        }

        #[test]
        fn test_vole_pool_exhaustion() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let mut pool = VolePool::generate(&gk, 3, &mut rng);

            // Use all correlations
            assert!(pool.commit(&gk, TestField::new(1)).is_some());
            assert!(pool.commit(&gk, TestField::new(2)).is_some());
            assert!(pool.commit(&gk, TestField::new(3)).is_some());

            // Pool should be exhausted
            assert_eq!(pool.remaining(), 0);
            assert!(pool.commit(&gk, TestField::new(4)).is_none());
        }

        #[test]
        fn test_vole_commit_preserves_randomness() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let mut pool = VolePool::generate(&gk, 5, &mut rng);

            // Commit to different values
            let values = [10u64, 20, 30, 40, 50];
            let mut committed = Vec::new();

            for &v in &values {
                let (mac, _) = pool.commit(&gk, TestField::new(v)).unwrap();
                committed.push(mac);
            }

            // All should verify and have correct values
            for (mac, &v) in committed.iter().zip(values.iter()) {
                assert!(mac.verify(&gk));
                assert_eq!(mac.value(), TestField::new(v));
            }
        }

        #[test]
        fn test_batch_scalar_mul() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let values: Vec<_> = (1..=5).map(|i| TestField::new(i * 10)).collect();
            let batch = ItMacBatch::commit(&gk, &values, &mut rng);

            let c = TestField::new(7);
            let scaled_batch = batch.scalar_mul(c);

            assert!(scaled_batch.verify_all(&gk));

            for (i, mac) in scaled_batch.macs().iter().enumerate() {
                assert_eq!(mac.value(), values[i] * c);
            }
        }

        #[test]
        fn test_random_mac_values_are_uniform() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            // Generate many random MACs
            let n = 100;
            let macs: Vec<ItMac<TestField>> =
                (0..n).map(|_| ItMac::random(&gk, &mut rng)).collect();

            // All should verify
            for mac in &macs {
                assert!(mac.verify(&gk));
            }

            // Values should be different (with high probability)
            let values: Vec<_> = macs.iter().map(|m| m.value()).collect();
            let unique: std::collections::HashSet<_> = values.iter().map(|v| v.0).collect();
            // With high probability, most values should be unique
            assert!(unique.len() > n / 2);
        }

        #[test]
        fn test_chained_operations() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            // Compute: 3 * (a + b) - 2 * c + 5
            let a = TestField::new(10);
            let b = TestField::new(20);
            let c = TestField::new(15);

            let mac_a = ItMac::commit(&gk, a, &mut rng);
            let mac_b = ItMac::commit(&gk, b, &mut rng);
            let mac_c = ItMac::commit(&gk, c, &mut rng);

            let result = mac_a
                .add(&mac_b)
                .scalar_mul(TestField::new(3))
                .sub(&mac_c.scalar_mul(TestField::new(2)))
                .add_constant(&gk, TestField::new(5));

            assert!(result.verify(&gk));

            let three = TestField::new(3);
            let two = TestField::new(2);
            let five = TestField::new(5);
            let expected = three * (a + b) - two * c + five;
            assert_eq!(result.value(), expected);
        }

        #[test]
        fn test_global_key_from_delta() {
            let delta = TestField::new(12345);
            let gk = GlobalKey::from_delta(delta);
            assert_eq!(gk.delta(), delta);
        }

        #[test]
        fn test_batch_empty() {
            let batch = ItMacBatch::<TestField>::new(vec![]);
            assert!(batch.is_empty());
            assert_eq!(batch.len(), 0);
        }

        #[test]
        fn test_batch_get() {
            let mut rng = Prg::from_seed(Block::ZERO);
            let gk = GlobalKey::<TestField>::generate(&mut rng);

            let values: Vec<_> = (1..=3).map(|i| TestField::new(i)).collect();
            let batch = ItMacBatch::commit(&gk, &values, &mut rng);

            assert!(batch.get(0).is_some());
            assert!(batch.get(2).is_some());
            assert!(batch.get(3).is_none());
        }
    }
}
