//! Residue Number System (RNS) for large modulus representation.
//!
//! RNS allows representing large integers using their residues modulo several
//! coprime moduli. This enables efficient arithmetic on numbers larger than
//! native integer types by performing operations independently on each residue.
//!
//! # Usage in BGV
//!
//! For BGV encryption with large plaintext modulus (like Goldilocks), we need
//! a ciphertext modulus q >> t. Using RNS, we represent q = q_1 × q_2 × ... × q_k
//! where each q_i fits in 64 bits and satisfies q_i ≡ 1 (mod 2N) for NTT support.
//!
//! # Mathematical Background
//!
//! By the Chinese Remainder Theorem (CRT), for coprime q_1, ..., q_k:
//!   Z_q ≅ Z_{q_1} × Z_{q_2} × ... × Z_{q_k}
//!
//! An element x ∈ Z_q is represented as (x mod q_1, x mod q_2, ..., x mod q_k).
//! Addition and multiplication work component-wise.

use super::ring::{BarrettReducer, RingPoly};

/// RNS moduli configuration.
///
/// Holds a set of coprime NTT-friendly primes for RNS representation.
#[derive(Clone, Debug)]
pub struct RnsParams {
    /// The individual prime moduli q_i.
    moduli: Vec<u64>,
    /// Barrett reducers for efficient modular reduction.
    reducers: Vec<BarrettReducer>,
    /// Primitive 2n-th roots of unity for each modulus (for NTT).
    roots: Vec<u64>,
    /// Ring dimension N.
    ring_dim: usize,
    /// Precomputed CRT reconstruction values.
    /// q_star[i] = (q / q_i) mod q_i
    q_stars: Vec<u64>,
    /// q_star_inv[i] = (q / q_i)^(-1) mod q_i
    q_star_invs: Vec<u64>,
}

/// Precomputed NTT-friendly primes for various ring dimensions.
/// Each prime q satisfies q ≡ 1 (mod 2N) for NTT support.
/// These are ~60-bit primes verified via Miller-Rabin primality test.
const NTT_PRIMES_8192: [u64; 8] = [
    // q ≡ 1 (mod 16384) primes, ~60 bits each
    1152921504606994433,
    1152921504607191041,
    1152921504607223809,
    1152921504607338497,
    1152921504607518721,
    1152921504608206849,
    1152921504608747521,
    1152921504609239041,
];

const NTT_PRIMES_4096: [u64; 8] = [
    // q ≡ 1 (mod 8192) primes
    1152921504606904321,
    1152921504606994433,
    1152921504607019009,
    1152921504607117313,
    1152921504607191041,
    1152921504607223809,
    1152921504607338497,
    1152921504607461377,
];

const NTT_PRIMES_1024: [u64; 8] = [
    // q ≡ 1 (mod 2048) primes, ~50 bits (smaller to avoid overflow in schoolbook)
    1125899906856961,
    1125899906949121,
    1125899906977793,
    1125899906990081,
    1125899907004417,
    1125899907063809,
    1125899907096577,
    1125899907100673,
];

const NTT_PRIMES_256: [u64; 4] = [
    // q ≡ 1 (mod 512) primes, ~50 bits
    1125899906844161,
    1125899906849281,
    1125899906856961,
    1125899906859521,
];

const NTT_PRIMES_64: [u64; 4] = [
    // q ≡ 1 (mod 128) primes, ~50 bits
    1125899906843009,
    1125899906844161,
    1125899906845057,
    1125899906849281,
];

impl RnsParams {
    /// Creates RNS parameters for the given ring dimension.
    ///
    /// Uses precomputed NTT-friendly primes for efficiency.
    ///
    /// # Arguments
    /// * `ring_dim` - Ring dimension N (must be power of 2)
    /// * `num_moduli` - Number of prime moduli to use
    /// * `_min_bit_size` - Ignored, uses precomputed primes
    pub fn new(ring_dim: usize, num_moduli: usize, _min_bit_size: u32) -> Self {
        assert!(ring_dim.is_power_of_two(), "ring_dim must be power of 2");
        assert!(num_moduli > 0, "need at least one modulus");

        // Select precomputed primes based on ring dimension
        let available_primes: &[u64] = match ring_dim {
            8192 => &NTT_PRIMES_8192,
            4096 => &NTT_PRIMES_4096,
            1024 => &NTT_PRIMES_1024,
            256 => &NTT_PRIMES_256,
            64 => &NTT_PRIMES_64,
            _ => {
                // For other dimensions, compute primes (slower)
                return Self::new_computed(ring_dim, num_moduli);
            }
        };

        assert!(
            num_moduli <= available_primes.len(),
            "requested {} moduli but only {} available for ring_dim {}",
            num_moduli,
            available_primes.len(),
            ring_dim
        );

        let moduli: Vec<u64> = available_primes[..num_moduli].to_vec();

        // Verify primes are NTT-friendly (debug check)
        let order = 2 * ring_dim as u64;
        for &q in &moduli {
            debug_assert_eq!(
                (q - 1) % order,
                0,
                "prime {} not NTT-friendly for N={}",
                q,
                ring_dim
            );
        }

        // Build reducers
        let reducers: Vec<_> = moduli.iter().map(|&q| BarrettReducer::new(q)).collect();

        // Find primitive 2N-th roots of unity for each modulus
        let roots: Vec<_> = moduli
            .iter()
            .map(|&q| Self::find_primitive_root(ring_dim, q).expect("prime should have root"))
            .collect();

        let q_stars = vec![0; num_moduli];
        let q_star_invs = vec![0; num_moduli];

        Self {
            moduli,
            reducers,
            roots,
            ring_dim,
            q_stars,
            q_star_invs,
        }
    }

    /// Creates RNS parameters by computing primes (slower, for non-standard dimensions).
    fn new_computed(ring_dim: usize, num_moduli: usize) -> Self {
        let order = 2 * ring_dim as u64;
        let mut moduli = Vec::with_capacity(num_moduli);

        // Start from a reasonable point
        let mut candidate = (1u64 << 50) + order - ((1u64 << 50) % order);
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 100000;

        while moduli.len() < num_moduli && attempts < MAX_ATTEMPTS {
            if Self::is_prime(candidate) && !moduli.contains(&candidate) {
                moduli.push(candidate);
            }
            candidate += order;
            attempts += 1;
        }

        if moduli.len() < num_moduli {
            panic!(
                "Could not find {} NTT-friendly primes for ring_dim {}",
                num_moduli, ring_dim
            );
        }

        let reducers: Vec<_> = moduli.iter().map(|&q| BarrettReducer::new(q)).collect();
        let roots: Vec<_> = moduli
            .iter()
            .map(|&q| Self::find_primitive_root(ring_dim, q).expect("prime should have root"))
            .collect();

        Self {
            moduli,
            reducers,
            roots,
            ring_dim,
            q_stars: vec![0; num_moduli],
            q_star_invs: vec![0; num_moduli],
        }
    }

    /// Creates RNS parameters from explicit moduli.
    ///
    /// Use this when you have specific primes you want to use.
    pub fn from_moduli(ring_dim: usize, moduli: Vec<u64>) -> Self {
        assert!(ring_dim.is_power_of_two());
        let order = 2 * ring_dim as u64;

        // Verify all moduli are NTT-friendly
        for &q in &moduli {
            assert!(
                (q - 1) % order == 0,
                "modulus {} is not NTT-friendly (need q ≡ 1 mod {})",
                q,
                order
            );
        }

        let reducers: Vec<_> = moduli.iter().map(|&q| BarrettReducer::new(q)).collect();
        let roots: Vec<_> = moduli
            .iter()
            .map(|&q| Self::find_primitive_root(ring_dim, q).expect("prime should have root"))
            .collect();

        let num_moduli = moduli.len();
        Self {
            moduli,
            reducers,
            roots,
            ring_dim,
            q_stars: vec![0; num_moduli],
            q_star_invs: vec![0; num_moduli],
        }
    }

    /// Returns the number of moduli.
    pub fn num_moduli(&self) -> usize {
        self.moduli.len()
    }

    /// Returns the ring dimension.
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Returns the moduli.
    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }

    /// Returns the Barrett reducers.
    pub fn reducers(&self) -> &[BarrettReducer] {
        &self.reducers
    }

    /// Returns the primitive roots.
    pub fn roots(&self) -> &[u64] {
        &self.roots
    }

    /// Simple primality test (Miller-Rabin with small bases).
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        // Write n-1 = 2^r * d
        let mut d = n - 1;
        let mut r = 0;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Test with small bases (sufficient for 64-bit)
        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        'outer: for &a in &witnesses {
            if a >= n {
                continue;
            }

            let mut x = Self::mod_pow(a, d, n);
            if x == 1 || x == n - 1 {
                continue;
            }

            for _ in 0..r - 1 {
                x = Self::mod_mul(x, x, n);
                if x == n - 1 {
                    continue 'outer;
                }
            }

            return false;
        }

        true
    }

    /// Finds primitive 2N-th root of unity modulo q.
    fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
        let order = 2 * n as u64;
        if (q - 1) % order != 0 {
            return None;
        }

        let exp = (q - 1) / order;

        for g in 2..1000u64 {
            let root = Self::mod_pow(g, exp, q);
            let root_n = Self::mod_pow(root, n as u64, q);
            if root_n == q - 1 {
                return Some(root);
            }
        }

        None
    }

    #[inline]
    fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
        ((a as u128 * b as u128) % m as u128) as u64
    }

    fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
        let mut result = 1u64;
        base %= m;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mod_mul(result, base, m);
            }
            exp >>= 1;
            base = Self::mod_mul(base, base, m);
        }
        result
    }
}

/// A polynomial in RNS representation.
///
/// Coefficients are stored as residues modulo each RNS modulus.
/// For a polynomial with n coefficients and k moduli, we store
/// a k × n matrix where entry [i][j] is coefficient j mod modulus i.
#[derive(Clone, Debug)]
pub struct RnsPoly {
    /// Coefficients in RNS form: residues[i] contains all coefficients mod moduli[i].
    residues: Vec<Vec<u64>>,
    /// Reference to RNS parameters.
    params: RnsParams,
}

impl RnsPoly {
    /// Creates a zero polynomial.
    pub fn zero(params: &RnsParams) -> Self {
        let residues = vec![vec![0; params.ring_dim]; params.num_moduli()];
        Self {
            residues,
            params: params.clone(),
        }
    }

    /// Creates a polynomial from a single small value (fits in u64).
    ///
    /// The value is placed in coefficient 0 (constant term).
    pub fn from_scalar(value: u64, params: &RnsParams) -> Self {
        let mut poly = Self::zero(params);
        for (i, &q) in params.moduli().iter().enumerate() {
            poly.residues[i][0] = value % q;
        }
        poly
    }

    /// Creates a polynomial from coefficient values (each must fit in u64).
    ///
    /// Each coefficient is reduced modulo each RNS modulus.
    pub fn from_coeffs(coeffs: &[u64], params: &RnsParams) -> Self {
        let n = params.ring_dim();
        let mut residues = vec![vec![0; n]; params.num_moduli()];

        for (i, &q) in params.moduli().iter().enumerate() {
            for (j, &c) in coeffs.iter().take(n).enumerate() {
                residues[i][j] = c % q;
            }
        }

        Self {
            residues,
            params: params.clone(),
        }
    }

    /// Returns the residues.
    pub fn residues(&self) -> &[Vec<u64>] {
        &self.residues
    }

    /// Returns mutable residues.
    pub fn residues_mut(&mut self) -> &mut [Vec<u64>] {
        &mut self.residues
    }

    /// Returns the RNS parameters.
    pub fn params(&self) -> &RnsParams {
        &self.params
    }

    /// Adds two RNS polynomials component-wise.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.params.num_moduli(), other.params.num_moduli());

        let mut result = Self::zero(&self.params);
        for i in 0..self.params.num_moduli() {
            let q = self.params.moduli()[i];
            for j in 0..self.params.ring_dim() {
                let sum = self.residues[i][j] + other.residues[i][j];
                result.residues[i][j] = if sum >= q { sum - q } else { sum };
            }
        }
        result
    }

    /// Subtracts two RNS polynomials component-wise.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.params.num_moduli(), other.params.num_moduli());

        let mut result = Self::zero(&self.params);
        for i in 0..self.params.num_moduli() {
            let q = self.params.moduli()[i];
            for j in 0..self.params.ring_dim() {
                let a = self.residues[i][j];
                let b = other.residues[i][j];
                result.residues[i][j] = if a >= b { a - b } else { q - (b - a) };
            }
        }
        result
    }

    /// Negates an RNS polynomial.
    pub fn neg(&self) -> Self {
        let mut result = Self::zero(&self.params);
        for i in 0..self.params.num_moduli() {
            let q = self.params.moduli()[i];
            for j in 0..self.params.ring_dim() {
                let c = self.residues[i][j];
                result.residues[i][j] = if c == 0 { 0 } else { q - c };
            }
        }
        result
    }

    /// Multiplies by a scalar (must fit in u64).
    pub fn scalar_mul(&self, scalar: u64) -> Self {
        let mut result = Self::zero(&self.params);
        for i in 0..self.params.num_moduli() {
            let reducer = &self.params.reducers()[i];
            let s = scalar % self.params.moduli()[i];
            for j in 0..self.params.ring_dim() {
                result.residues[i][j] =
                    reducer.reduce((self.residues[i][j] as u128) * (s as u128));
            }
        }
        result
    }

    /// Multiplies two RNS polynomials in the ring R_q = Z_q[X]/(X^n + 1).
    ///
    /// Uses NTT for each RNS component independently.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.params.num_moduli(), other.params.num_moduli());

        let mut result = Self::zero(&self.params);

        for i in 0..self.params.num_moduli() {
            let q = self.params.moduli()[i];
            let omega = self.params.roots()[i];

            // Create RingPoly for this component and multiply
            let a = RingPoly::new_with_omega(self.residues[i].clone(), q, omega);
            let b = RingPoly::new_with_omega(other.residues[i].clone(), q, omega);
            let c = a.mul(&b);

            result.residues[i] = c.coeffs().to_vec();
        }

        result
    }

    /// Converts a single-modulus RingPoly to RNS representation.
    ///
    /// The RingPoly coefficients are reduced modulo each RNS modulus.
    pub fn from_ring_poly(poly: &RingPoly, params: &RnsParams) -> Self {
        Self::from_coeffs(poly.coeffs(), params)
    }
}

/// Predefined RNS parameter sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnsParamSet {
    /// Small set for testing (2 primes, ~120 bit modulus).
    Test,
    /// Medium set (~180 bit modulus, 3 primes).
    Medium,
    /// Goldilocks (~180 bit modulus, 3 primes). IT-PAC noise ≤ 80 bits.
    Goldilocks,
}

impl RnsParamSet {
    /// Creates RNS parameters for the given set and ring dimension.
    pub fn params(self, ring_dim: usize) -> RnsParams {
        match self {
            RnsParamSet::Test => RnsParams::new(ring_dim, 2, 58),
            RnsParamSet::Medium => RnsParams::new(ring_dim, 3, 59),
            RnsParamSet::Goldilocks => RnsParams::new(ring_dim, 3, 60),
        }
    }
}

#[cfg(test)]
mod rns_tests {
    use super::*;

    #[test]
    fn test_precomputed_primes_valid() {
        // Verify all precomputed primes are valid
        for &q in &NTT_PRIMES_8192 {
            assert!(RnsParams::is_prime(q), "{} is not prime", q);
            assert_eq!((q - 1) % 16384, 0, "{} not NTT-friendly for N=8192", q);
        }
        for &q in &NTT_PRIMES_4096 {
            assert!(RnsParams::is_prime(q), "{} is not prime", q);
            assert_eq!((q - 1) % 8192, 0, "{} not NTT-friendly for N=4096", q);
        }
        for &q in &NTT_PRIMES_1024 {
            assert!(RnsParams::is_prime(q), "{} is not prime", q);
            assert_eq!((q - 1) % 2048, 0, "{} not NTT-friendly for N=1024", q);
        }
        for &q in &NTT_PRIMES_64 {
            assert!(RnsParams::is_prime(q), "{} is not prime", q);
            assert_eq!((q - 1) % 128, 0, "{} not NTT-friendly for N=64", q);
        }
    }

    #[test]
    fn test_rns_params_creation() {
        let params = RnsParams::new(1024, 2, 58);
        assert_eq!(params.num_moduli(), 2);
        assert_eq!(params.ring_dim(), 1024);

        // Verify moduli are NTT-friendly
        let order = 2 * 1024u64;
        for &q in params.moduli() {
            assert_eq!((q - 1) % order, 0);
        }
    }

    #[test]
    fn test_rns_poly_add() {
        let params = RnsParams::new(64, 2, 50);

        let a = RnsPoly::from_coeffs(&[1, 2, 3, 4], &params);
        let b = RnsPoly::from_coeffs(&[10, 20, 30, 40], &params);
        let c = a.add(&b);

        // Check first modulus
        assert_eq!(c.residues()[0][0], 11);
        assert_eq!(c.residues()[0][1], 22);
        assert_eq!(c.residues()[0][2], 33);
        assert_eq!(c.residues()[0][3], 44);
    }

    #[test]
    fn test_rns_poly_mul_simple() {
        let params = RnsParams::new(64, 2, 50);

        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let a = RnsPoly::from_coeffs(&[1, 1], &params);
        let c = a.mul(&a);

        // Both RNS components should have same result
        for residue in c.residues() {
            assert_eq!(residue[0], 1);
            assert_eq!(residue[1], 2);
            assert_eq!(residue[2], 1);
        }
    }

    #[test]
    fn test_rns_scalar_mul() {
        let params = RnsParams::new(64, 2, 50);

        let a = RnsPoly::from_coeffs(&[1, 2, 3], &params);
        let b = a.scalar_mul(5);

        for residue in b.residues() {
            assert_eq!(residue[0], 5);
            assert_eq!(residue[1], 10);
            assert_eq!(residue[2], 15);
        }
    }

    #[test]
    fn test_goldilocks_params() {
        // Ring dimension 8192 for Goldilocks slot packing
        let params = RnsParamSet::Goldilocks.params(8192);

        assert_eq!(params.num_moduli(), 3);
        assert_eq!(params.ring_dim(), 8192);

        // Verify NTT-friendliness
        let order = 2 * 8192u64;
        for &q in params.moduli() {
            assert_eq!((q - 1) % order, 0, "modulus {} not NTT-friendly", q);
        }

        // Total modulus should be ~180 bits (3 * 60-bit primes)
        let total_bits: u32 = params.moduli().iter().map(|&q| 64 - q.leading_zeros()).sum();
        assert!(total_bits >= 170, "total bits {} too small", total_bits);
    }
}
