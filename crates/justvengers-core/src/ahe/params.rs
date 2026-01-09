//! BGV parameter sets.
//!
//! Parameters determine security level, noise budget, and performance.

/// Goldilocks prime: p = 2^64 - 2^32 + 1
/// This prime is used in many ZK proof systems and supports NTT.
/// For slot packing with N=8192: p-1 = 2^64 - 2^32 = 2^32(2^32 - 1)
/// Since 2^32 is divisible by 16384 = 2^14, we have p ≡ 1 (mod 16384) ✓
pub const GOLDILOCKS: u64 = 0xFFFFFFFF00000001; // 2^64 - 2^32 + 1 = 18446744069414584321

/// BGV encryption parameters.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BgvParams {
    /// Ring dimension (must be power of 2).
    /// The ring is R = Z[X]/(X^n + 1).
    pub n: usize,

    /// Log2 of ring dimension.
    pub log_n: u32,

    /// Ciphertext modulus q.
    /// Coefficients are in Z_q.
    pub q: u64,

    /// Plaintext modulus t.
    /// Messages are in Z_t.
    pub t: u64,

    /// Scaling factor Δ = ⌊q/t⌋.
    pub delta: u64,

    /// Standard deviation for error sampling.
    pub sigma: f64,

    /// Bound for uniform ternary distribution {-1, 0, 1}.
    pub ternary_bound: u64,

    /// Primitive 2n-th root of unity for NTT.
    /// Satisfies: omega^n ≡ -1 (mod q), omega^(2n) ≡ 1 (mod q).
    pub omega: u64,
}

impl BgvParams {
    /// Creates new parameters with validation.
    ///
    /// # Panics
    ///
    /// Panics if parameters are invalid.
    pub fn new(n: usize, q: u64, t: u64, sigma: f64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(n >= 64, "n must be at least 64");
        assert!(q > t, "q must be greater than t");
        assert!(t > 1, "t must be greater than 1");
        assert!(sigma > 0.0, "sigma must be positive");

        let log_n = n.trailing_zeros();
        let delta = q / t;

        // Try to find primitive 2n-th root of unity for NTT (optional)
        // NTT requires q ≡ 1 (mod 2n). If not satisfied, omega = 0 (NTT disabled)
        let omega = Self::find_primitive_root(n, q).unwrap_or(0);

        Self {
            n,
            log_n,
            q,
            t,
            delta,
            sigma,
            ternary_bound: 1,
            omega,
        }
    }

    /// Finds a primitive 2n-th root of unity modulo q.
    ///
    /// Returns omega such that omega^n ≡ -1 (mod q).
    /// Requires q ≡ 1 (mod 2n).
    fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
        let order = 2 * n as u64;
        if (q - 1) % order != 0 {
            return None;
        }

        let exp = (q - 1) / order;

        // Try small primes as potential generators
        for g in 2..1000u64 {
            let omega = Self::mod_pow(g, exp, q);

            // Verify: omega^n should be -1 (i.e., q-1)
            let omega_n = Self::mod_pow(omega, n as u64, q);
            if omega_n == q - 1 {
                return Some(omega);
            }
        }

        None
    }

    /// Modular exponentiation: base^exp mod modulus
    fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % modulus as u128) as u64;
            }
            exp >>= 1;
            base = ((base as u128 * base as u128) % modulus as u128) as u64;
        }

        result
    }

    /// Returns the maximum noise that can be tolerated before decryption fails.
    ///
    /// For correct decryption, noise must satisfy: |noise| < Δ/2 = q/(2t)
    pub fn noise_bound(&self) -> u64 {
        self.delta / 2
    }
}

/// Predefined parameter sets for different use cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSet {
    /// Toy parameters for testing (NOT SECURE).
    /// n=256, q≈2^30, t=2^16
    Toy,

    /// Small parameters for development (NOT SECURE).
    /// n=1024, q≈2^40, t=2^16
    Small,

    /// Medium parameters (~80-bit security estimate).
    /// n=2048, q≈2^54, t=2^16
    Medium,

    /// Large parameters (~128-bit security estimate).
    /// n=4096, q≈2^60, t=2^16
    Large,
}

impl ParamSet {
    /// Converts parameter set to concrete parameters.
    pub fn params(self) -> BgvParams {
        match self {
            ParamSet::Toy => BgvParams::new(
                256,
                1073738753,   // Prime ≡ 1 (mod 512) for NTT support
                65537,        // 2^16 + 1 (prime)
                3.2,
            ),
            ParamSet::Small => BgvParams::new(
                1024,
                1099511592961, // Prime ≡ 1 (mod 2048) for NTT support
                65537,
                3.2,
            ),
            ParamSet::Medium => BgvParams::new(
                2048,
                18014398509404161, // Prime ≡ 1 (mod 4096) for NTT support
                65537,
                3.2,
            ),
            ParamSet::Large => BgvParams::new(
                4096,
                1152921504606830593, // Prime ≡ 1 (mod 8192) for NTT support
                65537,
                3.2,
            ),
        }
    }
}

impl Default for BgvParams {
    fn default() -> Self {
        // Use Large (n=4096) for ~128-bit security
        // Small (n=1024) only provides ~40-60 bit security
        ParamSet::Large.params()
    }
}

/// RNS-based BGV parameters for large plaintext modulus (like Goldilocks).
///
/// When the plaintext modulus t is large (e.g., Goldilocks ≈ 2^64), the ciphertext
/// modulus q must be much larger to maintain noise budget. This requires RNS
/// representation where q = q_1 × q_2 × ... × q_k with each q_i fitting in 64 bits.
#[derive(Clone, Debug)]
pub struct RnsBgvParams {
    /// Ring dimension (must be power of 2).
    pub n: usize,

    /// Plaintext modulus t.
    pub t: u64,

    /// Number of RNS moduli for ciphertext modulus q.
    pub num_moduli: usize,

    /// Standard deviation for error sampling.
    pub sigma: f64,

    /// Whether slot packing is supported (t ≡ 1 mod 2n).
    pub supports_slots: bool,

    /// Number of slots (equals n when slot packing is supported).
    pub num_slots: usize,
}

impl RnsBgvParams {
    /// Creates RNS BGV parameters for the Goldilocks field.
    ///
    /// Uses N=8192 ring dimension with 8192 slots and 3 RNS moduli (~180 bit q).
    /// In IT-PAC, noise never exceeds 80 bits, so 3 moduli (~180 bits) suffices.
    pub fn goldilocks() -> Self {
        let n = 8192;
        let t = GOLDILOCKS;

        // Verify slot packing is supported
        let order = 2 * n as u64;
        let supports_slots = (t - 1) % order == 0;
        assert!(supports_slots, "Goldilocks must support slot packing with N=8192");

        Self {
            n,
            t,
            num_moduli: 3, // ~180 bit q; IT-PAC noise ≤ 80 bits
            sigma: 3.2,
            supports_slots,
            num_slots: n,
        }
    }

    /// Creates RNS BGV parameters for Goldilocks with reduced modulus.
    ///
    /// Uses only 2 RNS moduli (~120 bit q) for lower noise in rotations.
    /// This provides less noise budget but allows more key-switching operations.
    pub fn goldilocks_reduced() -> Self {
        let n = 8192;
        let t = GOLDILOCKS;

        let order = 2 * n as u64;
        let supports_slots = (t - 1) % order == 0;
        assert!(supports_slots, "Goldilocks must support slot packing with N=8192");

        Self {
            n,
            t,
            num_moduli: 2, // ~120 bit ciphertext modulus - less noise
            sigma: 3.2,
            supports_slots,
            num_slots: n,
        }
    }

    /// Creates RNS BGV parameters for Goldilocks with extended modulus (testing only).
    ///
    /// Uses 4 RNS moduli (~240 bit q) for tests requiring larger noise budget.
    /// Production IT-PAC uses `goldilocks()` with 3 moduli since noise ≤ 80 bits.
    pub fn goldilocks_test() -> Self {
        let n = 8192;
        let t = GOLDILOCKS;

        let order = 2 * n as u64;
        let supports_slots = (t - 1) % order == 0;
        assert!(supports_slots, "Goldilocks must support slot packing with N=8192");

        Self {
            n,
            t,
            num_moduli: 4, // ~240 bit q; for tests with higher noise operations
            sigma: 3.2,
            supports_slots,
            num_slots: n,
        }
    }

    /// Creates RNS BGV parameters with custom settings.
    pub fn new(n: usize, t: u64, num_moduli: usize, sigma: f64) -> Self {
        assert!(n.is_power_of_two(), "n must be power of 2");
        assert!(n >= 64, "n must be at least 64");
        assert!(t > 1, "t must be greater than 1");
        assert!(num_moduli > 0, "need at least one modulus");

        let order = 2 * n as u64;
        let supports_slots = (t - 1) % order == 0;
        let num_slots = if supports_slots { n } else { 1 };

        Self {
            n,
            t,
            num_moduli,
            sigma,
            supports_slots,
            num_slots,
        }
    }

    /// Returns whether this configuration supports slot packing.
    pub fn supports_slot_packing(&self) -> bool {
        self.supports_slots
    }

    /// Returns the number of plaintext slots per ciphertext.
    pub fn slots(&self) -> usize {
        self.num_slots
    }
}

#[cfg(test)]
mod param_tests {
    use super::*;

    #[test]
    fn test_param_sets_valid() {
        for param_set in [ParamSet::Toy, ParamSet::Small, ParamSet::Medium, ParamSet::Large] {
            let params = param_set.params();
            assert!(params.n.is_power_of_two());
            assert!(params.q > params.t);
            assert!(params.delta > 0);
        }
    }

    #[test]
    fn test_noise_bound() {
        let params = ParamSet::Small.params();
        assert!(params.noise_bound() > 0);
        assert!(params.noise_bound() < params.q / 2);
    }

    #[test]
    fn test_goldilocks_constant() {
        // Verify Goldilocks = 2^64 - 2^32 + 1
        let expected = (1u64 << 32).wrapping_neg().wrapping_add(1);
        assert_eq!(GOLDILOCKS, expected);

        // Verify it's the right value
        assert_eq!(GOLDILOCKS, 18446744069414584321);
    }

    #[test]
    fn test_goldilocks_slot_packing_support() {
        // Goldilocks should support slot packing with N=8192
        let order = 2 * 8192u64; // 16384
        let remainder = (GOLDILOCKS - 1) % order;
        assert_eq!(remainder, 0, "Goldilocks-1 must be divisible by 16384");
    }

    #[test]
    fn test_rns_bgv_params_goldilocks() {
        let params = RnsBgvParams::goldilocks();
        assert_eq!(params.n, 8192);
        assert_eq!(params.t, GOLDILOCKS);
        assert_eq!(params.num_moduli, 3);
        assert!(params.supports_slot_packing());
        assert_eq!(params.slots(), 8192);
    }
}
