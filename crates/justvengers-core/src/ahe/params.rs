//! BGV parameter sets.
//!
//! Parameters determine security level, noise budget, and performance.

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
        ParamSet::Small.params()
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
}
