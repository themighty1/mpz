//! Sampling distributions for BGV encryption.
//!
//! Provides discrete Gaussian and ternary ({-1, 0, 1}) distributions
//! used for error sampling in BGV.

use rand::Rng;

use super::params::BgvParams;
use super::ring::RingPoly;

/// Discrete Gaussian sampler with given standard deviation.
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    sigma: f64,
    /// Precomputed CDF table for efficient sampling.
    cdf_table: Vec<f64>,
    /// Range of values in the table: [-bound, bound].
    bound: i64,
}

impl DiscreteGaussian {
    /// Creates a new discrete Gaussian sampler with given standard deviation.
    ///
    /// The sampler will produce integers in [-6σ, 6σ] with probability
    /// proportional to exp(-x²/(2σ²)).
    pub fn new(sigma: f64) -> Self {
        let bound = (6.0 * sigma).ceil() as i64;
        let mut cdf_table = Vec::with_capacity((2 * bound + 1) as usize);

        // Compute unnormalized probabilities
        let mut total = 0.0;
        for x in -bound..=bound {
            let prob = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
            total += prob;
            cdf_table.push(total);
        }

        // Normalize to [0, 1]
        for p in &mut cdf_table {
            *p /= total;
        }

        Self {
            sigma,
            cdf_table,
            bound,
        }
    }

    /// Samples a single integer from the discrete Gaussian distribution.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> i64 {
        let u: f64 = rng.random();

        // Binary search in CDF table
        let idx = match self.cdf_table.binary_search_by(|p| {
            p.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i,
        };

        -self.bound + idx as i64
    }

    /// Samples a ring polynomial with coefficients from discrete Gaussian.
    pub fn sample_poly<R: Rng>(&self, params: &BgvParams, rng: &mut R) -> RingPoly {
        let coeffs: Vec<u64> = (0..params.n)
            .map(|_| {
                let s = self.sample(rng);
                if s >= 0 {
                    s as u64
                } else {
                    params.q - ((-s) as u64)
                }
            })
            .collect();

        RingPoly::new(coeffs, params.q)
    }

    /// Returns the standard deviation.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

/// Samples a polynomial with ternary coefficients {-1, 0, 1}.
///
/// Each coefficient is independently sampled with equal probability.
pub(crate) fn sample_ternary<R: Rng>(params: &BgvParams, rng: &mut R) -> RingPoly {
    let coeffs: Vec<u64> = (0..params.n)
        .map(|_| {
            let r: u32 = rng.random_range(0..3);
            match r {
                0 => params.q - 1, // -1 mod q
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            }
        })
        .collect();

    RingPoly::new(coeffs, params.q)
}

/// Samples a polynomial with uniformly random coefficients in [0, q).
pub(crate) fn sample_uniform<R: Rng>(params: &BgvParams, rng: &mut R) -> RingPoly {
    let coeffs: Vec<u64> = (0..params.n)
        .map(|_| rng.random_range(0..params.q))
        .collect();

    RingPoly::new(coeffs, params.q)
}

/// Samples a polynomial with coefficients in [0, bound).
#[allow(dead_code)]
pub(crate) fn sample_bounded<R: Rng>(params: &BgvParams, bound: u64, rng: &mut R) -> RingPoly {
    let coeffs: Vec<u64> = (0..params.n)
        .map(|_| rng.random_range(0..bound))
        .collect();

    RingPoly::new(coeffs, params.q)
}

#[cfg(test)]
mod sample_tests {
    use super::*;
    use crate::ahe::params::ParamSet;
    use mpz_core::{Block, prg::Prg};
    use rand::SeedableRng;

    #[test]
    fn test_discrete_gaussian_basic() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let dg = DiscreteGaussian::new(3.2);

        // Sample many values and check they're in expected range
        let bound = (6.0f64 * 3.2f64).ceil() as i64;
        for _ in 0..1000 {
            let s = dg.sample(&mut rng);
            assert!(
                s >= -bound && s <= bound,
                "sample {} out of range [-{}, {}]",
                s,
                bound,
                bound
            );
        }
    }

    #[test]
    fn test_discrete_gaussian_mean_near_zero() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let dg = DiscreteGaussian::new(3.2);

        let samples: Vec<i64> = (0..10000).map(|_| dg.sample(&mut rng)).collect();
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;

        // Mean should be close to 0
        assert!(
            mean.abs() < 0.5,
            "mean {} too far from 0",
            mean
        );
    }

    #[test]
    fn test_discrete_gaussian_poly() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();
        let dg = DiscreteGaussian::new(params.sigma);

        let poly = dg.sample_poly(&params, &mut rng);

        assert_eq!(poly.dimension(), params.n);
        // All coefficients should be valid (< q)
        assert!(poly.coeffs().iter().all(|&c| c < params.q));
    }

    #[test]
    fn test_sample_ternary() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();

        let poly = sample_ternary(&params, &mut rng);

        // All coefficients should be in {0, 1, q-1}
        for &c in poly.coeffs() {
            assert!(
                c == 0 || c == 1 || c == params.q - 1,
                "invalid ternary coefficient: {}",
                c
            );
        }
    }

    #[test]
    fn test_sample_uniform() {
        let mut rng = Prg::from_seed(Block::ZERO);
        let params = ParamSet::Toy.params();

        let poly = sample_uniform(&params, &mut rng);

        assert_eq!(poly.dimension(), params.n);
        assert!(poly.coeffs().iter().all(|&c| c < params.q));
    }
}
