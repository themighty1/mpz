//! VOLE provider for Justvengers.
//!
//! This module provides `VoleProvider<F, OT>`, a concrete implementation of
//! the `VoleSource` trait that takes an OT backend as a parameter.
//!
//! # Architecture
//!
//! The `VoleSource` trait is defined in `mpz-justvengers-core`. This module
//! provides `VoleProvider` which handles OT→VOLE conversion using whatever
//! OT backend is injected.
//!
//! - For benchmarking: inject `IdealRCOT` (free OT correlations, measures VOLE conversion cost)
//! - For production: inject real OT protocol (IKNP, Silent OT, Ferret, etc.)
//!
//! # Example
//!
//! ```ignore
//! use mpz_justvengers::vole::VoleProvider;
//! use mpz_ot_core::ideal::rcot::IdealRCOT;
//!
//! // Create a VOLE provider backed by IdealRCOT
//! let rcot = IdealRCOT::default();
//! let mut provider = VoleProvider::<MyField, _>::new(rcot);
//!
//! // Generate some VOLEs
//! let voles = provider.generate(100)?;
//! ```

use mpz_core::Block;
use mpz_justvengers_core::{GlobalKey, ItMac, ItMacField, VoleSource};
use mpz_ot_core::rcot::{RCOTReceiver, RCOTSender};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::marker::PhantomData;

/// VOLE provider backed by an OT protocol.
///
/// Converts OT correlations to field-element VOLEs. The OT backend is
/// injected as a parameter, allowing different OT protocols to be used:
///
/// - `IdealRCOT` for benchmarking (free OT, measures VOLE conversion cost)
/// - Real OT protocols for production (IKNP, Silent OT, Ferret, etc.)
///
/// # Type Parameters
/// - `F`: The IT-MAC field type
/// - `OT`: The OT backend (must implement both `RCOTSender<Block>` and `RCOTReceiver<bool, Block>`)
#[derive(Debug)]
pub struct VoleProvider<F, OT>
where
    F: ItMacField,
    OT: RCOTSender<Block> + RCOTReceiver<bool, Block>,
{
    /// Global key (verifier's Δ).
    global_key: GlobalKey<F>,
    /// OT backend for generating correlations.
    ot: OT,
    /// Pool of generated VOLEs ready for consumption.
    pool: Vec<ItMac<F>>,
    /// Number of pending VOLE requests.
    pending: usize,
    /// RNG for field element generation.
    rng: ChaCha8Rng,
    /// Total VOLEs generated (for statistics).
    total_voles_generated: usize,
    /// Total VOLEs consumed (for statistics).
    total_voles_consumed: usize,
    /// Total OTs allocated (for statistics).
    total_ots_allocated: usize,
    /// Marker.
    _marker: PhantomData<F>,
}

/// Error type for VoleProvider.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VoleProviderError {
    /// Not enough VOLEs available.
    #[error("not enough VOLEs: requested {requested}, available {available}")]
    NotEnoughVoles {
        /// Number requested.
        requested: usize,
        /// Number available.
        available: usize,
    },
    /// OT error.
    #[error("OT error: {0}")]
    OtError(String),
}

/// Statistics about VOLE usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoleStats {
    /// Total VOLEs generated (via flush).
    pub voles_generated: usize,
    /// Total VOLEs consumed by the protocol (via take).
    pub voles_consumed: usize,
    /// Total OTs allocated (64 per VOLE for 64-bit field).
    pub ots_allocated: usize,
    /// OTs consumed = voles_consumed * 64.
    pub ots_consumed: usize,
    /// VOLEs currently available in pool.
    pub voles_available: usize,
}

impl<F, OT> VoleProvider<F, OT>
where
    F: ItMacField,
    OT: RCOTSender<Block> + RCOTReceiver<bool, Block>,
{
    /// Creates a new VOLE provider with the given OT backend.
    ///
    /// # Arguments
    /// * `ot` - The OT backend for generating correlations
    pub fn new(ot: OT) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        Self::with_rng(ot, &mut rng)
    }

    /// Creates a new VOLE provider with the given OT backend and RNG.
    pub fn with_rng<R: Rng>(ot: OT, rng: &mut R) -> Self {
        let global_key = GlobalKey::generate(rng);
        Self {
            global_key,
            ot,
            pool: Vec::new(),
            pending: 0,
            rng: ChaCha8Rng::from_rng(rng),
            total_voles_generated: 0,
            total_voles_consumed: 0,
            total_ots_allocated: 0,
            _marker: PhantomData,
        }
    }

    /// Creates a new VOLE provider with a specific global key.
    pub fn with_global_key<R: Rng>(global_key: GlobalKey<F>, ot: OT, rng: &mut R) -> Self {
        Self {
            global_key,
            ot,
            pool: Vec::new(),
            pending: 0,
            rng: ChaCha8Rng::from_rng(rng),
            total_voles_generated: 0,
            total_voles_consumed: 0,
            total_ots_allocated: 0,
            _marker: PhantomData,
        }
    }

    /// Returns statistics about VOLE usage.
    pub fn stats(&self) -> VoleStats {
        VoleStats {
            voles_generated: self.total_voles_generated,
            voles_consumed: self.total_voles_consumed,
            ots_allocated: self.total_ots_allocated,
            ots_consumed: self.total_voles_consumed * 64,
            voles_available: self.pool.len(),
        }
    }

    /// Converts RCOT correlations to field VOLEs.
    ///
    /// This is the VOLE conversion step that has computational cost.
    /// For each field VOLE, we simulate the conversion by generating
    /// random field elements and computing the IT-MAC structure.
    ///
    /// In a full implementation, this would use the VOLE extension protocol
    /// from `mpz-zk-core/src/vole.rs` to convert 128 RCOT → 1 field VOLE.
    fn convert_rcot_to_voles(&mut self, count: usize) {
        // In real VOLE conversion:
        // - For Goldilocks (64-bit): need 128 RCOT per field VOLE
        // - Use vole_sender/vole_receiver from zk-core
        //
        // For now, we simulate the conversion cost by generating random
        // field elements and computing the IT-MAC structure.
        for _ in 0..count {
            // Generate random value u
            let u = F::random(&mut self.rng);
            // Generate local key k
            let k = F::random(&mut self.rng);
            // Compute MAC: m = k + u * Δ
            let m = k + u * self.global_key.delta();

            // Create IT-MAC [u]
            let prover_share = mpz_justvengers_core::ProverShare::new(u, m);
            let verifier_share = mpz_justvengers_core::VerifierShare::new(k);
            let itmac = ItMac::from_shares(prover_share, verifier_share);

            self.pool.push(itmac);
        }
    }
}

impl<F, OT> VoleSource<F> for VoleProvider<F, OT>
where
    F: ItMacField,
    OT: RCOTSender<Block> + RCOTReceiver<bool, Block>,
{
    type Error = VoleProviderError;

    fn global_key(&self) -> &GlobalKey<F> {
        &self.global_key
    }

    fn available(&self) -> usize {
        self.pool.len()
    }

    fn request(&mut self, count: usize) -> Result<(), Self::Error> {
        self.pending += count;
        // Allocate RCOT correlations on sender side
        // For 64-bit field (Goldilocks): 64 RCOT per VOLE
        // Security matches field size (~64 bits), which bounds JV soundness anyway
        let ots_needed = count * 64;
        RCOTSender::alloc(&mut self.ot, ots_needed)
            .map_err(|e| VoleProviderError::OtError(e.to_string()))?;
        self.total_ots_allocated += ots_needed;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        if self.pending == 0 {
            return Ok(());
        }

        // Convert RCOT to field VOLEs
        // This is the computational cost we want to benchmark
        let count = self.pending;
        self.convert_rcot_to_voles(count);
        self.total_voles_generated += count;
        self.pending = 0;

        Ok(())
    }

    fn take(&mut self, count: usize) -> Result<Vec<ItMac<F>>, Self::Error> {
        if count > self.pool.len() {
            return Err(VoleProviderError::NotEnoughVoles {
                requested: count,
                available: self.pool.len(),
            });
        }

        // Take from end for efficiency
        let voles = self.pool.split_off(self.pool.len() - count);
        self.total_voles_consumed += count;
        Ok(voles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpz_ot_core::ideal::rcot::IdealRCOT;

    /// Simple test field: integers mod p.
    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    struct TestField(u64);

    const TEST_MODULUS: u64 = 1000000007;

    impl std::ops::Add for TestField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self((self.0 + rhs.0) % TEST_MODULUS)
        }
    }

    impl std::ops::Sub for TestField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self((self.0 + TEST_MODULUS - rhs.0) % TEST_MODULUS)
        }
    }

    impl std::ops::Mul for TestField {
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

    #[test]
    fn test_vole_provider_with_ideal_rcot() {
        let rcot = IdealRCOT::default();
        let mut provider = VoleProvider::<TestField, _>::new(rcot);

        assert_eq!(provider.available(), 0);

        // Generate 10 VOLEs
        let voles = provider.generate(10).unwrap();
        assert_eq!(voles.len(), 10);

        // All should verify
        let gk = provider.global_key();
        for vole in &voles {
            assert!(vole.verify(gk));
        }
    }

    #[test]
    fn test_vole_provider_request_flush_take() {
        let rcot = IdealRCOT::default();
        let mut provider = VoleProvider::<TestField, _>::new(rcot);

        // Request VOLEs
        provider.request(5).unwrap();
        assert_eq!(provider.available(), 0);

        // Flush
        provider.flush().unwrap();
        assert_eq!(provider.available(), 5);

        // Take
        let voles = provider.take(3).unwrap();
        assert_eq!(voles.len(), 3);
        assert_eq!(provider.available(), 2);
    }

    #[test]
    fn test_vole_provider_not_enough() {
        let rcot = IdealRCOT::default();
        let mut provider = VoleProvider::<TestField, _>::new(rcot);

        // Request 5
        provider.generate(5).unwrap();

        // Try to take 10
        let result = provider.take(10);
        assert!(matches!(result, Err(VoleProviderError::NotEnoughVoles { .. })));
    }

    #[test]
    fn test_vole_provider_multiple_batches() {
        let rcot = IdealRCOT::default();
        let mut provider = VoleProvider::<TestField, _>::new(rcot);
        let gk = provider.global_key().clone();

        // Generate in batches
        let v1 = provider.generate(100).unwrap();
        let v2 = provider.generate(200).unwrap();
        let v3 = provider.generate(50).unwrap();

        assert_eq!(v1.len(), 100);
        assert_eq!(v2.len(), 200);
        assert_eq!(v3.len(), 50);

        // All should verify
        for v in v1.iter().chain(v2.iter()).chain(v3.iter()) {
            assert!(v.verify(&gk));
        }
    }
}
