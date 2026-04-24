//! VOLE test utilities.

use mpz_fields::Field;
use poly_proof_core::SubfieldOf;

/// Asserts the correctness of vector oblivious linear evaluation.
pub fn assert_vole<W, E>(delta: E, keys: &[E], values: &[W], macs: &[E])
where
    W: SubfieldOf<E>,
    E: Field,
{
    assert_eq!(keys.len(), values.len());
    assert_eq!(keys.len(), macs.len());
    assert!(
        keys.iter()
            .zip(values.iter().zip(macs))
            .all(|(key, (value, mac))| *mac == *key + delta * value.embed())
    );
}

/// Asserts the correctness of vector oblivious polynomial evaluation.
pub fn assert_vope<E: Field>(delta: E, polynomials: &[Vec<E>], evaluations: &[E]) {
    assert_eq!(polynomials.len(), evaluations.len());
    for (poly, expected) in polynomials.iter().zip(evaluations) {
        let mut computed = E::zero();
        let mut power = E::one();
        for &c in poly {
            computed = computed + c * power;
            power = power * delta;
        }
        assert_eq!(*expected, computed);
    }
}
