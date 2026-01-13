//! Math helper functions for modular arithmetic.

/// Computes base^i mod modulus for i = 0..n-1.
pub fn compute_powers(base: u64, n: usize, modulus: u64) -> Vec<u64> {
    let mut powers = Vec::with_capacity(n);
    let mut current = 1u64;
    for _ in 0..n {
        powers.push(current);
        current = mod_mul(current, base, modulus);
    }
    powers
}

/// Modular multiplication: (a * b) mod m.
pub fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: base^exp mod m.
pub fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

/// Modular inverse using extended Euclidean algorithm.
pub fn mod_inverse(a: u64, modulus: u64) -> u64 {
    let mut t: i128 = 0;
    let mut new_t: i128 = 1;
    let mut r: i128 = modulus as i128;
    let mut new_r: i128 = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        let temp = t - quotient * new_t;
        t = new_t;
        new_t = temp;
        let temp = r - quotient * new_r;
        r = new_r;
        new_r = temp;
    }

    if t < 0 {
        (t + modulus as i128) as u64
    } else {
        t as u64
    }
}

/// Finds a primitive 2n-th root of unity modulo q.
pub fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
    let order = 2 * n as u64;
    if (q - 1) % order != 0 {
        return None;
    }
    let exp = (q - 1) / order;
    for g in 2..1000u64 {
        let root = mod_pow(g, exp, q);
        let root_n = mod_pow(root, n as u64, q);
        if root_n == q - 1 {
            return Some(root);
        }
    }
    None
}

/// Finds psi (primitive 2n-th root of unity) for NTT.
pub fn find_psi(n: usize, q: u64) -> Option<u64> {
    find_primitive_root(n, q)
}
