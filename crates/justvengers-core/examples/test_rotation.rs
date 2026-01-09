//! Quick test to debug rotation correctness.

use mpz_justvengers_core::{
    RnsKeyPair, RnsCiphertext, RnsGaloisKeys, RnsBgvParams,
};
use rand::rng;

fn main() {
    let mut rng = rng();

    // Use standard 3 moduli
    let params = RnsBgvParams::goldilocks();
    let keypair = RnsKeyPair::generate(&params, &mut rng);

    println!("=== Test 1: Basic encrypt/decrypt ===");
    let slots: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let ct = RnsCiphertext::encrypt_slots(&keypair.pk, &slots, &mut rng);
    let decrypted = ct.decrypt_slots(&keypair.sk);
    println!("Original:  {:?}", &slots);
    println!("Decrypted: {:?}", &decrypted[..8]);
    println!("Match: {}", slots == decrypted[..8]);

    println!("\n=== Test 2: Single rotation ===");
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);
    let gk0 = galois_keys.get_key(0).unwrap();

    // Small slots to track what happens
    let n = params.n;
    let mut test_slots = vec![0u64; n];
    test_slots[0] = 100;
    test_slots[1] = 200;
    test_slots[2] = 300;

    let ct2 = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);
    let dec_before = ct2.decrypt_slots(&keypair.sk);
    println!("Before rotation slots[0..5]: {:?}", &dec_before[..5]);

    let rotated = ct2.apply_automorphism(gk0);
    let dec_after = rotated.decrypt_slots(&keypair.sk);
    println!("After rotation slots[0..5]:  {:?}", &dec_after[..5]);

    println!("\n=== Test 3: Add original + rotated ===");
    let added = ct2.add(&rotated);
    let dec_added = added.decrypt_slots(&keypair.sk);
    println!("After add slots[0..5]: {:?}", &dec_added[..5]);

    println!("\n=== Test 4: Full sum_slots (small values) ===");
    // Use very small values
    let small_slots: Vec<u64> = (0..n).map(|i| if i < 8 { 1 } else { 0 }).collect();
    let expected_sum: u64 = small_slots.iter().sum();
    println!("Expected sum: {}", expected_sum);

    let ct3 = RnsCiphertext::encrypt_slots(&keypair.pk, &small_slots, &mut rng);
    let summed = ct3.sum_slots(&galois_keys);
    let dec_summed = summed.decrypt_slots(&keypair.sk);
    println!("Decrypted slot 0: {}", dec_summed[0]);
    println!("Decrypted slot 1: {}", dec_summed[1]);
    println!("Decrypted slot 100: {}", dec_summed[100]);
}
