//! Generates BGV test fixture data in binary format.
//!
//! Run with: cargo run -p mpz-justvengers-core --example generate_bgv_fixture_binary
//!
//! This generates deterministic BGV keys from a fixed seed and writes them
//! to separate binary files that can be loaded later.
//!
//! Output files (in ./bgv_fixtures/ directory):
//! - secret_key.bin: The secret key
//! - public_key.bin: The public key
//! - galois_keys.bin: All Galois keys for slot operations
//! - test_ciphertext.bin: A test ciphertext for verification
//! - params.bin: The BGV parameters used

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use mpz_justvengers_core::{RnsBgvParams, RnsCiphertext, RnsGaloisKeys, RnsKeyPair};
use mpz_core::prg::Prg;

const OUTPUT_DIR: &str = "bgv_fixtures";

fn main() {
    eprintln!("Generating BGV fixture data in binary format...");
    eprintln!("This will take a while due to key generation...");

    // Create output directory
    let output_path = Path::new(OUTPUT_DIR);
    fs::create_dir_all(output_path).expect("Failed to create output directory");

    // Use fixed seed for deterministic generation
    let mut rng = Prg::new_with_seed([0u8; 16]);

    // Use production Goldilocks parameters (3 moduli)
    let params = RnsBgvParams::goldilocks();
    let n = params.n;
    let num_moduli = params.num_moduli;

    eprintln!("Parameters: n={}, num_moduli={}", n, num_moduli);

    // Generate keypair
    eprintln!("Generating keypair...");
    let keypair = RnsKeyPair::generate(&params, &mut rng);

    // Generate Galois keys for sum_slots (13 keys for n=8192)
    eprintln!("Generating Galois keys (this is slow)...");
    let galois_keys = RnsGaloisKeys::generate(&keypair.sk, &mut rng);

    // Create a test ciphertext with known slot values
    eprintln!("Creating test ciphertext...");
    let test_slots: Vec<u64> = (0..n).map(|i| (i as u64) % 1000).collect();
    let test_ct = RnsCiphertext::encrypt_slots(&keypair.pk, &test_slots, &mut rng);

    // Compute expected sum
    let t = params.t;
    let expected_sum: u64 =
        test_slots.iter().fold(0u128, |acc, &x| (acc + x as u128) % t as u128) as u64;

    eprintln!("Expected sum of slots: {}", expected_sum);

    // Write parameters
    eprintln!("Writing params.bin...");
    write_bincode(output_path.join("params.bin"), &params);

    // Write secret key
    eprintln!("Writing secret_key.bin...");
    write_bincode(output_path.join("secret_key.bin"), &keypair.sk);

    // Write public key
    eprintln!("Writing public_key.bin...");
    write_bincode(output_path.join("public_key.bin"), &keypair.pk);

    // Write Galois keys
    eprintln!("Writing galois_keys.bin...");
    write_bincode(output_path.join("galois_keys.bin"), &galois_keys);

    // Write test ciphertext
    eprintln!("Writing test_ciphertext.bin...");
    write_bincode(output_path.join("test_ciphertext.bin"), &test_ct);

    // Write expected sum for verification
    eprintln!("Writing expected_sum.txt...");
    let mut f = File::create(output_path.join("expected_sum.txt")).expect("Failed to create file");
    writeln!(f, "{}", expected_sum).expect("Failed to write");

    // Print file sizes
    eprintln!("\nGenerated files:");
    for entry in fs::read_dir(output_path).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read entry");
        let metadata = entry.metadata().expect("Failed to read metadata");
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  {:30} {:8.2} MB", entry.file_name().to_string_lossy(), size_mb);
    }

    eprintln!("\nDone! Binary fixtures generated in ./{}/", OUTPUT_DIR);
}

fn write_bincode<T: serde::Serialize, P: AsRef<Path>>(path: P, value: &T) {
    let bytes = bincode::serialize(value).expect("Failed to serialize");
    let mut file = File::create(path).expect("Failed to create file");
    file.write_all(&bytes).expect("Failed to write");
}
