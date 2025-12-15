//! Benchmarks for cache-optimized garbling using wire remapping.
//!
//! Compares different remapping strategies:
//! - Original circuit (576 KB labels buffer for AES)
//! - Runtime remapping via slot_map indirection
//! - Circuit remapping (renumbered wire IDs, 24 KB buffer)
//!
//! Run with: `cargo bench -p mpz-garble-core --bench remapped`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mpz_circuits::remap::RemappedCircuit;
use mpz_circuits::AES128;
use mpz_garble_core::half_gates::{RemappedGarbler, WireRemapping};
use mpz_garble_core::{half_gates, Key};
use mpz_memory_core::correlated::Delta;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Benchmark different remapping strategies for garbling
fn bench_remapped_garbling(c: &mut Criterion) {
    const N: usize = 100;

    let mut group = c.benchmark_group("remapped_garble_100x_aes128");
    group.throughput(Throughput::Elements(N as u64));

    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();

    // Original circuit (no remapping)
    group.bench_function("original", |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            for _ in 0..N {
                let mut iter = gb.generate(&AES128, delta, &hg_inputs).unwrap();
                let _: Vec<_> = iter.by_ref().collect();
                black_box(iter.finish().unwrap());
            }
        })
    });

    // Runtime remapping via slot_map indirection
    group.bench_function("runtime_slot_map", |b| {
        let remapping = WireRemapping::compute(&AES128);
        let mut gb = RemappedGarbler::new(remapping);
        b.iter(|| {
            for _ in 0..N {
                let mut iter = gb.generate(&AES128, delta, &hg_inputs).unwrap();
                let _: Vec<_> = iter.by_ref().collect();
                black_box(iter.finish().unwrap());
            }
        })
    });

    // Circuit remapping (wire IDs renumbered, no runtime indirection)
    group.bench_function("circuit_remap", |b| {
        let remapped = RemappedCircuit::new(&AES128);
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            for _ in 0..N {
                let mut iter = gb.generate(remapped.circuit(), delta, &hg_inputs).unwrap();
                let _: Vec<_> = iter.by_ref().collect();
                black_box(iter.finish().unwrap());
            }
        })
    });

    group.finish();
}

/// Print remapping statistics
fn bench_remapping_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("remapping_stats");

    // Just measure the one-time cost of computing remapping
    group.bench_function("compute_wire_remapping", |b| {
        b.iter(|| {
            let remapping = WireRemapping::compute(&AES128);
            black_box(remapping.num_slots())
        })
    });

    group.bench_function("compute_circuit_remap", |b| {
        b.iter(|| {
            let remapped = RemappedCircuit::new(&AES128);
            black_box(remapped.num_slots())
        })
    });

    group.finish();

    // Print stats (not part of benchmark timing)
    let remapping = WireRemapping::compute(&AES128);
    let remapped = RemappedCircuit::new(&AES128);
    println!("\n=== Remapping Statistics for AES-128 ===");
    println!("Original feed_count: {}", AES128.feed_count());
    println!(
        "WireRemapping slots: {} ({:.1}x reduction)",
        remapping.num_slots(),
        AES128.feed_count() as f64 / remapping.num_slots() as f64
    );
    println!(
        "RemappedCircuit slots: {} ({:.1}x reduction)",
        remapped.num_slots(),
        AES128.feed_count() as f64 / remapped.num_slots() as f64
    );
    println!(
        "Original buffer size: {} KB",
        AES128.feed_count() * 16 / 1024
    );
    println!(
        "Remapped buffer size: {} KB",
        remapped.num_slots() * 16 / 1024
    );
}

criterion_group!(benches, bench_remapped_garbling, bench_remapping_stats);
criterion_main!(benches);
