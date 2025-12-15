//! Micro-benchmarks for isolating iteration and memory access overhead.
//!
//! These benchmarks measure the non-crypto overhead in garbling by stubbing
//! out the AES hash operations.
//!
//! Run with: `cargo bench -p mpz-garble-core --bench iteration`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mpz_circuits::remap::RemappedCircuit;
use mpz_circuits::{Gate, AES128};
use mpz_core::Block;
use mpz_garble_core::{half_gates, Key};
use mpz_memory_core::correlated::Delta;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// AND gate logic WITHOUT rtccr_many hash - does all XORs and bit ops
#[inline]
fn and_gate_no_hash(x_0: &Block, y_0: &Block, delta: &Block, gid: usize) -> (Block, [Block; 2]) {
    let x_1 = *x_0 ^ *delta;
    let y_1 = *y_0 ^ *delta;

    let p_a = x_0.lsb();
    let p_b = y_0.lsb();
    let _j = Block::new((gid as u128).to_be_bytes());
    let _k = Block::new(((gid + 1) as u128).to_be_bytes());

    // Skip: cipher.rtccr_many(&[j, k, j, k], &mut h);
    // Instead, just use the inputs as fake hash outputs
    let hx_0 = *x_0;
    let hy_0 = *y_0;
    let hx_1 = x_1;
    let hy_1 = y_1;

    // Garbled row of garbler half-gate
    let t_g = hx_0 ^ hx_1 ^ (Block::SELECT_MASK[p_b as usize] & *delta);
    let w_g = hx_0 ^ (Block::SELECT_MASK[p_a as usize] & t_g);

    // Garbled row of evaluator half-gate
    let t_e = hy_0 ^ hy_1 ^ *x_0;
    let w_e = hy_0 ^ (Block::SELECT_MASK[p_b as usize] & (t_e ^ *x_0));

    let z_0 = w_g ^ w_e;

    (z_0, [t_g, t_e])
}

/// Benchmark iteration overhead: comparing hash vs no-hash
fn bench_iteration_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_100x_aes128");
    group.throughput(Throughput::Elements(100));

    let mut rng = StdRng::seed_from_u64(0);

    // Setup for original circuit
    let original_circ = &*AES128;
    let original_labels: Vec<Block> = (0..original_circ.feed_count())
        .map(|_| rng.random())
        .collect();

    // Setup for remapped circuit
    let remapped = RemappedCircuit::new(&AES128);
    let remapped_labels: Vec<Block> = (0..remapped.num_slots()).map(|_| rng.random()).collect();

    let delta_block: Block = rng.random();

    // Benchmark: Original circuit, full AND gate logic WITHOUT hash
    group.bench_function("original_no_hash", |b| {
        let mut labels = original_labels.clone();
        let mut gid = 1usize;
        b.iter(|| {
            for _ in 0..100 {
                gid = 1;
                for gate in original_circ.gates() {
                    match gate {
                        Gate::Xor { x, y, z } => {
                            let x_0 = labels[x.id()];
                            let y_0 = labels[y.id()];
                            labels[z.id()] = x_0 ^ y_0;
                        }
                        Gate::And { x, y, z } => {
                            let x_0 = labels[x.id()];
                            let y_0 = labels[y.id()];
                            let (z_0, encrypted_gate) =
                                and_gate_no_hash(&x_0, &y_0, &delta_block, gid);
                            labels[z.id()] = z_0;
                            gid += 2;
                            black_box(encrypted_gate);
                        }
                        Gate::Inv { x, z } => {
                            let x_0 = labels[x.id()];
                            labels[z.id()] = x_0 ^ delta_block;
                        }
                        Gate::Id { x, z } => {
                            let x_0 = labels[x.id()];
                            labels[z.id()] = x_0;
                        }
                    }
                }
            }
            black_box(labels[0])
        })
    });

    // Benchmark: Remapped circuit, full AND gate logic WITHOUT hash
    group.bench_function("remapped_no_hash", |b| {
        let mut labels = remapped_labels.clone();
        let mut gid = 1usize;
        b.iter(|| {
            for _ in 0..100 {
                gid = 1;
                for gate in remapped.circuit().gates() {
                    match gate {
                        Gate::Xor { x, y, z } => {
                            let x_0 = labels[x.id()];
                            let y_0 = labels[y.id()];
                            labels[z.id()] = x_0 ^ y_0;
                        }
                        Gate::And { x, y, z } => {
                            let x_0 = labels[x.id()];
                            let y_0 = labels[y.id()];
                            let (z_0, encrypted_gate) =
                                and_gate_no_hash(&x_0, &y_0, &delta_block, gid);
                            labels[z.id()] = z_0;
                            gid += 2;
                            black_box(encrypted_gate);
                        }
                        Gate::Inv { x, z } => {
                            let x_0 = labels[x.id()];
                            labels[z.id()] = x_0 ^ delta_block;
                        }
                        Gate::Id { x, z } => {
                            let x_0 = labels[x.id()];
                            labels[z.id()] = x_0;
                        }
                    }
                }
            }
            black_box(labels[0])
        })
    });

    // Full garbling WITH crypto for comparison
    let delta = Delta::random(&mut rng);
    let hg_inputs: Vec<Key> = (0..256).map(|_| rng.random()).collect();

    group.bench_function("original_with_hash", |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            for _ in 0..100 {
                let mut iter = gb.generate(original_circ, delta, &hg_inputs).unwrap();
                let _: Vec<_> = iter.by_ref().collect();
                black_box(iter.finish().unwrap());
            }
        })
    });

    group.bench_function("remapped_with_hash", |b| {
        let mut gb = half_gates::Garbler::default();
        b.iter(|| {
            for _ in 0..100 {
                let mut iter = gb.generate(remapped.circuit(), delta, &hg_inputs).unwrap();
                let _: Vec<_> = iter.by_ref().collect();
                black_box(iter.finish().unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_iteration_overhead);
criterion_main!(benches);
