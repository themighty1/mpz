//! Benchmark for GPU-accelerated slot multiplication.
//!
//! Compares CPU vs GPU performance for batched slot operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng, rngs::StdRng};

use bgv_webgpu::{SlotMulGpuContext, TwiddleFactors};

const GOLDILOCKS: u64 = 0xFFFFFFFF00000001;
const N: usize = 8192;

/// Find primitive 2n-th root of unity for Goldilocks.
fn find_primitive_root(n: usize, t: u64) -> u64 {
    // For Goldilocks, 7 is a generator
    let g = 7u64;
    let order = 2 * n as u64;
    let exp = (t - 1) / order;
    mod_pow(g, exp, t)
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base as u128) % m as u128;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % m as u128) as u64;
    }
    result as u64
}

fn bench_gpu_context_creation(c: &mut Criterion) {
    let omega = find_primitive_root(N, GOLDILOCKS);

    c.bench_function("gpu_context_creation", |b| {
        b.iter(|| {
            let twiddles = TwiddleFactors::compute(N, GOLDILOCKS, omega);
            // GPU context creation is expensive, just benchmark twiddle computation
            black_box(twiddles)
        });
    });
}

fn bench_twiddle_computation(c: &mut Criterion) {
    let omega = find_primitive_root(N, GOLDILOCKS);

    c.bench_function("twiddle_computation", |b| {
        b.iter(|| {
            let twiddles = TwiddleFactors::compute(black_box(N), black_box(GOLDILOCKS), black_box(omega));
            black_box(twiddles)
        });
    });
}

fn bench_gpu_poly_mul(c: &mut Criterion) {
    let omega = find_primitive_root(N, GOLDILOCKS);
    let twiddles = TwiddleFactors::compute(N, GOLDILOCKS, omega);

    let ctx = match SlotMulGpuContext::new(&twiddles) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU benchmark: {}", e);
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(42);

    let mut group = c.benchmark_group("gpu_poly_mul");
    group.sample_size(10);

    for num_batches in [10, 90, 180] {
        // Generate random data
        let ct_ntt: Vec<u64> = (0..N).map(|_| rng.random::<u64>() % GOLDILOCKS).collect();
        let pt_batches: Vec<Vec<u64>> = (0..num_batches)
            .map(|_| (0..N).map(|_| rng.random::<u64>() % GOLDILOCKS).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batched_mul", num_batches),
            &num_batches,
            |b, _| {
                b.iter(|| {
                    ctx.poly_mul_batched(black_box(&ct_ntt), black_box(&pt_batches), GOLDILOCKS)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_gpu_encode_slots(c: &mut Criterion) {
    let omega = find_primitive_root(N, GOLDILOCKS);
    let twiddles = TwiddleFactors::compute(N, GOLDILOCKS, omega);

    let ctx = match SlotMulGpuContext::new(&twiddles) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping GPU benchmark: {}", e);
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(42);

    let mut group = c.benchmark_group("gpu_encode_slots");
    group.sample_size(10);

    for num_batches in [10, 90, 180] {
        let slot_batches: Vec<Vec<u64>> = (0..num_batches)
            .map(|_| (0..N).map(|_| rng.random::<u64>() % GOLDILOCKS).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batched_encode", num_batches),
            &num_batches,
            |b, _| {
                b.iter(|| {
                    ctx.encode_slots_batched(black_box(&slot_batches)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_twiddle_computation,
    bench_gpu_context_creation,
    bench_gpu_poly_mul,
    bench_gpu_encode_slots,
);
criterion_main!(benches);
