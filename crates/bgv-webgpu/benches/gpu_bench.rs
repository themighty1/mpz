//! Benchmarks for GPU-accelerated BGV slot-wise multiplication.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bgv_webgpu::{GpuContext, GpuCiphertext, SlotWiseMul, BatchParams, GOLDILOCKS_Q};

fn bench_slot_wise_mul(c: &mut Criterion) {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("GPU not available: {}, skipping benchmarks", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_slot_wise_mul");

    // Test configurations: (n, num_batches)
    for (n, num_batches) in [
        (256, 10),      // Small test
        (1024, 40),     // Medium test
        (8192, 80),     // JustVengers standard
    ] {
        let params = BatchParams::new(n, GOLDILOCKS_Q, num_batches);

        // Create test ciphertext
        let ct_c0: Vec<u64> = (0..n as u64).collect();
        let ct_c1: Vec<u64> = (0..n as u64).map(|x| x + 1).collect();
        let ct = GpuCiphertext::from_coeffs(&ctx, &ct_c0, &ct_c1).unwrap();

        // Create scalars: num_batches × n
        let scalars: Vec<u64> = (0..(num_batches * n))
            .map(|i| (i as u64) % GOLDILOCKS_Q)
            .collect();

        let mul = SlotWiseMul::new(&ctx, params).unwrap();

        group.bench_with_input(
            BenchmarkId::new("slotwise", format!("n={}_batches={}", n, num_batches)),
            &(&ct, &scalars),
            |b, (ct, scalars)| {
                b.iter(|| {
                    mul.run(&ctx, black_box(ct), black_box(scalars)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_slot_wise_mul);
criterion_main!(benches);
