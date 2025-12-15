//! Benchmark for chi coefficient generation.
//!
//! Run with: cargo bench -p mpz-zk-core --bench chi

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mpz_core::Block;

/// Sequential chi generation (current implementation)
fn compute_chis_sequential(chi: Block, count: usize) -> Vec<Block> {
    let mut chis = Vec::with_capacity(count);
    let mut current = chi;
    chis.push(current);
    for _ in 1..count {
        current = current.gfmul(current);
        chis.push(current);
    }
    chis
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("chi_generation");

    let chi = Block::from([0x42u8; 16]);

    // Test various AND gate counts
    for count in [6_400, 64_000, 640_000, 1_600_000] {
        group.bench_with_input(
            BenchmarkId::new("sequential", count),
            &count,
            |b, &count| {
                b.iter(|| compute_chis_sequential(chi, count));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
