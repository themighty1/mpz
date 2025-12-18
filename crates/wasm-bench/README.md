# mpz-wasm-bench

WASM benchmarks for mpz libraries. Runs in headless Chrome via chromiumoxide to measure real browser performance with Web Workers and SharedArrayBuffer.

## Prerequisites

- Rust with `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- Chrome/Chromium browser

## Quick Start

```bash
# Build WASM module
./build-wasm.sh

# Build runner
cargo build --release --bin wasm-bench-runner

# Run all benchmarks
../../target/release/wasm-bench-runner

# Run specific group
../../target/release/wasm-bench-runner -g garble --iterations 1 --samples 1

# List available benchmarks
../../target/release/wasm-bench-runner --list
```

## Available Benchmarks

### garble_core (raw garbling primitives)

Single-threaded benchmarks measuring raw garbling/evaluation speed without protocol overhead:

| Benchmark | Description |
|-----------|-------------|
| `garble_core/half_gates_garble` | Half-gates garbling of AES-128 circuit |
| `garble_core/half_gates_evaluate` | Half-gates evaluation of AES-128 circuit |

### garble (garbler/evaluator with message replay)

Isolated garbler and evaluator benchmarks using recorded messages for replay:

| Benchmark | Description |
|-----------|-------------|
| `garble/garbler_100k` | Garbler with 100K AND gates |
| `garble/garbler_1m` | Garbler with 1M AND gates |
| `garble/garbler_10m` | Garbler with 10M AND gates |
| `garble/evaluator_100k` | Evaluator with 100K AND gates |
| `garble/evaluator_1m` | Evaluator with 1M AND gates |
| `garble/evaluator_10m` | Evaluator with 10M AND gates |

### zk_core (QuickSilver ZK core)

Single-threaded benchmarks measuring the QuickSilver ZK core proving/verification performance:

| Benchmark | Description |
|-----------|-------------|
| `zk_core/prover_execute` | Prover execute phase only |
| `zk_core/verifier_execute` | Verifier execute phase only |
| `zk_core/full_protocol` | Complete ZK protocol (execute + check) |
| `zk_core/check_only` | SVOLE-based consistency check only |

### zk (full ZK protocol with VM)

End-to-end ZK protocol benchmarks:

| Benchmark | Description |
|-----------|-------------|
| `zk/zk_st_batched` | Single-threaded context |
| `zk/zk_mt_batched` | Multi-threaded context |

### zk_overhead (recording overhead measurement)

Compares baseline MT context vs recording MT context:

| Benchmark | Description |
|-----------|-------------|
| `zk_overhead/baseline_100k` | Baseline context, 100K gates |
| `zk_overhead/baseline_1m` | Baseline context, 1M gates |
| `zk_overhead/baseline_10m` | Baseline context, 10M gates |
| `zk_overhead/recording_100k` | Recording context, 100K gates |
| `zk_overhead/recording_1m` | Recording context, 1M gates |
| `zk_overhead/recording_10m` | Recording context, 10M gates |

### zk_prover / zk_verifier (isolated with message replay)

Isolated prover/verifier benchmarks with various batch sizes (200k-1000k) in both ST and MT variants.

### ferret (Ferret OT)

| Benchmark | Description |
|-----------|-------------|
| `ferret/sender_st` | Single-threaded Ferret sender |
| `ferret/sender_mt` | Multi-threaded Ferret sender |

## CLI Options

```
Usage: wasm-bench-runner [OPTIONS]

Options:
  --iterations <N>      Number of iterations per benchmark (default: 100)
  --samples <N>         Number of samples per benchmark (default: 10)
  --concurrency, -c <N> Thread count for MT benchmarks (default: auto, min: 2)
  --sweep               Run MT benchmarks with 2,3,4,6,8,12,16 threads
  --group, -g <GROUP>   Run all benchmarks in a group (can be repeated)
  --bench, -b <NAME>    Run specific benchmark (can be repeated)
  --list, -l            List available groups and benchmarks
  --headed              Run with visible browser window (for debugging)
  --help, -h            Show help

Groups: garble_core, zk_core, zk, zk_overhead, zk_prover, zk_verifier, garble, ferret, test
```

## Examples

```bash
# Quick test run
../../target/release/wasm-bench-runner -g garble --iterations 1 --samples 1

# Run garble benchmarks with more accuracy
../../target/release/wasm-bench-runner -g garble --iterations 3 --samples 5

# Compare recording overhead
../../target/release/wasm-bench-runner -b zk_overhead/baseline_1m -b zk_overhead/recording_1m --iterations 3 --samples 5

# Thread scaling analysis
../../target/release/wasm-bench-runner --sweep -g garble
```

## Architecture Notes

### Web Worker Requirement

MT benchmarks that use rayon internally (zk, zk_overhead) must run on Web Workers because `Atomics.wait` is forbidden on the main browser thread. These benchmarks use `web_spawn::spawn` to run on workers.

### SharedArrayBuffer Requirements

MT benchmarks require SharedArrayBuffer which needs specific HTTP headers:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

The built-in HTTP server sets these headers automatically.
