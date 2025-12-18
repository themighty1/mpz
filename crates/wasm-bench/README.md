# mpz-wasm-bench

WASM benchmarks for mpz garbling libraries. Runs in headless Chrome via chromiumoxide to measure real browser performance with Web Workers and SharedArrayBuffer.

## Prerequisites

- Rust with `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- Chrome/Chromium browser

## Quick Start

```bash
# Build WASM module
./build-wasm.sh

# Run all benchmarks
cargo run --release --bin wasm-bench-runner

# Run specific benchmark
cargo run --release --bin wasm-bench-runner -- -b garble/semihonest_aes_mt_batched

# List available benchmarks
cargo run --release --bin wasm-bench-runner -- --list
```

## Available Benchmarks

### garble_core (raw garbling primitives)

Single-threaded benchmarks measuring raw garbling/evaluation speed without protocol overhead:

| Benchmark | Description |
|-----------|-------------|
| `garble_core/half_gates_garble` | Half-gates garbling of AES-128 circuit |
| `garble_core/half_gates_evaluate` | Half-gates evaluation of AES-128 circuit |

### zk_core (QuickSilver ZK core)

Single-threaded benchmarks measuring the QuickSilver ZK core proving/verification performance:

| Benchmark | Description |
|-----------|-------------|
| `zk_core/prover_execute` | Prover execute phase only (generate adjustments) |
| `zk_core/verifier_execute` | Verifier execute phase only (consume adjustments) |
| `zk_core/full_protocol` | Complete ZK protocol (execute + check phases) |
| `zk_core/check_only` | SVOLE-based consistency check phase only |

### zk (full ZK protocol with VM)

End-to-end ZK protocol benchmarks including proof generation, verification, and communication:

| Benchmark | Description |
|-----------|-------------|
| `zk/zk_st_batched` | 256 AES circuits batched, single-threaded context |
| `zk/zk_mt_batched` | 256 AES circuits batched, multi-threaded context |

### garble (full semihonest 2PC protocol)

End-to-end protocol benchmarks including garbling, evaluation, and communication:

| Benchmark | Description |
|-----------|-------------|
| `garble/semihonest_aes` | Single AES circuit, single-threaded context |
| `garble/semihonest_aes_st_batched` | 256 AES circuits batched, single-threaded context |
| `garble/semihonest_aes_mt_batched` | 256 AES circuits batched, multi-threaded context |

### test (debugging)

| Benchmark | Description |
|-----------|-------------|
| `test/mt_context_only` | Minimal MT context ping-pong test |

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

Groups: garble_core, zk_core, zk, garble, test
```

## Examples

### Basic Usage

```bash
# Run all benchmarks with defaults (100 iterations, 10 samples)
cargo run --release --bin wasm-bench-runner

# Quick test run
cargo run --release --bin wasm-bench-runner -- --iterations 10 --samples 3

# Run with visible browser for debugging
cargo run --release --bin wasm-bench-runner -- --headed -b garble_core/half_gates_garble
```

### Comparing ST vs MT

```bash
# Compare single-threaded vs multi-threaded batched benchmarks
cargo run --release --bin wasm-bench-runner -- \
  -b garble/semihonest_aes_st_batched \
  -b garble/semihonest_aes_mt_batched
```

### Thread Scaling Analysis

```bash
# Sweep thread counts to analyze scaling
cargo run --release --bin wasm-bench-runner -- \
  --sweep \
  -b garble/semihonest_aes_mt_batched

# Run MT benchmark with specific thread count
cargo run --release --bin wasm-bench-runner -- \
  -c 4 \
  -b garble/semihonest_aes_mt_batched
```

### Run by Group

```bash
# Run all garble_core benchmarks
cargo run --release --bin wasm-bench-runner -- -g garble_core

# Run all zk_core benchmarks
cargo run --release --bin wasm-bench-runner -- -g zk_core

# Run all garble benchmarks (includes MT)
cargo run --release --bin wasm-bench-runner -- -g garble

# Run multiple groups
cargo run --release --bin wasm-bench-runner -- -g garble_core -g zk_core
```

## Output Format

Results are displayed as a table with:

- **Median (ms)**: Median time for all iterations in a sample
- **Per-iter (us)**: Time per iteration (circuit) in microseconds
- **AND gates/s**: Throughput in AND gates processed per second

Example output:
```
=== garble_core ===
Name                                     Median (ms)   Per-iter (us)   AND gates/s
----------------------------------------------------------------------------------
garble_core/half_gates_garble                  45.23          452.30       14.52M
```

## Architecture Notes

### Multi-threaded Context

The MT benchmarks use `web_spawn` for WASM threading via Web Workers. The `--concurrency` parameter controls the maximum number of worker threads used for parallel garbling.

**Minimum concurrency is 2** because the garbler internally uses `ctx.try_join()` which forks into 2 threads (one for OT setup, one for circuit preprocessing).

### Batched Benchmarks

The batched benchmarks process 256 AES circuits in one batch to amortize context setup overhead and better demonstrate parallelism benefits. This is more representative of real-world usage where multiple circuits are processed together.

### SharedArrayBuffer Requirements

MT benchmarks require SharedArrayBuffer which needs specific HTTP headers:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

The built-in HTTP server sets these headers automatically.

## Development

### Manual Browser Testing

Start the server and open in browser:
```bash
# Start dev server (requires a simple HTTP server with COOP/COEP headers)
cd crates/wasm-bench
python3 -m http.server 8080  # Note: won't work for MT without proper headers

# Or use the runner in headed mode
cargo run --release --bin wasm-bench-runner -- --headed
```

### Rebuilding WASM

After modifying Rust code:
```bash
./build-wasm.sh
```

The script builds with:
- `--target web` for ES module output
- Atomics and bulk-memory features enabled
- Release optimizations
