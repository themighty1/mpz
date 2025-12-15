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
| `garble_core/three_halves_garble` | Three-halves garbling of AES-128 circuit |
| `garble_core/half_gates_evaluate` | Half-gates evaluation of AES-128 circuit |
| `garble_core/three_halves_evaluate` | Three-halves evaluation of AES-128 circuit |

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
  --bench, -b <NAME>    Run specific benchmark (can be repeated)
  --list, -l            List available benchmarks
  --headed              Run with visible browser window (for debugging)
  --help, -h            Show help
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

### garble-core Only

```bash
# Run only the raw garbling primitive benchmarks
cargo run --release --bin wasm-bench-runner -- \
  -b garble_core/half_gates_garble \
  -b garble_core/half_gates_evaluate \
  -b garble_core/three_halves_garble \
  -b garble_core/three_halves_evaluate
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
garble_core/three_halves_garble                38.91          389.10       16.89M
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
