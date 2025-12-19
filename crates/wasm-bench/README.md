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

## Available Benchmark Groups

| Group | Description |
|-------|-------------|
| `garbler_core` | Half-gates garbling primitives |
| `evaluator_core` | Half-gates evaluation primitives |
| `garble` | Garbler and evaluator |
| `zk_prover_core` | QuickSilver ZK prover primitives |
| `zk_verifier_core` | QuickSilver ZK verifier primitives |
| `zk_prover` | ZK prover |
| `zk_verifier` | ZK verifier |
| `ferret_sender` | Ferret OT sender |

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

Groups: garbler_core, evaluator_core, zk_prover_core, zk_verifier_core, zk_prover, zk_verifier, garble, ferret_sender
```

## Examples

```bash
# Quick test run
../../target/release/wasm-bench-runner -g garble --iterations 1 --samples 1

# Run garble benchmarks with more accuracy
../../target/release/wasm-bench-runner -g garble --iterations 3 --samples 5

# Thread scaling analysis
../../target/release/wasm-bench-runner --sweep -g zk_prover_core

# Run both prover and verifier
../../target/release/wasm-bench-runner -g zk_prover -g zk_verifier --iterations 1 --samples 5
```

## Architecture Notes

### Web Worker Requirement

MT benchmarks that use rayon internally must run on Web Workers because `Atomics.wait` is forbidden on the main browser thread. These benchmarks use `web_spawn::spawn` to run on workers.

### SharedArrayBuffer Requirements

MT benchmarks require SharedArrayBuffer which needs specific HTTP headers:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

The built-in HTTP server sets these headers automatically.
