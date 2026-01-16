# MPZ JustVengers BGV

## WASM zkVM Performance Benchmarking

To emulate WASM zkVM performance, run:

```bash
cd crates/wasm-bench
cargo build --release
xvfb-run -a ../../target/release/wasm-bench-runner --reps 4000 --iterations 1 --samples 1 -v
```

### Options

- `--reps <N>` - Number of steps (repetitions) in the ZK protocol
- `--iterations <N>` - Number of benchmark iterations per sample
- `--samples <N>` - Number of samples to collect
- `-v` - Verbose output with detailed timing breakdown

### Example

```bash
# Run with 1000 steps
xvfb-run -a ../../target/release/wasm-bench-runner --reps 1000 --iterations 1 --samples 1 -v

# Run with 8192 steps
xvfb-run -a ../../target/release/wasm-bench-runner --reps 8192 --iterations 1 --samples 1 -v
```

The benchmark outputs a detailed timing breakdown showing GPU vs CPU time for each phase of the protocol.
