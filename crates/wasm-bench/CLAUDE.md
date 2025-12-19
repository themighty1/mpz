# Claude Instructions for wasm-bench

## Modification Protocol

- **Always ask before modifying ANY file** - including benchmarks, configs, and wasm-bench files
- When user says "yes" to a modification request, provide a brief overview of what will be changed BEFORE making edits
- This applies to ALL files in the mpz repo, not just files outside wasm-bench

## Scope Restrictions

- **Never modify files outside `crates/wasm-bench/`** without explicitly asking first
- This includes other mpz crates like mpz-garble-core, mpz-zk-core, mpz-common, etc.

## Benchmark Execution

- **NEVER run benchmarks or builds yourself** - always give the user the command to run
- After making changes, provide the appropriate command with notes on what needs rebuilding

## Rebuild Requirements

When providing benchmark commands, specify what needs rebuilding:

1. **WASM rebuild required** (changes to `src/*.rs` Rust code):
   ```bash
   ./build-wasm.sh
   ```

2. **Runner rebuild required** (changes to `src/bin/runner.rs`):
   ```bash
   cargo build --release --bin wasm-bench-runner
   ```

3. **No rebuild required** (changes to `js/*.js`, `index.html`, or just re-running):
   - Use the existing runner directly from target:
   ```bash
   ../../target/release/wasm-bench-runner [options]
   ```

## Command Examples

**ALWAYS use `--iterations 1 --samples 1` when giving benchmark commands:**

```bash
# Run specific group
../../target/release/wasm-bench-runner -g zk_core --iterations 1 --samples 1

# Run specific benchmark
../../target/release/wasm-bench-runner -b zk_core/full_protocol --iterations 1 --samples 1
```
