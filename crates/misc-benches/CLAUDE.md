# Claude Instructions for misc-benches

## Modification Protocol

- **Always ask before modifying ANY file** - including benchmarks, configs, and misc-benches files
- When user says "yes" to a modification request, provide a brief overview of what will be changed BEFORE making edits
- This applies to ALL files in the mpz repo, not just files outside misc-benches

## Git Commits

- **Make a git commit after every change** - each task/modification should have its own commit
- Use a one-line commit message describing the task
- Format: `git commit -m "misc-benches: <brief description>"`
- Examples:
  - `misc-benches: add trivial parallel benchmark`
  - `misc-benches: fix black_box optimization in loop`
  - `misc-benches: add private memory WAT module`
- This enables easy rollback if a change causes issues

## Scope Restrictions

- **Never modify files outside `crates/misc-benches/`** without explicitly asking first
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
   cargo build --release --bin misc-bench-runner
   ```

3. **No rebuild required** (changes to `js/*.js`, `index.html`, or just re-running):
   - Use the existing runner directly from target:
   ```bash
   ../../target/release/misc-bench-runner [options]
   ```

## Command Examples

**ALWAYS use `--iterations 1 --samples 1` when giving benchmark commands:**

```bash
# Run all benchmarks
../../target/release/misc-bench-runner -g aes_compare --iterations 1 --samples 1

# Run specific benchmark
../../target/release/misc-bench-runner -b aes_compare/aes_crate --iterations 1 --samples 1
```
