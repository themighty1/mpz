#!/bin/bash
# Build chi-wasm WITHOUT atomics (for private memory workers)

set -e

cd "$(dirname "$0")"

echo "Building chi-wasm (no atomics)..."

# Standard wasm-pack build - no atomics, no shared memory
wasm-pack build --target web --release --out-dir pkg

echo "Done. Output in pkg/"
