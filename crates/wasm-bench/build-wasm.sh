#!/bin/sh
# Build WASM benchmark with simd128 (requires nightly)
# NOTE: atomics disabled - not needed for WebGPU and causes Atomics.wait errors
set -e
cd "$(dirname "$0")"

# Ensure correct wasm-pack version is installed
echo "Ensuring wasm-pack version (rev 32e52ca)..."
cargo install --git https://github.com/rustwasm/wasm-pack.git --rev 32e52ca

echo "Building with nightly (simd128, build-std, no atomics)..."
rustup run nightly \
    wasm-pack build . \
        --profile wasm \
        --target web \
        --out-dir pkg \
        -- -Zbuild-std=panic_abort,std

echo "Done. WASM output in pkg/"
