#!/bin/sh
# Build WASM benchmark with atomics/simd (requires nightly)
set -e
cd "$(dirname "$0")"

# Ensure correct wasm-pack version is installed
echo "Ensuring wasm-pack version (rev 32e52ca)..."
cargo install --git https://github.com/rustwasm/wasm-pack.git --rev 32e52ca

echo "Building main wasm-bench with nightly (atomics, simd128, build-std)..."
rustup run nightly \
    wasm-pack build . \
        --profile wasm \
        --target web \
        --out-dir pkg \
        -- -Zbuild-std=panic_abort,std

# Copy JS bridge files to pkg/ (imported by WASM via raw_module)
echo "Copying JS bridge files to pkg/..."
cp js/chi-bridge.js pkg/
cp js/check-workers.js pkg/
cp js/terms-worker.js pkg/
cp js/terms-bridge.js pkg/

echo "Done. WASM output in pkg/"

# Build chi-wasm (no atomics, with simd128) for private memory workers
echo ""
echo "Building chi-wasm (simd128, no atomics) for chi workers..."
cd ../chi-wasm
RUSTFLAGS='-Ctarget-feature=+simd128' wasm-pack build --target web --release --out-dir ../wasm-bench/pkg-chi

echo "Done. Chi WASM output in pkg-chi/"

# Build terms-wasm (no atomics, with simd128) for private memory workers
echo ""
echo "Building terms-wasm (simd128, no atomics) for terms workers..."
cd ../terms-wasm
RUSTFLAGS='-Ctarget-feature=+simd128' wasm-pack build --target web --release --out-dir ../wasm-bench/pkg-terms

echo "Done. Terms WASM output in pkg-terms/"
