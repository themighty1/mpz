// Chi Bridge - Provides request_chi_computation import for WASM
//
// This module is imported by the WASM code (via wasm-bindgen raw_module).
// It runs in web-spawn worker context where Atomics.wait() is allowed.
//
// Note: This file is copied to pkg/ during build, so paths are relative to pkg/.

import { getChiSignalBuffer, setWasmMemory } from '../js/bench.js';

// Cached references
let signalBuffer = null;
let signalView = null;
let wasmMemory = null;

// Initialize the bridge (called once per worker)
function ensureInit() {
    if (!signalBuffer) {
        signalBuffer = getChiSignalBuffer();
        signalView = new Int32Array(signalBuffer);
    }
}

// Set WASM memory reference (must be called before request_chi_computation)
export function setChiBridgeMemory(memory) {
    wasmMemory = memory;
    setWasmMemory(memory);
}

// Request chi computation from JS coordinator.
// This function BLOCKS via Atomics.wait() until result is ready.
//
// Signal buffer layout (32 bytes):
//   Int32[0]: status (0=idle, 1=pending, 2=ready, 3=error)
//   Int32[1]: count
//   Int32[2]: resultPtr
//   Int32[3]: reserved
//   Uint8[16..31]: chi input (16 bytes)
//
// Parameters are raw pointers into WASM linear memory.
export function request_chi_computation(chiPtr, count, resultPtr) {
    ensureInit();

    if (!wasmMemory) {
        throw new Error('chi-bridge: WASM memory not set. Call setChiBridgeMemory() first.');
    }

    // Read chi from WASM memory
    const wasmView = new Uint8Array(wasmMemory.buffer);
    const chi = wasmView.slice(chiPtr, chiPtr + 16);

    // Write chi to signal buffer at offset 16
    const chiSignalView = new Uint8Array(signalBuffer, 16, 16);
    chiSignalView.set(chi);

    // Write count and resultPtr
    Atomics.store(signalView, 1, count);
    Atomics.store(signalView, 2, resultPtr);

    // Set status = 1 (request pending) - this triggers the monitor
    Atomics.store(signalView, 0, 1);

    // Block until result is ready (status changes from 1)
    // This is allowed because we're in a web-spawn worker, not main thread
    const result = Atomics.wait(signalView, 0, 1);

    // Check final status
    const finalStatus = Atomics.load(signalView, 0);

    if (finalStatus === 3) {
        throw new Error('chi-bridge: Chi computation failed (status=error)');
    }

    if (finalStatus !== 2) {
        throw new Error(`chi-bridge: Unexpected status after wait: ${finalStatus} (wait result: ${result})`);
    }

    // Reset status to idle for next request
    Atomics.store(signalView, 0, 0);

    // Result has been written to resultPtr by the monitor - nothing to return
}
