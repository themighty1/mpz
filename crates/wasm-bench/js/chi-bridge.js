// Chi Bridge - Provides request_chi_computation import for WASM
//
// This module is imported by the WASM code (via wasm-bindgen raw_module).
// It runs in web-spawn worker context where Atomics.wait() is allowed.
//
// Note: This file is copied to pkg/ during build, so paths are relative to pkg/.
//
// Multi-slot signal buffer layout (uses WASM memory at CHI_SIGNAL_OFFSET):
// Each slot is 32 bytes:
//   Int32[0]: status (0=idle, 1=pending, 2=ready, 3=error)
//   Int32[1]: count
//   Int32[2]: resultPtr
//   Int32[3]: reserved
//   Uint8[16..31]: chi input (16 bytes)
//
// Multiple slots allow concurrent requests from different workers.

// Signal buffer is stored at a fixed offset in WASM shared memory
// This offset should be in a safe region not used by the allocator
// We use 1MB offset (1048576) which should be safe for most cases
const CHI_SIGNAL_OFFSET = 1048576;
const NUM_SLOTS = 32;
const SLOT_SIZE = 32; // bytes per slot

// Export constants for monitor
export { CHI_SIGNAL_OFFSET, NUM_SLOTS, SLOT_SIZE };

// Legacy: module-level memory for setChiBridgeMemory (used by main thread)
let wasmMemory = null;

// Set WASM memory reference (legacy - used by main thread for monitor)
export function setChiBridgeMemory(memory) {
    wasmMemory = memory;
}

// Request chi computation from JS coordinator.
// This function BLOCKS via Atomics.wait() until result is ready.
//
// Parameters:
//   memory - WASM memory object (passed from Rust via wasm_bindgen::memory())
//   chiPtr, count, resultPtr - raw pointers into WASM linear memory
export function request_chi_computation(memory, chiPtr, count, resultPtr) {
    // Memory is passed directly from Rust - works in any context (main thread or workers)
    const mem = memory;
    const wasmView = new Uint8Array(mem.buffer);

    // Claim a free slot using atomic compare-exchange
    let slotIndex = -1;
    for (let attempt = 0; attempt < 1000; attempt++) {
        for (let i = 0; i < NUM_SLOTS; i++) {
            const slotOffset = CHI_SIGNAL_OFFSET + i * SLOT_SIZE;
            const slotView = new Int32Array(mem.buffer, slotOffset, 4);

            // Try to claim slot: change status from 0 (idle) to 1 (pending)
            const oldStatus = Atomics.compareExchange(slotView, 0, 0, 1);
            if (oldStatus === 0) {
                // Successfully claimed slot i
                slotIndex = i;
                break;
            }
        }
        if (slotIndex >= 0) break;
        // All slots busy, spin briefly and retry
    }

    if (slotIndex < 0) {
        throw new Error('chi-bridge: Could not claim a signal slot (all busy)');
    }

    const slotOffset = CHI_SIGNAL_OFFSET + slotIndex * SLOT_SIZE;
    const signalView = new Int32Array(mem.buffer, slotOffset, 4);
    const chiSignalView = new Uint8Array(mem.buffer, slotOffset + 16, 16);

    // Read chi from WASM memory at chiPtr
    const chi = wasmView.slice(chiPtr, chiPtr + 16);

    // Write chi to signal region at offset 16
    chiSignalView.set(chi);

    // Write count and resultPtr (status is already 1 from compareExchange)
    Atomics.store(signalView, 1, count);
    Atomics.store(signalView, 2, resultPtr);

    // Block until result is ready (status changes from 1)
    // This is allowed because we're in a web-spawn worker, not main thread
    const result = Atomics.wait(signalView, 0, 1);

    // Check final status
    const finalStatus = Atomics.load(signalView, 0);

    if (finalStatus === 3) {
        // Reset status to idle before throwing
        Atomics.store(signalView, 0, 0);
        throw new Error('chi-bridge: Chi computation failed (status=error)');
    }

    if (finalStatus !== 2) {
        // Reset status to idle before throwing
        Atomics.store(signalView, 0, 0);
        throw new Error(`chi-bridge: Unexpected status after wait: ${finalStatus} (wait result: ${result})`);
    }

    // Reset status to idle for next request
    Atomics.store(signalView, 0, 0);

    // Result has been written to resultPtr by the monitor - nothing to return
}
