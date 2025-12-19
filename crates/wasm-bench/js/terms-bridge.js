// Terms Bridge - Provides request_terms_computation import for WASM
//
// This module is imported by the WASM code (via wasm-bindgen raw_module).
// It runs in web-spawn worker context where Atomics.wait() is allowed.
//
// Note: This file is copied to pkg/ during build, so paths are relative to pkg/.
//
// Multi-slot signal buffer layout (uses WASM memory at TERMS_SIGNAL_OFFSET):
// Each slot is 32 bytes:
//   Int32[0]: status (0=idle, 1=pending, 2=ready, 3=error)
//   Int32[1]: triples_ptr (pointer to triples in WASM memory)
//   Int32[2]: chis_ptr (pointer to chis in WASM memory)
//   Int32[3]: count (number of triples)
//   Int32[4]: result_ptr (where to write 32-byte result)
//   Int32[5..7]: reserved
//
// Multiple slots allow concurrent requests from different workers.

// Signal buffer offset - different from chi to avoid conflicts
// Chi uses 1MB, we use 1.5MB
const TERMS_SIGNAL_OFFSET = 1572864; // 1.5MB
const NUM_SLOTS = 32;
const SLOT_SIZE = 32; // bytes per slot

// Export constants for monitor
export { TERMS_SIGNAL_OFFSET, NUM_SLOTS, SLOT_SIZE };

// Request terms computation from JS coordinator.
// This function BLOCKS via Atomics.wait() until result is ready.
//
// Parameters:
//   memory - WASM memory object (passed from Rust via wasm_bindgen::memory())
//   triples_ptr - pointer to triples in WASM memory (48 bytes per triple)
//   chis_ptr - pointer to chis in WASM memory (16 bytes per chi)
//   count - number of triples
//   result_ptr - where to write 32-byte result (u: 16 bytes, v: 16 bytes)
export function request_terms_computation(memory, triples_ptr, chis_ptr, count, result_ptr) {
    const mem = memory;

    // Claim a free slot using atomic compare-exchange
    let slotIndex = -1;
    for (let attempt = 0; attempt < 1000; attempt++) {
        for (let i = 0; i < NUM_SLOTS; i++) {
            const slotOffset = TERMS_SIGNAL_OFFSET + i * SLOT_SIZE;
            const slotView = new Int32Array(mem.buffer, slotOffset, 8);

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
        throw new Error('terms-bridge: Could not claim a signal slot (all busy)');
    }

    const slotOffset = TERMS_SIGNAL_OFFSET + slotIndex * SLOT_SIZE;
    const signalView = new Int32Array(mem.buffer, slotOffset, 8);

    // Write request parameters (status is already 1 from compareExchange)
    Atomics.store(signalView, 1, triples_ptr);
    Atomics.store(signalView, 2, chis_ptr);
    Atomics.store(signalView, 3, count);
    Atomics.store(signalView, 4, result_ptr);

    // Block until result is ready (status changes from 1)
    // This is allowed because we're in a web-spawn worker, not main thread
    const result = Atomics.wait(signalView, 0, 1);

    // Check final status
    const finalStatus = Atomics.load(signalView, 0);

    if (finalStatus === 3) {
        // Reset status to idle before throwing
        Atomics.store(signalView, 0, 0);
        throw new Error('terms-bridge: Terms computation failed (status=error)');
    }

    if (finalStatus !== 2) {
        // Reset status to idle before throwing
        Atomics.store(signalView, 0, 0);
        throw new Error(`terms-bridge: Unexpected status after wait: ${finalStatus} (wait result: ${result})`);
    }

    // Reset status to idle for next request
    Atomics.store(signalView, 0, 0);

    // Result has been written to result_ptr by the monitor - nothing to return
}
