// Terms computation worker with private memory
// Each worker loads terms-wasm (built WITHOUT atomics) for compute_terms
// Computes: u = x.gfmul(y).gfmul(chi), v = (a_10 ^ a_11 ^ z).gfmul(chi)

let wasmModule = null;

// Forward logs to main thread (only used for errors)
function log(...args) {
    const msg = args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' ');
    self.postMessage({ type: 'log', message: msg });
}

// Initialize WASM module
async function initWasm(wasmUrl) {
    if (wasmModule) return;

    try {
        wasmModule = await import(wasmUrl);
        await wasmModule.default();
    } catch (err) {
        log('[terms-worker] Failed to init WASM:', err.message);
        throw err;
    }
}

// Handle messages from main thread
self.onmessage = async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            try {
                await initWasm(data.wasmUrl);
                self.postMessage({ type: 'ready' });
            } catch (err) {
                self.postMessage({ type: 'error', error: err.toString() });
            }
            break;

        case 'compute_terms':
            // Compute terms for a batch of triples
            // Input: triples (ArrayBuffer, 48 bytes each), chis (ArrayBuffer, 16 bytes each)
            // Output: partial (u, v) as 32 bytes
            const { triples, chis, requestId } = data;
            try {
                // Wrap transferred ArrayBuffers as Uint8Array views
                const triplesArr = new Uint8Array(triples);
                const chisArr = new Uint8Array(chis);

                const result = wasmModule.compute_terms_batch(triplesArr, chisArr);

                self.postMessage({
                    type: 'terms_result',
                    data: result,
                    requestId
                });
            } catch (err) {
                log('[terms-worker] compute_terms failed:', err.message);
                self.postMessage({ type: 'error', error: err.toString(), requestId });
            }
            break;
    }
};
