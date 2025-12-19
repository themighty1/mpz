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
            // DUMMY MODE: generate data locally to test without transfer overhead
            const { triples, chis, requestId } = data;
            try {
                // Get count from transferred buffer size
                const count = new Uint8Array(triples).length / 48;

                // Generate dummy data locally (no transfer overhead)
                const triplesArr = new Uint8Array(count * 48);
                const chisArr = new Uint8Array(count * 16);
                for (let i = 0; i < triplesArr.length; i++) triplesArr[i] = i & 0xff;
                for (let i = 0; i < chisArr.length; i++) chisArr[i] = i & 0xff;

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
