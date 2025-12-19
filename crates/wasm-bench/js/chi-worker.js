// Chi computation worker with private memory
// Each worker loads chi-wasm (built WITHOUT atomics) for fast gfmul

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
        log('[chi-worker] Failed to init WASM:', err.message);
        throw err;
    }
}

// Compute chi starting points for parallel computation
// Segment k starts at chi[k * segmentSize] = seed^(2^(k * segmentSize))
// To compute seed^(2^N), do N squarings: seed -> seed^2 -> seed^4 -> ... -> seed^(2^N)
async function computeChiStarts(chi, segmentSize) {
    const PARALLELISM = 16;

    const starts = [];
    let current = new Uint8Array(chi);

    for (let seg = 0; seg < PARALLELISM; seg++) {
        // Current is now at chi[seg * segmentSize] = seed^(2^(seg * segmentSize))
        starts.push(new Uint8Array(current));

        // Advance to next segment start by doing segmentSize squarings
        // chi[(seg+1) * segmentSize] = chi[seg * segmentSize]^(2^segmentSize)
        for (let i = 0; i < segmentSize; i++) {
            current = wasmModule.gfmul(current, current);
        }
    }

    return starts;
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

        case 'compute_segment':
            const { segmentIndex, start, count, requestId } = data;
            try {
                const startArr = new Uint8Array(start);
                const segment = wasmModule.compute_chi_segment(startArr, count);
                self.postMessage({
                    type: 'segment_result',
                    segmentIndex,
                    data: segment,
                    requestId
                });
            } catch (err) {
                log('[chi-worker] compute_segment failed:', err.message);
                self.postMessage({ type: 'error', error: err.toString(), requestId });
            }
            break;

        case 'compute_starts':
            const { chi, segmentSize, requestId: startsRequestId } = data;
            try {
                const starts = await computeChiStarts(new Uint8Array(chi), segmentSize);
                self.postMessage({
                    type: 'starts_result',
                    starts: starts.map(s => Array.from(s)),
                    requestId: startsRequestId
                });
            } catch (err) {
                self.postMessage({ type: 'error', error: err.toString(), requestId: startsRequestId });
            }
            break;
    }
};
