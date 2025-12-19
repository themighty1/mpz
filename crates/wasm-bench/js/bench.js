// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

// Note: chi-bridge.js import is deferred to avoid circular dependency during WASM load
let setChiBridgeMemory = null;

let wasm = null;
let wasmUrl = null;  // URL to WASM module for chi workers
let andGateCount = 0;

// ============================================================================
// Private Memory Worker Pool for Chi Computation
// ============================================================================

const CHI_WORKER_COUNT = 16; // Force 16 workers for max parallelism

// ============================================================================
// Chi Signal Region in WASM Shared Memory
// ============================================================================
// Uses a fixed offset in WASM memory so main thread and web-spawn workers
// can share the same signal buffer (they share the same WASM memory).
//
// Multi-slot layout - each slot is 32 bytes:
//   Int32[0]: status (0=idle, 1=pending, 2=ready, 3=error)
//   Int32[1]: count
//   Int32[2]: resultPtr
//   Int32[3]: reserved
//   Uint8[16..31]: chi input (16 bytes)
//
// Multiple slots allow concurrent requests from different workers.
const CHI_SIGNAL_OFFSET = 1048576; // 1MB offset into WASM memory
const NUM_SLOTS = 32;
const SLOT_SIZE = 32; // bytes per slot
let chiMonitorRunning = false;
let wasmMemoryRef = null; // Reference to shared WASM memory

// ============================================================================
// Terms Signal Region in WASM Shared Memory
// ============================================================================
// Similar to chi signal region but for compute_terms requests.
// Uses different offset (1.5MB) to avoid conflicts with chi region.
//
// Multi-slot layout - each slot is 32 bytes:
//   Int32[0]: status (0=idle, 1=pending, 2=ready, 3=error)
//   Int32[1]: triples_ptr (pointer to triples in WASM memory)
//   Int32[2]: chis_ptr (pointer to chis in WASM memory)
//   Int32[3]: count (number of triples)
//   Int32[4]: result_ptr (where to write 32-byte result)
//   Int32[5..7]: reserved
const TERMS_SIGNAL_OFFSET = 1572864; // 1.5MB offset
const TERMS_WORKER_COUNT = 4;
let termsMonitorRunning = false;

// Set reference to shared WASM memory (must be called before monitor can write results)
export function setWasmMemory(memory) {
    wasmMemoryRef = memory;
    console.log('[bench.js] WASM memory reference set');
}

// Legacy export for compatibility (now uses WASM memory instead)
export function getChiSignalBuffer() {
    if (!wasmMemoryRef) {
        throw new Error('WASM memory not set. Call setWasmMemory() first.');
    }
    // Return a view of WASM memory at the signal offset
    return wasmMemoryRef.buffer;
}

// Start the chi request monitor (runs in main thread event loop)
// This watches for requests from web-spawn workers and dispatches to chi pool
export async function startChiRequestMonitor() {
    if (chiMonitorRunning) return;

    if (!wasmMemoryRef) {
        throw new Error('WASM memory not set. Call setWasmMemory() first.');
    }

    chiMonitorRunning = true;

    if (!chiWorkersReady) {
        await initChiWorkerPool();
    }

    // Initialize all slots to idle
    for (let i = 0; i < NUM_SLOTS; i++) {
        const slotOffset = CHI_SIGNAL_OFFSET + i * SLOT_SIZE;
        const slotView = new Int32Array(wasmMemoryRef.buffer, slotOffset, 4);
        Atomics.store(slotView, 0, 0);
    }

    console.log(`[bench.js] Chi request monitor started (${NUM_SLOTS} slots at offset ${CHI_SIGNAL_OFFSET})`);

    // Track in-flight requests per slot to avoid double-processing
    const processingSlots = new Set();

    // Use setInterval to check for requests (can't block main thread)
    const checkInterval = setInterval(() => {
        for (let i = 0; i < NUM_SLOTS; i++) {
            if (processingSlots.has(i)) continue; // Skip slots being processed

            const slotOffset = CHI_SIGNAL_OFFSET + i * SLOT_SIZE;
            const slotView = new Int32Array(wasmMemoryRef.buffer, slotOffset, 4);
            const status = Atomics.load(slotView, 0);

            if (status === 1) { // request_pending
                processingSlots.add(i);

                const count = Atomics.load(slotView, 1);
                const resultPtr = Atomics.load(slotView, 2);

                // Read chi input from slot
                const chiInput = new Uint8Array(wasmMemoryRef.buffer, slotOffset + 16, 16);
                const chi = new Uint8Array(chiInput); // Copy

                // Process asynchronously
                (async () => {
                    try {
                        // Dispatch to chi worker pool
                        const result = await computeChisWithWorkerPool(chi, count);

                        // Write result to shared WASM memory at resultPtr
                        const wasmView = new Uint8Array(wasmMemoryRef.buffer);
                        wasmView.set(result, resultPtr);

                        // Signal completion
                        Atomics.store(slotView, 0, 2); // result_ready
                        Atomics.notify(slotView, 0);
                    } catch (err) {
                        console.error(`[chi-monitor] Slot ${i} error:`, err);
                        Atomics.store(slotView, 0, 3); // error
                        Atomics.notify(slotView, 0);
                    } finally {
                        processingSlots.delete(i);
                    }
                })();
            }
        }
    }, 1); // Check every 1ms

    // Return cleanup function
    return () => {
        clearInterval(checkInterval);
        chiMonitorRunning = false;
        console.log('[bench.js] Chi request monitor stopped');
    };
}

// Test the chi signal infrastructure without Rust
// Tested 2025-12-19 18:40: `await bench.testChiSignalInfra(10000)` passed in browser console
export async function testChiSignalInfra(count = 1000) {
    console.log(`[test] Testing chi signal infra with count=${count}`);

    // 0. Set wasmUrl for chi workers if not already set
    if (!wasmUrl) {
        wasmUrl = new URL('../pkg-chi/chi_wasm.js', import.meta.url).href;
        console.log(`[test] Set wasmUrl to: ${wasmUrl}`);
    }

    // 1. Create fake "WASM memory" (must be large enough to include signal region at 1MB + result data)
    const fakeWasmMemorySize = CHI_SIGNAL_OFFSET + 32 + count * 16 + 1024;
    const fakeWasmMemory = new SharedArrayBuffer(fakeWasmMemorySize);
    setWasmMemory({ buffer: fakeWasmMemory });

    // 2. Start monitor (uses WASM memory at CHI_SIGNAL_OFFSET)
    await startChiRequestMonitor();

    // 3. Create views on signal region
    const signalView = new Int32Array(fakeWasmMemory, CHI_SIGNAL_OFFSET, 4);
    const chiView = new Uint8Array(fakeWasmMemory, CHI_SIGNAL_OFFSET + 16, 16);

    // 4. Write request to signal buffer
    const chi = new Uint8Array(16);
    crypto.getRandomValues(chi);
    const resultPtr = CHI_SIGNAL_OFFSET + 32; // Write result after signal region

    // Write chi to signal region at offset 16
    chiView.set(chi);

    // Write count and resultPtr
    Atomics.store(signalView, 1, count);
    Atomics.store(signalView, 2, resultPtr);

    // 5. Set status=1 (request pending)
    const startTime = performance.now();
    Atomics.store(signalView, 0, 1);

    // 6. Poll for completion (can't use Atomics.wait on main thread without blocking)
    await new Promise((resolve, reject) => {
        const pollInterval = setInterval(() => {
            const status = Atomics.load(signalView, 0);
            if (status === 2) { // result_ready
                clearInterval(pollInterval);
                resolve();
            } else if (status === 3) { // error
                clearInterval(pollInterval);
                reject(new Error('Chi computation failed'));
            }
        }, 1);

        // Timeout after 30s
        setTimeout(() => {
            clearInterval(pollInterval);
            reject(new Error('Timeout waiting for chi result'));
        }, 30000);
    });

    const elapsed = performance.now() - startTime;

    // 7. Verify result
    const resultView = new Uint8Array(fakeWasmMemory, resultPtr, count * 16);
    const nonZeroBytes = Array.from(resultView).filter(b => b !== 0).length;

    console.log(`[test] Chi signal infra test passed!`);
    console.log(`[test]   count: ${count}`);
    console.log(`[test]   elapsed: ${elapsed.toFixed(2)}ms`);
    console.log(`[test]   result bytes: ${resultView.length}`);
    console.log(`[test]   non-zero bytes: ${nonZeroBytes}`);

    // Return BenchResult format
    return {
        elapsed_ms: elapsed,
        and_gates: BigInt(count) // Use count as "gates" for throughput calc
    };
}

let chiWorkers = [];
let chiWorkersReady = false;
let pendingChiRequests = new Map();
let chiRequestId = 0;

// Initialize the chi worker pool
export async function initChiWorkerPool(customWasmUrl = null) {
    if (customWasmUrl) wasmUrl = customWasmUrl;
    if (chiWorkersReady) return;
    if (!wasmUrl) throw new Error("WASM URL not set. Call init() first.");

    console.log(`[initChiWorkerPool] Starting with ${CHI_WORKER_COUNT} workers`);
    console.log(`[initChiWorkerPool] wasmUrl: ${wasmUrl}`);

    // Load chi-wasm module in main thread for compute_chi_starts
    if (!chiWasm) {
        console.log(`[initChiWorkerPool] Loading chi-wasm module...`);
        chiWasm = await import(wasmUrl);
        await chiWasm.default(); // Initialize WASM
        console.log(`[initChiWorkerPool] chi-wasm module loaded`);
    }

    const workerUrl = new URL('./chi-worker.js', import.meta.url);
    console.log(`[initChiWorkerPool] workerUrl: ${workerUrl}`);

    const initPromises = [];
    for (let i = 0; i < CHI_WORKER_COUNT; i++) {
        console.log(`[initChiWorkerPool] Creating worker ${i}`);
        const worker = new Worker(workerUrl, { type: 'module' });

        const readyPromise = new Promise((resolve, reject) => {
            // Timeout after 30 seconds
            const timeout = setTimeout(() => {
                reject(new Error(`Worker ${i} timed out after 30s`));
            }, 30000);

            worker.onmessage = (e) => {
                if (e.data.type === 'log') {
                    // Forward worker logs to main console
                    console.log(`[worker ${i}]`, e.data.message);
                } else if (e.data.type === 'ready') {
                    clearTimeout(timeout);
                    console.log(`[initChiWorkerPool] Worker ${i} ready`);
                    resolve();
                } else if (e.data.type === 'error') {
                    clearTimeout(timeout);
                    console.error(`[chi-worker ${i}] Error:`, e.data.error);
                    reject(new Error(e.data.error));
                } else if (e.data.type === 'segment_result') {
                    const pending = pendingChiRequests.get(e.data.requestId);
                    if (pending) {
                        pending.segments[e.data.segmentIndex] = e.data.data;
                        pending.completed++;
                        if (pending.completed === pending.total) {
                            pendingChiRequests.delete(e.data.requestId);
                            pending.resolve(pending.segments);
                        }
                    }
                } else if (e.data.type === 'starts_result') {
                    const pending = pendingChiRequests.get(e.data.requestId);
                    if (pending) {
                        pendingChiRequests.delete(e.data.requestId);
                        pending.resolve(e.data.starts);
                    }
                }
            };

            worker.onerror = (e) => {
                clearTimeout(timeout);
                console.error(`[chi-worker ${i}] Worker error:`, e.message);
                reject(new Error(`Worker ${i} error: ${e.message}`));
            };
        });

        // Pass the WASM URL so worker can load its own instance
        worker.postMessage({ type: 'init', data: { wasmUrl } });
        chiWorkers.push(worker);
        initPromises.push(readyPromise);
    }

    console.log(`[initChiWorkerPool] Waiting for all workers...`);
    await Promise.all(initPromises);
    chiWorkersReady = true;
    console.log(`[initChiWorkerPool] All ${CHI_WORKER_COUNT} workers ready`);
}

// Reference to chi-wasm module (loaded during initChiWorkerPool)
let chiWasm = null;

// Dispatch chi computation to workers
export async function computeChisWithWorkerPool(chi, count) {
    if (!chiWorkersReady) {
        await initChiWorkerPool();
    }

    if (count === 0) {
        return new Uint8Array(0);
    }

    const PARALLELISM = 16;
    const segmentSize = Math.ceil(count / PARALLELISM);

    // Compute starting points using WASM (O(1) - 16 squarings + hash)
    const startsBytes = chiWasm.compute_chi_starts(chi, segmentSize);
    // startsBytes is 256 bytes (16 blocks of 16 bytes each)

    // Dispatch segments to workers
    const requestId = chiRequestId++;
    const segmentPromise = new Promise((resolve) => {
        pendingChiRequests.set(requestId, {
            resolve,
            segments: new Array(PARALLELISM),
            completed: 0,
            total: PARALLELISM
        });
    });

    let dispatchedCount = 0;
    for (let i = 0; i < PARALLELISM; i++) {
        const segStart = i * segmentSize;
        const segEnd = Math.min((i + 1) * segmentSize, count);
        const segLen = segEnd - segStart;

        // Skip segments with no work (can happen when count < PARALLELISM * segmentSize)
        if (segLen <= 0) {
            pendingChiRequests.get(requestId).segments[i] = new Uint8Array(0);
            pendingChiRequests.get(requestId).completed++;
            continue;
        }

        dispatchedCount++;
        const workerIdx = i % chiWorkers.length;
        // Extract 16-byte starting point for this segment
        const start = startsBytes.slice(i * 16, (i + 1) * 16);
        chiWorkers[workerIdx].postMessage({
            type: 'compute_segment',
            data: {
                segmentIndex: i,
                start,
                count: segLen,
                requestId
            }
        });
    }

    // If no segments were dispatched (count was 0), resolve immediately
    if (dispatchedCount === 0) {
        pendingChiRequests.delete(requestId);
        return new Uint8Array(0);
    }

    const segments = await segmentPromise;

    // Concatenate segments (segments are ArrayBuffers from transfer)
    const result = new Uint8Array(count * 16);
    let offset = 0;
    for (let i = 0; i < PARALLELISM; i++) {
        if (segments[i]) {
            const segArr = new Uint8Array(segments[i]);
            if (segArr.length > 0) {
                result.set(segArr, offset);
                offset += segArr.length;
            }
        }
    }

    return result;
}

// Benchmark: chi computation with private memory worker pool
async function benchChiWorkerPool(gateCount) {
    if (!chiWorkersReady) {
        await initChiWorkerPool();
    }

    const chi = new Uint8Array(16);
    crypto.getRandomValues(chi);

    const start = performance.now();
    const result = await computeChisWithWorkerPool(chi, gateCount);
    const elapsed = performance.now() - start;

    return {
        elapsed_ms: elapsed,
        and_gates: BigInt(gateCount),
        result_bytes: result.length
    };
}

// Benchmark: chi computation synchronous (main thread, using WASM)
function benchChiSync(gateCount) {
    const chi = new Uint8Array(16);
    crypto.getRandomValues(chi);

    const start = performance.now();
    // Use WASM compute_chi_segment for fair comparison
    const result = wasm.compute_chi_segment(chi, gateCount);
    const elapsed = performance.now() - start;

    return {
        elapsed_ms: elapsed,
        and_gates: BigInt(gateCount),
        result_bytes: result.length
    };
}

// ============================================================================
// Private Memory Worker Pool for Terms Computation
// ============================================================================

let termsWorkers = [];
let termsWorkersReady = false;
let pendingTermsRequests = new Map();
let termsRequestId = 0;
let termsWasmUrl = null;

// Initialize the terms worker pool
export async function initTermsWorkerPool(customWasmUrl = null) {
    if (customWasmUrl) termsWasmUrl = customWasmUrl;
    if (termsWorkersReady) return;
    if (!termsWasmUrl) {
        termsWasmUrl = new URL('../pkg-terms/terms_wasm.js', import.meta.url).href;
    }

    console.log(`[initTermsWorkerPool] Starting with ${TERMS_WORKER_COUNT} workers`);
    console.log(`[initTermsWorkerPool] termsWasmUrl: ${termsWasmUrl}`);

    const workerUrl = new URL('./terms-worker.js', import.meta.url);
    console.log(`[initTermsWorkerPool] workerUrl: ${workerUrl}`);

    const initPromises = [];
    for (let i = 0; i < TERMS_WORKER_COUNT; i++) {
        console.log(`[initTermsWorkerPool] Creating worker ${i}`);
        const worker = new Worker(workerUrl, { type: 'module' });

        const readyPromise = new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Terms worker ${i} timed out after 30s`));
            }, 30000);

            worker.onmessage = (e) => {
                if (e.data.type === 'log') {
                    console.log(`[terms-worker ${i}]`, e.data.message);
                } else if (e.data.type === 'ready') {
                    clearTimeout(timeout);
                    console.log(`[initTermsWorkerPool] Worker ${i} ready`);
                    resolve();
                } else if (e.data.type === 'error') {
                    clearTimeout(timeout);
                    console.error(`[terms-worker ${i}] Error:`, e.data.error);
                    if (e.data.requestId !== undefined) {
                        const pending = pendingTermsRequests.get(e.data.requestId);
                        if (pending) {
                            pendingTermsRequests.delete(e.data.requestId);
                            pending.reject(new Error(e.data.error));
                        }
                    } else {
                        reject(new Error(e.data.error));
                    }
                } else if (e.data.type === 'terms_result') {
                    const receiveTime = performance.now();
                    const pending = pendingTermsRequests.get(e.data.requestId);
                    if (pending) {
                        pending.results.push(e.data.data);
                        const roundTrip = receiveTime - e.data.dispatchTime;
                        pending.workerTimes.push({
                            workerId: e.data.workerIndex,
                            elapsedMs: e.data.elapsedMs,
                            tripleCount: e.data.tripleCount,
                            roundTripMs: roundTrip,
                            overheadMs: roundTrip - e.data.elapsedMs
                        });
                        pending.completed++;
                        if (pending.completed === pending.total) {
                            pendingTermsRequests.delete(e.data.requestId);
                            pending.resolve({ results: pending.results, workerTimes: pending.workerTimes });
                        }
                    }
                }
            };

            worker.onerror = (e) => {
                clearTimeout(timeout);
                console.error(`[terms-worker ${i}] Worker error:`, e.message);
                reject(new Error(`Terms worker ${i} error: ${e.message}`));
            };
        });

        worker.postMessage({ type: 'init', data: { wasmUrl: termsWasmUrl } });
        termsWorkers.push(worker);
        initPromises.push(readyPromise);
    }

    console.log(`[initTermsWorkerPool] Waiting for all workers...`);
    await Promise.all(initPromises);
    termsWorkersReady = true;
    console.log(`[initTermsWorkerPool] All ${TERMS_WORKER_COUNT} workers ready`);
}

// Dispatch terms computation to workers and combine results
// triples: Uint8Array (48 bytes per triple: x, y, z)
// chis: Uint8Array (16 bytes per chi)
// Returns: Uint8Array (32 bytes: u, v)
async function computeTermsWithWorkerPool(triples, chis, count) {
    if (!termsWorkersReady) {
        await initTermsWorkerPool();
    }

    if (count === 0) {
        return new Uint8Array(32);
    }

    // Split work across workers
    const workersToUse = Math.min(TERMS_WORKER_COUNT, count);
    const baseSegment = Math.floor(count / workersToUse);
    const remainder = count % workersToUse;

    const requestId = termsRequestId++;
    const dispatchStartTime = performance.now();
    const resultPromise = new Promise((resolve, reject) => {
        pendingTermsRequests.set(requestId, {
            resolve,
            reject,
            results: [],
            workerTimes: [],
            dispatchTimes: [],
            completed: 0,
            total: workersToUse
        });
    });

    let totalSliceTime = 0;
    let offset = 0;
    for (let i = 0; i < workersToUse; i++) {
        // Distribute remainder evenly
        const segmentCount = baseSegment + (i < remainder ? 1 : 0);
        if (segmentCount === 0) continue;

        // Extract segment data
        const tripleStart = offset * 48;
        const tripleEnd = (offset + segmentCount) * 48;
        const chiStart = offset * 16;
        const chiEnd = (offset + segmentCount) * 16;

        const sliceStart = performance.now();
        const segTriples = triples.slice(tripleStart, tripleEnd);
        const segChis = chis.slice(chiStart, chiEnd);
        totalSliceTime += performance.now() - sliceStart;

        const workerDispatchTime = performance.now();
        termsWorkers[i].postMessage({
            type: 'compute_terms',
            data: {
                triples: segTriples.buffer,
                chis: segChis.buffer,
                requestId,
                workerIndex: i,
                dispatchTime: workerDispatchTime
            }
        }, [segTriples.buffer, segChis.buffer]);

        offset += segmentCount;
    }

    // Wait for all workers to complete
    const { results, workerTimes } = await resultPromise;
    const totalWallTime = performance.now() - dispatchStartTime;

    // Print timing stats
    const totalWorkerTime = workerTimes.reduce((sum, w) => sum + w.elapsedMs, 0);
    const maxWorkerTime = Math.max(...workerTimes.map(w => w.elapsedMs));
    const minWorkerTime = Math.min(...workerTimes.map(w => w.elapsedMs));
    const totalTriples = workerTimes.reduce((sum, w) => sum + w.tripleCount, 0);

    const maxRoundTrip = Math.max(...workerTimes.map(w => w.roundTripMs));
    const minRoundTrip = Math.min(...workerTimes.map(w => w.roundTripMs));
    const avgRoundTrip = workerTimes.reduce((sum, w) => sum + w.roundTripMs, 0) / workerTimes.length;
    const avgOverhead = workerTimes.reduce((sum, w) => sum + w.overheadMs, 0) / workerTimes.length;
    const maxOverhead = Math.max(...workerTimes.map(w => w.overheadMs));

    console.log(`[terms-pool] ${count} triples, ${workersToUse} workers:`);
    console.log(`  Wall time: ${totalWallTime.toFixed(2)}ms, Slice time: ${totalSliceTime.toFixed(2)}ms`);
    console.log(`  Compute: min=${minWorkerTime.toFixed(2)}ms, max=${maxWorkerTime.toFixed(2)}ms`);
    console.log(`  RoundTrip: min=${minRoundTrip.toFixed(2)}ms, max=${maxRoundTrip.toFixed(2)}ms, avg=${avgRoundTrip.toFixed(2)}ms`);
    console.log(`  Per-worker overhead (roundtrip - compute): avg=${avgOverhead.toFixed(2)}ms, max=${maxOverhead.toFixed(2)}ms`);
    console.log(`  Throughput: ${(totalTriples / maxWorkerTime * 1000 / 1e6).toFixed(2)}M triples/s`);

    // XOR all partial results together
    const finalResult = new Uint8Array(32);
    for (const partial of results) {
        const partialArr = new Uint8Array(partial);
        for (let i = 0; i < 32; i++) {
            finalResult[i] ^= partialArr[i];
        }
    }

    return finalResult;
}

// Start the terms request monitor (runs in main thread event loop)
// This watches for requests from web-spawn workers and dispatches to terms pool
export async function startTermsRequestMonitor() {
    if (termsMonitorRunning) return;

    if (!wasmMemoryRef) {
        throw new Error('WASM memory not set. Call setWasmMemory() first.');
    }

    termsMonitorRunning = true;

    if (!termsWorkersReady) {
        await initTermsWorkerPool();
    }

    // Initialize all slots to idle
    for (let i = 0; i < NUM_SLOTS; i++) {
        const slotOffset = TERMS_SIGNAL_OFFSET + i * SLOT_SIZE;
        const slotView = new Int32Array(wasmMemoryRef.buffer, slotOffset, 8);
        Atomics.store(slotView, 0, 0);
    }

    console.log(`[bench.js] Terms request monitor started (${NUM_SLOTS} slots at offset ${TERMS_SIGNAL_OFFSET})`);

    // Track in-flight requests per slot to avoid double-processing
    const processingSlots = new Set();

    // Use setInterval to check for requests (can't block main thread)
    const checkInterval = setInterval(() => {
        for (let i = 0; i < NUM_SLOTS; i++) {
            if (processingSlots.has(i)) continue;

            const slotOffset = TERMS_SIGNAL_OFFSET + i * SLOT_SIZE;
            const slotView = new Int32Array(wasmMemoryRef.buffer, slotOffset, 8);
            const status = Atomics.load(slotView, 0);

            if (status === 1) { // request_pending
                processingSlots.add(i);

                const triplesPtr = Atomics.load(slotView, 1);
                const chisPtr = Atomics.load(slotView, 2);
                const count = Atomics.load(slotView, 3);
                const resultPtr = Atomics.load(slotView, 4);

                // Read triples and chis from WASM memory
                const triples = new Uint8Array(wasmMemoryRef.buffer, triplesPtr, count * 48);
                const chis = new Uint8Array(wasmMemoryRef.buffer, chisPtr, count * 16);

                // Copy data since we're passing to workers
                const triplesCopy = new Uint8Array(triples);
                const chisCopy = new Uint8Array(chis);

                // Process asynchronously
                const monitorSeenTime = performance.now();
                (async () => {
                    try {
                        const beforeCompute = performance.now();
                        const result = await computeTermsWithWorkerPool(triplesCopy, chisCopy, count);
                        const afterCompute = performance.now();

                        // Write result to shared WASM memory at resultPtr
                        const wasmView = new Uint8Array(wasmMemoryRef.buffer);
                        wasmView.set(result, resultPtr);

                        const beforeNotify = performance.now();
                        console.log(`[terms-monitor] Timing breakdown:`);
                        console.log(`  Copy time: ${(beforeCompute - monitorSeenTime).toFixed(2)}ms`);
                        console.log(`  computeTermsWithWorkerPool: ${(afterCompute - beforeCompute).toFixed(2)}ms`);
                        console.log(`  Write result: ${(beforeNotify - afterCompute).toFixed(2)}ms`);

                        // Signal completion
                        Atomics.store(slotView, 0, 2); // result_ready
                        Atomics.notify(slotView, 0);
                    } catch (err) {
                        console.error(`[terms-monitor] Slot ${i} error:`, err);
                        Atomics.store(slotView, 0, 3); // error
                        Atomics.notify(slotView, 0);
                    } finally {
                        processingSlots.delete(i);
                    }
                })();
            }
        }
    }, 1); // Check every 1ms

    // Return cleanup function
    return () => {
        clearInterval(checkInterval);
        termsMonitorRunning = false;
        console.log('[bench.js] Terms request monitor stopped');
    };
}

// ============================================================================
// End Worker Pool Section
// ============================================================================

// Initialize WASM module
export async function init(wasmModule, wasmModuleUrl = null) {
    wasm = wasmModule;
    // Chi workers use chi-wasm (pkg-chi/) which is built WITHOUT atomics
    // This allows workers to have private memory (no SharedArrayBuffer)
    wasmUrl = wasmModuleUrl || new URL('../pkg-chi/chi_wasm.js', import.meta.url).href;
    console.log('[bench.js] chi wasmUrl set to:', wasmUrl);
    andGateCount = wasm.garble_core_aes128_and_count();

    // Initialize chi-bridge with WASM memory (deferred import to avoid circular dep)
    // This enables request_chi_computation to access WASM linear memory
    const wasmMemory = wasm.get_wasm_memory ? wasm.get_wasm_memory() : wasm.memory;
    if (wasmMemory) {
        // Set WASM memory reference for the monitor (so it can read signal region and write results)
        setWasmMemory(wasmMemory);

        try {
            const chiBridge = await import('./chi-bridge.js');
            setChiBridgeMemory = chiBridge.setChiBridgeMemory;
            setChiBridgeMemory(wasmMemory);
            console.log('[bench.js] Chi bridge initialized with WASM memory');

            // Start chi request monitor for chi_pool feature
            // This allows web-spawn workers to request chi computation from chi worker pool
            await startChiRequestMonitor();
            console.log('[bench.js] Chi infrastructure ready');
        } catch (e) {
            console.warn('[bench.js] Chi bridge not available:', e.message);
        }

        // Initialize terms-bridge for terms_pool feature
        try {
            const termsBridge = await import('./terms-bridge.js');
            // terms-bridge doesn't need memory set - it receives memory as parameter
            console.log('[bench.js] Terms bridge module loaded');

            // Start terms request monitor for terms_pool feature
            // This allows web-spawn workers to request terms computation from terms worker pool
            await startTermsRequestMonitor();
            console.log('[bench.js] Terms infrastructure ready');
        } catch (e) {
            console.warn('[bench.js] Terms bridge not available:', e.message);
        }
    }
}

// Progress callback (set by runner)
let progressCallback = null;

export function setProgressCallback(cb) {
    progressCallback = cb;
}

function reportProgress(message) {
    if (progressCallback) progressCallback(message);
    window.__benchProgress = message;
}

// Run a single benchmark with warmup and multiple samples (sync version)
function runBenchSync(name, fn, iterations, samples = 10, warmupSamples = 3) {
    reportProgress(`Running: ${name} (warmup ${warmupSamples} runs)...`);

    // Warmup runs
    for (let i = 0; i < warmupSamples; i++) {
        fn(iterations);
    }

    reportProgress(`Running: ${name} (0/${samples} samples)...`);

    // Timed runs
    const times = [];
    for (let i = 0; i < samples; i++) {
        const start = performance.now();
        fn(iterations);
        const elapsed = performance.now() - start;
        times.push(elapsed);

        // Estimate remaining time
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const remaining = avgTime * (samples - i - 1);
        const remainingSec = (remaining / 1000).toFixed(1);
        reportProgress(`Running: ${name} (${i + 1}/${samples} samples, ~${remainingSec}s remaining)...`);
    }

    return calcStats(name, iterations, samples, times);
}

// Run a single benchmark with warmup and multiple samples (async version)
async function runBenchAsync(name, fn, iterations, samples = 10, warmupSamples = 3) {
    reportProgress(`Running: ${name} (warmup ${warmupSamples} runs)...`);

    // Warmup runs
    for (let i = 0; i < warmupSamples; i++) {
        await fn(iterations);
    }

    reportProgress(`Running: ${name} (0/${samples} samples)...`);

    // Timed runs
    const times = [];
    for (let i = 0; i < samples; i++) {
        const start = performance.now();
        await fn(iterations);
        const elapsed = performance.now() - start;
        times.push(elapsed);

        // Estimate remaining time
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const remaining = avgTime * (samples - i - 1);
        const remainingSec = (remaining / 1000).toFixed(1);
        reportProgress(`Running: ${name} (${i + 1}/${samples} samples, ~${remainingSec}s remaining)...`);
    }

    return calcStats(name, iterations, samples, times);
}

// Run benchmark that returns BenchResult { elapsed_ms, and_gates }
async function runBenchWithResult(name, fn, iterations, samples = 10, warmupSamples = 3) {
    reportProgress(`Running: ${name} (warmup ${warmupSamples} runs)...`);

    // Warmup runs
    for (let i = 0; i < warmupSamples; i++) {
        await fn(iterations);
    }

    reportProgress(`Running: ${name} (0/${samples} samples)...`);

    // Timed runs - function returns { elapsed_ms, and_gates }
    const times = [];
    let totalAndGates = 0;
    for (let i = 0; i < samples; i++) {
        const result = await fn(iterations);
        times.push(result.elapsed_ms);
        totalAndGates = Number(result.and_gates); // Same for all samples, convert BigInt if needed

        // Estimate remaining time
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const remaining = avgTime * (samples - i - 1);
        const remainingSec = (remaining / 1000).toFixed(1);
        reportProgress(`Running: ${name} (${i + 1}/${samples} samples, ~${remainingSec}s remaining)...`);
    }

    return calcStatsFromResult(name, iterations, samples, times, totalAndGates);
}

// Calculate statistics from BenchResult (elapsed_ms, and_gates)
function calcStatsFromResult(name, iterations, samples, times, andGates) {
    times.sort((a, b) => a - b);
    const min = times[0];
    const max = times[times.length - 1];
    const median = times[Math.floor(times.length / 2)];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;

    // Throughput: AND gates per second
    const andGatesPerSec = (andGates * 1000) / mean;

    return {
        name,
        iterations,
        samples,
        min_ms: min,
        max_ms: max,
        median_ms: median,
        mean_ms: mean,
        per_iter_ms: mean / iterations,
        per_iter_us: (mean / iterations) * 1000,
        throughput: andGatesPerSec,
    };
}

// Calculate statistics from timing data
function calcStats(name, iterations, samples, times, circuitsPerIter = 1) {
    times.sort((a, b) => a - b);
    const min = times[0];
    const max = times[times.length - 1];
    const median = times[Math.floor(times.length / 2)];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const totalCircuits = iterations * circuitsPerIter;
    const perIter = mean / totalCircuits;

    // Throughput in AND gates per second
    // perIter is ms per AES circuit, each circuit has andGateCount AND gates
    const andGatesPerSec = (andGateCount * 1000) / perIter;

    return {
        name,
        iterations,
        samples,
        min_ms: min,
        max_ms: max,
        median_ms: median,
        mean_ms: mean,
        per_iter_ms: perIter,
        per_iter_us: perIter * 1000,
        throughput: andGatesPerSec, // AND gates per second
    };
}

// Define all benchmarks with their categories
// concurrency is passed to MT benchmarks to control thread count
function getAllBenchmarkDefs(concurrency = 8) {
    return [
        // garbler_core benchmarks (raw garbling primitives)
        { category: "garbler_core", name: "garbler_core/half_gates", fn: (n) => wasm.garble_core_half_gates_garble(n), async: false },
        // evaluator_core benchmarks (raw evaluation primitives)
        { category: "evaluator_core", name: "evaluator_core/half_gates", fn: (n) => wasm.garble_core_half_gates_evaluate(n), async: false, returnsBenchResult: true },
        { category: "evaluator_core", name: "evaluator_core/half_gates_batched", fn: (n) => wasm.garble_core_half_gates_evaluate_batched(n), async: false, returnsBenchResult: true },
        { category: "evaluator_core", name: "evaluator_core/half_gates_parallel", fn: (n) => wasm.garble_core_half_gates_evaluate_parallel(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        // zk_prover_core benchmarks (QuickSilver ZK prover primitives)
        { category: "zk_prover_core", name: "zk_prover_core/execute", fn: (n) => wasm.zk_core_prover_execute(n), async: false, returnsBenchResult: true },
        { category: "zk_prover_core", name: "zk_prover_core/check_200k", fn: (n) => wasm.zk_core_prover_check_200k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_prover_core", name: "zk_prover_core/check_400k", fn: (n) => wasm.zk_core_prover_check_400k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_prover_core", name: "zk_prover_core/check_600k", fn: (n) => wasm.zk_core_prover_check_600k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_prover_core", name: "zk_prover_core/check_1m", fn: (n) => wasm.zk_core_prover_check_1m(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_prover_core", name: "zk_prover_core/check_10m", fn: (n) => wasm.zk_core_prover_check_10m(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        // Async worker pool benchmarks (wasm_workers feature - no rayon/SharedArrayBuffer)
        { category: "zk_prover_core", name: "zk_prover_core/check_async_400k", fn: (n) => wasm.zk_core_prover_check_async_400k?.(n), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "zk_prover_core", name: "zk_prover_core/check_async_1m", fn: (n) => wasm.zk_core_prover_check_async_1m?.(n), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "zk_prover_core", name: "zk_prover_core/check_async_10m", fn: (n) => wasm.zk_core_prover_check_async_10m?.(n), async: true, returnsBenchResult: true, warmup: 1 },
        // zk_verifier_core benchmarks (QuickSilver ZK verifier primitives)
        { category: "zk_verifier_core", name: "zk_verifier_core/execute", fn: (n) => wasm.zk_core_verifier_execute(n), async: false, returnsBenchResult: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_200k", fn: (n) => wasm.zk_core_verifier_check_200k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_400k", fn: (n) => wasm.zk_core_verifier_check_400k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_600k", fn: (n) => wasm.zk_core_verifier_check_600k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        // zk_prover benchmarks
        { category: "zk_prover", name: "zk_prover/100k", fn: (n) => wasm.zk_prover(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "zk_prover", name: "zk_prover/400k", fn: (n) => wasm.zk_prover(n, 400000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "zk_prover", name: "zk_prover/1m", fn: (n) => wasm.zk_prover(n, 1000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "zk_prover", name: "zk_prover/10m", fn: (n) => wasm.zk_prover(n, 10000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        // zk_verifier benchmarks
        { category: "zk_verifier", name: "zk_verifier/100k", fn: (n) => wasm.zk_verifier(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "zk_verifier", name: "zk_verifier/1m", fn: (n) => wasm.zk_verifier(n, 1000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "zk_verifier", name: "zk_verifier/10m", fn: (n) => wasm.zk_verifier(n, 10000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        // garble benchmarks
        { category: "garble", name: "garble/garbler_100k", fn: (n) => wasm.garble_garbler(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "garble", name: "garble/garbler_1m", fn: (n) => wasm.garble_garbler(n, 1000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "garble", name: "garble/garbler_10m", fn: (n) => wasm.garble_garbler(n, 10000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "garble", name: "garble/evaluator_100k", fn: (n) => wasm.garble_evaluator(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "garble", name: "garble/evaluator_1m", fn: (n) => wasm.garble_evaluator(n, 1000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "garble", name: "garble/evaluator_10m", fn: (n) => wasm.garble_evaluator(n, 10000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        // ferret_sender benchmarks
        { category: "ferret_sender", name: "ferret_sender/100k", fn: (n) => wasm.ferret_sender(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "ferret_sender", name: "ferret_sender/1m", fn: (n) => wasm.ferret_sender(n, 1000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        { category: "ferret_sender", name: "ferret_sender/10m", fn: (n) => wasm.ferret_sender(n, 10000000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
        // chi_pool benchmarks (private memory worker pool for compute_chis)
        { category: "chi_pool", name: "chi_pool/async_100k", fn: () => benchChiWorkerPool(100000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/async_400k", fn: () => benchChiWorkerPool(400000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/async_1m", fn: () => benchChiWorkerPool(1000000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/async_10m", fn: () => benchChiWorkerPool(10000000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_100k", fn: () => benchChiSync(100000), async: false, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_1m", fn: () => benchChiSync(1000000), async: false, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_10m", fn: () => benchChiSync(10000000), async: false, returnsBenchResult: true, warmup: 1 },
        // chi_signal test (tests signal buffer + monitor + chi pool integration)
        { category: "chi_signal", name: "chi_signal/test_10k", fn: () => testChiSignalInfra(10000), async: true, returnsBenchResult: true, warmup: 0, samples: 1 },
        { category: "chi_signal", name: "chi_signal/test_100k", fn: () => testChiSignalInfra(100000), async: true, returnsBenchResult: true, warmup: 0, samples: 1 },
    ];
}

// Check if any of the given benchmark names require MT (thread pool)
export function needsThreadPool(benchmarkNames, concurrency = 8) {
    if (!benchmarkNames || benchmarkNames.length === 0) {
        // No filter means all benchmarks, some of which need MT
        return true;
    }
    const defs = getAllBenchmarkDefs(concurrency);
    return benchmarkNames.some(name => {
        const def = defs.find(d => d.name === name);
        return def && def.mt;
    });
}

// Run garble-core benchmarks
export function runGarbleCoreBenchmarks(iterations = 100, samples = 10) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    const defs = getAllBenchmarkDefs().filter(d => d.category === "garble_core");
    const results = [];

    for (let i = 0; i < defs.length; i++) {
        const def = defs[i];
        reportProgress(`[garble-core ${i + 1}/${defs.length}] Starting ${def.name}...`);
        results.push(runBench(def.name, def.fn, iterations, samples));
    }

    return results;
}

// Run all benchmarks (or filtered subset)
// concurrency controls thread count for MT benchmarks
export async function runAllBenchmarks(iterations = 100, samples = 10, filter = null, concurrency = 8) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    let allDefs = getAllBenchmarkDefs(concurrency);

    // Filter benchmarks if specified
    if (filter && filter.length > 0) {
        allDefs = allDefs.filter(def => filter.includes(def.name));
    }

    const total = allDefs.length;
    const results = {};
    const completedTimes = [];

    for (let i = 0; i < allDefs.length; i++) {
        const def = allDefs[i];

        // Estimate remaining time based on completed benchmarks
        let etaStr = "";
        if (completedTimes.length > 0) {
            const avgTime = completedTimes.reduce((a, b) => a + b, 0) / completedTimes.length;
            const remaining = avgTime * (total - i);
            const remainingSec = (remaining / 1000).toFixed(0);
            etaStr = ` (~${remainingSec}s remaining)`;
        }

        reportProgress(`[${i + 1}/${total}] Starting ${def.name}...${etaStr}`);

        const startTime = performance.now();
        const warmup = def.warmup !== undefined ? def.warmup : 3;
        let result;
        if (def.returnsBenchResult) {
            result = await runBenchWithResult(def.name, def.fn, iterations, samples, warmup);
        } else if (def.async) {
            result = await runBenchAsync(def.name, def.fn, iterations, samples, warmup);
        } else {
            result = runBenchSync(def.name, def.fn, iterations, samples, warmup);
        }
        const elapsed = performance.now() - startTime;
        completedTimes.push(elapsed);

        // Initialize category array if needed
        if (!results[def.category]) {
            results[def.category] = [];
        }
        results[def.category].push(result);
    }

    return results;
}

// Format results as a table string (for console output)
export function formatResults(results) {
    let output = "";

    const formatSection = (name, benchmarks) => {
        output += `\n=== ${name} ===\n`;
        output += "Name                                    | Median (ms) | Per-iter (µs) | AND gates/s\n";
        output += "-".repeat(88) + "\n";
        for (const b of benchmarks) {
            const name = b.name.padEnd(39);
            const median = b.median_ms.toFixed(2).padStart(11);
            const perIter = b.per_iter_us.toFixed(2).padStart(13);
            const throughput = (b.throughput / 1e6).toFixed(2).padStart(11) + "M";
            output += `${name} | ${median} | ${perIter} | ${throughput}\n`;
        }
    };

    // Format each category dynamically
    for (const [category, benchmarks] of Object.entries(results)) {
        formatSection(category, benchmarks);
    }

    return output;
}

// Test MT context in isolation
export async function testMtContext() {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");
    const result = await wasm.test_mt_context_only();
    return result;
}

// Main entry point for browser/chromiumoxide
export async function runBenchmark(config = {}) {
    const iterations = config.iterations || 100;
    const samples = config.samples || 10;

    const results = runAllBenchmarks(iterations, samples);
    const formatted = formatResults(results);

    console.log(formatted);

    return {
        results,
        formatted,
    };
}
