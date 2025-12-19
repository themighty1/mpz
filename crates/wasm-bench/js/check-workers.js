// Check Workers - Async worker pool interface for check_prover
//
// This module provides async functions for Rust to call via wasm-bindgen.
// Uses private memory workers for both chi and terms computation.
// No SharedArrayBuffer contention.
//
// Note: This file is copied to pkg/ during build, so paths are relative to pkg/.

import { initChiWorkerPool, computeChisWithWorkerPool } from '../js/bench.js';

// Terms worker pool state
let termsWorkers = [];
let termsWorkersReady = false;
let termsRequestId = 0;
const termsPendingRequests = new Map();

const TERMS_WORKER_COUNT = 16;

// Initialize chi worker pool (reuse from bench.js)
let chiPoolInitialized = false;

async function ensureChiPool() {
    if (!chiPoolInitialized) {
        await initChiWorkerPool();
        chiPoolInitialized = true;
    }
}

// Initialize terms worker pool
export async function initTermsWorkerPool(customWasmUrl = null) {
    if (termsWorkersReady) return;

    const wasmUrl = customWasmUrl || new URL('../pkg-terms/terms_wasm.js', import.meta.url).href;

    const initPromises = [];

    for (let i = 0; i < TERMS_WORKER_COUNT; i++) {
        const worker = new Worker(new URL('./terms-worker.js', import.meta.url), { type: 'module' });

        const readyPromise = new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error(`Terms worker ${i} init timeout`)), 10000);

            worker.onmessage = (e) => {
                if (e.data.type === 'ready') {
                    clearTimeout(timeout);
                    resolve();
                } else if (e.data.type === 'error') {
                    clearTimeout(timeout);
                    reject(new Error(e.data.error));
                } else if (e.data.type === 'log') {
                    console.log(`[terms-worker ${i}]`, e.data.message);
                } else if (e.data.type === 'terms_result') {
                    const pending = termsPendingRequests.get(e.data.requestId);
                    if (pending) {
                        pending.resolve(e.data.data);
                        termsPendingRequests.delete(e.data.requestId);
                    }
                }
            };

            worker.onerror = (err) => {
                clearTimeout(timeout);
                reject(err);
            };
        });

        worker.postMessage({ type: 'init', data: { wasmUrl } });
        termsWorkers.push(worker);
        initPromises.push(readyPromise);
    }

    await Promise.all(initPromises);
    termsWorkersReady = true;
    console.log(`[check-workers] Terms worker pool initialized with ${TERMS_WORKER_COUNT} workers`);
}

// XOR two Uint8Arrays of equal length
function xorBytes(a, b) {
    const result = new Uint8Array(a.length);
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

// Compute terms using worker pool with zero-copy transfers
// Input: triples (48 bytes each), chis (16 bytes each)
// Output: accumulated (u, v) as 32 bytes
async function computeTermsWithWorkerPool(triples, chis) {
    if (!termsWorkersReady) {
        await initTermsWorkerPool();
    }

    const tripleCount = triples.length / 48;
    const chunkSize = Math.ceil(tripleCount / TERMS_WORKER_COUNT);

    const promises = [];

    // Pre-slice data into transferable ArrayBuffers
    const chunks = [];
    for (let i = 0; i < TERMS_WORKER_COUNT; i++) {
        const startIdx = i * chunkSize;
        const endIdx = Math.min((i + 1) * chunkSize, tripleCount);

        if (startIdx >= tripleCount) break;

        // Create new ArrayBuffers for this chunk (one copy here)
        const triplesBuffer = new ArrayBuffer((endIdx - startIdx) * 48);
        const chisBuffer = new ArrayBuffer((endIdx - startIdx) * 16);

        // Copy data into the buffers
        new Uint8Array(triplesBuffer).set(triples.subarray(startIdx * 48, endIdx * 48));
        new Uint8Array(chisBuffer).set(chis.subarray(startIdx * 16, endIdx * 16));

        chunks.push({ triplesBuffer, chisBuffer, startIdx, endIdx });
    }

    // Send chunks to workers with transfer (zero-copy from here)
    for (let i = 0; i < chunks.length; i++) {
        const { triplesBuffer, chisBuffer } = chunks[i];

        const requestId = termsRequestId++;
        const promise = new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                termsPendingRequests.delete(requestId);
                reject(new Error(`Terms worker ${i} computation timeout`));
            }, 60000);

            termsPendingRequests.set(requestId, {
                resolve: (data) => {
                    clearTimeout(timeout);
                    resolve(data);
                },
                reject: (err) => {
                    clearTimeout(timeout);
                    reject(err);
                }
            });
        });

        // Transfer ownership of buffers to worker (zero-copy)
        termsWorkers[i].postMessage({
            type: 'compute_terms',
            data: {
                triples: triplesBuffer,
                chis: chisBuffer,
                requestId
            }
        }, [triplesBuffer, chisBuffer]);

        promises.push(promise);
    }

    // Wait for all workers to complete
    const partials = await Promise.all(promises);

    // XOR reduce all partial (u, v) results
    let result = new Uint8Array(32); // Zero initialized
    for (const partial of partials) {
        result = xorBytes(result, new Uint8Array(partial));
    }

    return result;
}

// ============================================================================
// Exported async functions for Rust (via wasm-bindgen)
// ============================================================================

// Compute chi values using chi worker pool
// Input: chi seed (16 bytes), count
// Output: Uint8Array of count * 16 bytes
export async function computeChisAsync(chi, count) {
    await ensureChiPool();
    const result = await computeChisWithWorkerPool(chi, count);
    return result;
}

// Compute terms using terms worker pool
// Input: triples (48 bytes each), chis (16 bytes each)
// Output: Uint8Array of 32 bytes (u: 16, v: 16)
export async function computeTermsAsync(triples, chis) {
    const result = await computeTermsWithWorkerPool(triples, chis);
    return result;
}

// Initialize both worker pools
export async function initCheckWorkers(chiWasmUrl = null, termsWasmUrl = null) {
    await Promise.all([
        ensureChiPool(),
        initTermsWorkerPool(termsWasmUrl)
    ]);
}
