// Step 2 Validation Tests - Chi Bridge Integration
//
// Tests for the chi-bridge.js module that enables Atomics.wait blocking
// in web-spawn workers for parallel chi computation.
//
// Run in browser console:
//   import * as tests from './js/test-chi-bridge.js';
//   await tests.runAll();

import {
    setWasmMemory,
    startChiRequestMonitor,
    initChiWorkerPool,
    computeChisWithWorkerPool
} from './bench.js';

// Must match CHI_SIGNAL_OFFSET in bench.js and chi-bridge.js
const CHI_SIGNAL_OFFSET = 1048576;

// ============================================================================
// Test 1: Verify chi-bridge.js module loads and exports correct functions
// ============================================================================
export async function testChiBridgeModule() {
    console.log('[test] Testing chi-bridge module exports...');

    try {
        // chi-bridge.js is in pkg/ (copied during build)
        const chiBridge = await import('../pkg/chi-bridge.js');

        const hasRequestChiComputation = typeof chiBridge.request_chi_computation === 'function';
        const hasSetChiBridgeMemory = typeof chiBridge.setChiBridgeMemory === 'function';

        console.log(`[test]   request_chi_computation: ${hasRequestChiComputation ? 'OK' : 'MISSING'}`);
        console.log(`[test]   setChiBridgeMemory: ${hasSetChiBridgeMemory ? 'OK' : 'MISSING'}`);

        if (!hasRequestChiComputation || !hasSetChiBridgeMemory) {
            throw new Error('Chi-bridge module missing required exports');
        }

        console.log('[test] Chi-bridge module test PASSED');
        return { success: true };
    } catch (e) {
        console.error('[test] Chi-bridge module test FAILED:', e);
        return { success: false, error: e.message };
    }
}

// ============================================================================
// Test 2: Test Atomics.wait in a Worker context (simulates web-spawn worker)
// Tested 2025-12-19 19:12: PASSED in browser
// ============================================================================
export async function testAtomicsWaitInWorker() {
    console.log('[test] Testing Atomics.wait in Worker context...');

    // Create a simple worker that uses Atomics.wait
    const workerCode = `
        self.onmessage = function(e) {
            const { signalBuffer, expectedValue, timeout } = e.data;
            const view = new Int32Array(signalBuffer);

            try {
                // This should block until value changes or timeout
                const result = Atomics.wait(view, 0, expectedValue, timeout);
                self.postMessage({ success: true, result, finalValue: Atomics.load(view, 0) });
            } catch (err) {
                self.postMessage({ success: false, error: err.message });
            }
        };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));

    const signalBuffer = new SharedArrayBuffer(4);
    const signalView = new Int32Array(signalBuffer);
    Atomics.store(signalView, 0, 0); // Initial value

    return new Promise((resolve) => {
        worker.onmessage = (e) => {
            worker.terminate();
            if (e.data.success) {
                console.log(`[test]   Atomics.wait result: ${e.data.result}`);
                console.log(`[test]   Final value: ${e.data.finalValue}`);
                console.log('[test] Atomics.wait in Worker test PASSED');
                resolve({ success: true, ...e.data });
            } else {
                console.error('[test] Atomics.wait in Worker test FAILED:', e.data.error);
                resolve({ success: false, error: e.data.error });
            }
        };

        worker.onerror = (e) => {
            worker.terminate();
            console.error('[test] Worker error:', e.message);
            resolve({ success: false, error: e.message });
        };

        // Start worker waiting on value 0
        worker.postMessage({ signalBuffer, expectedValue: 0, timeout: 5000 });

        // After 100ms, change value and notify
        setTimeout(() => {
            Atomics.store(signalView, 0, 42);
            Atomics.notify(signalView, 0);
        }, 100);
    });
}

// ============================================================================
// Test 3: Test chi-bridge request_chi_computation in a Worker (simulates web-spawn)
// ============================================================================
export async function testChiBridgeInWorker(count = 1000, wasmUrl = null) {
    console.log(`[test] Testing chi-bridge in Worker context (count=${count})...`);

    // Create fake WASM memory (large enough for signal region at 1MB + result data)
    const fakeWasmMemorySize = CHI_SIGNAL_OFFSET + 32 + count * 16 + 1024;
    const fakeWasmMemory = new SharedArrayBuffer(fakeWasmMemorySize);
    setWasmMemory({ buffer: fakeWasmMemory });

    // Start monitor (uses WASM memory at CHI_SIGNAL_OFFSET)
    await startChiRequestMonitor();

    // Worker code - uses WASM memory at CHI_SIGNAL_OFFSET for signal buffer
    // This simulates what web-spawn workers do when calling request_chi_computation
    const workerCode = `
        const CHI_SIGNAL_OFFSET = ${CHI_SIGNAL_OFFSET};

        self.onmessage = async function(e) {
            const { wasmMemory, chiPtr, count, resultPtr } = e.data;

            try {
                // Create views on WASM memory at signal offset (same as main thread)
                const signalView = new Int32Array(wasmMemory, CHI_SIGNAL_OFFSET, 4);
                const chiSignalView = new Uint8Array(wasmMemory, CHI_SIGNAL_OFFSET + 16, 16);

                // Write chi seed to WASM memory at chiPtr
                const wasmView = new Uint8Array(wasmMemory);
                const chi = new Uint8Array(16);
                crypto.getRandomValues(chi);
                wasmView.set(chi, chiPtr);

                // Write chi to signal region at offset 16
                chiSignalView.set(chi);

                // Write count and resultPtr to signal region
                Atomics.store(signalView, 1, count);
                Atomics.store(signalView, 2, resultPtr);

                console.log('[worker] Setting status=1 and calling Atomics.wait...');
                const startTime = performance.now();

                // Set status = 1 (request pending) - triggers monitor
                Atomics.store(signalView, 0, 1);

                // Block until result is ready (status changes from 1)
                const waitResult = Atomics.wait(signalView, 0, 1);

                const elapsed = performance.now() - startTime;
                const finalStatus = Atomics.load(signalView, 0);
                console.log('[worker] Atomics.wait returned: ' + waitResult + ', status=' + finalStatus + ', elapsed=' + elapsed.toFixed(2) + 'ms');

                // Reset status to idle
                Atomics.store(signalView, 0, 0);

                // Verify result was written
                const resultView = new Uint8Array(wasmMemory, resultPtr, count * 16);
                const nonZeroBytes = Array.from(resultView.slice(0, 100)).filter(b => b !== 0).length;

                self.postMessage({
                    success: finalStatus === 2,
                    elapsed,
                    resultBytes: count * 16,
                    nonZeroSample: nonZeroBytes,
                    waitResult,
                    finalStatus
                });
            } catch (err) {
                console.error('[worker] Error:', err);
                self.postMessage({ success: false, error: err.message });
            }
        };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));

    return new Promise((resolve) => {
        worker.onmessage = (e) => {
            worker.terminate();
            if (e.data.success) {
                console.log(`[test]   Elapsed: ${e.data.elapsed.toFixed(2)}ms`);
                console.log(`[test]   Result bytes: ${e.data.resultBytes}`);
                console.log(`[test]   Non-zero sample (first 100): ${e.data.nonZeroSample}`);
                console.log('[test] Chi-bridge in Worker test PASSED');
                resolve({ success: true, ...e.data });
            } else {
                console.error('[test] Chi-bridge in Worker test FAILED:', e.data.error);
                resolve({ success: false, error: e.data.error });
            }
        };

        worker.onerror = (e) => {
            worker.terminate();
            console.error('[test] Worker error:', e.message);
            resolve({ success: false, error: e.message });
        };

        // Chi at offset 512, result after signal region
        const chiPtr = 512;
        const resultPtr = CHI_SIGNAL_OFFSET + 32;

        // Pass the WASM memory (SharedArrayBuffer) to the worker
        worker.postMessage({
            wasmMemory: fakeWasmMemory,
            chiPtr,
            count,
            resultPtr
        });
    });
}

// ============================================================================
// Test 4: Verify chi computation correctness (compare worker pool vs sequential)
// ============================================================================
export async function testChiComputationCorrectness(wasm, count = 100, wasmUrl = null) {
    console.log(`[test] Testing chi computation correctness (count=${count})...`);

    if (!wasm) {
        console.error('[test] WASM module required for correctness test');
        return { success: false, error: 'WASM not provided' };
    }

    // Generate random chi seed
    const chi = new Uint8Array(16);
    crypto.getRandomValues(chi);

    // Compute using sequential WASM (reference)
    const seqResult = wasm.compute_chi_segment(chi, count);

    // Initialize worker pool if needed
    const actualWasmUrl = wasmUrl || new URL('../pkg-chi/chi_wasm.js', import.meta.url).href;
    await initChiWorkerPool(actualWasmUrl);

    const poolResult = await computeChisWithWorkerPool(chi, count);

    // Compare results
    if (seqResult.length !== poolResult.length) {
        console.error(`[test] Length mismatch: seq=${seqResult.length}, pool=${poolResult.length}`);
        return { success: false, error: 'Length mismatch' };
    }

    let mismatches = 0;
    for (let i = 0; i < seqResult.length; i++) {
        if (seqResult[i] !== poolResult[i]) {
            mismatches++;
            if (mismatches <= 5) {
                console.error(`[test] Mismatch at byte ${i}: seq=${seqResult[i]}, pool=${poolResult[i]}`);
            }
        }
    }

    if (mismatches > 0) {
        console.error(`[test] Total mismatches: ${mismatches} / ${seqResult.length}`);
        return { success: false, error: `${mismatches} byte mismatches` };
    }

    console.log(`[test]   Compared ${seqResult.length} bytes`);
    console.log('[test] Chi computation correctness test PASSED');
    return { success: true, bytesCompared: seqResult.length };
}

// ============================================================================
// Run all step 2 validation tests
// ============================================================================
export async function runAll(wasm = null, wasmUrl = null) {
    console.log('');
    console.log('='.repeat(60));
    console.log('Step 2 Validation Tests - Chi Bridge Integration');
    console.log('='.repeat(60));
    console.log('');

    const results = {};

    // Test 1: Module exports
    results.moduleExports = await testChiBridgeModule();
    console.log('');

    // Test 2: Atomics.wait in Worker
    results.atomicsWait = await testAtomicsWaitInWorker();
    console.log('');

    // Test 3: Chi-bridge in Worker (skip if module test failed)
    if (results.moduleExports.success) {
        results.chiBridgeWorker = await testChiBridgeInWorker(1000, wasmUrl);
    } else {
        results.chiBridgeWorker = { success: false, error: 'Skipped - module test failed' };
        console.log('[test] Skipping chi-bridge worker test - module test failed');
    }
    console.log('');

    // Test 4: Computation correctness (only if WASM is provided)
    if (wasm) {
        results.correctness = await testChiComputationCorrectness(wasm, 100, wasmUrl);
    } else {
        results.correctness = { success: false, error: 'Skipped - WASM not provided' };
        console.log('[test] Skipping correctness test - WASM not provided');
    }
    console.log('');

    // Summary
    console.log('='.repeat(60));
    console.log('Test Summary:');
    console.log('='.repeat(60));
    let allPassed = true;
    for (const [name, result] of Object.entries(results)) {
        const status = result.success ? 'PASS' : 'FAIL';
        console.log(`  ${name}: ${status}${result.error ? ' - ' + result.error : ''}`);
        if (!result.success) allPassed = false;
    }
    console.log('');
    console.log(allPassed ? 'All tests PASSED!' : 'Some tests FAILED');
    console.log('');

    return { allPassed, results };
}
