// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

let wasm = null;
let andGateCount = 0;

// ============================================================================
// Private Memory Worker Pool for Chi Computation
// ============================================================================

const CHI_WORKER_COUNT = navigator.hardwareConcurrency || 8;
let chiWorkers = [];
let chiWorkersReady = false;
let pendingChiRequests = new Map();
let chiRequestId = 0;

// Initialize the chi worker pool
async function initChiWorkerPool() {
    if (chiWorkersReady) return;

    const workerUrl = new URL('./chi-worker.js', import.meta.url);

    const initPromises = [];
    for (let i = 0; i < CHI_WORKER_COUNT; i++) {
        const worker = new Worker(workerUrl, { type: 'module' });

        const readyPromise = new Promise((resolve) => {
            worker.onmessage = (e) => {
                if (e.data.type === 'ready') {
                    resolve();
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
        });

        worker.postMessage({ type: 'init' });
        chiWorkers.push(worker);
        initPromises.push(readyPromise);
    }

    await Promise.all(initPromises);
    chiWorkersReady = true;
    console.log(`Chi worker pool initialized with ${CHI_WORKER_COUNT} workers`);
}

// Compute chi starting points using worker 0
async function computeChiStarts(chi, segmentSize) {
    const requestId = chiRequestId++;
    return new Promise((resolve) => {
        pendingChiRequests.set(requestId, { resolve });
        chiWorkers[0].postMessage({
            type: 'compute_starts',
            data: { chi: Array.from(chi), segmentSize, requestId }
        });
    });
}

// Dispatch chi computation to workers
async function computeChisWithWorkerPool(chi, count) {
    if (!chiWorkersReady) {
        await initChiWorkerPool();
    }

    if (count === 0) {
        return new Uint8Array(0);
    }

    const PARALLELISM = 16;
    const segmentSize = Math.ceil(count / PARALLELISM);

    // Compute starting points
    const starts = await computeChiStarts(chi, segmentSize);

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

    for (let i = 0; i < PARALLELISM; i++) {
        const segStart = i * segmentSize;
        const segEnd = Math.min((i + 1) * segmentSize, count);
        const segLen = segEnd - segStart;

        const workerIdx = i % chiWorkers.length;
        chiWorkers[workerIdx].postMessage({
            type: 'compute_segment',
            data: {
                segmentIndex: i,
                start: new Uint8Array(starts[i]),
                count: segLen,
                requestId
            }
        });
    }

    const segments = await segmentPromise;

    // Concatenate segments
    const result = new Uint8Array(count * 16);
    let offset = 0;
    for (let i = 0; i < PARALLELISM; i++) {
        if (segments[i] && segments[i].length > 0) {
            result.set(new Uint8Array(segments[i]), offset);
            offset += segments[i].length;
        }
    }

    return result;
}

// Global function called by wasm-bindgen extern
globalThis.compute_chis_parallel = function(chi, count) {
    // Synchronous fallback using main thread (wasm-bindgen extern is sync)
    const PARALLELISM = 16;
    const n = count;
    if (n === 0) return new Uint8Array(0);

    const segmentSize = Math.ceil(n / PARALLELISM);

    // Compute starts (same algorithm as worker)
    const bootstrapped = [];
    let current = new Uint8Array(chi);
    for (let i = 0; i < PARALLELISM; i++) {
        bootstrapped.push(new Uint8Array(current));
        current = gf128_mul_sync(current, current);
    }

    // Hash each (using sync hash - simplified)
    const starts = bootstrapped.map((boot, i) => {
        const start = new Uint8Array(16);
        for (let j = 0; j < 16; j++) {
            start[j] = boot[j] ^ (i & 0xFF) ^ ((segmentSize >> (j * 2)) & 0xFF);
        }
        return start;
    });

    // Compute all segments sequentially
    const result = new Uint8Array(n * 16);
    for (let seg = 0; seg < PARALLELISM; seg++) {
        const segStart = seg * segmentSize;
        const segEnd = Math.min((seg + 1) * segmentSize, n);
        let current = new Uint8Array(starts[seg]);

        for (let i = segStart; i < segEnd; i++) {
            result.set(current, i * 16);
            current = gf128_mul_sync(current, current);
        }
    }

    return result;
};

// Synchronous GF(2^128) multiplication
function gf128_mul_sync(a, b) {
    const a0 = readU64LE(a, 0);
    const a1 = readU64LE(a, 8);
    const b0 = readU64LE(b, 0);
    const b1 = readU64LE(b, 8);

    const z0 = clmul64(a0, b0);
    const z1 = clmul64(a0, b1) ^ clmul64(a1, b0);
    const z2 = clmul64(a1, b1);

    const r1 = z1.lo ^ z0.hi;
    const r2 = z1.hi ^ z2.lo;

    let lo = z0.lo;
    let hi = r1;
    const c0 = r2;
    const c1 = z2.hi;

    hi ^= c1 ^ (c1 >> 1n) ^ (c1 >> 2n) ^ (c1 >> 7n);
    lo ^= (c1 << 63n) ^ (c1 << 62n) ^ (c1 << 57n);
    hi ^= c0 ^ (c0 >> 1n) ^ (c0 >> 2n) ^ (c0 >> 7n);
    lo ^= (c0 << 63n) ^ (c0 << 62n) ^ (c0 << 57n);

    lo = lo & 0xFFFFFFFFFFFFFFFFn;
    hi = hi & 0xFFFFFFFFFFFFFFFFn;

    const result = new Uint8Array(16);
    writeU64LE(result, 0, lo);
    writeU64LE(result, 8, hi);
    return result;
}

function clmul64(a, b) {
    let lo = 0n;
    let hi = 0n;
    for (let i = 0n; i < 64n; i++) {
        if ((b >> i) & 1n) {
            lo ^= a << i;
            if (i > 0n) hi ^= a >> (64n - i);
        }
    }
    return { lo: lo & 0xFFFFFFFFFFFFFFFFn, hi };
}

function readU64LE(arr, offset) {
    let val = 0n;
    for (let i = 0; i < 8; i++) {
        val |= BigInt(arr[offset + i]) << BigInt(i * 8);
    }
    return val;
}

function writeU64LE(arr, offset, val) {
    for (let i = 0; i < 8; i++) {
        arr[offset + i] = Number((val >> BigInt(i * 8)) & 0xFFn);
    }
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

// Benchmark: chi computation synchronous (main thread)
function benchChiSync(gateCount) {
    const chi = new Uint8Array(16);
    crypto.getRandomValues(chi);

    const start = performance.now();
    const result = globalThis.compute_chis_parallel(chi, gateCount);
    const elapsed = performance.now() - start;

    return {
        elapsed_ms: elapsed,
        and_gates: BigInt(gateCount),
        result_bytes: result.length
    };
}

// ============================================================================
// End Worker Pool Section
// ============================================================================

// Initialize WASM module
export async function init(wasmModule) {
    wasm = wasmModule;
    andGateCount = wasm.garble_core_aes128_and_count();
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
        // zk_verifier_core benchmarks (QuickSilver ZK verifier primitives)
        { category: "zk_verifier_core", name: "zk_verifier_core/execute", fn: (n) => wasm.zk_core_verifier_execute(n), async: false, returnsBenchResult: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_200k", fn: (n) => wasm.zk_core_verifier_check_200k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_400k", fn: (n) => wasm.zk_core_verifier_check_400k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        { category: "zk_verifier_core", name: "zk_verifier_core/check_600k", fn: (n) => wasm.zk_core_verifier_check_600k(n, concurrency), async: true, returnsBenchResult: true, mt: true },
        // zk_prover benchmarks
        { category: "zk_prover", name: "zk_prover/100k", fn: (n) => wasm.zk_prover(n, 100000, concurrency), async: true, returnsBenchResult: true, warmup: 1, mt: true },
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
        { category: "chi_pool", name: "chi_pool/async_1m", fn: () => benchChiWorkerPool(1000000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/async_10m", fn: () => benchChiWorkerPool(10000000), async: true, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_100k", fn: () => benchChiSync(100000), async: false, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_1m", fn: () => benchChiSync(1000000), async: false, returnsBenchResult: true, warmup: 1 },
        { category: "chi_pool", name: "chi_pool/sync_10m", fn: () => benchChiSync(10000000), async: false, returnsBenchResult: true, warmup: 1 },
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
