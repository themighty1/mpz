// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

let wasm = null;
let zigWasm = null;
let privateWasm = null;
let privateWasmBytes = null;  // Keep bytes for per-worker instantiation

// Initialize WASM module
export async function init(wasmModule) {
    wasm = wasmModule;
}

// Initialize Zig WASM module (loaded separately)
export async function initZigWasm() {
    try {
        const response = await fetch('/gf128.wasm');
        const bytes = await response.arrayBuffer();
        const { instance } = await WebAssembly.instantiate(bytes);
        zigWasm = instance.exports;
        console.log("Zig WASM loaded, exports:", Object.keys(zigWasm));
    } catch (e) {
        console.warn("Failed to load Zig WASM:", e);
    }
}

// Initialize private-memory WASM module
export async function initPrivateWasm() {
    try {
        const response = await fetch('/gf128_private.wasm');
        privateWasmBytes = await response.arrayBuffer();
        const { instance } = await WebAssembly.instantiate(privateWasmBytes.slice(0));
        privateWasm = instance.exports;
        console.log("Private WASM loaded, exports:", Object.keys(privateWasm));
    } catch (e) {
        console.warn("Failed to load Private WASM:", e);
    }
}

// Benchmark private-memory gf128 (single-threaded)
function privateGf128Bench(n) {
    if (!privateWasm) throw new Error("Private WASM not loaded");
    const start = performance.now();
    // gf128_bench returns [lo, hi] via multi-value return
    const [lo, hi] = privateWasm.gf128_bench(n);
    const elapsed_ms = performance.now() - start;
    // Prevent optimization
    if (lo === 0n && hi === 0n) console.log("zero result");
    return { elapsed_ms, and_gates: n };
}

// Benchmark private-memory gf128 (parallel - each worker gets own instance)
async function privateGf128BenchParallel(n) {
    const numWorkers = navigator.hardwareConcurrency || 8;

    // Create workers that each instantiate their own private WASM
    const workerCode = `
        let wasmInstance = null;

        self.onmessage = async (e) => {
            const { type, wasmBytes, iterations } = e.data;

            if (type === 'init') {
                const { instance } = await WebAssembly.instantiate(wasmBytes);
                wasmInstance = instance.exports;
                self.postMessage({ type: 'ready' });
            } else if (type === 'run') {
                const start = performance.now();
                const [lo, hi] = wasmInstance.gf128_bench(iterations);
                const elapsed_ms = performance.now() - start;
                self.postMessage({ type: 'done', elapsed_ms, lo, hi });
            }
        };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);

    // Create and initialize workers
    const workers = [];
    const initPromises = [];

    for (let i = 0; i < numWorkers; i++) {
        const worker = new Worker(workerUrl);
        workers.push(worker);

        const initPromise = new Promise((resolve) => {
            worker.onmessage = (e) => {
                if (e.data.type === 'ready') resolve();
            };
        });
        initPromises.push(initPromise);

        // Send WASM bytes to worker (each gets a copy)
        worker.postMessage({ type: 'init', wasmBytes: privateWasmBytes.slice(0) });
    }

    // Wait for all workers to initialize
    await Promise.all(initPromises);

    // Run benchmark on all workers in parallel
    const start = performance.now();

    const runPromises = workers.map((worker) => {
        return new Promise((resolve) => {
            worker.onmessage = (e) => {
                if (e.data.type === 'done') {
                    resolve(e.data);
                }
            };
            worker.postMessage({ type: 'run', iterations: n });
        });
    });

    // Wait for all workers to complete
    const results = await Promise.all(runPromises);
    const elapsed_ms = performance.now() - start;

    // Cleanup
    workers.forEach(w => w.terminate());
    URL.revokeObjectURL(workerUrl);

    // Total work = n iterations * numWorkers
    const total_ops = n * numWorkers;

    return { elapsed_ms, and_gates: total_ops };
}

// Wrapper for Zig gf128_bench - times from JS since Zig has no performance.now()
function zigGf128Bench(n) {
    if (!zigWasm) throw new Error("Zig WASM not loaded");
    const start = performance.now();
    // gf128_bench(ret_ptr, n) - use address 0 as scratch for return value
    zigWasm.gf128_bench(0, n);
    const elapsed_ms = performance.now() - start;
    return { elapsed_ms, and_gates: n };
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

// Run benchmark that returns BenchResult { elapsed_ms, and_gates }
async function runBenchWithResult(name, fn, iterations, samples = 10, warmupSamples = 1) {
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
        console.log("BenchResult:", result, "elapsed_ms:", result.elapsed_ms, "and_gates:", result.and_gates);
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

    // Throughput: blocks per second
    const blocksPerSec = (andGates * 1000) / mean;

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
        throughput: blocksPerSec,
    };
}

// Define all benchmarks with their categories
// concurrency is passed to MT benchmarks to control thread count
function getAllBenchmarkDefs(concurrency = 8) {
    return [
        // AES implementation comparison benchmarks
        { category: "aes_compare", name: "aes_compare/aes_crate", fn: (n) => wasm.aes_crate_encrypt(n), async: false, returnsBenchResult: true },
        { category: "aes_compare", name: "aes_compare/aes_crate_alloc", fn: (n) => wasm.aes_crate_encrypt_alloc(n), async: false, returnsBenchResult: true },
        { category: "aes_compare", name: "aes_compare/aes_crate_batch", fn: (n) => wasm.aes_crate_encrypt_batch(n), async: false, returnsBenchResult: true },
        { category: "aes_compare", name: "aes_compare/aes_wasm_ctr", fn: (n) => wasm.aes_wasm_ctr(n), async: false, returnsBenchResult: true },
        { category: "aes_compare", name: "aes_compare/aes_crate_parallel", fn: (n) => wasm.aes_crate_parallel(n), async: true, returnsBenchResult: true, mt: true },
        // GF(2^128) multiplication comparison benchmarks
        { category: "gf128_compare", name: "gf128_compare/gf128_mpz", fn: (n) => wasm.gf128_mpz(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_ghash", fn: (n) => wasm.gf128_ghash(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_polyval", fn: (n) => wasm.gf128_polyval(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_polyval_no_red", fn: (n) => wasm.gf128_polyval_no_red(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_polyval_parallel", fn: (n) => wasm.gf128_polyval_parallel(n), async: true, returnsBenchResult: true, mt: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_polyval_no_red_parallel", fn: (n) => wasm.gf128_polyval_no_red_parallel(n), async: true, returnsBenchResult: true, mt: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_aes_wasm_gcm", fn: (n) => wasm.gf128_aes_wasm_gcm(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_aes_wasm_ctr", fn: (n) => wasm.gf128_aes_wasm_ctr(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_zig_rust", fn: (n) => wasm.gf128_zig(n), async: false, returnsBenchResult: true },
        { category: "gf128_compare", name: "gf128_compare/gf128_zig_native", fn: (n) => zigGf128Bench(n), async: false, returnsBenchResult: true },
        // Trivial loop benchmarks to test WASM threading overhead
        { category: "trivial", name: "trivial/single", fn: (n) => wasm.trivial_single(n), async: false, returnsBenchResult: true },
        { category: "trivial", name: "trivial/parallel", fn: (n) => wasm.trivial_parallel(n), async: true, returnsBenchResult: true, mt: true },
        // Private memory benchmarks (no SharedArrayBuffer overhead)
        { category: "private", name: "private/gf128_single", fn: (n) => privateGf128Bench(n), async: false, returnsBenchResult: true },
        { category: "private", name: "private/gf128_parallel", fn: (n) => privateGf128BenchParallel(n), async: true, returnsBenchResult: true },
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
        const warmup = def.warmup !== undefined ? def.warmup : 1;
        const result = await runBenchWithResult(def.name, def.fn, iterations, samples, warmup);
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
        output += "Name                                    | Median (ms) | Per-iter (us) | Blocks/s\n";
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
