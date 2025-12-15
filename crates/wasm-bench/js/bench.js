// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

let wasm = null;
let andGateCount = 0;

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
function getAllBenchmarkDefs() {
    return [
        // garble-core benchmarks (raw garbling primitives)
        { category: "garble_core", name: "garble_core/half_gates_garble", fn: (n) => wasm.garble_core_half_gates_garble(n), async: false },
        { category: "garble_core", name: "garble_core/three_halves_garble", fn: (n) => wasm.garble_core_three_halves_garble(n), async: false },
        { category: "garble_core", name: "garble_core/half_gates_evaluate", fn: (n) => wasm.garble_core_half_gates_evaluate(n), async: false },
        { category: "garble_core", name: "garble_core/three_halves_evaluate", fn: (n) => wasm.garble_core_three_halves_evaluate(n), async: false },
        // garble benchmarks (full semihonest 2PC protocol)
        { category: "garble", name: "garble/semihonest_aes", fn: (n) => wasm.garble_semihonest_aes(n), async: true },
        { category: "garble", name: "garble/semihonest_aes_st_batched", fn: (n) => wasm.garble_semihonest_aes_st_batched(n), async: true, returnsBenchResult: true },
        { category: "garble", name: "garble/semihonest_aes_mt_batched", fn: (n) => wasm.garble_semihonest_aes_batched(n), async: true, returnsBenchResult: true },
        // test/debug benchmarks
        { category: "test", name: "test/mt_context_only", fn: async (n) => { for (let i = 0; i < n; i++) await wasm.test_mt_context_only(); return n; }, async: true },
    ];
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
export async function runAllBenchmarks(iterations = 100, samples = 10, filter = null) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    let allDefs = getAllBenchmarkDefs();

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
        let result;
        if (def.returnsBenchResult) {
            result = await runBenchWithResult(def.name, def.fn, iterations, samples);
        } else if (def.async) {
            result = await runBenchAsync(def.name, def.fn, iterations, samples);
        } else {
            result = runBenchSync(def.name, def.fn, iterations, samples);
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
    console.log("Starting MT context test...");
    const result = await wasm.test_mt_context_only();
    console.log("MT context test result:", result);
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
