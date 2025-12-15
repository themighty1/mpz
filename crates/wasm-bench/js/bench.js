// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

let wasm = null;
let andGateCount = 0;

// Initialize WASM module
export async function init(wasmModule) {
    wasm = wasmModule;
    andGateCount = wasm.aes128_and_count();
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

// Run a single benchmark with warmup and multiple samples
function runBench(name, fn, iterations, samples = 10, warmupSamples = 3) {
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

    // Calculate statistics
    times.sort((a, b) => a - b);
    const min = times[0];
    const max = times[times.length - 1];
    const median = times[Math.floor(times.length / 2)];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const perIter = mean / iterations;

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
        { category: "garble", name: "half_gates_garble", fn: (n) => wasm.bench_half_gates_garble(n) },
        { category: "garble", name: "three_halves_garble", fn: (n) => wasm.bench_three_halves_garble(n) },
        { category: "evaluate", name: "half_gates_evaluate", fn: (n) => wasm.bench_half_gates_evaluate(n) },
        { category: "evaluate", name: "three_halves_evaluate", fn: (n) => wasm.bench_three_halves_evaluate(n) },
    ];
}

// Run all garbling benchmarks
export function runGarbleBenchmarks(iterations = 100, samples = 10) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    const defs = getAllBenchmarkDefs().filter(d => d.category === "garble");
    const results = [];

    for (let i = 0; i < defs.length; i++) {
        const def = defs[i];
        reportProgress(`[Garble ${i + 1}/${defs.length}] Starting ${def.name}...`);
        results.push(runBench(def.name, def.fn, iterations, samples));
    }

    return results;
}

// Run all evaluation benchmarks
export function runEvaluateBenchmarks(iterations = 100, samples = 10) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    const defs = getAllBenchmarkDefs().filter(d => d.category === "evaluate");
    const results = [];

    for (let i = 0; i < defs.length; i++) {
        const def = defs[i];
        reportProgress(`[Evaluate ${i + 1}/${defs.length}] Starting ${def.name}...`);
        results.push(runBench(def.name, def.fn, iterations, samples));
    }

    return results;
}

// Run all benchmarks (or filtered subset)
export function runAllBenchmarks(iterations = 100, samples = 10, filter = null) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    let allDefs = getAllBenchmarkDefs();

    // Filter benchmarks if specified
    if (filter && filter.length > 0) {
        allDefs = allDefs.filter(def => filter.includes(def.name));
    }

    const total = allDefs.length;
    const results = { garble: [], evaluate: [] };
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
        const result = runBench(def.name, def.fn, iterations, samples);
        const elapsed = performance.now() - startTime;
        completedTimes.push(elapsed);

        results[def.category].push(result);
    }

    // Remove empty categories
    if (results.garble.length === 0) delete results.garble;
    if (results.evaluate.length === 0) delete results.evaluate;

    return results;
}

// Format results as a table string (for console output)
export function formatResults(results) {
    let output = "";

    const formatSection = (name, benchmarks) => {
        output += `\n=== ${name} ===\n`;
        output += "Name                          | Median (ms) | Per-iter (µs) | AND gates/s\n";
        output += "-".repeat(78) + "\n";
        for (const b of benchmarks) {
            const name = b.name.padEnd(29);
            const median = b.median_ms.toFixed(2).padStart(11);
            const perIter = b.per_iter_us.toFixed(2).padStart(13);
            const throughput = (b.throughput / 1e6).toFixed(2).padStart(11) + "M";
            output += `${name} | ${median} | ${perIter} | ${throughput}\n`;
        }
    };

    if (results.garble) formatSection("Garble (AES-128)", results.garble);
    if (results.evaluate) formatSection("Evaluate (AES-128)", results.evaluate);

    return output;
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
