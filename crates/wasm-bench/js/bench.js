// WASM Benchmark Runner
// Uses performance.now() for high-resolution timing

let wasm = null;

// Initialize WASM module
export async function init(wasmModule) {
    wasm = wasmModule;
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
        throughput: 1000 / perIter, // ops per second
    };
}

// Run all garbling benchmarks
export function runGarbleBenchmarks(iterations = 100, samples = 10) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    const results = [];

    results.push(runBench(
        "half_gates_garble",
        (n) => wasm.bench_half_gates_garble(n),
        iterations,
        samples
    ));

    results.push(runBench(
        "three_halves_garble",
        (n) => wasm.bench_three_halves_garble(n),
        iterations,
        samples
    ));

    return results;
}

// Run all evaluation benchmarks
export function runEvaluateBenchmarks(iterations = 100, samples = 10) {
    if (!wasm) throw new Error("WASM not initialized. Call init() first.");

    const results = [];

    results.push(runBench(
        "half_gates_evaluate",
        (n) => wasm.bench_half_gates_evaluate(n),
        iterations,
        samples
    ));

    results.push(runBench(
        "three_halves_evaluate",
        (n) => wasm.bench_three_halves_evaluate(n),
        iterations,
        samples
    ));

    return results;
}

// Run all benchmarks
export function runAllBenchmarks(iterations = 100, samples = 10) {
    return {
        garble: runGarbleBenchmarks(iterations, samples),
        evaluate: runEvaluateBenchmarks(iterations, samples),
    };
}

// Format results as a table string (for console output)
export function formatResults(results) {
    let output = "";

    const formatSection = (name, benchmarks) => {
        output += `\n=== ${name} ===\n`;
        output += "Name                          | Median (ms) | Per-iter (µs) | Throughput (ops/s)\n";
        output += "-".repeat(85) + "\n";
        for (const b of benchmarks) {
            const name = b.name.padEnd(29);
            const median = b.median_ms.toFixed(2).padStart(11);
            const perIter = b.per_iter_us.toFixed(2).padStart(13);
            const throughput = b.throughput.toFixed(1).padStart(18);
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
