// Web Worker for JV benchmark - manual implementation to bypass web_spawn bug
// See commit 240d7d58 on debug/webspawn-bug-investigation

console.log('[worker] JV Worker script loading...');

// Get base URL from worker location
const baseUrl = self.location.origin;

// Dynamically import WASM module
import(`${baseUrl}/pkg/mpz_wasm_bench.js`).then(async (wasmModule) => {
    console.log('[worker] WASM module imported');

    const { default: init, jv_vm_prover_worker } = wasmModule;

    console.log('[worker] Initializing WASM...');
    await init();
    console.log('[worker] WASM initialized');

    self.onmessage = async (e) => {
        console.log('[worker] Received message:', e.data);

        try {
            const { cmd, n, reps } = e.data;

            if (cmd === 'jv_vm_prover') {
                console.log(`[worker] Running jv_vm_prover_worker(${n}, ${reps})`);
                const result = await jv_vm_prover_worker(n, reps);
                console.log('[worker] Benchmark complete');

                const resultJson = JSON.stringify({
                    elapsed_ms: result.elapsed_ms,
                    and_gates: Number(result.and_gates),
                });
                console.log('[worker] Sending result back');
                self.postMessage(resultJson);
            } else {
                throw new Error(`Unknown command: ${cmd}`);
            }
        } catch (error) {
            console.error('[worker] Error:', error);
            self.postMessage(JSON.stringify({ error: error.toString() }));
        }
    };

    console.log('[worker] Ready to receive messages');
}).catch(error => {
    console.error('[worker] Failed to load WASM:', error);
    self.postMessage(JSON.stringify({ error: 'Failed to load WASM: ' + error.toString() }));
});
