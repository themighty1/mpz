/* tslint:disable */
/* eslint-disable */

export class BenchResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  elapsed_ms: number;
  and_gates: bigint;
}

/**
 * Benchmark JustVengers BGV pattern with WebGPU-accelerated sum_slots.
 *
 * Pattern: CPU (copies + mults + adds) + GPU sum_slots + CPU (mask + additions)
 *
 * # Arguments
 * * `n` - Number of benchmark iterations
 *
 * # Returns
 * BenchResult with elapsed_ms
 */
export function bgv_justvengers_pattern_webgpu(n: number): Promise<BenchResult>;

/**
 * Test version: runs bgv_webgpu benchmark directly (no worker).
 * This tests if async GPU code works at all before trying workers.
 */
export function bgv_webgpu_worker_test(n: number): Promise<BenchResult>;

/**
 * WASM entry point for JV VM benchmark WITH manual Web Worker.
 *
 * **WORKAROUND**: Creates a manual Web Worker to bypass the web_spawn bug.
 * See commit 240d7d58 on debug/webspawn-bug-investigation for details.
 *
 * **GPU SUPPORT**: Includes GPU acceleration running in worker context!
 *
 * # Arguments
 * * `n` - Number of benchmark iterations
 * * `reps` - Number of repetitions (1000, 2000, 3000, 8192, 16384, 32768, 65536, 131072)
 *
 * # Returns
 * BenchResult with elapsed_ms and total multiplications
 */
export function jv_vm_prover_main_thread(n: number, reps: number): Promise<BenchResult>;

/**
 * Worker entry point - runs the benchmark in a Web Worker context.
 * This is called FROM the worker thread, not from main thread.
 */
export function jv_vm_prover_worker(n: number, reps: number): Promise<BenchResult>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly jv_vm_prover_main_thread: (a: number, b: number) => any;
  readonly jv_vm_prover_worker: (a: number, b: number) => any;
  readonly bgv_justvengers_pattern_webgpu: (a: number) => any;
  readonly bgv_webgpu_worker_test: (a: number) => any;
  readonly __wbg_benchresult_free: (a: number, b: number) => void;
  readonly __wbg_get_benchresult_and_gates: (a: number) => bigint;
  readonly __wbg_get_benchresult_elapsed_ms: (a: number) => number;
  readonly __wbg_set_benchresult_and_gates: (a: number, b: bigint) => void;
  readonly __wbg_set_benchresult_elapsed_ms: (a: number, b: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__h9f5043ed1d626f22: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen__closure__destroy__hf3c2b16c614122f5: (a: number, b: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__h1ee7c78686776a5b: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen__closure__destroy__ha67683606eccb2e6: (a: number, b: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__h02003cbf49c8e665: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
