//! Isolated ZK verifier benchmarks for WASM.
//!
//! Records protocol messages for replay-based isolated benchmarking.
//! This allows benchmarking verifier performance without network overhead.

use wasm_bindgen::prelude::*;

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Waker};

use futures::{AsyncRead, AsyncWrite};
use mpz_circuits::AES128;
use mpz_common::Context;
use mpz_common::context::replay_st_context;
#[cfg(target_arch = "wasm32")]
use mpz_common::context::{
    recording_mt_context_with_spawn_and_limit, replay_mt_context_with_spawn_and_limit,
    Multithread, RecordedMtData,
};
use mpz_core::Block;
use mpz_ot::ideal::rcot::{IdealRCOTSender, ideal_rcot};
use mpz_memory_core::{Array, binary::U8, correlated::Delta};
use mpz_vm_core::{Call, prelude::*};
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::BenchResult;

// ============================================================================
// Browser yield helper
// ============================================================================

/// Yields to the browser event loop, allowing console logs to flush and UI to update.
async fn yield_to_browser() {
    use wasm_bindgen_futures::JsFuture;
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    let _ = JsFuture::from(promise).await;
}

// ============================================================================
// WASM-compatible async byte channel (no tokio dependency)
// ============================================================================

/// Shared state for one direction of the channel
struct ChannelState {
    buffer: VecDeque<u8>,
    waker: Option<Waker>,
    closed: bool,
}

/// Write half of an async byte channel
struct ChannelWriter {
    state: Arc<Mutex<ChannelState>>,
    recorded: Option<Arc<Mutex<Vec<u8>>>>,
}

/// Read half of an async byte channel
struct ChannelReader {
    state: Arc<Mutex<ChannelState>>,
}

fn byte_channel() -> (ChannelWriter, ChannelReader) {
    let state = Arc::new(Mutex::new(ChannelState {
        buffer: VecDeque::new(),
        waker: None,
        closed: false,
    }));
    (
        ChannelWriter { state: state.clone(), recorded: None },
        ChannelReader { state },
    )
}

/// Bidirectional async byte stream (reader + writer from different channels)
struct BiStream {
    reader: ChannelReader,
    writer: ChannelWriter,
}

impl AsyncRead for BiStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let mut state = self.reader.state.lock().unwrap();

        if state.buffer.is_empty() {
            if state.closed {
                return Poll::Ready(Ok(0));
            }
            state.waker = Some(cx.waker().clone());
            return Poll::Pending;
        }

        let to_read = buf.len().min(state.buffer.len());
        let (front, back) = state.buffer.as_slices();
        if to_read <= front.len() {
            buf[..to_read].copy_from_slice(&front[..to_read]);
        } else {
            buf[..front.len()].copy_from_slice(front);
            buf[front.len()..to_read].copy_from_slice(&back[..to_read - front.len()]);
        }
        state.buffer.drain(..to_read);
        Poll::Ready(Ok(to_read))
    }
}

impl AsyncWrite for BiStream {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let mut state = self.writer.state.lock().unwrap();

        if let Some(ref recorded) = self.writer.recorded {
            recorded.lock().unwrap().extend_from_slice(buf);
        }

        state.buffer.extend(buf);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        let mut state = self.writer.state.lock().unwrap();
        state.closed = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
        Poll::Ready(Ok(()))
    }
}

/// Creates recording context pair for WASM.
/// Writes from ctx_1 to ctx_0 are recorded.
fn wasm_recording_context(max_frame_length: usize) -> (Context, Context, Arc<Mutex<Vec<u8>>>) {
    let (writer_a, reader_a) = byte_channel();
    let (mut writer_b, reader_b) = byte_channel();

    let recorded = Arc::new(Mutex::new(Vec::new()));
    writer_b.recorded = Some(recorded.clone());

    let stream_0 = BiStream { reader: reader_b, writer: writer_a };
    let stream_1 = BiStream { reader: reader_a, writer: writer_b };

    (
        Context::new_single_threaded_with_limit(stream_0, max_frame_length),
        Context::new_single_threaded_with_limit(stream_1, max_frame_length),
        recorded,
    )
}

// ============================================================================
// Benchmark code
// ============================================================================

const BLOCK_COUNT: usize = 1000;

/// Calculate max frame length based on workload size.
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    let bytes_per_correlation = 1 + 16; // choice bit + MAC
    let overhead = 1.2; // serialization overhead
    let correlations = circuit.and_count() * circuit_count;
    ((correlations * bytes_per_correlation) as f64 * overhead) as usize
}

/// Runs the full ZK protocol with prover and verifier.
/// Records prover->verifier messages (ctx_p is the recording context).
async fn run_protocol_record_prover(
    ctx_p: &mut Context,
    ctx_v: &mut Context,
    seed: u64,
    batch_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

    let prover_config = ProverConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();

    let mut prover = Prover::new(prover_config, ot_recv);
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    futures::join!(
        {
            let key: Array<U8, 16> = prover.alloc().unwrap();
            prover.mark_private(key).unwrap();
            prover.assign(key, [0u8; 16]).unwrap();
            prover.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
                let msg: Array<U8, 16> = prover.alloc().unwrap();
                prover.mark_public(msg).unwrap();
                prover.assign(msg, [42u8; 16]).unwrap();
                prover.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = prover
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(prover.decode(ciphertext).unwrap());
            }

            async {
                prover.flush(ctx_p).await.unwrap();
                prover.execute(ctx_p).await.unwrap();
                prover.flush(ctx_p).await.unwrap();
            }
        },
        {
            let key: Array<U8, 16> = verifier.alloc().unwrap();
            verifier.mark_blind(key).unwrap();
            verifier.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
                let msg: Array<U8, 16> = verifier.alloc().unwrap();
                verifier.mark_public(msg).unwrap();
                verifier.assign(msg, [42u8; 16]).unwrap();
                verifier.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = verifier
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(verifier.decode(ciphertext).unwrap());
            }

            async {
                verifier.flush(ctx_v).await.unwrap();
                verifier.execute(ctx_v).await.unwrap();
                verifier.flush(ctx_v).await.unwrap();
            }
        }
    );
}

/// Records prover->verifier messages for verifier replay.
/// Returns (recorded_bytes, ot_seed, delta) needed for deterministic replay.
async fn record_for_verifier(seed: u64, batch_size: usize) -> (Vec<u8>, Block, Delta) {
    // ctx_1's writes are recorded, so prover uses ctx_1
    let (mut ctx_v, mut ctx_p, recorded) =
        wasm_recording_context(max_frame_length(&AES128, BLOCK_COUNT));

    // Capture delta and ot_seed for verifier replay
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);
    let ot_seed: Block = rng.random();

    run_protocol_record_prover(&mut ctx_p, &mut ctx_v, seed, batch_size).await;

    (recorded.lock().unwrap().clone(), ot_seed, delta)
}

/// Runs verifier only with replay context.
async fn run_verifier_with_replay(
    ctx: &mut Context,
    batch_size: usize,
    delta: Delta,
    ot_seed: Block,
) {
    // OT sender needs seed and delta to generate consistent correlations
    let ot_send = IdealRCOTSender::new(ot_seed, delta.into_inner());
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    let key: Array<U8, 16> = verifier.alloc().unwrap();
    verifier.mark_blind(key).unwrap();
    verifier.commit(key).unwrap();

    for _ in 0..BLOCK_COUNT {
        let msg: Array<U8, 16> = verifier.alloc().unwrap();
        verifier.mark_public(msg).unwrap();
        verifier.assign(msg, [42u8; 16]).unwrap();
        verifier.commit(msg).unwrap();

        let ciphertext: Array<U8, 16> = verifier
            .call(
                Call::builder(AES128.clone())
                    .arg(key)
                    .arg(msg)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        std::mem::drop(verifier.decode(ciphertext).unwrap());
    }

    verifier.flush(ctx).await.unwrap();
    verifier.execute(ctx).await.unwrap();
    verifier.flush(ctx).await.unwrap();
}

/// Benchmark isolated verifier with message replay.
///
/// Records prover->verifier messages once during setup, then benchmarks
/// verifier execution in isolation using replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `batch_size` - Batch size for consistency checks (e.g., 200000, 400000, etc.)
#[wasm_bindgen]
pub async fn zk_verifier(n: u32, batch_size: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;
    let batch_size = batch_size as usize;

    let performance = web_sys::window().unwrap().performance().unwrap();

    // Record messages once (not timed)
    web_sys::console::log_1(&format!("[rust] Recording messages for verifier, batch_size={}...", batch_size).into());
    yield_to_browser().await;

    let (recorded, ot_seed, delta) = record_for_verifier(0, batch_size).await;
    web_sys::console::log_1(&format!("[rust] Recorded {} bytes", recorded.len()).into());
    yield_to_browser().await;

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        if i % 10 == 0 {
            web_sys::console::log_1(&format!("[rust] Iteration {}/{}", i, n).into());
            yield_to_browser().await;
        }

        // Timed section: verifier replay
        let start = performance.now();

        let mut ctx = replay_st_context(recorded.clone(), max_frame_length(&AES128, BLOCK_COUNT));
        run_verifier_with_replay(&mut ctx, batch_size, delta, ot_seed).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BLOCK_COUNT as u64 * and_gates_per_circuit,
    }
}

// ============================================================================
// Multi-threaded isolated verifier benchmark
// ============================================================================

/// Runs the full ZK protocol with MT contexts.
/// Records prover->verifier messages.
#[cfg(target_arch = "wasm32")]
async fn run_protocol_record_prover_mt(
    exec_p: &mut Multithread,
    exec_v: &mut Multithread,
    seed: u64,
    batch_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

    let prover_config = ProverConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();

    let mut prover = Prover::new(prover_config, ot_recv);
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    let mut ctx_p = exec_p.new_context().await.unwrap();
    let mut ctx_v = exec_v.new_context().await.unwrap();

    futures::join!(
        {
            let key: Array<U8, 16> = prover.alloc().unwrap();
            prover.mark_private(key).unwrap();
            prover.assign(key, [0u8; 16]).unwrap();
            prover.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
                let msg: Array<U8, 16> = prover.alloc().unwrap();
                prover.mark_public(msg).unwrap();
                prover.assign(msg, [42u8; 16]).unwrap();
                prover.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = prover
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(prover.decode(ciphertext).unwrap());
            }

            async {
                prover.flush(&mut ctx_p).await.unwrap();
                prover.execute(&mut ctx_p).await.unwrap();
                prover.flush(&mut ctx_p).await.unwrap();
            }
        },
        {
            let key: Array<U8, 16> = verifier.alloc().unwrap();
            verifier.mark_blind(key).unwrap();
            verifier.commit(key).unwrap();

            for _ in 0..BLOCK_COUNT {
                let msg: Array<U8, 16> = verifier.alloc().unwrap();
                verifier.mark_public(msg).unwrap();
                verifier.assign(msg, [42u8; 16]).unwrap();
                verifier.commit(msg).unwrap();

                let ciphertext: Array<U8, 16> = verifier
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                std::mem::drop(verifier.decode(ciphertext).unwrap());
            }

            async {
                verifier.flush(&mut ctx_v).await.unwrap();
                verifier.execute(&mut ctx_v).await.unwrap();
                verifier.flush(&mut ctx_v).await.unwrap();
            }
        }
    );
}

/// Records prover->verifier messages for MT verifier replay.
#[cfg(target_arch = "wasm32")]
async fn record_for_verifier_mt(seed: u64, batch_size: usize, concurrency: usize) -> (RecordedMtData, Block, Delta) {
    // exec_1's writes are recorded, so prover uses exec_1
    let (mut exec_v, mut exec_p, recorded) = recording_mt_context_with_spawn_and_limit(
        1024 * 1024,
        max_frame_length(&AES128, BLOCK_COUNT),
        concurrency,
        |f| {
            let _ = web_spawn::spawn(f);
            Ok(())
        },
    );

    // Capture delta and ot_seed for verifier replay
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);
    let ot_seed: Block = rng.random();

    run_protocol_record_prover_mt(&mut exec_p, &mut exec_v, seed, batch_size).await;

    (recorded.lock().unwrap().clone(), ot_seed, delta)
}

/// Runs MT verifier only with replay context.
#[cfg(target_arch = "wasm32")]
async fn run_verifier_with_replay_mt(
    exec: &mut Multithread,
    batch_size: usize,
    delta: Delta,
    ot_seed: Block,
) {
    let ot_send = IdealRCOTSender::new(ot_seed, delta.into_inner());
    let verifier_config = VerifierConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut verifier = Verifier::new(verifier_config, delta, ot_send);

    let mut ctx = exec.new_context().await.unwrap();

    let key: Array<U8, 16> = verifier.alloc().unwrap();
    verifier.mark_blind(key).unwrap();
    verifier.commit(key).unwrap();

    for _ in 0..BLOCK_COUNT {
        let msg: Array<U8, 16> = verifier.alloc().unwrap();
        verifier.mark_public(msg).unwrap();
        verifier.assign(msg, [42u8; 16]).unwrap();
        verifier.commit(msg).unwrap();

        let ciphertext: Array<U8, 16> = verifier
            .call(
                Call::builder(AES128.clone())
                    .arg(key)
                    .arg(msg)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        std::mem::drop(verifier.decode(ciphertext).unwrap());
    }

    verifier.flush(&mut ctx).await.unwrap();
    verifier.execute(&mut ctx).await.unwrap();
    verifier.flush(&mut ctx).await.unwrap();
}

/// Benchmark isolated verifier with MT context and message replay.
///
/// Records prover->verifier messages once during setup using MT contexts,
/// then benchmarks verifier execution in isolation using MT replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `batch_size` - Batch size for consistency checks (e.g., 200000, 400000, etc.)
/// * `concurrency` - Number of worker threads for parallel execution
///
/// # Implementation Note
///
/// This benchmark runs on a Web Worker (via `web_spawn::spawn`) rather than
/// the main browser thread. This is required because rayon parallel iterators
/// call `Atomics.wait` which is forbidden on the main browser thread.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_verifier_mt(n: u32, batch_size: u32, concurrency: u32) -> BenchResult {
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    let and_gates_per_circuit = AES128.and_count() as u64;

    // Shared slot for benchmark result
    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    web_sys::console::log_1(&"[rust] zk_verifier_mt: spawning worker...".into());

    // Run benchmark on web worker thread (where Atomics.wait is allowed)
    let _handle = web_spawn::spawn(move || {
        web_sys::console::log_1(&"[rust] verifier worker started".into());
        let bench_result = pollster::block_on(async {
            let batch_size = batch_size as usize;
            let global = js_sys::global();
            let performance: web_sys::Performance = js_sys::Reflect::get(&global, &"performance".into())
                .expect("performance should exist")
                .unchecked_into();

            web_sys::console::log_1(
                &format!(
                    "[rust] Recording MT messages for verifier, batch_size={}, concurrency={}...",
                    batch_size, concurrency
                )
                .into(),
            );

            let (recorded, ot_seed, delta) = record_for_verifier_mt(0, batch_size, concurrency as usize).await;
            let total_bytes: usize = recorded.channels.values().map(|v| v.len()).sum();
            web_sys::console::log_1(
                &format!(
                    "[rust] Recorded {} channels, {} total bytes",
                    recorded.channels.len(),
                    total_bytes
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                if i % 10 == 0 {
                    web_sys::console::log_1(&format!("[rust] MT Verifier Iteration {}/{}", i, n).into());
                }

                let start = performance.now();

                let mut exec = replay_mt_context_with_spawn_and_limit(
                    recorded.clone(),
                    max_frame_length(&AES128, BLOCK_COUNT),
                    concurrency as usize,
                    |f| {
                        let _ = web_spawn::spawn(f);
                        Ok(())
                    },
                );
                run_verifier_with_replay_mt(&mut exec, batch_size, delta, ot_seed).await;

                total_elapsed_ms += performance.now() - start;
            }

            BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: n as u64 * BLOCK_COUNT as u64 * and_gates_per_circuit,
            }
        });
        web_sys::console::log_1(&"[rust] verifier worker storing result".into());
        *result_clone.lock().unwrap() = Some(bench_result);
    });

    // Initial yield to let the worker start
    JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await.unwrap();

    // Poll for result on main thread (non-blocking)
    loop {
        if let Some(r) = result.lock().unwrap().take() {
            return r;
        }
        // Yield with 10ms delay to avoid busy-spinning
        JsFuture::from(js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 10)
                .unwrap();
        }))
        .await
        .unwrap();
    }
}
