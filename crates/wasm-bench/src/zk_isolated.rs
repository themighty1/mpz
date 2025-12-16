//! Isolated ZK benchmarks for WASM.
//!
//! Records protocol messages for replay-based isolated benchmarking.
//! This allows benchmarking prover performance without network overhead.

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
use mpz_ot::ideal::rcot::{IdealRCOTReceiver, ideal_rcot};
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
        // Use as_slices for zero-allocation bulk copy
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

/// Creates recording context pair for WASM (no tokio dependency).
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
/// Records verifier->prover messages (ctx_v is the recording context).
async fn run_protocol_record_verifier(
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

/// Records verifier->prover messages for prover replay.
async fn record_for_prover(seed: u64, batch_size: usize) -> Vec<u8> {
    let (mut ctx_p, mut ctx_v, recorded) =
        wasm_recording_context(max_frame_length(&AES128, BLOCK_COUNT));
    run_protocol_record_verifier(&mut ctx_p, &mut ctx_v, seed, batch_size).await;
    recorded.lock().unwrap().clone()
}

/// Runs prover only with replay context.
async fn run_prover_with_replay(ctx: &mut Context, batch_size: usize) {
    let ot_recv = IdealRCOTReceiver::new();
    let prover_config = ProverConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut prover = Prover::new(prover_config, ot_recv);

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

    prover.flush(ctx).await.unwrap();
    prover.execute(ctx).await.unwrap();
    prover.flush(ctx).await.unwrap();
}

/// Simple ping-pong test for recording layer.
///
/// Verifies that wasm_recording_context records correctly and replay works.
/// Returns number of messages successfully round-tripped.
#[wasm_bindgen]
pub async fn zk_isolated_test_recording() -> Result<u32, JsValue> {
    use mpz_common::context::replay_st_context;
    use serio::{SinkExt, stream::IoStreamExt};

    web_sys::console::log_1(&"[rust] Testing recording layer...".into());

    // Test 1: Record some messages
    // Note: wasm_recording_context records ctx_1's writes (second context)
    // So ctx_1 should be the sender for recording to work
    let (mut ctx_receiver, mut ctx_sender, recorded_arc) =
        wasm_recording_context(1024 * 1024);

    let test_messages: Vec<Vec<u8>> = vec![
        vec![1, 2, 3, 4],
        vec![10; 100],
        vec![42; 1000],
    ];

    for msg in &test_messages {
        futures::try_join!(
            ctx_sender.io_mut().send(msg.clone()),
            ctx_receiver.io_mut().expect_next::<Vec<u8>>()
        ).map_err(|e| JsValue::from_str(&format!("Record failed: {}", e)))?;
    }

    let recorded = recorded_arc.lock().unwrap().clone();
    web_sys::console::log_1(&format!("[rust] Recorded {} bytes", recorded.len()).into());

    drop(ctx_sender);
    drop(ctx_receiver);

    // Test 2: Replay and verify
    let mut replay_ctx = replay_st_context(recorded, 1024 * 1024);

    let mut success_count = 0u32;
    for expected in &test_messages {
        let received: Vec<u8> = replay_ctx.io_mut().expect_next().await
            .map_err(|e| JsValue::from_str(&format!("Replay failed: {}", e)))?;

        if received == *expected {
            success_count += 1;
        } else {
            web_sys::console::log_1(&format!(
                "[rust] Mismatch: expected {} bytes, got {} bytes",
                expected.len(), received.len()
            ).into());
        }
    }

    web_sys::console::log_1(&format!(
        "[rust] Recording test: {}/{} messages OK",
        success_count, test_messages.len()
    ).into());

    Ok(success_count)
}

/// Benchmark ReplayDuplex read throughput.
///
/// Measures replay performance (reading from pre-recorded buffer).
/// This is what the isolated prover uses during benchmarking.
///
/// NOTE: This benchmark may not be needed - in the isolated prover benchmark,
/// the replay overhead is negligible compared to crypto work. The replay layer
/// itself is not what's being benchmarked. Consider removing this later.
///
/// # Arguments
/// * `n` - Number of iterations (each reads 1 MiB)
#[wasm_bindgen]
pub async fn zk_isolated_replay_throughput(n: u32) -> BenchResult {
    use mpz_common::context::replay_st_context;
    use serio::{SinkExt, stream::IoStreamExt};

    const MSG_SIZE: usize = 1024 * 1024; // 1 MiB per message

    let performance = web_sys::window().unwrap().performance().unwrap();
    let msg: Vec<u8> = vec![0u8; MSG_SIZE];

    // Record n messages using wasm_recording_context (setup, not timed)
    // Note: ctx_1 (second returned) writes are recorded, so it should be sender
    let (mut ctx_receiver, mut ctx_sender, recorded_arc) =
        wasm_recording_context(16 * 1024 * 1024);

    for _ in 0..n {
        futures::try_join!(
            ctx_sender.io_mut().send(msg.clone()),
            ctx_receiver.io_mut().expect_next::<Vec<u8>>()
        ).unwrap();
    }

    let recorded = recorded_arc.lock().unwrap().clone();
    drop(ctx_sender);
    drop(ctx_receiver);

    // Benchmark replay (timed)
    let start = performance.now();

    let mut replay_ctx = replay_st_context(recorded, 16 * 1024 * 1024);
    for _ in 0..n {
        let received: Vec<u8> = replay_ctx.io_mut().expect_next().await.unwrap();
        std::hint::black_box(received);
    }

    let elapsed_ms = performance.now() - start;
    let total_bytes = n as u64 * MSG_SIZE as u64;

    BenchResult {
        elapsed_ms,
        and_gates: total_bytes,
    }
}

/// Benchmark BiStream channel throughput.
///
/// Measures raw channel performance to compare against native (~400 MiB/s).
/// Sends 1 MiB messages back and forth.
///
/// # Arguments
/// * `n` - Number of iterations (each sends 1 MiB)
#[wasm_bindgen]
pub async fn zk_isolated_channel_throughput(n: u32) -> BenchResult {
    use mpz_common::Context;
    use serio::{SinkExt, stream::IoStreamExt};

    const MSG_SIZE: usize = 1024 * 1024; // 1 MiB per message

    let performance = web_sys::window().unwrap().performance().unwrap();

    // Create BiStream pair wrapped in Context (same as protocol uses)
    let (writer_a, reader_a) = byte_channel();
    let (writer_b, reader_b) = byte_channel();
    let stream_0 = BiStream { reader: reader_b, writer: writer_a };
    let stream_1 = BiStream { reader: reader_a, writer: writer_b };
    let mut ctx_0 = Context::new_single_threaded_with_limit(stream_0, 16 * 1024 * 1024);
    let mut ctx_1 = Context::new_single_threaded_with_limit(stream_1, 16 * 1024 * 1024);

    let data: Vec<u8> = vec![0u8; MSG_SIZE];

    let start = performance.now();

    for _ in 0..n {
        let (_, received): (_, Vec<u8>) = futures::try_join!(
            ctx_0.io_mut().send(data.clone()),
            ctx_1.io_mut().expect_next()
        ).unwrap();
        std::hint::black_box(received);
    }

    let elapsed_ms = performance.now() - start;
    let total_bytes = n as u64 * MSG_SIZE as u64;

    // Return bytes as "and_gates" for throughput calculation
    BenchResult {
        elapsed_ms,
        and_gates: total_bytes,
    }
}

/// Benchmark isolated prover with message replay.
///
/// Records verifier->prover messages once during setup, then benchmarks
/// prover execution in isolation using replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `batch_size` - Batch size for consistency checks (e.g., 200000, 400000, etc.)
#[wasm_bindgen]
pub async fn zk_isolated_prover(n: u32, batch_size: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;
    let batch_size = batch_size as usize;

    let performance = web_sys::window().unwrap().performance().unwrap();

    // Record messages once (not timed)
    web_sys::console::log_1(&format!("[rust] Recording messages for batch_size={}...", batch_size).into());
    yield_to_browser().await;

    let recorded = record_for_prover(0, batch_size).await;
    web_sys::console::log_1(&format!("[rust] Recorded {} bytes", recorded.len()).into());
    yield_to_browser().await;

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        if i % 10 == 0 {
            web_sys::console::log_1(&format!("[rust] Iteration {}/{}", i, n).into());
            yield_to_browser().await;
        }

        // Timed section: prover replay
        let start = performance.now();

        let mut ctx = replay_st_context(recorded.clone(), max_frame_length(&AES128, BLOCK_COUNT));
        run_prover_with_replay(&mut ctx, batch_size).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BLOCK_COUNT as u64 * and_gates_per_circuit,
    }
}

// ============================================================================
// Multi-threaded isolated prover benchmark
// ============================================================================

/// Runs the full ZK protocol with MT contexts.
/// Records verifier->prover messages.
#[cfg(target_arch = "wasm32")]
async fn run_protocol_record_verifier_mt(
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

/// Records verifier->prover messages for MT prover replay.
#[cfg(target_arch = "wasm32")]
async fn record_for_prover_mt(seed: u64, batch_size: usize, concurrency: usize) -> RecordedMtData {
    let (mut exec_p, mut exec_v, recorded) = recording_mt_context_with_spawn_and_limit(
        1024 * 1024,
        max_frame_length(&AES128, BLOCK_COUNT),
        concurrency,
        |f| {
            let _ = web_spawn::spawn(f);
            Ok(())
        },
    );
    run_protocol_record_verifier_mt(&mut exec_p, &mut exec_v, seed, batch_size).await;
    recorded.lock().unwrap().clone()
}

/// Runs MT prover only with replay context.
#[cfg(target_arch = "wasm32")]
async fn run_prover_with_replay_mt(exec: &mut Multithread, batch_size: usize) {
    let ot_recv = IdealRCOTReceiver::new();
    let prover_config = ProverConfig::builder()
        .batch_size(batch_size)
        .build()
        .unwrap();
    let mut prover = Prover::new(prover_config, ot_recv);

    let mut ctx = exec.new_context().await.unwrap();

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

    prover.flush(&mut ctx).await.unwrap();
    prover.execute(&mut ctx).await.unwrap();
    prover.flush(&mut ctx).await.unwrap();
}

/// Benchmark isolated prover with MT context and message replay.
///
/// Records verifier->prover messages once during setup using MT contexts,
/// then benchmarks prover execution in isolation using MT replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `batch_size` - Batch size for consistency checks (e.g., 200000, 400000, etc.)
/// * `concurrency` - Number of worker threads for parallel execution
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_isolated_prover_mt(n: u32, batch_size: u32, concurrency: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;
    let batch_size = batch_size as usize;

    let performance = web_sys::window().unwrap().performance().unwrap();

    // Record messages once (not timed)
    web_sys::console::log_1(
        &format!(
            "[rust] Recording MT messages for batch_size={}, concurrency={}...",
            batch_size, concurrency
        )
        .into(),
    );
    yield_to_browser().await;

    let recorded = record_for_prover_mt(0, batch_size, concurrency as usize).await;
    let total_bytes: usize = recorded.channels.values().map(|v| v.len()).sum();
    web_sys::console::log_1(
        &format!(
            "[rust] Recorded {} channels, {} total bytes",
            recorded.channels.len(),
            total_bytes
        )
        .into(),
    );
    yield_to_browser().await;

    let mut total_elapsed_ms = 0.0;

    for i in 0..n {
        if i % 10 == 0 {
            web_sys::console::log_1(&format!("[rust] MT Iteration {}/{}", i, n).into());
            yield_to_browser().await;
        }

        // Timed section: prover replay with MT context
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
        run_prover_with_replay_mt(&mut exec, batch_size).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * BLOCK_COUNT as u64 * and_gates_per_circuit,
    }
}
