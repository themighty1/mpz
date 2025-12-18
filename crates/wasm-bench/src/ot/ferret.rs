//! Isolated Ferret OT benchmarks for WASM.
//!
//! Records protocol messages for replay-based isolated benchmarking of Ferret sender.
//! This allows benchmarking sender performance without network overhead.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use std::collections::VecDeque;
#[cfg(target_arch = "wasm32")]
use std::pin::Pin;
#[cfg(target_arch = "wasm32")]
use std::sync::{Arc, Mutex};
#[cfg(target_arch = "wasm32")]
use std::task::{Context as TaskContext, Poll, Waker};

#[cfg(target_arch = "wasm32")]
use futures::{AsyncRead, AsyncWrite};
#[cfg(target_arch = "wasm32")]
use mpz_common::context::replay_st_context;
#[cfg(target_arch = "wasm32")]
use mpz_common::{Context, Flush};
#[cfg(target_arch = "wasm32")]
use mpz_core::Block;
#[cfg(target_arch = "wasm32")]
use mpz_ot::ferret::{FerretConfig, Receiver, Sender};
#[cfg(target_arch = "wasm32")]
use mpz_ot::ideal::rcot::ideal_rcot;
#[cfg(target_arch = "wasm32")]
use mpz_ot_core::rcot::{RCOTReceiver, RCOTSender};
#[cfg(target_arch = "wasm32")]
use rand::{Rng, SeedableRng, rngs::StdRng};

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

// ============================================================================
// WASM-compatible async byte channel (no tokio dependency)
// ============================================================================

/// Shared state for one direction of the channel
#[cfg(target_arch = "wasm32")]
struct ChannelState {
    buffer: VecDeque<u8>,
    waker: Option<Waker>,
    closed: bool,
}

/// Write half of an async byte channel
#[cfg(target_arch = "wasm32")]
struct ChannelWriter {
    state: Arc<Mutex<ChannelState>>,
    recorded: Option<Arc<Mutex<Vec<u8>>>>,
}

/// Read half of an async byte channel
#[cfg(target_arch = "wasm32")]
struct ChannelReader {
    state: Arc<Mutex<ChannelState>>,
}

#[cfg(target_arch = "wasm32")]
fn byte_channel() -> (ChannelWriter, ChannelReader) {
    let state = Arc::new(Mutex::new(ChannelState {
        buffer: VecDeque::new(),
        waker: None,
        closed: false,
    }));
    (
        ChannelWriter {
            state: state.clone(),
            recorded: None,
        },
        ChannelReader { state },
    )
}

/// Bidirectional async byte stream (reader + writer from different channels)
#[cfg(target_arch = "wasm32")]
struct BiStream {
    reader: ChannelReader,
    writer: ChannelWriter,
}

#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
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
#[cfg(target_arch = "wasm32")]
fn wasm_recording_context(max_frame_length: usize) -> (Context, Context, Arc<Mutex<Vec<u8>>>) {
    let (writer_a, reader_a) = byte_channel();
    let (mut writer_b, reader_b) = byte_channel();

    let recorded = Arc::new(Mutex::new(Vec::new()));
    writer_b.recorded = Some(recorded.clone());

    let stream_0 = BiStream {
        reader: reader_b,
        writer: writer_a,
    };
    let stream_1 = BiStream {
        reader: reader_a,
        writer: writer_b,
    };

    (
        Context::new_single_threaded_with_limit(stream_0, max_frame_length),
        Context::new_single_threaded_with_limit(stream_1, max_frame_length),
        recorded,
    )
}

// ============================================================================
// Benchmark parameters
// ============================================================================

/// Number of OTs to generate per benchmark iteration.
/// Using 1M for WASM (smaller than native 10M due to memory/time constraints).
#[cfg(target_arch = "wasm32")]
const OT_COUNT: usize = 1_000_000;

/// Calculate max frame length based on workload size.
#[cfg(target_arch = "wasm32")]
fn max_frame_length() -> usize {
    // Ferret messages include SPCOT data which can be large
    // Use large buffer for production parameters
    64 * 1024 * 1024 // 64 MB
}

/// Creates the Ferret config for benchmarking.
#[cfg(target_arch = "wasm32")]
fn bench_config() -> FerretConfig {
    FerretConfig::default()
}

// ============================================================================
// ST isolated sender benchmark
// ============================================================================

/// Recorded data needed for deterministic replay.
#[cfg(target_arch = "wasm32")]
struct RecordedData {
    /// Recorded bytes from receiver -> sender.
    bytes: Vec<u8>,
    /// Delta correlation.
    delta: Block,
    /// Seed for IdealRCOTSender.
    cot_seed: Block,
    /// Seed for Ferret sender.
    sender_seed: Block,
}

/// Runs the full Ferret protocol with sender and receiver.
/// Records receiver->sender messages.
#[cfg(target_arch = "wasm32")]
async fn run_protocol_record_receiver(
    ctx_sender: &mut Context,
    ctx_receiver: &mut Context,
    config: FerretConfig,
    delta: Block,
    cot_seed: Block,
    sender_seed: Block,
    receiver_seed: Block,
) {
    let (cot_send, cot_recv) = ideal_rcot(cot_seed, delta);

    let mut sender = Sender::new(config.clone(), sender_seed, cot_send);
    let mut receiver = Receiver::new(config, receiver_seed, cot_recv);

    futures::join!(
        async {
            sender.alloc(OT_COUNT).unwrap();
            let output = sender.queue_send_rcot(OT_COUNT).unwrap();
            sender.flush(ctx_sender).await.unwrap();
            let _ = output.await.unwrap();
        },
        async {
            receiver.alloc(OT_COUNT).unwrap();
            let output = receiver.queue_recv_rcot(OT_COUNT).unwrap();
            receiver.flush(ctx_receiver).await.unwrap();
            let _ = output.await.unwrap();
        }
    );
}

/// Records receiver->sender messages for sender replay.
#[cfg(target_arch = "wasm32")]
async fn record_for_sender(seed: u64) -> RecordedData {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta: Block = rng.random();
    let cot_seed: Block = rng.random();
    let sender_seed: Block = rng.random();
    let receiver_seed: Block = rng.random();

    // ctx_1 (receiver) is recorded, ctx_0 (sender) receives
    let (mut ctx_sender, mut ctx_receiver, recorded) =
        wasm_recording_context(max_frame_length());

    let config = bench_config();

    run_protocol_record_receiver(
        &mut ctx_sender,
        &mut ctx_receiver,
        config,
        delta,
        cot_seed,
        sender_seed,
        receiver_seed,
    )
    .await;

    RecordedData {
        bytes: recorded.lock().unwrap().clone(),
        delta,
        cot_seed,
        sender_seed,
    }
}

/// Runs sender only with replay context.
#[cfg(target_arch = "wasm32")]
async fn run_sender_with_replay(ctx: &mut Context, data: &RecordedData) {
    let (cot_send, _) = ideal_rcot(data.cot_seed, data.delta);
    let config = bench_config();
    let mut sender = Sender::new(config, data.sender_seed, cot_send);

    sender.alloc(OT_COUNT).unwrap();
    let output = sender.queue_send_rcot(OT_COUNT).unwrap();
    sender.flush(ctx).await.unwrap();
    let _ = output.await.unwrap();
}

/// Benchmark isolated Ferret sender with ST context and message replay.
///
/// Records receiver->sender messages once during setup, then benchmarks
/// sender execution in isolation using replay.
///
/// Runs in a web worker because Ferret uses Atomics.wait internally.
///
/// # Arguments
/// * `n` - Number of iterations
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn ferret_sender_st(n: u32) -> BenchResult {
    use wasm_bindgen_futures::JsFuture;

    // Shared slot for benchmark result
    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    web_sys::console::log_1(&"[rust] Starting Ferret sender ST benchmark...".into());

    // Run benchmark on web worker thread (where Atomics.wait is allowed)
    let _handle = web_spawn::spawn(move || {
        web_sys::console::log_1(&"[rust] Ferret ST web_spawn started".into());
        let bench_result = pollster::block_on(async {
            // Workers don't have `window`, use global scope to get performance
            let global = js_sys::global();
            let performance: web_sys::Performance =
                js_sys::Reflect::get(&global, &"performance".into())
                    .expect("performance should exist")
                    .unchecked_into();

            // Record messages once (not timed)
            web_sys::console::log_1(&"[rust] Recording Ferret sender messages...".into());

            let recorded = record_for_sender(0).await;
            web_sys::console::log_1(
                &format!("[rust] Recorded {} bytes", recorded.bytes.len()).into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                if i % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[rust] Ferret ST Iteration {}/{}", i, n).into(),
                    );
                }

                // Timed section: sender replay
                let start = performance.now();

                let mut ctx = replay_st_context(recorded.bytes.clone(), max_frame_length());
                run_sender_with_replay(&mut ctx, &recorded).await;

                total_elapsed_ms += performance.now() - start;
            }

            BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: n as u64 * OT_COUNT as u64,
            }
        });
        *result_clone.lock().unwrap() = Some(bench_result);
        web_sys::console::log_1(&"[rust] Ferret ST benchmark done".into());
    });

    // Initial yield to let the worker start
    JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
        .await
        .unwrap();

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

// ============================================================================
// MT isolated sender benchmark
// ============================================================================

#[cfg(target_arch = "wasm32")]
use mpz_common::context::{
    recording_mt_context_with_spawn_and_limit, replay_mt_context_with_spawn_and_limit,
    Multithread, RecordedMtData,
};

/// Recorded data needed for deterministic MT replay.
#[cfg(target_arch = "wasm32")]
struct RecordedDataMt {
    /// Recorded bytes from receiver -> sender (per channel).
    data: RecordedMtData,
    /// Delta correlation.
    delta: Block,
    /// Seed for IdealRCOTSender.
    cot_seed: Block,
    /// Seed for Ferret sender.
    sender_seed: Block,
}

/// Runs the full Ferret protocol with MT contexts.
/// Records receiver->sender messages.
#[cfg(target_arch = "wasm32")]
async fn run_protocol_record_receiver_mt(
    exec_sender: &mut Multithread,
    exec_receiver: &mut Multithread,
    config: FerretConfig,
    delta: Block,
    cot_seed: Block,
    sender_seed: Block,
    receiver_seed: Block,
) {
    let (cot_send, cot_recv) = ideal_rcot(cot_seed, delta);

    let mut sender = Sender::new(config.clone(), sender_seed, cot_send);
    let mut receiver = Receiver::new(config, receiver_seed, cot_recv);

    let mut ctx_sender = exec_sender.new_context().await.unwrap();
    let mut ctx_receiver = exec_receiver.new_context().await.unwrap();

    futures::join!(
        async {
            sender.alloc(OT_COUNT).unwrap();
            let output = sender.queue_send_rcot(OT_COUNT).unwrap();
            sender.flush(&mut ctx_sender).await.unwrap();
            let _ = output.await.unwrap();
        },
        async {
            receiver.alloc(OT_COUNT).unwrap();
            let output = receiver.queue_recv_rcot(OT_COUNT).unwrap();
            receiver.flush(&mut ctx_receiver).await.unwrap();
            let _ = output.await.unwrap();
        }
    );
}

/// Records receiver->sender messages for MT sender replay.
#[cfg(target_arch = "wasm32")]
async fn record_for_sender_mt(seed: u64, concurrency: usize) -> RecordedDataMt {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta: Block = rng.random();
    let cot_seed: Block = rng.random();
    let sender_seed: Block = rng.random();
    let receiver_seed: Block = rng.random();

    // exec_1 (receiver) is recorded, exec_0 (sender) receives
    let (mut exec_sender, mut exec_receiver, recorded) = recording_mt_context_with_spawn_and_limit(
        1024 * 1024,
        max_frame_length(),
        concurrency,
        |f| {
            let _ = web_spawn::spawn(f);
            Ok(())
        },
    );

    let config = bench_config();

    run_protocol_record_receiver_mt(
        &mut exec_sender,
        &mut exec_receiver,
        config,
        delta,
        cot_seed,
        sender_seed,
        receiver_seed,
    )
    .await;

    RecordedDataMt {
        data: recorded.lock().unwrap().clone(),
        delta,
        cot_seed,
        sender_seed,
    }
}

/// Runs MT sender only with replay context.
#[cfg(target_arch = "wasm32")]
async fn run_sender_with_replay_mt(exec: &mut Multithread, data: &RecordedDataMt) {
    let (cot_send, _) = ideal_rcot(data.cot_seed, data.delta);
    let config = bench_config();
    let mut sender = Sender::new(config, data.sender_seed, cot_send);

    let mut ctx = exec.new_context().await.unwrap();

    sender.alloc(OT_COUNT).unwrap();
    let output = sender.queue_send_rcot(OT_COUNT).unwrap();
    sender.flush(&mut ctx).await.unwrap();
    let _ = output.await.unwrap();
}

/// Benchmark isolated Ferret sender with MT context and message replay.
///
/// Records receiver->sender messages once during setup using MT contexts,
/// then benchmarks sender execution in isolation using MT replay.
///
/// # Arguments
/// * `n` - Number of iterations
/// * `concurrency` - Maximum parallelism level (max children per parent thread)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn ferret_sender_mt(n: u32, concurrency: u32) -> BenchResult {
    use wasm_bindgen_futures::JsFuture;

    // Shared slot for benchmark result
    let result: Arc<Mutex<Option<BenchResult>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    web_sys::console::log_1(&"[rust] Starting Ferret sender MT benchmark...".into());

    // Run benchmark on web worker thread (where Atomics.wait is allowed)
    let _handle = web_spawn::spawn(move || {
        web_sys::console::log_1(&"[rust] Ferret MT web_spawn started".into());
        let bench_result = pollster::block_on(async {
            // Workers don't have `window`, use global scope to get performance
            let global = js_sys::global();
            let performance: web_sys::Performance =
                js_sys::Reflect::get(&global, &"performance".into())
                    .expect("performance should exist")
                    .unchecked_into();

            // Record messages once (not timed)
            web_sys::console::log_1(
                &format!(
                    "[rust] Recording MT Ferret messages, concurrency={}...",
                    concurrency
                )
                .into(),
            );

            let recorded = record_for_sender_mt(0, concurrency as usize).await;
            let total_bytes: usize = recorded.data.channels.values().map(|v| v.len()).sum();
            web_sys::console::log_1(
                &format!(
                    "[rust] Recorded {} channels, {} total bytes",
                    recorded.data.channels.len(),
                    total_bytes
                )
                .into(),
            );

            let mut total_elapsed_ms = 0.0;

            for i in 0..n {
                if i % 10 == 0 {
                    web_sys::console::log_1(
                        &format!("[rust] Ferret MT Iteration {}/{}", i, n).into(),
                    );
                }

                // Timed section: sender replay with MT context
                let start = performance.now();

                let mut exec = replay_mt_context_with_spawn_and_limit(
                    recorded.data.clone(),
                    max_frame_length(),
                    concurrency as usize,
                    |f| {
                        let _ = web_spawn::spawn(f);
                        Ok(())
                    },
                );
                run_sender_with_replay_mt(&mut exec, &recorded).await;

                total_elapsed_ms += performance.now() - start;
            }

            BenchResult {
                elapsed_ms: total_elapsed_ms,
                and_gates: n as u64 * OT_COUNT as u64,
            }
        });
        *result_clone.lock().unwrap() = Some(bench_result);
        web_sys::console::log_1(&"[rust] Ferret MT benchmark done".into());
    });

    // Initial yield to let the worker start
    JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
        .await
        .unwrap();

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
