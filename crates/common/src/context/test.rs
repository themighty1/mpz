use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context as TaskContext, Poll},
};

use futures::{AsyncRead, AsyncWrite};
use serio::channel::duplex;
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt};
use uid_mux::test_utils::test_framed_mux;

use crate::{
    context::{Context, Multithread, SpawnError},
    io::Io,
    mux::Mux,
    ThreadId,
};

/// Creates a pair of single-threaded contexts using memory I/O channels.
pub fn test_st_context(io_buffer: usize) -> (Context, Context) {
    let (io_0, io_1) = duplex(io_buffer);

    (
        Context::from_io(Io::from_channel(io_0)),
        Context::from_io(Io::from_channel(io_1)),
    )
}

/// Creates a pair of multi-threaded contexts using multiplexed I/O channels.
pub fn test_mt_context(io_buffer: usize) -> (Multithread, Multithread) {
    let (mux_0, mux_1) = test_framed_mux(io_buffer);

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder().mux_internal(mux_0).build().unwrap(),
        Multithread::builder().mux_internal(mux_1).build().unwrap(),
    )
}

/// Creates a pair of multi-threaded contexts with a custom spawn handler.
///
/// This is useful for WASM environments where `std::thread::spawn` is not available
/// and a custom spawner like `web_spawn` is needed.
pub fn test_mt_context_with_spawn<F>(io_buffer: usize, spawn: F) -> (Multithread, Multithread)
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let (mux_0, mux_1) = test_framed_mux(io_buffer);

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder()
            .spawn_handler(spawn.clone())
            .mux_internal(mux_0)
            .build()
            .unwrap(),
        Multithread::builder()
            .spawn_handler(spawn)
            .mux_internal(mux_1)
            .build()
            .unwrap(),
    )
}

/// Creates a pair of multi-threaded contexts with a custom spawn handler and concurrency.
///
/// Like [`test_mt_context_with_spawn`], but allows configuring the maximum concurrency
/// level (number of worker threads) per context.
pub fn test_mt_context_with_concurrency<F>(
    io_buffer: usize,
    concurrency: usize,
    spawn: F,
) -> (Multithread, Multithread)
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let (mux_0, mux_1) = test_framed_mux(io_buffer);

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder()
            .concurrency(concurrency)
            .spawn_handler(spawn.clone())
            .mux_internal(mux_0)
            .build()
            .unwrap(),
        Multithread::builder()
            .concurrency(concurrency)
            .spawn_handler(spawn)
            .mux_internal(mux_1)
            .build()
            .unwrap(),
    )
}

/// A duplex stream that records all bytes written.
///
/// Used for recording protocol messages for replay in isolated benchmarks.
pub struct RecordingDuplex {
    inner: Compat<tokio::io::DuplexStream>,
    recorded: Arc<Mutex<Vec<u8>>>,
}

impl RecordingDuplex {
    /// Creates a new recording duplex wrapping the given stream.
    pub fn new(inner: tokio::io::DuplexStream, recorded: Arc<Mutex<Vec<u8>>>) -> Self {
        Self {
            inner: inner.compat(),
            recorded,
        }
    }
}

impl AsyncRead for RecordingDuplex {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for RecordingDuplex {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &result {
            self.recorded.lock().unwrap().extend_from_slice(&buf[..*n]);
        }
        result
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_close(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }
}

/// A duplex stream that replays recorded bytes on read and discards writes.
///
/// Used for replay-based isolated benchmarking where one party receives
/// pre-recorded messages without a real counterparty.
pub struct ReplayDuplex {
    /// Recorded bytes to replay.
    data: std::io::Cursor<Vec<u8>>,
}

impl ReplayDuplex {
    /// Creates a new replay duplex from recorded bytes.
    pub fn new(recorded: Vec<u8>) -> Self {
        Self {
            data: std::io::Cursor::new(recorded),
        }
    }
}

impl AsyncRead for ReplayDuplex {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        use std::io::Read;
        Poll::Ready(self.data.read(buf))
    }
}

impl AsyncWrite for ReplayDuplex {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        // Discard writes - just report success
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Creates a single-threaded context that replays recorded bytes.
///
/// The context will read from the recorded bytes and discard all writes.
/// Use this for replay-based isolated benchmarking of a single party.
///
/// # Arguments
///
/// * `recorded` - The recorded bytes to replay.
/// * `max_frame_length` - Maximum frame size in bytes.
pub fn replay_st_context(recorded: Vec<u8>, max_frame_length: usize) -> Context {
    let replay = ReplayDuplex::new(recorded);
    Context::new_single_threaded_with_limit(replay, max_frame_length)
}

/// Creates a pair of single-threaded contexts where writes from ctx_1 to ctx_0 are recorded.
///
/// Returns `(ctx_0, ctx_1, recorded)` where `recorded` contains all bytes written by ctx_1.
/// This is useful for recording protocol messages for replay in isolated benchmarks.
///
/// Note: Unlike `test_st_context`, this uses framed byte transport instead of memory channels,
/// which may have slightly different performance characteristics.
pub fn recording_st_context(io_buffer: usize) -> (Context, Context, Arc<Mutex<Vec<u8>>>) {
    let (io_0, io_1) = tokio::io::duplex(io_buffer);

    let recorded = Arc::new(Mutex::new(Vec::new()));
    let recording_io_1 = RecordingDuplex::new(io_1, recorded.clone());

    (
        Context::new_single_threaded(io_0.compat()),
        Context::new_single_threaded(recording_io_1),
        recorded,
    )
}

/// Creates a pair of single-threaded contexts with a custom frame limit where writes from
/// ctx_1 to ctx_0 are recorded.
///
/// Like [`recording_st_context`], but allows setting a custom maximum frame size.
/// Use this when protocol messages exceed the default 8MB frame limit.
///
/// # Arguments
///
/// * `io_buffer` - Size of the I/O buffer.
/// * `max_frame_length` - Maximum frame size in bytes.
pub fn recording_st_context_with_limit(
    io_buffer: usize,
    max_frame_length: usize,
) -> (Context, Context, Arc<Mutex<Vec<u8>>>) {
    let (io_0, io_1) = tokio::io::duplex(io_buffer);

    let recorded = Arc::new(Mutex::new(Vec::new()));
    let recording_io_1 = RecordingDuplex::new(io_1, recorded.clone());

    (
        Context::new_single_threaded_with_limit(io_0.compat(), max_frame_length),
        Context::new_single_threaded_with_limit(recording_io_1, max_frame_length),
        recorded,
    )
}

// ============================================================================
// Multi-threaded recording/replay infrastructure
// ============================================================================

/// Recorded data for multi-threaded context replay.
///
/// Stores bytes recorded from each channel, keyed by thread ID.
#[derive(Debug, Clone, Default)]
pub struct RecordedMtData {
    /// Recorded bytes per channel.
    pub channels: HashMap<ThreadId, Vec<u8>>,
}

/// Shared state for recording test mux.
///
/// Uses byte-based tokio duplex channels (like ST recording) instead of
/// type-erased MemoryDuplex, allowing byte-level recording.
struct RecordingMuxState {
    /// Channels waiting to be opened by role A.
    waiting_a: HashMap<ThreadId, Compat<tokio::io::DuplexStream>>,
    /// Channels waiting to be opened by role B.
    waiting_b: HashMap<ThreadId, RecordingDuplexMt>,
    /// Track which channels have been opened.
    opened: std::collections::HashSet<ThreadId>,
}

impl Default for RecordingMuxState {
    fn default() -> Self {
        Self {
            waiting_a: HashMap::new(),
            waiting_b: HashMap::new(),
            opened: std::collections::HashSet::new(),
        }
    }
}

/// Role in the recording mux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordingRole {
    /// Role A: receives from role B (no recording on this side).
    A,
    /// Role B: sends to role A (recording enabled).
    B,
}

/// A test mux that records writes from role B using byte-level recording.
///
/// Similar to `TestFramedMux` but uses tokio duplex (byte streams) instead of
/// MemoryDuplex (type-erased channels), allowing us to record raw bytes.
#[derive(Clone)]
struct RecordingTestMux {
    role: RecordingRole,
    buffer: usize,
    max_frame_length: Option<usize>,
    state: Arc<Mutex<RecordingMuxState>>,
    recorded: Arc<Mutex<RecordedMtData>>,
}

impl std::fmt::Debug for RecordingTestMux {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingTestMux")
            .field("role", &self.role)
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl Mux for RecordingTestMux {
    fn open(
        &self,
        id: ThreadId,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Io, std::io::Error>> + Send>> {
        let mux = self.clone();
        Box::pin(async move {
            let mut state = mux.state.lock().unwrap();

            // Check if channel already exists from the other side
            match mux.role {
                RecordingRole::A => {
                    if let Some(stream) = state.waiting_a.remove(&id) {
                        return Ok(if let Some(limit) = mux.max_frame_length {
                            Io::from_io_with_limit(stream, limit)
                        } else {
                            Io::from_io(stream)
                        });
                    }
                }
                RecordingRole::B => {
                    if let Some(recording_stream) = state.waiting_b.remove(&id) {
                        return Ok(if let Some(limit) = mux.max_frame_length {
                            Io::from_io_with_limit(recording_stream, limit)
                        } else {
                            Io::from_io(recording_stream)
                        });
                    }
                }
            }

            // Check for duplicate
            if !state.opened.insert(id.clone()) {
                return Err(std::io::Error::other("duplicate stream id"));
            }

            // Create new byte-based channel pair
            let (stream_a, stream_b) = tokio::io::duplex(mux.buffer);

            // Role B's writes are recorded
            let recorded_for_channel = mux.recorded.clone();
            let channel_id = id.clone();

            match mux.role {
                RecordingRole::A => {
                    // A gets plain stream, B gets recording stream
                    let recording_stream =
                        RecordingDuplexWithId::new(stream_b, channel_id, recorded_for_channel);
                    state.waiting_b.insert(id, recording_stream.into_recording_duplex());
                    Ok(if let Some(limit) = mux.max_frame_length {
                        Io::from_io_with_limit(stream_a.compat(), limit)
                    } else {
                        Io::from_io(stream_a.compat())
                    })
                }
                RecordingRole::B => {
                    // B gets recording stream, A gets plain stream
                    state.waiting_a.insert(id, stream_a.compat());
                    let recording_stream =
                        RecordingDuplexWithId::new(stream_b, channel_id, recorded_for_channel);
                    Ok(if let Some(limit) = mux.max_frame_length {
                        Io::from_io_with_limit(recording_stream.into_recording_duplex(), limit)
                    } else {
                        Io::from_io(recording_stream.into_recording_duplex())
                    })
                }
            }
        })
    }
}

/// Helper to create RecordingDuplex with per-channel recording.
struct RecordingDuplexWithId {
    inner: tokio::io::DuplexStream,
    channel_id: ThreadId,
    recorded: Arc<Mutex<RecordedMtData>>,
}

impl RecordingDuplexWithId {
    fn new(
        inner: tokio::io::DuplexStream,
        channel_id: ThreadId,
        recorded: Arc<Mutex<RecordedMtData>>,
    ) -> Self {
        Self {
            inner,
            channel_id,
            recorded,
        }
    }

    fn into_recording_duplex(self) -> RecordingDuplexMt {
        RecordingDuplexMt {
            inner: self.inner.compat(),
            channel_id: self.channel_id,
            recorded: self.recorded,
        }
    }
}

/// A duplex stream that records all bytes written, tagged by channel ID.
///
/// Like `RecordingDuplex` but stores bytes per-channel for MT contexts.
struct RecordingDuplexMt {
    inner: Compat<tokio::io::DuplexStream>,
    channel_id: ThreadId,
    recorded: Arc<Mutex<RecordedMtData>>,
}

impl AsyncRead for RecordingDuplexMt {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for RecordingDuplexMt {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &result {
            let mut data = self.recorded.lock().unwrap();
            data.channels
                .entry(self.channel_id.clone())
                .or_default()
                .extend_from_slice(&buf[..*n]);
        }
        result
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_close(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }
}

/// Creates a pair of recording test mux instances.
///
/// Writes from mux_1 (role B) are recorded in the returned `RecordedMtData`.
fn recording_test_mux(
    buffer: usize,
    max_frame_length: Option<usize>,
) -> (RecordingTestMux, RecordingTestMux, Arc<Mutex<RecordedMtData>>) {
    let state = Arc::new(Mutex::new(RecordingMuxState::default()));
    let recorded = Arc::new(Mutex::new(RecordedMtData::default()));

    (
        RecordingTestMux {
            role: RecordingRole::A,
            buffer,
            max_frame_length,
            state: state.clone(),
            recorded: recorded.clone(),
        },
        RecordingTestMux {
            role: RecordingRole::B,
            buffer,
            max_frame_length,
            state,
            recorded: recorded.clone(),
        },
        recorded,
    )
}

/// A test mux that replays recorded data.
///
/// Provides channels that read from pre-recorded data and discard writes.
#[derive(Debug, Clone)]
struct ReplayTestMux {
    recorded: Arc<Mutex<RecordedMtData>>,
    max_frame_length: Option<usize>,
}

impl ReplayTestMux {
    /// Creates a new replay mux from recorded data.
    fn new(recorded: RecordedMtData, max_frame_length: Option<usize>) -> Self {
        Self {
            recorded: Arc::new(Mutex::new(recorded)),
            max_frame_length,
        }
    }
}

impl Mux for ReplayTestMux {
    fn open(
        &self,
        id: ThreadId,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Io, std::io::Error>> + Send>> {
        let recorded = self.recorded.clone();
        let max_frame_length = self.max_frame_length;
        Box::pin(async move {
            let data = {
                let mut rec = recorded.lock().unwrap();
                rec.channels.remove(&id).unwrap_or_default()
            };
            let replay = ReplayDuplex::new(data);
            if let Some(limit) = max_frame_length {
                Ok(Io::from_io_with_limit(replay, limit))
            } else {
                Ok(Io::from_io(replay))
            }
        })
    }
}

/// Creates a pair of multi-threaded contexts where writes from ctx_1 are recorded.
///
/// Returns `(ctx_0, ctx_1, recorded)` where `recorded` contains all bytes written
/// by ctx_1 on each channel.
///
/// # Arguments
///
/// * `io_buffer` - Size of the I/O buffer per channel.
pub fn recording_mt_context(
    io_buffer: usize,
) -> (Multithread, Multithread, Arc<Mutex<RecordedMtData>>) {
    let (mux_0, mux_1, recorded) = recording_test_mux(io_buffer, None);

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder().mux_internal(mux_0).build().unwrap(),
        Multithread::builder().mux_internal(mux_1).build().unwrap(),
        recorded,
    )
}

/// Creates a pair of multi-threaded contexts where writes from ctx_1 are recorded,
/// with a custom frame length limit.
///
/// # Arguments
///
/// * `io_buffer` - Size of the I/O buffer per channel.
/// * `max_frame_length` - Maximum frame size in bytes.
pub fn recording_mt_context_with_limit(
    io_buffer: usize,
    max_frame_length: usize,
) -> (Multithread, Multithread, Arc<Mutex<RecordedMtData>>) {
    let (mux_0, mux_1, recorded) = recording_test_mux(io_buffer, Some(max_frame_length));

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder().mux_internal(mux_0).build().unwrap(),
        Multithread::builder().mux_internal(mux_1).build().unwrap(),
        recorded,
    )
}

/// Creates a pair of multi-threaded contexts with custom spawn handler where writes
/// from ctx_1 are recorded.
///
/// # Arguments
///
/// * `io_buffer` - Size of the I/O buffer per channel.
/// * `spawn` - Custom spawn handler for worker threads.
pub fn recording_mt_context_with_spawn<F>(
    io_buffer: usize,
    spawn: F,
) -> (Multithread, Multithread, Arc<Mutex<RecordedMtData>>)
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let (mux_0, mux_1, recorded) = recording_test_mux(io_buffer, None);

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder()
            .spawn_handler(spawn.clone())
            .mux_internal(mux_0)
            .build()
            .unwrap(),
        Multithread::builder()
            .spawn_handler(spawn)
            .mux_internal(mux_1)
            .build()
            .unwrap(),
        recorded,
    )
}

/// Creates a pair of multi-threaded contexts with custom spawn handler and frame limit
/// where writes from ctx_1 are recorded.
///
/// # Arguments
///
/// * `io_buffer` - Size of the I/O buffer per channel.
/// * `max_frame_length` - Maximum frame size in bytes.
/// * `concurrency` - Maximum parallelism level (max children per parent thread).
/// * `spawn` - Custom spawn handler for worker threads.
pub fn recording_mt_context_with_spawn_and_limit<F>(
    io_buffer: usize,
    max_frame_length: usize,
    concurrency: usize,
    spawn: F,
) -> (Multithread, Multithread, Arc<Mutex<RecordedMtData>>)
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let (mux_0, mux_1, recorded) = recording_test_mux(io_buffer, Some(max_frame_length));

    let mux_0: Box<dyn Mux + Send> = Box::new(mux_0);
    let mux_1: Box<dyn Mux + Send> = Box::new(mux_1);

    (
        Multithread::builder()
            .spawn_handler(spawn.clone())
            .concurrency(concurrency)
            .mux_internal(mux_0)
            .build()
            .unwrap(),
        Multithread::builder()
            .spawn_handler(spawn)
            .concurrency(concurrency)
            .mux_internal(mux_1)
            .build()
            .unwrap(),
        recorded,
    )
}

/// Creates a multi-threaded context that replays recorded data.
///
/// The context will read from the recorded bytes and discard all writes.
/// Use this for replay-based isolated benchmarking of a single party in MT mode.
///
/// # Arguments
///
/// * `recorded` - The recorded data to replay (per-channel).
pub fn replay_mt_context(recorded: RecordedMtData) -> Multithread {
    let mux = ReplayTestMux::new(recorded, None);
    let mux: Box<dyn Mux + Send> = Box::new(mux);

    Multithread::builder().mux_internal(mux).build().unwrap()
}

/// Creates a multi-threaded context that replays recorded data with a custom frame length limit.
///
/// # Arguments
///
/// * `recorded` - The recorded data to replay (per-channel).
/// * `max_frame_length` - Maximum frame size in bytes.
pub fn replay_mt_context_with_limit(
    recorded: RecordedMtData,
    max_frame_length: usize,
) -> Multithread {
    let mux = ReplayTestMux::new(recorded, Some(max_frame_length));
    let mux: Box<dyn Mux + Send> = Box::new(mux);

    Multithread::builder().mux_internal(mux).build().unwrap()
}

/// Creates a multi-threaded context that replays recorded data with custom spawn handler.
///
/// # Arguments
///
/// * `recorded` - The recorded data to replay (per-channel).
/// * `spawn` - Custom spawn handler for worker threads.
pub fn replay_mt_context_with_spawn<F>(recorded: RecordedMtData, spawn: F) -> Multithread
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let mux = ReplayTestMux::new(recorded, None);
    let mux: Box<dyn Mux + Send> = Box::new(mux);

    Multithread::builder()
        .spawn_handler(spawn)
        .mux_internal(mux)
        .build()
        .unwrap()
}

/// Creates a multi-threaded context that replays recorded data with custom spawn handler
/// and frame length limit.
///
/// # Arguments
///
/// * `recorded` - The recorded data to replay (per-channel).
/// * `max_frame_length` - Maximum frame size in bytes.
/// * `concurrency` - Maximum parallelism level (max children per parent thread).
/// * `spawn` - Custom spawn handler for worker threads.
pub fn replay_mt_context_with_spawn_and_limit<F>(
    recorded: RecordedMtData,
    max_frame_length: usize,
    concurrency: usize,
    spawn: F,
) -> Multithread
where
    F: FnMut(Box<dyn FnOnce() + Send>) -> Result<(), SpawnError> + Clone + Send + 'static,
{
    let mux = ReplayTestMux::new(recorded, Some(max_frame_length));
    let mux: Box<dyn Mux + Send> = Box::new(mux);

    Multithread::builder()
        .spawn_handler(spawn)
        .concurrency(concurrency)
        .mux_internal(mux)
        .build()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serio::{SinkExt, stream::IoStreamExt};

    #[tokio::test]
    async fn test_recording_st_context() {
        let (mut ctx_0, mut ctx_1, recorded) = recording_st_context(1024 * 1024);

        // Send a message from ctx_1 to ctx_0 (this should be recorded)
        ctx_1.io_mut().send(42u32).await.unwrap();
        ctx_1.io_mut().send(vec![1u8, 2, 3, 4]).await.unwrap();

        // Receive on ctx_0
        let msg1: u32 = ctx_0.io_mut().expect_next().await.unwrap();
        let msg2: Vec<u8> = ctx_0.io_mut().expect_next().await.unwrap();

        assert_eq!(msg1, 42);
        assert_eq!(msg2, vec![1, 2, 3, 4]);

        // Verify something was recorded
        let recorded_bytes = recorded.lock().unwrap();
        assert!(!recorded_bytes.is_empty(), "should have recorded bytes");
    }

    #[tokio::test]
    async fn test_recording_determinism() {
        // Run the same protocol twice and verify recorded bytes are identical
        async fn run_protocol(ctx_0: &mut Context, ctx_1: &mut Context) {
            ctx_1.io_mut().send(123u64).await.unwrap();
            ctx_1.io_mut().send("hello".to_string()).await.unwrap();
            ctx_1.io_mut().send(vec![10u8; 100]).await.unwrap();

            let _: u64 = ctx_0.io_mut().expect_next().await.unwrap();
            let _: String = ctx_0.io_mut().expect_next().await.unwrap();
            let _: Vec<u8> = ctx_0.io_mut().expect_next().await.unwrap();
        }

        // First run
        let (mut ctx_0a, mut ctx_1a, recorded_a) = recording_st_context(1024 * 1024);
        run_protocol(&mut ctx_0a, &mut ctx_1a).await;

        // Second run
        let (mut ctx_0b, mut ctx_1b, recorded_b) = recording_st_context(1024 * 1024);
        run_protocol(&mut ctx_0b, &mut ctx_1b).await;

        // Verify recordings are identical
        let bytes_a = recorded_a.lock().unwrap();
        let bytes_b = recorded_b.lock().unwrap();
        assert_eq!(*bytes_a, *bytes_b, "recordings should be deterministic");
    }

    #[tokio::test]
    async fn test_recording_mt_context() {
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        // Send a message from ctx_1 to ctx_0 (this should be recorded)
        ctx_1.io_mut().send(42u32).await.unwrap();
        ctx_1.io_mut().send(vec![1u8, 2, 3, 4]).await.unwrap();

        // Receive on ctx_0
        let msg1: u32 = ctx_0.io_mut().expect_next().await.unwrap();
        let msg2: Vec<u8> = ctx_0.io_mut().expect_next().await.unwrap();

        assert_eq!(msg1, 42);
        assert_eq!(msg2, vec![1, 2, 3, 4]);

        // Verify something was recorded
        let recorded_data = recorded.lock().unwrap();
        assert!(!recorded_data.channels.is_empty(), "should have recorded channels");

        // Check that the recorded channel has data
        let total_bytes: usize = recorded_data.channels.values().map(|v| v.len()).sum();
        assert!(total_bytes > 0, "should have recorded bytes");
    }

    #[tokio::test]
    async fn test_replay_mt_context() {
        // First: record some messages
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        // Send messages from ctx_1 (verifier) to ctx_0 (prover)
        ctx_1.io_mut().send(42u32).await.unwrap();
        ctx_1.io_mut().send("hello".to_string()).await.unwrap();

        // Receive on ctx_0
        let _: u32 = ctx_0.io_mut().expect_next().await.unwrap();
        let _: String = ctx_0.io_mut().expect_next().await.unwrap();

        // Get recorded data
        let recorded_data = recorded.lock().unwrap().clone();

        // Now replay to a new context
        let mut replay_exec = replay_mt_context(recorded_data);
        let mut replay_ctx = replay_exec.new_context().await.unwrap();

        // Should be able to receive the same messages from replay
        let msg1: u32 = replay_ctx.io_mut().expect_next().await.unwrap();
        let msg2: String = replay_ctx.io_mut().expect_next().await.unwrap();

        assert_eq!(msg1, 42);
        assert_eq!(msg2, "hello");
    }

    #[tokio::test]
    async fn test_recording_mt_multiple_channels() {
        // Test that recording works correctly with multiple channels via ctx.try_join()
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        // Run both sides concurrently
        let (result, send_result) = futures::join!(
            // ctx_0 uses try_join to receive on multiple channels
            ctx_0.try_join(
                async |ctx: &mut Context| {
                    let msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: u64 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
            ),
            // ctx_1 uses try_join to send on multiple channels
            ctx_1.try_join(
                async |ctx: &mut Context| {
                    ctx.io_mut().send(42u32).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send(123u64).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
            )
        );

        let (msg_a, msg_b) = result.unwrap().unwrap();
        send_result.unwrap().unwrap();

        assert_eq!(msg_a, 42);
        assert_eq!(msg_b, 123);

        // Verify multiple channels were recorded
        let recorded_data = recorded.lock().unwrap();
        println!(
            "Recorded {} channels: {:?}",
            recorded_data.channels.len(),
            recorded_data.channels.keys().collect::<Vec<_>>()
        );

        // Should have more than 1 channel (main + at least one child)
        assert!(
            recorded_data.channels.len() > 1,
            "expected multiple channels, got {}",
            recorded_data.channels.len()
        );

        // Each channel should have some data
        for (id, bytes) in &recorded_data.channels {
            println!("Channel {:?}: {} bytes", id, bytes.len());
            assert!(bytes.len() > 0, "channel {:?} should have data", id);
        }
    }

    #[tokio::test]
    async fn test_recording_mt_try_join3() {
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        let (result, send_result) = futures::join!(
            ctx_0.try_join3(
                async |ctx: &mut Context| {
                    let msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: u64 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: String = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
            ),
            ctx_1.try_join3(
                async |ctx: &mut Context| {
                    ctx.io_mut().send(42u32).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send(123u64).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send("hello".to_string()).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
            )
        );

        let (msg_a, msg_b, msg_c) = result.unwrap().unwrap();
        send_result.unwrap().unwrap();

        assert_eq!(msg_a, 42);
        assert_eq!(msg_b, 123);
        assert_eq!(msg_c, "hello");

        let recorded_data = recorded.lock().unwrap();
        println!(
            "try_join3: Recorded {} channels: {:?}",
            recorded_data.channels.len(),
            recorded_data.channels.keys().collect::<Vec<_>>()
        );

        // Should have 3 channels (one per fork)
        assert!(
            recorded_data.channels.len() >= 3,
            "expected at least 3 channels, got {}",
            recorded_data.channels.len()
        );
    }

    #[tokio::test]
    async fn test_recording_mt_try_join4() {
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        let (result, send_result) = futures::join!(
            ctx_0.try_join4(
                async |ctx: &mut Context| {
                    let msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: u64 = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: String = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
                async |ctx: &mut Context| {
                    let msg: Vec<u8> = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
            ),
            ctx_1.try_join4(
                async |ctx: &mut Context| {
                    ctx.io_mut().send(42u32).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send(123u64).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send("hello".to_string()).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send(vec![1u8, 2, 3]).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
            )
        );

        let (msg_a, msg_b, msg_c, msg_d) = result.unwrap().unwrap();
        send_result.unwrap().unwrap();

        assert_eq!(msg_a, 42);
        assert_eq!(msg_b, 123);
        assert_eq!(msg_c, "hello");
        assert_eq!(msg_d, vec![1u8, 2, 3]);

        let recorded_data = recorded.lock().unwrap();
        println!(
            "try_join4: Recorded {} channels: {:?}",
            recorded_data.channels.len(),
            recorded_data.channels.keys().collect::<Vec<_>>()
        );

        assert!(
            recorded_data.channels.len() >= 4,
            "expected at least 4 channels, got {}",
            recorded_data.channels.len()
        );
    }

    #[tokio::test]
    async fn test_recording_mt_map() {
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        // Create items to map over
        let items: Vec<u32> = (0..8).collect();

        let (recv_results, send_results) = futures::join!(
            ctx_0.map(
                items.clone(),
                async |ctx: &mut Context, _item: u32| {
                    let msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                    msg
                },
                |_| 1, // weight
            ),
            ctx_1.map(
                items,
                async |ctx: &mut Context, item: u32| {
                    ctx.io_mut().send(item * 10).await.unwrap();
                },
                |_| 1,
            )
        );

        let recv_results = recv_results.unwrap();
        send_results.unwrap();

        // Results should be [0, 10, 20, 30, 40, 50, 60, 70] (order may vary)
        let mut sorted_results = recv_results.clone();
        sorted_results.sort();
        assert_eq!(sorted_results, vec![0, 10, 20, 30, 40, 50, 60, 70]);

        let recorded_data = recorded.lock().unwrap();
        println!(
            "map: Recorded {} channels: {:?}",
            recorded_data.channels.len(),
            recorded_data.channels.keys().collect::<Vec<_>>()
        );

        // Should have multiple channels (distributed across workers)
        assert!(
            recorded_data.channels.len() > 1,
            "expected multiple channels from map, got {}",
            recorded_data.channels.len()
        );
    }

    #[tokio::test]
    async fn test_recording_mt_nested_try_join() {
        let (mut exec_0, mut exec_1, recorded) = recording_mt_context(1024 * 1024);

        let mut ctx_0 = exec_0.new_context().await.unwrap();
        let mut ctx_1 = exec_1.new_context().await.unwrap();

        let (result, send_result) = futures::join!(
            // Outer try_join
            ctx_0.try_join(
                // Inner try_join in first branch
                async |ctx: &mut Context| {
                    // Receive the outer child's message first
                    let outer_msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                    assert_eq!(outer_msg, 999);
                    let inner_result = ctx
                        .try_join(
                            async |ctx: &mut Context| {
                                let msg: u32 = ctx.io_mut().expect_next().await.unwrap();
                                Ok::<_, std::io::Error>(msg)
                            },
                            async |ctx: &mut Context| {
                                let msg: u64 = ctx.io_mut().expect_next().await.unwrap();
                                Ok::<_, std::io::Error>(msg)
                            },
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    Ok::<_, std::io::Error>(inner_result)
                },
                // Simple receive in second branch
                async |ctx: &mut Context| {
                    let msg: String = ctx.io_mut().expect_next().await.unwrap();
                    Ok::<_, std::io::Error>(msg)
                },
            ),
            // Matching structure on sender side
            ctx_1.try_join(
                async |ctx: &mut Context| {
                    // Write something on outer child before inner try_join
                    ctx.io_mut().send(999u32).await.unwrap();
                    ctx.try_join(
                        async |ctx: &mut Context| {
                            ctx.io_mut().send(42u32).await.unwrap();
                            Ok::<_, std::io::Error>(())
                        },
                        async |ctx: &mut Context| {
                            ctx.io_mut().send(123u64).await.unwrap();
                            Ok::<_, std::io::Error>(())
                        },
                    )
                    .await
                    .unwrap()
                    .unwrap();
                    Ok::<_, std::io::Error>(())
                },
                async |ctx: &mut Context| {
                    ctx.io_mut().send("nested".to_string()).await.unwrap();
                    Ok::<_, std::io::Error>(())
                },
            )
        );

        let ((msg_a, msg_b), msg_c) = result.unwrap().unwrap();
        send_result.unwrap().unwrap();

        assert_eq!(msg_a, 42);
        assert_eq!(msg_b, 123);
        assert_eq!(msg_c, "nested");

        let recorded_data = recorded.lock().unwrap();
        println!(
            "nested: Recorded {} channels: {:?}",
            recorded_data.channels.len(),
            recorded_data.channels.keys().collect::<Vec<_>>()
        );

        // Should have at least 4 channels (outer 2 + inner 2)
        assert!(
            recorded_data.channels.len() >= 4,
            "expected at least 4 channels from nested try_join, got {}",
            recorded_data.channels.len()
        );
    }
}
