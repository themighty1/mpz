use std::{
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
}
