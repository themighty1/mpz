use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll, Waker};

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use futures::{AsyncRead, AsyncWrite};
use mpz_common::context::{recording_st_context_with_limit, test_st_context};
use mpz_common::io::Io;
use pollster::FutureExt;
use serde::{Deserialize, Serialize};
use serio::{SinkExt, stream::IoStreamExt};
use tokio::{
    io::duplex,
    net::{TcpListener, TcpStream},
    sync::Mutex,
};
use tokio_util::compat::TokioAsyncReadCompatExt;

// ============================================================================
// BiStream implementation (same as wasm-bench for comparison)
// ============================================================================

struct ChannelState {
    buffer: VecDeque<u8>,
    waker: Option<Waker>,
    closed: bool,
}

struct ChannelWriter {
    state: Arc<std::sync::Mutex<ChannelState>>,
}

struct ChannelReader {
    state: Arc<std::sync::Mutex<ChannelState>>,
}

fn byte_channel() -> (ChannelWriter, ChannelReader) {
    let state = Arc::new(std::sync::Mutex::new(ChannelState {
        buffer: VecDeque::new(),
        waker: None,
        closed: false,
    }));
    (
        ChannelWriter { state: state.clone() },
        ChannelReader { state },
    )
}

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

fn bistream_pair() -> (BiStream, BiStream) {
    let (writer_a, reader_a) = byte_channel();
    let (writer_b, reader_b) = byte_channel();

    let stream_0 = BiStream { reader: reader_b, writer: writer_a };
    let stream_1 = BiStream { reader: reader_a, writer: writer_b };

    (stream_0, stream_1)
}

#[derive(Clone, Serialize, Deserialize)]
struct Packet {
    data: Vec<u8>,
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("io");

    const SIZE: usize = 1024 * 1024;

    group.throughput(Throughput::Bytes(SIZE as u64));
    group.bench_function("memory", |b| {
        let (io_0, io_1) = duplex(16 * 1024 * 1024); // 16 MB buffer.
        let mut io_0 = Io::from_io(io_0.compat());
        let mut io_1 = Io::from_io(io_1.compat());
        let packet = Packet {
            data: vec![0; SIZE],
        };
        b.iter(|| {
            async {
                let (_, out): (_, Packet) =
                    futures::try_join!(io_0.send(packet.clone()), io_1.expect_next()).unwrap();
                black_box(out);
            }
            .block_on()
        });
    });

    group.bench_function("tcp", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let (io_0, io_1): (TcpStream, TcpStream) = rt.block_on(async {
            let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::join!(
                async {
                    let (io, _) = listener.accept().await.unwrap();
                    io
                },
                async { TcpStream::connect(addr).await.unwrap() }
            )
        });

        let io_0 = Arc::new(Mutex::new(Io::from_io(io_0.compat())));
        let io_1 = Arc::new(Mutex::new(Io::from_io(io_1.compat())));

        let packet = Packet {
            data: vec![0; SIZE],
        };

        b.to_async(&rt).iter(|| async {
            let mut io_0 = io_0.try_lock().unwrap();
            let mut io_1 = io_1.try_lock().unwrap();
            let (_, out): (_, Packet) =
                tokio::try_join!(io_0.send(packet.clone()), io_1.expect_next()).unwrap();
            black_box(out);
        });
    });

    // Benchmark plain st_context (serio memory channel) vs recording context
    // to measure recording overhead
    group.bench_function("st_context_plain", |b| {
        let (mut ctx_0, mut ctx_1) = test_st_context(16 * 1024 * 1024);
        let packet = Packet {
            data: vec![0; SIZE],
        };
        b.iter(|| {
            async {
                let (_, out): (_, Packet) = futures::try_join!(
                    ctx_0.io_mut().send(packet.clone()),
                    ctx_1.io_mut().expect_next()
                )
                .unwrap();
                black_box(out);
            }
            .block_on()
        });
    });

    group.bench_function("st_context_recording", |b| {
        let (mut ctx_0, mut ctx_1, _recorded) =
            recording_st_context_with_limit(16 * 1024 * 1024, 16 * 1024 * 1024);
        let packet = Packet {
            data: vec![0; SIZE],
        };
        b.iter(|| {
            async {
                let (_, out): (_, Packet) = futures::try_join!(
                    ctx_0.io_mut().send(packet.clone()),
                    ctx_1.io_mut().expect_next()
                )
                .unwrap();
                black_box(out);
            }
            .block_on()
        });
    });

    // Benchmark BiStream (WASM-compatible channel) vs tokio duplex
    group.bench_function("bistream", |b| {
        let (io_0, io_1) = bistream_pair();
        let mut io_0 = Io::from_io(io_0);
        let mut io_1 = Io::from_io(io_1);
        let packet = Packet {
            data: vec![0; SIZE],
        };
        b.iter(|| {
            async {
                let (_, out): (_, Packet) =
                    futures::try_join!(io_0.send(packet.clone()), io_1.expect_next()).unwrap();
                black_box(out);
            }
            .block_on()
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
