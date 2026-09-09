// TEMPORARY debug repro — delete after pin-down.
//
// Reproduces the TLS-harness `mux_tls_stress` failure one level down:
// full MPC-TLS preprocessing needs >32 concurrent mux streams; with
// `max_num_streams = 32` it dies fast in OT bootstrap with torn-down
// streams (UnexpectedEof / "bytes remaining on stream" / "connection is
// closed", plus `Task polled after completion` on an mpz worker).
//
// This test runs N parallel real base-OT setups (chou_orlandi, the exact
// BaseOT in the first failure chain) over real `Executor`s multiplexed
// through real SYN-mux `Connection`s with the Session-style config, and
// instruments live-stream concurrency.
//
// Outcomes and what they mean:
// - PASS: budget pressure alone (with this fan-out shape) does not break;
//   the TLS failure needs something more specific.
// - HANG (timeout): over-subscription deadlocks in backpressure.
// - FAIL with torn-down-stream errors: mux-level repro of the TLS failure.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context as TaskContext, Poll},
    time::Duration,
};

use futures::{AsyncRead, AsyncWrite, future::poll_fn};
use mpz_common::{Context, Executor, Flush, io::Io, mux::Mux};
use mpz_core::Block;
use mpz_ot::chou_orlandi::{Receiver as BaseReceiver, Sender as BaseSender};
use mpz_ot_core::ot::{OTReceiver, OTSender};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokio::task;
use tokio_util::compat::TokioAsyncReadCompatExt;

// Parallel base-OT setups; each is its own stream per side, all fanned out
// at once like OT bootstrap. Roles alternate per pair so both sides'
// first-writes race (bidirectional SYN pressure).
//
// Ramp-up knobs (env, with defaults): TMP_N_PAIRS, TMP_OT_COUNT,
// TMP_MAX_STREAMS.
fn n_pairs() -> usize {
    std::env::var("TMP_N_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48)
}

fn ot_count() -> usize {
    std::env::var("TMP_OT_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn max_streams() -> usize {
    std::env::var("TMP_MAX_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32)
}

fn session_like_cfg() -> tlsn_mux::Config {
    let mut cfg = tlsn_mux::Config::default();
    cfg.set_max_num_streams(max_streams());
    cfg.set_keep_alive(true);
    cfg.set_close_sync(true);
    cfg
}

/// Mux over a real SYN-mux connection, counting live streams.
#[derive(Clone)]
struct CountingMux {
    handle: tlsn_mux::Handle,
    live: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl Mux for CountingMux {
    fn open(&self, id: &[u8]) -> Result<Io, std::io::Error> {
        let stream = self
            .handle
            .new_stream(id)
            .map_err(std::io::Error::other)?;
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(live, Ordering::SeqCst);
        Ok(Io::from_io(TrackedIo {
            inner: stream,
            live: self.live.clone(),
        }))
    }
}

/// Stream wrapper that decrements the live counter on drop (= close).
struct TrackedIo {
    inner: tlsn_mux::Stream,
    live: Arc<AtomicUsize>,
}

impl Drop for TrackedIo {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

impl AsyncRead for TrackedIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
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

async fn run_base_ot<S, R>(
    mut sender: S,
    mut receiver: R,
    ctx_s: &mut Context,
    ctx_r: &mut Context,
    seed: u64,
) where
    S: OTSender<Block> + Flush,
    R: OTReceiver<bool, Block> + Flush,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let msgs: Vec<[Block; 2]> = (0..ot_count()).map(|_| [rng.random(), rng.random()]).collect();
    let choices: Vec<bool> = (0..ot_count()).map(|_| rng.random()).collect();

    let (out_s, out_r) = futures::join!(
        async {
            sender.alloc(msgs.len()).unwrap();
            let output = sender.queue_send_ot(&msgs).unwrap();
            sender.flush(ctx_s).await.unwrap();
            output.await.unwrap()
        },
        async {
            receiver.alloc(choices.len()).unwrap();
            let output = receiver.queue_recv_ot(&choices).unwrap();
            receiver.flush(ctx_r).await.unwrap();
            output.await.unwrap()
        }
    );

    assert_eq!(out_s.id, out_r.id);
    assert_eq!(out_r.msgs.len(), ot_count());
    for ((choice, pair), got) in choices.iter().zip(msgs.iter()).zip(out_r.msgs.iter()) {
        assert_eq!(got, &pair[*choice as usize]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tmp_over_budget_base_ot_fanout() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    // Drive connect and accept concurrently: `connect` is lazy and makes no
    // progress until polled, so awaiting accept first deadlocks.
    let (conn_a, conn_b) = futures::join!(
        tokio::net::TcpStream::connect(addr),
        listener.accept()
    );
    let sock_a = conn_a.unwrap();
    let (sock_b, _) = conn_b.unwrap();
    sock_a.set_nodelay(true).unwrap();
    sock_b.set_nodelay(true).unwrap();

    let mut conn_a = tlsn_mux::Connection::new(sock_a.compat(), session_like_cfg());
    let mut conn_b = tlsn_mux::Connection::new(sock_b.compat(), session_like_cfg());

    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let mux_a = CountingMux {
        handle: conn_a.handle().unwrap(),
        live: live.clone(),
        peak: peak.clone(),
    };
    let mux_b = CountingMux {
        handle: conn_b.handle().unwrap(),
        live: live.clone(),
        peak: peak.clone(),
    };

    task::spawn(async move {
        poll_fn(|cx| conn_a.poll(cx)).await.ok();
    });
    task::spawn(async move {
        poll_fn(|cx| conn_b.poll(cx)).await.ok();
    });

    let exec_a = Executor::builder().num_threads(8).build(mux_a);
    let exec_b = Executor::builder().num_threads(8).build(mux_b);

    let run = async {
        let pairs: Vec<_> = (0..n_pairs())
            .map(|i| {
                let mut ctx_a = exec_a.new_context().unwrap();
                let mut ctx_b = exec_b.new_context().unwrap();
                async move {
                    if i % 2 == 0 {
                        run_base_ot(
                            BaseSender::new(),
                            BaseReceiver::new(),
                            &mut ctx_a,
                            &mut ctx_b,
                            i as u64,
                        )
                        .await;
                    } else {
                        run_base_ot(
                            BaseSender::new(),
                            BaseReceiver::new(),
                            &mut ctx_b,
                            &mut ctx_a,
                            i as u64,
                        )
                        .await;
                    }
                }
            })
            .collect();
        futures::future::join_all(pairs).await;
    };

    tokio::time::timeout(Duration::from_secs(180), run)
        .await
        .expect("TMP REPRO: did not complete in 180s (hang/deadlock under budget pressure)");

    eprintln!(
        "TMP REPRO: pairs={} ots={} budget={} peak concurrent live streams: {}",
        n_pairs(),
        ot_count(),
        max_streams(),
        peak.load(Ordering::SeqCst)
    );

    exec_a.shutdown();
    exec_b.shutdown();
}
