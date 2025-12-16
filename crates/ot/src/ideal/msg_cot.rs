//! Message-based ideal functionality for correlated OT.
//!
//! Unlike the barrier-based `ideal_cot`, this implementation uses separate
//! instances for sender and receiver that exchange actual OT data through
//! the context channel, following the ideal-vm pattern.

use async_trait::async_trait;
use mpz_common::{Context, Flush, future::{MaybeDone, Sender, new_output}};
use mpz_core::Block;
use mpz_ot_core::{
    TransferId,
    cot::{COTReceiver, COTReceiverOutput, COTSender, COTSenderOutput},
};
use serde::{Deserialize, Serialize};
use serio::{SinkExt, stream::IoStreamExt};

/// Message sent from sender to receiver during flush.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlushMsg {
    /// Keys (m0 values) for each OT.
    keys: Vec<Block>,
    /// Global correlation delta.
    delta: Block,
}

/// Returns a new message-based ideal COT sender and receiver.
///
/// Unlike `ideal_cot`, this version creates separate instances that
/// exchange actual OT data through the context during flush.
pub fn msg_ideal_cot(delta: Block) -> (MsgIdealCOTSender, MsgIdealCOTReceiver) {
    (
        MsgIdealCOTSender::new(delta),
        MsgIdealCOTReceiver::new(),
    )
}

/// Message-based ideal COT sender.
///
/// Has its own state and sends OT data to receiver via messages.
pub struct MsgIdealCOTSender {
    delta: Block,
    /// Pending keys queued via queue_send_cot().
    pending_keys: Vec<Block>,
    /// Queue of (count, sender) for deferred output.
    queue: Vec<(usize, Sender<COTSenderOutput>)>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl MsgIdealCOTSender {
    /// Creates a new sender with the given delta.
    pub fn new(delta: Block) -> Self {
        Self {
            delta,
            pending_keys: Vec::new(),
            queue: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl COTSender<Block> for MsgIdealCOTSender {
    type Error = MsgIdealCOTError;
    type Future = MaybeDone<COTSenderOutput>;

    fn alloc(&mut self, _count: usize) -> Result<(), Self::Error> {
        // No pre-allocation needed for COT - keys are provided via queue_send_cot
        Ok(())
    }

    fn available(&self) -> usize {
        self.pending_keys.len()
    }

    fn delta(&self) -> Block {
        self.delta
    }

    fn queue_send_cot(&mut self, keys: &[Block]) -> Result<Self::Future, Self::Error> {
        // Queue keys for sending during flush
        self.pending_keys.extend_from_slice(keys);

        let (send, recv) = new_output();
        self.queue.push((keys.len(), send));
        Ok(recv)
    }
}

#[async_trait]
impl Flush for MsgIdealCOTSender {
    type Error = MsgIdealCOTError;

    fn wants_flush(&self) -> bool {
        !self.queue.is_empty()
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.queue.is_empty() {
            return Ok(());
        }

        // Take pending keys
        let keys = std::mem::take(&mut self.pending_keys);

        // Send keys and delta to receiver
        let flush_msg = FlushMsg {
            keys,
            delta: self.delta,
        };
        ctx.io_mut().send(flush_msg).await?;

        // Send outputs through queued senders
        for (_count, sender) in std::mem::take(&mut self.queue) {
            sender.send(COTSenderOutput {
                id: self.transfer_id.next(),
            });
        }

        Ok(())
    }
}

/// Message-based ideal COT receiver.
///
/// Has its own state and receives OT data from sender via messages.
pub struct MsgIdealCOTReceiver {
    /// Pending choices queued via queue_recv_cot().
    pending_choices: Vec<bool>,
    /// Queue of (count, sender) for deferred output.
    queue: Vec<(usize, Sender<COTReceiverOutput<Block>>)>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl MsgIdealCOTReceiver {
    /// Creates a new receiver.
    pub fn new() -> Self {
        Self {
            pending_choices: Vec::new(),
            queue: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl Default for MsgIdealCOTReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl COTReceiver<bool, Block> for MsgIdealCOTReceiver {
    type Error = MsgIdealCOTError;
    type Future = MaybeDone<COTReceiverOutput<Block>>;

    fn alloc(&mut self, _count: usize) -> Result<(), Self::Error> {
        // No pre-allocation needed - choices are provided via queue_recv_cot
        Ok(())
    }

    fn available(&self) -> usize {
        self.pending_choices.len()
    }

    fn queue_recv_cot(&mut self, choices: &[bool]) -> Result<Self::Future, Self::Error> {
        // Queue choices for processing during flush
        self.pending_choices.extend_from_slice(choices);

        let (send, recv) = new_output();
        self.queue.push((choices.len(), send));
        Ok(recv)
    }
}

#[async_trait]
impl Flush for MsgIdealCOTReceiver {
    type Error = MsgIdealCOTError;

    fn wants_flush(&self) -> bool {
        !self.queue.is_empty()
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.queue.is_empty() {
            return Ok(());
        }

        // Receive keys and delta from sender
        let flush_msg: FlushMsg = ctx.io_mut().expect_next().await?;

        let choices = std::mem::take(&mut self.pending_choices);

        if flush_msg.keys.len() != choices.len() {
            return Err(MsgIdealCOTError::CountMismatch {
                expected: choices.len(),
                received: flush_msg.keys.len(),
            });
        }

        // Compute messages based on choices: msg = key XOR (choice * delta)
        let mut msgs = flush_msg
            .keys
            .iter()
            .zip(&choices)
            .map(|(key, &choice)| {
                if choice {
                    *key ^ flush_msg.delta
                } else {
                    *key
                }
            });

        // Send outputs through queued senders
        for (count, sender) in std::mem::take(&mut self.queue) {
            sender.send(COTReceiverOutput {
                id: self.transfer_id.next(),
                msgs: msgs.by_ref().take(count).collect(),
            });
        }

        Ok(())
    }
}

/// Error for message-based ideal COT.
#[derive(Debug, thiserror::Error)]
pub enum MsgIdealCOTError {
    /// Count mismatch between expected and received.
    #[error("count mismatch: expected={expected}, received={received}")]
    CountMismatch {
        /// Expected count.
        expected: usize,
        /// Received count.
        received: usize,
    },
    /// I/O error during message exchange.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use mpz_common::context::test_st_context;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    #[tokio::test]
    async fn test_msg_ideal_cot() {
        let mut rng = StdRng::seed_from_u64(0);
        let delta: Block = rng.random();

        let (mut sender, mut receiver) = msg_ideal_cot(delta);
        let (mut ctx_s, mut ctx_r) = test_st_context(1024 * 1024);

        const COUNT: usize = 128;

        // Generate random keys and choices
        let keys: Vec<Block> = (0..COUNT).map(|_| rng.random()).collect();
        let choices: Vec<bool> = (0..COUNT).map(|_| rng.random()).collect();

        // Queue send/recv - returns futures
        let _sender_fut = sender.queue_send_cot(&keys).unwrap();
        let receiver_fut = receiver.queue_recv_cot(&choices).unwrap();

        // Flush (exchange OT data)
        let (r1, r2) = futures::join!(
            sender.flush(&mut ctx_s),
            receiver.flush(&mut ctx_r)
        );
        r1.unwrap();
        r2.unwrap();

        // Get output from future
        let recv_output = receiver_fut.await.unwrap();

        // Verify correctness: msg_i = key_i XOR (choice_i * delta)
        assert_eq!(recv_output.msgs.len(), COUNT);
        for i in 0..COUNT {
            let expected = if choices[i] {
                keys[i] ^ delta
            } else {
                keys[i]
            };
            assert_eq!(recv_output.msgs[i], expected);
        }
    }
}
