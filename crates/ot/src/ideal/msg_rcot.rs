//! Message-based ideal functionality for random correlated OT.
//!
//! Unlike the barrier-based `ideal_rcot`, this implementation uses separate
//! instances for sender and receiver that exchange actual OT data through
//! the context channel, following the ideal-vm pattern.

use async_trait::async_trait;
use mpz_common::{Context, Flush, future::{MaybeDone, new_output}};
use mpz_core::{Block, prg::Prg};
use mpz_ot_core::{
    TransferId,
    rcot::{RCOTReceiver, RCOTReceiverOutput, RCOTSender, RCOTSenderOutput},
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serio::{SinkExt, stream::IoStreamExt};

/// Message sent from sender to receiver during flush.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlushMsg {
    /// Random choice bits.
    choices: Vec<bool>,
    /// Receiver's OT messages: msg_i = key_i XOR (choice_i * delta).
    msgs: Vec<Block>,
}

/// Returns a new message-based ideal RCOT sender and receiver.
///
/// Unlike `ideal_rcot`, this version creates separate instances that
/// exchange actual OT data through the context during flush.
pub fn msg_ideal_rcot(seed: Block, delta: Block) -> (MsgIdealRCOTSender, MsgIdealRCOTReceiver) {
    (
        MsgIdealRCOTSender::new(seed, delta),
        MsgIdealRCOTReceiver::new(),
    )
}

/// Message-based ideal RCOT sender.
///
/// Has its own state and sends OT data to receiver via messages.
pub struct MsgIdealRCOTSender {
    delta: Block,
    prg: Prg,
    /// Pending allocation count.
    pending: usize,
    /// Generated keys (sender's OT outputs).
    keys: Vec<Block>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl MsgIdealRCOTSender {
    /// Creates a new sender with the given seed and delta.
    pub fn new(seed: Block, delta: Block) -> Self {
        Self {
            delta,
            prg: Prg::new_with_seed(seed.to_bytes()),
            pending: 0,
            keys: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl RCOTSender<Block> for MsgIdealRCOTSender {
    type Error = MsgIdealRCOTError;
    type Future = MaybeDone<RCOTSenderOutput<Block>>;

    fn alloc(&mut self, count: usize) -> Result<(), Self::Error> {
        self.pending += count;
        Ok(())
    }

    fn available(&self) -> usize {
        self.keys.len()
    }

    fn delta(&self) -> Block {
        self.delta
    }

    fn try_send_rcot(&mut self, count: usize) -> Result<RCOTSenderOutput<Block>, Self::Error> {
        if count > self.keys.len() {
            return Err(MsgIdealRCOTError::NotEnoughOTs {
                available: self.keys.len(),
                requested: count,
            });
        }

        let keys_len = self.keys.len();
        let keys = self.keys.split_off(keys_len - count);

        Ok(RCOTSenderOutput {
            id: self.transfer_id.next(),
            keys,
        })
    }

    fn queue_send_rcot(&mut self, count: usize) -> Result<Self::Future, Self::Error> {
        let (send, recv) = new_output();
        let output = self.try_send_rcot(count)?;
        send.send(output);
        Ok(recv)
    }
}

#[async_trait]
impl Flush for MsgIdealRCOTSender {
    type Error = MsgIdealRCOTError;

    fn wants_flush(&self) -> bool {
        self.pending > 0
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.pending == 0 {
            return Ok(());
        }

        let count = self.pending;

        // Generate random keys
        let keys: Vec<Block> = (0..count).map(|_| self.prg.random()).collect();

        // Generate random choices
        let choices: Vec<bool> = (0..count).map(|_| self.prg.random()).collect();

        // Compute receiver's messages: msg_i = key_i XOR (choice_i * delta)
        let msgs: Vec<Block> = keys
            .iter()
            .zip(&choices)
            .map(|(key, &choice)| {
                if choice {
                    *key ^ self.delta
                } else {
                    *key
                }
            })
            .collect();

        // Store keys for sender
        self.keys.extend(keys);

        // Send choices and msgs to receiver
        let flush_msg = FlushMsg { choices, msgs };
        ctx.io_mut().send(flush_msg).await?;

        self.pending = 0;

        Ok(())
    }
}

/// Message-based ideal RCOT receiver.
///
/// Has its own state and receives OT data from sender via messages.
pub struct MsgIdealRCOTReceiver {
    /// Pending allocation count.
    pending: usize,
    /// Received choice bits.
    choices: Vec<bool>,
    /// Received messages.
    msgs: Vec<Block>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl MsgIdealRCOTReceiver {
    /// Creates a new receiver.
    pub fn new() -> Self {
        Self {
            pending: 0,
            choices: Vec::new(),
            msgs: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl Default for MsgIdealRCOTReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl RCOTReceiver<bool, Block> for MsgIdealRCOTReceiver {
    type Error = MsgIdealRCOTError;
    type Future = MaybeDone<RCOTReceiverOutput<bool, Block>>;

    fn alloc(&mut self, count: usize) -> Result<(), Self::Error> {
        self.pending += count;
        Ok(())
    }

    fn available(&self) -> usize {
        self.choices.len()
    }

    fn try_recv_rcot(&mut self, count: usize) -> Result<RCOTReceiverOutput<bool, Block>, Self::Error> {
        if count > self.choices.len() {
            return Err(MsgIdealRCOTError::NotEnoughOTs {
                available: self.choices.len(),
                requested: count,
            });
        }

        let len = self.choices.len();
        let choices = self.choices.split_off(len - count);
        let msgs = self.msgs.split_off(len - count);

        Ok(RCOTReceiverOutput {
            id: self.transfer_id.next(),
            choices,
            msgs,
        })
    }

    fn queue_recv_rcot(&mut self, count: usize) -> Result<Self::Future, Self::Error> {
        let (send, recv) = new_output();
        let output = self.try_recv_rcot(count)?;
        send.send(output);
        Ok(recv)
    }
}

#[async_trait]
impl Flush for MsgIdealRCOTReceiver {
    type Error = MsgIdealRCOTError;

    fn wants_flush(&self) -> bool {
        self.pending > 0
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.pending == 0 {
            return Ok(());
        }

        // Receive choices and msgs from sender
        let flush_msg: FlushMsg = ctx.io_mut().expect_next().await?;

        if flush_msg.choices.len() != self.pending {
            return Err(MsgIdealRCOTError::CountMismatch {
                expected: self.pending,
                received: flush_msg.choices.len(),
            });
        }

        // Store received data
        self.choices.extend(flush_msg.choices);
        self.msgs.extend(flush_msg.msgs);

        self.pending = 0;

        Ok(())
    }
}

/// Error for message-based ideal RCOT.
#[derive(Debug, thiserror::Error)]
pub enum MsgIdealRCOTError {
    /// Not enough OTs available.
    #[error("not enough OTs: available={available}, requested={requested}")]
    NotEnoughOTs {
        /// Number of OTs available.
        available: usize,
        /// Number of OTs requested.
        requested: usize,
    },
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
    async fn test_msg_ideal_rcot() {
        let mut rng = StdRng::seed_from_u64(0);
        let delta: Block = rng.random();

        let (mut sender, mut receiver) = msg_ideal_rcot(rng.random(), delta);
        let (mut ctx_s, mut ctx_r) = test_st_context(1024 * 1024);

        const COUNT: usize = 128;

        // Allocate
        sender.alloc(COUNT).unwrap();
        receiver.alloc(COUNT).unwrap();

        // Flush (exchange OT data)
        let (r1, r2) = futures::join!(
            sender.flush(&mut ctx_s),
            receiver.flush(&mut ctx_r)
        );
        r1.unwrap();
        r2.unwrap();

        // Transfer
        let sender_out = sender.try_send_rcot(COUNT).unwrap();
        let receiver_out = receiver.try_recv_rcot(COUNT).unwrap();

        // Verify correctness: msg_i = key_i XOR (choice_i * delta)
        for i in 0..COUNT {
            let expected = if receiver_out.choices[i] {
                sender_out.keys[i] ^ delta
            } else {
                sender_out.keys[i]
            };
            assert_eq!(receiver_out.msgs[i], expected);
        }
    }
}
