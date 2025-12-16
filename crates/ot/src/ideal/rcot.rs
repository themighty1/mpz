//! Ideal functionality for random correlated OT.
//!
//! This implementation sends only a seed during flush, and the receiver
//! regenerates keys locally via counter addition. Keys are generated as
//! `key[i] = seed + offset + i` (simple u128 addition).

use async_trait::async_trait;
use mpz_common::{Context, Flush, future::{MaybeDone, Sender, new_output}};
use mpz_core::{Block, prg::Prg};
use mpz_ot_core::{
    TransferId,
    rcot::{RCOTReceiver, RCOTReceiverOutput, RCOTSender, RCOTSenderOutput},
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serio::{SinkExt, stream::IoStreamExt};

/// Message sent from sender to receiver during flush.
/// Only contains seed + offset + count + delta, not the actual data.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlushMsg {
    /// Base seed for key generation.
    seed: Block,
    /// Offset into the key sequence.
    offset: u64,
    /// Number of OTs to generate.
    count: usize,
    /// Global correlation delta.
    delta: Block,
}

/// Generate keys via simple counter addition: key[i] = seed + offset + i
#[inline]
fn generate_keys(seed: Block, offset: u64, count: usize) -> Vec<Block> {
    let base = u128::from_le_bytes(seed.to_bytes());
    (0..count)
        .map(|i| {
            let val = base.wrapping_add(offset as u128).wrapping_add(i as u128);
            Block::from(val.to_le_bytes())
        })
        .collect()
}

/// Returns a new ideal RCOT sender and receiver.
///
/// This implementation sends only a seed during flush, and the receiver
/// regenerates keys locally via counter addition.
pub fn ideal_rcot(seed: Block, delta: Block) -> (IdealRCOTSender, IdealRCOTReceiver) {
    (
        IdealRCOTSender::new(seed, delta),
        IdealRCOTReceiver::new(),
    )
}

/// Ideal RCOT sender.
///
/// Sends only seed + offset + count + delta, not the actual keys/msgs.
pub struct IdealRCOTSender {
    delta: Block,
    seed: Block,
    /// Current offset into key sequence.
    offset: u64,
    /// Pending allocation count.
    pending: usize,
    /// Generated keys (sender's OT outputs).
    keys: Vec<Block>,
    /// Queue of (count, sender) for deferred output.
    queue: Vec<(usize, Sender<RCOTSenderOutput<Block>>)>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl IdealRCOTSender {
    /// Creates a new sender with the given seed and delta.
    pub fn new(seed: Block, delta: Block) -> Self {
        Self {
            delta,
            seed,
            offset: 0,
            pending: 0,
            keys: Vec::new(),
            queue: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl RCOTSender<Block> for IdealRCOTSender {
    type Error = IdealRCOTError;
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
            return Err(IdealRCOTError::NotEnoughOTs {
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

        // If enough keys available, send immediately
        if self.keys.len() >= count {
            let keys_len = self.keys.len();
            let keys = self.keys.split_off(keys_len - count);
            send.send(RCOTSenderOutput {
                id: self.transfer_id.next(),
                keys,
            });
        } else {
            // Otherwise queue for flush() to fulfill
            self.queue.push((count, send));
        }

        Ok(recv)
    }
}

#[async_trait]
impl Flush for IdealRCOTSender {
    type Error = IdealRCOTError;

    fn wants_flush(&self) -> bool {
        self.pending > 0 || !self.queue.is_empty()
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.pending == 0 && self.queue.is_empty() {
            return Ok(());
        }

        let count = self.pending;
        let current_offset = self.offset;

        if count > 0 {
            // Generate keys via counter addition
            let keys = generate_keys(self.seed, current_offset, count);

            // Store keys for sender
            self.keys.extend(keys);

            // Advance offset
            self.offset += count as u64;

            // Send only seed + offset + count + delta (not the actual data!)
            let flush_msg = FlushMsg {
                seed: self.seed,
                offset: current_offset,
                count,
                delta: self.delta,
            };
            ctx.io_mut().send(flush_msg).await?;

            self.pending = 0;
        }

        // Fulfill queued requests
        for (count, sender) in std::mem::take(&mut self.queue) {
            let keys_len = self.keys.len();
            let keys = self.keys.split_off(keys_len - count);
            sender.send(RCOTSenderOutput {
                id: self.transfer_id.next(),
                keys,
            });
        }

        Ok(())
    }
}

/// Ideal RCOT receiver.
///
/// Regenerates keys locally from the received seed via counter addition.
pub struct IdealRCOTReceiver {
    /// PRG for generating choices.
    prg: Prg,
    /// Pending allocation count.
    pending: usize,
    /// Received choice bits.
    choices: Vec<bool>,
    /// Received messages.
    msgs: Vec<Block>,
    /// Queue of (count, sender) for deferred output.
    queue: Vec<(usize, Sender<RCOTReceiverOutput<bool, Block>>)>,
    /// Transfer ID counter.
    transfer_id: TransferId,
}

impl IdealRCOTReceiver {
    /// Creates a new receiver.
    pub fn new() -> Self {
        Self {
            prg: Prg::new(),
            pending: 0,
            choices: Vec::new(),
            msgs: Vec::new(),
            queue: Vec::new(),
            transfer_id: TransferId::default(),
        }
    }
}

impl Default for IdealRCOTReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl RCOTReceiver<bool, Block> for IdealRCOTReceiver {
    type Error = IdealRCOTError;
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
            return Err(IdealRCOTError::NotEnoughOTs {
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

        // If enough available, send immediately
        if self.choices.len() >= count {
            let len = self.choices.len();
            let choices = self.choices.split_off(len - count);
            let msgs = self.msgs.split_off(len - count);
            send.send(RCOTReceiverOutput {
                id: self.transfer_id.next(),
                choices,
                msgs,
            });
        } else {
            // Otherwise queue for flush() to fulfill
            self.queue.push((count, send));
        }

        Ok(recv)
    }
}

#[async_trait]
impl Flush for IdealRCOTReceiver {
    type Error = IdealRCOTError;

    fn wants_flush(&self) -> bool {
        self.pending > 0 || !self.queue.is_empty()
    }

    async fn flush(&mut self, ctx: &mut Context) -> Result<(), Self::Error> {
        if self.pending == 0 && self.queue.is_empty() {
            return Ok(());
        }

        if self.pending > 0 {
            // Receive seed + offset + count + delta from sender
            let flush_msg: FlushMsg = ctx.io_mut().expect_next().await?;

            if flush_msg.count != self.pending {
                return Err(IdealRCOTError::CountMismatch {
                    expected: self.pending,
                    received: flush_msg.count,
                });
            }

            // Regenerate keys via counter addition (same as sender)
            let keys = generate_keys(flush_msg.seed, flush_msg.offset, flush_msg.count);

            // Generate random choices via PRG
            let choices: Vec<bool> = (0..flush_msg.count).map(|_| self.prg.random()).collect();

            // Compute receiver's messages: msg_i = key_i XOR (choice_i * delta)
            let msgs: Vec<Block> = keys
                .iter()
                .zip(&choices)
                .map(|(key, &choice)| {
                    if choice {
                        *key ^ flush_msg.delta
                    } else {
                        *key
                    }
                })
                .collect();

            // Store received data
            self.choices.extend(choices);
            self.msgs.extend(msgs);

            self.pending = 0;
        }

        // Fulfill queued requests
        for (count, sender) in std::mem::take(&mut self.queue) {
            let len = self.choices.len();
            let choices = self.choices.split_off(len - count);
            let msgs = self.msgs.split_off(len - count);
            sender.send(RCOTReceiverOutput {
                id: self.transfer_id.next(),
                choices,
                msgs,
            });
        }

        Ok(())
    }
}

/// Ideal RCOT error.
#[derive(Debug, thiserror::Error)]
pub enum IdealRCOTError {
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
    async fn test_ideal_rcot() {
        let mut rng = StdRng::seed_from_u64(0);
        let delta: Block = rng.random();

        let (mut sender, mut receiver) = ideal_rcot(rng.random(), delta);
        let (mut ctx_s, mut ctx_r) = test_st_context(1024 * 1024);

        const COUNT: usize = 128;

        // Allocate
        sender.alloc(COUNT).unwrap();
        receiver.alloc(COUNT).unwrap();

        // Flush (exchange seed only)
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
