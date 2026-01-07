//! Authenticated garbling protocol.

mod auth_eval;
mod auth_gen;

pub use auth_eval::AuthEval;
pub use auth_gen::AuthGen;


#[cfg(test)]
mod tests {
    use aes::Aes128;
    use aes::cipher::{BlockCipherEncrypt, KeyInit};
    use mpz_circuits::AES128;
    use mpz_common::context::test_st_context;
    use mpz_memory_core::{
        Array, MemoryExt, ViewExt,
        binary::{Binary, U8},
        correlated::Delta,
    };
    use mpz_ot::ideal::cot::{IdealCOTReceiver, IdealCOTSender, ideal_cot};
    use mpz_vm_core::{Call, CallableExt, Execute, Vm};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    /// Compute cleartext AES-128 encryption for testing
    fn aes128_encrypt(key: [u8; 16], plaintext: [u8; 16]) -> [u8; 16] {
        let cipher = Aes128::new(&key.into());
        let mut block = plaintext.into();
        cipher.encrypt_block(&mut block);
        block.into()
    }

    #[test]
    fn test_semihonest_is_vm() {
        fn is_vm<T: Vm<Binary>>() {}
        is_vm::<AuthGen<IdealCOTSender, IdealCOTReceiver>>();
        is_vm::<AuthEval<IdealCOTSender, IdealCOTReceiver>>();
    }

    #[tokio::test]
    async fn test_authenticated() {
        let mut rng = StdRng::seed_from_u64(0);
        
        let delta_a = Delta::random(&mut rng).set_lsb(true);
        let delta_b = Delta::random(&mut rng).set_lsb(false);

        let (mut ctx_a, mut ctx_b) = test_st_context(8);
        let (cot_gen_send, cot_eval_recv) = ideal_cot(delta_a.into_inner());
        let (cot_eval_send, cot_gen_recv) = ideal_cot(delta_b.into_inner());

        let mut gb = AuthGen::new([0u8; 16], delta_a, cot_gen_send, cot_gen_recv);
        let mut ev = AuthEval::new([0u8; 16], delta_b, cot_eval_send, cot_eval_recv);

        let ((gen_key, gen_msg, gen_ciphertext), (ev_key, ev_msg, ev_ciphertext)) = futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();
                let msg: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.mark_blind(msg).unwrap();

                let mut decoded_key = gb.decode(key).unwrap();
                let mut decoded_msg = gb.decode(msg).unwrap();

                let ciphertext: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let mut ciphertext = gb.decode(ciphertext).unwrap();

                gb.assign(key, [0u8; 16]).unwrap();
                gb.commit(key).unwrap();
                gb.commit(msg).unwrap();

                gb.execute_all(&mut ctx_a).await.unwrap();
                (decoded_key.try_recv().unwrap().unwrap(), decoded_msg.try_recv().unwrap().unwrap(), ciphertext.try_recv().unwrap().unwrap())
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                let msg: Array<U8, 16> = ev.alloc().unwrap();

                ev.mark_blind(key).unwrap();
                ev.mark_private(msg).unwrap();

                let mut decoded_key = ev.decode(key).unwrap();
                let mut decoded_msg = ev.decode(msg).unwrap();

                let ciphertext: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let mut ciphertext = ev.decode(ciphertext).unwrap();

                ev.assign(msg, [42u8; 16]).unwrap();
                ev.commit(key).unwrap();
                ev.commit(msg).unwrap();

                ev.execute_all(&mut ctx_b).await.unwrap();
                (decoded_key.try_recv().unwrap().unwrap(), decoded_msg.try_recv().unwrap().unwrap(), ciphertext.try_recv().unwrap().unwrap())
            }
        );

        // Verify both parties agree on the values
        assert_eq!(gen_key, ev_key);
        assert_eq!(gen_msg, ev_msg);
        assert_eq!(gen_ciphertext, ev_ciphertext);

        // Verify the garbled circuit output matches cleartext AES computation
        let expected = aes128_encrypt(gen_key, gen_msg);
        assert_eq!(gen_ciphertext, expected, "Garbled circuit output does not match cleartext AES");
        assert_eq!(ev_ciphertext, expected, "Evaluator output does not match cleartext AES");
    }

    #[tokio::test]
    async fn test_authenticated_nothing_to_do() {
        let mut rng = StdRng::seed_from_u64(0);
        
        let delta_a = Delta::random(&mut rng).set_lsb(true);
        let delta_b = Delta::random(&mut rng).set_lsb(false);

        let (mut ctx_a, mut ctx_b) = test_st_context(8);
        let (cot_gen_send, cot_eval_recv) = ideal_cot(delta_a.into_inner());
        let (cot_eval_send, cot_gen_recv) = ideal_cot(delta_b.into_inner());

        let mut gb = AuthGen::new([0u8; 16], delta_a, cot_gen_send, cot_gen_recv);
        let mut ev = AuthEval::new([0u8; 16], delta_b, cot_eval_send, cot_eval_recv);

        gb.flush(&mut ctx_a).await.unwrap();
        ev.flush(&mut ctx_b).await.unwrap();
    }

    #[tokio::test]
    async fn test_authenticated_chain() {
        let mut rng = StdRng::seed_from_u64(0);
        
        let delta_a = Delta::random(&mut rng).set_lsb(true);
        let delta_b = Delta::random(&mut rng).set_lsb(false);

        let (mut ctx_a, mut ctx_b) = test_st_context(8);
        let (cot_gen_send, cot_eval_recv) = ideal_cot(delta_a.into_inner());
        let (cot_eval_send, cot_gen_recv) = ideal_cot(delta_b.into_inner());

        let mut gb = AuthGen::new([0u8; 16], delta_a, cot_gen_send, cot_gen_recv);
        let mut ev = AuthEval::new([0u8; 16], delta_b, cot_eval_send, cot_eval_recv);

        let (gen_out, ev_out) = futures::join!(
            async {
                let key: Array<U8, 16> = gb.alloc().unwrap();
                let key_2: Array<U8, 16> = gb.alloc().unwrap();
                let msg: Array<U8, 16> = gb.alloc().unwrap();
                let msg_2: Array<U8, 16> = gb.alloc().unwrap();

                gb.mark_private(key).unwrap();
                gb.mark_public(key_2).unwrap();
                gb.mark_blind(msg).unwrap();
                gb.mark_blind(msg_2).unwrap();

                let output: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                // Parallel AES calls.
                let ciphertext_2: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key_2)
                            .arg(msg_2)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                // Chain the AES calls.
                let ciphertext: Array<U8, 16> = gb
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(output)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let mut ciphertext = gb.decode(ciphertext).unwrap();
                let mut ciphertext_2 = gb.decode(ciphertext_2).unwrap();

                gb.assign(key, [0u8; 16]).unwrap();
                gb.assign(key_2, [69u8; 16]).unwrap();
                gb.commit(key).unwrap();
                gb.commit(key_2).unwrap();
                gb.commit(msg).unwrap();
                gb.commit(msg_2).unwrap();

                gb.execute_all(&mut ctx_a).await.unwrap();
                (ciphertext.try_recv().unwrap().unwrap(), ciphertext_2.try_recv().unwrap().unwrap())
            },
            async {
                let key: Array<U8, 16> = ev.alloc().unwrap();
                let key_2: Array<U8, 16> = ev.alloc().unwrap();
                let msg: Array<U8, 16> = ev.alloc().unwrap();
                let msg_2: Array<U8, 16> = ev.alloc().unwrap();

                ev.mark_blind(key).unwrap();
                ev.mark_public(key_2).unwrap();
                ev.mark_private(msg).unwrap();
                ev.mark_private(msg_2).unwrap();    

                let output: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(msg)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                // Parallel AES calls.
                let ciphertext_2: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key_2)
                            .arg(msg_2)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                // Chain the AES calls.
                let ciphertext: Array<U8, 16> = ev
                    .call(
                        Call::builder(AES128.clone())
                            .arg(key)
                            .arg(output)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();

                let mut ciphertext = ev.decode(ciphertext).unwrap();
                let mut ciphertext_2 = ev.decode(ciphertext_2).unwrap();

                ev.assign(key_2, [69u8; 16]).unwrap();
                ev.assign(msg, [42u8; 16]).unwrap();
                ev.assign(msg_2, [42u8; 16]).unwrap();
                ev.commit(key).unwrap();
                ev.commit(key_2).unwrap();
                ev.commit(msg).unwrap();
                ev.commit(msg_2).unwrap();

                ev.execute_all(&mut ctx_b).await.unwrap();
                (ciphertext.try_recv().unwrap().unwrap(), ciphertext_2.try_recv().unwrap().unwrap())
            }
        );

        // Verify both parties agree on the outputs
        assert_eq!(gen_out, ev_out);

        // Verify the garbled circuit outputs match cleartext AES computation
        // First parallel call: AES([69u8;16], [42u8;16])
        let expected_ciphertext_2 = aes128_encrypt([69u8; 16], [42u8; 16]);
        assert_eq!(gen_out.1, expected_ciphertext_2, "Parallel AES output does not match cleartext");

        // Chained calls: output = AES([0u8;16], [42u8;16]), then final = AES([0u8;16], output)
        let intermediate = aes128_encrypt([0u8; 16], [42u8; 16]);
        let expected_ciphertext = aes128_encrypt([0u8; 16], intermediate);
        assert_eq!(gen_out.0, expected_ciphertext, "Chained AES output does not match cleartext");
    }
}
