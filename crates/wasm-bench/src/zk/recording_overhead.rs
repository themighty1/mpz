//! Recording overhead benchmarks.
//!
//! Compares baseline MT context vs recording MT context to measure
//! the overhead of the recording infrastructure.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use mpz_circuits::AES128;
#[cfg(target_arch = "wasm32")]
use mpz_common::context::{
    recording_mt_context_with_spawn_and_limit, test_mt_context_with_spawn, Multithread,
};
#[cfg(target_arch = "wasm32")]
use mpz_memory_core::{Array, binary::U8, correlated::Delta};
#[cfg(target_arch = "wasm32")]
use mpz_ot::ideal::rcot::ideal_rcot;
#[cfg(target_arch = "wasm32")]
use mpz_vm_core::{Call, prelude::*};
#[cfg(target_arch = "wasm32")]
use mpz_zk::{Prover, ProverConfig, Verifier, VerifierConfig};
#[cfg(target_arch = "wasm32")]
use rand::{Rng, SeedableRng, rngs::StdRng};

#[cfg(target_arch = "wasm32")]
use crate::BenchResult;

#[cfg(target_arch = "wasm32")]
async fn yield_to_browser() {
    use wasm_bindgen_futures::JsFuture;
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    let _ = JsFuture::from(promise).await;
}

#[cfg(target_arch = "wasm32")]
fn max_frame_length(circuit: &mpz_circuits::Circuit, circuit_count: usize) -> usize {
    let bytes_per_correlation = 1 + 16;
    let overhead = 1.2;
    let correlations = circuit.and_count() * circuit_count;
    ((correlations * bytes_per_correlation) as f64 * overhead) as usize
}

/// Runs the full ZK protocol with given MT contexts.
#[cfg(target_arch = "wasm32")]
async fn run_full_protocol(
    exec_p: &mut Multithread,
    exec_v: &mut Multithread,
    circuit_count: usize,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let delta = Delta::random(&mut rng);

    let (ot_send, ot_recv) = ideal_rcot(rng.random(), delta.into_inner());

    let mut prover = Prover::new(ProverConfig::default(), ot_recv);
    let mut verifier = Verifier::new(VerifierConfig::default(), delta, ot_send);

    let mut ctx_p = exec_p.new_context().await.unwrap();
    let mut ctx_v = exec_v.new_context().await.unwrap();

    futures::join!(
        async {
            let key: Array<U8, 16> = prover.alloc().unwrap();
            prover.mark_private(key).unwrap();
            prover.assign(key, [0u8; 16]).unwrap();
            prover.commit(key).unwrap();

            for _ in 0..circuit_count {
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

            prover.flush(&mut ctx_p).await.unwrap();
            prover.execute(&mut ctx_p).await.unwrap();
            prover.flush(&mut ctx_p).await.unwrap();
        },
        async {
            let key: Array<U8, 16> = verifier.alloc().unwrap();
            verifier.mark_blind(key).unwrap();
            verifier.commit(key).unwrap();

            for _ in 0..circuit_count {
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

            verifier.flush(&mut ctx_v).await.unwrap();
            verifier.execute(&mut ctx_v).await.unwrap();
            verifier.flush(&mut ctx_v).await.unwrap();
        }
    );
}

/// Benchmark baseline MT context (no recording).
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `batch_size` - Number of AND gates per iteration
/// * `concurrency` - Maximum parallelism level
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_overhead_baseline(n: u32, batch_size: u32, concurrency: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;
    let circuit_count = (batch_size as u64).div_ceil(and_gates_per_circuit) as usize;
    let actual_gates = circuit_count as u64 * and_gates_per_circuit;

    let performance = web_sys::window().unwrap().performance().unwrap();

    yield_to_browser().await;

    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        let start = performance.now();

        let (mut exec_p, mut exec_v) = test_mt_context_with_spawn(concurrency as usize, |f| {
            let _ = web_spawn::spawn(f);
            Ok(())
        });
        run_full_protocol(&mut exec_p, &mut exec_v, circuit_count, 0).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * actual_gates,
    }
}

/// Benchmark recording MT context.
///
/// # Arguments
/// * `n` - Number of benchmark iterations
/// * `batch_size` - Number of AND gates per iteration
/// * `concurrency` - Maximum parallelism level
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn zk_overhead_recording(n: u32, batch_size: u32, concurrency: u32) -> BenchResult {
    let and_gates_per_circuit = AES128.and_count() as u64;
    let circuit_count = (batch_size as u64).div_ceil(and_gates_per_circuit) as usize;
    let actual_gates = circuit_count as u64 * and_gates_per_circuit;

    let performance = web_sys::window().unwrap().performance().unwrap();

    yield_to_browser().await;

    let mut total_elapsed_ms = 0.0;

    for _ in 0..n {
        let start = performance.now();

        let (mut exec_p, mut exec_v, _recorded) = recording_mt_context_with_spawn_and_limit(
            1024 * 1024,
            max_frame_length(&AES128, circuit_count),
            concurrency as usize,
            |f| {
                let _ = web_spawn::spawn(f);
                Ok(())
            },
        );
        run_full_protocol(&mut exec_p, &mut exec_v, circuit_count, 0).await;

        total_elapsed_ms += performance.now() - start;
    }

    BenchResult {
        elapsed_ms: total_elapsed_ms,
        and_gates: n as u64 * actual_gates,
    }
}
