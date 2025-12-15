//! Benchmarks for mpz-zk-core (QuickSilver ZK protocol).
//!
//! Mirrors the WASM zk_core benchmarks:
//! - prover_execute: only prover execute phase
//! - verifier_execute: only verifier execute phase
//! - full_protocol: execute + check

use blake3::Hasher;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use mpz_circuits::AES128;
use mpz_memory_core::correlated::{Delta, Key, Mac};
use mpz_ot_core::{
    ideal::rcot::IdealRCOT,
    rcot::{RCOTReceiverOutput, RCOTSenderOutput},
};
use mpz_zk_core::{Prover, Verifier};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

const CIRCUIT_COUNT: usize = 1000;

fn bench_prover_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    // Throughput in AND gates (elem/s = AND gates/s)
    group.throughput(Throughput::Elements((and_count * CIRCUIT_COUNT) as u64));

    // Setup correlations
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

    let total_inputs = inputs_per_circuit * CIRCUIT_COUNT;
    rcot.alloc(total_inputs);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { .. },
        RCOTReceiverOutput {
            msgs: mut macs,
            choices,
            ..
        },
    ) = rcot.transfer(total_inputs).unwrap();
    macs.iter_mut()
        .zip(&choices)
        .for_each(|(mac, &choice)| mac.set_lsb(choice));
    let input_macs = Mac::from_blocks(macs);

    let total_and_gates = and_count * CIRCUIT_COUNT;
    rcot.alloc(total_and_gates);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { .. },
        RCOTReceiverOutput {
            choices: gate_masks,
            msgs: macs,
            ..
        },
    ) = rcot.transfer(total_and_gates).unwrap();
    let gate_macs = Mac::from_blocks(macs);

    group.bench_function("prover_execute", |b| {
        b.iter(|| {
            let mut prover = Prover::default();

            for i in 0..CIRCUIT_COUNT {
                let input_start = i * inputs_per_circuit;
                let input_end = input_start + inputs_per_circuit;
                let gate_start = i * and_count;
                let gate_end = gate_start + and_count;

                let mut prover_exec = prover
                    .execute(
                        circuit.clone(),
                        &input_macs[input_start..input_end],
                        &gate_masks[gate_start..gate_end],
                        &gate_macs[gate_start..gate_end],
                    )
                    .unwrap();

                for _ in prover_exec.iter() {}
                let _ = prover_exec.finish().unwrap();
            }

            black_box(())
        })
    });

    group.finish();
}

fn bench_verifier_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    // Throughput in AND gates (elem/s = AND gates/s)
    group.throughput(Throughput::Elements((and_count * CIRCUIT_COUNT) as u64));

    // Setup correlations
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

    let total_inputs = inputs_per_circuit * CIRCUIT_COUNT;
    rcot.alloc(total_inputs);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { mut keys, .. },
        RCOTReceiverOutput {
            msgs: mut macs,
            choices,
            ..
        },
    ) = rcot.transfer(total_inputs).unwrap();
    keys.iter_mut().for_each(|key| key.set_lsb(false));
    macs.iter_mut()
        .zip(&choices)
        .for_each(|(mac, &choice)| mac.set_lsb(choice));
    let input_keys = Key::from_blocks(keys);
    let input_macs = Mac::from_blocks(macs);

    let total_and_gates = and_count * CIRCUIT_COUNT;
    rcot.alloc(total_and_gates);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { keys, .. },
        RCOTReceiverOutput {
            choices: gate_masks,
            msgs: macs,
            ..
        },
    ) = rcot.transfer(total_and_gates).unwrap();
    let gate_keys = Key::from_blocks(keys);
    let gate_macs = Mac::from_blocks(macs);

    // Pre-generate adjustments
    let adjustments: Vec<Vec<bool>> = {
        let mut prover = Prover::default();
        let mut all_adjustments = Vec::with_capacity(CIRCUIT_COUNT);

        for i in 0..CIRCUIT_COUNT {
            let input_start = i * inputs_per_circuit;
            let input_end = input_start + inputs_per_circuit;
            let gate_start = i * and_count;
            let gate_end = gate_start + and_count;

            let mut prover_exec = prover
                .execute(
                    circuit.clone(),
                    &input_macs[input_start..input_end],
                    &gate_masks[gate_start..gate_end],
                    &gate_macs[gate_start..gate_end],
                )
                .unwrap();

            let adj: Vec<bool> = prover_exec.iter().collect();
            let _ = prover_exec.finish().unwrap();
            all_adjustments.push(adj);
        }

        all_adjustments
    };

    group.bench_function("verifier_execute", |b| {
        b.iter(|| {
            let mut verifier = Verifier::new(delta);

            for i in 0..CIRCUIT_COUNT {
                let input_start = i * inputs_per_circuit;
                let input_end = input_start + inputs_per_circuit;
                let gate_start = i * and_count;
                let gate_end = gate_start + and_count;

                let mut verifier_exec = verifier
                    .execute(
                        circuit.clone(),
                        &input_keys[input_start..input_end],
                        &gate_keys[gate_start..gate_end],
                    )
                    .unwrap();

                let mut consumer = verifier_exec.consumer();
                for &adjust in &adjustments[i] {
                    consumer.next(adjust);
                }

                let _ = verifier_exec.finish().unwrap();
            }

            black_box(())
        })
    });

    group.finish();
}

fn bench_full_protocol(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    // Throughput in AND gates (elem/s = AND gates/s)
    group.throughput(Throughput::Elements((and_count * CIRCUIT_COUNT) as u64));

    // Setup correlations
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

    let total_inputs = inputs_per_circuit * CIRCUIT_COUNT;
    rcot.alloc(total_inputs);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { mut keys, .. },
        RCOTReceiverOutput {
            msgs: mut macs,
            choices,
            ..
        },
    ) = rcot.transfer(total_inputs).unwrap();
    keys.iter_mut().for_each(|key| key.set_lsb(false));
    macs.iter_mut()
        .zip(&choices)
        .for_each(|(mac, &choice)| mac.set_lsb(choice));
    let input_keys = Key::from_blocks(keys);
    let input_macs = Mac::from_blocks(macs);

    let total_and_gates = and_count * CIRCUIT_COUNT;
    rcot.alloc(total_and_gates);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { keys, .. },
        RCOTReceiverOutput {
            choices: gate_masks,
            msgs: macs,
            ..
        },
    ) = rcot.transfer(total_and_gates).unwrap();
    let gate_keys = Key::from_blocks(keys);
    let gate_macs = Mac::from_blocks(macs);

    // SVOLE for check phase
    rcot.alloc(128);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput {
            keys: svole_keys, ..
        },
        RCOTReceiverOutput {
            choices: svole_choices,
            msgs: svole_ev,
            ..
        },
    ) = rcot.transfer(128).unwrap();

    group.bench_function("full_protocol", |b| {
        b.iter(|| {
            let mut prover = Prover::default();
            let mut verifier = Verifier::new(delta);
            let mut prover_transcript = Hasher::default();
            let mut verifier_transcript = Hasher::default();

            for i in 0..CIRCUIT_COUNT {
                let input_start = i * inputs_per_circuit;
                let input_end = input_start + inputs_per_circuit;
                let gate_start = i * and_count;
                let gate_end = gate_start + and_count;

                let mut prover_exec = prover
                    .execute(
                        circuit.clone(),
                        &input_macs[input_start..input_end],
                        &gate_masks[gate_start..gate_end],
                        &gate_macs[gate_start..gate_end],
                    )
                    .unwrap();
                let mut verifier_exec = verifier
                    .execute(
                        circuit.clone(),
                        &input_keys[input_start..input_end],
                        &gate_keys[gate_start..gate_end],
                    )
                    .unwrap();

                let mut consumer = verifier_exec.consumer();
                for adjust in prover_exec.iter() {
                    consumer.next(adjust);
                }

                let _ = prover_exec.finish().unwrap();
                let _ = verifier_exec.finish().unwrap();
            }

            // Check phase
            let uv = prover
                .check(&mut prover_transcript, &svole_choices, &svole_ev)
                .unwrap();
            verifier
                .check(&mut verifier_transcript, &svole_keys, uv)
                .unwrap();

            black_box(())
        })
    });

    group.finish();
}

fn bench_prover_check_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    // Throughput in AND gates (elem/s = AND gates/s)
    group.throughput(Throughput::Elements((and_count * CIRCUIT_COUNT) as u64));

    // Setup correlations
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

    let total_inputs = inputs_per_circuit * CIRCUIT_COUNT;
    rcot.alloc(total_inputs);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { mut keys, .. },
        RCOTReceiverOutput {
            msgs: mut macs,
            choices,
            ..
        },
    ) = rcot.transfer(total_inputs).unwrap();
    keys.iter_mut().for_each(|key| key.set_lsb(false));
    macs.iter_mut()
        .zip(&choices)
        .for_each(|(mac, &choice)| mac.set_lsb(choice));
    let input_keys = Key::from_blocks(keys);
    let input_macs = Mac::from_blocks(macs);

    let total_and_gates = and_count * CIRCUIT_COUNT;
    rcot.alloc(total_and_gates);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { keys, .. },
        RCOTReceiverOutput {
            choices: gate_masks,
            msgs: macs,
            ..
        },
    ) = rcot.transfer(total_and_gates).unwrap();
    let gate_keys = Key::from_blocks(keys);
    let gate_macs = Mac::from_blocks(macs);

    // SVOLE for check phase
    rcot.alloc(128);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { .. },
        RCOTReceiverOutput {
            choices: svole_choices,
            msgs: svole_ev,
            ..
        },
    ) = rcot.transfer(128).unwrap();

    group.bench_function("prover_check_only", |b| {
        b.iter_batched(
            || {
                // SETUP (not timed): build prover+verifier, run execute
                let mut prover = Prover::default();
                let mut verifier = Verifier::new(delta);

                for i in 0..CIRCUIT_COUNT {
                    let input_start = i * inputs_per_circuit;
                    let input_end = input_start + inputs_per_circuit;
                    let gate_start = i * and_count;
                    let gate_end = gate_start + and_count;

                    let mut prover_exec = prover
                        .execute(
                            circuit.clone(),
                            &input_macs[input_start..input_end],
                            &gate_masks[gate_start..gate_end],
                            &gate_macs[gate_start..gate_end],
                        )
                        .unwrap();
                    let mut verifier_exec = verifier
                        .execute(
                            circuit.clone(),
                            &input_keys[input_start..input_end],
                            &gate_keys[gate_start..gate_end],
                        )
                        .unwrap();

                    let mut consumer = verifier_exec.consumer();
                    for adjust in prover_exec.iter() {
                        consumer.next(adjust);
                    }

                    let _ = prover_exec.finish().unwrap();
                    let _ = verifier_exec.finish().unwrap();
                }

                (prover, Hasher::default())
            },
            |(mut prover, mut prover_transcript)| {
                // ROUTINE (timed): only prover check
                let _uv = prover
                    .check(&mut prover_transcript, &svole_choices, &svole_ev)
                    .unwrap();
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn bench_verifier_check_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone().into();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();

    // Throughput in AND gates (elem/s = AND gates/s)
    group.throughput(Throughput::Elements((and_count * CIRCUIT_COUNT) as u64));

    // Setup correlations
    let mut rng = StdRng::seed_from_u64(0);
    let delta = Delta::random(&mut rng);
    let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

    let total_inputs = inputs_per_circuit * CIRCUIT_COUNT;
    rcot.alloc(total_inputs);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { mut keys, .. },
        RCOTReceiverOutput {
            msgs: mut macs,
            choices,
            ..
        },
    ) = rcot.transfer(total_inputs).unwrap();
    keys.iter_mut().for_each(|key| key.set_lsb(false));
    macs.iter_mut()
        .zip(&choices)
        .for_each(|(mac, &choice)| mac.set_lsb(choice));
    let input_keys = Key::from_blocks(keys);
    let input_macs = Mac::from_blocks(macs);

    let total_and_gates = and_count * CIRCUIT_COUNT;
    rcot.alloc(total_and_gates);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput { keys, .. },
        RCOTReceiverOutput {
            choices: gate_masks,
            msgs: macs,
            ..
        },
    ) = rcot.transfer(total_and_gates).unwrap();
    let gate_keys = Key::from_blocks(keys);
    let gate_macs = Mac::from_blocks(macs);

    // SVOLE for check phase
    rcot.alloc(128);
    rcot.flush().unwrap();
    let (
        RCOTSenderOutput {
            keys: svole_keys, ..
        },
        RCOTReceiverOutput {
            choices: svole_choices,
            msgs: svole_ev,
            ..
        },
    ) = rcot.transfer(128).unwrap();

    group.bench_function("verifier_check_only", |b| {
        b.iter_batched(
            || {
                // SETUP (not timed): build prover+verifier, run execute, run prover check
                let mut prover = Prover::default();
                let mut verifier = Verifier::new(delta);

                for i in 0..CIRCUIT_COUNT {
                    let input_start = i * inputs_per_circuit;
                    let input_end = input_start + inputs_per_circuit;
                    let gate_start = i * and_count;
                    let gate_end = gate_start + and_count;

                    let mut prover_exec = prover
                        .execute(
                            circuit.clone(),
                            &input_macs[input_start..input_end],
                            &gate_masks[gate_start..gate_end],
                            &gate_macs[gate_start..gate_end],
                        )
                        .unwrap();
                    let mut verifier_exec = verifier
                        .execute(
                            circuit.clone(),
                            &input_keys[input_start..input_end],
                            &gate_keys[gate_start..gate_end],
                        )
                        .unwrap();

                    let mut consumer = verifier_exec.consumer();
                    for adjust in prover_exec.iter() {
                        consumer.next(adjust);
                    }

                    let _ = prover_exec.finish().unwrap();
                    let _ = verifier_exec.finish().unwrap();
                }

                // Run prover check to get UV (not timed)
                let mut prover_transcript = Hasher::default();
                let uv = prover
                    .check(&mut prover_transcript, &svole_choices, &svole_ev)
                    .unwrap();

                (verifier, Hasher::default(), uv)
            },
            |(mut verifier, mut verifier_transcript, uv)| {
                // ROUTINE (timed): only verifier check
                verifier
                    .check(&mut verifier_transcript, &svole_keys, uv)
                    .unwrap();
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_prover_execute, bench_verifier_execute, bench_prover_check_only, bench_verifier_check_only, bench_full_protocol);
criterion_main!(benches);
