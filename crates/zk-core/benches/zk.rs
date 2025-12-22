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

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("zk-core");

    group.throughput(Throughput::Bytes(16));
    group.bench_function("aes128", |b| {
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        rcot.alloc(AES128.inputs().len());
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { mut keys, .. },
            RCOTReceiverOutput {
                msgs: mut macs,
                choices,
                ..
            },
        ) = rcot.transfer(AES128.inputs().len()).unwrap();
        keys.iter_mut().for_each(|key| key.set_lsb(false));
        macs.iter_mut()
            .zip(choices)
            .for_each(|(mac, choice)| mac.set_lsb(choice));

        let input_keys = Key::from_blocks(keys);
        let input_macs = Mac::from_blocks(macs);

        rcot.alloc(AES128.and_count());
        rcot.flush().unwrap();
        let (
            RCOTSenderOutput { keys, .. },
            RCOTReceiverOutput {
                choices: gate_masks,
                msgs: macs,
                ..
            },
        ) = rcot.transfer(AES128.and_count()).unwrap();
        let gate_keys = Key::from_blocks(keys);
        let gate_macs = Mac::from_blocks(macs);

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

        let mut prover = Prover::default();
        let mut verifier = Verifier::new(delta);
        let mut prover_transcript = Hasher::default();
        let mut verifier_transcript = Hasher::default();

        b.iter(|| {
            let mut prover_execute = prover
                .execute(AES128.clone(), &input_macs, &gate_masks, &gate_macs)
                .unwrap();
            let mut verifier_execute = verifier
                .execute(AES128.clone(), &input_keys, &gate_keys)
                .unwrap();

            let mut verifier_consumer = verifier_execute.consumer();
            for adjust in prover_execute.iter() {
                verifier_consumer.next(adjust);
            }

            let output_macs = prover_execute.finish().unwrap();
            let output_keys = verifier_execute.finish().unwrap();

            let uv = prover
                .check(&mut prover_transcript, &svole_choices, &svole_ev)
                .unwrap();
            verifier
                .check(&mut verifier_transcript, &svole_keys, uv)
                .unwrap();

            black_box((output_macs, output_keys))
        })
    });
}

/// Benchmark prover check phase with 10M gates (comparable to WASM benchmark).
fn bench_prover_check_10m(c: &mut Criterion) {
    const TARGET_GATES: usize = 10_000_000;

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();
    let circuit_count = TARGET_GATES.div_ceil(and_count);
    let actual_gates = circuit_count * and_count;

    let mut group = c.benchmark_group("zk-core-check");
    group.throughput(Throughput::Elements(actual_gates as u64));
    group.sample_size(10);

    group.bench_function("prover_check_10m", |b| {
        // Setup correlations (once)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        // Input correlations
        let total_inputs = inputs_per_circuit * circuit_count;
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

        // Gate correlations
        let total_and_gates = and_count * circuit_count;
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

        // Use iter_batched to separate setup from measurement
        b.iter_batched(
            || {
                // Setup: accumulate all circuits (not timed)
                let mut prover = Prover::default();

                for i in 0..circuit_count {
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

                    // Consume adjustments
                    for _ in prover_exec.iter() {}
                    let _ = prover_exec.finish().unwrap();
                }

                prover
            },
            |mut prover| {
                // Timed: only the check phase
                let mut prover_transcript = Hasher::default();
                let uv = prover
                    .check(&mut prover_transcript, &svole_choices, &svole_ev)
                    .unwrap();
                black_box(uv)
            },
            criterion::BatchSize::PerIteration,
        )
    });
}

/// Benchmark prover check phase with 400K gates.
fn bench_prover_check_400k(c: &mut Criterion) {
    const TARGET_GATES: usize = 400_000;

    let circuit: Arc<mpz_circuits::Circuit> = AES128.clone();
    let and_count = circuit.and_count();
    let inputs_per_circuit = circuit.inputs().len();
    let circuit_count = TARGET_GATES.div_ceil(and_count);
    let actual_gates = circuit_count * and_count;

    let mut group = c.benchmark_group("zk-core-check");
    group.throughput(Throughput::Elements(actual_gates as u64));
    group.sample_size(50);

    group.bench_function("prover_check_400k", |b| {
        // Setup correlations (once)
        let mut rng = StdRng::seed_from_u64(0);
        let delta = Delta::random(&mut rng);
        let mut rcot = IdealRCOT::new(rng.random(), delta.into_inner());

        // Input correlations
        let total_inputs = inputs_per_circuit * circuit_count;
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

        // Gate correlations
        let total_and_gates = and_count * circuit_count;
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

        // Use iter_batched to separate setup from measurement
        b.iter_batched(
            || {
                // Setup: accumulate all circuits (not timed)
                let mut prover = Prover::default();

                for i in 0..circuit_count {
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

                    // Consume adjustments
                    for _ in prover_exec.iter() {}
                    let _ = prover_exec.finish().unwrap();
                }

                prover
            },
            |mut prover| {
                // Timed: only the check phase
                let mut prover_transcript = Hasher::default();
                let uv = prover
                    .check(&mut prover_transcript, &svole_choices, &svole_ev)
                    .unwrap();
                black_box(uv)
            },
            criterion::BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, criterion_benchmark, bench_prover_check_400k, bench_prover_check_10m);
criterion_main!(benches);
