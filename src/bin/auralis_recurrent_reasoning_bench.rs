use auralis::brain_ab::AbProtocol;
use auralis::manifest::fingerprint_params;
use auralis::model::{Config, Gpt};
use auralis::optim::{Adam, Optimizer};
use auralis::recurrent::RecurrentConfig;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Measurement {
    reasoning_steps: usize,
    optimizer_steps: usize,
    block_applications: usize,
    tokens_seen: usize,
    train_ns: u64,
    inference_ns: u64,
    eval_loss: f32,
    perplexity: f32,
    parameter_count: usize,
    state_fingerprint: u64,
    max_pre_clip_grad_norm: f32,
}

fn model_config() -> Config {
    Config {
        vocab: 8,
        n_embd: 8,
        n_head: 2,
        n_layer: 1,
        block: 8,
        n_ff: 16,
    }
}

fn token_stream(vocab: usize, count: usize) -> Vec<usize> {
    (0..count).map(|i| (i * 37 + i / 7 + 11) % vocab).collect()
}

fn global_l2(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt() as f32
}

fn clip_in_place(values: &mut [f32], threshold: f32) -> f32 {
    let norm = global_l2(values);
    if norm > threshold {
        let scale = threshold / norm;
        for value in values {
            *value *= scale;
        }
    }
    norm
}

fn run_once(
    reasoning_steps: usize,
    block_budget: usize,
    protocol: AbProtocol,
) -> Measurement {
    let cfg = model_config();
    let recurrent = RecurrentConfig::new(reasoning_steps).unwrap();
    let applications_per_update = recurrent.block_applications(cfg.n_layer).unwrap();
    assert_eq!(block_budget % applications_per_update, 0);
    let optimizer_steps = block_budget / applications_per_update;
    assert!(optimizer_steps > 0);

    let mut rng = StdRng::seed_from_u64(protocol.seed);
    let mut gpt = Gpt::new(cfg, &mut rng);
    let parameter_count = gpt.collect_params().len();
    let mut optimizer = Adam::new(parameter_count, protocol.learning_rate);
    let mut grads = vec![0.0f32; parameter_count];
    let stream = token_stream(cfg.vocab, protocol.token_count);

    let started = Instant::now();
    let mut max_pre_clip_grad_norm = 0.0f32;
    for update in 0..optimizer_steps {
        let max_start = stream.len() - cfg.block - 1;
        let start = (update * 17 + 3) % max_start;
        let x = &stream[start..start + cfg.block];
        let y = &stream[start + 1..start + cfg.block + 1];

        grads.fill(0.0);
        let loss = gpt
            .backward_recurrent_into(x, y, &mut grads, recurrent)
            .expect("recurrent backward");
        assert!(loss.is_finite());
        let grad_norm = clip_in_place(&mut grads, protocol.grad_clip_norm);
        assert!(grad_norm.is_finite());
        max_pre_clip_grad_norm = max_pre_clip_grad_norm.max(grad_norm);

        let mut params = gpt.collect_params();
        optimizer.update(&mut params, &grads).unwrap();
        gpt.write_params(&params);
    }
    let train_ns = started.elapsed().as_nanos().max(1) as u64;

    let eval_x = &stream[64..64 + cfg.block];
    let eval_y = &stream[65..65 + cfg.block];
    let eval_loss = gpt.loss_recurrent(eval_x, eval_y, recurrent).unwrap();
    let perplexity = eval_loss.exp();

    let inference_started = Instant::now();
    for _ in 0..100 {
        std::hint::black_box(gpt.logits_recurrent(eval_x, recurrent).unwrap());
    }
    let inference_ns = inference_started.elapsed().as_nanos().max(1) as u64;

    let params = gpt.collect_params();
    Measurement {
        reasoning_steps,
        optimizer_steps,
        block_applications: optimizer_steps * applications_per_update,
        tokens_seen: optimizer_steps * cfg.block,
        train_ns,
        inference_ns,
        eval_loss,
        perplexity,
        parameter_count,
        state_fingerprint: fingerprint_params(&params),
        max_pre_clip_grad_norm,
    }
}

fn median_u64(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_f32(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let protocol = AbProtocol {
        seed: 659_918,
        steps: 12,
        repeats: 3,
        batch_size: 1,
        gradient_accumulation_steps: 1,
        learning_rate: 1e-3,
        grad_clip_norm: 1.0,
        token_count: 512,
    };
    protocol.validate().unwrap();
    let block_budget = 12usize;

    println!(
        "recurrent_protocol | schema=1 seed={} repeats={} block_budget={} batch={} accum={} lr={} clip={} token_count={} physical_layers=1",
        protocol.seed,
        protocol.repeats,
        block_budget,
        protocol.batch_size,
        protocol.gradient_accumulation_steps,
        protocol.learning_rate,
        protocol.grad_clip_norm,
        protocol.token_count,
    );

    for reasoning_steps in [1usize, 2, 3] {
        let mut measurements = Vec::with_capacity(protocol.repeats);
        for repetition in 0..protocol.repeats {
            let m = run_once(reasoning_steps, block_budget, protocol);
            println!(
                "recurrent_run | reasoning_steps={} repetition={} optimizer_steps={} block_applications={} tokens_seen={} train_ns={} inference_100_ns={} eval_loss={:.6} ppl={:.6} params={} state_fingerprint={:016x} max_pre_clip_grad_norm={:.6}",
                m.reasoning_steps,
                repetition + 1,
                m.optimizer_steps,
                m.block_applications,
                m.tokens_seen,
                m.train_ns,
                m.inference_ns,
                m.eval_loss,
                m.perplexity,
                m.parameter_count,
                m.state_fingerprint,
                m.max_pre_clip_grad_norm,
            );
            measurements.push(m);
        }

        let first = &measurements[0];
        assert!(measurements.iter().all(|m| {
            m.state_fingerprint == first.state_fingerprint
                && m.eval_loss.to_bits() == first.eval_loss.to_bits()
                && m.perplexity.to_bits() == first.perplexity.to_bits()
                && m.parameter_count == first.parameter_count
                && m.block_applications == block_budget
        }));

        let mut train: Vec<u64> = measurements.iter().map(|m| m.train_ns).collect();
        let mut inference: Vec<u64> = measurements.iter().map(|m| m.inference_ns).collect();
        let mut loss: Vec<f32> = measurements.iter().map(|m| m.eval_loss).collect();
        let mut ppl: Vec<f32> = measurements.iter().map(|m| m.perplexity).collect();
        let median_train = median_u64(&mut train);
        let median_inference = median_u64(&mut inference);
        let median_loss = median_f32(&mut loss);
        let median_ppl = median_f32(&mut ppl);

        println!(
            "recurrent_curve | reasoning_steps={} optimizer_steps={} block_applications={} tokens_seen={} params={} median_train_ns={} ns_per_block_application={:.3} median_inference_100_ns={} eval_loss={:.6} ppl={:.6} state_reproducible=true",
            reasoning_steps,
            first.optimizer_steps,
            block_budget,
            first.tokens_seen,
            first.parameter_count,
            median_train,
            median_train as f64 / block_budget as f64,
            median_inference,
            median_loss,
            median_ppl,
        );
    }
}
