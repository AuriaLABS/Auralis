//! Reproducible reference training loop primitives.
//!
//! Foundation keeps the simple allocation-heavy path as the semantic reference.
//! Engine adds explicitly reusable workspaces and must remain equivalent to it.

use crate::batch::{
    backward_batch_into, backward_deterministic_batch_from_stream_into,
    deterministic_batch_from_stream,
};
use crate::model::{BackwardWorkspace, Gpt};
use crate::optim::Adam;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    pub seed: u64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub grad_clip_norm: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 0xA11CE,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            grad_clip_norm: 1.0,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.batch_size == 0 {
            return Err("batch_size must be positive");
        }
        if self.gradient_accumulation_steps == 0 {
            return Err("gradient_accumulation_steps must be positive");
        }
        if !self.grad_clip_norm.is_finite() || self.grad_clip_norm <= 0.0 {
            return Err("grad_clip_norm must be finite and positive");
        }
        Ok(())
    }

    pub fn effective_batch_size(&self) -> Result<usize, &'static str> {
        self.batch_size
            .checked_mul(self.gradient_accumulation_steps)
            .ok_or("effective batch size overflow")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepMetrics {
    pub global_step: u64,
    pub loss: f32,
    pub grad_norm_before_clip: f32,
    pub grad_scale: f32,
    pub tokens: usize,
    pub microbatches: usize,
    pub effective_batch_size: usize,
}

/// Reusable Engine buffers that sit outside the optimizer-step hot path.
///
/// Besides micro/sample gradients, Engine keeps a flat parameter mirror for
/// Adam and an opaque model backward workspace. All are initialized once and
/// reused across samples and optimizer steps.
#[derive(Debug)]
pub struct TrainWorkspace {
    micro_grads: Vec<f32>,
    sample_grads: Vec<f32>,
    params: Vec<f32>,
    backward: BackwardWorkspace,
}

impl TrainWorkspace {
    pub fn new(gpt: &Gpt) -> Self {
        let params = gpt.collect_params();
        let param_count = params.len();
        Self {
            micro_grads: vec![0.0; param_count],
            sample_grads: vec![0.0; param_count],
            params,
            backward: BackwardWorkspace::new(gpt),
        }
    }

    pub fn len(&self) -> usize {
        self.micro_grads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.micro_grads.is_empty()
    }

    fn matches(&self, gpt: &Gpt, param_count: usize) -> bool {
        self.micro_grads.len() == param_count
            && self.sample_grads.len() == param_count
            && self.params.len() == param_count
            && self.backward.matches(gpt)
    }
}

pub fn global_l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt() as f32
}

/// Execute one optimizer step using deterministic microbatch accumulation.
///
/// This is the Foundation reference path. It intentionally materializes
/// batches and allocates scratch buffers so future Engine paths can be checked
/// against a simple implementation.
pub fn train_step(
    gpt: &mut Gpt,
    adam: &mut Adam,
    train_tokens: &[usize],
    cfg: TrainConfig,
    global_step: u64,
    grads: &mut [f32],
) -> Result<StepMetrics, &'static str> {
    cfg.validate()?;
    let effective_batch_size = cfg.effective_batch_size()?;
    if grads.len() != gpt.collect_params().len() {
        return Err("gradient buffer has wrong size");
    }

    grads.fill(0.0);
    let mut micro_grads = vec![0.0f32; grads.len()];
    let mut loss_sum = 0.0f32;

    for micro in 0..cfg.gradient_accumulation_steps {
        let stream_offset = (micro as u64)
            .checked_mul(cfg.batch_size as u64)
            .ok_or("batch stream overflow")?;
        let batch = deterministic_batch_from_stream(
            train_tokens,
            gpt.cfg.block,
            cfg.batch_size,
            cfg.seed,
            global_step,
            stream_offset,
        )?;

        micro_grads.fill(0.0);
        loss_sum += backward_batch_into(gpt, &batch, &mut micro_grads)?;
        for (dst, src) in grads.iter_mut().zip(&micro_grads) {
            *dst += *src;
        }
    }

    finish_step(
        gpt,
        adam,
        cfg,
        global_step,
        effective_batch_size,
        grads,
        loss_sum,
    )
}

/// Engine optimizer step with reusable data/gradient/parameter buffers.
///
/// Data selection, example order, gradient accumulation order, clipping and
/// Adam updates are identical to `train_step`. The only difference is memory
/// behavior: deterministic windows are borrowed directly from the token stream,
/// flat buffers are reused, model-structured gradients are reset in place, and
/// Adam updates a persistent flat parameter mirror.
pub fn train_step_reuse(
    gpt: &mut Gpt,
    adam: &mut Adam,
    train_tokens: &[usize],
    cfg: TrainConfig,
    global_step: u64,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> Result<StepMetrics, &'static str> {
    cfg.validate()?;
    let effective_batch_size = cfg.effective_batch_size()?;
    if !workspace.matches(gpt, grads.len()) {
        return Err("training workspace has wrong size or model config");
    }

    grads.fill(0.0);
    let mut loss_sum = 0.0f32;

    for micro in 0..cfg.gradient_accumulation_steps {
        let stream_offset = (micro as u64)
            .checked_mul(cfg.batch_size as u64)
            .ok_or("batch stream overflow")?;
        loss_sum += backward_deterministic_batch_from_stream_into(
            gpt,
            train_tokens,
            gpt.cfg.block,
            cfg.batch_size,
            cfg.seed,
            global_step,
            stream_offset,
            &mut workspace.micro_grads,
            &mut workspace.sample_grads,
            &mut workspace.backward,
        )?;
        for (dst, src) in grads.iter_mut().zip(&workspace.micro_grads) {
            *dst += *src;
        }
    }

    finish_step_with_params(
        gpt,
        adam,
        cfg,
        global_step,
        effective_batch_size,
        grads,
        loss_sum,
        &mut workspace.params,
    )
}

fn prepare_grads(
    cfg: TrainConfig,
    grads: &mut [f32],
    loss_sum: f32,
) -> Result<(f32, f32, f32), &'static str> {
    let inv_accum = 1.0 / cfg.gradient_accumulation_steps as f32;
    for g in grads.iter_mut() {
        *g *= inv_accum;
    }
    let loss = loss_sum * inv_accum;

    let grad_norm = global_l2_norm(grads);
    if !loss.is_finite() || !grad_norm.is_finite() {
        return Err("non-finite training state");
    }

    let grad_scale = if grad_norm > cfg.grad_clip_norm {
        cfg.grad_clip_norm / grad_norm
    } else {
        1.0
    };
    if grad_scale < 1.0 {
        for g in grads.iter_mut() {
            *g *= grad_scale;
        }
    }
    Ok((loss, grad_norm, grad_scale))
}

fn make_metrics(
    gpt: &Gpt,
    cfg: TrainConfig,
    global_step: u64,
    effective_batch_size: usize,
    loss: f32,
    grad_norm: f32,
    grad_scale: f32,
) -> Result<StepMetrics, &'static str> {
    let tokens = effective_batch_size
        .checked_mul(gpt.cfg.block)
        .ok_or("processed token count overflow")?;

    Ok(StepMetrics {
        global_step,
        loss,
        grad_norm_before_clip: grad_norm,
        grad_scale,
        tokens,
        microbatches: cfg.gradient_accumulation_steps,
        effective_batch_size,
    })
}

fn finish_step(
    gpt: &mut Gpt,
    adam: &mut Adam,
    cfg: TrainConfig,
    global_step: u64,
    effective_batch_size: usize,
    grads: &mut [f32],
    loss_sum: f32,
) -> Result<StepMetrics, &'static str> {
    let (loss, grad_norm, grad_scale) = prepare_grads(cfg, grads, loss_sum)?;

    let mut params = gpt.collect_params();
    adam.step(&mut params, grads);
    if params.iter().any(|x| !x.is_finite()) {
        return Err("optimizer produced non-finite parameters");
    }
    gpt.write_params(&params);

    make_metrics(
        gpt,
        cfg,
        global_step,
        effective_batch_size,
        loss,
        grad_norm,
        grad_scale,
    )
}

fn finish_step_with_params(
    gpt: &mut Gpt,
    adam: &mut Adam,
    cfg: TrainConfig,
    global_step: u64,
    effective_batch_size: usize,
    grads: &mut [f32],
    loss_sum: f32,
    params: &mut [f32],
) -> Result<StepMetrics, &'static str> {
    let (loss, grad_norm, grad_scale) = prepare_grads(cfg, grads, loss_sum)?;

    adam.step(params, grads);
    if params.iter().any(|x| !x.is_finite()) {
        return Err("optimizer produced non-finite parameters");
    }
    gpt.write_params(params);

    make_metrics(
        gpt,
        cfg,
        global_step,
        effective_batch_size,
        loss,
        grad_norm,
        grad_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn model(seed: u64) -> Gpt {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(seed);
        Gpt::new(cfg, &mut rng)
    }

    fn cfg(seed: u64, batch_size: usize) -> TrainConfig {
        TrainConfig {
            seed,
            batch_size,
            gradient_accumulation_steps: 1,
            grad_clip_norm: 1.0,
        }
    }

    #[test]
    fn same_seed_state_and_step_produce_identical_update() {
        let mut a = model(9);
        let mut b = model(9);
        assert_eq!(a.collect_params(), b.collect_params());
        let tokens: Vec<usize> = (0..80).map(|i| i % 7).collect();
        let cfg = cfg(1234, 3);
        let n = a.collect_params().len();
        let mut adam_a = Adam::new(n, 3e-3);
        let mut adam_b = Adam::new(n, 3e-3);
        let mut ga = vec![0.0; n];
        let mut gb = vec![0.0; n];
        let ma = train_step(&mut a, &mut adam_a, &tokens, cfg, 0, &mut ga).unwrap();
        let mb = train_step(&mut b, &mut adam_b, &tokens, cfg, 0, &mut gb).unwrap();
        assert_eq!(ma, mb);
        assert_eq!(ga, gb);
        assert_eq!(a.collect_params(), b.collect_params());
        assert_eq!(adam_a.export().1, adam_b.export().1);
    }

    #[test]
    fn different_global_step_selects_a_different_training_batch() {
        let mut a = model(12);
        let mut b = a.clone();
        let tokens: Vec<usize> = (0..100).map(|i| (i * 3) % 7).collect();
        let cfg = cfg(88, 4);
        let n = a.collect_params().len();
        let mut adam_a = Adam::new(n, 1e-3);
        let mut adam_b = Adam::new(n, 1e-3);
        let mut ga = vec![0.0; n];
        let mut gb = vec![0.0; n];
        train_step(&mut a, &mut adam_a, &tokens, cfg, 0, &mut ga).unwrap();
        train_step(&mut b, &mut adam_b, &tokens, cfg, 1, &mut gb).unwrap();
        assert_ne!(ga, gb);
        assert_ne!(a.collect_params(), b.collect_params());
    }

    #[test]
    fn clipping_never_increases_gradient_norm() {
        let mut gpt = model(77);
        let tokens: Vec<usize> = (0..60).map(|i| i % 7).collect();
        let cfg = TrainConfig {
            seed: 5,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            grad_clip_norm: 1e-4,
        };
        let n = gpt.collect_params().len();
        let mut adam = Adam::new(n, 1e-3);
        let mut grads = vec![0.0; n];
        let m = train_step(&mut gpt, &mut adam, &tokens, cfg, 0, &mut grads).unwrap();
        assert!(m.grad_norm_before_clip >= global_l2_norm(&grads));
        assert!(global_l2_norm(&grads) <= cfg.grad_clip_norm * 1.001);
        assert!(m.grad_scale <= 1.0);
    }

    #[test]
    fn accumulation_matches_one_larger_effective_batch() {
        let mut accumulated = model(41);
        let mut reference = accumulated.clone();
        let tokens: Vec<usize> = (0..120).map(|i| (i * 5 + 1) % 7).collect();
        let n = accumulated.collect_params().len();
        let mut adam_acc = Adam::new(n, 1e-3);
        let mut adam_ref = Adam::new(n, 1e-3);
        let mut g_acc = vec![0.0; n];
        let mut g_ref = vec![0.0; n];

        let acc_cfg = TrainConfig {
            seed: 9001,
            batch_size: 2,
            gradient_accumulation_steps: 3,
            grad_clip_norm: 1000.0,
        };
        let ref_cfg = TrainConfig {
            seed: 9001,
            batch_size: 6,
            gradient_accumulation_steps: 1,
            grad_clip_norm: 1000.0,
        };

        let a = train_step(
            &mut accumulated,
            &mut adam_acc,
            &tokens,
            acc_cfg,
            7,
            &mut g_acc,
        )
        .unwrap();
        let b = train_step(
            &mut reference,
            &mut adam_ref,
            &tokens,
            ref_cfg,
            7,
            &mut g_ref,
        )
        .unwrap();

        assert_eq!(a.effective_batch_size, 6);
        assert_eq!(a.tokens, b.tokens);
        assert!((a.loss - b.loss).abs() < 1e-6);
        for (x, y) in g_acc.iter().zip(&g_ref) {
            assert!((*x - *y).abs() < 1e-5);
        }
        for (x, y) in accumulated
            .collect_params()
            .iter()
            .zip(reference.collect_params())
        {
            assert!((*x - y).abs() < 1e-5);
        }
        assert_eq!(adam_acc.t, 1);
        assert_eq!(adam_ref.t, 1);
    }

    #[test]
    fn reuse_workspace_matches_reference_step_exactly() {
        let mut reference = model(101);
        let mut candidate = reference.clone();
        let tokens: Vec<usize> = (0..128).map(|i| (i * 5 + 2) % 7).collect();
        let cfg = TrainConfig {
            seed: 4242,
            batch_size: 3,
            gradient_accumulation_steps: 2,
            grad_clip_norm: 1.0,
        };
        let n = reference.collect_params().len();
        let mut reference_adam = Adam::new(n, 2e-3);
        let mut candidate_adam = Adam::new(n, 2e-3);
        let mut reference_grads = vec![0.0; n];
        let mut candidate_grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&candidate);

        let a = train_step(
            &mut reference,
            &mut reference_adam,
            &tokens,
            cfg,
            6,
            &mut reference_grads,
        )
        .unwrap();
        let b = train_step_reuse(
            &mut candidate,
            &mut candidate_adam,
            &tokens,
            cfg,
            6,
            &mut candidate_grads,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(b, a);
        assert_eq!(candidate_grads, reference_grads);
        assert_eq!(candidate.collect_params(), reference.collect_params());
        let (_, at, am, av) = reference_adam.export();
        let (_, bt, bm, bv) = candidate_adam.export();
        assert_eq!(bt, at);
        assert_eq!(bm, am);
        assert_eq!(bv, av);
    }

    #[test]
    fn reuse_workspace_stays_exact_across_multiple_steps() {
        let mut reference = model(202);
        let mut candidate = reference.clone();
        let tokens: Vec<usize> = (0..160).map(|i| (i * 3 + 4) % 7).collect();
        let cfg = TrainConfig {
            seed: 991,
            batch_size: 2,
            gradient_accumulation_steps: 2,
            grad_clip_norm: 1.0,
        };
        let n = reference.collect_params().len();
        let mut reference_adam = Adam::new(n, 2e-3);
        let mut candidate_adam = Adam::new(n, 2e-3);
        let mut reference_grads = vec![0.0; n];
        let mut candidate_grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&candidate);

        for step in 0..3 {
            let a = train_step(
                &mut reference,
                &mut reference_adam,
                &tokens,
                cfg,
                step,
                &mut reference_grads,
            )
            .unwrap();
            let b = train_step_reuse(
                &mut candidate,
                &mut candidate_adam,
                &tokens,
                cfg,
                step,
                &mut candidate_grads,
                &mut workspace,
            )
            .unwrap();
            assert_eq!(b, a);
            assert_eq!(candidate_grads, reference_grads);
            assert_eq!(candidate.collect_params(), reference.collect_params());
        }

        let (_, at, am, av) = reference_adam.export();
        let (_, bt, bm, bv) = candidate_adam.export();
        assert_eq!(bt, at);
        assert_eq!(bm, am);
        assert_eq!(bv, av);
    }

    #[test]
    fn zero_accumulation_is_rejected() {
        let bad = TrainConfig {
            gradient_accumulation_steps: 0,
            ..TrainConfig::default()
        };
        assert!(bad.validate().is_err());
    }
}
