//! Reproducible reference training loop primitives.
//!
//! Foundation keeps the simple allocation-heavy path as the semantic reference.
//! Engine adds explicitly reusable workspaces and must remain equivalent to it.

use crate::batch::{
    backward_batch_into, backward_deterministic_batch_from_stream_into,
    deterministic_batch_from_stream,
};
use crate::metrics::EngineStepTiming;
use crate::model::{BackwardWorkspace, Gpt};
use crate::numeric::{explain, Diagnostics, Scan, Stage};
use crate::numeric_state::{fault_context, summarize_training_state, TrainingStateSummary};
use crate::optim::{Adam, Optimizer};
use std::time::Instant;

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

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingDiagnosticsReport {
    pub loss: Scan,
    pub pre_optimizer: TrainingStateSummary,
    pub post_optimizer: TrainingStateSummary,
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
    optimizer: &mut dyn Optimizer,
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
        optimizer,
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
/// optimizer updates are identical to `train_step`. The only difference is memory
/// behavior: deterministic windows are borrowed directly from the token stream,
/// flat buffers are reused, model-structured gradients are reset in place, and
/// the optimizer updates a persistent flat parameter mirror.
pub fn train_step_reuse(
    gpt: &mut Gpt,
    optimizer: &mut dyn Optimizer,
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
        optimizer,
        cfg,
        global_step,
        effective_batch_size,
        grads,
        loss_sum,
        &mut workspace.params,
    )
}

pub fn train_step_reuse_timing(
    gpt: &mut Gpt,
    optimizer: &mut dyn Optimizer,
    train_tokens: &[usize],
    cfg: TrainConfig,
    global_step: u64,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
    enabled: bool,
) -> Result<(StepMetrics, Option<EngineStepTiming>), &'static str> {
    if !enabled {
        return train_step_reuse(
            gpt, adam, train_tokens, cfg, global_step, grads, workspace,
        )
        .map(|metrics| (metrics, None));
    }
    train_step_reuse_timed(
        gpt, adam, train_tokens, cfg, global_step, grads, workspace,
    )
    .map(|(metrics, timing)| (metrics, Some(timing)))
}
fn current_rss_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                return rest.split_whitespace().find_map(|part| part.parse().ok());
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Opt-in Engine step timing. The production timing-off path remains
/// train_step_reuse and is intentionally unchanged.
pub fn train_step_reuse_timed(
    gpt: &mut Gpt,
    optimizer: &mut dyn Optimizer,
    train_tokens: &[usize],
    cfg: TrainConfig,
    global_step: u64,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> Result<(StepMetrics, EngineStepTiming), &'static str> {
    cfg.validate()?;
    let effective_batch_size = cfg.effective_batch_size()?;
    if !workspace.matches(gpt, grads.len()) {
        return Err("training workspace has wrong size or model config");
    }

    let total_start = Instant::now();
    grads.fill(0.0);
    let mut loss_sum = 0.0f32;

    let backward_start = Instant::now();
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
    let backward_accum_ns = backward_start.elapsed().as_nanos() as u64;

    let grad_start = Instant::now();
    let (loss, grad_norm, grad_scale) = prepare_grads(cfg, grads, loss_sum)?;
    let grad_process_ns = grad_start.elapsed().as_nanos() as u64;

    let optimizer_start = Instant::now();
    optimizer.update(&mut workspace.params, grads)?;
    if workspace.params.iter().any(|x| !x.is_finite()) {
        return Err("optimizer produced non-finite parameters");
    }
    let optimizer_ns = optimizer_start.elapsed().as_nanos() as u64;

    let writeback_start = Instant::now();
    gpt.write_params(&workspace.params);
    let writeback_ns = writeback_start.elapsed().as_nanos() as u64;

    let metrics = make_metrics(
        gpt,
        cfg,
        global_step,
        effective_batch_size,
        loss,
        grad_norm,
        grad_scale,
    )?;
    let total_ns = total_start.elapsed().as_nanos() as u64;

    Ok((
        metrics,
        EngineStepTiming {
            schema_version: EngineStepTiming::SCHEMA_VERSION,
            global_step,
            tokens: metrics.tokens,
            total_ns,
            backward_accum_ns,
            grad_process_ns,
            optimizer_ns,
            writeback_ns,
            data_ns: None,
            checkpoint_ns: None,
            rss_kib: current_rss_kib(),
            alloc_calls: None,
            alloc_bytes: None,
        },
    ))
}
/// Opt-in Engine training step with numerical diagnostics.
///
/// The existing `train_step_reuse` remains the production diagnostics-off path
/// unchanged. Passing `Diagnostics::off()` here delegates directly to that
/// path. Enabled mode checks loss first, then gradients/parameters/optimizer state
/// before and after the optimizer update, and returns contextual first-fault
/// errors without changing arithmetic on clean runs.
pub fn train_step_reuse_diagnostics(
    gpt: &mut Gpt,
    optimizer: &mut dyn Optimizer,
    train_tokens: &[usize],
    cfg: TrainConfig,
    global_step: u64,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
    diagnostics: Diagnostics,
) -> Result<(StepMetrics, Option<TrainingDiagnosticsReport>), String> {
    if !diagnostics.enabled {
        let metrics = train_step_reuse(
            gpt,
            optimizer,
            train_tokens,
            cfg,
            global_step,
            grads,
            workspace,
        )
        .map_err(str::to_string)?;
        return Ok((metrics, None));
    }

    cfg.validate().map_err(str::to_string)?;
    let effective_batch_size = cfg.effective_batch_size().map_err(str::to_string)?;
    if !workspace.matches(gpt, grads.len()) {
        return Err("training workspace has wrong size or model config".into());
    }

    grads.fill(0.0);
    let mut loss_sum = 0.0f32;
    for micro in 0..cfg.gradient_accumulation_steps {
        let stream_offset = (micro as u64)
            .checked_mul(cfg.batch_size as u64)
            .ok_or_else(|| "batch stream overflow".to_string())?;
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
        )
        .map_err(str::to_string)?;
        for (dst, src) in grads.iter_mut().zip(&workspace.micro_grads) {
            *dst += *src;
        }
    }

    let inv_accum = 1.0 / cfg.gradient_accumulation_steps as f32;
    let raw_loss = loss_sum * inv_accum;
    let loss_values = [raw_loss];
    let loss_scan = diagnostics
        .scan(&loss_values)
        .expect("enabled diagnostics must return a scan");
    if !loss_scan.is_finite() {
        return Err(explain(Stage::Loss, "loss", &loss_scan));
    }

    let optimizer_before = optimizer.diagnostics();
    let pre_optimizer = summarize_training_state(
        diagnostics,
        grads,
        &workspace.params,
        optimizer_before.first,
        optimizer_before.second,
    )
    .expect("enabled diagnostics must summarize state");
    if let Some(context) = fault_context(&pre_optimizer) {
        return Err(context);
    }

    let (loss, grad_norm, grad_scale) =
        prepare_grads(cfg, grads, loss_sum).map_err(str::to_string)?;

    optimizer.update(&mut workspace.params, grads)?;

    let (_, _, adam_m_after, adam_v_after) = adam.export();
    let post_optimizer = summarize_training_state(
        diagnostics,
        grads,
        &workspace.params,
        adam_m_after,
        adam_v_after,
    )
    .expect("enabled diagnostics must summarize state");
    if let Some(context) = fault_context(&post_optimizer) {
        return Err(context);
    }

    gpt.write_params(&workspace.params);
    let metrics = make_metrics(
        gpt,
        cfg,
        global_step,
        effective_batch_size,
        loss,
        grad_norm,
        grad_scale,
    )
    .map_err(str::to_string)?;

    Ok((
        metrics,
        Some(TrainingDiagnosticsReport {
            loss: loss_scan,
            pre_optimizer,
            post_optimizer,
        }),
    ))
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
    optimizer: &mut dyn Optimizer,
    cfg: TrainConfig,
    global_step: u64,
    effective_batch_size: usize,
    grads: &mut [f32],
    loss_sum: f32,
) -> Result<StepMetrics, &'static str> {
    let (loss, grad_norm, grad_scale) = prepare_grads(cfg, grads, loss_sum)?;

    let mut params = gpt.collect_params();
    optimizer.update(&mut params, grads)?;
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
    optimizer: &mut dyn Optimizer,
    cfg: TrainConfig,
    global_step: u64,
    effective_batch_size: usize,
    grads: &mut [f32],
    loss_sum: f32,
    params: &mut [f32],
) -> Result<StepMetrics, &'static str> {
    let (loss, grad_norm, grad_scale) = prepare_grads(cfg, grads, loss_sum)?;

    optimizer.update(params, grads)?;
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
    fn timing_off_wrapper_delegates_to_exact_reuse_path() {
        let mut reference = model(811);
        let mut observed = reference.clone();
        let tokens: Vec<usize> = (0..96).map(|i| (i * 3 + 2) % 7).collect();
        let cfg = cfg(812, 2);
        let n = reference.collect_params().len();
        let mut adam_reference = Adam::new(n, 1e-3);
        let mut adam_observed = Adam::new(n, 1e-3);
        let mut grads_reference = vec![0.0; n];
        let mut grads_observed = vec![0.0; n];
        let mut workspace_reference = TrainWorkspace::new(&reference);
        let mut workspace_observed = TrainWorkspace::new(&observed);

        let metrics_reference = train_step_reuse(
            &mut reference, &mut adam_reference, &tokens, cfg, 0,
            &mut grads_reference, &mut workspace_reference,
        ).unwrap();
        let (metrics_observed, timing) = train_step_reuse_timing(
            &mut observed, &mut adam_observed, &tokens, cfg, 0,
            &mut grads_observed, &mut workspace_observed, false,
        ).unwrap();

        assert!(timing.is_none());
        assert_eq!(metrics_observed, metrics_reference);
        assert_eq!(grads_observed, grads_reference);
        assert_eq!(observed.collect_params(), reference.collect_params());
        assert_eq!(adam_observed.export().1, adam_reference.export().1);
        assert_eq!(adam_observed.export().2, adam_reference.export().2);
        assert_eq!(adam_observed.export().3, adam_reference.export().3);
    }
    #[test]
    fn timed_path_preserves_clean_training_state_exactly() {
        let mut reference = model(912);
        let mut observed = reference.clone();
        let tokens: Vec<usize> = (0..96).map(|i| (i * 5 + 1) % 7).collect();
        let cfg = cfg(913, 2);
        let n = reference.collect_params().len();
        let mut adam_reference = Adam::new(n, 1e-3);
        let mut adam_observed = Adam::new(n, 1e-3);
        let mut grads_reference = vec![0.0; n];
        let mut grads_observed = vec![0.0; n];
        let mut workspace_reference = TrainWorkspace::new(&reference);
        let mut workspace_observed = TrainWorkspace::new(&observed);

        let metrics_reference = train_step_reuse(
            &mut reference,
            &mut adam_reference,
            &tokens,
            cfg,
            0,
            &mut grads_reference,
            &mut workspace_reference,
        )
        .unwrap();
        let (metrics_observed, timing) = train_step_reuse_timed(
            &mut observed,
            &mut adam_observed,
            &tokens,
            cfg,
            0,
            &mut grads_observed,
            &mut workspace_observed,
        )
        .unwrap();

        assert_eq!(metrics_observed, metrics_reference);
        assert_eq!(grads_observed, grads_reference);
        assert_eq!(observed.collect_params(), reference.collect_params());
        assert_eq!(adam_observed.export().1, adam_reference.export().1);
        assert_eq!(adam_observed.export().2, adam_reference.export().2);
        assert_eq!(adam_observed.export().3, adam_reference.export().3);
        assert_eq!(timing.schema_version, EngineStepTiming::SCHEMA_VERSION);
        assert_eq!(timing.global_step, 0);
        assert_eq!(timing.tokens, metrics_observed.tokens);
        assert!(timing.known_phase_ns() <= timing.total_ns);
        assert!(timing.data_ns.is_none());
        assert!(timing.checkpoint_ns.is_none());
        assert!(timing.alloc_calls.is_none());
        assert!(timing.alloc_bytes.is_none());
    }
    #[test]
    fn diagnostics_off_delegates_to_exact_reuse_path() {
        let mut reference = model(123);
        let mut observed = reference.clone();
        let tokens: Vec<usize> = (0..96).map(|i| (i * 3 + 1) % 7).collect();
        let cfg = cfg(456, 2);
        let n = reference.collect_params().len();
        let mut adam_reference = Adam::new(n, 1e-3);
        let mut adam_observed = Adam::new(n, 1e-3);
        let mut grads_reference = vec![0.0; n];
        let mut grads_observed = vec![0.0; n];
        let mut workspace_reference = TrainWorkspace::new(&reference);
        let mut workspace_observed = TrainWorkspace::new(&observed);

        let metrics_reference = train_step_reuse(
            &mut reference,
            &mut adam_reference,
            &tokens,
            cfg,
            0,
            &mut grads_reference,
            &mut workspace_reference,
        )
        .unwrap();
        let (metrics_observed, report) = train_step_reuse_diagnostics(
            &mut observed,
            &mut adam_observed,
            &tokens,
            cfg,
            0,
            &mut grads_observed,
            &mut workspace_observed,
            Diagnostics::off(),
        )
        .unwrap();

        assert!(report.is_none());
        assert_eq!(metrics_observed, metrics_reference);
        assert_eq!(grads_observed, grads_reference);
        assert_eq!(observed.collect_params(), reference.collect_params());
        assert_eq!(adam_observed.export().1, adam_reference.export().1);
        assert_eq!(adam_observed.export().2, adam_reference.export().2);
        assert_eq!(adam_observed.export().3, adam_reference.export().3);
    }

    #[test]
    fn diagnostics_on_preserves_clean_training_state_exactly() {
        let mut reference = model(321);
        let mut observed = reference.clone();
        let tokens: Vec<usize> = (0..96).map(|i| (i * 5 + 2) % 7).collect();
        let cfg = cfg(654, 2);
        let n = reference.collect_params().len();
        let mut adam_reference = Adam::new(n, 1e-3);
        let mut adam_observed = Adam::new(n, 1e-3);
        let mut grads_reference = vec![0.0; n];
        let mut grads_observed = vec![0.0; n];
        let mut workspace_reference = TrainWorkspace::new(&reference);
        let mut workspace_observed = TrainWorkspace::new(&observed);

        let metrics_reference = train_step_reuse(
            &mut reference,
            &mut adam_reference,
            &tokens,
            cfg,
            0,
            &mut grads_reference,
            &mut workspace_reference,
        )
        .unwrap();
        let (metrics_observed, report) = train_step_reuse_diagnostics(
            &mut observed,
            &mut adam_observed,
            &tokens,
            cfg,
            0,
            &mut grads_observed,
            &mut workspace_observed,
            Diagnostics::on(),
        )
        .unwrap();

        let report = report.expect("diagnostics enabled");
        assert!(report.loss.is_finite());
        assert!(report.pre_optimizer.first_fault().is_none());
        assert!(report.post_optimizer.first_fault().is_none());
        assert_eq!(metrics_observed, metrics_reference);
        assert_eq!(grads_observed, grads_reference);
        assert_eq!(observed.collect_params(), reference.collect_params());
        assert_eq!(adam_observed.export().1, adam_reference.export().1);
        assert_eq!(adam_observed.export().2, adam_reference.export().2);
        assert_eq!(adam_observed.export().3, adam_reference.export().3);
    }

    #[test]
    fn diagnostics_reject_corrupt_adam_state_before_optimizer() {
        let mut gpt = model(777);
        let tokens: Vec<usize> = (0..96).map(|i| (i * 2 + 3) % 7).collect();
        let cfg = cfg(999, 2);
        let n = gpt.collect_params().len();
        let mut m = vec![0.0; n];
        m[3] = f32::NAN;
        let mut adam = Adam::from_state(1e-3, 0, m, vec![0.0; n]);
        let mut grads = vec![0.0; n];
        let mut workspace = TrainWorkspace::new(&gpt);

        let error = train_step_reuse_diagnostics(
            &mut gpt,
            &mut adam,
            &tokens,
            cfg,
            0,
            &mut grads,
            &mut workspace,
            Diagnostics::on(),
        )
        .unwrap_err();

        assert!(error.contains("adam_m"));
        assert!(error.contains("NaN"));
        assert_eq!(
            adam.global_step(),
            0,
            "pre-optimizer fault must stop before optimizer update"
        );
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
