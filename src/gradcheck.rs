//! Finite differences versus the explicit analytical backward pass.
//!
//! With f32, relative error can look large when both gradients are extremely
//! close to zero. We therefore require a small absolute error and a bounded
//! mean relative error instead of judging correctness from one near-zero
//! outlier.

use crate::model::{Config, Gpt};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct GradCheckReport {
    pub checked: usize,
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    pub mean_rel_err: f32,
}

impl GradCheckReport {
    pub fn ok(&self, abs_tol: f32, mean_rel_tol: f32) -> bool {
        self.checked > 0
            && self.max_abs_err.is_finite()
            && self.mean_rel_err.is_finite()
            && self.max_abs_err <= abs_tol
            && self.mean_rel_err <= mean_rel_tol
    }
}

pub fn check_random_params(
    gpt: &Gpt,
    x: &[usize],
    y: &[usize],
    n_params: usize,
    eps: f32,
    rng: &mut impl Rng,
) -> GradCheckReport {
    let mut params = gpt.collect_params();
    let mut grads = vec![0.0; params.len()];
    let _loss = gpt.backward_into(x, y, &mut grads);
    let n = params.len().min(n_params.max(1));
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_rel = 0.0f32;

    for _ in 0..n {
        let i = rng.gen_range(0..params.len());
        let saved = params[i];

        params[i] = saved + eps;
        let lp = loss_only(gpt, &params, x, y);

        params[i] = saved - eps;
        let lm = loss_only(gpt, &params, x, y);

        params[i] = saved;
        let num = (lp - lm) / (2.0 * eps);
        let ana = grads[i];
        let abs = (num - ana).abs();
        let rel = abs / (ana.abs() + num.abs() + 1e-6);

        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }

    GradCheckReport {
        checked: n,
        max_abs_err: max_abs,
        max_rel_err: max_rel,
        mean_rel_err: sum_rel / n as f32,
    }
}

fn loss_only(template: &Gpt, params: &[f32], x: &[usize], y: &[usize]) -> f32 {
    let mut clone = {
        let mut rng = rand::thread_rng();
        Gpt::new(template.cfg, &mut rng)
    };
    clone.write_params(params);
    let mut g = vec![0.0; params.len()];
    clone.backward_into(x, y, &mut g)
}

pub fn tiny_check_config(vocab: usize) -> Config {
    Config {
        vocab,
        n_embd: 16,
        n_head: 2,
        n_layer: 1,
        block: 8,
        n_ff: 32,
    }
}
