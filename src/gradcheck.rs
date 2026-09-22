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
    /// Backwards-compatible convenience criterion used by the CLI.
    pub fn ok(&self, mean_rel_tol: f32) -> bool {
        self.ok_with(5e-4, mean_rel_tol)
    }

    pub fn ok_with(&self, abs_tol: f32, mean_rel_tol: f32) -> bool {
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
        Gpt::new_with_attention_policy(
            template.cfg,
            template.normalization(),
            template.position_kind(),
            template.n_kv_head(),
            template.attention_window(),
            &mut rng,
        )
    };
    clone.write_params(params);
    clone.loss(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NormalizationKind;
    use crate::position::PositionKind;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn local_attention_qkv_gradients_match_finite_differences() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 4,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let x = [0usize, 1, 2, 3];
        let y = [1usize, 2, 3, 4];
        let eps = 1e-3f32;
        let n_kv_head = 2usize;
        let mut rng = StdRng::seed_from_u64(0xA11CE_1112);
        let gpt = Gpt::new_with_attention_policy(
            cfg,
            NormalizationKind::LayerNorm,
            PositionKind::LearnedAbsolute,
            n_kv_head,
            2,
            &mut rng,
        );
        let mut params = gpt.collect_params();
        let mut grads = vec![0.0f32; params.len()];
        let loss = gpt.backward_into(&x, &y, &mut grads);
        assert!(loss.is_finite());
        assert!(grads.iter().all(|g| g.is_finite()));

        let d = cfg.n_embd;
        let kv_width = (d / cfg.n_head) * n_kv_head;
        let block_start = cfg.vocab * d + cfg.block * d;
        let wq_start = block_start + 2 * d;
        let wk_start = wq_start + d * d;
        let wv_start = wk_start + d * kv_width;
        let indices = [
            wq_start,
            wq_start + d * d - 1,
            wk_start,
            wk_start + d * kv_width - 1,
            wv_start,
            wv_start + d * kv_width - 1,
        ];

        for index in indices {
            let saved = params[index];
            params[index] = saved + eps;
            let lp = loss_only(&gpt, &params, &x, &y);
            params[index] = saved - eps;
            let lm = loss_only(&gpt, &params, &x, &y);
            params[index] = saved;

            let numerical = (lp - lm) / (2.0 * eps);
            let analytical = grads[index];
            let abs_err = (numerical - analytical).abs();
            assert!(
                abs_err <= 4e-3,
                "index={index} analytical={analytical} numerical={numerical} abs_err={abs_err}"
            );
        }
    }

    #[test]
    fn grouped_kv_projection_gradients_match_finite_differences() {
        let cfg = Config {
            vocab: 7,
            n_embd: 8,
            n_head: 4,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let x = [0usize, 1, 2, 3];
        let y = [1usize, 2, 3, 4];
        let eps = 1e-3f32;

        for n_kv_head in [1usize, 2] {
            let mut rng = StdRng::seed_from_u64(0xA11CE_1403 + n_kv_head as u64);
            let gpt = Gpt::new_with_attention_heads(
                cfg,
                NormalizationKind::LayerNorm,
                PositionKind::LearnedAbsolute,
                n_kv_head,
                &mut rng,
            );
            let mut params = gpt.collect_params();
            let mut grads = vec![0.0f32; params.len()];
            let loss = gpt.backward_into(&x, &y, &mut grads);
            assert!(loss.is_finite());
            assert!(grads.iter().all(|g| g.is_finite()));

            let d = cfg.n_embd;
            let kv_width = (d / cfg.n_head) * n_kv_head;
            let block_start = cfg.vocab * d + cfg.block * d;
            let wk_start = block_start + 2 * d + d * d;
            let wk_len = d * kv_width;
            let wv_start = wk_start + wk_len;
            let indices = [
                wk_start,
                wk_start + wk_len - 1,
                wv_start,
                wv_start + wk_len - 1,
            ];

            for index in indices {
                let saved = params[index];
                params[index] = saved + eps;
                let lp = loss_only(&gpt, &params, &x, &y);
                params[index] = saved - eps;
                let lm = loss_only(&gpt, &params, &x, &y);
                params[index] = saved;

                let numerical = (lp - lm) / (2.0 * eps);
                let analytical = grads[index];
                let abs_err = (numerical - analytical).abs();
                assert!(
                    abs_err <= 3e-3,
                    "n_kv_head={n_kv_head} index={index} analytical={analytical} numerical={numerical} abs_err={abs_err}"
                );
            }
        }
    }
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
