use std::error::Error;
use std::fmt;

pub const MOE_ROUTER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowPolicy {
    Reject,
    DenseFallback,
}

impl OverflowPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reject => "reject",
            Self::DenseFallback => "dense-fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRouterConfig {
    pub experts: usize,
    pub top_k: usize,
    pub capacity: usize,
    pub overflow_policy: OverflowPolicy,
}

impl MoeRouterConfig {
    pub fn validate(self) -> Result<Self, MoeError> {
        if self.experts == 0 {
            return Err(MoeError::InvalidConfig("experts must be > 0"));
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err(MoeError::InvalidConfig("top_k must be in 1..=experts"));
        }
        if self.capacity == 0 {
            return Err(MoeError::InvalidConfig("capacity must be > 0"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertAssignment {
    pub expert: usize,
    pub slot: usize,
    pub gate: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenRoute {
    pub token: usize,
    pub assignments: Vec<ExpertAssignment>,
    pub dense_fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoutePlan {
    pub schema_version: u32,
    pub config: MoeRouterConfig,
    pub routes: Vec<TokenRoute>,
    pub expert_loads: Vec<usize>,
}

impl RoutePlan {
    pub fn token_count(&self) -> usize {
        self.routes.len()
    }

    pub fn fallback_tokens(&self) -> usize {
        self.routes.iter().filter(|r| r.dense_fallback).count()
    }

    pub fn assignment_count(&self) -> usize {
        self.routes.iter().map(|r| r.assignments.len()).sum()
    }

    pub fn metrics(&self) -> MoeMetrics {
        let tokens = self.token_count();
        let fallback_tokens = self.fallback_tokens();
        let routed_tokens = tokens - fallback_tokens;
        let assignments = self.assignment_count();
        let empty_experts = self.expert_loads.iter().filter(|&&n| n == 0).count();
        let max_load = self.expert_loads.iter().copied().max().unwrap_or(0);
        let min_load = self.expert_loads.iter().copied().min().unwrap_or(0);
        let mean_load = if self.expert_loads.is_empty() {
            0.0
        } else {
            assignments as f64 / self.expert_loads.len() as f64
        };
        let variance = if self.expert_loads.is_empty() {
            0.0
        } else {
            self.expert_loads
                .iter()
                .map(|&n| {
                    let d = n as f64 - mean_load;
                    d * d
                })
                .sum::<f64>()
                / self.expert_loads.len() as f64
        };
        let load_cv = if mean_load == 0.0 {
            0.0
        } else {
            variance.sqrt() / mean_load
        };
        let total_capacity = self
            .config
            .capacity
            .saturating_mul(self.config.experts);
        let capacity_utilization = if total_capacity == 0 {
            0.0
        } else {
            assignments as f64 / total_capacity as f64
        };
        MoeMetrics {
            tokens,
            routed_tokens,
            fallback_tokens,
            assignments,
            dropped_tokens: 0,
            expert_loads: self.expert_loads.clone(),
            empty_experts,
            min_load,
            max_load,
            mean_load,
            load_cv,
            capacity_utilization,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeMetrics {
    pub tokens: usize,
    pub routed_tokens: usize,
    pub fallback_tokens: usize,
    pub assignments: usize,
    pub dropped_tokens: usize,
    pub expert_loads: Vec<usize>,
    pub empty_experts: usize,
    pub min_load: usize,
    pub max_load: usize,
    pub mean_load: f64,
    pub load_cv: f64,
    pub capacity_utilization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertDispatch {
    pub token_indices: Vec<usize>,
    pub gates: Vec<f32>,
    pub activations: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DispatchBatch {
    pub width: usize,
    pub experts: Vec<ExpertDispatch>,
    pub dense_fallback_tokens: Vec<usize>,
}

pub trait MoeRouter {
    fn route(
        &self,
        logits: &[f32],
        tokens: usize,
        config: MoeRouterConfig,
    ) -> Result<RoutePlan, MoeError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeterministicTopKRouter;

impl MoeRouter for DeterministicTopKRouter {
    fn route(
        &self,
        logits: &[f32],
        tokens: usize,
        config: MoeRouterConfig,
    ) -> Result<RoutePlan, MoeError> {
        let config = config.validate()?;
        let expected = tokens
            .checked_mul(config.experts)
            .ok_or(MoeError::SizeOverflow)?;
        if logits.len() != expected {
            return Err(MoeError::LengthMismatch {
                what: "router logits",
                expected,
                actual: logits.len(),
            });
        }

        let mut loads = vec![0usize; config.experts];
        let mut routes = Vec::with_capacity(tokens);
        let mut ranked = Vec::with_capacity(config.experts);

        for token in 0..tokens {
            ranked.clear();
            let row = &logits[token * config.experts..(token + 1) * config.experts];
            for (expert, &score) in row.iter().enumerate() {
                if !score.is_finite() {
                    return Err(MoeError::NonFiniteLogit { token, expert });
                }
                ranked.push((expert, score));
            }
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let selected = &ranked[..config.top_k];

            let overflow = selected
                .iter()
                .any(|(expert, _)| loads[*expert] >= config.capacity);
            if overflow {
                match config.overflow_policy {
                    OverflowPolicy::Reject => {
                        let expert = selected
                            .iter()
                            .find(|(expert, _)| loads[*expert] >= config.capacity)
                            .map(|(expert, _)| *expert)
                            .expect("overflow expert exists");
                        return Err(MoeError::CapacityExceeded {
                            token,
                            expert,
                            capacity: config.capacity,
                        });
                    }
                    OverflowPolicy::DenseFallback => {
                        routes.push(TokenRoute {
                            token,
                            assignments: Vec::new(),
                            dense_fallback: true,
                        });
                        continue;
                    }
                }
            }

            let max_score = selected
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            let mut weights = Vec::with_capacity(config.top_k);
            for &(_, score) in selected {
                let weight = (score - max_score).exp();
                denominator += weight;
                weights.push(weight);
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(MoeError::InvalidGateNormalization { token });
            }

            let mut assignments = Vec::with_capacity(config.top_k);
            for ((expert, _), weight) in selected.iter().zip(weights) {
                let slot = loads[*expert];
                loads[*expert] += 1;
                assignments.push(ExpertAssignment {
                    expert: *expert,
                    slot,
                    gate: weight / denominator,
                });
            }
            routes.push(TokenRoute {
                token,
                assignments,
                dense_fallback: false,
            });
        }

        Ok(RoutePlan {
            schema_version: MOE_ROUTER_SCHEMA_VERSION,
            config,
            routes,
            expert_loads: loads,
        })
    }
}

impl DeterministicTopKRouter {
    pub fn dispatch(
        &self,
        plan: &RoutePlan,
        activations: &[f32],
        width: usize,
    ) -> Result<DispatchBatch, MoeError> {
        if width == 0 {
            return Err(MoeError::InvalidWidth);
        }
        let expected = plan
            .token_count()
            .checked_mul(width)
            .ok_or(MoeError::SizeOverflow)?;
        if activations.len() != expected {
            return Err(MoeError::LengthMismatch {
                what: "activations",
                expected,
                actual: activations.len(),
            });
        }

        let mut experts = plan
            .expert_loads
            .iter()
            .map(|&load| ExpertDispatch {
                token_indices: Vec::with_capacity(load),
                gates: Vec::with_capacity(load),
                activations: Vec::with_capacity(load.saturating_mul(width)),
            })
            .collect::<Vec<_>>();
        let mut dense_fallback_tokens = Vec::with_capacity(plan.fallback_tokens());

        for route in &plan.routes {
            if route.dense_fallback {
                dense_fallback_tokens.push(route.token);
                continue;
            }

            let row = &activations[route.token * width..(route.token + 1) * width];
            for assignment in &route.assignments {
                let expert = experts
                    .get_mut(assignment.expert)
                    .ok_or(MoeError::PlanInvariant("assignment expert out of range"))?;
                if expert.token_indices.len() != assignment.slot {
                    return Err(MoeError::PlanInvariant(
                        "assignment slot is not contiguous",
                    ));
                }
                expert.token_indices.push(route.token);
                expert.gates.push(assignment.gate);
                expert.activations.extend_from_slice(row);
            }
        }

        Ok(DispatchBatch {
            width,
            experts,
            dense_fallback_tokens,
        })
    }

    pub fn gather(
        &self,
        plan: &RoutePlan,
        expert_outputs: &[Vec<f32>],
        dense_fallback: Option<&[f32]>,
        width: usize,
    ) -> Result<Vec<f32>, MoeError> {
        if width == 0 {
            return Err(MoeError::InvalidWidth);
        }
        if expert_outputs.len() != plan.config.experts {
            return Err(MoeError::LengthMismatch {
                what: "expert output groups",
                expected: plan.config.experts,
                actual: expert_outputs.len(),
            });
        }
        for (expert, (&load, output)) in plan.expert_loads.iter().zip(expert_outputs).enumerate() {
            let expected = load.checked_mul(width).ok_or(MoeError::SizeOverflow)?;
            if output.len() != expected {
                return Err(MoeError::ExpertOutputLength {
                    expert,
                    expected,
                    actual: output.len(),
                });
            }
        }

        let total = plan
            .token_count()
            .checked_mul(width)
            .ok_or(MoeError::SizeOverflow)?;
        if plan.fallback_tokens() > 0 {
            let dense = dense_fallback.ok_or(MoeError::MissingDenseFallback)?;
            if dense.len() != total {
                return Err(MoeError::LengthMismatch {
                    what: "dense fallback",
                    expected: total,
                    actual: dense.len(),
                });
            }
        }

        let mut out = vec![0.0f32; total];
        for route in &plan.routes {
            let dst = &mut out[route.token * width..(route.token + 1) * width];
            if route.dense_fallback {
                let dense = dense_fallback.expect("validated above");
                dst.copy_from_slice(&dense[route.token * width..(route.token + 1) * width]);
                continue;
            }
            if route.assignments.is_empty() {
                return Err(MoeError::PlanInvariant(
                    "routed token has no assignments",
                ));
            }
            for assignment in &route.assignments {
                let start = assignment
                    .slot
                    .checked_mul(width)
                    .ok_or(MoeError::SizeOverflow)?;
                let src = &expert_outputs[assignment.expert][start..start + width];
                for (dst_value, &src_value) in dst.iter_mut().zip(src) {
                    *dst_value += assignment.gate * src_value;
                }
            }
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoeError {
    InvalidConfig(&'static str),
    InvalidWidth,
    SizeOverflow,
    LengthMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        token: usize,
        expert: usize,
    },
    InvalidGateNormalization {
        token: usize,
    },
    CapacityExceeded {
        token: usize,
        expert: usize,
        capacity: usize,
    },
    MissingDenseFallback,
    ExpertOutputLength {
        expert: usize,
        expected: usize,
        actual: usize,
    },
    PlanInvariant(&'static str),
}

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid MoE router config: {message}"),
            Self::InvalidWidth => write!(f, "MoE dispatch width must be > 0"),
            Self::SizeOverflow => write!(f, "MoE size arithmetic overflow"),
            Self::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteLogit { token, expert } => {
                write!(f, "non-finite router logit at token {token}, expert {expert}")
            }
            Self::InvalidGateNormalization { token } => {
                write!(f, "invalid router gate normalization at token {token}")
            }
            Self::CapacityExceeded {
                token,
                expert,
                capacity,
            } => write!(
                f,
                "expert capacity exceeded at token {token}, expert {expert}, capacity {capacity}"
            ),
            Self::MissingDenseFallback => write!(f, "dense fallback outputs are required"),
            Self::ExpertOutputLength {
                expert,
                expected,
                actual,
            } => write!(
                f,
                "expert {expert} output length mismatch: expected {expected}, got {actual}"
            ),
            Self::PlanInvariant(message) => write!(f, "invalid MoE route plan: {message}"),
        }
    }
}

impl Error for MoeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        experts: usize,
        top_k: usize,
        capacity: usize,
        overflow_policy: OverflowPolicy,
    ) -> MoeRouterConfig {
        MoeRouterConfig {
            experts,
            top_k,
            capacity,
            overflow_policy,
        }
    }

    #[test]
    fn config_validation_is_fail_closed() {
        for bad in [
            cfg(0, 1, 1, OverflowPolicy::Reject),
            cfg(2, 0, 1, OverflowPolicy::Reject),
            cfg(2, 3, 1, OverflowPolicy::Reject),
            cfg(2, 1, 0, OverflowPolicy::Reject),
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn routing_is_deterministic_and_ties_choose_lower_expert_index() {
        let router = DeterministicTopKRouter;
        let logits = [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, -1.0, -1.0];
        let config = cfg(4, 2, 4, OverflowPolicy::Reject);
        let a = router.route(&logits, 2, config).unwrap();
        let b = router.route(&logits, 2, config).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.routes[0]
                .assignments
                .iter()
                .map(|x| x.expert)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            a.routes[1]
                .assignments
                .iter()
                .map(|x| x.expert)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        for route in &a.routes {
            let sum: f32 = route.assignments.iter().map(|a| a.gate).sum();
            assert!((sum - 1.0).abs() <= 1e-6);
        }
    }

    #[test]
    fn reject_policy_reports_first_saturated_assignment() {
        let router = DeterministicTopKRouter;
        let logits = [3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0];
        let err = router
            .route(&logits, 2, cfg(4, 2, 1, OverflowPolicy::Reject))
            .unwrap_err();
        assert!(matches!(
            err,
            MoeError::CapacityExceeded {
                token: 1,
                expert: 0,
                capacity: 1
            }
        ));
    }

    #[test]
    fn dense_fallback_is_explicit_and_no_token_is_silently_dropped() {
        let router = DeterministicTopKRouter;
        let logits = [
            3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0,
        ];
        let plan = router
            .route(
                &logits,
                3,
                cfg(4, 2, 1, OverflowPolicy::DenseFallback),
            )
            .unwrap();
        assert_eq!(plan.routes.len(), 3);
        assert_eq!(plan.fallback_tokens(), 1);
        assert!(plan.routes[1].dense_fallback);
        assert!(plan.expert_loads.iter().all(|&n| n <= 1));
        let metrics = plan.metrics();
        assert_eq!(
            metrics.tokens,
            metrics.routed_tokens + metrics.fallback_tokens
        );
        assert_eq!(metrics.dropped_tokens, 0);
    }

    #[test]
    fn dispatch_and_identity_experts_gather_back_to_input_with_dense_fallback() {
        let router = DeterministicTopKRouter;
        let logits = [
            3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0,
        ];
        let plan = router
            .route(
                &logits,
                3,
                cfg(4, 2, 1, OverflowPolicy::DenseFallback),
            )
            .unwrap();
        let width = 3;
        let activations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dispatch = router.dispatch(&plan, &activations, width).unwrap();
        assert_eq!(dispatch.dense_fallback_tokens, vec![1]);
        let expert_outputs = dispatch
            .experts
            .iter()
            .map(|expert| expert.activations.clone())
            .collect::<Vec<_>>();
        let gathered = router
            .gather(&plan, &expert_outputs, Some(&activations), width)
            .unwrap();
        for (actual, expected) in gathered.iter().zip(activations) {
            assert!((*actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn metrics_report_empty_and_saturated_experts() {
        let router = DeterministicTopKRouter;
        let logits = [9.0, 8.0, 0.0, -1.0, 9.0, 8.0, 0.0, -1.0];
        let plan = router
            .route(&logits, 2, cfg(4, 1, 2, OverflowPolicy::Reject))
            .unwrap();
        let metrics = plan.metrics();
        assert_eq!(metrics.expert_loads, vec![2, 0, 0, 0]);
        assert_eq!(metrics.empty_experts, 3);
        assert_eq!(metrics.max_load, 2);
        assert_eq!(metrics.dropped_tokens, 0);
    }

    #[test]
    fn non_finite_logits_and_missing_fallback_fail_closed() {
        let router = DeterministicTopKRouter;
        assert!(matches!(
            router.route(
                &[0.0, f32::NAN],
                1,
                cfg(2, 1, 1, OverflowPolicy::Reject)
            ),
            Err(MoeError::NonFiniteLogit {
                token: 0,
                expert: 1
            })
        ));
        let plan = router
            .route(
                &[1.0, 0.0, 2.0, 1.0],
                2,
                cfg(2, 1, 1, OverflowPolicy::DenseFallback),
            )
            .unwrap();
        let expert_outputs = vec![vec![1.0], vec![]];
        assert!(matches!(
            router.gather(&plan, &expert_outputs, None, 1),
            Err(MoeError::MissingDenseFallback)
        ));
    }
}
