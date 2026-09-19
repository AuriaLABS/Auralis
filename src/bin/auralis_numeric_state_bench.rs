use auralis::numeric::{l2_norm, Diagnostics, Stage};
use auralis::numeric_state::{
    summarize_training_state, GradientSummary, StateSliceSummary, TrainingStateSummary,
};
use std::hint::black_box;
use std::time::Instant;

struct StateData {
    gradients: Vec<f32>,
    parameters: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
}

fn data(n: usize) -> StateData {
    StateData {
        gradients: (0..n)
            .map(|i| ((i % 257) as f32 - 128.0) / 257.0)
            .collect(),
        parameters: (0..n)
            .map(|i| ((i % 193) as f32 - 96.0) / 193.0)
            .collect(),
        adam_m: (0..n).map(|i| (i % 17) as f32 * 1e-4).collect(),
        adam_v: (0..n).map(|i| (i % 23) as f32 * 1e-5).collect(),
    }
}

fn legacy_summarize_training_state(
    diagnostics: Diagnostics,
    gradients: &[f32],
    parameters: &[f32],
    adam_m: &[f32],
    adam_v: &[f32],
) -> Option<TrainingStateSummary> {
    if !diagnostics.enabled {
        return None;
    }

    let grad_scan = diagnostics.scan(gradients).expect("enabled diagnostics");
    let parameter_scan = diagnostics.scan(parameters).expect("enabled diagnostics");
    let adam_m_scan = diagnostics.scan(adam_m).expect("enabled diagnostics");
    let adam_v_scan = diagnostics.scan(adam_v).expect("enabled diagnostics");

    let gradient = GradientSummary {
        len: gradients.len(),
        l2: l2_norm(gradients),
        max_abs: grad_scan.max_abs,
        finite: grad_scan.is_finite(),
    };

    Some(TrainingStateSummary {
        gradient,
        slices: vec![
            StateSliceSummary {
                stage: Stage::Gradient,
                name: "gradients",
                scan: grad_scan,
            },
            StateSliceSummary {
                stage: Stage::Parameter,
                name: "parameters",
                scan: parameter_scan,
            },
            StateSliceSummary {
                stage: Stage::AdamMoment,
                name: "adam_m",
                scan: adam_m_scan,
            },
            StateSliceSummary {
                stage: Stage::AdamMoment,
                name: "adam_v",
                scan: adam_v_scan,
            },
        ],
    })
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn run_current(enabled: bool, iters: usize, repeats: usize, state: &StateData) -> f64 {
    let diagnostics = Diagnostics { enabled };
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(summarize_training_state(
                black_box(diagnostics),
                black_box(&state.gradients),
                black_box(&state.parameters),
                black_box(&state.adam_m),
                black_box(&state.adam_v),
            ));
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iters as f64);
    }
    median(samples)
}

fn run_legacy(iters: usize, repeats: usize, state: &StateData) -> f64 {
    let diagnostics = Diagnostics::on();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(legacy_summarize_training_state(
                black_box(diagnostics),
                black_box(&state.gradients),
                black_box(&state.parameters),
                black_box(&state.adam_m),
                black_box(&state.adam_v),
            ));
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iters as f64);
    }
    median(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000usize);
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    let n = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(28_580usize);
    let state = data(n);

    let legacy = legacy_summarize_training_state(
        Diagnostics::on(),
        &state.gradients,
        &state.parameters,
        &state.adam_m,
        &state.adam_v,
    );
    let current = summarize_training_state(
        Diagnostics::on(),
        &state.gradients,
        &state.parameters,
        &state.adam_m,
        &state.adam_v,
    );
    assert_eq!(legacy, current, "legacy/current summary mismatch");

    let off_ns = run_current(false, iters, repeats, &state);
    let legacy_on_ns = run_legacy(iters, repeats, &state);
    let current_on_ns = run_current(true, iters, repeats, &state);

    println!(
        "numeric_state_bench | len={} iters={} repeats={} off_ns={:.3} legacy_on_ns={:.3} current_on_ns={:.3} current_over_legacy={:.4} speedup={:.4}",
        n,
        iters,
        repeats,
        off_ns,
        legacy_on_ns,
        current_on_ns,
        current_on_ns / legacy_on_ns,
        legacy_on_ns / current_on_ns
    );
}
