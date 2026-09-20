use auralis::layer_diagnostics::LayerHooks;
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

fn model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(659_918);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn batch() -> (Vec<usize>, Vec<usize>) {
    let x: Vec<usize> = (0..32).map(|i| (i * 37 + 11) % 100).collect();
    let y: Vec<usize> = (0..32).map(|i| (i * 37 + 12) % 100).collect();
    (x, y)
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn time_baseline(gpt: &Gpt, x: &[usize], y: &[usize], iterations: usize) -> u64 {
    let mut grads = vec![0.0; gpt.collect_params().len()];
    let started = Instant::now();
    for _ in 0..iterations {
        let loss = gpt.backward_into(x, y, &mut grads);
        std::hint::black_box((loss, &grads));
    }
    started.elapsed().as_nanos().max(1) as u64
}

fn time_hooks(
    gpt: &Gpt,
    x: &[usize],
    y: &[usize],
    iterations: usize,
    hooks: LayerHooks,
) -> u64 {
    let mut grads = vec![0.0; gpt.collect_params().len()];
    let started = Instant::now();
    for _ in 0..iterations {
        let (loss, report) = gpt.backward_with_layer_diagnostics(x, y, &mut grads, hooks);
        std::hint::black_box((loss, &grads, report));
    }
    started.elapsed().as_nanos().max(1) as u64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let repeats = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5usize);
    assert!(iterations > 0 && repeats >= 3);

    let gpt = model();
    let (x, y) = batch();
    let n = gpt.collect_params().len();

    let mut baseline_grads = vec![0.0; n];
    let mut off_grads = vec![0.0; n];
    let mut full_grads = vec![0.0; n];

    let baseline_loss = gpt.backward_into(&x, &y, &mut baseline_grads);
    let (off_loss, off_report) =
        gpt.backward_with_layer_diagnostics(&x, &y, &mut off_grads, LayerHooks::off());
    let (full_loss, full_report) =
        gpt.backward_with_layer_diagnostics(&x, &y, &mut full_grads, LayerHooks::full());

    assert_eq!(off_loss.to_bits(), baseline_loss.to_bits());
    assert_eq!(full_loss.to_bits(), baseline_loss.to_bits());
    assert_eq!(off_grads, baseline_grads);
    assert_eq!(full_grads, baseline_grads);
    assert!(off_report.is_none());
    let full_report = full_report.expect("full diagnostics report");
    println!("layer_diagnostics_json | {}", full_report.json());

    let mut baseline = Vec::with_capacity(repeats);
    let mut off = Vec::with_capacity(repeats);
    let mut full = Vec::with_capacity(repeats);

    for repetition in 0..repeats {
        let order = repetition % 3;
        let (b, o, f) = match order {
            0 => (
                time_baseline(&gpt, &x, &y, iterations),
                time_hooks(&gpt, &x, &y, iterations, LayerHooks::off()),
                time_hooks(&gpt, &x, &y, iterations, LayerHooks::full()),
            ),
            1 => {
                let o = time_hooks(&gpt, &x, &y, iterations, LayerHooks::off());
                let f = time_hooks(&gpt, &x, &y, iterations, LayerHooks::full());
                let b = time_baseline(&gpt, &x, &y, iterations);
                (b, o, f)
            }
            _ => {
                let f = time_hooks(&gpt, &x, &y, iterations, LayerHooks::full());
                let b = time_baseline(&gpt, &x, &y, iterations);
                let o = time_hooks(&gpt, &x, &y, iterations, LayerHooks::off());
                (b, o, f)
            }
        };
        baseline.push(b);
        off.push(o);
        full.push(f);
        println!(
            "layer_diagnostics_run | repetition={} baseline_ns={} off_ns={} full_ns={} off_ratio={:.4} full_ratio={:.4} exact=true",
            repetition + 1,
            b,
            o,
            f,
            o as f64 / b as f64,
            f as f64 / b as f64,
        );
    }

    let b = median(&mut baseline);
    let o = median(&mut off);
    let f = median(&mut full);
    println!(
        "layer_diagnostics_summary | iterations={} repeats={} layers={} activations={} gradients={} adjacent_cosine={} baseline_ns={} off_ns={} full_ns={} off_ratio={:.4} full_ratio={:.4} exact=true",
        iterations,
        repeats,
        gpt.cfg.n_layer,
        full_report.activations.len(),
        full_report.gradients.len(),
        full_report.adjacent_gradient_cosine.len(),
        b,
        o,
        f,
        o as f64 / b as f64,
        f as f64 / b as f64,
    );
}
