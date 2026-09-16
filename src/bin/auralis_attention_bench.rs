use auralis::kernels::{attention_forward_reference_into, attention_forward_row_slices_into};
use std::env;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    Reference,
    Slices,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Slices => "slices",
        }
    }
}

const T: usize = 32;
const D: usize = 32;
const N_HEAD: usize = 4;
const CALLS_PER_ITERATION: usize = 2;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn inputs() -> Inputs {
    Inputs {
        q: data(T * D, 37),
        k: data(T * D, 41),
        v: data(T * D, 43),
    }
}

fn run_one(mode: Mode, input: &Inputs, out: &mut [f32], probs: &mut [f32]) {
    match mode {
        Mode::Reference => attention_forward_reference_into(
            &input.q, &input.k, &input.v, T, D, N_HEAD, out, probs,
        ),
        Mode::Slices => attention_forward_row_slices_into(
            &input.q, &input.k, &input.v, T, D, N_HEAD, out, probs,
        ),
    }
}

fn validate_exact(input: &Inputs) -> bool {
    let mut reference_out = vec![f32::NAN; T * D];
    let mut reference_probs = vec![f32::NAN; N_HEAD * T * T];
    let mut sliced_out = vec![f32::NAN; T * D];
    let mut sliced_probs = vec![f32::NAN; N_HEAD * T * T];
    run_one(
        Mode::Reference,
        input,
        &mut reference_out,
        &mut reference_probs,
    );
    run_one(
        Mode::Slices,
        input,
        &mut sliced_out,
        &mut sliced_probs,
    );
    reference_out == sliced_out && reference_probs == sliced_probs
}

fn run_suite(mode: Mode, iterations: usize, input: &Inputs) -> f64 {
    let mut out = vec![0.0; T * D];
    let mut probs = vec![0.0; N_HEAD * T * T];
    let mut checksum = 0.0f64;
    let started = Instant::now();
    for iteration in 0..iterations {
        for call in 0..CALLS_PER_ITERATION {
            run_one(mode, input, &mut out, &mut probs);
            checksum += out[(iteration + call) % out.len()] as f64;
            checksum += probs[(iteration * 7 + call) % probs.len()] as f64;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    black_box(checksum);
    seconds
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn parse_arg(index: usize, default: usize) -> usize {
    env::args()
        .nth(index)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let warmup_iterations = parse_arg(1, 5);
    let measured_iterations = parse_arg(2, 80);
    let repeats = parse_arg(3, 5);
    if measured_iterations == 0 || repeats == 0 {
        eprintln!("attention benchmark measured iterations and repeats must be positive");
        std::process::exit(2);
    }

    let input = inputs();
    let exact_output = validate_exact(&input);
    if !exact_output {
        eprintln!("slice-based attention forward does not match reference exactly");
        std::process::exit(3);
    }

    run_suite(Mode::Reference, warmup_iterations, &input);
    run_suite(Mode::Slices, warmup_iterations, &input);

    let mut reference_seconds = Vec::with_capacity(repeats);
    let mut sliced_seconds = Vec::with_capacity(repeats);
    let mut speedups = Vec::with_capacity(repeats);

    for repetition in 0..repeats {
        let reference_first = repetition % 2 == 0;
        let (reference, sliced) = if reference_first {
            (
                run_suite(Mode::Reference, measured_iterations, &input),
                run_suite(Mode::Slices, measured_iterations, &input),
            )
        } else {
            let sliced = run_suite(Mode::Slices, measured_iterations, &input);
            let reference = run_suite(Mode::Reference, measured_iterations, &input);
            (reference, sliced)
        };
        let speedup = reference / sliced;
        reference_seconds.push(reference);
        sliced_seconds.push(sliced);
        speedups.push(speedup);
        println!(
            "attention_bench_run | repetition={} first={} iterations={} calls_per_iteration={} reference_ms={:.3} slices_ms={:.3} speedup={:.4} exact_output={}",
            repetition + 1,
            if reference_first { Mode::Reference.label() } else { Mode::Slices.label() },
            measured_iterations,
            CALLS_PER_ITERATION,
            reference * 1000.0,
            sliced * 1000.0,
            speedup,
            exact_output,
        );
    }

    let median_reference = median(&mut reference_seconds);
    let median_slices = median(&mut sliced_seconds);
    let median_speedup = median(&mut speedups);
    println!(
        "attention_bench_summary | warmup_iterations={} measured_iterations={} repeats={} calls_per_iteration={} median_reference_ms={:.3} median_slices_ms={:.3} median_speedup={:.4} exact_output={}",
        warmup_iterations,
        measured_iterations,
        repeats,
        CALLS_PER_ITERATION,
        median_reference * 1000.0,
        median_slices * 1000.0,
        median_speedup,
        exact_output,
    );
}

#[cfg(test)]
mod tests {
    use super::{inputs, median, validate_exact};

    #[test]
    fn representative_case_is_exact() {
        assert!(validate_exact(&inputs()));
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        let mut odd = [3.0, 1.0, 2.0];
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);
        assert_eq!(median(&mut even), 2.5);
    }
}
