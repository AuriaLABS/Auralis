use auralis::kernels::{
    attention_backward_reference_into, attention_backward_row_slices_into,
    attention_forward_reference_into,
};
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
    dout: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probs: Vec<f32>,
}

fn inputs() -> Inputs {
    let q = data(T * D, 37);
    let k = data(T * D, 41);
    let v = data(T * D, 43);
    let dout = data(T * D, 47);
    let mut forward_out = vec![0.0; T * D];
    let mut probs = vec![0.0; N_HEAD * T * T];
    attention_forward_reference_into(
        &q,
        &k,
        &v,
        T,
        D,
        N_HEAD,
        &mut forward_out,
        &mut probs,
    );
    black_box(forward_out);
    Inputs {
        dout,
        q,
        k,
        v,
        probs,
    }
}

fn run_one(
    mode: Mode,
    input: &Inputs,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    dp: &mut [f32],
) {
    match mode {
        Mode::Reference => attention_backward_reference_into(
            &input.dout,
            &input.q,
            &input.k,
            &input.v,
            &input.probs,
            T,
            D,
            N_HEAD,
            dq,
            dk,
            dv,
            dp,
        ),
        Mode::Slices => attention_backward_row_slices_into(
            &input.dout,
            &input.q,
            &input.k,
            &input.v,
            &input.probs,
            T,
            D,
            N_HEAD,
            dq,
            dk,
            dv,
            dp,
        ),
    }
}

fn validate_exact(input: &Inputs) -> bool {
    let mut reference_dq = vec![f32::NAN; T * D];
    let mut reference_dk = vec![f32::NAN; T * D];
    let mut reference_dv = vec![f32::NAN; T * D];
    let mut reference_dp = vec![f32::NAN; T];
    let mut sliced_dq = vec![f32::NAN; T * D];
    let mut sliced_dk = vec![f32::NAN; T * D];
    let mut sliced_dv = vec![f32::NAN; T * D];
    let mut sliced_dp = vec![f32::NAN; T];

    run_one(
        Mode::Reference,
        input,
        &mut reference_dq,
        &mut reference_dk,
        &mut reference_dv,
        &mut reference_dp,
    );
    run_one(
        Mode::Slices,
        input,
        &mut sliced_dq,
        &mut sliced_dk,
        &mut sliced_dv,
        &mut sliced_dp,
    );

    reference_dq == sliced_dq
        && reference_dk == sliced_dk
        && reference_dv == sliced_dv
        && reference_dp == sliced_dp
}

fn run_suite(mode: Mode, iterations: usize, input: &Inputs) -> f64 {
    let mut dq = vec![0.0; T * D];
    let mut dk = vec![0.0; T * D];
    let mut dv = vec![0.0; T * D];
    let mut dp = vec![0.0; T];
    let mut checksum = 0.0f64;
    let started = Instant::now();
    for iteration in 0..iterations {
        for call in 0..CALLS_PER_ITERATION {
            run_one(mode, input, &mut dq, &mut dk, &mut dv, &mut dp);
            checksum += dq[(iteration + call) % dq.len()] as f64;
            checksum += dk[(iteration * 3 + call) % dk.len()] as f64;
            checksum += dv[(iteration * 5 + call) % dv.len()] as f64;
            checksum += dp[(iteration * 7 + call) % dp.len()] as f64;
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
        eprintln!("attention backward benchmark measured iterations and repeats must be positive");
        std::process::exit(2);
    }

    let input = inputs();
    let exact_output = validate_exact(&input);
    if !exact_output {
        eprintln!("slice-based attention backward does not match reference exactly");
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
            "attention_backward_bench_run | repetition={} first={} iterations={} calls_per_iteration={} reference_ms={:.3} slices_ms={:.3} speedup={:.4} exact_output={}",
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
        "attention_backward_bench_summary | warmup_iterations={} measured_iterations={} repeats={} calls_per_iteration={} median_reference_ms={:.3} median_slices_ms={:.3} median_speedup={:.4} exact_output={}",
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
