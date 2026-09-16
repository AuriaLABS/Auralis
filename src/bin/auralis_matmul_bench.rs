use auralis::kernels::{matmul_reference_into, matmul_row_slices_into};
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

struct Case {
    rows: usize,
    inner: usize,
    cols: usize,
    weight: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
}

fn cases() -> Vec<Case> {
    [
        // Approximate forward call mix for one tiny-model sample:
        // q/k/v + attention projection across two layers = eight d*d calls,
        // plus two FF expansion/contraction calls and final logits.
        (32, 32, 100, 1),
        (32, 96, 32, 2),
        (32, 32, 96, 2),
        (32, 32, 32, 8),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (rows, inner, cols, weight))| Case {
        rows,
        inner,
        cols,
        weight,
        a: data(rows * inner, index + 3),
        b: data(inner * cols, index + 17),
    })
    .collect()
}

fn validate_exact(cases: &[Case]) -> bool {
    for case in cases {
        let mut reference = vec![f32::NAN; case.rows * case.cols];
        let mut sliced = vec![f32::NAN; case.rows * case.cols];
        matmul_reference_into(
            &case.a,
            case.rows,
            case.inner,
            &case.b,
            case.cols,
            &mut reference,
        );
        matmul_row_slices_into(
            &case.a,
            case.rows,
            case.inner,
            &case.b,
            case.cols,
            &mut sliced,
        );
        if reference != sliced {
            return false;
        }
    }
    true
}

fn run_suite(mode: Mode, iterations: usize, cases: &[Case]) -> f64 {
    let mut outputs: Vec<Vec<f32>> = cases
        .iter()
        .map(|case| vec![0.0; case.rows * case.cols])
        .collect();
    let mut checksum = 0.0f64;

    let started = Instant::now();
    for iteration in 0..iterations {
        for (case_index, case) in cases.iter().enumerate() {
            for repetition in 0..case.weight {
                let out = &mut outputs[case_index];
                match mode {
                    Mode::Reference => matmul_reference_into(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.b,
                        case.cols,
                        out,
                    ),
                    Mode::Slices => matmul_row_slices_into(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.b,
                        case.cols,
                        out,
                    ),
                }
                let probe = (iteration + repetition + case_index) % out.len();
                checksum += out[probe] as f64;
            }
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
    let measured_iterations = parse_arg(2, 40);
    let repeats = parse_arg(3, 5);
    if measured_iterations == 0 || repeats == 0 {
        eprintln!("matmul benchmark measured iterations and repeats must be positive");
        std::process::exit(2);
    }

    let cases = cases();
    let exact_output = validate_exact(&cases);
    if !exact_output {
        eprintln!("slice-based matmul does not match scalar reference exactly");
        std::process::exit(3);
    }

    run_suite(Mode::Reference, warmup_iterations, &cases);
    run_suite(Mode::Slices, warmup_iterations, &cases);

    let calls_per_iteration: usize = cases.iter().map(|case| case.weight).sum();
    let mut reference_seconds = Vec::with_capacity(repeats);
    let mut sliced_seconds = Vec::with_capacity(repeats);
    let mut speedups = Vec::with_capacity(repeats);

    for repetition in 0..repeats {
        let reference_first = repetition % 2 == 0;
        let (reference, sliced) = if reference_first {
            (
                run_suite(Mode::Reference, measured_iterations, &cases),
                run_suite(Mode::Slices, measured_iterations, &cases),
            )
        } else {
            let sliced = run_suite(Mode::Slices, measured_iterations, &cases);
            let reference = run_suite(Mode::Reference, measured_iterations, &cases);
            (reference, sliced)
        };

        let speedup = reference / sliced;
        reference_seconds.push(reference);
        sliced_seconds.push(sliced);
        speedups.push(speedup);

        println!(
            concat!(
                "matmul_bench_run | repetition={} first={} iterations={} calls_per_iteration={} ",
                "reference_ms={:.3} slices_ms={:.3} speedup={:.4} exact_output={}"
            ),
            repetition + 1,
            if reference_first {
                Mode::Reference.label()
            } else {
                Mode::Slices.label()
            },
            measured_iterations,
            calls_per_iteration,
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
        concat!(
            "matmul_bench_summary | warmup_iterations={} measured_iterations={} repeats={} ",
            "calls_per_iteration={} median_reference_ms={:.3} median_slices_ms={:.3} ",
            "median_speedup={:.4} exact_output={}"
        ),
        warmup_iterations,
        measured_iterations,
        repeats,
        calls_per_iteration,
        median_reference * 1000.0,
        median_slices * 1000.0,
        median_speedup,
        exact_output,
    );
}

#[cfg(test)]
mod tests {
    use super::{cases, median, validate_exact};

    #[test]
    fn representative_cases_are_exact() {
        assert!(validate_exact(&cases()));
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        let mut odd = [3.0, 1.0, 2.0];
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);
        assert_eq!(median(&mut even), 2.5);
    }
}
