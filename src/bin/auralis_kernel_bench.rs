use auralis::kernels::{matmul_grad_b_reference, matmul_grad_b_rowwise_zeroed};
use std::env;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
enum KernelMode {
    Reference,
    Rowwise,
}

impl KernelMode {
    fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Rowwise => "rowwise",
        }
    }
}

struct Case {
    rows: usize,
    inner: usize,
    cols: usize,
    weight: usize,
    a: Vec<f32>,
    dy: Vec<f32>,
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
        // Approximate call mix for one tiny-model backward sample:
        // output projection once, FF matrices twice each, and four d*d
        // attention matrices per layer across two layers.
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
        a: data(rows * inner, index + 1),
        dy: data(rows * cols, index + 11),
    })
    .collect()
}

fn validate_exact(cases: &[Case]) -> bool {
    for case in cases {
        let mut reference = vec![0.0; case.inner * case.cols];
        let mut rowwise = vec![0.0; case.inner * case.cols];
        matmul_grad_b_reference(
            &case.a,
            case.rows,
            case.inner,
            &case.dy,
            case.cols,
            &mut reference,
        );
        matmul_grad_b_rowwise_zeroed(
            &case.a,
            case.rows,
            case.inner,
            &case.dy,
            case.cols,
            &mut rowwise,
        );
        if reference != rowwise {
            return false;
        }
    }
    true
}

fn run_suite(mode: KernelMode, iterations: usize, cases: &[Case]) -> f64 {
    let mut outputs: Vec<Vec<f32>> = cases
        .iter()
        .map(|case| vec![0.0; case.inner * case.cols])
        .collect();
    let mut checksum = 0.0f64;

    let started = Instant::now();
    for iteration in 0..iterations {
        for (case_index, case) in cases.iter().enumerate() {
            for repetition in 0..case.weight {
                let out = &mut outputs[case_index];
                out.fill(0.0);
                match mode {
                    KernelMode::Reference => matmul_grad_b_reference(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.dy,
                        case.cols,
                        out,
                    ),
                    KernelMode::Rowwise => matmul_grad_b_rowwise_zeroed(
                        &case.a,
                        case.rows,
                        case.inner,
                        &case.dy,
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
        eprintln!("kernel benchmark measured iterations and repeats must be positive");
        std::process::exit(2);
    }

    let cases = cases();
    let exact_output = validate_exact(&cases);
    if !exact_output {
        eprintln!("row-wise grad-B kernel does not match scalar reference exactly");
        std::process::exit(3);
    }

    run_suite(KernelMode::Reference, warmup_iterations, &cases);
    run_suite(KernelMode::Rowwise, warmup_iterations, &cases);

    let calls_per_iteration: usize = cases.iter().map(|case| case.weight).sum();
    let mut reference_seconds = Vec::with_capacity(repeats);
    let mut rowwise_seconds = Vec::with_capacity(repeats);
    let mut speedups = Vec::with_capacity(repeats);

    for repetition in 0..repeats {
        let reference_first = repetition % 2 == 0;
        let (reference, rowwise) = if reference_first {
            (
                run_suite(KernelMode::Reference, measured_iterations, &cases),
                run_suite(KernelMode::Rowwise, measured_iterations, &cases),
            )
        } else {
            let rowwise = run_suite(KernelMode::Rowwise, measured_iterations, &cases);
            let reference = run_suite(KernelMode::Reference, measured_iterations, &cases);
            (reference, rowwise)
        };

        let speedup = reference / rowwise;
        reference_seconds.push(reference);
        rowwise_seconds.push(rowwise);
        speedups.push(speedup);

        println!(
            concat!(
                "kernel_bench_run | kernel=matmul_grad_b repetition={} first={} ",
                "iterations={} calls_per_iteration={} reference_ms={:.3} rowwise_ms={:.3} ",
                "speedup={:.4} exact_output={}"
            ),
            repetition + 1,
            if reference_first {
                KernelMode::Reference.label()
            } else {
                KernelMode::Rowwise.label()
            },
            measured_iterations,
            calls_per_iteration,
            reference * 1000.0,
            rowwise * 1000.0,
            speedup,
            exact_output,
        );
    }

    let median_reference = median(&mut reference_seconds);
    let median_rowwise = median(&mut rowwise_seconds);
    let median_speedup = median(&mut speedups);
    println!(
        concat!(
            "kernel_bench_summary | kernel=matmul_grad_b warmup_iterations={} measured_iterations={} ",
            "repeats={} calls_per_iteration={} median_reference_ms={:.3} median_rowwise_ms={:.3} ",
            "median_speedup={:.4} exact_output={}"
        ),
        warmup_iterations,
        measured_iterations,
        repeats,
        calls_per_iteration,
        median_reference * 1000.0,
        median_rowwise * 1000.0,
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
