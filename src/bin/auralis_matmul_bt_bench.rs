use auralis::kernels::{
    matmul_b_t_reference_add_into, matmul_b_t_reference_into,
    matmul_b_t_row_slices_add_into, matmul_b_t_row_slices_into,
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

#[derive(Clone, Copy)]
enum Op {
    Write,
    Add,
}

struct Case {
    rows: usize,
    out_cols: usize,
    result_cols: usize,
    weight: usize,
    op: Op,
    dy: Vec<f32>,
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
        // One output projection gradient-to-input.
        (32, 100, 32, 1, Op::Write),
        // Two transformer layers: FF contraction and expansion backward.
        (32, 32, 96, 2, Op::Write),
        (32, 96, 32, 2, Op::Write),
        // Per layer: attention output + dQ write, then dK/dV add.
        (32, 32, 32, 4, Op::Write),
        (32, 32, 32, 4, Op::Add),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (rows, out_cols, result_cols, weight, op))| Case {
        rows,
        out_cols,
        result_cols,
        weight,
        op,
        dy: data(rows * out_cols, index + 5),
        b: data(result_cols * out_cols, index + 29),
    })
    .collect()
}

fn run_one(mode: Mode, case: &Case, out: &mut [f32]) {
    match (mode, case.op) {
        (Mode::Reference, Op::Write) => matmul_b_t_reference_into(
            &case.dy,
            case.rows,
            case.out_cols,
            &case.b,
            case.result_cols,
            out,
        ),
        (Mode::Slices, Op::Write) => matmul_b_t_row_slices_into(
            &case.dy,
            case.rows,
            case.out_cols,
            &case.b,
            case.result_cols,
            out,
        ),
        (Mode::Reference, Op::Add) => matmul_b_t_reference_add_into(
            &case.dy,
            case.rows,
            case.out_cols,
            &case.b,
            case.result_cols,
            out,
        ),
        (Mode::Slices, Op::Add) => matmul_b_t_row_slices_add_into(
            &case.dy,
            case.rows,
            case.out_cols,
            &case.b,
            case.result_cols,
            out,
        ),
    }
}

fn validate_exact(cases: &[Case]) -> bool {
    for (case_index, case) in cases.iter().enumerate() {
        let initial = data(case.rows * case.result_cols, case_index + 41);
        let mut reference = initial.clone();
        let mut sliced = initial;
        run_one(Mode::Reference, case, &mut reference);
        run_one(Mode::Slices, case, &mut sliced);
        if reference != sliced {
            return false;
        }
    }
    true
}

fn run_suite(mode: Mode, iterations: usize, cases: &[Case]) -> f64 {
    let mut outputs: Vec<Vec<f32>> = cases
        .iter()
        .enumerate()
        .map(|(i, case)| data(case.rows * case.result_cols, i + 53))
        .collect();
    let mut checksum = 0.0f64;

    let started = Instant::now();
    for iteration in 0..iterations {
        for (case_index, case) in cases.iter().enumerate() {
            for repetition in 0..case.weight {
                let out = &mut outputs[case_index];
                if matches!(case.op, Op::Add) {
                    out.fill(0.125);
                }
                run_one(mode, case, out);
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
        eprintln!("B-transpose benchmark measured iterations and repeats must be positive");
        std::process::exit(2);
    }

    let cases = cases();
    let exact_output = validate_exact(&cases);
    if !exact_output {
        eprintln!("slice-based B-transpose matmul does not match reference exactly");
        std::process::exit(3);
    }

    run_suite(Mode::Reference, warmup_iterations, &cases);
    run_suite(Mode::Slices, warmup_iterations, &cases);

    let calls_per_iteration: usize = cases.iter().map(|case| case.weight).sum();
    let add_calls_per_iteration: usize = cases
        .iter()
        .filter(|case| matches!(case.op, Op::Add))
        .map(|case| case.weight)
        .sum();
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
                "matmul_bt_bench_run | repetition={} first={} iterations={} calls_per_iteration={} ",
                "add_calls_per_iteration={} reference_ms={:.3} slices_ms={:.3} speedup={:.4} exact_output={}"
            ),
            repetition + 1,
            if reference_first {
                Mode::Reference.label()
            } else {
                Mode::Slices.label()
            },
            measured_iterations,
            calls_per_iteration,
            add_calls_per_iteration,
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
            "matmul_bt_bench_summary | warmup_iterations={} measured_iterations={} repeats={} ",
            "calls_per_iteration={} add_calls_per_iteration={} median_reference_ms={:.3} ",
            "median_slices_ms={:.3} median_speedup={:.4} exact_output={}"
        ),
        warmup_iterations,
        measured_iterations,
        repeats,
        calls_per_iteration,
        add_calls_per_iteration,
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
