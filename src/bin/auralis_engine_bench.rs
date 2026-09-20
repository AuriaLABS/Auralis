use auralis::bench_runner::{run_legacy, BenchProtocol};
use std::env;

fn parse_arg(index: usize, default: usize) -> usize {
    env::args()
        .nth(index)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() {
    let protocol = BenchProtocol {
        warmup: parse_arg(1, 3),
        iterations: parse_arg(2, 20),
        repetitions: parse_arg(3, 5),
    };
    match run_legacy("engine", protocol) {
        Ok(out) => print!("{out}"),
        Err(e) => {
            eprintln!("Engine benchmark failed: {e}");
            std::process::exit(2);
        }
    }
}
