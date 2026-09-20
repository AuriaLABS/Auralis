use auralis::brain_ab::{run_experiment, AbProtocol, AbVariant};
use auralis::model::{Config, NormalizationKind};
use auralis::optim::OptimizerId;
use auralis::position::PositionKind;
use std::env;

fn parse_usize(index: usize, default: usize) -> usize {
    env::args()
        .nth(index)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn config(layers: usize) -> Config {
    Config {
        vocab: 8,
        n_embd: 8,
        n_head: 2,
        n_layer: layers,
        block: 8,
        n_ff: 16,
    }
}

fn main() {
    let steps = parse_usize(1, 8);
    let repeats = parse_usize(2, 3);
    let scenario = env::args().nth(3).unwrap_or_else(|| "depth".into());
    let json = env::args().any(|arg| arg == "--json");

    let (a, b) = match scenario.as_str() {
        "same" => (
            AbVariant { label: "same-a".into(), config: config(1), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
            AbVariant { label: "same-b".into(), config: config(1), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
        ),
        "depth" => (
            AbVariant { label: "1-layer".into(), config: config(1), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
            AbVariant { label: "2-layer".into(), config: config(2), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
        ),
        "normalization" => (
            AbVariant { label: "layernorm".into(), config: config(2), normalization: NormalizationKind::LayerNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
            AbVariant { label: "rmsnorm".into(), config: config(2), normalization: NormalizationKind::RmsNorm, position: PositionKind::LearnedAbsolute, optimizer: OptimizerId::Adam },
        ),
        "rope" => (
            AbVariant {
                label: "learned-absolute".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "rope".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::Rope,
                optimizer: OptimizerId::Adam,
            },
        ),
        "optimizer-adamw" => (
            AbVariant {
                label: "adam".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "adamw".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::AdamW,
            },
        ),
        "optimizer-lion" => (
            AbVariant {
                label: "adam".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Adam,
            },
            AbVariant {
                label: "lion".into(),
                config: config(2),
                normalization: NormalizationKind::LayerNorm,
                position: PositionKind::LearnedAbsolute,
                optimizer: OptimizerId::Lion,
            },
        ),
        other => {
            eprintln!(
                "unknown A/B scenario {other}; expected same|depth|normalization|rope|optimizer-adamw|optimizer-lion"
            );
            std::process::exit(2);
        }
    };

    let protocol = AbProtocol {
        steps,
        repeats,
        token_count: 512,
        ..AbProtocol::default()
    };
    match run_experiment(protocol, a, b) {
        Ok(result) => {
            if json {
                println!("{}", result.json());
            } else {
                print!("{}", result.human());
            }
        }
        Err(e) => {
            eprintln!("A/B experiment failed: {e}");
            std::process::exit(3);
        }
    }
}
