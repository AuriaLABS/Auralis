use auralis::layer_diagnostics::LayerHooks;
use auralis::model::{Config, Gpt};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn model(layers: usize) -> Gpt {
    let cfg = Config {
        vocab: 11,
        n_embd: 8,
        n_head: 2,
        n_layer: layers,
        block: 8,
        n_ff: 16,
    };
    let mut rng = StdRng::seed_from_u64(659_918);
    Gpt::new(cfg, &mut rng)
}

fn batch() -> (Vec<usize>, Vec<usize>) {
    let x = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let y = vec![2, 3, 4, 5, 6, 7, 8, 9];
    (x, y)
}

#[test]
fn hooks_off_are_exact_historical_backward() {
    let gpt = model(3);
    let (x, y) = batch();
    let n = gpt.collect_params().len();
    let mut expected = vec![0.0; n];
    let mut observed = vec![0.0; n];

    let baseline = gpt.backward_into(&x, &y, &mut expected);
    let (off_loss, report) =
        gpt.backward_with_layer_diagnostics(&x, &y, &mut observed, LayerHooks::off());

    assert_eq!(off_loss.to_bits(), baseline.to_bits());
    assert_eq!(observed, expected);
    assert!(report.is_none());
}

#[test]
fn full_hooks_report_every_layer_without_changing_gradients() {
    let gpt = model(3);
    let (x, y) = batch();
    let n = gpt.collect_params().len();
    let mut expected = vec![0.0; n];
    let mut observed = vec![0.0; n];

    let baseline = gpt.backward_into(&x, &y, &mut expected);
    let (loss, report) =
        gpt.backward_with_layer_diagnostics(&x, &y, &mut observed, LayerHooks::full());
    let report = report.expect("full hooks report");

    assert_eq!(loss.to_bits(), baseline.to_bits());
    assert_eq!(observed, expected);
    assert_eq!(report.activations.len(), 27);
    assert_eq!(report.gradients.len(), 3);
    assert_eq!(report.adjacent_gradient_cosine.len(), 2);

    for (layer, chunk) in report.activations.chunks_exact(9).enumerate() {
        assert!(chunk.iter().all(|entry| entry.layer == layer));
        assert!(chunk.iter().all(|entry| entry.stats.l2.is_some()));
        assert!(chunk.iter().all(|entry| entry.stats.mean.is_some()));
        assert!(chunk.iter().all(|entry| entry.stats.histogram.is_some()));
    }
    for (layer, entry) in report.gradients.iter().enumerate() {
        assert_eq!(entry.layer, layer);
        assert!(entry.stats.l2.is_some());
        assert!(entry.stats.mean.is_some());
        assert!(entry.stats.histogram.is_some());
        assert!(entry.stats.max_abs >= 0.0);
    }

    let json = report.json();
    assert!(json.starts_with("{\"activations\":["));
    assert!(json.contains("\"gradients\":["));
    assert!(json.contains("\"adjacent_gradient_cosine\":["));
    assert!(json.contains("\"histogram\":{"));
}

#[test]
fn compact_hooks_skip_histogram_and_cosine() {
    let gpt = model(2);
    let (x, y) = batch();
    let mut grads = vec![0.0; gpt.collect_params().len()];
    let (_, report) =
        gpt.backward_with_layer_diagnostics(&x, &y, &mut grads, LayerHooks::compact());
    let report = report.unwrap();

    assert_eq!(report.activations.len(), 18);
    assert_eq!(report.gradients.len(), 2);
    assert!(report.adjacent_gradient_cosine.is_empty());
    assert!(report.activations.iter().all(|entry| entry.stats.histogram.is_none()));
    assert!(report.gradients.iter().all(|entry| entry.stats.histogram.is_none()));
}
