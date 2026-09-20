use auralis::attention::{Attention, AttentionShape, RowSlicesAttention};
use auralis::kernels::{
    attention_backward_row_slices_into, attention_forward_row_slices_into,
};
use std::env;
use std::time::Instant;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i * 37 + salt * 19 + i / 3) % 101) as f32;
            (raw - 50.0) / 31.0
        })
        .collect()
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7usize);
    assert!(iterations > 0 && repeats >= 3);

    let shape = AttentionShape { tokens: 16, width: 32, heads: 4 };
    let n = shape.activation_len().unwrap();
    let p = shape.probs_len().unwrap();
    let q = data(n, 3);
    let k = data(n, 7);
    let v = data(n, 11);
    let dout = data(n, 13);

    let mut ref_out = vec![0.0; n];
    let mut ref_probs = vec![0.0; p];
    let mut opt_out = vec![0.0; n];
    let mut opt_probs = vec![0.0; p];

    attention_forward_row_slices_into(
        &q, &k, &v, shape.tokens, shape.width, shape.heads, &mut ref_out, &mut ref_probs,
    );
    RowSlicesAttention
        .forward_cached(&q, &k, &v, shape, &mut opt_out, &mut opt_probs)
        .unwrap();
    assert_eq!(opt_out, ref_out);
    assert_eq!(opt_probs, ref_probs);

    let mut direct_dq = vec![0.0; n];
    let mut direct_dk = vec![0.0; n];
    let mut direct_dv = vec![0.0; n];
    let mut direct_dp = vec![0.0; shape.tokens];
    let mut bound_dq = vec![0.0; n];
    let mut bound_dk = vec![0.0; n];
    let mut bound_dv = vec![0.0; n];
    let mut bound_dp = vec![0.0; shape.tokens];

    attention_backward_row_slices_into(
        &dout, &q, &k, &v, &ref_probs,
        shape.tokens, shape.width, shape.heads,
        &mut direct_dq, &mut direct_dk, &mut direct_dv, &mut direct_dp,
    );
    RowSlicesAttention.backward(
        &dout, &q, &k, &v, &ref_probs, shape,
        &mut bound_dq, &mut bound_dk, &mut bound_dv, &mut bound_dp,
    ).unwrap();
    assert_eq!(bound_dq, direct_dq);
    assert_eq!(bound_dk, direct_dk);
    assert_eq!(bound_dv, direct_dv);
    assert_eq!(bound_dp, direct_dp);

    let mut forward_direct = Vec::with_capacity(repeats);
    let mut forward_boundary = Vec::with_capacity(repeats);
    let mut backward_direct = Vec::with_capacity(repeats);
    let mut backward_boundary = Vec::with_capacity(repeats);

    for rep in 0..repeats {
        let direct_first = rep % 2 == 0;

        let run_forward_direct = || {
            let mut out = vec![0.0; n];
            let mut probs = vec![0.0; p];
            let start = Instant::now();
            for _ in 0..iterations {
                attention_forward_row_slices_into(
                    &q, &k, &v, shape.tokens, shape.width, shape.heads, &mut out, &mut probs,
                );
            }
            std::hint::black_box((&out, &probs));
            start.elapsed().as_nanos().max(1) as u64
        };
        let run_forward_boundary = || {
            let mut out = vec![0.0; n];
            let mut probs = vec![0.0; p];
            let start = Instant::now();
            for _ in 0..iterations {
                RowSlicesAttention
                    .forward_cached(&q, &k, &v, shape, &mut out, &mut probs)
                    .unwrap();
            }
            std::hint::black_box((&out, &probs));
            start.elapsed().as_nanos().max(1) as u64
        };

        let (fd, fb) = if direct_first {
            (run_forward_direct(), run_forward_boundary())
        } else {
            let fb = run_forward_boundary();
            let fd = run_forward_direct();
            (fd, fb)
        };

        let run_backward_direct = || {
            let mut dq = vec![0.0; n];
            let mut dk = vec![0.0; n];
            let mut dv = vec![0.0; n];
            let mut dp = vec![0.0; shape.tokens];
            let start = Instant::now();
            for _ in 0..iterations {
                attention_backward_row_slices_into(
                    &dout, &q, &k, &v, &ref_probs,
                    shape.tokens, shape.width, shape.heads,
                    &mut dq, &mut dk, &mut dv, &mut dp,
                );
            }
            std::hint::black_box((&dq, &dk, &dv, &dp));
            start.elapsed().as_nanos().max(1) as u64
        };
        let run_backward_boundary = || {
            let mut dq = vec![0.0; n];
            let mut dk = vec![0.0; n];
            let mut dv = vec![0.0; n];
            let mut dp = vec![0.0; shape.tokens];
            let start = Instant::now();
            for _ in 0..iterations {
                RowSlicesAttention
                    .backward(
                        &dout, &q, &k, &v, &ref_probs, shape,
                        &mut dq, &mut dk, &mut dv, &mut dp,
                    )
                    .unwrap();
            }
            std::hint::black_box((&dq, &dk, &dv, &dp));
            start.elapsed().as_nanos().max(1) as u64
        };
        let (bd, bb) = if direct_first {
            (run_backward_direct(), run_backward_boundary())
        } else {
            let bb = run_backward_boundary();
            let bd = run_backward_direct();
            (bd, bb)
        };

        forward_direct.push(fd);
        forward_boundary.push(fb);
        backward_direct.push(bd);
        backward_boundary.push(bb);
        println!(
            "attention_boundary_run | repetition={} first={} forward_ratio={:.4} backward_ratio={:.4} exact=true",
            rep + 1,
            if direct_first { "direct" } else { "boundary" },
            fb as f64 / fd as f64,
            bb as f64 / bd as f64,
        );
    }

    let fd = median(&mut forward_direct);
    let fb = median(&mut forward_boundary);
    let bd = median(&mut backward_direct);
    let bb = median(&mut backward_boundary);
    println!(
        "attention_boundary_summary | tokens={} width={} heads={} iterations={} repeats={} forward_direct_ns={} forward_boundary_ns={} forward_ratio={:.4} backward_direct_ns={} backward_boundary_ns={} backward_ratio={:.4} exact=true",
        shape.tokens, shape.width, shape.heads, iterations, repeats,
        fd, fb, fb as f64 / fd as f64,
        bd, bb, bb as f64 / bd as f64,
    );
}
