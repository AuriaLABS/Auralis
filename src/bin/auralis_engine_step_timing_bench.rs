use auralis::metrics::EngineStepTiming;
use auralis::model::{Config, Gpt};
use auralis::optim::{Adam, Optimizer};
use auralis::training::{
    train_step_reuse, train_step_reuse_timing, TrainConfig, TrainWorkspace,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    Baseline,
    OffWrapper,
    Timed,
}

struct Run {
    ns_per_step: f64,
    params: Vec<f32>,
    adam_t: i32,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    known_ns: u128,
    total_timing_ns: u128,
    rss_present: bool,
    sample: Option<EngineStepTiming>,
}

fn model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(0x225);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn cfg() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn tokens() -> Vec<usize> {
    (0..8192).map(|i| (i * 37 + i / 7 + 11) % 100).collect()
}

fn step(
    mode: Mode,
    gpt: &mut Gpt,
    adam: &mut Adam,
    stream: &[usize],
    cfg: TrainConfig,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> Option<EngineStepTiming> {
    let global_step = adam.global_step();
    match mode {
        Mode::Baseline => {
            train_step_reuse(gpt, adam, stream, cfg, global_step, grads, workspace)
                .expect("baseline step");
            None
        }
        Mode::OffWrapper => {
            let (_, timing) = train_step_reuse_timing(
                gpt, adam, stream, cfg, global_step, grads, workspace, false,
            )
            .expect("timing-off step");
            assert!(timing.is_none());
            None
        }
        Mode::Timed => {
            let (_, timing) = train_step_reuse_timing(
                gpt, adam, stream, cfg, global_step, grads, workspace, true,
            )
            .expect("timed step");
            let timing = timing.expect("timing enabled");
            assert_eq!(timing.schema_version, EngineStepTiming::SCHEMA_VERSION);
            assert!(timing.known_phase_ns() <= timing.total_ns);
            assert!(timing.data_ns.is_none());
            assert!(timing.checkpoint_ns.is_none());
            assert!(timing.alloc_calls.is_none());
            assert!(timing.alloc_bytes.is_none());
            Some(timing)
        }
    }
}

fn run(mode: Mode, warmup: usize, measured: usize) -> Run {
    let mut gpt = model();
    let n = gpt.collect_params().len();
    let mut adam = Adam::new(n, 3e-3);
    let mut grads = vec![0.0f32; n];
    let mut workspace = TrainWorkspace::new(&gpt);
    let stream = tokens();
    let cfg = cfg();

    for _ in 0..warmup {
        let _ = step(
            mode,
            &mut gpt,
            &mut adam,
            &stream,
            cfg,
            &mut grads,
            &mut workspace,
        );
    }

    let started = Instant::now();
    let mut known_ns = 0u128;
    let mut total_timing_ns = 0u128;
    let mut rss_present = false;
    let mut sample = None;
    for _ in 0..measured {
        if let Some(timing) = step(
            mode,
            &mut gpt,
            &mut adam,
            &stream,
            cfg,
            &mut grads,
            &mut workspace,
        ) {
            known_ns += timing.known_phase_ns() as u128;
            total_timing_ns += timing.total_ns as u128;
            rss_present |= timing.rss_kib.is_some();
            sample = Some(timing);
        }
    }
    let ns_per_step = started.elapsed().as_nanos() as f64 / measured as f64;
    let params = gpt.collect_params();
    let (_, adam_t, m, v) = adam.export();

    Run {
        ns_per_step,
        params,
        adam_t,
        adam_m: m.to_vec(),
        adam_v: v.to_vec(),
        known_ns,
        total_timing_ns,
        rss_present,
        sample,
    }
}

fn exact(a: &Run, b: &Run) -> bool {
    a.params == b.params
        && a.adam_t == b.adam_t
        && a.adam_m == b.adam_m
        && a.adam_v == b.adam_v
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let measured = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60usize);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    assert!(measured >= 50, "measured steps must be at least 50");
    assert!(repeats >= 3, "repeats must be at least 3");

    let mut baseline_ns = Vec::with_capacity(repeats);
    let mut off_ns = Vec::with_capacity(repeats);
    let mut timed_ns = Vec::with_capacity(repeats);
    let mut known_share = Vec::with_capacity(repeats);
    let mut rss_seen = false;
    let mut sample_json = None;
    let mut sample_human = None;

    for rep in 0..repeats {
        let order = match rep % 3 {
            0 => [Mode::Baseline, Mode::OffWrapper, Mode::Timed],
            1 => [Mode::Timed, Mode::Baseline, Mode::OffWrapper],
            _ => [Mode::OffWrapper, Mode::Timed, Mode::Baseline],
        };
        let mut baseline = None;
        let mut off = None;
        let mut timed = None;

        for mode in order {
            let run = run(mode, warmup, measured);
            match mode {
                Mode::Baseline => baseline = Some(run),
                Mode::OffWrapper => off = Some(run),
                Mode::Timed => timed = Some(run),
            }
        }

        let baseline = baseline.unwrap();
        let off = off.unwrap();
        let timed = timed.unwrap();
        assert!(exact(&baseline, &off), "timing-off changed final state");
        assert!(exact(&baseline, &timed), "timed path changed final state");

        let sample = timed.sample.expect("timed sample");
        let json = sample.json();
        let human = sample.human();
        assert!(json.starts_with("{\"schema_version\":1,"));
        assert!(json.contains("\"data_ns\":null"));
        assert!(human.starts_with("engine_step_timing | schema=1"));

        baseline_ns.push(baseline.ns_per_step);
        off_ns.push(off.ns_per_step);
        timed_ns.push(timed.ns_per_step);
        known_share.push(if timed.total_timing_ns > 0 {
            timed.known_ns as f64 / timed.total_timing_ns as f64
        } else {
            0.0
        });
        rss_seen |= timed.rss_present;
        sample_json = Some(json);
        sample_human = Some(human);

        println!(
            "engine_step_timing_run | repetition={} baseline_ns_per_step={:.3} off_ns_per_step={:.3} timed_ns_per_step={:.3} exact_state=true known_share={:.4}",
            rep + 1,
            baseline.ns_per_step,
            off.ns_per_step,
            timed.ns_per_step,
            *known_share.last().unwrap(),
        );
    }

    let base = median(&mut baseline_ns);
    let off = median(&mut off_ns);
    let timed = median(&mut timed_ns);
    let share = median(&mut known_share);
    println!("{}", sample_human.unwrap());
    println!("engine_step_timing_json | {}", sample_json.unwrap());
    println!(
        "engine_step_timing_bench | warmup={} measured_steps={} repeats={} baseline_ns_per_step={:.3} off_ns_per_step={:.3} off_over_baseline={:.4} timed_ns_per_step={:.3} timed_over_baseline={:.4} median_known_share={:.4} rss_present={}",
        warmup,
        measured,
        repeats,
        base,
        off,
        off / base,
        timed,
        timed / base,
        share,
        rss_seen,
    );
}
