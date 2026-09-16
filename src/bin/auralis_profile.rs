use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::training::{train_step, train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct CountingAllocator;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_OLD_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_NEW_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            REALLOC_OLD_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            REALLOC_NEW_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, Default)]
struct AllocationSnapshot {
    alloc_calls: u64,
    alloc_bytes: u64,
    realloc_calls: u64,
    realloc_old_bytes: u64,
    realloc_new_bytes: u64,
    dealloc_calls: u64,
    dealloc_bytes: u64,
}

#[derive(Clone, Copy)]
enum ProfileMode {
    Reference,
    Reuse,
}

impl ProfileMode {
    fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Reuse => "reuse",
        }
    }
}

struct ProfileRun {
    counts: AllocationSnapshot,
    seconds: f64,
    processed_tokens: u64,
    params: usize,
    final_params: Vec<f32>,
    adam_t: i32,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
}

fn reset_counters() {
    for counter in [
        &ALLOC_CALLS,
        &ALLOC_BYTES,
        &REALLOC_CALLS,
        &REALLOC_OLD_BYTES,
        &REALLOC_NEW_BYTES,
        &DEALLOC_CALLS,
        &DEALLOC_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
        realloc_old_bytes: REALLOC_OLD_BYTES.load(Ordering::Relaxed),
        realloc_new_bytes: REALLOC_NEW_BYTES.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn make_model(seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn profile_config() -> TrainConfig {
    TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    }
}

fn token_stream() -> Vec<usize> {
    (0..4096)
        .map(|i| ((i * 37 + i / 7 + 11) % 100) as usize)
        .collect()
}

fn execute_step(
    mode: ProfileMode,
    model: &mut Gpt,
    adam: &mut Adam,
    tokens: &[usize],
    cfg: TrainConfig,
    grads: &mut [f32],
    workspace: &mut TrainWorkspace,
) -> usize {
    let global_step = adam.t.max(0) as u64;
    let metrics = match mode {
        ProfileMode::Reference => train_step(model, adam, tokens, cfg, global_step, grads),
        ProfileMode::Reuse => {
            train_step_reuse(model, adam, tokens, cfg, global_step, grads, workspace)
        }
    }
    .expect("profile training step");
    metrics.tokens
}

fn warm(mode: ProfileMode, cfg: TrainConfig, tokens: &[usize]) {
    let mut model = make_model(1234);
    let n = model.collect_params().len();
    let mut adam = Adam::new(n, 3e-3);
    let mut grads = vec![0.0; n];
    let mut workspace = TrainWorkspace::new(n);
    execute_step(mode, &mut model, &mut adam, tokens, cfg, &mut grads, &mut workspace);
}

fn measure(mode: ProfileMode, steps: usize, cfg: TrainConfig, tokens: &[usize]) -> ProfileRun {
    let mut model = make_model(1234);
    let params = model.collect_params().len();
    let mut adam = Adam::new(params, 3e-3);
    let mut grads = vec![0.0; params];
    let mut workspace = TrainWorkspace::new(params);

    reset_counters();
    let started = Instant::now();
    let mut processed_tokens = 0u64;
    for _ in 0..steps {
        processed_tokens += execute_step(
            mode,
            &mut model,
            &mut adam,
            tokens,
            cfg,
            &mut grads,
            &mut workspace,
        ) as u64;
    }
    let seconds = started.elapsed().as_secs_f64();
    let counts = snapshot();

    // Everything below happens after the allocation snapshot and therefore does
    // not contaminate the measured hot path.
    let final_params = model.collect_params();
    let (_, adam_t, m, v) = adam.export();
    ProfileRun {
        counts,
        seconds,
        processed_tokens,
        params,
        final_params,
        adam_t,
        adam_m: m.to_vec(),
        adam_v: v.to_vec(),
    }
}

fn tokens_per_second(run: &ProfileRun) -> f64 {
    if run.seconds > 0.0 {
        run.processed_tokens as f64 / run.seconds
    } else {
        f64::INFINITY
    }
}

fn print_run(mode: ProfileMode, steps: usize, run: &ProfileRun) {
    let alloc_calls_per_step = run.counts.alloc_calls as f64 / steps as f64;
    let alloc_bytes_per_step = run.counts.alloc_bytes as f64 / steps as f64;
    let alloc_bytes_per_token = if run.processed_tokens > 0 {
        run.counts.alloc_bytes as f64 / run.processed_tokens as f64
    } else {
        0.0
    };

    println!(
        concat!(
            "engine_profile | mode={} steps={} tokens={} params={} seconds={:.6} tok_per_s={:.3} ",
            "alloc_calls={} alloc_bytes={} realloc_calls={} realloc_old_bytes={} realloc_new_bytes={} ",
            "dealloc_calls={} dealloc_bytes={} alloc_calls_per_step={:.3} alloc_bytes_per_step={:.3} ",
            "alloc_bytes_per_token={:.3}"
        ),
        mode.label(),
        steps,
        run.processed_tokens,
        run.params,
        run.seconds,
        tokens_per_second(run),
        run.counts.alloc_calls,
        run.counts.alloc_bytes,
        run.counts.realloc_calls,
        run.counts.realloc_old_bytes,
        run.counts.realloc_new_bytes,
        run.counts.dealloc_calls,
        run.counts.dealloc_bytes,
        alloc_calls_per_step,
        alloc_bytes_per_step,
        alloc_bytes_per_token,
    );
}

fn reduction_pct(reference: u64, candidate: u64) -> f64 {
    if reference == 0 {
        0.0
    } else {
        100.0 * (reference as f64 - candidate as f64) / reference as f64
    }
}

fn main() {
    let steps = env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
    if steps == 0 {
        eprintln!("profile steps must be positive");
        std::process::exit(2);
    }

    let cfg = profile_config();
    let tokens = token_stream();
    warm(ProfileMode::Reference, cfg, &tokens);
    warm(ProfileMode::Reuse, cfg, &tokens);

    let reference = measure(ProfileMode::Reference, steps, cfg, &tokens);
    let reuse = measure(ProfileMode::Reuse, steps, cfg, &tokens);

    let exact_state = reference.final_params == reuse.final_params
        && reference.adam_t == reuse.adam_t
        && reference.adam_m == reuse.adam_m
        && reference.adam_v == reuse.adam_v;
    if !exact_state {
        eprintln!("Engine candidate changed the final training state");
        std::process::exit(3);
    }

    print_run(ProfileMode::Reference, steps, &reference);
    print_run(ProfileMode::Reuse, steps, &reuse);
    println!(
        concat!(
            "engine_profile_delta | exact_state={} alloc_calls_reduction_pct={:.3} ",
            "alloc_bytes_reduction_pct={:.3} realloc_calls_reduction_pct={:.3} tok_per_s_ratio={:.3}"
        ),
        exact_state,
        reduction_pct(reference.counts.alloc_calls, reuse.counts.alloc_calls),
        reduction_pct(reference.counts.alloc_bytes, reuse.counts.alloc_bytes),
        reduction_pct(reference.counts.realloc_calls, reuse.counts.realloc_calls),
        tokens_per_second(&reuse) / tokens_per_second(&reference),
    );
}
