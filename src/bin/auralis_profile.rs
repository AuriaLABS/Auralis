use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::training::{train_step, TrainConfig};
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

    // Warm the code path on a separate state so none of its allocations or
    // optimizer updates contaminate the measured experiment.
    {
        let mut warm_model = make_model(1234);
        let n = warm_model.collect_params().len();
        let mut warm_adam = Adam::new(n, 3e-3);
        let mut warm_grads = vec![0.0; n];
        train_step(
            &mut warm_model,
            &mut warm_adam,
            &tokens,
            cfg,
            0,
            &mut warm_grads,
        )
        .expect("warmup training step");
    }

    let mut model = make_model(1234);
    let params = model.collect_params().len();
    let mut adam = Adam::new(params, 3e-3);
    let mut grads = vec![0.0; params];

    reset_counters();
    let started = Instant::now();
    let mut processed_tokens = 0u64;

    for _ in 0..steps {
        let global_step = adam.t.max(0) as u64;
        let metrics = train_step(
            &mut model,
            &mut adam,
            &tokens,
            cfg,
            global_step,
            &mut grads,
        )
        .expect("profile training step");
        processed_tokens += metrics.tokens as u64;
    }

    let elapsed = started.elapsed().as_secs_f64();
    let counts = snapshot();
    let tok_per_s = if elapsed > 0.0 {
        processed_tokens as f64 / elapsed
    } else {
        f64::INFINITY
    };
    let alloc_calls_per_step = counts.alloc_calls as f64 / steps as f64;
    let alloc_bytes_per_step = counts.alloc_bytes as f64 / steps as f64;
    let alloc_bytes_per_token = if processed_tokens > 0 {
        counts.alloc_bytes as f64 / processed_tokens as f64
    } else {
        0.0
    };

    println!(
        concat!(
            "engine_profile | steps={} tokens={} params={} seconds={:.6} tok_per_s={:.3} ",
            "alloc_calls={} alloc_bytes={} realloc_calls={} realloc_old_bytes={} realloc_new_bytes={} ",
            "dealloc_calls={} dealloc_bytes={} alloc_calls_per_step={:.3} alloc_bytes_per_step={:.3} ",
            "alloc_bytes_per_token={:.3}"
        ),
        steps,
        processed_tokens,
        params,
        elapsed,
        tok_per_s,
        counts.alloc_calls,
        counts.alloc_bytes,
        counts.realloc_calls,
        counts.realloc_old_bytes,
        counts.realloc_new_bytes,
        counts.dealloc_calls,
        counts.dealloc_bytes,
        alloc_calls_per_step,
        alloc_bytes_per_step,
        alloc_bytes_per_token,
    );
}
