use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::training::{train_step, train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct CountingAllocator;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

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
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    DEALLOC_CALLS.store(0, Ordering::Relaxed);
}

fn print_line(mode: &str, iterations: usize, seconds: f64, checksum: f32) {
    let calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    let deallocs = DEALLOC_CALLS.load(Ordering::Relaxed);
    println!(
        concat!(
            "train_profile | mode={} iterations={} seconds={:.6} ",
            "alloc_calls={} alloc_bytes={} dealloc_calls={} ",
            "alloc_calls_per_iter={:.3} alloc_bytes_per_iter={:.3} checksum={:.6}"
        ),
        mode,
        iterations,
        seconds,
        calls,
        bytes,
        deallocs,
        calls as f64 / iterations as f64,
        bytes as f64 / iterations as f64,
        checksum,
    );
}

fn make_model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(0xA11CE_2801);
    Gpt::new(Config::tiny(64), &mut rng)
}

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);
    if iterations == 0 {
        eprintln!("iterations must be positive");
        std::process::exit(2);
    }

    let tokens: Vec<usize> = (0..256).map(|i| i % 64).collect();
    let cfg = TrainConfig {
        seed: 0xA11CE_2802,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    };

    let mut reference = make_model();
    let mut reuse = make_model();
    let n = reference.collect_params().len();
    let mut adam_ref = Adam::new(n, 3e-3);
    let mut adam_reuse = Adam::new(n, 3e-3);
    let mut grads_ref = vec![0.0; n];
    let mut grads_reuse = vec![0.0; n];
    let mut workspace = TrainWorkspace::new(&reuse);

    let _ = train_step(&mut reference, &mut adam_ref, &tokens, cfg, 0, &mut grads_ref);
    let _ = train_step_reuse(
        &mut reuse,
        &mut adam_reuse,
        &tokens,
        cfg,
        0,
        &mut grads_reuse,
        &mut workspace,
    );

    reset();
    let started = Instant::now();
    let mut checksum = 0.0f32;
    for step in 1..=iterations {
        let m = train_step(
            black_box(&mut reference),
            black_box(&mut adam_ref),
            black_box(&tokens),
            cfg,
            step as u64,
            black_box(&mut grads_ref),
        )
        .expect("reference step");
        checksum += m.loss;
    }
    print_line("reference", iterations, started.elapsed().as_secs_f64(), checksum);

    reset();
    let started = Instant::now();
    let mut checksum = 0.0f32;
    for step in 1..=iterations {
        let m = train_step_reuse(
            black_box(&mut reuse),
            black_box(&mut adam_reuse),
            black_box(&tokens),
            cfg,
            step as u64,
            black_box(&mut grads_reuse),
            black_box(&mut workspace),
        )
        .expect("reuse step");
        checksum += m.loss;
    }
    print_line("reuse", iterations, started.elapsed().as_secs_f64(), checksum);
}
