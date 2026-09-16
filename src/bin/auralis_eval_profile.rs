use auralis::model::{Config, Gpt};
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
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
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
struct Snapshot {
    alloc_calls: u64,
    alloc_bytes: u64,
    realloc_calls: u64,
    realloc_new_bytes: u64,
    dealloc_calls: u64,
    dealloc_bytes: u64,
}

fn reset() {
    for counter in [
        &ALLOC_CALLS,
        &ALLOC_BYTES,
        &REALLOC_CALLS,
        &REALLOC_NEW_BYTES,
        &DEALLOC_CALLS,
        &DEALLOC_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn snapshot() -> Snapshot {
    Snapshot {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
        realloc_new_bytes: REALLOC_NEW_BYTES.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn make_model() -> Gpt {
    let mut rng = StdRng::seed_from_u64(0xA11CE_2701);
    Gpt::new(Config::tiny(100), &mut rng)
}

fn print_snapshot(mode: &str, iterations: usize, elapsed_s: f64, counts: Snapshot) {
    println!(
        concat!(
            "eval_profile | mode={} iterations={} seconds={:.6} calls_per_s={:.3} ",
            "alloc_calls={} alloc_bytes={} realloc_calls={} realloc_new_bytes={} ",
            "dealloc_calls={} dealloc_bytes={} alloc_calls_per_iter={:.3} alloc_bytes_per_iter={:.3}"
        ),
        mode,
        iterations,
        elapsed_s,
        iterations as f64 / elapsed_s.max(f64::MIN_POSITIVE),
        counts.alloc_calls,
        counts.alloc_bytes,
        counts.realloc_calls,
        counts.realloc_new_bytes,
        counts.dealloc_calls,
        counts.dealloc_bytes,
        counts.alloc_calls as f64 / iterations as f64,
        counts.alloc_bytes as f64 / iterations as f64,
    );
}

fn profile_loss(model: &Gpt, iterations: usize) {
    let x: Vec<usize> = (0..model.cfg.block).map(|i| i % model.cfg.vocab).collect();
    let y: Vec<usize> = (0..model.cfg.block)
        .map(|i| (i + 1) % model.cfg.vocab)
        .collect();

    black_box(model.loss(&x, &y));
    reset();
    let started = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iterations {
        checksum += black_box(model.loss(black_box(&x), black_box(&y)));
    }
    let elapsed_s = started.elapsed().as_secs_f64();
    let counts = snapshot();
    black_box(checksum);
    print_snapshot("loss", iterations, elapsed_s, counts);
}

fn profile_generate(model: &Gpt, iterations: usize) {
    let prefix_len = model.cfg.block.min(16);
    let prefix: Vec<usize> = (0..prefix_len).map(|i| i % model.cfg.vocab).collect();
    let mut ids = Vec::with_capacity(prefix_len + 1);
    let mut rng = StdRng::seed_from_u64(0xA11CE_2702);

    ids.extend_from_slice(&prefix);
    model.generate(&mut ids, 1, 0.0, &mut rng);

    reset();
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        ids.clear();
        ids.extend_from_slice(&prefix);
        model.generate(black_box(&mut ids), 1, 0.0, black_box(&mut rng));
        checksum ^= ids[prefix_len];
    }
    let elapsed_s = started.elapsed().as_secs_f64();
    let counts = snapshot();
    black_box(checksum);
    print_snapshot("generate_one", iterations, elapsed_s, counts);
}

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    if iterations == 0 {
        eprintln!("iterations must be positive");
        std::process::exit(2);
    }

    let model = make_model();
    profile_loss(&model, iterations);
    profile_generate(&model, iterations);
}
