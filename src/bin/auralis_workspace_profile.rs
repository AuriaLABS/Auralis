use auralis::model::{Config, Gpt};
use auralis::optim::Adam;
use auralis::training::{train_step_reuse, TrainConfig, TrainWorkspace};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
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
struct Counts {
    alloc_calls: u64,
    alloc_bytes: u64,
    realloc_calls: u64,
    realloc_new_bytes: u64,
    dealloc_calls: u64,
    dealloc_bytes: u64,
}

fn reset_counts() {
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

fn counts() -> Counts {
    Counts {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
        realloc_new_bytes: REALLOC_NEW_BYTES.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn config(block: usize) -> Config {
    Config {
        vocab: 100,
        n_embd: 32,
        n_head: 4,
        n_layer: 2,
        block,
        n_ff: 96,
    }
}

fn model(block: usize, seed: u64) -> Gpt {
    let mut rng = StdRng::seed_from_u64(seed);
    Gpt::new(config(block), &mut rng)
}

fn tokens(len: usize) -> Vec<usize> {
    (0..len).map(|i| (i * 37 + i / 3 + 11) % 100).collect()
}

fn peak_rss_kib() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return 0;
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(value) = rest.split_whitespace().next() {
                    if let Ok(kib) = value.parse() {
                        return kib;
                    }
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

fn print_profile(
    phase: &str,
    block: usize,
    elapsed_s: f64,
    c: Counts,
    work_tokens: usize,
) {
    let peak_rss_kib = peak_rss_kib();
    let tok_per_s = work_tokens as f64 / elapsed_s.max(f64::MIN_POSITIVE);
    println!(
        concat!(
            "workspace_profile | phase={} block={} seconds={:.9} ",
            "alloc_calls={} alloc_bytes={} realloc_calls={} realloc_new_bytes={} ",
            "dealloc_calls={} dealloc_bytes={} work_tokens={} tok_per_s={:.3} peak_rss_kib={}"
        ),
        phase,
        block,
        elapsed_s,
        c.alloc_calls,
        c.alloc_bytes,
        c.realloc_calls,
        c.realloc_new_bytes,
        c.dealloc_calls,
        c.dealloc_bytes,
        work_tokens,
        tok_per_s,
        peak_rss_kib,
    );
}

fn profile_loss(block: usize) {
    let gpt = model(block, 0xA11CE_2801 + block as u64);
    let x = tokens(block);
    let y = tokens(block + 1)[1..].to_vec();

    let warm = gpt.loss(&x, &y);
    assert!(warm.is_finite());

    reset_counts();
    let started = Instant::now();
    let loss = gpt.loss(&x, &y);
    let elapsed = started.elapsed().as_secs_f64();
    let c = counts();
    assert!(loss.is_finite());
    print_profile("loss", block, elapsed, c, block);
}

fn profile_backward(block: usize) {
    let gpt = model(block, 0xA11CE_2802 + block as u64);
    let x = tokens(block);
    let y = tokens(block + 1)[1..].to_vec();
    let param_count = gpt.collect_params().len();
    let mut grads = vec![0.0f32; param_count];

    let warm = gpt.backward_into(&x, &y, &mut grads);
    assert!(warm.is_finite());
    grads.fill(0.0);

    reset_counts();
    let started = Instant::now();
    let loss = gpt.backward_into(&x, &y, &mut grads);
    let elapsed = started.elapsed().as_secs_f64();
    let c = counts();
    assert!(loss.is_finite());
    assert!(grads.iter().all(|v| v.is_finite()));
    print_profile("backward", block, elapsed, c, block);
}

fn profile_train_reuse(block: usize) {
    let cfg = TrainConfig {
        seed: 659_918,
        batch_size: 2,
        gradient_accumulation_steps: 2,
        grad_clip_norm: 1.0,
    };
    let stream = tokens(block * 24 + 17);

    let mut warm_model = model(block, 0xA11CE_2803 + block as u64);
    let warm_n = warm_model.collect_params().len();
    let mut warm_adam = Adam::new(warm_n, 3e-3);
    let mut warm_grads = vec![0.0f32; warm_n];
    let mut warm_workspace = TrainWorkspace::new(&warm_model);
    train_step_reuse(
        &mut warm_model,
        &mut warm_adam,
        &stream,
        cfg,
        0,
        &mut warm_grads,
        &mut warm_workspace,
    )
    .expect("warm train step");

    let mut gpt = model(block, 0xA11CE_2803 + block as u64);
    let n = gpt.collect_params().len();
    let mut adam = Adam::new(n, 3e-3);
    let mut grads = vec![0.0f32; n];
    let mut workspace = TrainWorkspace::new(&gpt);

    reset_counts();
    let started = Instant::now();
    let metrics = train_step_reuse(
        &mut gpt,
        &mut adam,
        &stream,
        cfg,
        0,
        &mut grads,
        &mut workspace,
    )
    .expect("profile train step");
    let elapsed = started.elapsed().as_secs_f64();
    let c = counts();
    assert!(metrics.loss.is_finite());
    print_profile("train_reuse", block, elapsed, c, metrics.tokens);
}

fn run_one(phase: &str, block: usize) {
    match phase {
        "loss" => profile_loss(block),
        "backward" => profile_backward(block),
        "train_reuse" => profile_train_reuse(block),
        other => panic!("unknown workspace profile phase: {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let phase = args.get(2).expect("missing child phase");
        let block: usize = args
            .get(3)
            .expect("missing child block")
            .parse()
            .expect("invalid child block");
        run_one(phase, block);
        return;
    }

    let exe = std::env::current_exe().expect("resolve workspace profiler executable");
    for block in [4usize, 8, 16, 32] {
        for phase in ["loss", "backward", "train_reuse"] {
            let output = Command::new(&exe)
                .args(["--child", phase, &block.to_string()])
                .output()
                .expect("spawn isolated workspace profiler child");
            assert!(
                output.status.success(),
                "workspace profiler child failed for phase={phase} block={block}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
    }
}
