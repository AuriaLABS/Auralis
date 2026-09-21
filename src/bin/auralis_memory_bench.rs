use auralis::memory::{
    ExternalMemory, InMemoryExternalMemory, MemoryClass, MemoryQuery, MemoryReset, MemoryWrite,
};
use std::env;
use std::hint::black_box;
use std::time::Instant;

fn populate(capacity: usize, working_set: usize) -> (InMemoryExternalMemory, usize) {
    let mut memory = InMemoryExternalMemory::new(capacity).expect("valid capacity");
    let mut rejected = 0usize;
    for i in 0..working_set {
        let write = MemoryWrite::new(
            "facts",
            format!("key-{i:04}"),
            vec![i as f32, (i as f32) * 0.5, -(i as f32)],
            MemoryClass::Persistent,
        );
        if memory.write(write).is_err() {
            rejected += 1;
        }
    }
    (memory, rejected)
}

fn exact_retrieval(memory: &InMemoryExternalMemory, working_set: usize) -> usize {
    let mut hits = 0usize;
    for i in 0..working_set {
        let query = MemoryQuery::exact("facts", format!("key-{i:04}"));
        let result = memory.query(&query).expect("valid query");
        if let Some(record) = result.first() {
            if record.value.first().copied() == Some(i as f32) {
                hits += 1;
            }
        }
    }
    hits
}

fn query_latency_ns(
    memory: &InMemoryExternalMemory,
    working_set: usize,
    repeats: usize,
) -> f64 {
    let started = Instant::now();
    let mut observed = 0usize;
    for repeat in 0..repeats {
        for i in 0..working_set {
            let query = MemoryQuery::exact("facts", format!("key-{i:04}"));
            observed = observed.wrapping_add(memory.query(&query).unwrap().len());
        }
        black_box(observed ^ repeat);
    }
    let operations = working_set.saturating_mul(repeats).max(1);
    started.elapsed().as_nanos() as f64 / operations as f64
}

fn contamination_probe() -> usize {
    let mut memory = InMemoryExternalMemory::new(8).unwrap();
    memory
        .write(MemoryWrite::new(
            "session-a",
            "private",
            vec![1.0],
            MemoryClass::Session,
        ))
        .unwrap();
    memory
        .write(MemoryWrite::new(
            "shared",
            "persistent",
            vec![2.0],
            MemoryClass::Persistent,
        ))
        .unwrap();
    memory.reset(&MemoryReset::Session).unwrap();
    memory
        .query(&MemoryQuery::exact("session-a", "private"))
        .unwrap()
        .len()
}

fn persistence_probe() -> (usize, usize, u64) {
    let mut memory = InMemoryExternalMemory::new(8).unwrap();
    memory
        .write(MemoryWrite::new(
            "session",
            "ephemeral",
            vec![1.0],
            MemoryClass::Session,
        ))
        .unwrap();
    memory
        .write(MemoryWrite::new(
            "shared",
            "persistent",
            vec![2.0],
            MemoryClass::Persistent,
        ))
        .unwrap();
    memory
        .write(MemoryWrite::new(
            "weights",
            "trainable",
            vec![3.0],
            MemoryClass::Trainable,
        ))
        .unwrap();
    let encoded = memory.encode().unwrap();
    let restored = InMemoryExternalMemory::decode(&encoded).unwrap();
    (
        encoded.len(),
        restored.len(),
        restored.snapshot_fingerprint().unwrap(),
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let working_set = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(128usize);
    let repeats = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(40usize);
    assert!(working_set >= 128, "working_set must be >= 128");
    assert!(repeats > 0, "repeats must be positive");

    let contamination = contamination_probe();
    let (snapshot_bytes, restored_records, snapshot_fingerprint) = persistence_probe();
    println!(
        "memory_protocol | schema=1 working_set={} repeats={} capacity_policy=reject_new session_persisted=false contamination={} snapshot_bytes={} restored_records={} snapshot_fingerprint={:016x}",
        working_set,
        repeats,
        contamination,
        snapshot_bytes,
        restored_records,
        snapshot_fingerprint,
    );

    for capacity in [8usize, 32, 128] {
        let (memory, rejected) = populate(capacity, working_set);
        let hits = exact_retrieval(&memory, working_set);
        let retrieval = hits as f64 / working_set as f64;
        let latency_ns = query_latency_ns(&memory, working_set, repeats);
        println!(
            "memory_capacity | capacity={} working_set={} stored={} rejected={} exact_hits={} exact_retrieval={:.6} query_ns={:.3} estimated_heap_bytes={} snapshot_bytes={}",
            capacity,
            working_set,
            memory.len(),
            rejected,
            hits,
            retrieval,
            latency_ns,
            memory.estimated_heap_bytes(),
            memory.encode().unwrap().len(),
        );
    }
}
