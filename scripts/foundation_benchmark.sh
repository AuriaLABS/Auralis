#!/usr/bin/env bash
set -euo pipefail

# Reproducible Foundation CPU benchmark.
#
# The training workload itself is deterministic. Wall-clock speed and RSS are
# measured as regression signals with deliberately broad gates because hosted
# CI hardware is shared and can vary between runs.

STEPS="${AURALIS_BENCH_STEPS:-20}"
REPEATS="${AURALIS_BENCH_REPEATS:-3}"
SEED="${AURALIS_BENCH_SEED:-659918}"
BATCH="${AURALIS_BENCH_BATCH:-2}"
ACCUM="${AURALIS_BENCH_ACCUM:-2}"
BASELINE_FILE="${AURALIS_BENCH_BASELINE:-benchmarks/foundation-baseline.env}"
BIN="${AURALIS_BENCH_BIN:-target/release/auralis}"

if [[ ! -x "$BIN" ]]; then
  echo "benchmark binary not found: $BIN" >&2
  echo "build it first with: cargo build --release" >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "/usr/bin/time is required to measure peak RSS" >&2
  exit 2
fi
if (( REPEATS < 3 )); then
  echo "AURALIS_BENCH_REPEATS must be at least 3" >&2
  exit 2
fi

field() {
  local line="$1"
  local key="$2"
  awk -v key="$key" '{
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == key) { print a[2]; exit }
    }
  }' <<< "$line"
}

float_le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'; }
float_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

declare -a speeds=()
declare -a rss_values=()
reference_fingerprint=""
reference_validation=""
reference_test=""
reference_checkpoint_bytes=""

for ((i = 1; i <= REPEATS; i++)); do
  ckpt="$workdir/run-${i}.bin"
  out="$workdir/run-${i}.out"
  mem="$workdir/run-${i}.rss"

  /usr/bin/time -f '%M' -o "$mem" \
    "$BIN" train-fresh "$STEPS" "$ckpt" "$SEED" "$BATCH" "$ACCUM" \
    > "$out" 2>&1

  summary="$(grep '^run_summary |' "$out" | tail -n 1)"
  if [[ -z "$summary" ]]; then
    cat "$out" >&2
    echo "benchmark run ${i} did not emit run_summary" >&2
    exit 1
  fi

  speed="$(field "$summary" tok_per_s)"
  validation="$(field "$summary" validation_loss)"
  test_loss="$(field "$summary" test_loss)"
  checkpoint_bytes="$(field "$summary" checkpoint_bytes)"
  rss="$(tr -d '[:space:]' < "$mem")"
  fingerprint="$(awk -F= '$1 == "parameter_fingerprint" { print $2 }' "$ckpt.manifest")"

  [[ -n "$speed" && -n "$validation" && -n "$test_loss" && -n "$checkpoint_bytes" ]] || {
    echo "benchmark run ${i} is missing summary fields" >&2
    exit 1
  }
  [[ "$rss" =~ ^[0-9]+$ ]] || { echo "invalid RSS reading: $rss" >&2; exit 1; }
  [[ -n "$fingerprint" ]] || { echo "manifest is missing parameter_fingerprint" >&2; exit 1; }

  if [[ -z "$reference_fingerprint" ]]; then
    reference_fingerprint="$fingerprint"
    reference_validation="$validation"
    reference_test="$test_loss"
    reference_checkpoint_bytes="$checkpoint_bytes"
  else
    [[ "$fingerprint" == "$reference_fingerprint" ]] || {
      echo "determinism regression: parameter fingerprint differs on run ${i}" >&2
      exit 1
    }
    [[ "$validation" == "$reference_validation" ]] || {
      echo "determinism regression: validation loss differs on run ${i}" >&2
      exit 1
    }
    [[ "$test_loss" == "$reference_test" ]] || {
      echo "determinism regression: test loss differs on run ${i}" >&2
      exit 1
    }
    [[ "$checkpoint_bytes" == "$reference_checkpoint_bytes" ]] || {
      echo "determinism regression: checkpoint size differs on run ${i}" >&2
      exit 1
    }
  fi

  speeds+=("$speed")
  rss_values+=("$rss")
  printf 'benchmark_run | repetition=%d tok_per_s=%s max_rss_kb=%s validation_loss=%s test_loss=%s checkpoint_bytes=%s fingerprint=%s\n' \
    "$i" "$speed" "$rss" "$validation" "$test_loss" "$checkpoint_bytes" "$fingerprint"
done

median_index=$(( (REPEATS + 1) / 2 ))
median_speed="$(printf '%s\n' "${speeds[@]}" | sort -n | sed -n "${median_index}p")"
max_rss="$(printf '%s\n' "${rss_values[@]}" | sort -nr | head -n 1)"

if [[ -f "$BASELINE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$BASELINE_FILE"

  float_le "$reference_validation" "$MAX_VALIDATION_LOSS" || {
    echo "quality regression: validation_loss=$reference_validation > $MAX_VALIDATION_LOSS" >&2
    exit 1
  }
  float_le "$reference_test" "$MAX_TEST_LOSS" || {
    echo "quality regression: test_loss=$reference_test > $MAX_TEST_LOSS" >&2
    exit 1
  }
  float_ge "$median_speed" "$MIN_MEDIAN_TOK_PER_S" || {
    echo "speed regression: median_tok_per_s=$median_speed < $MIN_MEDIAN_TOK_PER_S" >&2
    exit 1
  }
  (( max_rss <= MAX_RSS_KB )) || {
    echo "memory regression: max_rss_kb=$max_rss > $MAX_RSS_KB" >&2
    exit 1
  }
  (( reference_checkpoint_bytes <= MAX_CHECKPOINT_BYTES )) || {
    echo "artifact regression: checkpoint_bytes=$reference_checkpoint_bytes > $MAX_CHECKPOINT_BYTES" >&2
    exit 1
  }
fi

printf 'benchmark_summary | steps=%s repeats=%s seed=%s batch=%s accum=%s effective_batch=%s median_tok_per_s=%s max_rss_kb=%s validation_loss=%s test_loss=%s checkpoint_bytes=%s fingerprint=%s\n' \
  "$STEPS" "$REPEATS" "$SEED" "$BATCH" "$ACCUM" "$((BATCH * ACCUM))" \
  "$median_speed" "$max_rss" "$reference_validation" "$reference_test" \
  "$reference_checkpoint_bytes" "$reference_fingerprint"
