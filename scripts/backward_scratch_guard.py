#!/usr/bin/env python3
"""Policy for Backward Scratch Comparison.

Automatic PR runs are a no-regression guard. A PR that intentionally claims a
scratch optimization must put the exact commit trailer
"Auralis-Scratch-Optimization: true" on its HEAD commit; only then are the
historical material-improvement thresholds applied.
"""

from __future__ import annotations

import argparse


def evaluate_guard(
    *,
    base_calls: int,
    candidate_calls: int,
    base_bytes: int,
    candidate_bytes: int,
    base_reallocs: int,
    candidate_reallocs: int,
    base_realloc_bytes: int,
    candidate_realloc_bytes: int,
    base_tps: float,
    candidate_tps: float,
    base_rss_kib: float,
    candidate_rss_kib: float,
    require_improvement: bool,
) -> str:
    if candidate_calls > base_calls:
        raise ValueError(
            f"allocation calls regressed: {base_calls} -> {candidate_calls}"
        )
    if candidate_bytes > base_bytes:
        raise ValueError(
            f"allocation bytes regressed: {base_bytes} -> {candidate_bytes}"
        )
    if candidate_reallocs > base_reallocs:
        raise ValueError(
            f"reallocation calls regressed: {base_reallocs} -> {candidate_reallocs}"
        )
    if candidate_realloc_bytes > base_realloc_bytes:
        raise ValueError(
            "reallocation bytes regressed: "
            f"{base_realloc_bytes} -> {candidate_realloc_bytes}"
        )

    call_reduction = 1.0 - candidate_calls / base_calls if base_calls else 0.0
    byte_reduction = 1.0 - candidate_bytes / base_bytes if base_bytes else 0.0

    if require_improvement:
        if call_reduction < 0.25:
            raise ValueError(
                f"scratch optimization call reduction is not material: "
                f"{call_reduction:.2%}"
            )
        if byte_reduction < 0.20:
            raise ValueError(
                f"scratch optimization byte reduction is not material: "
                f"{byte_reduction:.2%}"
            )

    if base_tps > 0 and candidate_tps < base_tps * 0.85:
        raise ValueError(
            f"throughput regressed materially: {base_tps:.3f} -> "
            f"{candidate_tps:.3f} tok/s"
        )

    if base_rss_kib > 0 and candidate_rss_kib > base_rss_kib * 1.10 + 1024:
        raise ValueError(
            f"peak RSS regressed materially: {base_rss_kib:.0f} -> "
            f"{candidate_rss_kib:.0f} KiB"
        )

    if require_improvement:
        return "material-improvement-pass"
    if candidate_calls == base_calls and candidate_bytes == base_bytes:
        return "neutral-no-regression-pass"
    return "improvement-no-regression-pass"


def self_test() -> None:
    common = dict(
        base_calls=136,
        base_bytes=728_256,
        base_reallocs=0,
        base_realloc_bytes=0,
        base_tps=45_000.0,
        base_rss_kib=4_860.0,
    )

    assert evaluate_guard(
        **common,
        candidate_calls=136,
        candidate_bytes=728_256,
        candidate_reallocs=0,
        candidate_realloc_bytes=0,
        candidate_tps=44_990.0,
        candidate_rss_kib=4_828.0,
        require_improvement=False,
    ) == "neutral-no-regression-pass"

    assert evaluate_guard(
        **common,
        candidate_calls=90,
        candidate_bytes=500_000,
        candidate_reallocs=0,
        candidate_realloc_bytes=0,
        candidate_tps=46_000.0,
        candidate_rss_kib=4_700.0,
        require_improvement=True,
    ) == "material-improvement-pass"

    failing = [
        dict(
            candidate_calls=137,
            candidate_bytes=728_256,
            candidate_reallocs=0,
            candidate_realloc_bytes=0,
            candidate_tps=45_000.0,
            candidate_rss_kib=4_860.0,
            require_improvement=False,
        ),
        dict(
            candidate_calls=120,
            candidate_bytes=650_000,
            candidate_reallocs=0,
            candidate_realloc_bytes=0,
            candidate_tps=45_000.0,
            candidate_rss_kib=4_860.0,
            require_improvement=True,
        ),
        dict(
            candidate_calls=136,
            candidate_bytes=728_256,
            candidate_reallocs=0,
            candidate_realloc_bytes=0,
            candidate_tps=37_000.0,
            candidate_rss_kib=4_860.0,
            require_improvement=False,
        ),
        dict(
            candidate_calls=136,
            candidate_bytes=728_256,
            candidate_reallocs=0,
            candidate_realloc_bytes=0,
            candidate_tps=45_000.0,
            candidate_rss_kib=6_500.0,
            require_improvement=False,
        ),
    ]
    for case in failing:
        try:
            evaluate_guard(**common, **case)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected guard failure: {case}")

    print("backward_scratch_guard_self_test | ok=true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported from CLI")
    self_test()


if __name__ == "__main__":
    main()
