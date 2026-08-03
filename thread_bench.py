#!/usr/bin/env python3
"""A/B benchmark for graph.rs's CPU thread pinning.

Toggles the `ggml_backend_cpu_set_n_threads` pinned-block in graph.rs in and
out, rebuilding and running N times per configuration, and reports mean
decode tok/s for pinned vs. default thread count. This is what produced the
~6-9% figure cited in graph.rs's comment and the README.

Usage: python3 thread_bench.py [n_runs]   (default 5 runs per configuration)

MODEL below must point at a real GGUF file on whatever machine this runs
on — override with the QFZ3_BENCH_MODEL env var if ~/qfz3/ai_playground/
doesn't exist or doesn't have this file locally.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

GRAPH_RS = Path.home() / "qfz3/qfz3-engine/src/graph.rs"
MODEL = os.environ.get(
    "QFZ3_BENCH_MODEL",
    str(Path.home() / "qfz3/ai_playground/qwen2.5-coder-1.5b-q4_K_M.gguf"),
)
PROMPT = "explain how photosynthesis works in three sentences"
COOLDOWN_SECONDS = 5

ANCHOR = "        if backend.is_null() {\n            bail!(\"[Z.1 Graph] ggml_backend_cpu_init failed\");\n        }"
PINNED_BLOCK = (
    "\n        unsafe { ffi::ggml_backend_cpu_set_n_threads(backend, 2) };\n"
    "        log::info!(\"[Z.1 Graph] CPU backend threads pinned to 2 (physical cores)\");"
)


def set_pinned(enabled: bool):
    src = GRAPH_RS.read_text()
    if PINNED_BLOCK in src:
        src = src.replace(PINNED_BLOCK, "")
    if enabled:
        assert src.count(ANCHOR) == 1, f"anchor count: {src.count(ANCHOR)}"
        src = src.replace(ANCHOR, ANCHOR + PINNED_BLOCK)
    GRAPH_RS.write_text(src)


def build():
    r = subprocess.run(["cargo", "build", "--release"], cwd=Path.home() / "qfz3",
                        capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED:")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(1)


def run_once():
    r = subprocess.run(
        ["cargo", "run", "--release", "--bin", "qfz3-engine", "--",
         "--prompt", PROMPT, "--model", MODEL],
        cwd=Path.home() / "qfz3", capture_output=True, text=True, timeout=120,
    )
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) prompt tokens, (\d+) generated \((\d+)ms prefill, (\d+)ms decode\)", out)
    if not m:
        print("Could not parse output:")
        print(out[-1500:])
        return None
    prompt_tok, gen_tok, prefill_ms, decode_ms = (int(x) for x in m.groups())
    tok_per_sec = gen_tok / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    return {"generated": gen_tok, "decode_ms": decode_ms, "tok_per_sec": tok_per_sec}


def bench_config(label, pinned, n_runs):
    print(f"\n=== {label} ({n_runs} runs) ===")
    set_pinned(pinned)
    build()
    results = []
    for i in range(1, n_runs + 1):
        print(f"  run {i}/{n_runs}...", end=" ", flush=True)
        res = run_once()
        if res is None:
            print("FAILED (skipped)")
            continue
        print(f"{res['generated']} tok, {res['decode_ms']}ms decode, {res['tok_per_sec']:.2f} tok/s")
        results.append(res)
        if i < n_runs:
            time.sleep(COOLDOWN_SECONDS)
    return results


def summarize(label, results):
    if not results:
        print(f"{label}: no successful runs")
        return None
    rates = [r["tok_per_sec"] for r in results]
    mean = sum(rates) / len(rates)
    print(f"\n{label} summary:")
    print(f"  runs: {len(rates)}")
    print(f"  individual tok/s: {[f'{r:.2f}' for r in rates]}")
    print(f"  mean tok/s: {mean:.2f}")
    print(f"  min/max: {min(rates):.2f} / {max(rates):.2f}")
    return mean


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Benchmarking qfz3-engine: {n_runs} runs per configuration")
    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT!r}")

    pinned_results = bench_config("PINNED (2 threads)", pinned=True, n_runs=n_runs)
    default_results = bench_config("DEFAULT (unset)", pinned=False, n_runs=n_runs)

    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    pinned_mean = summarize("PINNED (2 threads)", pinned_results)
    default_mean = summarize("DEFAULT (unset)", default_results)

    if pinned_mean and default_mean:
        diff_pct = (default_mean - pinned_mean) / pinned_mean * 100
        print(f"\nDefault is {diff_pct:+.1f}% vs pinned (positive = default faster)")

    # Pinning is the shipped default (see graph.rs) as of this benchmark's
    # own result, not an experimental flag — restore to *pinned*, not
    # unpinned, so a benchmark run doesn't silently revert it on disk.
    set_pinned(True)
    build()
    print("\ngraph.rs restored to its shipped (pinned) state.")


if __name__ == "__main__":
    main()
