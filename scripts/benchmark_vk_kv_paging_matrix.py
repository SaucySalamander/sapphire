#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ELAPSED_RE = re.compile(r"perform_inference: elapsed=([0-9]*\.?[0-9]+) sec")
PAGER_RE = re.compile(
    r"VK KV pager stats: hits=(\d+) misses=(\d+) hit_rate=([0-9]*\.?[0-9]+)%"
)


@dataclass
class RunSample:
    elapsed_s: float
    latency_ms_per_token: float
    hit_rate_pct: Optional[float]
    hits: int
    misses: int


@dataclass
class MatrixCase:
    name: str
    paging_enabled: bool
    page_tokens: int
    vram_budget_mb: int


def parse_csv_ints(value: str) -> List[int]:
    out: List[int] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("list cannot be empty")
    return out


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def run_once(binary: Path,
             model: str,
             prompt: str,
             max_tokens: int,
             context_len: int,
             case: MatrixCase,
             base_env: dict) -> RunSample:
    env = dict(base_env)
    env["SAPPHIRE_BACKEND"] = "vulkan"
    env["SAPPHIRE_VK_KV_PAGING"] = "1" if case.paging_enabled else "0"
    if case.paging_enabled:
        env["SAPPHIRE_KV_PAGE_TOKENS"] = str(case.page_tokens)
        env["SAPPHIRE_KV_VRAM_BUDGET_MB"] = str(case.vram_budget_mb)

    cmd = [
        str(binary),
        "-m", model,
        "-t", "0.0",
        "-c", str(context_len),
        "-p", prompt,
        "-n", str(max_tokens),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed for case={case.name} rc={proc.returncode}\n{output}"
        )

    elapsed_match = ELAPSED_RE.search(output)
    if not elapsed_match:
        raise RuntimeError(
            f"Could not parse elapsed time for case={case.name}.\n{output}"
        )
    elapsed_s = float(elapsed_match.group(1))

    hit_rate_pct: Optional[float] = None
    hits = 0
    misses = 0
    pager_match = PAGER_RE.search(output)
    if pager_match:
        hits = int(pager_match.group(1))
        misses = int(pager_match.group(2))
        hit_rate_pct = float(pager_match.group(3))

    latency_ms_per_token = (elapsed_s * 1000.0) / float(max_tokens)
    return RunSample(
        elapsed_s=elapsed_s,
        latency_ms_per_token=latency_ms_per_token,
        hit_rate_pct=hit_rate_pct,
        hits=hits,
        misses=misses,
    )


def build_cases(page_tokens: List[int], vram_budgets: List[int]) -> List[MatrixCase]:
    cases: List[MatrixCase] = [
        MatrixCase(
            name="baseline_no_paging",
            paging_enabled=False,
            page_tokens=page_tokens[0],
            vram_budget_mb=vram_budgets[0],
        )
    ]
    for tokens in page_tokens:
        for vram in vram_budgets:
            cases.append(
                MatrixCase(
                    name=f"paging_pt{tokens}_vram{vram}",
                    paging_enabled=True,
                    page_tokens=tokens,
                    vram_budget_mb=vram,
                )
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Vulkan KV paging perf matrix (hit-rate + p50/p95/p99 latency)."
    )
    parser.add_argument("--binary", default="./out/sapphire")
    parser.add_argument("--model", default="gemma-3-270m-it")
    parser.add_argument("--prompt", default="Summarize this benchmark run in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--page-tokens", default="64,128")
    parser.add_argument("--vram-budgets", default="2048,1024,512")
    parser.add_argument("--out-csv", default="reports/vk_kv_paging_matrix.csv")
    parser.add_argument("--out-md", default="reports/vk_kv_paging_matrix.md")
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.runs <= 0 or args.context <= 0:
        raise ValueError("--max-tokens, --runs, and --context must be > 0")

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}. Build first with `make bin`.")

    page_tokens = parse_csv_ints(args.page_tokens)
    vram_budgets = parse_csv_ints(args.vram_budgets)
    cases = build_cases(page_tokens, vram_budgets)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    rows = []

    for case in cases:
        samples: List[RunSample] = []
        for run_idx in range(args.runs):
            sample = run_once(
                binary=binary,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                context_len=args.context,
                case=case,
                base_env=base_env,
            )
            samples.append(sample)
            print(
                f"[{case.name}] run {run_idx + 1}/{args.runs}: "
                f"elapsed={sample.elapsed_s:.3f}s "
                f"latency={sample.latency_ms_per_token:.3f}ms/token"
            )

        elapsed_values = [s.elapsed_s for s in samples]
        latency_values = [s.latency_ms_per_token for s in samples]
        hit_rates = [s.hit_rate_pct for s in samples if s.hit_rate_pct is not None]

        row = {
            "case": case.name,
            "paging": int(case.paging_enabled),
            "page_tokens": case.page_tokens,
            "vram_budget_mb": case.vram_budget_mb,
            "runs": len(samples),
            "elapsed_p50_s": percentile(elapsed_values, 0.50),
            "elapsed_p95_s": percentile(elapsed_values, 0.95),
            "elapsed_p99_s": percentile(elapsed_values, 0.99),
            "latency_p50_ms_per_token": percentile(latency_values, 0.50),
            "latency_p95_ms_per_token": percentile(latency_values, 0.95),
            "latency_p99_ms_per_token": percentile(latency_values, 0.99),
            "hit_rate_mean_pct": statistics.mean(hit_rates) if hit_rates else float("nan"),
            "hit_rate_p50_pct": percentile(hit_rates, 0.50) if hit_rates else float("nan"),
            "hits_total": sum(s.hits for s in samples),
            "misses_total": sum(s.misses for s in samples),
        }
        rows.append(row)

    fieldnames = [
        "case",
        "paging",
        "page_tokens",
        "vram_budget_mb",
        "runs",
        "elapsed_p50_s",
        "elapsed_p95_s",
        "elapsed_p99_s",
        "latency_p50_ms_per_token",
        "latency_p95_ms_per_token",
        "latency_p99_ms_per_token",
        "hit_rate_mean_pct",
        "hit_rate_p50_pct",
        "hits_total",
        "misses_total",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Vulkan KV Paging Perf Matrix\n\n")
        f.write("Generated by `scripts/benchmark_vk_kv_paging_matrix.py`.\n\n")
        f.write("| Case | Paging | Page Tokens | VRAM MB | Runs | p50 ms/token | p95 ms/token | p99 ms/token | Mean hit-rate % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {case} | {paging} | {page_tokens} | {vram_budget_mb} | {runs} | {latency_p50_ms_per_token:.3f} | {latency_p95_ms_per_token:.3f} | {latency_p99_ms_per_token:.3f} | {hit_rate_mean_pct:.2f} |\n".format(
                    **row
                )
            )

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
