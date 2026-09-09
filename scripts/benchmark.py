"""
Benchmark harness for the CP-SAT knitting-CRP solver.

Runs 3-iteration timing at 200 / 500 / 1000 tasks using synthetic payloads
identical in structure to the production Go payload, then reports:
  - mean and p95 solve time
  - peak RSS memory
  - solve status
  - factory load ratio

Usage:
    python scripts/benchmark.py                  # all three scales
    python scripts/benchmark.py --scales 200     # single scale
    python scripts/benchmark.py --runs 5         # more iterations

Targets from TechDesign:
    200 tasks  → ≤ 60s, ≤ 500 MB
    500 tasks  → ≤ 180s, ≤ 1.5 GB
    1000 tasks → ≤ max_search_time, ≤ 6 GB
"""
import argparse
import copy
import resource
import statistics
import sys
import time
from pathlib import Path

# Allow `from app.engine...` imports when run from the project root or scripts/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.engine.model import Engine  # noqa: E402
from tests.conftest import make_payload  # noqa: E402

# ── Targets (TechDesign §Performance Benchmarking) ────────────────────────────
TARGETS = {
    200:  {"max_time_s": 60,           "max_ram_mb": 500},
    500:  {"max_time_s": 180,          "max_ram_mb": 1_500},
    1000: {"max_time_s": float("inf"), "max_ram_mb": 6_000},  # best-effort
}

_PASS = "✅"
_FAIL = "❌"
_WARN = "⚠️ "


def _rss_mb() -> float:
    """Current process peak RSS in MB (Linux getrusage returns KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _make_bench_payload(n_tasks: int, max_search_time: int) -> dict:
    """
    Build a benchmark payload.  n_tasks must be even; each order produces
    1 knitting + 1 linking task, so n_orders = n_tasks // 2.
    """
    n_orders = max(1, n_tasks // 2)
    n_km = max(1, n_orders // 5)   # ~1 machine per 5 orders (realistic density)
    n_lm = max(1, n_orders // 20)
    return make_payload(
        n_orders=n_orders,
        n_knitting_machines=n_km,
        n_linking_machines=n_lm,
        max_factory_machines=n_km,
        horizon_minutes=10080,     # 1 week
        max_search_time=max_search_time,
        num_search_workers=8,      # production mode (not deterministic)
        random_seed=42,
    )


def run_scale(n_tasks: int, n_runs: int, max_search_time: int) -> dict:
    """Run n_runs solves for a given task count; return timing + status stats."""
    base_payload = _make_bench_payload(n_tasks, max_search_time)

    times: list[float] = []
    statuses: list[str] = []
    ram_before = _rss_mb()

    for i in range(n_runs):
        payload = copy.deepcopy(base_payload)
        t0 = time.perf_counter()
        result = Engine(payload).solve()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        statuses.append(result.get("status", "unknown"))
        print(f"  run {i+1}/{n_runs}: {elapsed:.1f}s  status={statuses[-1]}")

    ram_after = _rss_mb()
    return {
        "n_tasks":  n_tasks,
        "n_runs":   n_runs,
        "mean_s":   statistics.mean(times),
        "p95_s":    sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[-1],
        "max_s":    max(times),
        "statuses": statuses,
        "ram_delta_mb": ram_after - ram_before,
    }


def _check(label: str, value: float, target: float, unit: str) -> str:
    icon = _PASS if value <= target else _FAIL
    return f"{icon} {label}: {value:.1f}{unit}  (target ≤ {target}{unit})"


def print_report(stats: dict) -> None:
    n = stats["n_tasks"]
    tgt = TARGETS.get(n, {})
    feasible = sum(1 for s in stats["statuses"] if s in ("feasible", "optimal"))

    print(f"\n{'─'*60}")
    print(f"  n={n} tasks  ({stats['n_runs']} runs)")
    print(f"{'─'*60}")
    print(f"  mean={stats['mean_s']:.1f}s  "
          f"p95={stats['p95_s']:.1f}s  "
          f"max={stats['max_s']:.1f}s")
    print(f"  RAM delta: {stats['ram_delta_mb']:.0f} MB")
    print(f"  feasible/optimal: {feasible}/{stats['n_runs']}")
    if tgt:
        print(_check("  solve time (mean)", stats["mean_s"], tgt["max_time_s"], "s"))
        print(_check("  RAM delta        ", stats["ram_delta_mb"], tgt["max_ram_mb"], " MB"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CP-SAT solver benchmark")
    parser.add_argument("--scales", nargs="+", type=int,
                        default=[200, 500, 1000],
                        help="Task counts to benchmark (default: 200 500 1000)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed runs per scale (default: 3)")
    parser.add_argument("--max-search-time", type=int, default=60,
                        help="Solver time limit in seconds (default: 60)")
    args = parser.parse_args()

    print(f"Benchmark: scales={args.scales}  runs={args.runs}  "
          f"max_search_time={args.max_search_time}s")

    all_stats = []
    for n in args.scales:
        print(f"\n▶ n={n} tasks …")
        stats = run_scale(n, args.runs, args.max_search_time)
        print_report(stats)
        all_stats.append(stats)

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    print(f"{'Tasks':>6}  {'Mean':>7}  {'P95':>7}  {'RAM Δ':>8}  Status")
    print(f"{'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*20}")
    for s in all_stats:
        feasible = sum(1 for st in s["statuses"] if st in ("feasible", "optimal"))
        status_str = f"{feasible}/{s['n_runs']} ok"
        tgt = TARGETS.get(s["n_tasks"], {})
        time_ok = _PASS if not tgt or s["mean_s"] <= tgt["max_time_s"] else _FAIL
        print(f"{s['n_tasks']:>6}  {s['mean_s']:>6.1f}s  "
              f"{s['p95_s']:>6.1f}s  {s['ram_delta_mb']:>7.0f}MB  "
              f"{time_ok} {status_str}")


if __name__ == "__main__":
    main()
