#!/usr/bin/env python
"""Determinism checker — confirm whether non-determinism comes from PYTHONHASHSEED.

Runs the SAME payload several times, each in a FRESH subprocess (so each gets its
own hash seed when PYTHONHASHSEED is unset — in-process replays share one seed and
cannot expose hash-order drift), under two conditions:

  * PYTHONHASHSEED unset  → random per process  (production-local default)
  * PYTHONHASHSEED=0      → the determinism requirement (what the Dockerfile sets)

It hashes the knitting assignments and the full schedule.  Verdict:
  - unset DIVERGES but =0 IDENTICAL  → confirmed: it's the missing hash seed.
  - both IDENTICAL                   → determinism is fine here; look elsewhere.
  - =0 also DIVERGES                 → a non-hashseed source (report it).

Usage:
    python tools/det_check.py path/to/payload.json [N=4]

The payload JSON is the dict passed to Engine(...).solve() — i.e. the same body
your API receives (keys: config, machines, resources, tasks, reschedule_hint,
material_capacities).
"""
import sys
import os
import json
import hashlib
import subprocess


def _hash_rows(assignments, keep_ids=None):
    rows = sorted(
        (a["task_id"], a.get("machine_id"), a.get("start_time"),
         a.get("end_time"), a.get("batch_slot_id", ""))
        for a in assignments
        if keep_ids is None or a["task_id"] in keep_ids
    )
    return hashlib.md5(repr(rows).encode()).hexdigest()


def _worker(payload_path):
    """Solve once; print a JSON line of hashes."""
    import logging
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, os.getcwd())
    from app.engine.model import Engine

    with open(payload_path) as fh:
        payload = json.load(fh)

    knit_ids = {
        t["task_id"] for t in payload.get("tasks", [])
        if str(t.get("operation", "")).lower() == "knitting"
    }
    result = Engine(payload).solve()
    assigns = result.get("assignments", [])
    print(json.dumps({
        "status": result.get("status"),
        "n": len(assigns),
        "knitting": _hash_rows(assigns, knit_ids),
        "full": _hash_rows(assigns),
    }))


def _spawn(payload_path, hashseed):
    env = dict(os.environ)
    if hashseed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(hashseed)
    env["_DET_CHECK_WORKER"] = "1"
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), payload_path],
        capture_output=True, text=True, env=env,
    )
    line = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
    if not line:
        return {"status": "ERROR", "stderr": out.stderr[-500:]}
    return json.loads(line[-1])


def main():
    if os.environ.get("_DET_CHECK_WORKER") == "1":
        _worker(sys.argv[1])
        return

    payload_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    for label, seed in (("PYTHONHASHSEED unset", None), ("PYTHONHASHSEED=0", 0)):
        print(f"\n=== {label} — {n} runs ===")
        results = [_spawn(payload_path, seed) for _ in range(n)]
        for i, r in enumerate(results, 1):
            print(f"  run {i}: status={r.get('status'):>10}  "
                  f"knitting={r.get('knitting')}  full={r.get('full')}")
        kset = {r.get("knitting") for r in results}
        fset = {r.get("full") for r in results}
        print(f"  -> knitting {'IDENTICAL' if len(kset)==1 else f'DIVERGES ({len(kset)} distinct)'}"
              f" | full {'IDENTICAL' if len(fset)==1 else f'DIVERGES ({len(fset)} distinct)'}")


if __name__ == "__main__":
    main()
