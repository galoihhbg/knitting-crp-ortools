"""
Regression guard for determinism leg 3: INPUT TASK ORDER must not change output.

Measured root cause (det_study/perm.py): build_resource_model creates CP-SAT
vars/intervals/constraints in task-list order, and the fixed-seed search keys off
that order.  Without a stable ingest sort, shuffling `payload["tasks"]` (which Go
may do via map iteration) moved 2–8/90 tasks → "B trôi" with no input change.

Fix (app/engine/pipeline.py, Pipeline.__init__): sort tasks by the STABLE key
`task_id` at ingest.  These tests assert the fix holds: for one payload, ANY
permutation of the task list yields a BYTE-IDENTICAL schedule.

If someone later reorders the ingest, removes the sort, or sorts by a non-stable
key (due/priority — which churn on re-schedule), these go red immediately.

NOTE: PYTHONHASHSEED is constant within a single pytest process, so the only
varying input between the two solves here is task order — exactly what we test.
"""
import copy
import hashlib
import json
import random

import pytest

from app.engine.model import Engine
from tests.conftest import make_payload


def _canonical(result):
    rows = sorted(
        [a["task_id"], a["machine_id"], int(a["start_time"]), int(a["end_time"])]
        for a in result.get("assignments", [])
    )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def contended_payload():
    """Maximally SYMMETRIC payload: many orders on few machines, all affinity
    stripped, uniform priority/due, every task compatible with every machine in its
    op.  This creates many equally-optimal pairings, so CP-SAT's choice is decided
    by variable-creation order — i.e. task-list order leaks into the schedule unless
    the ingest sort normalises it.  (Mirrors the symmetric_payload pattern that
    empirically gives keep_rate≈0.31 without a hint.)"""
    base = make_payload(
        n_orders=10,
        n_knitting_machines=3,
        n_linking_machines=2,
        max_factory_machines=3,
        max_search_time=10,
        num_search_workers=1,
        random_seed=42,
        rng_seed=11,
    )
    k_ids = [r["id"] for r in base["resources"] if r.get("operation") == "knitting"]
    l_ids = [r["id"] for r in base["resources"] if r.get("operation") == "linking"]
    for r in base["resources"]:
        r["design_item_id"] = ""
        r["color_config"] = ""
    for m in base["machines"]:
        m["design_item_id"] = ""
        m["color_config"] = ""
    for t in base["tasks"]:
        t["design_item_id"] = ""
        t["color_config"] = ""
        t["priority"] = 3
        t["due_at_min"] = 5000
        if t["operation"] == "knitting":
            t["duration"] = 200
            t["compatible_resource_ids"] = list(k_ids)
        elif t["operation"] == "linking":
            t["duration"] = 120
            t["compatible_resource_ids"] = list(l_ids)
    return base


@pytest.mark.parametrize("shuffle_seed", [1, 2, 3, 7, 99])
def test_task_order_does_not_change_schedule(contended_payload, shuffle_seed):
    base = Engine(copy.deepcopy(contended_payload)).solve()
    assert base["status"] in ("feasible", "optimal")
    base_hash = _canonical(base)

    shuffled = copy.deepcopy(contended_payload)
    random.Random(shuffle_seed).shuffle(shuffled["tasks"])
    perm = Engine(shuffled).solve()

    assert _canonical(perm) == base_hash, (
        f"shuffling task input order (seed={shuffle_seed}) changed the schedule — "
        f"determinism leg 3 (ingest sort by task_id) is broken"
    )


def test_baseline_is_self_consistent(contended_payload):
    """Sanity: same input twice → same output (isolates the permutation test above)."""
    a = Engine(copy.deepcopy(contended_payload)).solve()
    b = Engine(copy.deepcopy(contended_payload)).solve()
    assert _canonical(a) == _canonical(b)
