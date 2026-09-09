# Debug Log — Multi-PO Knitting Scheduling (Linking Wait) — 2026-05-06

## Failure

- **Action:** Schedule an order where the Knitting stage has multiple POs (e.g. PO1 and PO2 are two separate Knitting tasks that both feed the same Linking task for one order).
- **Expected:** Solver schedules all Knitting POs in parallel (or at least pipelined) so that Linking can start as soon as enough knitted output is ready. PO1 and PO2 run concurrently on different machines, allowing Linking to begin earlier.
- **Actual:** All batches of PO1 are completed before PO2 even starts. Linking must wait for *both* PO1 and PO2 to finish before it can begin, causing the Linking workers to idle for a long time and the order to be late.
- **Error:** No exception; this is a silent logic / scheduling quality bug.

---

## Data Trace

| Step | Location (file:line) | Expected Value | Actual Value |
|------|----------------------|----------------|--------------|
| Input | `apply_dependency_constraints()` | L task has `final_depends_on = [K_PO1_batch, K_PO2_batch]` | Same — deps are correct |
| Inferred K→L | `builder.py:1016-1024` | Both K_PO1 and K_PO2 resolved from translation map → L must wait for ALL of them | **Both resolved correctly** — L waits for end of K_PO1 AND end of K_PO2 ✅ |
| PO co-location | `builder.py:733-769` | PO-co-location only groups tasks of the *same* PO (same `original_order_id`) | ✅ Groups by `original_order_id` (= PO ID), not order ID — correct |
| Objective gap penalty | `builder.py:940-961` | Gap penalty per K→L edge drives Linking to start ASAP after *each* K finishes | **Bug: one gap_var is added per K→L edge**, each penalising `L.start - K.end`. When L has two parents (K_PO1 and K_PO2) the objective becomes `gap_PO1 + gap_PO2` where gap_PO1 = L.start - K_PO1.end and gap_PO2 = L.start - K_PO2.end |
| Solver incentive | CP-SAT objective | Solver should finish all Ks ASAP and start L immediately | **Bug manifests here:** to minimise gap_PO2, the solver finds it *cheaper* to run K_PO1 first, then K_PO2, then L — this minimises both gaps at cost zero, but forces L to wait for K_PO2 |
| Output | `extract_results()` | Parallel K assignments, L starts after whichever K finishes last | L starts after the sequentially-run K_PO1 AND K_PO2, making it late |

---

## First Point of Divergence

- **Step:** `build_resource_allocations()` — PO co-location bounding-box logic (lines 733–769)
- **Location:** `builder.py:733-769`
- **Expected:** The solver is *encouraged* to run K_PO1 and K_PO2 concurrently (on separate machines) so Linking can start sooner.
- **Got:** There is NO constraint or objective term that encourages K tasks from **different POs of the same order** to run in parallel. The only co-location constraint groups tasks within the *same* PO on the *same* machine. Cross-PO parallelism has no incentive.
- **Anomaly type:** Logic error — missing objective incentive for cross-PO parallelism.

**Secondary divergence (amplifier):**
- `_add_gap_penalty(K_PO1, L)` and `_add_gap_penalty(K_PO2, L)` both add `(L.start - K.end)` to the objective. This means the *total* gap penalty is minimised whether Ks run sequentially or in parallel — the solver sees no difference in objective cost between the two options. Without a concrete advantage to parallelism, the solver picks the sequential arrangement because it's simpler for the machine NoOverlap constraint (one machine, two sequential tasks).

---

## Root Cause Hypothesis

The bug is caused by **the absence of any objective term or constraint in `build_resource_allocations()` that incentivises K tasks of different POs within the same order to run on different machines concurrently**, because the co-location bounding-box logic (lines 733–769) only groups tasks that share the *same* `original_order_id` (= PO), so two POs feeding the same Linking task get no cross-PO parallelism push, and the gap-penalty terms per K→L edge provide equal cost whether the Ks run sequentially or in parallel.

---

## Proposed Fix

### Minimal Change

Add a **cross-PO parallelism incentive** in `build_resource_allocations()` (after the existing PO co-location block):

For every order that has **multiple Knitting PO groups** (i.e., multiple distinct `original_order_id` values among Knitting tasks that share the same `group_id`), add a soft penalty:

```
penalty += (K_PO1.end - K_PO2.start)_clipped_positive  ← max(0, K_PO1.end - K_PO2.start)
```

This incentivises the solver to start K_PO2 as early as possible rather than waiting for K_PO1 to finish.

**Implementation sketch (inside `build_resource_allocations()`):**

```python
# Cross-PO parallelism incentive:
# For orders with multiple knitting POs, encourage them to overlap by
# penalising the gap between the start of the later PO and the end of the earlier PO.
# This is a soft incentive — if machines are scarce the solver can still run them serially.

from collections import defaultdict
group_knitting_pos: Dict[str, Dict[str, List]] = defaultdict(dict)
for t in self.tasks:
    if t.get("operation", "").lower() == "knitting" and not t.get("is_pinned", False):
        gid = t.get("group_id", "")
        po_id = t.get("original_order_id", "")
        if gid and po_id:
            group_knitting_pos[gid].setdefault(po_id, []).append(t)

_cross_po_w = max(1, (10 ** (6 - 3)) * self.lateness_scale // 5)  # medium weight

for gid, po_map in group_knitting_pos.items():
    po_ids = sorted(po_map.keys())
    if len(po_ids) < 2:
        continue
    # For every pair of POs in this order, penalise the start-gap of the later one
    for i, po_a in enumerate(po_ids):
        for po_b in po_ids[i + 1:]:
            # Representatives: first task of each PO
            ta = po_map[po_a][0]["task_id"]
            tb = po_map[po_b][0]["task_id"]
            if ta not in self.task_vars or tb not in self.task_vars:
                continue
            # overlap_gap = max(0, start_b - start_a)  -- penalise late start of b
            # We want b to start as early as a (or earlier)
            late_start = self.model.NewIntVar(0, self.horizon, f"cross_po_gap_{gid}_{po_a}_{po_b}")
            self.model.Add(late_start >= self.task_vars[tb]["start"] - self.task_vars[ta]["start"])
            self.model.Add(late_start >= self.task_vars[ta]["start"] - self.task_vars[tb]["start"])
            self.objective_terms.append(late_start * _cross_po_w)
            logger.info(f"   ⚡ Cross-PO parallel incentive: {ta} ↔ {tb} (group={gid}, w={_cross_po_w})")
```

This penalises the difference in start times between the two PO batches, driving the solver to start them at the same time (on different machines) to minimise the objective.

---

## Fix Applied

- **File(s) changed:** `app/engine/shared.py`, `app/engine/phases/phase1_knitting.py`, `app/engine/phases/phase2_linking.py`, `app/engine/phases/phase4_downstream.py`
- **What changed:** Implemented an **Order Flow (Lead Time) Optimization** objective. For every `group_id`, the solver now penalizes both the group completion time (`max_end`) and the group elapsed span (`max_end - min_start`). 
- **Why this addresses root cause:** This provides a strong incentive (weighted at ~5% of lateness) to parallelize component POs across multiple machines. It effectively overcomes machine affinity/setup biases that previously drove the solver toward sequential processing on a single "setup-ready" machine, ensuring downstream linking and washing phases can start much earlier.

## Verification

- [x] Original bug no longer reproduces (Parallelism confirmed even with high affinity bias)
- [x] Order makespan minimized across all pipeline phases
- [x] Determinism preserved (verified with 200-task replay tests)
- [x] `pytest tests/` passes with new flow optimization tests
