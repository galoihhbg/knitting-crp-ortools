# PHASE V — Dyelot allocation: problem measurement & model spec (READ-ONLY)

Status: **measured on real payloads carrying `main_yarn_consumption` + `dyelot_stock`.**
No solver/builder/CP-SAT touched; no model built.

Instrument: [`tools/dyelot_probe.py`](../tools/dyelot_probe.py) (read-only). Run:

```
python tools/dyelot_probe.py PAYLOAD.json --output OUTPUT.json --md report.md
```

Measured on `logs/solver_input_CP_1781340900453448867.json` (+ paired output);
`...518176` is byte-identical in structure. 660 tasks / 174 knitting / 8 machines.

`main_yarn` mode here = **legacy** (these payloads have no `is_main` flag yet → every
consumption entry counted as main). The probe is forward-compatible: when `is_main`
arrives it keeps only `is_main:true` (default-include when the flag is absent). Re-run
once the flagged payload lands — see the 3038 caveat below.

---

## 1. Chunk-granularity check — NOT a blocker ✅

- 174/174 knitting units carry `main_yarn_consumption`; **0 Go-side slices**
  (`is_slice=False`, no `parent_task_id`). Each unit is a whole per-component
  `SolverTask`; rolling-wave never splits a task. Field rides at exact granularity.
- Residual risk remains a **Go-contract** item only: if Go ever slices a knitting task
  it must split `main_yarn_consumption` per slice. The probe auto-flags this. Not
  triggered here.

---

## 2. Per-VI summary (MEASURED)

| VI | #dyelots | orders | units | machines | demand kg | stock kg | demand/stock | max lot kg | packing_size |
|----|---------:|-------:|------:|---------:|----------:|---------:|-------------:|-----------:|-------------:|
| **3038** | **0** | 16 | 48 | 2 | 130.0 | **0.0** | ∞ | 0.0 | — |
| 3039 | 3 | 14 | 126 | 8 | 403.0 | 2000.0 | 0.20 | 1000.0 | 1.0 |
| 3216 | 2 | 30 | 174 | 8 | 574.0 | 1000.0 | 0.57 | 500.0 | 1.0 |
| 3781 | 1 | 30 | 174 | 8 | 246.0 | 500.0 | 0.49 | 500.0 | 1.0 |
| 4330 | 1 | 30 | 174 | 8 | 574.0 | 1500.0 | 0.38 | 1500.0 | 1.0 |

- **Max dyelots/VI = 3.** Allocatable VIs = 4 (3038 excluded). Stock is **abundant**
  (demand/stock 0.20–0.57) — no VI is short.
- Dyelots: 3039 = {dyelot01:1000, dyelot02:500, dyelot03:500}; 3216 = {52:500, 53:500};
  3781 = {NO_LOT:500}; 4330 = {"10.":1500}. (Lot names `NO_LOT` / `10.` look like
  placeholder/parse artifacts — worth a glance from the Go side, but they are valid lots.)

### VI coupling still holds
A single unit draws multiple VIs at once (e.g. 3039+3216+3781+4330), and a creel flush
discards all of them at one machine-handoff → per-VI dyelot choices share the per-machine
flush variables. Decompose **per machine chain**, not per VI.

### ⚠️ 3038 — consumed as main, ZERO stock
16 orders / 48 units / 130 kg consume VI 3038, but `dyelot_stock` has no 3038 lot. This is
exactly what the **`is_main` flag is for**: if 3038 is a *secondary* yarn it should be
`is_main:false` and drop out of allocation entirely (problem becomes a clean 4-VI one). If
it is genuinely main, it's a hard data gap (no lot can be assigned). **Action: confirm
3038's `is_main` on the next payload and re-run.**

---

## 3. Never-flush over-group baseline — BENIGN on this payload

- Largest never-flush cohort: machine `DT7hPmDj15YcFQW` = **18 orders / 80.5 kg** forced
  onto one dyelot for VI 3216 → largest lot 500 kg → **FITS (6× margin)**.
- Across all 34 (machine-cohort, VI) bindings, **the only 2 overflows are VI 3038**
  (zero-stock artifact). **Excluding 3038, the never-flush greedy fits every stocked VI** —
  lots are 6–18× the largest cohort demand.

> **Key takeaway:** on this payload the over-grouping pathology ("không lô nào chứa nổi")
> **does not occur** — lots dwarf cohorts. A flush-deciding post-pass would be near-inert
> here (greedy is already feasible; only marginal small-lot tid:ying possible). The model's
> real value needs a **tight-stock / large-order payload** to demonstrate. **If you have a
> payload where the greedy actually fails to fit a lot, send it — that is the case that
> sizes the objective weights.**

---

## 4. Residual / packing pivotalness — APPROXIMATE capacity confirmed ✅

`packing_size = 1.0` for every lot; lots are 500–1500 kg → **pk/lot ≈ 0.001–0.002**
(0.1–0.2%). Residual carried across a handoff is utterly negligible vs a lot.

→ **Use the approximate capacity model**: each cohort-segment's per-VI load =
`Σ kg of its units ≤ chosen lot.remaining_kg`, ignoring residual carry-in. No need for the
exact per-machine residual-flow network. (Decision rule the probe prints: exact only if
`pk/lot ≳ 0.15`; here it is 75–150× below that.)

---

## 5. Feasibility sanity (MEASURED)

- **1 VI consumed-as-main with zero stock: `3038`** (see §2 caveat).
- For every *stocked* VI: **no single (order, VI) exceeds its largest lot**, and **no VI is
  short on total stock**. The only feasibility issue is the 3038 flag/gap.

---

## Conclusion — design recommendation (now grounded in real data)

**(a) Capacity model: APPROXIMATE — confirmed, not provisional.** `pk/lot ≈ 0.002`.
Segment `Σkg ≤ lot.remaining_kg`; drop residual flow.

**(b) Lexicographic objective: feasibility ▸ minimize flush-waste ▸ small-lots-first.**
- Feasibility stays hard/top (binds on tight-stock payloads, not this one).
- Each flush wastes ≤ `packing_size` = 1 kg — essentially free here, so the model can flush
  liberally to satisfy feasibility; flush-waste only matters when packs are large.
- Small-lots-first (consume nearly-empty lots, avoid fragmenting a VI across dyelots) is the
  main *quality* lever once feasible. Weight 2-vs-3 needs a binding payload to tune; on
  abundant-stock inputs they barely trade off.

**(c) CP-SAT scale: trivial.** ≤3 dyelots/VI, 4 allocatable VIs, ≤30 orders, 8 machine
chains (18–28 units each). Per-machine flush booleans (≤28×8) + per-(segment, VI) lot pick.
Comfortably small — no scale concern at this size.

**Caveats / next inputs that would sharpen this:**
1. A payload where the never-flush greedy **actually overflows a lot** (tight stock / big
   orders) — needed to validate the post-pass's core value and set objective weights.
2. The **`is_main`-flagged** payload — to confirm 3038 (and any other) is secondary and the
   allocatable VI set is genuinely 4. Re-run `tools/dyelot_probe.py`; the report auto-switches
   to "flagged" mode and drops secondary yarns.
