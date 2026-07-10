"""A1a — placement helpers: chứng minh tương đương với các bản gốc.

Mỗi helper trong app/engine/phases/placement.py thay thế 2–8 bản cài đặt
trùng lặp trong các file phase.  Test này giữ BẢN THAM CHIẾU (copy nguyên văn
code gốc, prefix _ref_) và fuzz-đối-chiếu helper với từng bản trên hàng nghìn
case sinh bằng seed cố định — bằng chứng tương đương tồn tại vĩnh viễn kể cả
sau khi các bản gốc bị xoá khỏi file phase (Gate 3).

Với 2 balancer, test so sánh TRỰC TIẾP với hàm gốc đang tồn tại
(balance_linking_load / balance_downstream_load) trên scenario fuzz: sau khi
Gate 3 chuyển thân hàm gốc sang delegate, phép so trở thành tautology — vẫn
giữ để khoá hành vi.
"""
import copy
import random
from typing import Any, Dict

from app.engine.phases.placement import (
    avail_at,
    bump_earliest,
    earliest_candidates,
    earliest_sweep,
    overlaps,
    relabel_balance,
    release_from_deps,
    unavail_windows,
)
from app.engine.phases.phase2_linking import PHASE2_OPS, balance_linking_load
from app.engine.phases.phase4_downstream import balance_downstream_load


# ── Bản tham chiếu (copy nguyên văn từ code gốc — KHÔNG sửa) ────────────────


def _ref_earliest_slot_phase4(busy, release, dur):
    """phase4_downstream._earliest_slot (dòng 214) — nguyên văn."""
    t = release
    for s, e in busy:
        if e <= t:
            continue
        if s >= t + dur:
            break
        t = max(t, e)
    return t


def _ref_earliest_gap_phase1(busy, release, dur):
    """phase1_knitting._earliest_gap (dòng 1151) — nguyên văn."""
    t = release
    for s, e in busy:
        if e <= t:
            continue
        if s >= t + dur:
            break
        t = max(t, e)
    return t


def _ref_earliest_phase2(plist, release, dur):
    """phase2_linking.left_shift_cold_linking._earliest (dòng 282) — nguyên văn."""
    cur = release
    for s, e in plist:
        if s >= cur + dur:
            return cur
        if e > cur:
            cur = e
    return cur


def _ref_earliest_phase3(machine_busy, bounds, release, dur, limit):
    """phase3_batching.left_shift_cold_washing._earliest (dòng 1637) — nguyên văn
    (đóng gói machine_busy/bounds làm tham số thay vì closure)."""

    def _straddles(s, e):
        return any(s < b < e for b in bounds)

    def _free(s, e):
        for (bs, be) in machine_busy:
            if s < be and bs < e:
                return False
        return True

    cands = {release}
    for (bs, be) in machine_busy:
        cands.add(be)
    for b in bounds:
        cands.add(b)
    for s in sorted(c for c in cands if release <= c < limit):
        if _straddles(s, s + dur):
            continue
        if _free(s, s + dur):
            return s
    return None


def _ref_reentries(orders):
    """phase1_knitting._reentries — nguyên văn (dùng cho oracle nonfrag)."""
    runs = []
    for o in orders:
        if not runs or runs[-1] != o:
            runs.append(o)
    from collections import Counter
    return sum(v - 1 for v in Counter(runs).values() if v > 1)


def _ref_earliest_nonfrag_start_phase1(placed, release, dur, order):
    """phase1_knitting._earliest_nonfrag_start (dòng 1007) — nguyên văn."""
    base_re = _ref_reentries([o for _, _, o in placed])
    candidates = sorted({release} | {e for _, e, _ in placed if e >= release})
    for cs in candidates:
        ce = cs + dur
        if any(s < ce and e > cs for s, e, _ in placed):
            continue
        merged = sorted(placed + [(cs, ce, order)])
        if _ref_reentries([o for _, _, o in merged]) <= base_re:
            return cs
    return None


def _ref_machine_earliest_core_phase1(fixed_busy, start, dur):
    """Lõi phase1_knitting._machine_earliest (dòng 1690) sau seed — nguyên văn."""
    st = start
    moved = True
    while moved:
        moved = False
        for ws, we in fixed_busy:
            if st < we and st + dur > ws:
                st = we
                moved = True
    return st


# ── Fuzz generators (seed cố định — deterministic) ──────────────────────────


def _sorted_busy(rng, n, allow_overlap):
    out = []
    t = 0
    for _ in range(n):
        if allow_overlap:
            s = rng.randint(max(0, t - 5), t + 15)
        else:
            s = t + rng.randint(0, 10)
        e = s + rng.randint(0, 12)  # cho phép interval rỗng (s == e)
        out.append((s, e))
        t = max(t, e)
    return sorted(out)


# ── earliest_sweep ──────────────────────────────────────────────────────────


class TestEarliestSweep:
    def test_empty_busy_returns_release(self):
        assert earliest_sweep([], 7, 5) == 7

    def test_fits_first_gap(self):
        assert earliest_sweep([(0, 10), (20, 30)], 0, 5) == 10

    def test_appends_after_tail(self):
        assert earliest_sweep([(0, 10), (10, 30)], 0, 5) == 30

    def test_zero_duration(self):
        # dur=0: interval bắt đầu đúng tại t không đẩy (s >= t+0 → break)
        assert earliest_sweep([(5, 9)], 5, 0) == 5

    def test_fuzz_equiv_all_three_origins(self):
        rng = random.Random(20260709)
        for _ in range(4000):
            busy = _sorted_busy(rng, rng.randint(0, 8), rng.random() < 0.4)
            release = rng.randint(0, 40)
            dur = rng.randint(0, 8)
            got = earliest_sweep(busy, release, dur)
            assert got == _ref_earliest_slot_phase4(busy, release, dur)
            assert got == _ref_earliest_gap_phase1(busy, release, dur)
            assert got == _ref_earliest_phase2(busy, release, dur)


# ── earliest_candidates ─────────────────────────────────────────────────────


class TestEarliestCandidates:
    def test_none_when_no_slot_below_limit(self):
        assert earliest_candidates([(0, 10)], 0, 5, limit=10) is None

    def test_basic_gap(self):
        assert earliest_candidates([(0, 10), (20, 30)], 0, 5, limit=100) == 10

    def test_fuzz_equiv_phase3_washing(self):
        rng = random.Random(20260710)
        for _ in range(4000):
            busy = _sorted_busy(rng, rng.randint(0, 6), rng.random() < 0.3)
            bounds = sorted(rng.sample(range(0, 80), rng.randint(0, 4)))
            release = rng.randint(0, 40)
            dur = rng.randint(0, 10)
            limit = rng.randint(0, 90)

            def _no_straddle(s, e, _b=bounds):
                return not any(s < b < e for b in _b)

            got = earliest_candidates(
                busy, release, dur,
                extra_candidates=bounds, limit=limit, accept=_no_straddle,
            )
            want = _ref_earliest_phase3(busy, bounds, release, dur, limit)
            assert got == want, (busy, bounds, release, dur, limit)

    def test_fuzz_equiv_phase1_nonfrag(self):
        rng = random.Random(20260711)
        orders = ["A", "B", "C"]
        for _ in range(4000):
            raw = _sorted_busy(rng, rng.randint(0, 6), False)
            placed = sorted((s, e, rng.choice(orders)) for s, e in raw)
            release = rng.randint(0, 40)
            dur = rng.randint(0, 8)
            order = rng.choice(orders)

            def _no_new_reentry(cs, ce, _p=placed, _o=order):
                base = _ref_reentries([o for _, _, o in _p])
                merged = sorted(_p + [(cs, ce, _o)])
                return _ref_reentries([o for _, _, o in merged]) <= base

            got = earliest_candidates(
                [(s, e) for s, e, _ in placed], release, dur,
                accept=_no_new_reentry,
            )
            want = _ref_earliest_nonfrag_start_phase1(placed, release, dur, order)
            assert got == want, (placed, release, dur, order)


# ── bump_earliest ───────────────────────────────────────────────────────────


class TestBumpEarliest:
    def test_unsorted_overlapping_busy(self):
        busy = [(20, 30), (0, 25), (28, 40)]  # chưa sort, chồng nhau
        assert bump_earliest(busy, 0, 5) == 40

    def test_zero_duration_edge_no_bump(self):
        # dur=0 tại đúng mép ws: st+0 > ws sai → không bump (hành vi khoá-sort gốc)
        assert bump_earliest([(5, 9)], 5, 0) == 5
        # dur=0 nằm hẳn trong interval → bump tới we
        assert bump_earliest([(5, 9)], 6, 0) == 9

    def test_fuzz_equiv_phase1_machine_earliest(self):
        rng = random.Random(20260712)
        for _ in range(4000):
            n = rng.randint(0, 7)
            busy = [
                (s, s + rng.randint(0, 12))
                for s in (rng.randint(0, 60) for _ in range(n))
            ]
            rng.shuffle(busy)
            start = rng.randint(0, 70)
            dur = rng.randint(0, 8)
            assert bump_earliest(busy, start, dur) == \
                _ref_machine_earliest_core_phase1(busy, start, dur)


# ── release_from_deps / unavail_windows / avail_at ──────────────────────────


class TestReleaseFromDeps:
    def test_start_after_and_deps(self):
        t = {"start_after_min": 50, "final_depends_on": ["a", "b", "zz"]}
        ends = {"a": 40, "b": 90}
        assert release_from_deps(t, ends) == 90
        assert release_from_deps({"start_after_min": 50}, ends) == 50

    def test_washing_mode_ignores_start_after(self):
        # Quirk giữ nguyên: ready-time washing bỏ qua start_after_min (bản gốc).
        t = {"start_after_min": 500, "final_depends_on": ["a"]}
        assert release_from_deps(t, {"a": 40}, include_start_after=False) == 40
        assert release_from_deps(t, {}, include_start_after=False) == 0

    def test_none_start_after_tolerated(self):
        assert release_from_deps({"start_after_min": None}, {}) == 0


class TestResourceHelpers:
    def test_unavail_windows_filters_empty(self):
        r = {"unavailability": [
            {"start": 5, "end": 10}, {"start": 7, "end": 7}, {"start": 9, "end": 3},
        ]}
        assert unavail_windows(r) == [(5, 10)]
        assert unavail_windows({}) == []
        assert unavail_windows({"unavailability": None}) == []

    def test_avail_at(self):
        assert avail_at({"available_at_min": 30}) == 30
        assert avail_at({"available_at_min": None}) == 0
        assert avail_at({}) == 0


# ── relabel_balance ≡ 2 balancer gốc ────────────────────────────────────────


def _fuzz_scenario(rng, op: str, n_machines: int, n_tasks: int):
    """Sinh (assignments, all_tasks, resources) ngẫu nhiên cho một op."""
    machines = [f"M_{op}_{i:02d}" for i in range(n_machines)]
    resources = []
    for m in machines:
        r: Dict[str, Any] = {"id": m, "operation": op}
        if rng.random() < 0.3:
            s = rng.randint(0, 50)
            r["unavailability"] = [{"start": s, "end": s + rng.randint(1, 30)}]
        if rng.random() < 0.3:
            r["available_at_min"] = rng.randint(0, 40)
        resources.append(r)
    # thêm một resource op khác để test lọc theo ops
    resources.append({"id": "M_OTHER", "operation": "somethingelse"})

    tasks, assigns = [], []
    for i in range(n_tasks):
        tid = f"T_{op}_{i:03d}"
        s = rng.randint(0, 200)
        e = s + rng.randint(1, 40)
        compat = (
            rng.sample(machines, rng.randint(1, n_machines))
            if rng.random() < 0.7 else []
        )
        tasks.append({
            "task_id": tid,
            "operation": op if rng.random() < 0.9 else "somethingelse",
            "compatible_resource_ids": compat,
            "is_pinned": rng.random() < 0.15,
        })
        assigns.append({
            "task_id": tid,
            "machine_id": rng.choice(machines),
            "start_time": s,
            "end_time": e,
        })
    return assigns, tasks, resources


class TestRelabelBalanceEquivalence:
    def test_fuzz_equiv_balance_linking_load(self):
        rng = random.Random(20260713)
        for _ in range(300):
            assigns, tasks, resources = _fuzz_scenario(
                rng, "linking", rng.randint(1, 5), rng.randint(0, 25)
            )
            a1, a2 = copy.deepcopy(assigns), copy.deepcopy(assigns)
            r1 = balance_linking_load(a1, tasks, resources, {})
            r2 = relabel_balance(
                a2, tasks, resources, {},
                ops=frozenset(PHASE2_OPS), label="Linking",
            )
            assert r1 == r2
            assert a1 == a2

    def test_fuzz_equiv_balance_downstream_load(self):
        rng = random.Random(20260714)
        for _ in range(300):
            assigns, tasks, resources = _fuzz_scenario(
                rng, "iron", rng.randint(1, 5), rng.randint(0, 25)
            )
            a1, a2 = copy.deepcopy(assigns), copy.deepcopy(assigns)
            r1 = balance_downstream_load(
                a1, tasks, resources, {}, frozenset({"iron", "ironing"}), "Ironing",
            )
            r2 = relabel_balance(
                a2, tasks, resources, {},
                ops=frozenset({"iron", "ironing"}), label="Ironing",
            )
            assert r1 == r2
            assert a1 == a2

    def test_pinned_stays_and_occupies(self):
        tasks = [
            {"task_id": "P", "operation": "iron", "compatible_resource_ids": [],
             "is_pinned": True},
            {"task_id": "F", "operation": "iron", "compatible_resource_ids": [],
             "is_pinned": False},
        ]
        resources = [{"id": "M_iron_00", "operation": "iron"},
                     {"id": "M_iron_01", "operation": "iron"}]
        assigns = [
            {"task_id": "P", "machine_id": "M_iron_01", "start_time": 0, "end_time": 10},
            {"task_id": "F", "machine_id": "M_iron_01", "start_time": 5, "end_time": 15},
        ]
        relabel_balance(assigns, tasks, resources, {},
                        ops=frozenset({"iron", "ironing"}), label="Ironing")
        by = {a["task_id"]: a for a in assigns}
        assert by["P"]["machine_id"] == "M_iron_01"   # pinned giữ máy
        assert by["F"]["machine_id"] == "M_iron_00"   # movable né interval pinned

    def test_single_machine_noop(self):
        tasks = [{"task_id": "F", "operation": "iron",
                  "compatible_resource_ids": [], "is_pinned": False}]
        resources = [{"id": "M_iron_00", "operation": "iron"}]
        assigns = [{"task_id": "F", "machine_id": "M_iron_00",
                    "start_time": 0, "end_time": 5}]
        assert relabel_balance(
            assigns, tasks, resources, {},
            ops=frozenset({"iron", "ironing"}), label="Ironing",
        ) == 0


class TestOverlaps:
    def test_touching_endpoints_not_overlap(self):
        assert not overlaps(0, 5, 5, 10)
        assert not overlaps(5, 10, 0, 5)

    def test_containment_and_partial(self):
        assert overlaps(2, 4, 0, 10)
        assert overlaps(0, 10, 2, 4)
        assert overlaps(0, 6, 5, 10)

    def test_empty_interval_semantics_preserved(self):
        # Hành vi GỐC (giữ nguyên): interval rỗng nằm HẲN TRONG interval bận vẫn
        # tính là chồng lấn (s < be and bs < e với s == e) — nhất quán với cách
        # bump_earliest xử lý dur=0.  Chạm mép thì không.
        assert overlaps(5, 5, 0, 10)
        assert not overlaps(0, 0, 0, 10)
        assert not overlaps(10, 10, 0, 10)
