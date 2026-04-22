# Testing Strategy

## Frameworks

- **Unit / Integration Tests:** `pytest` (run via `pytest tests/ -v`)
- **Coverage:** `pytest-cov` (`pytest tests/ --cov=app --cov-report=term-missing`)
- **No E2E framework** — solver is a pure function; end-to-end is tested via fixture-driven integration tests

## Coverage Targets

| Module | Target | Method |
|--------|--------|--------|
| `builder.py` | 80% line coverage | `pytest-cov` |
| `model.py` | 90% | `pytest-cov` |
| `extract_results()` root-cause branches | 100% | Parametrized fixtures |
| `filter_dummy_tasks/overloads` | 100% | Unit test |

## Test Classes

### Priority 1: Determinism Regression

File: `tests/test_determinism.py`

```python
import copy, json
from app.engine.model import Engine

FIXTURE = json.load(open("tests/fixtures/payload_200_tasks.json"))

def test_deterministic_output():
    """5 runs with num_search_workers=1 → byte-identical assignments."""
    results = []
    for _ in range(5):
        payload = copy.deepcopy(FIXTURE)
        payload["config"]["num_search_workers"] = 1
        payload["config"]["random_seed"] = 42
        results.append(Engine(payload).solve())
    baseline = results[0]["assignments"]
    for run in results[1:]:
        assert run["assignments"] == baseline
```

### Priority 2: Root-Cause Classification

File: `tests/test_root_cause.py`

```python
@pytest.mark.parametrize("fixture,expected_code", [
    ("payload_pinned_conflict.json", "PINNED_TASK_CONFLICT"),
    ("payload_machine_overload.json", "MACHINE_OVERLOAD"),
    ("payload_capacity_full.json", "WORKFORCE_SHORTAGE"),
])
def test_root_cause_code(fixture, expected_code):
    payload = json.load(open(f"tests/fixtures/{fixture}"))
    result = Engine(payload).solve()
    late = [o for o in result["overloads"] if o["status"] == "LATE"]
    assert any(o["root_cause_code"] == expected_code for o in late)
```

### Priority 3: Hard Constraint Verification

File: `tests/test_constraints.py`

```python
def test_no_overlap_on_machine(solved_result):
    from collections import defaultdict
    slots = defaultdict(list)
    for a in solved_result["assignments"]:
        slots[a["machine_id"]].append((a["start_time"], a["end_time"]))
    for m_id, machine_slots in slots.items():
        machine_slots.sort()
        for i in range(len(machine_slots) - 1):
            assert machine_slots[i][1] <= machine_slots[i+1][0], \
                f"Overlap on {m_id}: {machine_slots[i]} vs {machine_slots[i+1]}"

def test_pinned_tasks_immovable(solved_result, payload):
    pinned = {t["task_id"]: t for t in payload["tasks"] if t.get("is_pinned")}
    for a in solved_result["assignments"]:
        if a["task_id"] in pinned:
            t = pinned[a["task_id"]]
            assert a["machine_id"] == t["pinned_machine_id"]
            assert a["start_time"] == t["pinned_start_time"]
            assert a["end_time"] == t["pinned_end_time"]
```

### Priority 4: Infeasibility Handling

```python
def test_infeasible_returns_structured_dict(impossible_payload):
    """Never returns HTTP 500 — always a structured dict."""
    result = Engine(impossible_payload).solve()
    assert result["status"] == "infeasible"
    assert result["assignments"] == []
    assert result["overloads"] == []
```

### Priority 5: Smart Batching

File: `tests/test_smart_batching.py`

Key checks:
- **Capacity respected**: No batch slot has `sum(qty) > washing_batch_capacity`
- **Multiple batches formed**: 6 tasks with capacity=2 → ≥ 3 distinct `batch_slot_id` values
- **Start-time sync**: If two tasks share `batch_slot_id`, their `start_time` must be identical
- **Urgent order override**: Tight-deadline task may get its own slot rather than waiting for others
- **No-op case**: Non-washing tasks produce `batch_slot_id == ""`

```python
# Pattern: verify slot capacity from response assignments
from collections import Counter
slot_counts = Counter(a["batch_slot_id"] for a in result["assignments"] if a["batch_slot_id"])
for slot_id, count in slot_counts.items():
    assert count <= capacity, f"Slot {slot_id} exceeded capacity"

# Pattern: verify start-time sync
by_id = {a["task_id"]: a for a in result["assignments"]}
wa, wb = by_id["WA"], by_id["WB"]
if wa["batch_slot_id"] and wa["batch_slot_id"] == wb["batch_slot_id"]:
    assert wa["start_time"] == wb["start_time"]
```

Fixture helpers in `test_smart_batching.py`:
- `_make_washing_task(task_id, order_id, duration, due, resource_ids)` — sets `color="red"`, `substance="cotton"` by default
- `_make_config(**overrides)` — sets `washing_batch_capacity=3` by default
- `_make_resource(r_id, op="washing")` — creates resource dict with `capacity=1`

For machine sync tests, set resource `capacity > 1` so `AddCumulative` is used instead of `AddNoOverlap`.

## Test Fixtures

```
tests/
├── conftest.py                        # Shared fixtures + Engine factory
├── fixtures/
│   ├── payload_10_tasks.json          # Smoke test (fast)
│   ├── payload_200_tasks.json         # Determinism baseline
│   ├── payload_pinned_conflict.json   # Triggers PINNED_TASK_CONFLICT
│   ├── payload_machine_overload.json  # Triggers MACHINE_OVERLOAD
│   └── payload_capacity_full.json     # Triggers WORKFORCE_SHORTAGE
├── test_determinism.py
├── test_root_cause.py
├── test_constraints.py
└── test_infeasibility.py
```

## Rules

- **NEVER skip tests or mock CP-SAT to make a pipeline pass.** If `builder.py` changes break a test, fix the builder — don't patch the test.
- **Determinism test must use `num_search_workers=1`** — higher worker counts are non-deterministic by design.
- **All fixture payloads must be valid `SolverPayload` JSON** — use `SolverPayload.model_validate(json.load(...))` to verify.
- **New builder methods must have at least one test** verifying the constraint is not trivially infeasible.

## Execution

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_determinism.py -v

# With coverage report
pytest tests/ --cov=app --cov-report=term-missing

# Only fast tests (skip 200-task fixture)
pytest tests/ -v -m "not slow"

# Run constraint tests only
pytest tests/test_constraints.py -v
```

## Pre-Commit Verification Loop

After every change to `builder.py` or `model.py`:
1. `pytest tests/test_constraints.py -v` — verify hard constraints hold
2. `pytest tests/test_determinism.py -v` — verify determinism not broken
3. `pytest tests/ -v` — full suite before committing
