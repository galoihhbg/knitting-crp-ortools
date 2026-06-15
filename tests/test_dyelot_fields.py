"""
Additive-schema tests for the dyelot post-pass groundwork.

Two new fields are added as DATA ONLY (no CP-SAT consumption yet):
  * SolverTask.main_yarn_consumption : List[YarnConsumption]
  * SolverPayload.dyelot_stock        : List[DyelotStock]

These tests prove the fields:
  1. round-trip through the schema with the right shape,
  2. SURVIVE the real ingestion path (route model_dump → Engine → Pipeline,
     including _sanitize_dummy_tasks) all the way to where the dyelot post-pass
     reads them (Pipeline.dyelot_stock / each task's main_yarn_consumption),
  3. default to empty for legacy payloads missing both, leaving old fields intact.
"""
from app.schemas.request_schema import (
    DyelotStock,
    SolverPayload,
    SolverTask,
    YarnConsumption,
)
from app.engine.model import Engine
from app.engine.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Fixture builders (minimal valid SolverTask / SolverPayload)
# ---------------------------------------------------------------------------

def _task(task_id: str = "K1", **overrides) -> dict:
    base = dict(
        task_id=task_id,
        original_order_id="ORDER_1",
        group_id="G1",
        operation="knitting",
        qty=10.0,
        total_qty=10.0,
        priority=3,
        duration=100,
        design_item_id="DESIGN_A",
        color_config="MAT_WHT:1",
        compatible_resource_ids=["KM_00"],
    )
    base.update(overrides)
    return base


def _payload(tasks, **overrides) -> dict:
    base = dict(
        job_id="JOB_1",
        config={},
        machines=[{"id": "KM_00", "design_item_id": "DESIGN_A", "color_config": "MAT_WHT:1"}],
        resources=[{"id": "KM_00", "operation": "knitting", "capacity": 1}],
        tasks=tasks,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1 — schema keeps the new fields with the right shape
# ---------------------------------------------------------------------------

def test_schema_keeps_new_fields():
    payload = SolverPayload(
        **_payload(
            tasks=[
                _task(main_yarn_consumption=[{"vi": "Sợi trắng", "kg": 12.5}])
            ],
            dyelot_stock=[
                {"vi": "Mẻ nhuộm A", "dyelot": "DL-001", "remaining_kg": 80.0, "packing_size": 25.0}
            ],
        )
    )

    # Per-task field
    yc = payload.tasks[0].main_yarn_consumption
    assert len(yc) == 1
    assert isinstance(yc[0], YarnConsumption)
    assert yc[0].vi == "Sợi trắng"
    assert yc[0].kg == 12.5
    assert yc[0].is_main is True  # default when flag absent (legacy = main)

    # Top-level field
    ds = payload.dyelot_stock
    assert len(ds) == 1
    assert isinstance(ds[0], DyelotStock)
    assert ds[0].vi == "Mẻ nhuộm A"
    assert ds[0].dyelot == "DL-001"
    assert ds[0].remaining_kg == 80.0
    assert ds[0].packing_size == 25.0


def test_is_main_flag_main_and_secondary():
    """is_main distinguishes main yarn (gets a dyelot) from secondary yarn.

    Mirrors the production shape: [{vi,kg,is_main:true}, {vi,kg,is_main:false}].
    """
    payload = SolverPayload(
        **_payload(
            tasks=[
                _task(main_yarn_consumption=[
                    {"vi": "3038", "kg": 47.3, "is_main": True},
                    {"vi": "3216", "kg": 12.0, "is_main": False},
                ])
            ],
        )
    )
    yc = payload.tasks[0].main_yarn_consumption
    assert [(c.vi, c.kg, c.is_main) for c in yc] == [
        ("3038", 47.3, True),
        ("3216", 12.0, False),
    ]
    # Survives real ingestion as plain dicts the dyelot post-pass can filter on.
    engine = Engine(payload.model_dump(by_alias=False))
    ingested = engine.tasks[0]["main_yarn_consumption"]
    main_only = [c for c in ingested if c.get("is_main", True)]
    assert [c["vi"] for c in main_only] == ["3038"]


# ---------------------------------------------------------------------------
# Test 2 — fields survive the REAL ingestion path (the core test)
# ---------------------------------------------------------------------------

def test_fields_survive_real_ingestion():
    """Route → model_dump(by_alias=False) → Engine → Pipeline (incl. _sanitize_dummy_tasks).

    Includes a real task carrying main_yarn_consumption AND a pinned dummy task
    (qty=0) so the _sanitize_dummy_tasks branch is exercised — proving real tasks
    pass through with the field intact.
    """
    real = _task(
        task_id="K1",
        main_yarn_consumption=[
            {"vi": "Sợi trắng", "kg": 12.5},
            {"vi": "Sợi đen", "kg": 3.0},
        ],
    )
    dummy = _task(
        task_id="DUMMY_1",
        qty=0.0,
        total_qty=0.0,
        is_pinned=True,
        pinned_machine_id="KM_00",
        pinned_start_time=0,
        pinned_end_time=50,
    )
    payload_model = SolverPayload(
        **_payload(
            tasks=[real, dummy],
            dyelot_stock=[
                {"vi": "Mẻ nhuộm A", "dyelot": "DL-001", "remaining_kg": 80.0, "packing_size": 25.0},
                {"vi": "Mẻ nhuộm B", "dyelot": "DL-002", "remaining_kg": 40.0, "packing_size": 10.0},
            ],
        )
    )

    # Exactly what the FastAPI route does (solver_route.py:17/46).
    data = payload_model.model_dump(by_alias=False)

    # Engine parses the raw dict (solver_task → Engine(payload)).
    engine = Engine(data)

    # Top-level dyelot_stock survived route → Engine.
    assert len(engine.dyelot_stock) == 2
    assert engine.dyelot_stock[0]["dyelot"] == "DL-001"
    assert engine.dyelot_stock[1]["remaining_kg"] == 40.0

    # Per-task main_yarn_consumption survived on the real task.
    real_ingested = next(t for t in engine.tasks if t["task_id"] == "K1")
    assert [y["vi"] for y in real_ingested["main_yarn_consumption"]] == ["Sợi trắng", "Sợi đen"]
    assert real_ingested["main_yarn_consumption"][0]["kg"] == 12.5

    # Build the Pipeline exactly as Engine.solve() does — this runs
    # _sanitize_dummy_tasks and is where the dyelot post-pass will read the data.
    pipeline = Pipeline(
        engine.config,
        engine.resources,
        engine.tasks,
        engine.material_capacities,
        reschedule_hint=engine.reschedule_hint,
        dyelot_stock=engine.dyelot_stock,
    )

    # dyelot_stock reached the post-pass location intact.
    assert len(pipeline.dyelot_stock) == 2
    assert pipeline.dyelot_stock[1]["dyelot"] == "DL-002"

    # The real task still carries main_yarn_consumption AFTER _sanitize_dummy_tasks.
    real_post = next(t for t in pipeline.tasks if t["task_id"] == "K1")
    assert [y["kg"] for y in real_post["main_yarn_consumption"]] == [12.5, 3.0]


# ---------------------------------------------------------------------------
# Test 3 — legacy payload missing both fields → defaults, old fields unchanged
# ---------------------------------------------------------------------------

def test_legacy_payload_defaults():
    legacy_task = _task(task_id="K1", material_demands={"MAT_WHT": 2})
    payload = SolverPayload(**_payload(tasks=[legacy_task]))  # no new fields

    # Both new fields default to empty.
    assert payload.dyelot_stock == []
    assert payload.tasks[0].main_yarn_consumption == []

    # Old fields untouched.
    assert payload.tasks[0].material_demands == {"MAT_WHT": 2}
    assert payload.tasks[0].duration == 100
    assert payload.material_capacities == {}

    # And they survive real ingestion as empty defaults.
    engine = Engine(payload.model_dump(by_alias=False))
    assert engine.dyelot_stock == []
    assert engine.tasks[0]["main_yarn_consumption"] == []
    assert engine.tasks[0]["material_demands"] == {"MAT_WHT": 2}
