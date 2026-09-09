"""Affinity fix part 2: derive each machine's current yarn config from its last
pinned task when the resource carries none.

On a re-schedule after convert, machine_materials is empty so
resources[].color_config arrives blank; without a current config the affinity
scores every machine as cold-start and won't pull a same-yarn new order onto the
machine whose converted (pinned) task already commits it to that yarn. The
pinned task now carries the real config (Go side), so it is the machine's
committed yarn.
"""
from app.engine.shared import seed_resource_config_from_pins


def _pin(task_id, machine, end, cc="", dii=""):
    return {
        "task_id": task_id,
        "operation": "knitting",
        "is_pinned": True,
        "pinned_machine_id": machine,
        "pinned_end_time": end,
        "color_config": cc,
        "design_item_id": dii,
    }


def test_fills_empty_resource_from_pin():
    tasks = [_pin("PIN_1", "SK01", end=300, cc="CY02-WHTC026:5", dii="fbm_KHL")]
    resources = {"SK01": {"id": "SK01", "color_config": "", "design_item_id": ""}}

    filled = seed_resource_config_from_pins(tasks, resources)

    assert filled == 1
    assert resources["SK01"]["color_config"] == "CY02-WHTC026:5"
    assert resources["SK01"]["design_item_id"] == "fbm_KHL"


def test_last_pin_wins():
    # Two pins on SK01: the later one (max pinned_end_time) is the current yarn.
    tasks = [
        _pin("PIN_early", "SK01", end=100, cc="CY02-MCHC304:2"),
        _pin("PIN_late", "SK01", end=800, cc="CY02-WHTC026:5"),
    ]
    resources = {"SK01": {"id": "SK01", "color_config": ""}}

    seed_resource_config_from_pins(tasks, resources)

    assert resources["SK01"]["color_config"] == "CY02-WHTC026:5"


def test_does_not_overwrite_existing_config():
    # A real Go-sent config must never be clobbered by a derived one.
    tasks = [_pin("PIN_1", "SK01", end=300, cc="CY02-WHTC026:5")]
    resources = {"SK01": {"id": "SK01", "color_config": "AY02-PNK016:3"}}

    filled = seed_resource_config_from_pins(tasks, resources)

    assert filled == 0
    assert resources["SK01"]["color_config"] == "AY02-PNK016:3"


def test_ignores_pins_without_machine_or_config():
    tasks = [
        _pin("PIN_nomachine", None, end=300, cc="CY02-WHTC026:5"),
        _pin("PIN_noconfig", "SK01", end=300, cc="", dii=""),
        {"task_id": "FREE", "is_pinned": False, "pinned_machine_id": "SK01",
         "color_config": "CY02-WHTC026:5"},
    ]
    resources = {"SK01": {"id": "SK01", "color_config": ""}}

    filled = seed_resource_config_from_pins(tasks, resources)

    assert filled == 0
    assert resources["SK01"]["color_config"] == ""


def test_noop_when_no_pins():
    resources = {"SK01": {"id": "SK01", "color_config": ""}}
    assert seed_resource_config_from_pins([], resources) == 0
    assert resources["SK01"]["color_config"] == ""
