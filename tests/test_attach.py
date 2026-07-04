"""Map-attachment tests: label match, no reuse, split provenance, determinism."""
from __future__ import annotations

import numpy as np

from wafer_rootcause.attach import assign_maps, wafer_combos


def test_every_wafer_assigned_exactly_once(assignment, tables):
    assert len(assignment) == len(tables["wafers"])
    assert assignment["wafer_id"].is_unique
    assert assignment["map_id"].notna().all()


def test_no_map_reused(assignment):
    assert assignment["map_id"].is_unique


def test_assigned_maps_match_simulated_labels(db, assignment, inventory):
    """Wafer's simulated label set == assigned map's TRUE label set."""
    merged = assignment.merge(inventory, on="map_id", how="left")
    combos = wafer_combos(db)
    assert merged["combo"].notna().all()  # every map_id is a test-split map
    got = merged.set_index("wafer_id")["combo"]
    expected = combos.loc[got.index]
    assert (got == expected).all()


def test_maps_come_from_test_split(assignment, acfg):
    """Independent of inventory: map_ids sit in wafer-mixed's persisted test split."""
    test_idx = np.load(acfg.splits_path)["test"]
    assert set(assignment["map_id"]) <= set(test_idx.tolist())


def test_inspections_carry_assignment(attached_db, assignment):
    """record_assignment persists exactly the sampled map per wafer."""
    df = attached_db.execute(
        "SELECT wafer_id, map_id FROM inspections ORDER BY wafer_id").df()
    exp = assignment.sort_values("wafer_id", ignore_index=True)
    assert (df["wafer_id"] == exp["wafer_id"]).all()
    assert (df["map_id"].to_numpy() == exp["map_id"].to_numpy()).all()


def test_assignment_deterministic(db, inventory, acfg):
    combos = wafer_combos(db)
    a = assign_maps(combos, inventory, acfg.assign_seed)
    b = assign_maps(combos, inventory, acfg.assign_seed)
    assert a.equals(b)
    c = assign_maps(combos, inventory, acfg.assign_seed + 1)
    assert not a["map_id"].equals(c["map_id"])
