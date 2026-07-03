"""DB-level tests: referential integrity, route coverage, monotonicity.

Checks run in SQL against the built DuckDB file — the same layer the
Phase 2 analysis queries will run in.
"""
from __future__ import annotations


def _one(db, query: str) -> int:
    return db.execute(query).fetchone()[0]


def test_no_orphan_foreign_keys(db):
    orphan_checks = {
        "wafers→lots": """
            SELECT count(*) FROM wafers w
            LEFT JOIN lots l USING (lot_id) WHERE l.lot_id IS NULL""",
        "tools→steps": """
            SELECT count(*) FROM tools t
            LEFT JOIN process_steps s USING (step_id) WHERE s.step_id IS NULL""",
        "chambers→tools": """
            SELECT count(*) FROM chambers c
            LEFT JOIN tools t USING (tool_id) WHERE t.tool_id IS NULL""",
        "history→wafers": """
            SELECT count(*) FROM wafer_process_history h
            LEFT JOIN wafers w USING (wafer_id) WHERE w.wafer_id IS NULL""",
        "history→chambers": """
            SELECT count(*) FROM wafer_process_history h
            LEFT JOIN chambers c USING (chamber_id) WHERE c.chamber_id IS NULL""",
        "inspections→wafers": """
            SELECT count(*) FROM inspections i
            LEFT JOIN wafers w USING (wafer_id) WHERE w.wafer_id IS NULL""",
        "faults→chambers": """
            SELECT count(*) FROM ground_truth_faults f
            LEFT JOIN chambers c USING (chamber_id) WHERE c.chamber_id IS NULL""",
        "gt_labels→wafers": """
            SELECT count(*) FROM ground_truth_wafer_labels g
            LEFT JOIN wafers w USING (wafer_id) WHERE w.wafer_id IS NULL""",
    }
    for name, query in orphan_checks.items():
        assert _one(db, query) == 0, f"orphans in {name}"


def test_history_covers_full_route(db):
    n_steps = _one(db, "SELECT count(*) FROM process_steps")
    incomplete = _one(db, f"""
        SELECT count(*) FROM (
            SELECT wafer_id FROM wafer_process_history
            GROUP BY wafer_id HAVING count(DISTINCT step_id) <> {n_steps}
        )""")
    assert incomplete == 0
    n_wafers = _one(db, "SELECT count(*) FROM wafers")
    assert _one(db, "SELECT count(*) FROM wafer_process_history") == n_wafers * n_steps


def test_history_chamber_belongs_to_step(db):
    mismatched = _one(db, """
        SELECT count(*) FROM wafer_process_history h
        JOIN chambers c USING (chamber_id)
        JOIN tools t USING (tool_id)
        WHERE t.step_id <> h.step_id""")
    assert mismatched == 0


def test_timestamps_monotone_along_route(db):
    violations = _one(db, """
        SELECT count(*) FROM (
            SELECT h.wafer_id,
                   h.start_ts,
                   h.end_ts,
                   lag(h.end_ts) OVER (
                       PARTITION BY h.wafer_id ORDER BY s.step_order
                   ) AS prev_end
            FROM wafer_process_history h
            JOIN process_steps s USING (step_id)
        )
        WHERE end_ts <= start_ts OR (prev_end IS NOT NULL AND start_ts < prev_end)""")
    assert violations == 0


def test_one_inspection_per_wafer_after_last_step(db):
    n_wafers = _one(db, "SELECT count(*) FROM wafers")
    assert _one(db, "SELECT count(DISTINCT wafer_id) FROM inspections") == n_wafers
    early = _one(db, """
        SELECT count(*) FROM inspections i
        JOIN (SELECT wafer_id, max(end_ts) AS last_end
              FROM wafer_process_history GROUP BY wafer_id) h USING (wafer_id)
        WHERE i.inspect_ts <= h.last_end""")
    assert early == 0


def test_classifier_outputs_empty_until_phase1(db):
    assert _one(db, "SELECT count(*) FROM classifier_outputs") == 0
