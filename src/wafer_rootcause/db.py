"""db.py — create the DuckDB file from sql/schema.sql and load sim tables."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wafer_rootcause.config import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"

# Insert order respects foreign keys. classifier_outputs is created by the
# schema but stays empty until Phase 1.
TABLE_ORDER = [
    "lots", "wafers", "process_steps", "tools", "chambers",
    "wafer_process_history", "inspections",
    "ground_truth_faults", "ground_truth_wafer_labels",
]


def build_db(tables: dict[str, pd.DataFrame], db_path: str | Path,
             schema_path: str | Path = SCHEMA_PATH) -> None:
    """Write `tables` into a fresh DuckDB file at `db_path`."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(Path(schema_path).read_text())
        for name in TABLE_ORDER:
            df = tables[name]  # noqa: F841 — resolved by DuckDB's replacement scan
            con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM df")
    finally:
        con.close()


def connect(db_path: str | Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=read_only)
