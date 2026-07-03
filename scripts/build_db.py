"""build_db.py — simulate the fab and load the DuckDB file.

Usage:
    python scripts/build_db.py [--config configs/sim_baseline.yaml] [--db PATH]

Deterministic: same config + seed → byte-identical table contents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_rootcause.config import REPO_ROOT, SimConfig
from wafer_rootcause.db import build_db, connect
from wafer_rootcause.simulate import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="wafer-rootcause DB builder")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "sim_baseline.yaml"))
    parser.add_argument("--db", default=None, help="override cfg.db_path")
    args = parser.parse_args()

    cfg = SimConfig.from_yaml(args.config)
    db_path = Path(args.db) if args.db else REPO_ROOT / cfg.db_path

    tables = simulate(cfg)
    build_db(tables, db_path)

    con = connect(db_path)
    try:
        print(f"Built {db_path}")
        for name in ["lots", "wafers", "wafer_process_history", "inspections",
                     "ground_truth_faults", "ground_truth_wafer_labels"]:
            n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"  {name:28s} {n:>7,}")
        print("\nTrue label prevalence (ground truth — sim summary only):")
        rows = con.execute("""
            SELECT label, count(*) AS n_wafers,
                   round(count(*) * 100.0 / (SELECT count(*) FROM wafers), 1) AS pct
            FROM ground_truth_wafer_labels GROUP BY label ORDER BY n_wafers DESC
        """).fetchall()
        for label, n, pct in rows:
            print(f"  {label:12s} {n:>5,}  ({pct}%)")
        n_clean = con.execute("""
            SELECT count(*) FROM wafers w
            WHERE NOT EXISTS (SELECT 1 FROM ground_truth_wafer_labels g
                              WHERE g.wafer_id = w.wafer_id)
        """).fetchone()[0]
        print(f"  {'(clean)':12s} {n_clean:>5,}")
        lo, hi = con.execute(
            "SELECT min(start_ts), max(end_ts) FROM wafer_process_history").fetchone()
        print(f"\nHorizon: {lo} → {hi}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
