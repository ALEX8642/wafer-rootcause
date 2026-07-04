"""attach_and_predict.py — Phase 1 pipeline: attach maps, predict, load DB.

Usage (after scripts/build_db.py):
    python scripts/attach_and_predict.py [--config configs/attach_baseline.yaml]
                                         [--db PATH] [--rebuild-cache]

Steps:
  1. Assign each simulated wafer a wafer-mixed TEST-split map whose true
     label set matches the wafer's simulated label set (without
     replacement); persist to outputs/map_assignment.parquet and
     inspections.map_id.
  2. Ensure the prediction cache (full test split, ~3 min CPU once);
     reused untouched on every later run.
  3. Load classifier_outputs (8 rows per wafer: calibrated prob + tau
     decision) from assignment × cache.

The closing summary compares predictions against simulated ground truth.
That is scorer-side output (like build_db.py's prevalence print) — the
analysis SQL in sql/ never touches ground truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from wafer_rootcause.attach import (assign_maps, load_test_inventory,  # noqa: E402
                                    record_assignment, wafer_combos)
from wafer_rootcause.config import REPO_ROOT, AttachConfig  # noqa: E402
from wafer_rootcause.db import write_parquet  # noqa: E402
from wafer_rootcause.labels import LABELS  # noqa: E402
from wafer_rootcause.predict import ensure_prediction_cache, load_classifier_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="wafer-rootcause Phase 1 pipeline")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "attach_baseline.yaml"))
    parser.add_argument("--db", default=None, help="override cfg.db_path")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="re-run test-split inference even if the cache exists")
    args = parser.parse_args()

    acfg = AttachConfig.from_yaml(args.config)
    db_path = Path(args.db) if args.db else REPO_ROOT / acfg.db_path
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found — run scripts/build_db.py first")

    con = duckdb.connect(str(db_path))
    try:
        # --- 1. map attachment (computed, not yet persisted) ---
        combos = wafer_combos(con)
        inventory = load_test_inventory(acfg)
        assignment = assign_maps(combos, inventory, acfg.assign_seed)
        used = combos.value_counts()
        have = inventory["combo"].value_counts()
        worst = (used / have.reindex(used.index)).sort_values(ascending=False)
        print(f"Assigned {len(assignment):,} maps "
              f"({len(used)} combos; tightest inventory: "
              f"{worst.index[0]} {worst.iloc[0]:.0%} of "
              f"{int(have[worst.index[0]])} test maps)")

        # --- 2. prediction cache (the slow, failure-prone step) ---
        cache = ensure_prediction_cache(acfg, REPO_ROOT / acfg.cache_path,
                                        rebuild=args.rebuild_cache)

        # --- 3. persist: all DB/artifact mutations happen together, after
        # everything that can fail slowly, so an aborted inference pass
        # can't leave a new assignment beside old classifier_outputs ---
        write_parquet(assignment, REPO_ROOT / acfg.assignment_path)
        record_assignment(con, assignment)
        n = load_classifier_outputs(con, assignment, cache)
        print(f"classifier_outputs loaded: {n:,} rows "
              f"({n // len(LABELS):,} wafers x {len(LABELS)} labels)")

        # --- scorer-side summary: predictions vs simulated ground truth ---
        print("\nPrediction vs ground truth (sim summary — scorer side only):")
        print(f"  {'label':12s} {'true':>5s} {'pred':>5s} {'escapes':>8s} {'false_alarms':>13s}")
        rows = con.execute("""
            SELECT co.label,
                   count(*) FILTER (WHERE g.wafer_id IS NOT NULL)   AS n_true,
                   count(*) FILTER (WHERE co.predicted)             AS n_pred,
                   count(*) FILTER (WHERE g.wafer_id IS NOT NULL
                                      AND NOT co.predicted)         AS escapes,
                   count(*) FILTER (WHERE g.wafer_id IS NULL
                                      AND co.predicted)             AS false_alarms
            FROM classifier_outputs co
            LEFT JOIN ground_truth_wafer_labels g
                   ON g.wafer_id = co.wafer_id AND g.label = co.label
            GROUP BY co.label
        """).fetchall()
        order = {name: i for i, name in enumerate(LABELS)}
        tot_e = tot_f = 0
        for label, n_true, n_pred, esc, fa in sorted(rows, key=lambda r: order[r[0]]):
            print(f"  {label:12s} {n_true:>5d} {n_pred:>5d} {esc:>8d} {fa:>13d}")
            tot_e += esc
            tot_f += fa
        print(f"  {'total':12s} {'':>5s} {'':>5s} {tot_e:>8d} {tot_f:>13d}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
