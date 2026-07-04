"""attribution.py — Phase 2: run the attribution SQL and score it.

The analytics live in sql/attr_*.sql (commonality z-tests + BH, window
localisation); this module only executes those files and turns the
scorer's per-fault rows into the headline metrics. The firewall runs
through the middle of this file: `suspects`/`windows`/`bucket_rates` are
analysis side (classifier_outputs only), `score` joins ground truth and
is scorer side.

Metric definitions (planted faults = ground_truth_faults rows):
  recall@k     — share of planted faults whose chamber is BH-significant
                 and ranked in the top k suspects for its signature label.
  precision@k  — of all BH-significant suspects ranked in any label's
                 top k, the share that are planted faults. Denominator is
                 what an engineer would actually walk down: the flagged
                 list, not the full grid.
  window IoU / latency — for faults recovered@K_DEFAULT that also got an
                 excursion window: interval IoU with the true window, and
                 detected-start minus true-start in hours (negative =
                 flagged early, the rolling window smears one bucket).
"""
from __future__ import annotations

import duckdb
import pandas as pd

from wafer_rootcause.config import REPO_ROOT

SQL_DIR = REPO_ROOT / "sql"

ALPHA = 0.05      # BH FDR level — must match params.alpha in attr_suspects.sql
K_DEFAULT = 3     # the "@3" in precision/recall@3, and the window-metric gate


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Execute sql/<name>.sql and return the result frame."""
    return con.execute((SQL_DIR / f"{name}.sql").read_text()).df()


# --------------------------- analysis side ---------------------------

def suspects(con) -> pd.DataFrame:
    """Full label x chamber grid: z, p, BH q, per-label suspect_rank."""
    return run_sql(con, "attr_suspects")


def windows(con) -> pd.DataFrame:
    """Best excursion window per (chamber, label) cell that has one."""
    return run_sql(con, "attr_windows")


def bucket_rates(con) -> pd.DataFrame:
    """Chamber x label x time-bucket rates (heatmap / inspection material)."""
    return run_sql(con, "attr_bucket_rates")


# ---------------------------- scorer side ----------------------------

def score(con, suspects_df: pd.DataFrame,
          windows_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join analysis verdicts against ground_truth_faults.

    Returns (per_fault, summary): one scored row per planted fault, and
    the headline metrics dict (see module docstring for definitions).
    """
    con.register("suspects", suspects_df)
    con.register("windows", windows_df)
    try:
        per_fault = run_sql(con, "score_faults")
    finally:
        con.unregister("suspects")
        con.unregister("windows")
    if len(per_fault) != con.execute(
            "SELECT count(*) FROM ground_truth_faults").fetchone()[0]:
        raise RuntimeError("scorer lost faults in the suspects join — "
                           "grid is missing (chamber, label) cells")

    fault_keys = set(zip(per_fault["chamber_id"], per_fault["label"]))
    summary: dict[str, float] = {"n_faults": len(per_fault)}
    for k in (1, K_DEFAULT):
        hit = per_fault["significant"] & (per_fault["suspect_rank"] <= k)
        flagged = suspects_df[suspects_df["significant"]
                              & (suspects_df["suspect_rank"] <= k)]
        true_pos = [(c, l) in fault_keys
                    for c, l in zip(flagged["chamber_id"], flagged["label"])]
        summary[f"recall@{k}"] = hit.mean()
        summary[f"n_flagged@{k}"] = len(flagged)
        summary[f"precision@{k}"] = (sum(true_pos) / len(flagged)
                                     if len(flagged) else float("nan"))

    found = per_fault[per_fault["significant"]
                      & (per_fault["suspect_rank"] <= K_DEFAULT)
                      & per_fault["window_iou"].notna()]
    summary["n_localised"] = len(found)
    summary["mean_iou"] = found["window_iou"].mean()
    summary["mean_abs_latency_h"] = found["latency_hours"].abs().mean()
    return per_fault, summary
