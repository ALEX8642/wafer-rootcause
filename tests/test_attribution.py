"""Attribution tests: grid coverage, statistics vs scipy/numpy references,
window well-formedness, scorer arithmetic, ground-truth firewall."""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from wafer_rootcause.attribution import ALPHA, SQL_DIR, score
from wafer_rootcause.labels import LABELS


def test_grid_covers_every_label_chamber_cell(attached_db, suspects_df):
    n_chambers = attached_db.execute("SELECT count(*) FROM chambers").fetchone()[0]
    n_wafers = attached_db.execute("SELECT count(*) FROM wafers").fetchone()[0]
    assert len(suspects_df) == len(LABELS) * n_chambers
    assert not suspects_df.duplicated(["label", "chamber_id"]).any()
    # every wafer visits every step, so chamber + rest always partition the line
    assert (suspects_df["n_cham"] + suspects_df["n_rest"] == n_wafers).all()
    assert (suspects_df["k_cham"] <= suspects_df["n_cham"]).all()


def test_z_matches_direct_computation(suspects_df):
    s = suspects_df
    pool = (s["k_cham"] + s["k_rest"]) / (s["n_cham"] + s["n_rest"])
    se = np.sqrt(pool * (1 - pool) * (1 / s["n_cham"] + 1 / s["n_rest"]))
    z = (s["k_cham"] / s["n_cham"] - s["k_rest"] / s["n_rest"]) / se
    ok = se > 0
    assert ok.any()
    assert np.allclose(s.loc[ok, "z"], z[ok], atol=1e-12)
    assert (s.loc[~ok, "z"] == 0).all()


def test_pvalues_match_scipy_within_approximation_bound(suspects_df):
    """SQL inlines Abramowitz–Stegun 7.1.26 (|erf error| <= 1.5e-7)."""
    p_ref = stats.norm.sf(suspects_df["z"])
    assert np.abs(suspects_df["p_value"] - p_ref).max() < 1.5e-7


def test_bh_matches_numpy_reference(suspects_df):
    p = suspects_df["p_value"].to_numpy()
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    q_ref = np.empty(m)
    q_ref[order] = q
    assert np.allclose(suspects_df["q_value"], q_ref, atol=1e-12)
    assert ((suspects_df["q_value"] <= ALPHA) == suspects_df["significant"]).all()


def test_suspect_rank_is_z_descending_per_label(suspects_df):
    for _, grp in suspects_df.groupby("label"):
        ordered = grp.sort_values("suspect_rank")
        assert (ordered["z"].diff().dropna() <= 1e-12).all()
        assert sorted(ordered["suspect_rank"]) == list(range(1, len(grp) + 1))


def test_windows_are_wellformed(attached_db, windows_df):
    lo, hi = attached_db.execute(
        "SELECT min(start_ts), max(end_ts) FROM wafer_process_history").fetchone()
    w = windows_df
    assert not w.duplicated(["label", "chamber_id"]).any()
    assert (w["win_start"] < w["win_end"]).all()
    # windows sit on the bucket spine, which cannot leave the horizon by
    # more than one bucket on each side
    pad = pd.Timedelta(hours=6)
    assert (w["win_start"] >= pd.Timestamp(lo) - pad).all()
    assert (w["win_end"] <= pd.Timestamp(hi) + pad).all()
    assert (w["n_buckets"] >= 2).all()          # params.min_run_buckets
    assert (w["excess_defects"] >= 5.0).all()   # params.min_excess
    assert (w["k_defect"] <= w["n_wafers"]).all()


def test_strong_faults_recovered_on_baseline_seed(attached_db, suspects_df,
                                                  windows_df):
    """Deterministic on the baseline config (seed 42): the two strongest
    planted faults (F1, F5) must be rank-1, BH-significant and localised.
    Scorer side — ground truth allowed here."""
    per_fault, summary = score(attached_db, suspects_df, windows_df)
    strong = per_fault[per_fault["p_acquire"] >= 0.6].set_index("fault_id")
    assert len(strong) >= 2
    assert strong["significant"].all()
    assert (strong["suspect_rank"] == 1).all()
    assert (strong["window_iou"] > 0.5).all()
    assert summary["n_faults"] == len(per_fault)
    # every BH-significant rank-1 suspect points at a real planted fault
    assert summary["precision@1"] == 1.0


def test_scorer_arithmetic_on_synthetic_case():
    """Hand-checkable score(): 2 faults, one recovered, one below the fold."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE ground_truth_faults (
            fault_id VARCHAR, chamber_id VARCHAR, signature_label VARCHAR,
            start_ts TIMESTAMP, end_ts TIMESTAMP, p_acquire DOUBLE);
        INSERT INTO ground_truth_faults VALUES
        ('FA', 'S1-T1-C1', 'Center', '2026-06-02 00:00', '2026-06-03 00:00', 0.5),
        ('FB', 'S2-T1-C1', 'Scratch', '2026-06-01 00:00', '2026-06-02 00:00', 0.2);
    """)
    suspects_df = pd.DataFrame({
        "chamber_id": ["S1-T1-C1", "S1-T1-C2", "S2-T1-C1", "S2-T1-C2"],
        "label": ["Center", "Center", "Scratch", "Scratch"],
        "excess": [0.3, 0.0, 0.01, 0.02],
        "z": [5.0, 0.1, 0.2, 0.4],
        "q_value": [1e-4, 0.9, 0.8, 0.7],
        "significant": [True, False, False, False],
        "suspect_rank": [1, 2, 2, 1],
    })
    windows_df = pd.DataFrame({  # 12 h detected vs 24 h true, 12 h overlap
        "chamber_id": ["S1-T1-C1"], "label": ["Center"],
        "win_start": [pd.Timestamp("2026-06-02 12:00")],
        "win_end": [pd.Timestamp("2026-06-03 00:00")],
    })
    per_fault, summary = score(con, suspects_df, windows_df)
    fa = per_fault.set_index("fault_id").loc["FA"]
    assert fa["window_iou"] == pytest.approx(0.5)
    assert fa["latency_hours"] == pytest.approx(12.0)
    assert math.isnan(per_fault.set_index("fault_id").loc["FB", "window_iou"])
    assert summary["recall@1"] == pytest.approx(0.5)   # FA yes, FB no
    assert summary["precision@1"] == pytest.approx(1.0)  # 1 flagged, 1 true
    assert summary["n_flagged@1"] == 1
    assert summary["n_localised"] == 1
    con.close()


def test_firewall_analysis_sql_never_reads_ground_truth():
    """Analysis queries (attr_*, eda_*) must never join ground_truth_*.
    Only the simulator and sql/score_faults.sql may."""
    for path in sorted(SQL_DIR.glob("*.sql")):
        if path.name in ("schema.sql", "score_faults.sql"):
            continue
        code = "\n".join(line.split("--", 1)[0]      # comments may (and do)
                         for line in path.read_text().splitlines())  # say it
        assert "ground_truth" not in code.lower(), (
            f"{path.name} references ground truth across the firewall")
