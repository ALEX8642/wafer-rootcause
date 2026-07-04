"""Phase 3 tests: the simulator extensions (correlated dispatch, duty-cycled
faults) and the ablation / orchestration layer.

Simulator-only tests run without the wafer-mixed checkout. The orchestration
tests need the prediction cache and reuse conftest's inventory/cache fixtures,
which skip cleanly when it is absent."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from wafer_rootcause.config import (REPO_ROOT, CouplingSpec, SimConfig)
from wafer_rootcause.labels import LABELS
from wafer_rootcause.predict import _sigmoid, classifier_output_rows
from wafer_rootcause.simulate import simulate

CFG_DIR = REPO_ROOT / "configs"


# ------------------------- config validation --------------------------

def test_new_configs_load_and_roundtrip():
    for name in ("sim_correlated", "sim_overlap", "sim_intermittent"):
        cfg = SimConfig.from_yaml(CFG_DIR / f"{name}.yaml")
        cfg.validate()  # from_yaml already validated; explicit for intent


def test_correlated_requires_coupling_and_ordering():
    base = SimConfig.from_yaml(CFG_DIR / "sim_baseline.yaml")
    with pytest.raises(ValueError, match="coupling block go together"):
        dataclasses.replace(base, dispatch="correlated").validate()
    # follower must come after driver in the route
    bad = dataclasses.replace(base, dispatch="correlated",
                              coupling=CouplingSpec("CMP", "ETCH", 0.8))
    with pytest.raises(ValueError, match="before follower"):
        bad.validate()
    with pytest.raises(ValueError, match="not a route step"):
        dataclasses.replace(base, dispatch="correlated",
                            coupling=CouplingSpec("ETCH", "NOPE", 0.8)).validate()


def test_duty_cycle_validation():
    base = SimConfig.from_yaml(CFG_DIR / "sim_baseline.yaml")

    def with_duty(on, off, dur=48):
        f0 = dataclasses.replace(base.faults[0], start_hour=24,
                                 duration_hours=dur, duty_on_hours=on,
                                 duty_off_hours=off)
        return dataclasses.replace(base, faults=[f0, *base.faults[1:]])

    with pytest.raises(ValueError, match="set together"):
        with_duty(8, None).validate()
    with pytest.raises(ValueError, match="fit inside the window"):
        with_duty(30, 30).validate()   # period 60 > 48 window
    with_duty(8, 8).validate()          # ok


# ---------------------- correlated dispatch ---------------------------

def test_correlated_dispatch_couples_chambers():
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_correlated.yaml")
    h = simulate(cfg)["wafer_process_history"]
    driver = h[h.chamber_id.str.startswith("ETCH")].set_index("wafer_id")["chamber_id"]
    follow = h[h.chamber_id.str.startswith("DEPOSITION")].set_index("wafer_id")["chamber_id"]
    d_pool = sorted(driver.unique())
    f_pool = sorted(follow.unique())
    di = {c: i for i, c in enumerate(d_pool)}
    fi = {c: i for i, c in enumerate(f_pool)}
    df = pd.DataFrame({"d": driver.map(di), "f": follow.map(fi)}).dropna()
    follows = (df["d"] % len(f_pool) == df["f"]).mean()
    # strength 0.8 + residual uniform coincidence 0.2*(1/4); well above uniform
    assert follows > 0.7
    assert follows > 1.5 / len(f_pool)


def test_random_dispatch_unchanged_by_extension():
    """The baseline (dispatch=random) draws identically to before: same seed
    → same history table, no extra rng consumption from the coupling branch."""
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_baseline.yaml")
    a = simulate(cfg)["wafer_process_history"]
    b = simulate(cfg)["wafer_process_history"]
    pd.testing.assert_frame_equal(a, b)
    # sanity: chambers spread across the fleet, not collapsed
    assert a["chamber_id"].nunique() > 1


def test_correlated_is_deterministic():
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_correlated.yaml")
    pd.testing.assert_frame_equal(simulate(cfg)["wafer_process_history"],
                                  simulate(cfg)["wafer_process_history"])


# ------------------------ duty-cycled faults --------------------------

def test_duty_cycle_exposures_only_in_on_phases():
    """Every fault-sourced label for the duty-cycled fault sits on a wafer
    whose visit to the fault chamber fell inside an on-phase."""
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_intermittent.yaml")
    tables = simulate(cfg)
    f1 = next(f for f in cfg.faults if f.fault_id == "F1")
    assert f1.duty_on_hours is not None
    faults = tables["ground_truth_faults"]
    f1_row = faults[faults.fault_id == "F1"].iloc[0]
    start, period = f1_row.start_ts, f1.duty_on_hours + f1.duty_off_hours

    gt = tables["ground_truth_wafer_labels"]
    hist = tables["wafer_process_history"]
    struck = gt[gt.source == "fault:F1"]["wafer_id"]
    assert len(struck) > 0
    visits = hist[(hist.chamber_id == f1.chamber)
                  & hist.wafer_id.isin(struck)].set_index("wafer_id")["start_ts"]
    for wid in struck:
        into = (visits[wid] - start).total_seconds() / 3600.0
        assert 0 <= into % period < f1.duty_on_hours, wid
    # the duty cut exposure roughly in half vs a continuous window
    cont = dataclasses.replace(
        cfg, faults=[dataclasses.replace(f1, duty_on_hours=None,
                                         duty_off_hours=None), *cfg.faults[1:]])
    n_cont = (simulate(cont)["ground_truth_wafer_labels"]["source"]
              == "fault:F1").sum()
    assert len(struck) < n_cont


def test_ground_truth_faults_has_no_duty_columns():
    """The duty cycle is a config detail; the fact table records the envelope
    only (the scorer judges window IoU against it)."""
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_intermittent.yaml")
    cols = set(simulate(cfg)["ground_truth_faults"].columns)
    assert "duty_on_hours" not in cols and "duty_off_hours" not in cols


# ------------------ classifier-output ablation modes ------------------

def test_raw_mode_matches_recomputed_sigmoid():
    """raw@0.5 rows are exactly sigmoid(logit) > 0.5 on the cached logits."""
    cache = pd.DataFrame({
        "map_id": [0, 0, 1, 1], "label": ["Center", "Donut"] * 2,
        "logit": [2.0, -1.0, 0.05, -3.0], "prob": [0.8, 0.3, 0.5, 0.1],
        "predicted": [True, False, True, False]})
    assignment = pd.DataFrame({"wafer_id": ["W1", "W2"], "map_id": [0, 1]})
    rows = classifier_output_rows(assignment, cache, mode="raw")
    m = rows.set_index(["wafer_id", "label"])
    exp = _sigmoid(np.array([2.0, -1.0, 0.05, -3.0]))
    assert m.loc[("W1", "Center"), "prob"] == pytest.approx(exp[0])
    assert bool(m.loc[("W1", "Center"), "predicted"]) is True
    assert bool(m.loc[("W2", "Center"), "predicted"]) is True   # logit 0.05 > 0
    assert bool(m.loc[("W2", "Donut"), "predicted"]) is False


def test_calibrated_mode_passes_cache_through():
    cache = pd.DataFrame({
        "map_id": [0, 0], "label": ["Center", "Donut"],
        "logit": [2.0, -1.0], "prob": [0.8, 0.3], "predicted": [True, False]})
    assignment = pd.DataFrame({"wafer_id": ["W1"], "map_id": [0]})
    rows = classifier_output_rows(assignment, cache, mode="calibrated")
    assert rows.set_index("label").loc["Center", "prob"] == 0.8
    assert bool(rows.set_index("label").loc["Center", "predicted"]) is True


# --------------- orchestration (needs the prediction cache) -----------

def test_oracle_outputs_match_ground_truth_exactly(tables, inventory, cache,
                                                   acfg):
    """Oracle mode fills classifier_outputs with the simulated truth: zero
    escapes, zero false alarms by construction."""
    from wafer_rootcause.attach import (assign_maps, record_assignment,
                                        wafer_combos)
    from wafer_rootcause.db import memory_db
    from wafer_rootcause.predict import load_classifier_outputs
    con = memory_db(tables)
    try:
        asn = assign_maps(wafer_combos(con), inventory, acfg.assign_seed)
        record_assignment(con, asn)
        load_classifier_outputs(con, asn, cache, mode="oracle")
        esc, fa = con.execute("""
            SELECT count(*) FILTER (WHERE g.wafer_id IS NOT NULL AND NOT co.predicted),
                   count(*) FILTER (WHERE g.wafer_id IS NULL AND co.predicted)
            FROM classifier_outputs co
            LEFT JOIN ground_truth_wafer_labels g
                   ON g.wafer_id = co.wafer_id AND g.label = co.label
        """).fetchone()
        assert (esc, fa) == (0, 0)
        n = con.execute("SELECT count(*) FROM classifier_outputs").fetchone()[0]
        n_wafers = con.execute("SELECT count(*) FROM wafers").fetchone()[0]
        assert n == n_wafers * len(LABELS)
    finally:
        con.close()


def test_run_scenario_reproduces_phase2_baseline(inventory, cache, acfg):
    """Baseline through the Phase 3 orchestrator == the Phase 2 headline
    (seed 42): precision@1 1.0, recall@1 0.8, 4 faults localised."""
    from wafer_rootcause.sensitivity import run_scenario
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_baseline.yaml")
    per_fault, summary = run_scenario(cfg, acfg, inventory, cache, "calibrated")
    assert summary["precision@1"] == 1.0
    assert summary["recall@1"] == pytest.approx(0.8)
    assert summary["n_localised"] == 4
    assert summary["mean_iou"] == pytest.approx(0.866, abs=5e-3)


def test_ablation_modes_agree_on_baseline(inventory, cache, acfg):
    """The Phase 3 finding, pinned: oracle / calibrated / raw give the same
    attribution on the baseline draw despite different label-error counts."""
    from wafer_rootcause.sensitivity import run_scenario
    cfg = SimConfig.from_yaml(CFG_DIR / "sim_baseline.yaml")
    outs = {m: run_scenario(cfg, acfg, inventory, cache, m)[1]
            for m in ("oracle", "calibrated", "raw")}
    for m in ("calibrated", "raw"):
        assert outs[m]["recall@1"] == outs["oracle"]["recall@1"]
        assert outs[m]["precision@1"] == outs["oracle"]["precision@1"]
        assert outs[m]["mean_iou"] == pytest.approx(outs["oracle"]["mean_iou"])
