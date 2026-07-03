"""Simulator-level tests: determinism, combo validity, fault effect."""
from __future__ import annotations

import dataclasses
import math

import pandas as pd

from wafer_rootcause.labels import is_valid_combo
from wafer_rootcause.simulate import simulate


def test_determinism_same_seed(cfg, tables):
    again = simulate(cfg)
    assert tables.keys() == again.keys()
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name])


def test_different_seed_differs(cfg, tables):
    other = simulate(dataclasses.replace(cfg, seed=cfg.seed + 1))
    assert not tables["wafer_process_history"].equals(
        other["wafer_process_history"])


def test_label_sets_are_valid_combos(tables):
    per_wafer = tables["ground_truth_wafer_labels"].groupby("wafer_id")["label"]
    for _, labels in per_wafer:
        assert is_valid_combo(set(labels))


def test_fault_windows_inside_horizon(tables):
    """Pins the shipped config at its seed: the horizon end is stochastic
    (sum of exponential interarrivals), so changing seed or arrival process
    in the YAML requires retuning the fault windows there — this test is
    what catches a drifted config, not a simulator property."""
    hist = tables["wafer_process_history"]
    lo, hi = hist["start_ts"].min(), hist["end_ts"].max()
    faults = tables["ground_truth_faults"]
    assert (faults["start_ts"] >= lo).all()
    assert (faults["end_ts"] <= hi).all()


def test_fault_effect_sanity(cfg, tables):
    """Exposed wafers carry the signature at ~p_acquire above the base rate.

    Expected rate among exposed = p + (1-p) * base, where base is the
    empirical rate among unexposed wafers at the same step. Tolerance is
    3 sigma binomial plus slack for exclusivity-group collisions.

    Like test_fault_windows_inside_horizon, the >= 20 exposed-wafer floor
    pins the shipped config at its seed — retune fault windows in the YAML
    if a config change starves a fault of exposed wafers.
    """
    hist = tables["wafer_process_history"]
    gt = tables["ground_truth_wafer_labels"]
    n_wafers = len(tables["wafers"])

    for f in tables["ground_truth_faults"].itertuples():
        in_chamber = hist["chamber_id"] == f.chamber_id
        in_window = hist["start_ts"].between(f.start_ts, f.end_ts)
        exposed = set(hist.loc[in_chamber & in_window, "wafer_id"])
        assert len(exposed) >= 20, f"{f.fault_id}: too few exposed wafers to test"

        carriers = set(gt.loc[gt["label"] == f.signature_label, "wafer_id"])
        rate_exposed = len(exposed & carriers) / len(exposed)
        base = len(carriers - exposed) / (n_wafers - len(exposed))
        expected = f.p_acquire + (1 - f.p_acquire) * base
        sigma = math.sqrt(expected * (1 - expected) / len(exposed))
        assert abs(rate_exposed - expected) < 3 * sigma + 0.03, (
            f"{f.fault_id}: rate {rate_exposed:.2f}, expected {expected:.2f}"
        )
        assert rate_exposed > base + f.p_acquire / 2


def test_near_full_and_random_only_on_clean(tables):
    gt = tables["ground_truth_wafer_labels"]
    isolated = gt[gt["label"].isin(["Near-full", "Random"])]["wafer_id"]
    counts = gt.groupby("wafer_id").size()
    assert (counts.loc[isolated] == 1).all()
