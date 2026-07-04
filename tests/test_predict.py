"""Prediction tests: cache round-trip, tau rule, and a live model spot-check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafer_rootcause.labels import LABELS
from wafer_rootcause.predict import load_thresholds, wafer_mixed_modules


def test_one_row_per_wafer_per_label(attached_db):
    bad = attached_db.execute(f"""
        SELECT wafer_id FROM classifier_outputs
        GROUP BY wafer_id HAVING count(*) <> {len(LABELS)}
    """).fetchall()
    assert bad == []
    n_wafers, n_rows = attached_db.execute(
        "SELECT count(DISTINCT wafer_id), count(*) FROM classifier_outputs"
    ).fetchone()
    assert n_rows == len(LABELS) * n_wafers


def test_roundtrip_db_matches_cache(attached_db, assignment, cache):
    """DB rows == parquet cache joined through the persisted assignment."""
    expected = (assignment.merge(cache, on="map_id")
                [["wafer_id", "label", "prob", "predicted"]]
                .sort_values(["wafer_id", "label"], ignore_index=True))
    got = attached_db.execute("""
        SELECT wafer_id, label, prob, predicted
        FROM classifier_outputs ORDER BY wafer_id, label
    """).df()
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_predicted_equals_prob_gt_tau(attached_db, acfg):
    """The stored decision is exactly wafer-mixed's rule: prob > per-label tau."""
    _, tau = load_thresholds(acfg.thresholds_path)
    df = attached_db.execute(
        "SELECT label, prob, predicted FROM classifier_outputs").df()
    tau_of = dict(zip(LABELS, tau))
    expected = df["prob"] > df["label"].map(tau_of)
    assert (df["predicted"] == expected).all()


def test_spot_check_matches_wafer_mixed_pipeline(attached_db, assignment, acfg):
    """Recompute a batch end-to-end with wafer-mixed's own functions.

    Loads the checkpoint, runs encode→resize→model→scale_probs→
    predict_multihot on 12 assigned maps and compares against the DB rows.
    Probabilities get a small tolerance (batch composition differs from the
    cache run, so kernel reduction order may differ); decisions must match
    wherever the probability is decisively away from tau — a prob within
    recompute tolerance of its threshold can legitimately flip either way.
    """
    if not acfg.checkpoint_path.exists():
        pytest.skip(f"checkpoint not available ({acfg.checkpoint_path})")
    import torch
    from torch.utils.data import DataLoader

    wm = wafer_mixed_modules(acfg.mixed_root)
    cfg = wm.MixedConfig(device="cpu", num_workers=0)
    cfg.device = "cpu"  # a WAFER_DEVICE leftover must not retarget the test
    maps, labels = wm.data.load_raw(cfg.data_root)
    model, _ = wm.model.load_checkpoint_model(cfg, acfg.checkpoint_path)

    # deterministic spread across the assignment, not just the first lot
    sample = assignment.sort_values("wafer_id").iloc[:: max(1, len(assignment) // 12)][:12]
    map_ids = sample["map_id"].to_numpy()
    ds = wm.data.MixedWaferDataset(maps[map_ids], labels[map_ids],
                                   cfg.input_size, augment=False)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    with torch.no_grad():
        logits = model(next(iter(loader))[0]).numpy()

    T, tau = load_thresholds(acfg.thresholds_path)
    probs = wm.calibrate.scale_probs(logits, T)      # rows in sample order
    pred = wm.metrics.predict_multihot(probs, tau).astype(bool)

    db = attached_db.execute("""
        SELECT wafer_id, label, prob, predicted FROM classifier_outputs
        WHERE wafer_id IN (SELECT unnest(?))
    """, [sample["wafer_id"].tolist()]).df()
    # reindex by LABELS, not by SQL collation order
    got_prob = (db.pivot(index="wafer_id", columns="label", values="prob")
                  .reindex(index=sample["wafer_id"], columns=LABELS).to_numpy())
    got_pred = (db.pivot(index="wafer_id", columns="label", values="predicted")
                  .reindex(index=sample["wafer_id"], columns=LABELS)
                  .to_numpy().astype(bool))
    atol = 1e-4
    np.testing.assert_allclose(got_prob, probs, atol=atol)
    decisive = np.abs(probs - tau) > atol
    np.testing.assert_array_equal(got_pred[decisive], pred[decisive])
