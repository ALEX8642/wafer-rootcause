"""predict.py — run the wafer-mixed checkpoint over test-split maps (CPU).

Reuses wafer-mixed's OWN modules (model, data, evaluate, calibrate,
metrics) from a sibling checkout instead of re-implementing the
architecture, encoding, temperature scaling or threshold rule: what lands
in classifier_outputs is definitionally what wafer-mixed produces, and the
Phase 1 spot-check test compares against those functions directly.

Caching: inference runs once over the FULL test split (7,603 maps, ~3 min
CPU) and is cached to parquet keyed by map_id. Any DB rebuild, re-seed or
Phase 3 re-configuration only re-joins the cache — it never re-runs the
model. Raw logits are cached alongside the calibrated probabilities so
Phase 3's raw-@0.5 ablation needs no new inference either.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd

from wafer_rootcause.config import AttachConfig
from wafer_rootcause.db import read_parquet, write_parquet
from wafer_rootcause.labels import LABELS


def wafer_mixed_modules(mixed_root: Path) -> SimpleNamespace:
    """Import wafer-mixed's package from a sibling checkout.

    Single owner of the cross-repo dependency: everything downstream goes
    through the namespace returned here. Two guards keep the bridge honest:
    the imported package must actually live under `mixed_root` (sys.modules
    pins the first import for the whole process, and an installed
    wafer_mixed could shadow the checkout), and its label ordering must
    equal this repo's LABELS — every logit column, threshold vector and
    combo string indexes by that order, and a silent drift would scramble
    all of them while staying self-consistent in tests.
    """
    src = mixed_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(
            f"{src} not found — set wafer_mixed_root in configs/attach_baseline.yaml "
            "to a wafer-mixed checkout")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from wafer_mixed import calibrate, data, evaluate, metrics, model
    from wafer_mixed.config import MixedConfig
    loaded_from = Path(data.__file__).resolve()
    if not loaded_from.is_relative_to(src.resolve()):
        raise RuntimeError(
            f"wafer_mixed resolved to {loaded_from}, not the configured "
            f"checkout {src} — another import (installed package or earlier "
            "config) got there first")
    if list(data.LABEL_NAMES) != LABELS:
        raise RuntimeError(
            f"label-order drift: wafer_mixed.data.LABEL_NAMES "
            f"{list(data.LABEL_NAMES)} != wafer_rootcause LABELS {LABELS}")
    return SimpleNamespace(calibrate=calibrate, data=data, evaluate=evaluate,
                           metrics=metrics, model=model, MixedConfig=MixedConfig)


def load_thresholds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(temperatures, taus) in LABELS order from wafer-mixed thresholds.json.

    The file is self-contained on purpose (wafer-mixed embeds the per-label
    temperatures the taus were tuned on); refuse label-set drift loudly.
    """
    raw = json.loads(path.read_text())
    temps, taus = raw["_temperatures"], raw["thresholds"]
    for name, d in (("_temperatures", temps), ("thresholds", taus)):
        if set(d) != set(LABELS):
            raise ValueError(f"{path}: {name} labels {sorted(d)} != {sorted(LABELS)}")
    return (np.array([temps[n] for n in LABELS]),
            np.array([taus[n] for n in LABELS]))


def build_prediction_cache(acfg: AttachConfig) -> pd.DataFrame:
    """One CPU pass over the full test split → long DataFrame.

    Columns: map_id, label, logit (raw), prob (temperature-scaled),
    predicted (prob > per-label tau). Exactly wafer-mixed's calibrated
    decision path: collect_logits → scale_probs → predict_multihot.
    """
    import torch
    from torch.utils.data import DataLoader

    wm = wafer_mixed_modules(acfg.mixed_root)
    cfg = wm.MixedConfig(device="cpu", batch_size=acfg.batch_size,
                         num_workers=acfg.num_workers)
    # This repo is CPU-only by policy. MixedConfig gives the WAFER_DEVICE
    # env var top priority (a training-shell leftover could resolve to
    # cuda and split model/inputs across devices) — override after init.
    cfg.device = "cpu"
    torch.manual_seed(0)  # inference is deterministic; belt-and-braces

    maps, labels = wm.data.load_raw(cfg.data_root)
    test_idx = wm.data.load_splits(cfg)["test"]
    model, ckpt = wm.model.load_checkpoint_model(cfg, acfg.checkpoint_path)
    print(f"Checkpoint: {acfg.checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val macro-F1 {ckpt.get('val_macro_f1', float('nan')):.4f}), "
          f"input {cfg.input_size}px, device cpu")

    ds = wm.data.MixedWaferDataset(maps[test_idx], labels[test_idx],
                                   cfg.input_size, augment=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    _, logits = wm.evaluate.collect_logits(model, loader, "cpu",
                                           desc="test-split inference")

    T, tau = load_thresholds(acfg.thresholds_path)
    probs = wm.calibrate.scale_probs(logits, T)          # float64
    pred = wm.metrics.predict_multihot(probs, tau)

    n = len(test_idx)
    return pd.DataFrame({
        "map_id": np.repeat(test_idx.astype(np.int64), len(LABELS)),
        "label": np.tile(np.array(LABELS, dtype=object), n),
        "logit": logits.ravel().astype(np.float32),
        "prob": probs.ravel(),
        "predicted": pred.ravel().astype(bool),
    })


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_fingerprint(acfg: AttachConfig) -> dict:
    """What the prediction cache's validity depends on."""
    return {"checkpoint_sha256": _sha256(acfg.checkpoint_path),
            "thresholds_sha256": _sha256(acfg.thresholds_path)}


def ensure_prediction_cache(acfg: AttachConfig, cache_path: Path,
                            rebuild: bool = False) -> pd.DataFrame:
    """Return the prediction cache, building it if missing or told to.

    A sidecar .meta.json fingerprints the checkpoint + thresholds the cache
    was built from; reuse with a stale fingerprint is a hard error rather
    than a silent join of old predictions against a retrained model.
    """
    meta_path = cache_path.with_suffix(".meta.json")
    fp = cache_fingerprint(acfg)
    if not rebuild and cache_path.exists():
        if meta_path.exists() and json.loads(meta_path.read_text()) == fp:
            print(f"Prediction cache reused: {cache_path}")
            return read_parquet(cache_path)
        raise SystemExit(
            f"{cache_path} was built from a different checkpoint/thresholds "
            "(or predates fingerprinting) — re-run with --rebuild-cache")
    cache = build_prediction_cache(acfg)
    write_parquet(cache, cache_path)
    meta_path.write_text(json.dumps(fp, indent=2))
    print(f"Prediction cache built: {cache_path} ({len(cache):,} rows)")
    return cache


def load_classifier_outputs(con: duckdb.DuckDBPyConnection,
                            assignment: pd.DataFrame,
                            cache: pd.DataFrame) -> int:
    """Join assignment × cache and (re)load classifier_outputs. Returns rows."""
    missing = set(assignment["map_id"]) - set(cache["map_id"])
    if missing:
        raise ValueError(f"{len(missing)} assigned map_ids absent from the "
                         "prediction cache — rebuild it (--rebuild-cache)")
    rows = assignment.merge(cache[["map_id", "label", "prob", "predicted"]],
                            on="map_id")  # noqa: F841 — DuckDB replacement scan
    con.execute("DELETE FROM classifier_outputs")
    con.execute("""
        INSERT INTO classifier_outputs BY NAME
        SELECT wafer_id, label, prob, predicted FROM rows
    """)
    return con.execute("SELECT count(*) FROM classifier_outputs").fetchone()[0]
