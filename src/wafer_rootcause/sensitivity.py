"""sensitivity.py — Phase 3 orchestration: run the whole pipeline end to end
on adversarial configs and classifier-noise ablations, and score each run.

One `run_scenario` = simulate → in-memory DB → attach maps → fill
classifier_outputs in the chosen ablation mode → attr_suspects/attr_windows →
score against ground truth. Nothing here is new analysis: it drives the same
SQL Phase 2 shipped. The map assignment (assign_seed) is held fixed across a
sweep so the only moving lever is the one under test.

Firewall: the analysis SQL still reads classifier_outputs only. This module
reads ground_truth_wafer_labels for the oracle mode and ground_truth_faults
through the scorer — both already-blessed harness/scorer uses.
"""
from __future__ import annotations

import dataclasses

import duckdb
import pandas as pd

from wafer_rootcause.attach import (assign_maps, load_test_inventory,
                                    record_assignment, wafer_combos)
from wafer_rootcause.attribution import K_DEFAULT, score, suspects, windows
from wafer_rootcause.config import AttachConfig, SimConfig
from wafer_rootcause.db import memory_db
from wafer_rootcause.predict import load_classifier_outputs
from wafer_rootcause.simulate import simulate


def run_scenario(sim_cfg: SimConfig, acfg: AttachConfig,
                 inventory: pd.DataFrame, cache: pd.DataFrame | None,
                 mode: str = "calibrated") -> tuple[pd.DataFrame, dict]:
    """Simulate `sim_cfg`, attach maps, score attribution under `mode`.

    Returns (per_fault, summary) exactly as attribution.score does. `cache`
    may be None only for mode='oracle' (which never touches the classifier).
    """
    tables = simulate(sim_cfg)
    con = memory_db(tables)
    try:
        assignment = assign_maps(wafer_combos(con), inventory, acfg.assign_seed)
        record_assignment(con, assignment)
        if mode != "oracle" and cache is None:
            raise ValueError(f"mode={mode!r} needs the prediction cache")
        load_classifier_outputs(con, assignment, cache, mode=mode)
        sus, wins = suspects(con), windows(con)
        return score(con, sus, wins)
    finally:
        con.close()


def run_over_seeds(sim_cfg: SimConfig, acfg: AttachConfig,
                   inventory: pd.DataFrame, cache: pd.DataFrame | None,
                   seeds, mode: str = "calibrated"
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`run_scenario` at each seed; returns (per_fault_seeded, summary_seeded).

    Reporting a scenario over a seed set (not one draw) is the point: a
    single seed conflates the config lever with RNG-phase noise — the extra
    draws a correlated dispatch consumes shift every later label draw, so
    an unrelated fault's z wanders seed to seed. Averaging over seeds
    isolates the lever. The second frame carries each seed's precision@1 and
    flagged-count (the "did an innocent chamber get blamed" metric, which
    needs the full suspect list, not just the planted-fault rows).
    """
    fault_frames, summ_rows = [], []
    for seed in seeds:
        per_fault, summary = run_scenario(
            dataclasses.replace(sim_cfg, seed=seed), acfg, inventory, cache, mode)
        per_fault = per_fault.copy()
        per_fault["seed"] = seed
        per_fault["recovered"] = _recovered(per_fault).to_numpy()
        fault_frames.append(per_fault)
        summ_rows.append({"seed": seed, **summary})
    return pd.concat(fault_frames, ignore_index=True), pd.DataFrame(summ_rows)


def seed_summary(per_fault_seeded: pd.DataFrame,
                 summary_seeded: pd.DataFrame) -> dict:
    """Seed-averaged headline metrics from a run_over_seeds pair.

    recall is averaged per seed then over seeds (each seed weights equally
    regardless of how many faults it planted); precision@1 and flagged-count
    are averaged over seeds from the per-seed summaries.
    """
    per_seed = per_fault_seeded.groupby("seed")
    recall1 = per_seed.apply(
        lambda g: (g["significant"] & (g["suspect_rank"] == 1)).mean(),
        include_groups=False)
    recallk = per_seed.apply(lambda g: g["recovered"].mean(),
                             include_groups=False)
    return {
        "n_seeds": per_fault_seeded["seed"].nunique(),
        "recall@1": float(recall1.mean()),
        "recall@1_std": float(recall1.std(ddof=0)),
        f"recall@{K_DEFAULT}": float(recallk.mean()),
        "precision@1": float(summary_seeded["precision@1"].mean()),
        "mean_n_flagged@1": float(summary_seeded["n_flagged@1"].mean()),
        "mean_iou": float(summary_seeded["mean_iou"].mean()),
        "per_fault_recovery": (
            per_fault_seeded.groupby("fault_id")["recovered"].mean().to_dict()),
    }


def _recovered(per_fault: pd.DataFrame, k: int = K_DEFAULT) -> pd.Series:
    """Boolean per fault: BH-significant and ranked in the top k."""
    return per_fault["significant"] & (per_fault["suspect_rank"] <= k)


def scenario_row(name: str, mode: str,
                 per_fault: pd.DataFrame, summary: dict) -> dict:
    """One flat record summarising a scenario run for the results table."""
    rec = _recovered(per_fault)
    return {
        "scenario": name, "mode": mode,
        "n_faults": summary["n_faults"],
        "recall@1": summary["recall@1"],
        f"recall@{K_DEFAULT}": summary[f"recall@{K_DEFAULT}"],
        "precision@1": summary["precision@1"],
        f"precision@{K_DEFAULT}": summary[f"precision@{K_DEFAULT}"],
        "n_recovered": int(rec.sum()),
        "n_flagged@1": summary["n_flagged@1"],
        "mean_iou": summary["mean_iou"],
        "mean_abs_latency_h": summary["mean_abs_latency_h"],
    }


def with_fault_intensity(base: SimConfig, fault_id: str, p_acquire: float,
                         seed: int) -> SimConfig:
    """Copy `base` with one fault's p_acquire and the sim seed overridden."""
    faults = [dataclasses.replace(f, p_acquire=p_acquire)
              if f.fault_id == fault_id else f for f in base.faults]
    if not any(f.fault_id == fault_id for f in base.faults):
        raise ValueError(f"{fault_id} not in the base config")
    return dataclasses.replace(base, seed=seed, faults=faults)


def sweep_intensity(base: SimConfig, acfg: AttachConfig, inventory: pd.DataFrame,
                    cache: pd.DataFrame, fault_id: str,
                    intensities, seeds, modes=("calibrated", "oracle"),
                    progress=None) -> pd.DataFrame:
    """Detection curve: vary one fault's p_acquire across seeds and modes.

    Returns one row per (mode, p_acquire, seed) for the swept fault: its
    rank, z, q, significance and whether it was recovered. recall at each
    (mode, intensity) is the mean of `recovered` over seeds.
    """
    rows = []
    for mode in modes:
        for p in intensities:
            for seed in seeds:
                cfg = with_fault_intensity(base, fault_id, p, seed)
                per_fault, _ = run_scenario(cfg, acfg, inventory, cache, mode)
                f = per_fault.set_index("fault_id").loc[fault_id]
                rows.append({
                    "mode": mode, "p_acquire": p, "seed": seed,
                    "suspect_rank": int(f["suspect_rank"]),
                    "z": float(f["z"]), "q_value": float(f["q_value"]),
                    "significant": bool(f["significant"]),
                    "recovered": bool(f["significant"]
                                      and f["suspect_rank"] <= K_DEFAULT),
                    "window_iou": (None if pd.isna(f["window_iou"])
                                   else float(f["window_iou"])),
                })
                if progress is not None:
                    progress(mode, p, seed)
    return pd.DataFrame(rows)


def load_context(acfg: AttachConfig,
                 cache_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(inventory, cache) from the wafer-mixed checkout + prediction cache."""
    from wafer_rootcause.config import REPO_ROOT
    from wafer_rootcause.db import read_parquet
    inventory = load_test_inventory(acfg)
    cache = read_parquet(cache_path or REPO_ROOT / acfg.cache_path)
    return inventory, cache
