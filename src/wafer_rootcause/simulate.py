"""simulate.py — schedule lots through the route, plant faults, emit tables.

Produces the full relational payload as pandas DataFrames keyed by table
name (see sql/schema.sql). Deterministic per (config, seed): one
`np.random.default_rng(seed)` drives every draw in a fixed iteration order.

Label mechanics
---------------
1. Faults first (config order): a wafer is *exposed* to a fault if its
   history row at the fault's chamber starts inside the fault window;
   exposed wafers acquire the signature label with p_acquire.
2. Baseline contamination everywhere: per exclusivity group (Center/Donut,
   Edge-Loc/Edge-Ring — member picked uniformly), then Loc, Scratch, and
   the single-only Near-full / Random (which labels.can_add lets through
   only on wafers that are still clean).
Every addition passes labels.can_add, so each wafer's label set is one of
MixedWM38's 38 valid combos and Phase 1 can always find a matching real map.

Scheduling is intentionally simple: per-wafer sequential flow with
exponential queue delays and jittered process times; chamber capacity is
not modelled (documented simplification — attribution needs who/where/when,
not a queueing-accurate fab).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from wafer_rootcause.config import SimConfig, route_chambers
from wafer_rootcause.labels import can_add


def simulate(cfg: SimConfig) -> dict[str, pd.DataFrame]:
    """Run the simulation; return {table_name: DataFrame} for all tables."""
    rng = np.random.default_rng(cfg.seed)
    sim_start = (cfg.sim_start if isinstance(cfg.sim_start, datetime)
                 else datetime.fromisoformat(cfg.sim_start))

    # ---- route: steps, tools, chambers -------------------------------
    steps = [{"step_id": i + 1, "step_order": i + 1,
              "step_name": spec.name, "process_type": spec.process_type}
             for i, spec in enumerate(cfg.route)]
    tools, chambers = [], []
    step_chambers: dict[int, list[str]] = {s["step_id"]: [] for s in steps}
    for i, _, tool_id, chamber_id in route_chambers(cfg.route):
        if not tools or tools[-1]["tool_id"] != tool_id:
            tools.append({"tool_id": tool_id, "step_id": i + 1,
                          "tool_name": tool_id})
        chambers.append({"chamber_id": chamber_id, "tool_id": tool_id,
                         "chamber_name": chamber_id})
        step_chambers[i + 1].append(chamber_id)

    # ---- lots and wafers ----------------------------------------------
    lots, wafers = [], []
    lot_start_by_id: dict[str, datetime] = {}
    interarrivals = rng.exponential(cfg.lot_interarrival_hours, cfg.n_lots)
    lot_starts = np.cumsum(interarrivals)
    for li in range(cfg.n_lots):
        lot_id = f"LOT{li + 1:04d}"
        lot_start_by_id[lot_id] = sim_start + timedelta(hours=float(lot_starts[li]))
        lots.append({"lot_id": lot_id, "product": cfg.product,
                     "start_ts": lot_start_by_id[lot_id],
                     "n_wafers": cfg.wafers_per_lot})
        for wi in range(1, cfg.wafers_per_lot + 1):
            wafers.append({"wafer_id": f"{lot_id}-W{wi:02d}",
                           "lot_id": lot_id, "wafer_index": wi})

    # ---- wafer process history ----------------------------------------
    history = []
    last_end: dict[str, datetime] = {}  # wafer_id -> end of its final step
    for w in wafers:
        t = lot_start_by_id[w["lot_id"]] + timedelta(
            minutes=(w["wafer_index"] - 1) * cfg.wafer_stagger_minutes)
        for spec, step_row in zip(cfg.route, steps):
            step_id = step_row["step_id"]
            chamber_id = step_chambers[step_id][
                rng.integers(len(step_chambers[step_id]))]
            start = t + timedelta(
                minutes=float(rng.exponential(cfg.queue_mean_minutes)))
            end = start + timedelta(
                minutes=spec.process_minutes * float(rng.uniform(0.9, 1.1)))
            history.append({"wafer_id": w["wafer_id"], "step_id": step_id,
                            "chamber_id": chamber_id,
                            "start_ts": start, "end_ts": end})
            t = end
        last_end[w["wafer_id"]] = t

    # ---- faults → absolute windows ------------------------------------
    faults = []
    for f in cfg.faults:
        f_start = sim_start + timedelta(hours=f.start_hour)
        faults.append({"fault_id": f.fault_id, "chamber_id": f.chamber,
                       "signature_label": f.label,
                       "start_ts": f_start,
                       "end_ts": f_start + timedelta(hours=f.duration_hours),
                       "p_acquire": f.p_acquire})

    # exposure lookup: wafer_id -> [fault rows], from its own history
    hist_by_chamber: dict[str, list[dict]] = {}
    for row in history:
        hist_by_chamber.setdefault(row["chamber_id"], []).append(row)
    exposures: dict[str, list[dict]] = {}
    for f in faults:
        for row in hist_by_chamber.get(f["chamber_id"], []):
            if f["start_ts"] <= row["start_ts"] <= f["end_ts"]:
                exposures.setdefault(row["wafer_id"], []).append(f)

    # ---- label assignment (fixed draw order per wafer) -----------------
    b = cfg.baseline
    group_rates = [
        (b.center_donut, ("Center", "Donut")),
        (b.edge, ("Edge-Loc", "Edge-Ring")),
        (b.loc, ("Loc",)),
        (b.scratch, ("Scratch",)),
        # single-only labels: can_add admits them on clean wafers only
        (b.near_full, ("Near-full",)),
        (b.random_pattern, ("Random",)),
    ]
    gt_labels = []
    for w in wafers:
        wid = w["wafer_id"]
        labels: set[str] = set()
        for f in exposures.get(wid, []):
            if rng.random() < f["p_acquire"] and can_add(labels, f["signature_label"]):
                labels.add(f["signature_label"])
                gt_labels.append({"wafer_id": wid, "label": f["signature_label"],
                                  "source": f"fault:{f['fault_id']}"})
        for rate, members in group_rates:
            draw, pick = rng.random(), rng.integers(len(members))
            if draw < rate:
                label = members[pick]
                if can_add(labels, label):
                    labels.add(label)
                    gt_labels.append({"wafer_id": wid, "label": label,
                                      "source": "baseline"})

    # ---- inspections ----------------------------------------------------
    inspections = []
    for w in wafers:
        inspections.append({
            "inspection_id": f"INSP-{w['wafer_id']}",
            "wafer_id": w["wafer_id"],
            "inspect_ts": last_end[w["wafer_id"]]
                          + timedelta(minutes=cfg.inspect_delay_minutes),
            "station": "EOL-INSPECT-1",
            "map_id": None,  # assigned in Phase 1
        })

    return {
        "lots": pd.DataFrame(lots),
        "wafers": pd.DataFrame(wafers),
        "process_steps": pd.DataFrame(steps),
        "tools": pd.DataFrame(tools),
        "chambers": pd.DataFrame(chambers),
        "wafer_process_history": pd.DataFrame(history),
        "inspections": pd.DataFrame(inspections),
        "ground_truth_faults": pd.DataFrame(
            faults, columns=["fault_id", "chamber_id", "signature_label",
                             "start_ts", "end_ts", "p_acquire"]),
        "ground_truth_wafer_labels": pd.DataFrame(
            gt_labels, columns=["wafer_id", "label", "source"]),
    }
