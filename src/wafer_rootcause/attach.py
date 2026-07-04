"""attach.py — assign real wafer-mixed test-split maps to simulated wafers.

Each simulated wafer receives a MixedWM38 map whose TRUE label set equals
the wafer's simulated label set, sampled without replacement from
wafer-mixed's persisted *test* split — maps the classifier never saw in
training, so the Phase 1 predictions carry the honest test-split error
rates. `map_id` is the row index into MixedWM38.npz (restricted to
test-split rows), i.e. it is meaningful in wafer-mixed's own terms:
arr_0[map_id] is the wafer image, arr_1[map_id] its multi-hot truth.

Firewall note: this module reads ground_truth_wafer_labels. That is
allowed — attachment is part of the simulation harness (it realises the
simulated truth as pixels before the classifier ever runs), not of the
analysis. The schema firewall applies to analysis queries in sql/.

Determinism: assignment is a pure function of (wafer label sets,
test-split inventory, seed) — combos and wafers are iterated in sorted
order and all draws come from one seeded Generator.
"""
from __future__ import annotations

from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from wafer_rootcause.config import AttachConfig
from wafer_rootcause.labels import LABELS


def combo_key(labels: Iterable[str]) -> str:
    """Canonical combo string: labels joined in LABELS order, '' → 'normal'.

    Matches wafer-mixed's combo_name convention so inventory printouts read
    the same in both repos.
    """
    present = set(labels)
    active = [name for name in LABELS if name in present]
    return "+".join(active) if active else "normal"


def load_test_inventory(acfg: AttachConfig) -> pd.DataFrame:
    """Test-split map inventory: DataFrame(map_id, combo).

    map_id is the global MixedWM38.npz row index. Only arr_1 (labels) is
    materialised — the 400 MB image array stays on disk until inference.
    """
    labels = np.load(acfg.npz_path)["arr_1"]
    test_idx = np.load(acfg.splits_path)["test"]
    combos = [
        combo_key(name for name, on in zip(LABELS, row) if on)
        for row in labels[test_idx]
    ]
    return pd.DataFrame({"map_id": test_idx.astype(np.int64), "combo": combos})


def wafer_combos(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Simulated label set per wafer as combo strings, indexed by wafer_id.

    Includes clean wafers (no ground-truth rows) as 'normal'.
    """
    df = con.execute("""
        SELECT w.wafer_id, g.label
        FROM wafers w
        LEFT JOIN ground_truth_wafer_labels g ON g.wafer_id = w.wafer_id
        ORDER BY w.wafer_id
    """).df()
    return (df.groupby("wafer_id")["label"]
              .apply(lambda s: combo_key(s.dropna()))
              .rename("combo"))


def assign_maps(combos: pd.Series, inventory: pd.DataFrame,
                seed: int) -> pd.DataFrame:
    """Sample one matching test-split map per wafer, without replacement.

    Returns DataFrame(wafer_id, map_id). Raises if any combo's demand
    exceeds the test-split supply — the fix is retuning the sim config
    (configs/sim_baseline.yaml), not sampling with replacement, which
    would let one map's classifier error count twice.
    """
    demand = combos.value_counts()
    supply = inventory["combo"].value_counts()
    short = {c: (int(n), int(supply.get(c, 0)))
             for c, n in demand.items() if n > supply.get(c, 0)}
    if short:
        lines = [f"  {c}: need {n}, have {s}" for c, (n, s) in sorted(short.items())]
        raise ValueError(
            "Test-split inventory oversubscribed — retune configs/sim_baseline.yaml:\n"
            + "\n".join(lines))

    rng = np.random.default_rng(seed)
    pools = inventory.groupby("combo")["map_id"].apply(lambda s: np.sort(s.to_numpy()))
    rows = []
    for combo in sorted(demand.index):
        wafer_ids = sorted(combos.index[combos == combo])
        chosen = rng.choice(pools[combo], size=len(wafer_ids), replace=False)
        rows.extend(zip(wafer_ids, chosen))
    return (pd.DataFrame(rows, columns=["wafer_id", "map_id"])
              .astype({"map_id": np.int64})
              .sort_values("wafer_id", ignore_index=True))


def record_assignment(con: duckdb.DuckDBPyConnection,
                      assignment: pd.DataFrame) -> None:
    """Write the assignment into inspections.map_id."""
    con.execute("""
        UPDATE inspections
        SET map_id = a.map_id
        FROM assignment a
        WHERE inspections.wafer_id = a.wafer_id
    """)
