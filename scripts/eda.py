"""eda.py — run the named EDA queries in sql/ and render the figures.

Usage (after scripts/attach_and_predict.py):
    python scripts/eda.py [--db PATH]

The analytics live in SQL: every figure is drawn verbatim from one named
query in sql/eda_*.sql; matplotlib only renders what the query returns.
Everything here reads classifier_outputs — the analysis side never touches
ground_truth_* tables, so no fault windows are overlaid: excursions must be
visible in the predictions alone or Phase 2 has nothing to find.

Figures → assets/, tables quoted by docs/EDA.md printed to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_rootcause.config import REPO_ROOT, AttachConfig  # noqa: E402
from wafer_rootcause.db import connect  # noqa: E402
from wafer_rootcause.labels import LABELS  # noqa: E402

ASSETS = REPO_ROOT / "assets"
SQL_DIR = REPO_ROOT / "sql"

# House style shared with the sibling repos' figures.
INK, MUTED, GRID, BLUE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
# Sequential ramp for magnitude cells: one hue, light → dark (no rainbow).
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#f7f7f3", "#123f75"])


def run_query(con, name: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / f"{name}.sql").read_text()).df()


def _style_axes(ax) -> None:
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _annotate_cells(ax, values: np.ndarray, fmt) -> None:
    """Small in-cell values; ink flips to white on dark cells."""
    vmax = np.nanmax(values) or 1.0
    for (r, c), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        ax.text(c, r, fmt(v), ha="center", va="center", fontsize=6.5,
                color="white" if v > 0.6 * vmax else INK)


def fig_rate_by_chamber(con) -> pd.DataFrame:
    df = run_query(con, "eda_rate_by_chamber")
    order = (df[["step_order", "chamber_id"]].drop_duplicates()
             .sort_values(["step_order", "chamber_id"]))
    grid = (df.pivot(index="chamber_id", columns="label", values="defect_rate")
              .reindex(index=order["chamber_id"], columns=LABELS))

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, aspect="auto")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(grid)), grid.index)
    # hairline separators between route steps
    step_sizes = order.groupby("step_order").size().to_numpy()
    for y in np.cumsum(step_sizes)[:-1]:
        ax.axhline(y - 0.5, color=GRID, linewidth=1)
    _annotate_cells(ax, grid.to_numpy(), lambda v: f"{v:.2f}")
    _style_axes(ax)
    ax.set_title("Predicted defect rate by chamber x label\n"
                 "(share of wafers through the chamber flagged with the label "
                 "at end-of-line)", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.6).ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_rate_by_chamber.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def fig_rate_over_time(con) -> pd.DataFrame:
    df = run_query(con, "eda_rate_over_time")
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    hours = pd.date_range(df["hour_bucket"].min(), df["hour_bucket"].max(),
                          freq="h")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), LABELS):
        s = (df[df["label"] == label]
             .set_index("hour_bucket")["defect_rate"]
             .reindex(hours))
        ax.plot(hours, s, linewidth=0.8, color=BLUE, alpha=0.25)
        ax.plot(hours, s.rolling(6, min_periods=3, center=True).mean(),
                linewidth=2, color=BLUE)
        ax.set_title(label, fontsize=10, color=INK)
        ax.grid(color=GRID, linewidth=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        _style_axes(ax)
    fig.suptitle("Line-wide predicted defect rate over time — hourly buckets "
                 "(faint) with 6 h centred rolling mean", fontsize=12, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_rate_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def fig_lot_yield(con) -> pd.DataFrame:
    df = run_query(con, "eda_lot_yield")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axisbelow(True)
    bins = np.arange(0, df["clean_yield"].max() + 0.08, 0.04)
    ax.hist(df["clean_yield"], bins=bins, color=BLUE,
            edgecolor="white", linewidth=1.5)
    med = df["clean_yield"].median()
    ax.axvline(med, color=INK, linewidth=1)
    ax.text(med + 0.005, ax.get_ylim()[1] * 0.95, f"median {med:.2f}",
            fontsize=9, color=INK, va="top")
    ax.set_xlabel("Lot clean yield (share of wafers with no predicted label)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("Lots", fontsize=9, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    _style_axes(ax)
    ax.set_title("Lot-level clean-yield distribution (classifier verdict)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_lot_yield.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def fig_label_cooccurrence(con) -> pd.DataFrame:
    df = run_query(con, "eda_label_cooccurrence")
    n = len(LABELS)
    counts = np.full((n, n), np.nan)
    idx = {name: i for i, name in enumerate(LABELS)}
    for row in df.itertuples():
        i, j = idx[row.label_a], idx[row.label_b]
        lo, hi = min(i, j), max(i, j)
        counts[hi, lo] = row.n_wafers  # lower triangle only

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(counts, cmap=BLUES, vmin=0)
    ax.set_xticks(range(n), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(n), LABELS)
    _annotate_cells(ax, counts, lambda v: f"{v:.0f}")
    _style_axes(ax)
    ax.set_title("Predicted label co-occurrence (wafer counts)\n"
                 "MixedWM38-forbidden pairs appearing here are stacked "
                 "false alarms", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7).ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "eda_label_cooccurrence.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="wafer-rootcause SQL EDA")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "attach_baseline.yaml"),
                        help="attach config supplying db_path")
    parser.add_argument("--db", default=None, help="override cfg.db_path")
    args = parser.parse_args()
    db_path = args.db or REPO_ROOT / AttachConfig.from_yaml(args.config).db_path
    con = connect(db_path)
    try:
        n_outputs = con.execute("SELECT count(*) FROM classifier_outputs").fetchone()[0]
        if n_outputs == 0:
            raise SystemExit("classifier_outputs is empty — run "
                             "scripts/attach_and_predict.py first")
        prevalence = run_query(con, "eda_label_prevalence")
        print("Predicted label prevalence:\n", prevalence.to_string(index=False))

        by_chamber = fig_rate_by_chamber(con)
        # top chamber-vs-step excursions, quoted in docs/EDA.md
        sums = by_chamber.groupby(["step_name", "label"])[["n_defect", "n_wafers"]].sum()
        step_rate = (sums["n_defect"] / sums["n_wafers"]).rename("step_rate")
        exc = by_chamber.join(step_rate, on=["step_name", "label"])
        exc["excess"] = exc["defect_rate"] - exc["step_rate"]
        print("\nTop chamber-vs-step excursions (defect_rate − step mean):\n",
              exc.nlargest(8, "excess")[["chamber_id", "label", "n_wafers",
                                         "defect_rate", "step_rate", "excess"]]
              .to_string(index=False))

        fig_rate_over_time(con)
        lot = fig_lot_yield(con)
        print(f"\nLot clean yield: median {lot['clean_yield'].median():.3f}, "
              f"min {lot['clean_yield'].min():.3f} ({lot.iloc[0]['lot_id']}), "
              f"max {lot['clean_yield'].max():.3f}")
        pairs = fig_label_cooccurrence(con)
        print("\nTop predicted co-occurrences:\n",
              pairs.head(6).to_string(index=False))
        print(f"\nFigures written to {ASSETS}/eda_*.png")
    finally:
        con.close()


if __name__ == "__main__":
    main()
