"""attribute.py — Phase 2: run attribution, score it, render the figures.

Usage (after scripts/attach_and_predict.py):
    python scripts/attribute.py [--db PATH] [--heatmap-label Scratch]

Order of business mirrors the firewall:
  1. ANALYSIS (classifier_outputs only): ranked suspects per label
     (sql/attr_suspects.sql) + excursion windows (sql/attr_windows.sql).
  2. SCORER (ground_truth_faults allowed): sql/score_faults.sql joins the
     verdicts against the planted faults — precision/recall, window IoU,
     latency. Everything printed under the SCORER banner, and the fault
     tags on the figures, come from this side.

Figures → assets/:
  attr_suspect_ranking.png  — the grid's top cells by z; BH survivors in
                              blue, planted faults tagged (scorer-side tag)
  attr_rate_by_chamber.png  — chamber-vs-step bars for a hit and a miss
  attr_timeline_heatmap.png — chamber x time rate for one label, detected
                              excursion windows outlined (analysis-side)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_rootcause.attribution import (ALPHA, K_DEFAULT, bucket_rates,  # noqa: E402
                                         score, suspects, windows)
from wafer_rootcause.config import REPO_ROOT, AttachConfig  # noqa: E402
from wafer_rootcause.db import connect  # noqa: E402
from wafer_rootcause.labels import LABELS  # noqa: E402

ASSETS = REPO_ROOT / "assets"

# House style shared with the sibling repos' figures (see scripts/eda.py).
INK, MUTED, GRID, BLUE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
FAINT = "#d8d6ce"  # de-emphasis fill for not-the-story marks
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#f7f7f3", "#123f75"])


def _style_axes(ax) -> None:
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def fault_tags(per_fault: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(chamber_id, label) -> 'F1' for the planted faults (scorer side)."""
    return {(r.chamber_id, r.label): r.fault_id for r in per_fault.itertuples()}


def fig_suspect_ranking(sus: pd.DataFrame, per_fault: pd.DataFrame) -> None:
    """Top grid cells by z — the ranked list BH is asked to prune."""
    tags = fault_tags(per_fault)
    # top 10 by z, then a break, then any planted fault that fell below the
    # fold at its true rank — the missed fault is the honest exhibit
    top = sus.nlargest(10, "z")
    is_fault = [(c, l) in tags for c, l in zip(sus["chamber_id"], sus["label"])]
    below = (sus[is_fault].drop(index=top.index, errors="ignore")
             .sort_values("z", ascending=False))
    rows = pd.concat([top, below]).reset_index(drop=True)
    gap = 1.4  # y-space marking the omitted cells between the two blocks

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 2.1))
    y = -np.arange(len(rows), dtype=float)
    y[len(top):] -= gap
    colors = [BLUE if s else FAINT for s in rows["significant"]]
    ax.barh(y, rows["z"], height=0.62, color=colors, edgecolor="white",
            linewidth=1.5)
    ax.set_yticks(y, [f"{c}  ·  {l}" for c, l in
                      zip(rows["chamber_id"], rows["label"])], fontsize=8)
    if len(below):
        n_omitted = int((sus["z"] > below["z"].max()).sum()) - len(top)
        ax.text(0.04, y[len(top) - 1] - 1 - gap / 2,
                f"⋯  {n_omitted} cells omitted  ⋯", fontsize=7.5, color=MUTED,
                va="center")
    for yi, row in zip(y, rows.itertuples()):
        tag = tags.get((row.chamber_id, row.label))
        note = f"z={row.z:.1f}, q={row.q_value:.2g}"
        if tag:
            rank = int((sus["z"] > row.z).sum() + 1)
            note += (f"   planted {tag}"
                     + ("" if row.significant
                        else f" — not flagged (rank {rank} of {len(sus)})"))
        ax.text(row.z + 0.08, yi, note, va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, top["z"].max() * 1.45)
    ax.set_xlabel("two-proportion z (chamber vs rest-of-step)",
                  fontsize=9, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6, axis="x")
    ax.set_axisbelow(True)
    _style_axes(ax)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE),
               Rectangle((0, 0), 1, 1, color=FAINT)]
    ax.legend(handles, [f"BH-significant (q ≤ {ALPHA})", "not significant"],
              fontsize=8, frameon=False, loc="center right")
    ax.set_title("Chamber suspects, ranked by evidence — full label x chamber "
                 f"grid ({len(sus)} tests)\nBenjamini–Hochberg keeps "
                 f"{int(sus['significant'].sum())} cells; planted faults "
                 "tagged (scorer annotation)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "attr_suspect_ranking.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _rate_panel(ax, sus: pd.DataFrame, chamber: str, label: str,
                verdict: str) -> None:
    step = sus.loc[sus["chamber_id"] == chamber, "step_name"].iloc[0]
    cells = (sus[(sus["label"] == label) & (sus["step_name"] == step)]
             .sort_values("chamber_id").reset_index(drop=True))
    fault_row = cells[cells["chamber_id"] == chamber].iloc[0]

    x = np.arange(len(cells))
    colors = [BLUE if c == chamber else FAINT for c in cells["chamber_id"]]
    ax.bar(x, cells["rate_cham"], width=0.62, color=colors,
           edgecolor="white", linewidth=1.5)
    ax.axhline(fault_row["rate_rest"], color=INK, linewidth=1)
    ax.text(len(cells) - 0.4, fault_row["rate_rest"] + 0.006,
            f"rest-of-step {fault_row['rate_rest']:.2f}",
            ha="right", fontsize=7.5, color=INK)
    i = int(cells.index[cells["chamber_id"] == chamber][0])
    ax.text(i, fault_row["rate_cham"] + 0.006, f"{fault_row['rate_cham']:.2f}",
            ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [c.removeprefix(f"{step}-") for c in cells["chamber_id"]],
                  fontsize=8)
    ax.set_title(f"{label} @ {step} — {verdict}\n"
                 f"suspect rank {int(fault_row['suspect_rank'])}, "
                 f"q = {fault_row['q_value']:.2g}", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    _style_axes(ax)


def fig_rate_by_chamber(sus: pd.DataFrame, per_fault: pd.DataFrame) -> None:
    """Chamber-vs-step bars for the clearest hit and the weakest case."""
    recovered = per_fault["significant"] & (per_fault["suspect_rank"] <= K_DEFAULT)
    hit = per_fault[recovered].nlargest(1, "z").iloc[0]
    missed = per_fault[~recovered]
    if len(missed):
        miss, verdict = missed.nsmallest(1, "z").iloc[0], "missed"
    else:  # all faults recovered on this draw — show the weakest instead
        miss, verdict = per_fault[recovered].nsmallest(1, "z").iloc[0], \
            "weakest hit"

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    _rate_panel(axes[0], sus, hit["chamber_id"], hit["label"],
                f"recovered ({hit['fault_id']})")
    _rate_panel(axes[1], sus, miss["chamber_id"], miss["label"],
                f"{verdict} ({miss['fault_id']})")
    axes[0].set_ylabel("predicted defect rate (whole horizon)",
                       fontsize=9, color=MUTED)
    fig.suptitle("Commonality contrast: fault chamber (blue) vs step siblings "
                 "— whole-horizon marginals", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "attr_rate_by_chamber.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_timeline_heatmap(rates: pd.DataFrame, wins: pd.DataFrame,
                         label: str) -> None:
    """Chamber x time rate heatmap for one label, detected windows outlined.

    Both layers are analysis-side: the heatmap is sql/attr_bucket_rates.sql,
    the outlines are sql/attr_windows.sql. No ground truth is drawn — the
    reader can check the outlined windows against docs/ANALYSIS.md.
    """
    df = rates[rates["label"] == label]
    order = (df[["step_order", "chamber_id"]].drop_duplicates()
             .sort_values(["step_order", "chamber_id"]))
    grid = (df.pivot(index="chamber_id", columns="bucket_ts", values="rate")
              .reindex(order["chamber_id"]))
    buckets = grid.columns
    bucket_h = (buckets[1] - buckets[0]) / pd.Timedelta(hours=1)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = BLUES.copy()
    cmap.set_bad("#ffffff")  # empty buckets (no wafers) stay surface-white
    im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), cmap=cmap,
                   vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(grid)), grid.index)
    tick_every = max(1, len(buckets) // 12)
    ax.set_xticks(range(0, len(buckets), tick_every),
                  [pd.Timestamp(b).strftime("%b %d %Hh")
                   for b in buckets[::tick_every]], rotation=30, ha="right")
    step_sizes = order.groupby("step_order").size().to_numpy()
    for yb in np.cumsum(step_sizes)[:-1]:
        ax.axhline(yb - 0.5, color=GRID, linewidth=1)

    col_of = {b: i for i, b in enumerate(buckets)}
    row_of = {c: i for i, c in enumerate(grid.index)}
    for w in wins[wins["label"] == label].itertuples():
        x0 = col_of[w.win_start] - 0.5
        n_cols = round((w.win_end - w.win_start) / pd.Timedelta(hours=bucket_h))
        ax.add_patch(Rectangle((x0, row_of[w.chamber_id] - 0.5), n_cols, 1,
                               fill=False, edgecolor=INK, linewidth=1.6))
    _style_axes(ax)
    ax.set_title(f"{label}: predicted rate by chamber over {bucket_h:.0f} h "
                 "buckets of chamber-processing time\nblack outline = detected "
                 "excursion window (sql/attr_windows.sql); white = no wafers "
                 "in bucket", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.6).ax.tick_params(colors=MUTED,
                                                       labelsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "attr_timeline_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="wafer-rootcause Phase 2")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "attach_baseline.yaml"),
                        help="attach config supplying db_path")
    parser.add_argument("--db", default=None, help="override cfg.db_path")
    parser.add_argument("--heatmap-label", default="Scratch",
                        help="label rendered in the timeline heatmap")
    args = parser.parse_args()
    if args.heatmap_label not in LABELS:
        raise SystemExit(f"--heatmap-label {args.heatmap_label!r} is not one "
                         f"of {LABELS}")
    db_path = args.db or REPO_ROOT / AttachConfig.from_yaml(args.config).db_path

    con = connect(db_path)
    try:
        if con.execute("SELECT count(*) FROM classifier_outputs").fetchone()[0] == 0:
            raise SystemExit("classifier_outputs is empty — run "
                             "scripts/attach_and_predict.py first")

        # --- analysis side ---
        sus = suspects(con)
        wins = windows(con)
        rates = bucket_rates(con)
        sig = sus[sus["significant"]]
        print(f"Grid: {len(sus)} (label x chamber) tests, BH keeps "
              f"{len(sig)} at FDR {ALPHA}\n")
        print("BH-significant suspects:")
        print(sig[["label", "chamber_id", "n_cham", "rate_cham", "rate_rest",
                   "excess", "z", "q_value", "suspect_rank"]]
              .to_string(index=False))
        print(f"\nExcursion windows detected: {len(wins)} "
              "(top 8 by excess defects):")
        print(wins.head(8).to_string(index=False))

        # --- scorer side ---
        per_fault, summary = score(con, sus, wins)
        print("\n" + "=" * 72)
        print("SCORER (ground_truth_faults joined — never used above)")
        print("=" * 72)
        print(per_fault[["fault_id", "chamber_id", "label", "p_acquire",
                         "suspect_rank", "z", "q_value", "significant",
                         "win_start", "win_end", "window_iou",
                         "latency_hours"]].to_string(index=False))
        print("\nHeadline metrics:")
        for k, v in summary.items():
            print(f"  {k:20s} {v:.3f}" if isinstance(v, float)
                  else f"  {k:20s} {v}")

        fig_suspect_ranking(sus, per_fault)
        fig_rate_by_chamber(sus, per_fault)
        fig_timeline_heatmap(rates, wins, args.heatmap_label)
        print(f"\nFigures written to {ASSETS}/attr_*.png")
    finally:
        con.close()


if __name__ == "__main__":
    main()
