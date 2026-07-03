"""make_erd.py — render the schema ERD to assets/erd.png.

Pure matplotlib (no graphviz dependency). Regenerate after any schema
change: python scripts/make_erd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_rootcause.config import REPO_ROOT  # noqa: E402

INK = "#1f2933"
MUTED = "#616e7c"
HEADER = "#334e68"
GT_HEADER = "#8d2b0b"
EDGE = "#9aa5b1"
ROW_H, TITLE_H, PAD = 3.2, 4.2, 1.0

# name -> (x, y_top, width, [(key_marker, "col type")])
TABLES = {
    "lots": (4, 88, 27, [
        ("PK", "lot_id"), ("", "product"), ("", "start_ts"), ("", "n_wafers")]),
    "process_steps": (46, 88, 27, [
        ("PK", "step_id"), ("", "step_order"), ("", "step_name"),
        ("", "process_type")]),
    "tools": (78, 88, 24, [
        ("PK", "tool_id"), ("FK", "step_id"), ("", "tool_name")]),
    "chambers": (110, 88, 24, [
        ("PK", "chamber_id"), ("FK", "tool_id"), ("", "chamber_name")]),
    "wafers": (4, 62, 27, [
        ("PK", "wafer_id"), ("FK", "lot_id"), ("", "wafer_index")]),
    "wafer_process_history": (60, 64, 31, [
        ("PK,FK", "wafer_id"), ("PK,FK", "step_id"), ("FK", "chamber_id"),
        ("", "start_ts"), ("", "end_ts")]),
    "inspections": (4, 34, 27, [
        ("PK", "inspection_id"), ("FK", "wafer_id  (unique)"),
        ("", "inspect_ts"), ("", "station"), ("", "map_id  (Phase 1)")]),
    "classifier_outputs": (36, 34, 27, [
        ("PK,FK", "wafer_id"), ("PK", "label"), ("", "prob  (calibrated)"),
        ("", "predicted  (@tau)")]),
    "ground_truth_faults": (104, 31, 30, [
        ("PK", "fault_id"), ("FK", "chamber_id"), ("", "signature_label"),
        ("", "start_ts"), ("", "end_ts"), ("", "p_acquire")]),
    "ground_truth_wafer_labels": (70, 31, 30, [
        ("PK,FK", "wafer_id"), ("PK", "label"), ("", "source")]),
}

# (child, parent, child_side, parent_side) — sides pick the anchor edge
EDGES = [
    ("wafers", "lots", "top", "bottom"),
    ("tools", "process_steps", "left", "right"),
    ("chambers", "tools", "left", "right"),
    ("wafer_process_history", "wafers", "left", "right"),
    ("wafer_process_history", "process_steps", "top", "bottom"),
    ("wafer_process_history", "chambers", "top-right", "bottom"),
    ("inspections", "wafers", "top", "bottom"),
    ("classifier_outputs", "wafers", "top", "right"),
    ("ground_truth_faults", "chambers", "top", "bottom"),
    ("ground_truth_wafer_labels", "wafers", "top", "right"),
]


def box_geom(name):
    x, y_top, w, rows = TABLES[name]
    h = TITLE_H + ROW_H * len(rows) + PAD
    return x, y_top - h, w, h  # x, y_bottom, w, h


def anchor(name, side):
    x, y, w, h = box_geom(name)
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
            "top-right": (x + 0.85 * w, y + h)}[side]


def draw_table(ax, name):
    x, y_top, w, rows = TABLES[name]
    _, y, _, h = box_geom(name)
    gt = name.startswith("ground_truth")
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4",
        facecolor="white", edgecolor=EDGE, linewidth=1.2))
    header = GT_HEADER if gt else HEADER
    ax.add_patch(FancyBboxPatch(
        (x, y + h - TITLE_H), w, TITLE_H, boxstyle="round,pad=0.4",
        facecolor=header, edgecolor=header))
    ax.text(x + w / 2, y + h - TITLE_H / 2, name, ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold", family="monospace")
    for i, (marker, col) in enumerate(rows):
        ry = y + h - TITLE_H - ROW_H * (i + 0.5) - PAD / 2
        ax.text(x + 1.5, ry, col, va="center", fontsize=9.5,
                family="monospace", color=INK)
        if marker:
            ax.text(x + w - 1.5, ry, marker, va="center", ha="right",
                    fontsize=8, family="monospace", color=MUTED)


def main():
    fig, ax = plt.subplots(figsize=(14.5, 9.2))
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#f5f7fa")

    # ground-truth firewall zone
    ax.add_patch(FancyBboxPatch(
        (66, 1.5), 72.5, 31.5, boxstyle="round,pad=0.6", facecolor="#fff3f0",
        edgecolor=GT_HEADER, linewidth=1.4, linestyle="--"))
    ax.text(102, 3.0, "GROUND TRUTH — simulator writes, scorer reads. "
            "Never joined by analysis queries.",
            ha="center", fontsize=8.5, color=GT_HEADER, style="italic")

    for child, parent, cs, ps in EDGES:
        p0, p1 = anchor(child, cs), anchor(parent, ps)
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
            color=MUTED, shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=0.04"))

    for name in TABLES:
        draw_table(ax, name)

    ax.text(4, 96.5, "wafer-rootcause — simulated MES schema",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(4, 93, "Arrows point child → parent (FK → PK). "
            "classifier_outputs is loaded in Phase 1 from the wafer-mixed "
            "checkpoint; map_id links inspections to test-split maps.",
            fontsize=9.5, color=MUTED)

    out = REPO_ROOT / "assets" / "erd.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
