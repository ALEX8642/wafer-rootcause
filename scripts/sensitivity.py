"""sensitivity.py — Phase 3: confounders + classifier-noise ablation.

Runs the Phase 2 attribution pipeline unchanged over adversarial sim configs
and classifier-noise ablations, scores every run against ground truth, and
reports where the method breaks. Everything analytical is still the Phase 2
SQL; this script only drives it (see src/wafer_rootcause/sensitivity.py).

What it does (all CPU, ~2 min):
  1. Confounders — baseline + three one-lever configs (correlated routing,
     two same-label overlapping faults, an intermittent fault), each over a
     seed set so the lever is separated from RNG-phase noise.
  2. Ablation — baseline three ways: oracle (true labels), raw (@0.5 on the
     cached logits), calibrated (@tau). How much does classifier quality buy
     in attribution terms?
  3. Intensity sweep — F4's p_acquire from near-baseline up, calibrated vs
     oracle, over seeds: the detection-vs-intensity curve.

Outputs → outputs/sensitivity_*.parquet + assets/sens_*.png. Narrative and
the reproduced tables live in docs/SENSITIVITY.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_rootcause.attribution import K_DEFAULT  # noqa: E402
from wafer_rootcause.config import REPO_ROOT, AttachConfig, SimConfig  # noqa: E402
from wafer_rootcause.db import memory_db, write_parquet  # noqa: E402
from wafer_rootcause.sensitivity import (load_context, run_over_seeds,  # noqa: E402
                                         seed_summary, sweep_intensity)

ASSETS = REPO_ROOT / "assets"
OUT = REPO_ROOT / "outputs"

# House style shared with scripts/eda.py and scripts/attribute.py.
INK, MUTED, GRID, BLUE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
FAINT, AMBER = "#d8d6ce", "#c6892b"      # de-emphasis fill; second categorical hue
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#f7f7f3", "#123f75"])

SCENARIOS = {
    "baseline": "sim_baseline",
    "correlated": "sim_correlated",
    "overlap": "sim_overlap",
    "intermittent": "sim_intermittent",
}
SWEEP_INTENSITIES = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25]


def _style(ax) -> None:
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


# ------------------------------- compute -------------------------------

def run_confounders(acfg, inv, cache, seeds):
    """Baseline + 3 confounder configs over the seed set (calibrated)."""
    per_fault, summaries = [], []
    for name, yaml in SCENARIOS.items():
        cfg = SimConfig.from_yaml(REPO_ROOT / "configs" / f"{yaml}.yaml")
        pf, ss = run_over_seeds(cfg, acfg, inv, cache, seeds, "calibrated")
        pf.insert(0, "scenario", name)
        per_fault.append(pf)
        s = seed_summary(pf, ss)
        s.pop("per_fault_recovery")
        summaries.append({"scenario": name, **s})
        print(f"  {name:13s} recall@1={s['recall@1']:.2f}±{s['recall@1_std']:.2f} "
              f"recall@{K_DEFAULT}={s[f'recall@{K_DEFAULT}']:.2f} "
              f"precision@1={s['precision@1']:.2f} "
              f"mean_n_flagged@1={s['mean_n_flagged@1']:.1f} iou={s['mean_iou']:.2f}")
    return pd.concat(per_fault, ignore_index=True), pd.DataFrame(summaries)


def run_ablation(acfg, inv, cache, seeds):
    """Baseline three ways, plus each mode's escape/false-alarm count."""
    cfg = SimConfig.from_yaml(REPO_ROOT / "configs" / "sim_baseline.yaml")
    rows = []
    for mode in ("oracle", "calibrated", "raw"):
        pf, ss = run_over_seeds(cfg, acfg, inv, cache, seeds, mode)
        s = seed_summary(pf, ss)
        esc, fa = _escape_counts(cfg, acfg, inv, cache, mode)
        rows.append({"mode": mode, "recall@1": s["recall@1"],
                     f"recall@{K_DEFAULT}": s[f"recall@{K_DEFAULT}"],
                     "precision@1": s["precision@1"], "mean_iou": s["mean_iou"],
                     "escapes": esc, "false_alarms": fa})
        print(f"  {mode:11s} recall@1={s['recall@1']:.2f} "
              f"precision@1={s['precision@1']:.2f} iou={s['mean_iou']:.3f} "
              f"(escapes={esc}, false_alarms={fa})")
    return pd.DataFrame(rows)


def _escape_counts(cfg, acfg, inv, cache, mode, seed=42):
    """Label-level escapes + false alarms vs ground truth, one seed."""
    import dataclasses

    from wafer_rootcause.attach import (assign_maps, record_assignment,
                                        wafer_combos)
    from wafer_rootcause.predict import load_classifier_outputs
    from wafer_rootcause.simulate import simulate
    con = memory_db(simulate(dataclasses.replace(cfg, seed=seed)))
    try:
        asn = assign_maps(wafer_combos(con), inv, acfg.assign_seed)
        record_assignment(con, asn)
        load_classifier_outputs(con, asn, cache, mode=mode)
        return con.execute("""
            SELECT count(*) FILTER (WHERE g.wafer_id IS NOT NULL AND NOT co.predicted),
                   count(*) FILTER (WHERE g.wafer_id IS NULL AND co.predicted)
            FROM classifier_outputs co
            LEFT JOIN ground_truth_wafer_labels g
                   ON g.wafer_id = co.wafer_id AND g.label = co.label
        """).fetchone()
    finally:
        con.close()


# ------------------------------- figures -------------------------------

def fig_scenarios(summ: pd.DataFrame, per_fault: pd.DataFrame) -> None:
    """Left: recall@1 (±std) and precision@1 per scenario. Right: per-fault
    recovery-rate heatmap (fault x scenario)."""
    order = list(SCENARIOS)
    summ = summ.set_index("scenario").reindex(order)
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})

    x = np.arange(len(order))
    w = 0.38
    axl.bar(x - w / 2, summ["recall@1"], w, yerr=summ["recall@1_std"],
            color=BLUE, edgecolor="white", linewidth=1.2, capsize=3,
            error_kw={"ecolor": MUTED, "elinewidth": 1}, label="recall@1")
    axl.bar(x + w / 2, summ["precision@1"], w, color=AMBER, edgecolor="white",
            linewidth=1.2, label="precision@1")
    for xi, (r, p) in enumerate(zip(summ["recall@1"], summ["precision@1"])):
        axl.text(xi - w / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7.5,
                 color=INK)
        axl.text(xi + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=7.5,
                 color=INK)
    axl.axhline(1.0, color=GRID, linewidth=1, zorder=0)
    axl.set_xticks(x, order, fontsize=8.5)
    axl.set_ylim(0, 1.18)
    axl.set_ylabel("rate (5 seeds)", fontsize=9, color=MUTED)
    axl.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    axl.set_title("Recall collapses, precision slips below 1.0\n"
                  "each confounder vs baseline, seed-averaged",
                  fontsize=10.5, color=INK)
    axl.grid(color=GRID, linewidth=0.6, axis="y")
    axl.set_axisbelow(True)
    _style(axl)

    faults = sorted(per_fault["fault_id"].unique())
    grid = (per_fault.groupby(["fault_id", "scenario"])["recovered"].mean()
            .unstack("scenario").reindex(index=faults, columns=order))
    im = axr.imshow(np.ma.masked_invalid(grid.to_numpy()), cmap=BLUES,
                    vmin=0, vmax=1, aspect="auto")
    axr.set_xticks(range(len(order)), order, fontsize=8.5)
    axr.set_yticks(range(len(faults)), faults, fontsize=8.5)
    for i in range(len(faults)):
        for j in range(len(order)):
            v = grid.to_numpy()[i, j]
            if not np.isnan(v):
                axr.text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=8, color="white" if v > 0.5 else INK)
    axr.set_title("Per-fault recovery rate\n(fraction of 5 seeds recovered "
                  f"@{K_DEFAULT})", fontsize=10.5, color=INK)
    _style(axr)
    fig.colorbar(im, ax=axr, shrink=0.7).ax.tick_params(colors=MUTED,
                                                        labelsize=8)
    fig.suptitle("Confounders — one lever at a time, same 5 seeds",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "sens_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame) -> None:
    """Three metrics, three modes — identical bars despite different noise."""
    modes = ["oracle", "calibrated", "raw"]
    abl = abl.set_index("mode").reindex(modes)
    metrics = [("recall@1", "recall@1"), (f"recall@{K_DEFAULT}", f"recall@{K_DEFAULT}"),
               ("precision@1", "precision@1"), ("mean_iou", "mean IoU")]
    colors = {"oracle": BLUE, "calibrated": AMBER, "raw": FAINT}

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(metrics))
    w = 0.26
    for i, mode in enumerate(modes):
        vals = [abl.loc[mode, m] for m, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[mode],
                      edgecolor="white", linewidth=1.2,
                      label=(f"{mode}  ({int(abl.loc[mode, 'escapes'])} esc, "
                             f"{int(abl.loc[mode, 'false_alarms'])} FA)"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [lab for _, lab in metrics], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value (5 seeds)", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper right", title="classifier mode",
              title_fontsize=8)
    ax.set_title("Classifier-noise ablation: attribution is identical across "
                 "oracle / calibrated / raw\ndespite 0 vs 9 vs 16 label escapes "
                 "— attribution is statistics-bound, not classifier-bound",
                 fontsize=10.5, color=INK)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "sens_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_detection_curve(sweep: pd.DataFrame) -> None:
    """F4 recall vs p_acquire, calibrated vs oracle. The two curves coincide
    (the ablation null) — the floor is set by intensity, not the classifier."""
    grp = (sweep.groupby(["mode", "p_acquire"])["recovered"].mean()
           .reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.4))
    styles = {"calibrated": (AMBER, "-", "o"), "oracle": (BLUE, "--", "s")}
    for mode, (c, ls, mk) in styles.items():
        d = grp[grp["mode"] == mode].sort_values("p_acquire")
        ax.plot(d["p_acquire"], d["recovered"], ls, color=c, marker=mk,
                markersize=6, linewidth=2, label=mode,
                markeredgecolor="white", markeredgewidth=1)
    ax.axvline(0.19, color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.19, 1.02, "Donut baseline ≈0.19", fontsize=7.5, color=MUTED,
            ha="center")
    ax.axvline(0.25, color=GRID, linewidth=1)
    ax.text(0.25, 0.55, "Phase 2 draw\n(p=0.25)", fontsize=7.5, color=MUTED,
            ha="right", va="center")
    n_seeds = sweep["seed"].nunique()
    ax.set_xlabel("F4 p_acquire (fault intensity)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"recall of F4 (fraction of {n_seeds} seeds recovered "
                  f"@{K_DEFAULT})", fontsize=9, color=MUTED)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left",
              title="classifier mode", title_fontsize=8)
    ax.set_title("Detection curve: how weak a fault can commonality find?\n"
                 "F4 (Donut @ OXIDATION-T2-C1, 40 h window) — calibrated and "
                 "oracle coincide", fontsize=10.5, color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "sens_detection_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------- main --------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="wafer-rootcause Phase 3")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "attach_baseline.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--skip-sweep", action="store_true",
                        help="scenarios + ablation only (the slow part is the sweep)")
    args = parser.parse_args()

    acfg = AttachConfig.from_yaml(args.config)
    inv, cache = load_context(acfg)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Seeds: {args.seeds}\n")

    print("Confounders (calibrated, seed-averaged):")
    conf_faults, conf_summ = run_confounders(acfg, inv, cache, args.seeds)
    write_parquet(conf_faults, OUT / "sensitivity_scenarios.parquet")
    write_parquet(conf_summ, OUT / "sensitivity_scenario_summary.parquet")

    print("\nClassifier-noise ablation (baseline, seed-averaged):")
    abl = run_ablation(acfg, inv, cache, args.seeds)
    write_parquet(abl, OUT / "sensitivity_ablation.parquet")

    fig_scenarios(conf_summ, conf_faults)
    fig_ablation(abl)

    if not args.skip_sweep:
        print(f"\nIntensity sweep (F4, {len(SWEEP_INTENSITIES)} points x "
              f"{len(args.seeds)} seeds x 2 modes):")
        base = SimConfig.from_yaml(REPO_ROOT / "configs" / "sim_baseline.yaml")
        done = [0]

        def tick(*_):
            done[0] += 1
            print(f"\r  {done[0]}/"
                  f"{2 * len(SWEEP_INTENSITIES) * len(args.seeds)} runs",
                  end="", flush=True)

        sweep = sweep_intensity(base, acfg, inv, cache, "F4",
                                SWEEP_INTENSITIES, args.seeds, progress=tick)
        print()
        write_parquet(sweep, OUT / "sensitivity_sweep.parquet")
        curve = (sweep.groupby(["mode", "p_acquire"])["recovered"].mean()
                 .unstack("mode"))
        print("F4 recall by intensity:")
        print(curve.to_string())
        fig_detection_curve(sweep)

    print(f"\nFigures → {ASSETS}/sens_*.png")
    print(f"Results → {OUT}/sensitivity_*.parquet")


if __name__ == "__main__":
    main()
