# wafer-rootcause

From *what defect pattern is on the wafer* to *which chamber put it there,
and when*: a simulated MES relational layer (lots, wafers, routes, tools,
chambers, timestamps) joined to **real classifier outputs** from
[wafer-mixed](https://github.com/ALEX8642/wafer-mixed), with commonality
analysis **in SQL** recovering planted faults against known ground truth.
Because the faults are planted, attribution quality is *scored* —
precision, recall, window overlap — not asserted. It doubles as the
rehearsal rig for a private MES extract that never touches GitHub.

**Status: complete (Phases 0–4).** Full session log with tables in
[STATUS.md](STATUS.md); schema in [docs/SCHEMA.md](docs/SCHEMA.md); the
attribution method in [docs/ANALYSIS.md](docs/ANALYSIS.md); the
confounder/sensitivity study in [docs/SENSITIVITY.md](docs/SENSITIVITY.md).

## Why simulated MES + real predictions

Simulation gives **ground-truth fault labels**, so attribution
precision/recall are computable — impossible with public data alone. Real
model outputs keep the noise **honest**: each simulated wafer is dressed
with a real MixedWM38 *test-split* map (never seen in training) whose true
label set matches, then classified by wafer-mixed's checkpoint at its
calibrated per-label thresholds. So `classifier_outputs` carries the
model's real test-split escapes and false alarms, and attribution has to
survive them.

**The analytics live in SQL** (CTEs, window functions, two-proportion
z-tests, Benjamini–Hochberg — kept portable enough to read as
Postgres-ready). Python is glue: simulator, orchestration, figures. That is
the portfolio point of this repo.

## Headline results

Attribution is scored against the planted faults (`ground_truth_faults`,
read by the scorer *only* — see the firewall below).

**Phase 2 — the favourable single draw (seed 42, 1,000 wafers, 5 faults):**

| metric | value |
|---|---|
| attribution precision@1 | **1.000** (4 rank-1 flags, all true faults) |
| attribution recall@1 | **0.800** (4 of 5 faults) |
| grid false discoveries | 0 of 184 tests at BH FDR 0.05 |
| mean window IoU (4 recovered) | **0.866** |
| mean abs. detection latency | 3.25 h |

**Phase 3 — the honest picture (5 seeds, one confounder lever at a time):**

| scenario | recall@1 | precision@1 | what the lever does |
|---|---|---|---|
| baseline | **0.48 ± 0.30** | **1.00** | seed 42's 0.80 was the top of the distribution, not the centre |
| correlated routing | 0.32 | 0.90 | DEPOSITION chamber choice follows ETCH's |
| overlapping faults | 0.30 | 0.92 | second same-signature fault on another step |
| intermittent fault | 0.36 | 0.90 | strongest fault duty-cycled 8 h on / 8 h off |

Two findings worth stating plainly:

- **Precision is reliable, recall is not.** BH holds ~0.90–1.00 even under
  confounders (and the correlated-routing decoy never out-ranks its true
  source — the per-step contrast holds), but recall is low and
  high-variance, set by fault strength × duration × baseline dilution. A
  single seed is not a recall measurement.
- **Classifier quality doesn't move attribution (a null).** Running
  attribution with oracle labels vs raw @0.5 vs calibrated @τ gives
  *identical* recall/precision/IoU despite 0 vs 9 vs 16 label escapes —
  attribution aggregates ~250 wafers per (chamber, label) cell, so a
  handful of errors can't shift a chamber-level proportion. Attribution
  here is statistics-bound, not classifier-bound. The detection floor for a
  40 h fault on a moderate-baseline label is p_acquire ≈ 0.16.

## Figure gallery

Ranked chamber suspects — the full label × chamber grid, BH survivors in
blue, planted faults tagged (scorer annotation):

![suspect ranking](assets/attr_suspect_ranking.png)

Time-resolved excursions — predicted rate by chamber over 6 h buckets,
detected windows outlined (analysis-side only, no ground truth drawn):

![timeline heatmap](assets/attr_timeline_heatmap.png)

Confounders and the classifier-noise ablation (Phase 3):

![confounders](assets/sens_scenarios.png)

![ablation null](assets/sens_ablation.png)

![detection curve](assets/sens_detection_curve.png)

## Quickstart

```bash
pip install -r requirements.txt
python scripts/build_db.py            # configs/sim_baseline.yaml → outputs/wafer_rootcause.duckdb
python scripts/attach_and_predict.py  # attach real test-split maps + run the classifier (CPU, ~3 min once, cached)
python scripts/attribute.py           # SQL commonality + BH + window localisation → score → assets/attr_*.png
python scripts/sensitivity.py         # confounders + classifier-noise ablation + detection curve → assets/sens_*.png
pytest                                # 44 tests
```

`attach_and_predict.py` needs a sibling checkout of
[wafer-mixed](https://github.com/ALEX8642/wafer-mixed) (trained checkpoint,
`thresholds.json`, MixedWM38 data + persisted split) — path in
`configs/attach_baseline.yaml` (`wafer_mixed_root`, default `../wafer-mixed`).
The bridge imports wafer-mixed's *own* modules (model, encoding, temperature
scaling, threshold rule) rather than re-implementing them, so
`classifier_outputs` is definitionally what wafer-mixed produces.

All-CPU, no GPU anywhere in this project. Builds are deterministic (same
configs + seeds → identical database); inference is cached to parquet and
never re-runs on a rebuild, re-seed, or Phase 3 re-configuration.

## Schema

Simulated MES: lot genealogy → route → equipment (tools, chambers) → unit
process history (the fact table) → inspection → classifier outputs, plus two
firewalled ground-truth tables. Data dictionary and design rules in
[docs/SCHEMA.md](docs/SCHEMA.md).

![ERD](assets/erd.png)

**Ground-truth firewall.** `ground_truth_faults` and
`ground_truth_wafer_labels` are written by the simulator and read by the
scorer *only*. Analysis queries (`sql/attr_*.sql`, `sql/eda_*.sql`) never
join them — that is what makes the reported precision/recall honest, and it
is enforced by a test that greps the analysis SQL.

## Rehearsal rig — swapping in a real MES extract

The schema and queries are designed so a private MES extract can replace the
simulator with a **loader swap and zero analysis-SQL changes**. What feeds
each table:

| schema table | real MES source | on the private extract |
|---|---|---|
| `lots`, `wafers` | lot genealogy / WIP tracking (lot open, wafer scribe) | loader maps genealogy → these columns |
| `process_steps` | route / recipe definition (the process flow) | one-time route load per product |
| `tools`, `chambers` | equipment master (tool + chamber IDs, step assignment) | equipment master export |
| `wafer_process_history` | equipment event log / run history (wafer × step × chamber × start/end) | **the load-bearing join** — MES run history |
| `inspections` | defect inspection / review events (with the wafer map) | inspection extract; `map_id` → the real map store |
| `classifier_outputs` | run the deployed classifier over the real maps | same inference bridge, real maps instead of MixedWM38 |
| `ground_truth_faults` | **does not exist in production** | attribution is unsupervised there — the scorer is a simulation-only artifact for validating the method before trusting it on real data |

What changes: only the loader (`src/wafer_rootcause/simulate.py` +
`scripts/build_db.py` → a real-MES loader). What stays identical: every
`sql/*.sql` analysis query, the attribution scoring math, the figures. What
never enters GitHub: the extract, the loader for it, and any derived table —
they live in a private repo only.

## Layout

- `sql/schema.sql` — schema DDL; `sql/attr_*.sql` — the attribution analytics
  (suspect z-tests + BH, window localisation, bucket rates); `sql/eda_*.sql` —
  EDA queries; `sql/score_faults.sql` — the scorer (the only analysis-era file
  allowed to read ground truth)
- `src/wafer_rootcause/` — simulator, config, map attachment, inference bridge,
  attribution runner, sensitivity orchestration, DB loading (Python is glue)
- `configs/` — declarative sim (`sim_baseline` + three Phase 3 adversarial
  configs) and attachment/inference (`attach_baseline`)
- `scripts/` — `build_db`, `attach_and_predict`, `eda`, `attribute`,
  `sensitivity`, `make_erd`
- `docs/` — `SCHEMA` (data dictionary + firewall), `EDA`, `ANALYSIS`
  (attribution method + the honest F3 miss), `SENSITIVITY` (confounders +
  ablation + where the method breaks)
- `tests/` — referential integrity, determinism, combo validity, fault-effect
  sanity, assignment consistency, prediction round-trip, a live spot-check
  against wafer-mixed's own pipeline, statistics vs scipy/numpy references, the
  firewall grep, and the Phase 3 confounder/ablation checks

## Simulated data only

Everything here is simulated or derived from the public MixedWM38 dataset. No
proprietary manufacturing data is present, and none will be committed; the
schema doubles as a rehearsal rig for a private MES extract that stays off
GitHub entirely.

Sibling repos: [wafer-defect-classifier](https://github.com/ALEX8642/wafer-defect-classifier) ·
[wafer-ssl](https://github.com/ALEX8642/wafer-ssl) ·
[wafer-mixed](https://github.com/ALEX8642/wafer-mixed)
