# STATUS — wafer-rootcause

Session handoff log. One phase per session (see workspace
`PLAN-wafer-rootcause.md`).

## Phase 0 — Scaffold + schema + simulator ✅ (2026-07-03)

**Done:**
- Repo scaffold mirroring the sibling repos (src/wafer_rootcause, sql/,
  configs, tests, scripts, docs, assets). MIT license; data, outputs and
  `*.duckdb` gitignored.
- `sql/schema.sql`: 10 tables — lots, wafers, process_steps, tools,
  chambers, wafer_process_history (fact), inspections, classifier_outputs
  (empty until Phase 1), plus firewalled ground_truth_faults and
  ground_truth_wafer_labels. Data dictionary in `docs/SCHEMA.md`, ERD in
  `assets/erd.png` (regenerate with `scripts/make_erd.py`).
- Simulator (`src/wafer_rootcause/simulate.py`), fully config-driven
  (`configs/sim_baseline.yaml`): 40 lots × 25 wafers through a 6-step
  route (17 chambers), uniform-random dispatch, exponential queue delays,
  5 planted faults (p_acquire 0.25–0.70, incl. a deliberately weak one
  for honest Phase 2 misses) + everywhere baseline contamination.
  Deterministic per seed.
- `scripts/build_db.py`: config → simulate → DuckDB, prints row counts +
  ground-truth prevalence + horizon.
- Tests: **12 passing** — referential integrity (all FKs), full route
  coverage, chamber-belongs-to-step, timestamp monotonicity (SQL window
  function), fault windows inside horizon, determinism (same seed →
  identical frames; different seed differs), MixedWM38 combo validity,
  fault-effect sanity (exposed-wafer rate ≈ p_acquire over base, 3σ),
  Near-full/Random isolation, classifier_outputs empty.

**Decisions / deviations from the plan sketch:**
- **The plan's "≤7,603 wafers" cap is not the binding constraint** — the
  per-combo test-split inventory is, especially **Normal (200 maps)** and
  Near-full (30). Phase 1 samples matching maps without replacement, so
  clean-wafer demand must stay under 200. Sized the sim at 1,000 wafers
  with a deliberately defect-dense line (~85 % of wafers carry ≥1 label).
  Verified over seeds {42,1,2,3,4}: no combo oversubscribed, worst-case
  utilization 84 % (Normal). Also the conservative direction for
  attribution — high common baseline makes excursions harder to isolate.
- Simulator enforces MixedWM38 combo validity by construction (≤1 of
  Center/Donut, ≤1 of Edge-Loc/Edge-Ring, Near-full/Random single-only;
  faults restricted to mixable labels), so every simulated label set has
  real maps available.
- Added `ground_truth_wafer_labels` (with per-label `source`) beyond the
  plan's table sketch — Phase 1 map matching and the Phase 3 oracle
  ablation both need per-wafer truth, and it belongs behind the same
  firewall as the fault table.
- Chamber capacity is not modelled (documented in SCHEMA.md) —
  attribution needs who/where/when, not a queueing-accurate fab.

**Verified for Phase 1 (plan asked Phase 0 to check):**
- `wafer-mixed/outputs/best.pt` and `outputs/thresholds.json` exist
  locally; thresholds.json embeds per-label temperatures + tuned τ.
- Test-split inventory counted from `data/splits.npz` + raw npz: 38
  combos, 200 maps for most, 400 for Center+Edge-Loc+Scratch, 173 Random,
  30 Near-full.

**Baseline DB (seed 42):** 1,000 wafers, 6,000 history rows, 144 clean,
mixable-label prevalence 20–33 % (Random 1.5 %, Near-full 0.1 %), horizon
2026-06-01 → 2026-06-06 (~140 h).

**Next (Phase 1, fresh session):** map attachment (sample matching
test-split maps without replacement, persist assignment), run the
wafer-mixed checkpoint on CPU over assigned maps, apply per-label T + τ,
load `classifier_outputs`, cache predictions to parquet; SQL EDA →
docs/EDA.md. Torch pins come from wafer-mixed's requirements.txt.

## Phase 1 — Classifier integration + SQL EDA ✅ (2026-07-04)

**Done:**
- Map attachment (`src/wafer_rootcause/attach.py` +
  `scripts/attach_and_predict.py`): each simulated wafer gets a wafer-mixed
  **test-split** map whose true label set matches its simulated label set,
  sampled without replacement, deterministic per `assign_seed` (own seed,
  independent of the sim seed; `configs/attach_baseline.yaml`). Persisted
  to `outputs/map_assignment.parquet` + `inspections.map_id` (= MixedWM38
  npz row index). Baseline demand fits comfortably: tightest combo is
  Normal at 72 % of its 200 test maps.
- Inference (`src/wafer_rootcause/predict.py`): imports wafer-mixed's OWN
  modules from the sibling checkout (model, data, evaluate, calibrate,
  metrics) — zero re-implementation of the architecture / encoding /
  T + τ rule. One CPU pass over the FULL test split (7,603 maps, 2 m 20 s,
  batch 64) cached to `outputs/predictions_test.parquet` (map_id, label,
  raw logit, calibrated prob, predicted) — rebuilds/re-seeds only re-join
  the cache, and the raw logits mean Phase 3's raw-@0.5 ablation needs no
  new inference either. `classifier_outputs` loaded: 8,000 rows.
- **Honest-noise check (scorer side):** on this 1,000-wafer draw the
  classifier makes 9 label-level escapes + 8 false alarms (worst: Scratch
  4 escapes, Loc 6 FAs). Checkpoint epoch 7, val macro-F1 0.9906.
- SQL EDA: 5 named queries in `sql/eda_*.sql` (prevalence, rate by
  step/tool/chamber, hourly rates over time, lot yield, co-occurrence),
  rendered by `scripts/eda.py` → 4 figures in assets/, narrative in
  `docs/EDA.md`. **Eyeball check passed:** 4 of 5 planted faults are the
  top-4 chamber-vs-step excursions in the raw rates (F5 +0.185, F1 +0.139,
  F4 +0.097, F2 +0.095); F3 (Scratch @ CMP-T1-C2) leads its step (0.324 vs
  0.261–0.294) but is diluted below noise entries in the whole-horizon
  marginal — the expected Phase 2 case for time-resolved localisation.
  A routing-noise coincidence (ETCH-T1-C1 Loc +0.075) sits at rank 5:
  the motivating example for Phase 2's significance testing + BH control.
- Tests: **22 passing** (12 Phase 0 + 10 new) — every wafer assigned
  exactly once, no map reused, assigned map's true labels == simulated
  labels, map_ids ⊆ test split (checked against splits.npz directly),
  assignment determinism, inspections round-trip, 8 rows/wafer, DB ==
  parquet cache round-trip, predicted == (prob > τ) for all rows, and a
  live 12-map spot-check that reruns wafer-mixed's encode→model→
  scale_probs→predict_multihot end-to-end and matches the DB. Phase 1
  tests skip cleanly if the wafer-mixed checkout or cache is absent.

**Pre-commit review (multi-angle) — applied:** forced `device="cpu"` after
MixedConfig init (WAFER_DEVICE env var otherwise outranks it — real risk on
the 5090-rig shell); prediction cache now fingerprinted
(`predictions_test.meta.json`, sha256 of checkpoint + thresholds; stale
reuse is a hard error, verified); DB mutations moved after the slow
inference step (no mixed assignment/predictions state on abort); import
bridge asserts the loaded wafer_mixed lives under the configured checkout
AND that `LABEL_NAMES == LABELS` (order drift would scramble every column
while staying test-green); quote-escaped parquet paths; eda.py errors
cleanly on empty classifier_outputs and reads db_path from the attach
config. Accepted-risk: τ rule (`>`) restated in docs — the live spot-check
recomputes through wafer-mixed's `predict_multihot`, so a rule change fails
the suite after any cache rebuild.

**Decisions / deviations:**
- `map_id` = global MixedWM38.npz row index (not position-within-test-split)
  so it's directly meaningful against wafer-mixed's arrays; schema comment +
  SCHEMA.md updated.
- Prediction cache covers the full test split, not just assigned maps —
  30 s more inference once, in exchange for Phase 3 never touching torch.
- Parquet IO via DuckDB (`COPY`/`read_parquet`) — avoids a pyarrow dep.
- Firewall extended to figures: EDA plots draw predictions only, no
  ground-truth fault-window overlays (excursions must be visible in
  predictions alone or Phase 2 has nothing to find).

**Next (Phase 2, fresh session):** commonality analysis in SQL — per
(label, step) two-proportion test chamber-vs-rest, BH correction across the
label × step × chamber grid, ranked suspects; window localisation via
rolling rates; scorer (precision@1/@3, recall, window IoU, latency) —
`ground_truth_faults` allowed in the scorer only. Figures + docs/ANALYSIS.md.
Watch F3 (needs time resolution) and the ETCH-T1-C1 Loc false suspect.

## Phase 2 — Root-cause attribution ✅ (2026-07-04)

**Scored attribution (baseline: seed 42, 1,000 wafers, 5 planted faults):**

| metric | value |
|---|---|
| attribution precision@1 | **1.000** (4 rank-1 flags, all true faults) |
| attribution recall@1 (= @3) | **0.800** (4 of 5 faults) |
| false discoveries | 0 of 184 grid tests at BH FDR 0.05 |
| mean window IoU (4 recovered) | **0.866** |
| mean abs. detection latency | 3.25 h |

Per fault: F1 rank 1, IoU 1.00 · F2 rank 1, IoU 0.86 · F4 rank 1, IoU 0.81
· F5 rank 1, IoU 0.80 · **F3 missed** (rank 3, q = 0.49 — 40 h window
diluted by ~100 h of clean traffic in the whole-horizon marginal; its
excursion window WAS detected at IoU 0.45 but nothing routes an analyst
there; see docs/ANALYSIS.md). The ETCH-T1-C1 Loc coincidence from Phase 1's
EDA died correctly under BH (q = 0.33).

**Done:**
- `sql/attr_suspects.sql` — the headline: chamber vs rest-of-step
  two-proportion z per (label, chamber) cell (184 tests), one-sided p via
  inlined Abramowitz–Stegun 7.1.26 (stock DuckDB has no erf; stays portable
  arithmetic), Benjamini–Hochberg via window functions, per-label suspect
  ranking. Verified in tests against scipy (|Δp| < 1.5e-7) and a numpy BH
  reimplementation (exact).
- `sql/attr_windows.sql` — localisation: 6 h buckets of chamber-processing
  time (4 h fragmented runs on sparse chambers — IoU for F1 went 0.42 → 1.00
  at 6 h), centred 3-bucket rolling rate, flag at rest + 2 binomial SE,
  gaps-and-islands runs, ≥ 2 buckets and ≥ 5 excess defects, best run per
  cell. `sql/attr_bucket_rates.sql` feeds the timeline heatmap.
- `sql/score_faults.sql` — scorer (the ONLY analysis-era file allowed to
  read ground_truth_faults): per-fault rank/q/IoU/latency;
  `attribution.score()` aggregates precision@k / recall@k. Firewall now
  test-enforced (comment-stripped grep over sql/ in test_attribution.py).
- `scripts/attribute.py` — analysis → scorer banner → 3 figures:
  attr_suspect_ranking (top 10 + below-fold planted fault at true rank),
  attr_rate_by_chamber (hit F5 vs miss F3 panels), attr_timeline_heatmap
  (Scratch, detected windows outlined — analysis-side only).
- `docs/ANALYSIS.md` — method + scored tables + three honesty sections:
  the F3 miss (whole-horizon dilution; scan stats deliberately not built),
  windows-localise-don't-detect (28 windows, only 4 on significant cells;
  the Edge-Ring echo mechanism), F4 recovered despite being planted weak
  (detection limit unmeasured → Phase 3(d) sweep).
- Tests: **31 passing** (22 prior + 9 new) — grid coverage/partition
  invariants, z vs direct computation, p vs scipy, BH vs numpy, rank
  ordering, window well-formedness, deterministic strong-fault recovery
  (F1/F5 rank 1 + IoU > 0.5), synthetic hand-checked scorer arithmetic,
  firewall grep.

**Decisions / deviations:**
- Suspect rank runs across all 23 chambers per label (not per step) — a
  fault could be at any step, so the per-label walk-down list is the
  deliverable. Cross-step "echo" excursions are why rank + BH must lead
  and windows follow (documented mechanism in ANALYSIS.md).
- Window detector gates: min 5 excess defects per run — without it,
  near-zero-baseline labels (Near-full, Random) emitted one-wafer "rate
  excursions". 62 → 28 windows, no effect on fault cells.
- p-values in SQL rather than Python: keeps the whole test in one
  readable/portable query; the A&S approximation error bound is quoted in
  the file header and asserted in tests.
- attribution params live as SQL `params` CTEs (bucket_hours 6, sigma 2,
  min_roll_n 10, min_run_buckets 2, min_excess 5; alpha 0.05 duplicated as
  `attribution.ALPHA` for the scorer/figures — keep in sync).

**Next (Phase 3, fresh session):** adversarial configs one lever at a time,
same seeds — (a) correlated routing (the echo mechanism above becomes a
real trap), (b) two same-label overlapping faults, (c) intermittent fault,
(d) intensity sweep p_acquire → baseline (F4 came back rank 1 at 0.25, so
the sweep must go lower to find the floor); classifier-noise ablation
(oracle labels vs raw @0.5 vs calibrated @τ) reusing the cached raw logits
— no new inference needed. → docs/SENSITIVITY.md + figures.
