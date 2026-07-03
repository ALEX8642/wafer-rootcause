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
