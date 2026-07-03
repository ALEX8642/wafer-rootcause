# wafer-rootcause

From *what defect pattern is on the wafer* to *which tool/chamber put it
there*: a simulated MES relational layer (lots, wafers, routes, tools,
chambers, timestamps) joined to **real classifier outputs** from
[wafer-mixed](https://github.com/ALEX8642/wafer-mixed), with commonality
analysis in SQL recovering planted faults against known ground truth —
attribution quality is *scored*, not asserted.

**Status: Phase 0 (scaffold + schema + simulator) complete.** See
[STATUS.md](STATUS.md) for the phase log and
[docs/SCHEMA.md](docs/SCHEMA.md) for the schema.

![ERD](assets/erd.png)

## Quickstart (Phase 0 scope)

```bash
pip install -r requirements.txt
python scripts/build_db.py            # configs/sim_baseline.yaml → outputs/wafer_rootcause.duckdb
pytest                                # integrity, monotonicity, determinism, fault-effect sanity
```

All-CPU, no GPU anywhere in this project. The build is deterministic:
same config + seed → identical database.

## Layout

- `sql/schema.sql` — the schema DDL (the analytics in this repo live in SQL)
- `src/wafer_rootcause/` — simulator, config, DB loading (Python is glue)
- `configs/sim_baseline.yaml` — declarative sim: route, fault list, rates, seed
- `docs/SCHEMA.md` — data dictionary + design rules (ground-truth firewall)
- `tests/` — referential integrity, timestamp monotonicity, determinism,
  combo validity, fault-effect sanity

## Simulated data only

Everything in this repo is simulated or derived from the public MixedWM38
dataset. No proprietary manufacturing data is present, and none will be
committed; the schema doubles as a rehearsal rig for a private MES extract
that stays off GitHub entirely.

Sibling repos: [wafer-defect-classifier](https://github.com/ALEX8642/wafer-defect-classifier) ·
[wafer-ssl](https://github.com/ALEX8642/wafer-ssl) ·
[wafer-mixed](https://github.com/ALEX8642/wafer-mixed)
