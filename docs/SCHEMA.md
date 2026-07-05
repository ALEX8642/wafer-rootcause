# Schema — simulated MES

DDL: [`sql/schema.sql`](../sql/schema.sql) (DuckDB; portable ANSI where
practical, designed to read as Postgres-ready). Built by
`scripts/build_db.py` from `configs/sim_baseline.yaml`.

![ERD](../assets/erd.png)

## Design rules

- **Ground-truth firewall.** `ground_truth_faults` and
  `ground_truth_wafer_labels` are written by the simulator and read by the
  Phase 2 scorer *only*. Analysis queries never join them — attribution
  quality is scored against them afterwards, which is what makes the
  reported precision/recall honest.
- **Rehearsal rig.** Tables and columns mirror what a real MES extract
  provides (lot genealogy, equipment/chamber tracking, unit process
  history, inspection events), so the private work-data extract can later
  be loaded by swapping the loader — analysis SQL unchanged. The
  MES-source mapping table lives in the README's *Rehearsal rig* section.
- **Label-set validity.** The simulator only emits label sets that exist
  in MixedWM38 (at most one of Center/Donut, at most one of
  Edge-Loc/Edge-Ring, Loc/Scratch free, Near-full/Random single-only), so
  every simulated wafer has matching real maps in the wafer-mixed test
  split for Phase 1.

## Data dictionary

### lots
| column | type | description |
|---|---|---|
| lot_id | VARCHAR PK | `LOT0001` … |
| product | VARCHAR | product code (single product in baseline sim) |
| start_ts | TIMESTAMP | release of the lot into the line |
| n_wafers | INTEGER | wafers in the lot |

### wafers
| column | type | description |
|---|---|---|
| wafer_id | VARCHAR PK | `<lot_id>-W<nn>` |
| lot_id | VARCHAR FK→lots | |
| wafer_index | INTEGER | 1-based slot within the lot |

### process_steps
| column | type | description |
|---|---|---|
| step_id | INTEGER PK | |
| step_order | INTEGER UNIQUE | position along the route (1-based) |
| step_name | VARCHAR UNIQUE | e.g. `ETCH` |
| process_type | VARCHAR | e.g. `dry_etch` |

### tools
| column | type | description |
|---|---|---|
| tool_id | VARCHAR PK | `<step_name>-T<n>` |
| step_id | INTEGER FK→process_steps | step this tool serves |
| tool_name | VARCHAR | display name |

### chambers
| column | type | description |
|---|---|---|
| chamber_id | VARCHAR PK | `<tool_id>-C<n>` |
| tool_id | VARCHAR FK→tools | |
| chamber_name | VARCHAR | display name |

### wafer_process_history — the fact table
| column | type | description |
|---|---|---|
| wafer_id | VARCHAR PK,FK→wafers | |
| step_id | INTEGER PK,FK→process_steps | |
| chamber_id | VARCHAR FK→chambers | chamber that processed the wafer |
| start_ts | TIMESTAMP | process start (fault windows match on this) |
| end_ts | TIMESTAMP | process end; `end_ts > start_ts`, monotone along the route |

One row per wafer per route step. Chamber capacity is not modelled
(documented simplification): wafers flow sequentially with exponential
queue delays and ±10 % jittered process times.

### inspections
| column | type | description |
|---|---|---|
| inspection_id | VARCHAR PK | `INSP-<wafer_id>` |
| wafer_id | VARCHAR FK→wafers, UNIQUE | one end-of-line inspection per wafer |
| inspect_ts | TIMESTAMP | after the last route step |
| station | VARCHAR | inspection station id |
| map_id | INTEGER, NULL | MixedWM38.npz row index of the attached map (always a test-split row); written by `scripts/attach_and_predict.py` |

### classifier_outputs *(loaded by `scripts/attach_and_predict.py`)*
| column | type | description |
|---|---|---|
| wafer_id | VARCHAR PK,FK→wafers | |
| label | VARCHAR PK | one of the 8 signature labels |
| prob | DOUBLE | temperature-calibrated probability (per-label T) |
| predicted | BOOLEAN | `prob > tau_label` from wafer-mixed `thresholds.json` (same strict rule as wafer-mixed's `predict_multihot`) |

8 rows per wafer — the model's view of the wafer, which is what all
attribution queries consume (never the ground truth).

### ground_truth_faults 🔒
| column | type | description |
|---|---|---|
| fault_id | VARCHAR PK | `F1` … |
| chamber_id | VARCHAR FK→chambers | faulty chamber |
| signature_label | VARCHAR | defect label the fault elevates (mixable labels only) |
| start_ts / end_ts | TIMESTAMP | fault window |
| p_acquire | DOUBLE | P(exposed wafer acquires the label) |

### ground_truth_wafer_labels 🔒
| column | type | description |
|---|---|---|
| wafer_id | VARCHAR PK,FK→wafers | absent rows ⇒ clean wafer |
| label | VARCHAR PK | true defect label |
| source | VARCHAR | `fault:<fault_id>` or `baseline` |

🔒 = ground-truth firewall: simulator writes, scorer reads, analysis never.

## Signature labels

Order matches wafer-mixed's verified label order:
`Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random`.
Validity rules live in `src/wafer_rootcause/labels.py`.
