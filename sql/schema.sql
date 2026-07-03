-- schema.sql — simulated-MES relational schema (DuckDB, portable ANSI where practical)
--
-- Layers:
--   entities      : lots, wafers
--   route         : process_steps, tools, chambers
--   facts         : wafer_process_history (wafer x step x chamber x time),
--                   inspections, classifier_outputs
--   ground truth  : ground_truth_faults, ground_truth_wafer_labels
--
-- FIREWALL: ground_truth_* tables are read by the simulator and the scorer
-- ONLY. Analysis queries (sql/*.sql outside scoring) must never touch them —
-- that is what makes the attribution scores honest.
--
-- Rehearsal-rig note: table/column names are chosen so a real MES extract
-- (lot genealogy, equipment tracking, defect inspection) can be loaded into
-- the same schema with a loader swap and zero analysis-SQL changes.

CREATE TABLE lots (
    lot_id    VARCHAR PRIMARY KEY,          -- 'LOT0001'
    product   VARCHAR NOT NULL,
    start_ts  TIMESTAMP NOT NULL,           -- release into the line
    n_wafers  INTEGER NOT NULL CHECK (n_wafers > 0)
);

CREATE TABLE wafers (
    wafer_id    VARCHAR PRIMARY KEY,        -- 'LOT0001-W03'
    lot_id      VARCHAR NOT NULL REFERENCES lots(lot_id),
    wafer_index INTEGER NOT NULL            -- 1-based slot within the lot
);

CREATE TABLE process_steps (
    step_id      INTEGER PRIMARY KEY,
    step_order   INTEGER NOT NULL UNIQUE,   -- position along the route, 1-based
    step_name    VARCHAR NOT NULL UNIQUE,   -- 'ETCH'
    process_type VARCHAR NOT NULL           -- 'dry_etch'
);

CREATE TABLE tools (
    tool_id   VARCHAR PRIMARY KEY,          -- 'ETCH-T1'
    step_id   INTEGER NOT NULL REFERENCES process_steps(step_id),
    tool_name VARCHAR NOT NULL
);

CREATE TABLE chambers (
    chamber_id   VARCHAR PRIMARY KEY,       -- 'ETCH-T1-C2'
    tool_id      VARCHAR NOT NULL REFERENCES tools(tool_id),
    chamber_name VARCHAR NOT NULL
);

-- The fact table: one row per wafer per route step.
CREATE TABLE wafer_process_history (
    wafer_id   VARCHAR NOT NULL REFERENCES wafers(wafer_id),
    step_id    INTEGER NOT NULL REFERENCES process_steps(step_id),
    chamber_id VARCHAR NOT NULL REFERENCES chambers(chamber_id),
    start_ts   TIMESTAMP NOT NULL,
    end_ts     TIMESTAMP NOT NULL,
    PRIMARY KEY (wafer_id, step_id),
    CHECK (end_ts > start_ts)
);

-- End-of-line inspection event that produces the wafer map.
CREATE TABLE inspections (
    inspection_id VARCHAR PRIMARY KEY,
    wafer_id      VARCHAR NOT NULL UNIQUE REFERENCES wafers(wafer_id),
    inspect_ts    TIMESTAMP NOT NULL,
    station       VARCHAR NOT NULL,
    map_id        INTEGER                   -- wafer-mixed test-split map index; NULL until Phase 1
);

-- Classifier predictions: 8 rows per inspected wafer (loaded in Phase 1).
CREATE TABLE classifier_outputs (
    wafer_id  VARCHAR NOT NULL REFERENCES wafers(wafer_id),
    label     VARCHAR NOT NULL,             -- one of the 8 signature labels
    prob      DOUBLE  NOT NULL CHECK (prob >= 0 AND prob <= 1),  -- calibrated (per-label T)
    predicted BOOLEAN NOT NULL,             -- prob >= per-label tau (thresholds.json)
    PRIMARY KEY (wafer_id, label)
);

-- ============================ GROUND TRUTH ============================
-- Simulator writes, scorer reads. NEVER joined by analysis queries.

CREATE TABLE ground_truth_faults (
    fault_id        VARCHAR PRIMARY KEY,    -- 'F1'
    chamber_id      VARCHAR NOT NULL REFERENCES chambers(chamber_id),
    signature_label VARCHAR NOT NULL,       -- defect label the fault elevates
    start_ts        TIMESTAMP NOT NULL,
    end_ts          TIMESTAMP NOT NULL,
    p_acquire       DOUBLE NOT NULL CHECK (p_acquire > 0 AND p_acquire <= 1),
    CHECK (end_ts > start_ts)
);

-- True defect label set per wafer (multi-label; absent rows = clean wafer).
CREATE TABLE ground_truth_wafer_labels (
    wafer_id VARCHAR NOT NULL REFERENCES wafers(wafer_id),
    label    VARCHAR NOT NULL,
    source   VARCHAR NOT NULL,              -- 'fault:<fault_id>' or 'baseline'
    PRIMARY KEY (wafer_id, label)
);
