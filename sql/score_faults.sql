-- score_faults.sql — SCORER: planted faults vs the analysis verdicts.
--
-- *** ground_truth_faults is allowed HERE and only here (plus the
-- simulator). Analysis queries (attr_*.sql, eda_*.sql) never touch it. ***
--
-- Expects two relations registered by the caller (scripts/attribute.py /
-- wafer_rootcause.attribution.score):
--   suspects — output of attr_suspects.sql
--   windows  — output of attr_windows.sql
--
-- One row per planted fault: where its chamber landed in the suspect
-- ranking for its signature label, whether it cleared BH significance,
-- and how well the detected excursion window localises the true one
-- (IoU + detection latency). LEFT JOIN on windows: a fault whose cell
-- produced no excursion window still gets a row (win_* NULL).

SELECT
    f.fault_id,
    f.chamber_id,
    f.signature_label            AS label,
    f.start_ts                   AS true_start,
    f.end_ts                     AS true_end,
    f.p_acquire,
    s.suspect_rank,
    s.excess,
    s.z,
    s.q_value,
    s.significant,
    w.win_start,
    w.win_end,
    -- interval IoU in hours; epoch() is DuckDB (Postgres: extract(epoch ...))
    CASE WHEN w.win_start IS NULL THEN NULL ELSE
        greatest(0, epoch(least(f.end_ts, w.win_end))
                  - epoch(greatest(f.start_ts, w.win_start)))
        / ( (epoch(f.end_ts) - epoch(f.start_ts))
          + (epoch(w.win_end) - epoch(w.win_start))
          - greatest(0, epoch(least(f.end_ts, w.win_end))
                      - epoch(greatest(f.start_ts, w.win_start))) )
    END                          AS window_iou,
    CASE WHEN w.win_start IS NULL THEN NULL ELSE
        (epoch(w.win_start) - epoch(f.start_ts)) / 3600.0
    END                          AS latency_hours
FROM ground_truth_faults f
JOIN suspects s
  ON s.chamber_id = f.chamber_id AND s.label = f.signature_label
LEFT JOIN windows w
  ON w.chamber_id = f.chamber_id AND w.label = f.signature_label
ORDER BY f.fault_id;
