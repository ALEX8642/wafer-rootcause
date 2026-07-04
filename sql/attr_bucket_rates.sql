-- attr_bucket_rates.sql — per-chamber defect rate over time buckets.
--
-- Wafers are bucketed by WHEN THEY WERE PROCESSED IN THE CHAMBER
-- (wafer_process_history.start_ts), not by inspection time: a chamber
-- fault is an event in chamber time, and the end-of-line label is carried
-- back to the pass that could have caused it. The bucket spine is
-- generated explicitly so quiet buckets appear as n = 0 rows rather than
-- silently vanishing (rolling windows in attr_windows.sql and the
-- timeline heatmap both need the gaps).
--
-- Analysis side: reads classifier_outputs only, never ground_truth_*.
-- Portability: time_bucket/generate_series are DuckDB spellings; Postgres
-- has generate_series and date_bin with the same semantics.

WITH params AS (
    SELECT 6 AS bucket_hours   -- keep in step with attr_windows.sql
),

pass AS (
    SELECT h.chamber_id, co.label, h.start_ts, co.predicted
    FROM wafer_process_history h
    JOIN classifier_outputs co ON co.wafer_id = h.wafer_id
),

horizon AS (
    SELECT
        time_bucket(INTERVAL 1 HOUR * p.bucket_hours, min(start_ts)) AS h_min,
        time_bucket(INTERVAL 1 HOUR * p.bucket_hours, max(start_ts)) AS h_max,
        p.bucket_hours
    FROM pass, params p
    GROUP BY p.bucket_hours
),

spine AS (
    SELECT ch.chamber_id, lab.label, b.bucket_ts
    FROM chambers ch
    CROSS JOIN (SELECT DISTINCT label FROM classifier_outputs) lab
    CROSS JOIN (
        SELECT unnest(generate_series(h_min, h_max,
                                      INTERVAL 1 HOUR * bucket_hours)) AS bucket_ts
        FROM horizon
    ) b
),

bucket_counts AS (
    SELECT p.chamber_id, p.label,
           time_bucket(INTERVAL 1 HOUR * pr.bucket_hours, p.start_ts) AS bucket_ts,
           count(*)                          AS n,
           count(*) FILTER (WHERE predicted) AS k
    FROM pass p, params pr
    GROUP BY 1, 2, 3
)

SELECT
    l.step_order, l.step_name, s.chamber_id, s.label, s.bucket_ts,
    coalesce(b.n, 0) AS n,
    coalesce(b.k, 0) AS k,
    CASE WHEN b.n > 0 THEN b.k * 1.0 / b.n END AS rate
FROM spine s
LEFT JOIN bucket_counts b
       ON b.chamber_id = s.chamber_id AND b.label = s.label
      AND b.bucket_ts  = s.bucket_ts
JOIN (SELECT ch.chamber_id, ps.step_order, ps.step_name
      FROM chambers ch
      JOIN tools t          ON t.tool_id  = ch.tool_id
      JOIN process_steps ps ON ps.step_id = t.step_id) l
  ON l.chamber_id = s.chamber_id
ORDER BY step_order, chamber_id, label, bucket_ts;
