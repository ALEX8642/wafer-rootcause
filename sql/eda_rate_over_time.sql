-- eda_rate_over_time.sql — line-wide predicted defect rate in hourly buckets.
-- Bucketed by inspection timestamp (the end-of-line view a fab dashboard
-- shows). A chamber fault dilutes into this line-wide rate by the chamber's
-- share of step traffic, so excursions appear softened here; per-chamber
-- time localisation is Phase 2's job.
SELECT
    date_trunc('hour', i.inspect_ts)                    AS hour_bucket,
    co.label,
    count(*)                                            AS n_wafers,
    count(*) FILTER (WHERE co.predicted)                AS n_defect,
    avg(CASE WHEN co.predicted THEN 1.0 ELSE 0.0 END)   AS defect_rate
FROM inspections i
JOIN classifier_outputs co ON co.wafer_id = i.wafer_id
GROUP BY hour_bucket, co.label
ORDER BY hour_bucket, co.label;
