-- attr_windows.sql — fault-window localisation per (chamber, label).
--
-- For each chamber x label cell: defect rate in bucket_hours buckets of
-- chamber-processing time (same construction as attr_bucket_rates.sql),
-- smoothed with a centred 3-bucket rolling window, then simple
-- threshold-crossing: a bucket is flagged when the rolling rate exceeds
-- the whole-horizon rest-of-step rate by `sigma` binomial standard errors
-- (given the rolling sample size). Contiguous flagged runs are grouped
-- (gaps-and-islands) and the run with the largest excess-defect mass is
-- reported as THE excursion window for the cell. Cells with no qualifying
-- run produce no row.
--
-- Deliberately simple (threshold-crossing, not CUSUM): localisation only
-- has to place the window well enough for a maintenance log pull, and the
-- scorer reports IoU against ground truth so the simplicity is priced.
--
-- Analysis side: reads classifier_outputs only, never ground_truth_*.
-- Portability: see attr_bucket_rates.sql note (time_bucket/generate_series).

WITH params AS (
    SELECT 6    AS bucket_hours,     -- 4 h fragments runs on sparse chambers
           2.0  AS sigma,            -- rolling-rate threshold, binomial SEs
           10   AS min_roll_n,       -- min wafers in the rolling window
           2    AS min_run_buckets,  -- min contiguous flagged buckets
           5.0  AS min_excess        -- min excess defects in the run: a rate
                                     -- blip carried by 1-2 wafers (near-zero
                                     -- baseline labels) is not an excursion
),

pass AS (
    SELECT h.chamber_id, co.label, h.start_ts, co.predicted
    FROM wafer_process_history h
    JOIN classifier_outputs co ON co.wafer_id = h.wafer_id
),

-- whole-horizon rest-of-step rate: the control level the rolling rate
-- must clear (same contrast as attr_suspects.sql)
cham AS (
    SELECT chamber_id, label,
           count(*)                          AS n_cham,
           count(*) FILTER (WHERE predicted) AS k_cham
    FROM pass
    GROUP BY chamber_id, label
),

lineage AS (
    SELECT ch.chamber_id, t.step_id
    FROM chambers ch
    JOIN tools t ON t.tool_id = ch.tool_id
),

rest AS (
    SELECT c.chamber_id, c.label,
           (sum(c2.k_cham) - c.k_cham) * 1.0
         / (sum(c2.n_cham) - c.n_cham) AS rate_rest
    FROM cham c
    JOIN lineage l  ON l.chamber_id  = c.chamber_id
    JOIN lineage l2 ON l2.step_id    = l.step_id
    JOIN cham   c2  ON c2.chamber_id = l2.chamber_id AND c2.label = c.label
    GROUP BY c.chamber_id, c.label, c.k_cham, c.n_cham
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
),

grid AS (
    SELECT s.chamber_id, s.label, s.bucket_ts,
           coalesce(b.n, 0) AS n,
           coalesce(b.k, 0) AS k
    FROM spine s
    LEFT JOIN bucket_counts b
           ON b.chamber_id = s.chamber_id AND b.label = s.label
          AND b.bucket_ts  = s.bucket_ts
),

rolled AS (
    SELECT *,
        sum(n) OVER w AS roll_n,
        sum(k) OVER w AS roll_k
    FROM grid
    WINDOW w AS (PARTITION BY chamber_id, label ORDER BY bucket_ts
                 ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
),

flagged AS (
    SELECT r.*, rest.rate_rest,
        CASE WHEN r.roll_n >= p.min_roll_n
              AND r.roll_k * 1.0 / r.roll_n
                  > rest.rate_rest
                    + p.sigma * sqrt(rest.rate_rest * (1 - rest.rate_rest)
                                     / r.roll_n)
             THEN 1 ELSE 0 END AS flag
    FROM rolled r
    JOIN rest ON rest.chamber_id = r.chamber_id AND rest.label = r.label
    CROSS JOIN params p
),

-- gaps-and-islands: contiguous same-flag runs share a group key
islands AS (
    SELECT *,
        row_number() OVER (PARTITION BY chamber_id, label
                           ORDER BY bucket_ts)
      - row_number() OVER (PARTITION BY chamber_id, label, flag
                           ORDER BY bucket_ts) AS grp
    FROM flagged
),

runs AS (
    SELECT chamber_id, label,
           min(bucket_ts)                                    AS win_start,
           max(bucket_ts) + INTERVAL 1 HOUR * min(p.bucket_hours) AS win_end,
           count(*)                                          AS n_buckets,
           sum(n)                                            AS n_wafers,
           sum(k)                                            AS k_defect,
           sum(k - rate_rest * n)                            AS excess_defects
    FROM islands, params p
    WHERE flag = 1
    GROUP BY chamber_id, label, grp
)

SELECT chamber_id, label, win_start, win_end,
       n_buckets,
       CAST(n_wafers AS BIGINT) AS n_wafers,
       CAST(k_defect AS BIGINT) AS k_defect,
       excess_defects
FROM (
    SELECT r.*,
        row_number() OVER (PARTITION BY chamber_id, label
                           ORDER BY excess_defects DESC) AS best
    FROM runs r, params p
    WHERE r.n_buckets >= p.min_run_buckets
      AND r.excess_defects >= p.min_excess
)
WHERE best = 1
ORDER BY excess_defects DESC;
