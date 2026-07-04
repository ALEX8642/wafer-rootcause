-- attr_suspects.sql — commonality analysis: ranked chamber suspects per label.
--
-- For every (signature label, chamber) cell, contrast the end-of-line
-- predicted defect rate of wafers that passed through the chamber against
-- wafers that went through the OTHER chambers of the same route step
-- (every wafer visits every step exactly once, so rest-of-step is the
-- natural control group). One-sided pooled two-proportion z-test
-- (H1: chamber rate > rest-of-step rate), Benjamini–Hochberg correction
-- across the full label x chamber grid, then a per-label suspect ranking.
--
-- Analysis side: reads classifier_outputs only, never ground_truth_*.
--
-- Portability: pure arithmetic + window functions (Postgres-ready). The
-- normal tail P(Z >= z) is inlined via the Abramowitz–Stegun 7.1.26 erfc
-- approximation (|error| <= 1.5e-7 — far below any BH decision made here)
-- because stock DuckDB has no erf/erfc.

WITH params AS (
    SELECT 0.05 AS alpha                    -- BH false-discovery rate
),

-- one row per wafer x chamber visited x label decision
pass AS (
    SELECT h.chamber_id, co.label, co.predicted
    FROM wafer_process_history h
    JOIN classifier_outputs co ON co.wafer_id = h.wafer_id
),

cham AS (
    SELECT chamber_id, label,
           count(*)                             AS n_cham,
           count(*) FILTER (WHERE predicted)    AS k_cham
    FROM pass
    GROUP BY chamber_id, label
),

-- chamber -> tool -> step lineage, plus step totals for the control group
lineage AS (
    SELECT ch.chamber_id, t.tool_id, ps.step_id, ps.step_order, ps.step_name
    FROM chambers ch
    JOIN tools t          ON t.tool_id  = ch.tool_id
    JOIN process_steps ps ON ps.step_id = t.step_id
),

step_tot AS (
    SELECT l.step_id, c.label,
           sum(c.n_cham) AS n_step,
           sum(c.k_cham) AS k_step
    FROM cham c
    JOIN lineage l ON l.chamber_id = c.chamber_id
    GROUP BY l.step_id, c.label
),

contrast AS (
    SELECT
        c.label, l.step_order, l.step_name, l.tool_id, c.chamber_id,
        c.n_cham, c.k_cham,
        c.k_cham * 1.0 / c.n_cham                       AS rate_cham,
        s.n_step - c.n_cham                             AS n_rest,
        s.k_step - c.k_cham                             AS k_rest,
        -- NULL, not a 0-division, when the chamber IS the whole step
        -- (single-chamber steps have no within-step control group)
        CASE WHEN s.n_step > c.n_cham
             THEN (s.k_step - c.k_cham) * 1.0 / (s.n_step - c.n_cham)
        END                                             AS rate_rest,
        s.k_step * 1.0 / s.n_step                       AS p_pool
    FROM cham c
    JOIN lineage  l ON l.chamber_id = c.chamber_id
    JOIN step_tot s ON s.step_id = l.step_id AND s.label = c.label
),

tested AS (
    SELECT *,
        rate_cham - rate_rest AS excess,
        CASE WHEN p_pool <= 0 OR p_pool >= 1 OR n_rest = 0
             THEN 0.0                                   -- degenerate: no evidence
             ELSE (rate_cham - rate_rest)
                  / sqrt(p_pool * (1 - p_pool) * (1.0 / n_cham + 1.0 / n_rest))
        END AS z
    FROM contrast
),

-- one-sided p = P(Z >= z), Abramowitz–Stegun 7.1.26 on |z|
pvalued AS (
    SELECT t.*,
        CASE WHEN z >= 0 THEN tail ELSE 1 - tail END AS p_value
    FROM (
        SELECT *,
            0.5 * exp(-z * z / 2) * (
                  0.254829592  * u
                - 0.284496736  * u * u
                + 1.421413741  * u * u * u
                - 1.453152027  * u * u * u * u
                + 1.061405429  * u * u * u * u * u
            ) AS tail
        FROM (SELECT *, 1.0 / (1.0 + 0.3275911 * abs(z) / sqrt(2.0)) AS u
              FROM tested)
    ) t
),

-- Benjamini–Hochberg across the full grid: q_i = min_{j >= i} p_(j) * m / j
ranked AS (
    SELECT *,
        row_number() OVER (ORDER BY p_value, z DESC) AS p_rank,
        count(*)     OVER ()                         AS m
    FROM pvalued
),

bh AS (
    SELECT *,
        least(1.0, min(p_value * m / p_rank) OVER (
            ORDER BY p_rank DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS q_value
    FROM ranked
)

SELECT
    label, step_order, step_name, tool_id, chamber_id,
    n_cham, k_cham, rate_cham, n_rest, k_rest, rate_rest,
    excess, z, p_value, q_value,
    q_value <= alpha AS significant,
    row_number() OVER (PARTITION BY label ORDER BY z DESC) AS suspect_rank
FROM bh, params
ORDER BY label, suspect_rank;
