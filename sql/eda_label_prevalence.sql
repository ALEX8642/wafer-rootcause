-- eda_label_prevalence.sql — predicted-label prevalence across the line.
-- Baseline context for every other EDA view: how common is each signature
-- as the CLASSIFIER sees it (analysis queries never read ground truth).
SELECT
    co.label,
    count(*)                                            AS n_wafers,
    count(*) FILTER (WHERE co.predicted)                AS n_predicted,
    avg(CASE WHEN co.predicted THEN 1.0 ELSE 0.0 END)   AS predicted_rate
FROM classifier_outputs co
GROUP BY co.label
ORDER BY predicted_rate DESC;
