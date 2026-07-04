-- eda_label_cooccurrence.sql — predicted label pair co-occurrence.
-- MixedWM38 structure should be visible in predictions: Center/Donut and
-- Edge-Loc/Edge-Ring almost never pair (mutually exclusive by construction),
-- Near-full/Random pair with nothing, Loc and Scratch mix freely. Pairs the
-- data "forbids" showing up here are classifier false alarms stacking on
-- real labels.
SELECT
    a.label      AS label_a,
    b.label      AS label_b,
    count(*)     AS n_wafers
FROM classifier_outputs a
JOIN classifier_outputs b
  ON b.wafer_id = a.wafer_id AND a.label < b.label
WHERE a.predicted AND b.predicted
GROUP BY a.label, b.label
ORDER BY n_wafers DESC, label_a, label_b;
