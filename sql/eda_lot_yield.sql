-- eda_lot_yield.sql — lot-level clean yield distribution.
-- Yield here = share of a lot's wafers with NO predicted signature label
-- (the classifier's "clean" verdict, not ground truth). Lots that ran
-- through a faulty chamber during its window should sink to the bottom.
WITH wafer_flags AS (
    SELECT
        w.lot_id,
        w.wafer_id,
        max(CASE WHEN co.predicted THEN 1 ELSE 0 END) AS any_defect
    FROM wafers w
    JOIN classifier_outputs co ON co.wafer_id = w.wafer_id
    GROUP BY w.lot_id, w.wafer_id
)
SELECT
    lot_id,
    count(*)                       AS n_wafers,
    sum(any_defect)                AS n_defective,
    1.0 - avg(any_defect)          AS clean_yield
FROM wafer_flags
GROUP BY lot_id
ORDER BY clean_yield, lot_id;
