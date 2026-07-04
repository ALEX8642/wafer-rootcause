-- eda_rate_by_chamber.sql — predicted defect rate per step/tool/chamber/label.
-- The raw material of Phase 2's commonality analysis: for wafers that passed
-- through a chamber, how often does end-of-line classification carry each
-- signature label? A planted chamber fault shows up as one chamber's rate
-- sitting far above its step siblings for one label.
SELECT
    ps.step_order,
    ps.step_name,
    t.tool_id,
    ch.chamber_id,
    co.label,
    count(*)                                            AS n_wafers,
    count(*) FILTER (WHERE co.predicted)                AS n_defect,
    avg(CASE WHEN co.predicted THEN 1.0 ELSE 0.0 END)   AS defect_rate
FROM wafer_process_history h
JOIN chambers        ch ON ch.chamber_id = h.chamber_id
JOIN tools           t  ON t.tool_id     = ch.tool_id
JOIN process_steps   ps ON ps.step_id    = h.step_id
JOIN classifier_outputs co ON co.wafer_id = h.wafer_id
GROUP BY ps.step_order, ps.step_name, t.tool_id, ch.chamber_id, co.label
ORDER BY ps.step_order, ch.chamber_id, co.label;
