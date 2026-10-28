-- Run against outputs/credit_risk.db using SQLite or a SQLite editor.
-- Question 1: do observed default rates rise across predicted risk bands?
SELECT risk_band, COUNT(*) AS customers,
       ROUND(AVG(pd), 4) AS predicted_default_rate,
       ROUND(AVG("default"), 4) AS observed_default_rate,
       SUM(CASE WHEN BILL_AMT1 > 0 THEN BILL_AMT1 ELSE 0 END) AS positive_bill_balance_ntd
FROM test_predictions
GROUP BY risk_band
ORDER BY predicted_default_rate;

-- Question 2: what proportion of defaults were captured by the frozen review rule?
SELECT SUM(review_flag) AS flagged_customers,
       SUM(review_flag * "default") AS flagged_defaults,
       1.0 * SUM(review_flag * "default") / NULLIF(SUM("default"), 0) AS default_recall
FROM test_predictions;
