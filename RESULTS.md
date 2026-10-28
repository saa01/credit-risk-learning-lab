# Initial verified run — 20 September 2026

The public dataset was downloaded and the full pipeline executed using Python 3.12.14 and scikit-learn 1.9.1. Exact installed packages are in `requirements-lock.txt`.

Training: 18,000 customers. Validation: 6,000. Final test: 6,000. Seed: 42. Splits are stratified random customer holdouts, not chronological holdouts.

The calibrated gradient-boosting model was selected using the lowest **validation log loss**. The table below reports its final-test performance; test scores were not used to change the chosen model or threshold.

| Test measure | Result |
|---|---:|
| ROC-AUC | 0.7901 |
| Average precision | 0.5708 |
| Brier score | 0.1320 |
| Log loss | 0.4220 |
| Observed default rate | 22.12% |
| Customers flagged for review | 20.95% |
| Precision among flagged customers | 56.72% |
| Share of defaults flagged (recall) | 53.73% |

The review threshold was fixed at approximately 31.11% predicted probability using the validation 80th percentile. On test customers, reviewing approximately 21% of accounts captures approximately 54% of observed defaults. This demonstrates prioritization within this historical cohort. It does not demonstrate prevention of defaults, financial savings, or performance on current UK customers.

The base-rate ranking reference has ROC-AUC 0.5 and average precision equal to prevalence. The achieved metrics show useful ranking in this split but require uncertainty intervals and temporal/external validation before stronger conclusions.

Verification passed: unique split assignments and expected sizes; final-test customer identity; probability bounds; target-rate consistency; both SQL queries; Streamlit application render through AppTest; and a slider rerun confirming that zero assumed LGD produces zero illustrative loss. This is a functional dashboard check, not a manual visual browser review.

Data SHA-256: `8c05bc24646b74bec483f240be19bd21483170f029c5a7e30c6553a8ab452e91`.

Next learning task: reproduce the target distribution in pandas and explain why the model's 53.73% recall and 56.72% precision answer different questions. Avoid optimizing against these final-test numbers in further experiments; use training cross-validation and the validation set.
