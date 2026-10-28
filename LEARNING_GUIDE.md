# Build and understand your credit risk project

Your project title: **Credit Card Default Risk: Prediction, Validation and Portfolio Review**.

Your business question: “Using information already available for an existing credit card customer, how likely is next-month payment default, and which accounts should an analyst review first?”

The work demonstrates Python, SQL, statistics, machine learning, financial reasoning and communication. You should be able to explain every major choice. A defensible small project is more useful in an interview than a complex system you cannot explain.

## 1. Understand the three risk quantities

**Probability of default (PD):** the chance of the defined default event within a specified period. In this project, that is the dataset's next-month payment-default event. A PD of 0.10 means an estimated 10% chance, not a certainty that this specific customer will or will not default.

**Loss given default (LGD):** the fraction of exposure lost if default happens, after recoveries under the chosen definition.

**Exposure at default (EAD):** the amount exposed when default occurs. Credit limit, current bill and actual EAD are different quantities.

For a simple educational calculation:

`Expected loss = PD × LGD × EAD`

If PD = 10%, assumed LGD = 45%, and assumed EAD = NT$10,000, illustrative expected loss is NT$450. For many accounts, add account-level expected losses. Keep the event, horizon and currency consistent. A full accounting expected-credit-loss calculation requires considerably more detail.

The dataset lets us learn a payment-default probability. It does not provide recovery outcomes or true EAD at default, so LGD and EAD are assumptions/proxies in the optional calculator.

**Checkpoint:** explain why a model with accurate ranking but exaggerated PDs could overstate losses.

## 2. Know what each software tool does

| Tool | Your use | Need it now? |
|---|---|---|
| Python + virtual environment | Execute the analysis with project-specific packages | Yes |
| VS Code | Edit scripts and inspect results | Useful; another editor is fine |
| pandas + NumPy | Data quality checks, feature construction and tables | Yes |
| scikit-learn | Preprocessing, classifiers, calibration and metrics | Yes |
| SQLite | Practice SQL on a local database; Python includes its driver | Included |
| Streamlit | Run the interactive results dashboard locally | For the demo |
| Git + GitHub | Track changes and publish a readable portfolio repository | At publication stage |
| Jupyter | Explore one concept at a time in notebooks | Optional |
| Power BI Desktop | Add a management-facing report from exported results | Optional later |
| SHAP | Explore individual model explanations | Optional later |

Use the setup commands in README. You do not need a GPU, paid cloud services, Docker or a complex database for the first version. Avoid installing every tool before you understand the model.

## 3. Understand the dataset before fitting anything

The UCI source describes 30,000 Taiwan credit card customers with repayment and bill history from April to September 2005. Target 1 denotes payment default and 0 denotes no payment default. Read the source dictionary before interpreting field values.

Look at `data/clients.csv` using pandas:

```python
import pandas as pd
df = pd.read_csv("data/clients.csv")
print(df.shape)
print(df.head())
print(df.isna().sum())
print(df["default"].value_counts(normalize=True))
print(df.groupby("PAY_0")["default"].agg(["size", "mean"]))
```

After the split, use training rows for exploratory choices about features. Full-data schema checks are fine, but do not use final-test relationships to redesign your model.

Investigate repeated IDs, unexpected category codes, negative bill balances, large payment values and unusual utilization. A negative bill is not automatically a data error. Document decisions instead of deleting inconvenient observations. Some repayment codes are not fully described by the public dictionary; the starter preserves their distinct categories.

**Deliverable:** a one-page data dictionary with input name, meaning, unit, availability at prediction time and quality concerns.

## 4. Protect yourself against misleading results

The starter assigns 60% to training, 20% to validation and 20% to final testing, stratified to preserve the approximate target proportion.

- Training teaches the models and their preprocessing parameters.
- Validation selects the model and fixes the review threshold.
- Test data estimates performance of that frozen choice.

Preprocessing is inside pipelines. For calibration, the entire model pipeline is fitted inside training-only cross-validation folds, so calibration examples are not used to fit their underlying classifier.

Data leakage means using information unavailable when a real prediction would have been made. Examples include later collection results, repayment after the prediction date, or scaling using all rows before splitting. Customer ID must not be a predictor.

This dataset is a single cohort. Its six history columns do not create six independent monthly prediction cohorts. Do not claim the random split is a time-based backtest. A later extension needs genuinely dated snapshots and fully observed future outcomes.

**Checkpoint:** explain why repeatedly changing the model after looking at test scores turns that test set into another validation set.

## 5. Build features with a reason

The starter uses credit limit, six bill amounts, six payment amounts, repayment categories and three derived features:

| Feature | Calculation | Hypothesis to check |
|---|---|---|
| Latest utilization | latest bill / credit limit | High utilization may signal financial pressure |
| Months delayed | count of positive repayment delay codes | Repeated late payment may predict future difficulty |
| Relative bill change | (latest bill − oldest bill) / credit limit | Increasing balances may provide useful context |

These are hypotheses about prediction, not causal claims. Ratios are computed per customer, and the code checks for nonpositive credit limits before dividing. Age and demographic categories are omitted from modeling in this first version and retained where useful for audit.

**Exercise:** add one feature, document why it was known at prediction time, and assess it within training cross-validation or validation. Avoid collecting many candidate improvements from final-test results.

## 6. Understand the algorithms in order

**A. Constant base-rate model.** Predict the training default rate for every customer. It cannot rank customers but is an essential reference for probability quality. If the complex model cannot improve on this, investigate.

**B. Logistic regression.** Form a weighted combination of inputs, then map it into a probability:

`z = intercept + b1*x1 + ... + bk*xk`

`PD = 1 / (1 + exp(-z))`

Despite its name, this is a classification algorithm. Coefficients act on log-odds, not directly on probability. Numeric variables are standardized; repayment status categories are one-hot encoded. Regularization discourages extreme coefficients. Coefficients can be unstable when inputs are strongly correlated and do not establish causes.

**C. Gradient-boosted trees.** Build successive small trees that improve the current prediction. They can learn nonlinear patterns and interactions, such as repayment delay having different implications at different utilization levels. The starter uses scikit-learn's histogram-based implementation to keep dependencies small. XGBoost or LightGBM are later alternatives, not automatic upgrades.

**D. Probability calibration.** Learn a correction from model scores to observed outcome frequencies using held-out training folds. Among customers with predicted PD around 20%, roughly 20% should default over the defined horizon if the model is well calibrated for that population. Calibration can help or hurt; compare calibrated and uncalibrated versions.

A decision tree is a useful teaching exercise; a random forest is a reasonable optional benchmark. Neural networks, clustering and large language models are unnecessary for the first version.

**Deliverable:** a comparison table with the same splits and metrics for every candidate, including the constant baseline. The code selects lowest validation log loss and then evaluates the chosen model once on test data.

## 7. Evaluate more than accuracy

| Measure | Question answered | Better direction |
|---|---|---|
| ROC-AUC | Does the model generally rank defaulters above non-defaulters? | Higher |
| Average precision | How well does it find defaults across precision/recall trade-offs? | Higher; compare with default prevalence |
| Log loss | Are probability predictions useful, including costly confident mistakes? | Lower |
| Brier score | How far are probabilities from 0/1 outcomes on average? | Lower |
| Calibration bins/curve | Do stated probabilities match observed frequencies? | Close agreement with adequate counts |
| Recall at review threshold | What share of actual defaulters are flagged? | Higher, subject to review capacity |
| Precision at review threshold | What share of flagged customers actually default? | Higher, subject to coverage |

Average precision summarizes a precision-recall curve; it is not identical to trapezoidal PR-AUC. Brier score reflects more than calibration, so inspect calibration bins as well.

If 80% of customers do not default, predicting no default for everyone gives 80% accuracy while identifying no defaulters. Do not use accuracy as your main success claim. Avoid automatic oversampling: altering prevalence can distort raw probabilities. If you later test class weighting or resampling, perform it inside training folds and reassess calibration.

**Exercise:** interpret one false positive and one false negative in the context of an account-review workflow.

## 8. Connect predictions to a business workflow

The starter fixes a probability threshold at the validation 80th percentile: an illustrative review workload of roughly the highest-risk 20%. The actual workload may differ because of tied scores or distribution differences. Report that difference honestly.

Use `analysis.sql` to answer which risk bands contain the most customers, which have higher observed default rates, and what share of defaults the fixed policy captures. Review capacity is an assumption, not a discovered optimal business policy. There is no measured cost or intervention benefit here.

Use the dashboard to compare models, examine calibration, inspect feature reliance and change loss assumptions. Raising a PD multiplier is a sensitivity exercise. It does not prove how a recession would affect these customers.

For a Power BI extension, import `outputs/test_predictions.csv` and create three pages: portfolio overview, model validation, and operational review. Display units, cohort, sample counts and assumptions beside charts. Do not describe flagging existing cardholders as approving new loan applicants.

## 9. Make the project stronger than a typical notebook

Prioritize evidence and reproducibility over the number of models:

1. **Clear specification:** population, prediction date, outcome definition and horizon.
2. **Reproducibility:** source citation, file hash, exact environment, seed and saved split assignments.
3. **Reliable probabilities:** compare calibration as well as ranking.
4. **Uncertainty:** bootstrap final-test customers for intervals around AUC, average precision and policy metrics. These intervals condition on the fitted model; they do not capture all model-selection uncertainty.
5. **Interpretability:** global permutation importance first; add local explanations later. Explain correlation limitations and avoid causal language.
6. **Fairness investigation:** inspect group sample sizes, calibration, recall and false-positive rates. Excluding sex does not eliminate demographic proxies. A group comparison is not a legal compliance certification.
7. **Temporal evidence:** extend with an appropriate longitudinal dataset, train on older fully observed cohorts and test on newer ones. This is a larger improvement in realism than adding another algorithm.
8. **Monitoring design:** track missingness, feature distributions and score distributions immediately; track outcome-based metrics after labels mature. Document investigation and retraining triggers. Artificial drift is a drill, not evidence of actual deterioration.
9. **Business memo:** explain what decision the tool supports, what performance you observed and what data is needed before use in practice.

**Portfolio package:** code repository, clear README, one management dashboard, a short model card, a two-page business memo and a three-minute recorded demonstration. Publish measured results only; never invent AUC, savings or production readiness.

## 10. A manageable learning schedule

| Sessions | Work | You are done when… |
|---|---|---|
| 1–2 | Setup, business problem and data dictionary | You can explain the target and each feature group |
| 3–4 | Quality checks, split design and training-only exploration | You can identify leakage and reproduce the splits |
| 5–6 | Baseline and logistic regression | You can explain PD, coefficients and log loss |
| 7–8 | Boosting and calibration | You can explain the comparison without reading the code |
| 9–10 | SQL, fixed review policy and dashboard | You can translate metrics into analyst workload |
| 11–12 | Limits, group audit and portfolio write-up | You can defend the work in a short interview demo |

Allow around 60–90 minutes per session and extra time for experiments. Build the extensions after you can explain the first version. Running the supplied code is a starting point; the exercises turn it into work you understand and can defend.

Your first session: run the downloader, inspect five rows and the target proportions, then write five sentences describing who is being scored, when, for which outcome, over what horizon and with which limitations.

## Official references

- [UCI dataset and dictionary](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Streamlit setup](https://docs.streamlit.io/get-started/installation)
