# Credit Risk Learning Lab

A portfolio project for someone comfortable with pandas and basic Python. Build an existing-customer credit card risk model, understand its probabilities, and communicate its limits.

Start with [LEARNING_GUIDE.md](LEARNING_GUIDE.md). This is an educational prototype, not a lending system.

## Run on Windows

Install a supported Python release (Python 3.12 is a conservative choice) and optionally VS Code. Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Skip environment creation if this folder already contains the working `.venv`. Using the interpreter directly avoids PowerShell activation-policy changes. If `python` is unavailable but the Windows Python launcher is installed, use `py -3.12` to create the environment.

The download requires internet access. Subsequent training uses the saved data. Training overwrites outputs reproducibly; preserve a run elsewhere before making experiments. Do not load model files from untrusted sources.

## What is included

| File | Purpose |
|---|---|
| `download_data.py` | Fetch UCI data; preserve customer IDs and provenance |
| `train.py` | Validate inputs, engineer features, split, train, select and evaluate |
| `app.py` | Local results dashboard with an illustrative loss sensitivity |
| `analysis.sql` | Portfolio aggregation and review-capture SQL exercises |
| `LEARNING_GUIDE.md` | Concepts, algorithms, phased build plan and interview preparation |
| `requirements.txt` | Compatible dependency ranges |

Training creates `outputs/report.json`, validation metrics, test predictions, calibration bins, feature importance, a SQLite database, split assignments and a saved model. `requirements-lock.txt`, when present, records the locally tested environment.

The initial run is complete; see [RESULTS.md](RESULTS.md). To reproduce the tested package versions in a new environment, install with `-r requirements-lock.txt` instead of `-r requirements.txt`. Run `.\.venv\Scripts\python.exe verify.py` after training to check artifacts, SQL and dashboard behavior.

Implemented: baseline, logistic regression, gradient boosting, cross-validated sigmoid calibration, 60/20/20 split, validation-only selection, fixed review threshold, SQL outputs, dashboard and a basic group audit.

Extensions to build yourself: bootstrap uncertainty intervals, richer fairness checks, local explanations, actual temporal validation on another suitable dataset, monitoring and Power BI reporting. These are not claimed as completed features.

## Evidence boundaries

- Prediction time: after the latest historical payment/bill information is available; predict the following month's payment-default label.
- Population: existing card customers, not new applicants. These records do not support learning approval policy for rejected applicants.
- Source: one historical cohort, so a random customer holdout measures same-cohort generalization, not future-year robustness.
- SEX, AGE, EDUCATION and MARRIAGE are excluded from model inputs. ID is only a join key. Exclusion alone does not establish fairness.
- Repayment status codes are one-hot encoded, including undocumented special codes; no unsupported ordering is imposed on those codes.
- The loss calculator uses assumed LGD and positive latest bill balance as an EAD proxy. Neither is a fitted recovery/exposure model. It is not IFRS 9 ECL or a capital model.
- Review flags do not demonstrate that interventions prevent defaults. Actual financial benefit would require additional data and causal evaluation.

## Data attribution

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. CC BY 4.0. The downloaded data is renamed and exported to CSV; engineered features are documented in `train.py`. Monetary amounts are New Taiwan dollars (NT$).

Official references: [UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html), [Streamlit installation](https://docs.streamlit.io/get-started/installation).
