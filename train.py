"""Train on 60%, choose on 20%, and evaluate a frozen choice on 20%."""
import hashlib
import json
import platform
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = ["LIMIT_BAL", *[f"BILL_AMT{i}" for i in range(1, 7)],
           *[f"PAY_AMT{i}" for i in range(1, 7)],
           "utilization", "months_delayed", "bill_change"]
FEATURES = NUMERIC + PAY


def features(df):
    x = df.copy()
    # Row-local calculations do not learn anything from validation/test customers.
    x["utilization"] = x["BILL_AMT1"] / x["LIMIT_BAL"]
    x["months_delayed"] = (x[PAY] > 0).sum(axis=1)
    x["bill_change"] = (x["BILL_AMT1"] - x["BILL_AMT6"]) / x["LIMIT_BAL"]
    # Preserve zero/negative repayment codes as categories; do not invent meanings.
    return x[FEATURES]


def preprocessing():
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("repayment", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PAY),
    ])


def metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)),
            "default_rate": float(np.mean(y))}


def main():
    path = ROOT / "data" / "clients.csv"
    if not path.exists():
        raise SystemExit("Run python download_data.py first.")
    df = pd.read_csv(path)
    if len(df) != 30000 or df.ID.isna().any() or df.ID.duplicated().any():
        raise ValueError("Expected 30,000 unique customer IDs.")
    if df.isna().any().any() or not set(df.default.unique()) == {0, 1}:
        raise ValueError("Unexpected missing values or target coding; inspect the source.")
    if (df.LIMIT_BAL <= 0).any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError("Invalid credit limits or non-finite values.")

    train, rest = train_test_split(df.index, test_size=.4, stratify=df.default, random_state=SEED)
    valid, test = train_test_split(rest, test_size=.5, stratify=df.loc[rest, "default"], random_state=SEED)
    assert not (set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test))
    x, y = features(df), df.default
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    assignments = pd.Series(index=df.index, dtype="object")
    for label, ix in [("train", train), ("validation", valid), ("test", test)]:
        assignments.loc[ix] = label
    pd.DataFrame({"ID": df.ID, "split": assignments}).to_csv(out / "splits.csv", index=False)

    logistic = make_pipeline(preprocessing(), LogisticRegression(max_iter=2000, C=1.0))
    boost = make_pipeline(preprocessing(), HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED))
    models = {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic": logistic,
        "boosting": boost,
        "calibrated_boosting": CalibratedClassifierCV(
            estimator=boost, method="sigmoid",
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED)),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(x.loc[train], y.loc[train])
        p = model.predict_proba(x.loc[valid])[:, 1]
        rows.append({"model": name, **metrics(y.loc[valid], p)})
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("log_loss")
    comparison.to_csv(out / "validation_metrics.csv", index=False)
    # Selection uses validation ONLY; never switch winner after viewing test scores.
    winner = comparison.iloc[0]["model"]
    model = fitted[winner]
    validation_pd = model.predict_proba(x.loc[valid])[:, 1]
    # Illustrative review capacity: flag roughly the highest-risk 20% in validation.
    threshold = float(np.quantile(validation_pd, .8))
    p = model.predict_proba(x.loc[test])[:, 1]
    flagged = p >= threshold
    report = {"winner": winner, "selection_rule": "Lowest validation log loss",
              "seed": SEED, "sizes": {"train": len(train), "validation": len(valid), "test": len(test)},
              "review_threshold": threshold,
              "test": {**metrics(y.loc[test], p), "flag_rate": float(flagged.mean()),
                       "precision": float(precision_score(y.loc[test], flagged, zero_division=0)),
                       "recall": float(recall_score(y.loc[test], flagged, zero_division=0))},
              "python": platform.python_version(), "sklearn": sklearn.__version__,
              "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "limitations": ["Single historical Taiwan cohort; no out-of-time validation.",
                               "Payment-default label is not a verified regulatory default definition.",
                               "No observed recoveries, EAD at default, or causal intervention effects."]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    predictions = df.loc[test, ["ID", "default", "SEX", "AGE", "LIMIT_BAL", "BILL_AMT1"]].copy()
    predictions["pd"] = p
    predictions["review_flag"] = flagged.astype(int)
    predictions["risk_band"] = pd.cut(p, [-.001, .1, .2, .4, 1], labels=["<10%", "10-20%", "20-40%", "40%+"])
    predictions.to_csv(out / "test_predictions.csv", index=False)
    with sqlite3.connect(out / "credit_risk.db") as con:
        predictions.to_sql("test_predictions", con, index=False, if_exists="replace")
    calibration = predictions.assign(bin=pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)).groupby(
        "bin", observed=True).agg(n=("pd", "size"), mean_pd=("pd", "mean"), actual_default=("default", "mean"))
    calibration.to_csv(out / "calibration.csv")
    # Model inspection on validation, keeping test outcomes out of feature decisions.
    importance = permutation_importance(model, x.loc[valid], y.loc[valid], scoring="neg_log_loss",
                                        n_repeats=3, random_state=SEED)
    pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean,
                  "std": importance.importances_std}).sort_values("importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, out / "model.joblib")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
