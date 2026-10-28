"""Read the saved results; do not tune models against the final test set."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent / "outputs"
st.set_page_config(page_title="Credit Risk Learning Lab", layout="wide")
st.title("Credit Risk Learning Lab")
st.caption("Historical Taiwan credit card cohort • next-month payment-default prediction • educational use")
if not (OUT / "report.json").exists():
    st.info("Run python download_data.py and python train.py first.")
    st.stop()
report = json.loads((OUT / "report.json").read_text())
df = pd.read_csv(OUT / "test_predictions.csv")
st.write(f"Selected model: **{report['winner']}**. Selection rule: {report['selection_rule']}.")
a, b, c = st.columns(3)
a.metric("Test ROC-AUC", f"{report['test']['roc_auc']:.3f}")
b.metric("Test average precision", f"{report['test']['average_precision']:.3f}")
c.metric("Observed default rate", f"{df['default'].mean():.1%}")
st.subheader("Model comparison — validation customers")
st.dataframe(pd.read_csv(OUT / "validation_metrics.csv"), hide_index=True)
st.subheader("Do predicted probabilities match observed outcomes?")
st.dataframe(pd.read_csv(OUT / "calibration.csv"), hide_index=True)
st.caption("Compare mean_pd with actual_default; small bins are uncertain. Brier score alone is not a calibration diagnosis.")
st.subheader("Frozen review policy — final test customers")
st.write({k: report["test"][k] for k in ["flag_rate", "precision", "recall"]})
st.caption("Threshold chosen from validation for approximately 20% review capacity. Ties and distribution changes can alter the actual flag rate. A flag means review, not decline.")
st.subheader("Illustrative loss sensitivity")
st.warning("These are assumption-driven loss proxies, not measured losses, regulatory ECL, or proven savings.")
lgd = st.slider("Assumed loss given default", 0.0, 1.0, .45, .05)
multiplier = st.slider("Hypothetical PD multiplier", 1.0, 3.0, 1.0, .1)
ead_proxy = df.BILL_AMT1.clip(lower=0)
loss = (df.pd.mul(multiplier).clip(upper=1) * lgd * ead_proxy).sum()
st.metric("Illustrative total loss proxy — test cohort (NT$)", f"{loss:,.0f}")
st.caption("EAD proxy = positive latest bill balance. Actual exposure at default is unavailable. The PD multiplier is an arbitrary sensitivity, not an estimated recession effect.")
st.subheader("Which inputs does the model rely on?")
st.bar_chart(pd.read_csv(OUT / "feature_importance.csv").set_index("feature")["importance"])
st.caption("Permutation importance measures predictive reliance, not causes. Correlated inputs can share or hide importance.")
st.subheader("Exploratory group audit")
st.dataframe(df.groupby("SEX").agg(customers=("ID", "size"), mean_pd=("pd", "mean"),
    observed_default=("default", "mean"), review_rate=("review_flag", "mean")))
st.caption("Sex and other demographics are excluded from training. Exclusion does not prove fairness; proxies and group calibration still need review.")
st.subheader("Limits of the evidence")
for limitation in report["limitations"]:
    st.write(f"• {limitation}")
