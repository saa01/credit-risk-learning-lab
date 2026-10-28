"""Check output integrity and exercise the dashboard, including a slider rerun."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
report = json.loads((OUT / "report.json").read_text())
splits = pd.read_csv(OUT / "splits.csv")
predictions = pd.read_csv(OUT / "test_predictions.csv")
assert len(splits) == 30000 and splits.ID.is_unique
assert splits.groupby("split").size().to_dict() == {"test": 6000, "train": 18000, "validation": 6000}
assert set(predictions.ID) == set(splits.loc[splits.split == "test", "ID"])
assert predictions.pd.between(0, 1).all() and np.isfinite(predictions.pd).all()
assert np.isclose(predictions.default.mean(), report["test"]["default_rate"])
with sqlite3.connect(OUT / "credit_risk.db") as connection:
    assert connection.execute("SELECT COUNT(*) FROM test_predictions").fetchone()[0] == 6000
    for query in (ROOT / "analysis.sql").read_text().split(";"):
        if query.strip():
            connection.execute(query).fetchall()
app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
assert not app.exception, app.exception
original_loss = app.metric[-1].value
app.slider[0].set_value(0.0).run(timeout=60)
assert not app.exception, app.exception
assert app.metric[-1].value == "0" and original_loss != "0"
print("Passed: split integrity, prediction bounds, target consistency, SQL queries, dashboard render and loss-slider rerun.")
