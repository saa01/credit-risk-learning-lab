"""Download the public UCI dataset once and record its provenance."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
         "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
         *[f"BILL_AMT{i}" for i in range(1, 7)],
         *[f"PAY_AMT{i}" for i in range(1, 7)]]


def main():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    target = folder / "clients.csv"
    if target.exists():
        raise SystemExit("Data already exists; reuse it to preserve reproducibility.")
    source = fetch_ucirepo(id=350)
    x = source.data.features.copy()
    x = x.rename(columns={f"X{i}": name for i, name in enumerate(NAMES, 1)})
    if set(x.columns) != set(NAMES):
        raise ValueError(f"Unexpected source columns: {list(x.columns)}")
    y = source.data.targets.iloc[:, 0].rename("default")
    ids = source.data.ids
    if ids is None:
        raise ValueError("Source customer IDs missing; inspect the source before proceeding.")
    df = pd.concat([ids.iloc[:, 0].rename("ID"), x, y], axis=1)
    df.to_csv(target, index=False)
    provenance = {
        "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. DOI: 10.24432/C55S3H",
        "license": "CC BY 4.0", "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "rows": len(df), "currency": "New Taiwan dollar (NT$)",
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Saved {len(df):,} customers to {target}")


if __name__ == "__main__":
    main()
