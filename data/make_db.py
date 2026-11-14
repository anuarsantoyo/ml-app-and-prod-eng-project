"""Build data/ames.db — the Ames housing dataset as a SQLite table.

Drops the identifier and the mostly-empty columns (nothing but noise).
Run once:  /usr/bin/python3 data/make_db.py
"""
import os
import sqlite3
import pandas as pd
from sklearn.datasets import fetch_openml

# Id = identifier. The rest are ~80-99% missing ("NA" mostly means "doesn't have it").
DROP = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]


def main():
    bunch = fetch_openml("house_prices", as_frame=True, parser="auto")
    X = bunch.data
    y = bunch.target
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    df = X.copy()
    df["SalePrice"] = y
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ames.db")
    conn = sqlite3.connect(out)
    df.to_sql("houses", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Wrote {out}")
    print(f"  {df.shape[0]} rows x {df.shape[1]} columns (dropped: {DROP})")


if __name__ == "__main__":
    main()
