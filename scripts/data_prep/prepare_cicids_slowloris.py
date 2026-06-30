"""
Build the Slowloris-vs-benign dataset from CICIDS2017 (UNB CIC).

The original "SlowlorisBig" set used by Hasanin et al. (2019) is not published, so
we reconstruct a comparable severely-imbalanced task from the public CICIDS2017
Wednesday capture, which labels the "DoS slowloris" attack against benign traffic.

Source: CICIDS2017, Sharafaldin, Lashkari & Ghorbani (2018), ICISSP. The flows are
mirrored on Hugging Face (c01dsnap/CIC-IDS2017); the dataset is publicly available
for research with citation of the paper above.

This downloads the Wednesday CSV (~225 MB), keeps only BENIGN and DoS-slowloris
rows, coerces features to numeric, drops infinities / missing values and
zero-variance columns, and caches the result to
``.data_cache/cicids_slowloris.pkl`` (gitignored) for
``scripts/training_scripts/train-cicids-slowloris.py``.
"""

import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / ".data_cache"
RAW = CACHE / "Wednesday-workingHours.pcap_ISCX.csv"
OUT = CACHE / "cicids_slowloris.pkl"
URL = (
    "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/"
    "Wednesday-workingHours.pcap_ISCX.csv"
)


def main():
    CACHE.mkdir(exist_ok=True)
    if not RAW.exists():
        print(f"downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, RAW)
    print("loading + cleaning ...", flush=True)

    df = pd.read_csv(RAW, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    label = df.columns[-1]
    df = df[df[label].isin(["BENIGN", "DoS slowloris"])].copy()

    y = (df[label] == "DoS slowloris").astype("int8").to_numpy()
    X = df.drop(columns=[label]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    keep = ~X.isna().any(axis=1)
    X, y = X[keep], y[keep]
    nunique = X.nunique()
    X = X.drop(columns=nunique[nunique <= 1].index)  # drop zero-variance columns

    out = {"X": X.to_numpy(dtype="float32"), "y": y, "features": list(X.columns)}
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(
        f"saved -> {OUT}\n"
        f"  rows: {len(y):,} | slowloris: {int(y.sum()):,} "
        f"({100 * y.mean():.3f}%) | features: {out['X'].shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
