"""Stage 05: match each UK city to its most similar US cities.

For each UK city, rank US cities by CSLS (cross-domain similarity local scaling, which
suppresses hub US cities that are everyone's nearest neighbor) on the country-neutralized
representation, and keep the top n. Each match carries a cosine similarity for display
(e.g. dot size / shade on the map).

Reads:  data/processed/reps_<model>.parquet (chosen method), data/processed/cities.parquet
Writes: data/processed/matches_<model>.json  (map-ready, keyed by UK city)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import PROCESSED

MODEL_KEY = "nomic"
N_OUT = 3
CSLS_NBRS = 10
REPORT = [
    "Manchester",
    "Liverpool",
    "Oxford",
    "Cambridge",
    "Brighton and Hove",
    "Blackpool",
    "Edinburgh",
    "Glasgow",
    "York",
    "Bath",
    "Milton Keynes",
]


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def csls_matrix(uk: np.ndarray, us: np.ndarray, kk: int) -> np.ndarray:
    """CSLS(uk,us) = 2*cos - r_us - r_uk on L2-normalized inputs; penalizes hub US cities."""
    cos = uk @ us.T
    r_us = np.sort(cos, axis=0)[-kk:, :].mean(0)
    r_uk = np.sort(cos, axis=1)[:, -kk:].mean(1)
    return 2 * cos - r_us[None, :] - r_uk[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    ap.add_argument(
        "--method", default="centroid", help="representation: centroid | leace | raw_pca"
    )
    args = ap.parse_args()

    reps = pd.read_parquet(PROCESSED / f"reps_{args.model}.parquet")
    reps = reps[reps["method"] == args.method].reset_index(drop=True)
    assert len(reps), f"no rows for method={args.method!r}"
    coords = pd.read_parquet(PROCESSED / "cities.parquet")[["qid", "lat", "lon"]]
    reps = reps.merge(coords, on="qid", how="left")

    us = reps[reps["country"] == "US"].reset_index(drop=True)
    uk = reps[reps["country"] == "UK"].reset_index(drop=True)
    us_name = us["city"].to_numpy()
    us_n = l2(np.vstack(us["embedding"].to_numpy()).astype("float64"))
    uk_n = l2(np.vstack(uk["embedding"].to_numpy()).astype("float64"))

    csls = csls_matrix(uk_n, us_n, CSLS_NBRS)  # ranking (hubness-corrected)
    cos = uk_n @ us_n.T  # cosine for display weights

    records = {}
    for i in range(len(uk)):
        top = np.argsort(-csls[i])[:N_OUT]
        row = uk.iloc[i]
        records[row["wikipedia_title"]] = {
            "city": row["city"],
            "qid": row["qid"],
            "lat": None if pd.isna(row["lat"]) else float(row["lat"]),
            "lon": None if pd.isna(row["lon"]) else float(row["lon"]),
            "matches": [
                {"city": us_name[j], "similarity": round(float(cos[i, j]), 3)} for j in top
            ],
        }

    print(f"model={args.model} method={args.method} | {len(uk)} UK x {len(us)} US dictionary\n")
    name_to_title = {r["city"]: t for t, r in records.items()}
    for q in REPORT:
        if q in name_to_title:
            ms = records[name_to_title[q]]["matches"]
            print(f"  {q:18s} -> " + ", ".join(f"{m['city']} ({m['similarity']:.2f})" for m in ms))

    out_path = PROCESSED / f"matches_{args.model}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    print(f"\nWrote {len(records)} UK cities -> {out_path}")


if __name__ == "__main__":
    main()
