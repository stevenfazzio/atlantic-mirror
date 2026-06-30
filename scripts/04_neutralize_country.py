"""Stage 04: reduce to a shared PCA space and neutralize the country signal.

Builds three representations in a shared ~50-dim PCA space and compares them:
  raw_pca   PCA only (control)
  centroid  PCA + per-country mean subtraction (removes the first-moment country offset)
  leace     PCA + LEACE (linear concept erasure: no linear classifier can recover country)

For each we report the country-confound diagnostics (nearest-neighbor same-country rate;
US-vs-UK linear separability) and a qualitative UK->US neighbor preview, so we can watch the
two country blobs merge while within-country character sharpens. CSLS / hubness correction
is deliberately left for the reconstruction stage; here we use plain cosine.

Reads:  data/processed/embeddings_<model>.parquet
Writes: data/processed/reps_<model>.parquet  (long: identity + method + embedding)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from _common import PROCESSED, write_df

MODEL_KEY = "nomic"
N_PCA = 50
ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]
SAMPLE_QUERIES = ["Manchester", "Oxford", "Liverpool", "Brighton and Hove", "Cambridge"]


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def diagnostics(x: np.ndarray, ctry: np.ndarray) -> tuple[float, float]:
    """(nearest-neighbor same-country rate, US-vs-UK 5-fold logistic accuracy)."""
    xn = l2(x)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -1.0)
    same = (ctry[sim.argmax(1)] == ctry).mean()
    y = (ctry == "US").astype(int)
    acc = cross_val_score(LogisticRegression(max_iter=5000), x, y, cv=5).mean()
    return same, acc


def neighbors(x: np.ndarray, ctry: np.ndarray, city: np.ndarray, q: str, k: int = 3) -> str:
    xn = l2(x)
    us = np.where(ctry == "US")[0]
    idx = np.where(city == q)[0]
    if not len(idx):
        return f"{q}: (not in set)"
    i = idx[0]
    sims = xn[us] @ xn[i]
    top = us[np.argsort(-sims)[:k]]
    return f"{q:18s} -> " + ", ".join(f"{city[j]} ({xn[i] @ xn[j]:.2f})" for j in top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    ap.add_argument("--n-pca", type=int, default=N_PCA)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"embeddings_{args.model}.parquet").reset_index(drop=True)
    x = np.vstack(df["embedding"].to_numpy()).astype("float64")  # float64 for LEACE stability
    ctry = df["country"].to_numpy()
    city = df["city"].to_numpy()

    pca = PCA(n_components=args.n_pca, svd_solver="full")
    xp = pca.fit_transform(x)
    print(
        f"PCA: {x.shape[1]} -> {args.n_pca} dims, "
        f"explained variance = {pca.explained_variance_ratio_.sum():.1%}\n"
    )

    reps: dict[str, np.ndarray] = {"raw_pca": xp}

    # centroid subtraction: remove each country's own mean (first-moment country offset)
    xc = xp.copy()
    for c in ("US", "UK"):
        m = ctry == c
        xc[m] = xp[m] - xp[m].mean(0)
    reps["centroid"] = xc

    # LEACE: erase the binary country concept (linear guardedness)
    import torch
    from concept_erasure import LeaceEraser

    eraser = LeaceEraser.fit(torch.from_numpy(xp), torch.from_numpy((ctry == "US").astype("int64")))
    reps["leace"] = eraser(torch.from_numpy(xp)).numpy()

    print(f"{'method':10s} {'NN same-country':>16s} {'US/UK separability':>20s}")
    for name, m in reps.items():
        same, acc = diagnostics(m, ctry)
        print(f"{name:10s} {same:>15.1%} {acc:>19.1%}")
    print()
    for name, m in reps.items():
        print(f"--- {name}: UK -> nearest US (plain cosine, no CSLS yet) ---")
        for q in SAMPLE_QUERIES:
            print("   " + neighbors(m, ctry, city, q))
        print()

    out = []
    for name, m in reps.items():
        sub = df[ID_COLS].copy()
        sub["method"] = name
        sub["embedding"] = [row.tolist() for row in m.astype("float32")]
        out.append(sub)
    out = pd.concat(out, ignore_index=True)
    write_df(out, PROCESSED / f"reps_{args.model}.parquet")
    print(f"Wrote reps_{args.model}.parquet: {len(out)} rows ({len(reps)}x{len(df)})")


if __name__ == "__main__":
    main()
