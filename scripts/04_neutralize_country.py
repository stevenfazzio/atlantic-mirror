"""Stage 04: reduce to a shared PCA space and neutralize the country signal.

Builds three representations in a shared ~50-dim PCA space and compares them:
  raw_pca   PCA only (control)
  centroid  PCA + per-GROUP mean subtraction (removes the first-moment US/Europe offset)
  leace     PCA + LEACE (linear concept erasure: no linear classifier can recover the group)

Two groups only -- "US" and "Europe" (Europe pooled, no per-country structure). For each rep we
report: nearest-neighbor same-GROUP rate, US-vs-Europe linear separability, and an "EU same-country
NN" tripwire -- among European cities, how often the nearest European neighbor shares the real
country (high => pooling Europe still leaves strong per-nationality clustering). Plus a qualitative
Europe->US neighbor preview. CSLS / hubness correction is left for stage 05; here we use plain cosine.

--source lead | profile (+ --profile-key) selects which embeddings to neutralize.
Reads:  data/processed/embeddings_<model>[_profile_<key>].parquet  (+ city_lists for country_name)
Writes: data/processed/reps_<model>[_profile_<key>].parquet  (long: identity + method + embedding)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from _common import INTERIM, PROCESSED, write_df

MODEL_KEY = "nomic"
N_PCA = 50
ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]
SAMPLE_QUERIES = ["Manchester", "Lyon", "Munich", "Naples", "Rotterdam"]


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def diagnostics(x: np.ndarray, ctry: np.ndarray) -> tuple[float, float]:
    """(nearest-neighbor same-group rate, US-vs-Europe 5-fold logistic accuracy)."""
    xn = l2(x)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -1.0)
    same = (ctry[sim.argmax(1)] == ctry).mean()
    y = (ctry == "US").astype(int)
    acc = cross_val_score(LogisticRegression(max_iter=5000), x, y, cv=5).mean()
    return same, acc


def nationality_nn(x: np.ndarray, ctry: np.ndarray, cname: np.ndarray) -> float:
    """Tripwire for residual nationality signal: among European cities, the fraction whose nearest
    European neighbor shares the real country. High => Europe-as-one-group still clusters by
    nationality (character crosses borders less than we'd like). Diluted by single-city countries."""
    eu = np.where(ctry == "Europe")[0]
    xe = l2(x[eu])
    sim = xe @ xe.T
    np.fill_diagonal(sim, -1.0)
    return (cname[eu][sim.argmax(1)] == cname[eu]).mean()


def neighbors(x: np.ndarray, ctry: np.ndarray, city: np.ndarray, q: str, k: int = 3) -> str:
    xn = l2(x)
    us = np.where(ctry == "US")[0]
    idx = np.where((city == q) & (ctry == "Europe"))[0]  # the European namesake, not the US one
    if not len(idx):
        return f"{q}: (not in set)"
    i = idx[0]
    sims = xn[us] @ xn[i]
    top = us[np.argsort(-sims)[:k]]
    return f"{q:18s} -> " + ", ".join(f"{city[j]} ({xn[i] @ xn[j]:.2f})" for j in top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    ap.add_argument(
        "--source",
        choices=["lead", "profile"],
        default="lead",
        help="lead embeddings or LLM character profiles",
    )
    ap.add_argument("--profile-key", default="haiku", help="distillation key for --source profile")
    ap.add_argument("--n-pca", type=int, default=N_PCA)
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    emb_path = PROCESSED / f"embeddings_{args.model}{suffix}.parquet"
    df = pd.read_parquet(emb_path).reset_index(drop=True)
    x = np.vstack(df["embedding"].to_numpy()).astype("float64")  # float64 for LEACE stability
    ctry = df["country"].to_numpy()
    city = df["city"].to_numpy()

    meta = pd.read_parquet(INTERIM / "city_lists.parquet")[["qid", "country_name"]]
    name_map = dict(zip(meta["qid"], meta["country_name"]))
    cname = np.array([name_map.get(q, "?") for q in df["qid"]])

    pca = PCA(n_components=args.n_pca, svd_solver="full")
    xp = pca.fit_transform(x)
    print(
        f"{emb_path.name}: PCA {x.shape[1]} -> {args.n_pca} dims, "
        f"explained variance = {pca.explained_variance_ratio_.sum():.1%}\n"
    )

    reps: dict[str, np.ndarray] = {"raw_pca": xp}

    # centroid subtraction: remove each group's own mean (first-moment US/Europe offset)
    xc = xp.copy()
    for c in ("US", "Europe"):
        m = ctry == c
        xc[m] = xp[m] - xp[m].mean(0)
    reps["centroid"] = xc

    # LEACE: erase the binary US-vs-Europe concept (linear guardedness)
    import torch
    from concept_erasure import LeaceEraser

    eraser = LeaceEraser.fit(torch.from_numpy(xp), torch.from_numpy((ctry == "US").astype("int64")))
    reps["leace"] = eraser(torch.from_numpy(xp)).numpy()

    print(f"{'method':10s} {'NN same-group':>14s} {'US/Eur sep':>12s} {'EU same-country NN':>20s}")
    for name, m in reps.items():
        same, acc = diagnostics(m, ctry)
        natl = nationality_nn(m, ctry, cname)
        print(f"{name:10s} {same:>13.1%} {acc:>11.1%} {natl:>19.1%}")
    print()
    for name, m in reps.items():
        print(f"--- {name}: Europe -> nearest US (plain cosine, no CSLS yet) ---")
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
    reps_path = PROCESSED / f"reps_{args.model}{suffix}.parquet"
    write_df(out, reps_path)
    print(f"Wrote {reps_path.name}: {len(out)} rows ({len(reps)}x{len(df)})")


if __name__ == "__main__":
    main()
