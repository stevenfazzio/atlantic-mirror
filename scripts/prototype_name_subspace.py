"""Prototype: counterfactual name-subspace erasure.

Delta_i = embed(lead_i) - embed(masked_lead_i) is the name's *contextual* contribution to
the embedding. The top-k principal directions of {Delta_i} form an empirical "name subspace";
we project it out of every embedding, then run the usual PCA-50 + centroid pipeline.

Sweep k and report: the namesake effect (mean CSLS rank should climb toward chance ~76, and
in-top3 toward 0), the country residual (does name-erasure also dent the 94% same-country NN?),
and detailed matches at a couple of k so we can see whether good (control) matches survive.

Reads: data/processed/cities.parquet, data/processed/embeddings_nomic.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analyze_name_collisions import csls_matrix, find_namesakes, l2, rank_of
from prototype_name_fixes import (
    CONTROLS,
    DOC_PREFIX,
    MODEL_NAME,
    NAMESAKE_CITIES,
    build_reps,
    mask_lead,
    show,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from _common import PROCESSED

KS = [0, 1, 3, 5, 10, 20, 40]


def country_diag(reps: np.ndarray, country: np.ndarray) -> tuple[float, float]:
    x = l2(reps)
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)
    same = (country[sim.argmax(1)] == country).mean()
    acc = cross_val_score(
        LogisticRegression(max_iter=2000), reps, (country == "US").astype(int), cv=5
    ).mean()
    return float(same), float(acc)


def namesake_metrics(reps, country, names, pairs) -> tuple[float, int]:
    csls = csls_matrix(l2(reps[country == "UK"]), l2(reps[country == "US"]))
    ranks = [rank_of(csls[i], j) for i, j, _ in pairs]
    return float(np.mean(ranks)), int(sum(r <= 3 for r in ranks))


def main() -> None:
    cit = pd.read_parquet(PROCESSED / "cities.parquet").reset_index(drop=True)
    raw_by_qid = pd.read_parquet(PROCESSED / "embeddings_nomic.parquet").set_index("qid")[
        "embedding"
    ]
    base = l2(np.vstack([np.asarray(raw_by_qid[q]) for q in cit["qid"]]).astype("float64"))
    country = cit["country"].to_numpy()
    names = cit["city"].to_numpy()
    pairs = find_namesakes(list(names[country == "UK"]), list(names[country == "US"]))

    from sentence_transformers import SentenceTransformer

    print("loading model ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    masked = [mask_lead(t, n) for t, n in zip(cit["lead_text"], names)]
    masked_emb = l2(
        model.encode(
            [DOC_PREFIX + t for t in masked], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float64")
    )

    delta = base - masked_emb
    _, _, vt = np.linalg.svd(delta, full_matrices=False)  # rows of vt = name directions

    def erase(k: int) -> np.ndarray:
        if k == 0:
            return base
        vk = vt[:k].T
        return base - (base @ vk) @ vk.T

    print(
        f"\n{'k':>3s} {'name mean rank':>14s} {'in-top3':>8s} {'NN same-ctry':>13s} {'separability':>12s}"
    )
    print("    (chance ~76)              (baseline 94.7%)  (chance 50%)")
    for k in KS:
        reps = build_reps(erase(k), country)
        mr, t3 = namesake_metrics(reps, country, names, pairs)
        same, acc = country_diag(reps, country)
        print(f"{k:3d} {mr:14.1f} {t3:6d}/6 {same:12.1%} {acc:11.1%}")

    for k in (5, 20):
        show(
            f"k={k}",
            build_reps(erase(k), country),
            country,
            names,
            pairs,
            NAMESAKE_CITIES + CONTROLS,
        )


if __name__ == "__main__":
    main()
