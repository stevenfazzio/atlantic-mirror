"""Prototype two ways to kill name-collision similarity, vs baseline.

  baseline   original lead embeddings
  masked     re-embed leads with each city's own name words replaced by a placeholder
  projected  remove each city's name-vector component from its original embedding

Each variant goes through the same reduction (PCA-50 + per-country centroid subtraction). We
report the namesake effect (mean CSLS rank should rise toward chance ~76) and the new top-3
US matches for the namesake cities (should turn character-driven), plus control cities to
confirm the good matches survive.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from analyze_name_collisions import csls_matrix, find_namesakes, l2, rank_of
from sklearn.decomposition import PCA

from _common import PROCESSED

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
STOP = {"city", "upon", "the", "and", "county", "borough", "district", "town", "royal"}
NAMESAKE_CITIES = ["Birmingham", "York", "Newport", "Worcester", "Lincoln", "Stockton-on-Tees"]
CONTROLS = ["Brighton and Hove", "Edinburgh", "Liverpool", "Cambridge"]


def mask_lead(text: str, name: str) -> str:
    targets = [name] + [
        w for w in re.split(r"[\s\-]+", name) if len(w) >= 4 and w.lower() not in STOP
    ]
    for t in sorted(set(targets), key=len, reverse=True):  # longest first
        text = re.sub(rf"\b{re.escape(t)}\b", "Anytown", text, flags=re.IGNORECASE)
    return text


def build_reps(emb: np.ndarray, country: np.ndarray) -> np.ndarray:
    """L2-normalize -> PCA(50) -> per-country centroid subtraction (matches stage 04)."""
    xp = PCA(n_components=50, svd_solver="full").fit_transform(l2(emb))
    reps = xp.copy()
    for c in ("US", "UK"):
        m = country == c
        reps[m] = xp[m] - xp[m].mean(0)
    return reps


def show(tag, reps, country, names, pairs, focus) -> None:
    us, uk = country == "US", country == "UK"
    usn = names[us]
    ukn = list(names[uk])
    UK, US = l2(reps[uk]), l2(reps[us])
    cos, csls = UK @ US.T, csls_matrix(l2(reps[uk]), l2(reps[us]))

    ranks, pcts, top3 = [], [], 0
    for i, j, _ in pairs:
        r = rank_of(csls[i], j)
        ranks.append(r)
        top3 += r <= 3
        pcts.append((cos[i] < cos[i, j]).sum() / (len(usn) - 1) * 100)
    print(
        f"\n[{tag}]  namesake mean CSLS rank={np.mean(ranks):5.1f}  "
        f"in-top3={top3}/{len(pairs)}  median pctile={np.median(pcts):3.0f}%"
    )
    for c in focus:
        if c in ukn:
            i = ukn.index(c)
            top = np.argsort(-csls[i])[:3]
            print(f"     {c:20s} -> {', '.join(usn[t] for t in top)}")


def main() -> None:
    cit = pd.read_parquet(PROCESSED / "cities.parquet").reset_index(drop=True)
    raw_by_qid = pd.read_parquet(PROCESSED / "embeddings_nomic.parquet").set_index("qid")[
        "embedding"
    ]
    base = np.vstack([np.asarray(raw_by_qid[q]) for q in cit["qid"]]).astype("float64")
    country = cit["country"].to_numpy()
    names = cit["city"].to_numpy()
    pairs = find_namesakes(list(names[country == "UK"]), list(names[country == "US"]))
    focus = NAMESAKE_CITIES + CONTROLS

    from sentence_transformers import SentenceTransformer

    print("loading model ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    masked_leads = [mask_lead(t, n) for t, n in zip(cit["lead_text"], names)]
    eg = cit[cit["city"] == "Birmingham"].index[0]
    print(f"masked-lead sample [Birmingham]: {masked_leads[eg][:130]!r}")

    masked_emb = model.encode(
        [DOC_PREFIX + t for t in masked_leads], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float64")
    name_emb = model.encode(
        [DOC_PREFIX + n for n in names], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float64")

    basen = l2(base)
    proj = basen - np.sum(basen * name_emb, axis=1, keepdims=True) * name_emb

    show("baseline ", build_reps(base, country), country, names, pairs, focus)
    show("masked   ", build_reps(masked_emb, country), country, names, pairs, focus)
    show("projected", build_reps(proj, country), country, names, pairs, focus)


if __name__ == "__main__":
    main()
