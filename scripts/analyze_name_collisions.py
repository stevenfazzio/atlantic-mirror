"""Measure the name-collision effect: do UK cities match their US namesakes above chance?

Detects namesake pairs (UK/US cities sharing a significant name word), then reports, for each
pair, where the namesake lands among the 150 US cities by: raw 768-d cosine, centroid-space
cosine, and CSLS rank. If names didn't matter, a namesake's expected rank is ~75/150 and its
expected similarity percentile ~50%.

Reads: data/processed/embeddings_<model>.parquet, data/processed/reps_<model>.parquet
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from _common import PROCESSED

MODEL_KEY = "qwen3"
STOP = {"city", "upon", "the", "and", "county", "borough", "district", "town", "royal"}


def core_words(name: str) -> set[str]:
    toks = re.split(r"[\s\-]+", name.lower())
    return {t for t in toks if len(t) >= 4 and t not in STOP}


def find_namesakes(uk_names, us_names) -> list[tuple[int, int, list[str]]]:
    uk_core = [core_words(n) for n in uk_names]
    us_core = [core_words(n) for n in us_names]
    pairs = []
    for i, uc in enumerate(uk_core):
        for j, sc in enumerate(us_core):
            shared = uc & sc
            if shared:
                pairs.append((i, j, sorted(shared)))
    return pairs


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def csls_matrix(uk: np.ndarray, us: np.ndarray, kk: int = 10) -> np.ndarray:
    cos = uk @ us.T
    r_us = np.sort(cos, axis=0)[-kk:, :].mean(0)
    r_uk = np.sort(cos, axis=1)[:, -kk:].mean(1)
    return 2 * cos - r_us[None, :] - r_uk[:, None]


def rank_of(scores: np.ndarray, j: int) -> int:
    return int((scores > scores[j]).sum()) + 1


def main() -> None:
    emb = pd.read_parquet(PROCESSED / f"embeddings_{MODEL_KEY}.parquet")
    reps = pd.read_parquet(PROCESSED / f"reps_{MODEL_KEY}.parquet")
    reps = reps[reps["method"] == "centroid"].reset_index(drop=True)

    raw_by_qid = emb.set_index("qid")["embedding"]
    us = reps[reps["country"] == "US"].reset_index(drop=True)
    uk = reps[reps["country"] == "UK"].reset_index(drop=True)

    def cen(sub):
        return l2(np.vstack(sub["embedding"].to_numpy()).astype("float64"))

    def raw(sub):
        return l2(np.vstack([np.asarray(raw_by_qid[q]) for q in sub["qid"]]).astype("float64"))

    cos_raw = raw(uk) @ raw(us).T
    cos_cen = cen(uk) @ cen(us).T
    csls = csls_matrix(cen(uk), cen(us))
    uk_names, us_names = uk["city"].to_numpy(), us["city"].to_numpy()
    n_us = len(us)

    pairs = find_namesakes(uk_names, us_names)
    n_uk_with = len({i for i, _, _ in pairs})
    print(f"{len(pairs)} namesake pairs across {n_uk_with} UK cities (of {len(uk)})\n")
    print(
        f"{'UK city':22s} {'US namesake':18s} {'shared':12s} {'cos':>5s} "
        f"{'rank_raw':>8s} {'rank_cen':>8s} {'rank_csls':>9s} {'pctile':>6s}"
    )

    rk_raw, rk_cen, rk_csls, pcts, top3 = [], [], [], [], 0
    for i, j, shared in pairs:
        rr, rc, rs = rank_of(cos_raw[i], j), rank_of(cos_cen[i], j), rank_of(csls[i], j)
        pct = (cos_cen[i] < cos_cen[i, j]).sum() / (n_us - 1) * 100
        rk_raw.append(rr)
        rk_cen.append(rc)
        rk_csls.append(rs)
        pcts.append(pct)
        top3 += rs <= 3
        print(
            f"{uk_names[i]:22s} {us_names[j]:18s} {','.join(shared):12s} {cos_cen[i, j]:5.2f} "
            f"{rr:8d} {rc:8d} {rs:9d} {pct:5.0f}%"
        )

    print(f"\naggregate over {len(pairs)} namesake pairs (vs chance):")
    print(
        f"  mean rank  raw={np.mean(rk_raw):.1f}  centroid={np.mean(rk_cen):.1f}  "
        f"csls={np.mean(rk_csls):.1f}   (chance ~{(n_us + 1) / 2:.0f})"
    )
    print(f"  median similarity percentile (centroid): {np.median(pcts):.0f}%   (chance 50%)")
    print(
        f"  namesake in CSLS top-3: {top3}/{len(pairs)} = {top3 / len(pairs):.0%}   "
        f"(chance {3 / n_us:.0%})"
    )


if __name__ == "__main__":
    main()
