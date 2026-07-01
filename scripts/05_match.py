"""Stage 05: bidirectional matching -- each European city <-> its most similar US cities.

Both directions on the country-neutralized representation, ranked by CSLS (cross-domain similarity
local scaling, which suppresses hub cities that are everyone's nearest neighbor): each European
city -> top-n US analogs, AND each US city -> top-n European analogs. The CSLS pairwise score is
symmetric, so we compute one Europe x US matrix and read it both ways -- top per ROW for
Europe->US, top per COLUMN for US->Europe. Memberships differ per side (the embraced asymmetry:
Manchester's #1 US match need not have Manchester as its #1 European match). Pools are disjoint,
so no same-group exclusion is needed. Each match carries a cosine similarity and the other city's
qid (so stage 07 can caption it and the map can look it up).

--source lead | profile (+ --profile-key); --method centroid | leace | raw_pca.
Reads:  reps_<model>[_profile_<key>].parquet, cities.parquet, city_lists.parquet (country_name)
Writes: matches_<model>[_profile_<key>].json  (map-ready: keyed by qid, both groups)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import INTERIM, PROCESSED

MODEL_KEY = "nomic"
N_OUT = 3
CSLS_NBRS = 10
REPORT_EU = ["Manchester", "Lyon", "Munich", "Naples", "Rotterdam", "Barcelona", "Edinburgh", "Hamburg"]
REPORT_US = ["Pittsburgh", "Milwaukee", "New Orleans", "Las Vegas", "Boston", "Portland, Oregon"]


def l2(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def csls_matrix(src: np.ndarray, tgt: np.ndarray, kk: int) -> np.ndarray:
    """CSLS(src,tgt) = 2*cos - r_tgt - r_src on L2-normalized inputs; penalizes hub targets.
    Symmetric up to transpose, so the same matrix serves both directions."""
    cos = src @ tgt.T
    r_tgt = np.sort(cos, axis=0)[-kk:, :].mean(0)
    r_src = np.sort(cos, axis=1)[:, -kk:].mean(1)
    return 2 * cos - r_tgt[None, :] - r_src[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    ap.add_argument("--source", choices=["lead", "profile"], default="lead")
    ap.add_argument("--profile-key", default="haiku", help="distillation key for --source profile")
    ap.add_argument("--method", default="centroid", help="representation: centroid | leace | raw_pca")
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    reps = pd.read_parquet(PROCESSED / f"reps_{args.model}{suffix}.parquet")
    reps = reps[reps["method"] == args.method].reset_index(drop=True)
    assert len(reps), f"no rows for method={args.method!r}"
    coords = pd.read_parquet(PROCESSED / "cities.parquet")[["qid", "lat", "lon"]]
    names = pd.read_parquet(INTERIM / "city_lists.parquet")[["qid", "country_name"]]
    reps = reps.merge(coords, on="qid", how="left").merge(names, on="qid", how="left")

    eu = reps[reps["country"] == "Europe"].reset_index(drop=True)
    us = reps[reps["country"] == "US"].reset_index(drop=True)
    eu_n = l2(np.vstack(eu["embedding"].to_numpy()).astype("float64"))
    us_n = l2(np.vstack(us["embedding"].to_numpy()).astype("float64"))
    eu_name, eu_qid = eu["city"].to_numpy(), eu["qid"].to_numpy()
    us_name, us_qid = us["city"].to_numpy(), us["qid"].to_numpy()

    csls = csls_matrix(eu_n, us_n, CSLS_NBRS)  # eu x us; ranking (hubness-corrected)
    cos = eu_n @ us_n.T  # cosine for display weights (symmetric across directions)

    records: dict[str, dict] = {}

    def add(row: pd.Series, matches: list[dict]) -> None:
        records[str(row["qid"])] = {
            "city": row["city"],
            "group": row["country"],
            "country": None if pd.isna(row["country_name"]) else row["country_name"],
            "lat": None if pd.isna(row["lat"]) else float(row["lat"]),
            "lon": None if pd.isna(row["lon"]) else float(row["lon"]),
            "matches": matches,
        }

    for i in range(len(eu)):  # Europe -> US (top per row)
        top = np.argsort(-csls[i])[:N_OUT]
        add(eu.iloc[i], [
            {"qid": str(us_qid[j]), "city": us_name[j], "similarity": round(float(cos[i, j]), 3)}
            for j in top
        ])
    for j in range(len(us)):  # US -> Europe (top per column)
        top = np.argsort(-csls[:, j])[:N_OUT]
        add(us.iloc[j], [
            {"qid": str(eu_qid[i]), "city": eu_name[i], "similarity": round(float(cos[i, j]), 3)}
            for i in top
        ])

    print(f"{args.source}/{args.method}: {len(eu)} Europe x {len(us)} US, both directions\n")
    eu_ix = {eu_name[i]: i for i in range(len(eu))}
    us_ix = {us_name[j]: j for j in range(len(us))}
    print("Europe -> US:")
    for q in REPORT_EU:
        if q in eu_ix:
            i = eu_ix[q]
            top = np.argsort(-csls[i])[:N_OUT]
            print(f"  {q:18s} -> " + ", ".join(f"{us_name[j]} ({cos[i, j]:.2f})" for j in top))
    print("\nUS -> Europe:")
    for q in REPORT_US:
        if q in us_ix:
            j = us_ix[q]
            top = np.argsort(-csls[:, j])[:N_OUT]
            print(f"  {q:18s} -> " + ", ".join(f"{eu_name[i]} ({cos[i, j]:.2f})" for i in top))

    out_path = PROCESSED / f"matches_{args.model}{suffix}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    print(f"\nWrote {len(records)} cities (both groups) -> {out_path}")


if __name__ == "__main__":
    main()
