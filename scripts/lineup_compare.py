"""Compare two lineup runs, paired where the design allows it, with cluster-robust intervals.

    uv run python scripts/lineup_compare.py BASE NEW     # label keys; differences are NEW - BASE

Reads the per-judgment cache lineup_eval.py writes. Calls no API and writes nothing.

Two differences are reported, and they answer different questions:

  pooled       Every judgment counts once: the difference between the two runs' headline numbers.
               It moves when the label improves AND when a run sends its matches to cities that are
               easier or harder to identify, and on sampled runs it includes cities that landed in
               only one sample.
  city-paired  Every city judged in both runs counts once: its mean in NEW minus its mean in BASE.
               A city gets the same lineup in every run, and about half the prob-mass variance
               belongs to the city, not the label (the same city judged through different
               counterparts correlates at r = .47 between qwen3nn and qwen38bnn), so this holds the
               city fixed and asks only whether its labels got more identifying. For a matcher A/B
               this is the sharper question. On sampled runs it covers the cities in both samples;
               with --sample 0 in both runs it covers every city.

For a caption A/B (same pairs, new captions) the two agree closely. For a matcher A/B they can
differ: when they do, the gap is the cities each run favours, not the labels.

Intervals are 95% normal intervals from a two-way cluster-robust standard error (Cameron, Gelbach
& Miller 2011), clustered on each pair's European city and its North American city. That covers
the dependence the design creates: one city sits in several pairs, the two cities of a pair share
a label (their prob-mass correlates at about +.3 to +.45), and a pair scored in both runs is
paired automatically. On a null simulation over the real pair graph these intervals cover 93-94%.

Top-1 splits ties (lineup_eval.top1_credit), so it runs 1-2 points below numbers recorded before
2026-11-15, which counted a tie as a hit. Pair score is the min of the two cities' prob-mass.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from lineup_eval import top1_credit

from _common import RAW

BASE, NEW = -1, 1


def load(key: str) -> pd.DataFrame:
    """One row per cached judgment, restricted to pairs with both sides cached."""
    folder = RAW / "lineup_judgments" / key
    files = sorted(folder.glob("*.json"))
    if not files:
        raise SystemExit(f"no cached judgments under {folder}")
    rows = []
    for path in files:
        eu, na, side = path.stem.split("__")
        v = json.loads(path.read_text())
        city = eu if side == "eu" else na
        assert v["lineup"][v["true_pos"]] == city, f"{path.name}: true city is not {city}"
        rows.append(
            {
                "pair": f"{eu}__{na}",
                "eu": eu,
                "na": na,
                "side": side,
                "city": city,
                "prob_mass": v["prob_mass"],
                "top1": top1_credit(v["scores"], v["true_pos"]),
                "lineup": tuple(sorted(v["lineup"])),
            }
        )
    j = pd.DataFrame(rows)
    sides = j.groupby("pair")["side"].transform("nunique")
    if (sides < 2).any():
        n = j.loc[sides < 2, "pair"].nunique()
        print(f"  {key}: leaving out {n} pairs with only one side cached")
        j = j[sides == 2].reset_index(drop=True)
    mixed = int((j.groupby("city")["lineup"].nunique() > 1).sum())
    if mixed:
        print(
            f"  WARNING {key}: {mixed} cities have more than one lineup inside this run, so it "
            f"mixes --distractors or --ref-reps settings."
        )
    return j


def pair_scores(j: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, pair): the pair score, min of its two cities' prob-mass."""
    keys = ["arm", "pair", "eu", "na"]
    return j.groupby(keys, as_index=False)["prob_mass"].min().rename(columns={"prob_mass": "y"})


def city_weights(j: pd.DataFrame) -> np.ndarray:
    """Weights that make each city judged in both runs count once per run; other cities get 0."""
    in_both = j.groupby("city")["arm"].transform("nunique").to_numpy() == 2
    per_run = j.groupby(["city", "arm"])["pair"].transform("size").to_numpy()
    return np.where(in_both, 1 / per_run, 0.0)


def difference(d: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """NEW - BASE difference of weighted means, with a 95% two-way cluster-robust interval.

    The estimate is a weighted sum of residuals from each run's own mean, so its variance is the
    sum of squared per-cluster totals: by European city, plus by North American city, minus by pair
    (their intersection, which the first two both count)."""
    arm = d["arm"].to_numpy()
    est, resid = 0.0, np.zeros(len(d))
    for sign in (NEW, BASE):
        m = arm == sign
        share = w[m] / w[m].sum()
        mean = (share * y[m]).sum()
        est += sign * mean
        resid[m] = sign * share * (y[m] - mean)

    def cluster_var(col: str) -> float:
        return float((np.bincount(pd.factorize(d[col])[0], weights=resid) ** 2).sum())

    var = cluster_var("eu") + cluster_var("na") - cluster_var("pair")
    half = 1.96 * np.sqrt(max(var, 0.0))
    return est, est - half, est + half


def show(pct: bool, label: str, res: tuple[float, float, float], units: str) -> None:
    est, lo, hi = res
    if pct:
        body = f"{100 * est:+5.1f} pts  [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
    else:
        body = f"{est:+.3f}      [{lo:+.3f}, {hi:+.3f}]"
    print(f"    {label:12s} {body}   {units}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="label key of the baseline run")
    ap.add_argument("new", help="label key of the run being tested")
    args = ap.parse_args()

    print(f"lineup compare: BASE={args.base}  NEW={args.new}  (differences are NEW - BASE)\n")
    jb, jn = load(args.base).assign(arm=BASE), load(args.new).assign(arm=NEW)
    j = pd.concat([jb, jn], ignore_index=True)
    p = pair_scores(j)

    pairs_b, pairs_n = set(jb["pair"]), set(jn["pair"])
    cities_b, cities_n = set(jb["city"]), set(jn["city"])
    both = sorted(cities_b & cities_n)
    print(f"  pairs:   {len(pairs_b)} base, {len(pairs_n)} new, {len(pairs_b & pairs_n)} in both")
    print(f"  cities:  {len(cities_b)} base, {len(cities_n)} new, {len(both)} in both")
    lu_b = jb.groupby("city")["lineup"].first()
    lu_n = jn.groupby("city")["lineup"].first()
    differ = int((lu_b[both] != lu_n[both]).sum())
    if differ:
        print(
            f"  WARNING: {differ} of {len(both)} shared cities got a different lineup in the two "
            f"runs, so for them the label is not the only thing that changed. A handful is a "
            f"city-list edit between the runs and is harmless; many means the runs used different "
            f"--distractors or --ref-reps and should not be compared."
        )
    else:
        print(f"  lineups: identical for all {len(both)} cities judged in both runs")

    is_b, is_n = (j["arm"] == BASE).to_numpy(), (j["arm"] == NEW).to_numpy()
    pair_b, pair_n = p.loc[p["arm"] == BASE, "y"].mean(), p.loc[p["arm"] == NEW, "y"].mean()
    print(f"\n  {'':12s} {'base':>8s} {'new':>8s}")
    print(f"  {'top-1':12s} {j.loc[is_b, 'top1'].mean():8.1%} {j.loc[is_n, 'top1'].mean():8.1%}")
    print(
        f"  {'prob-mass':12s} {j.loc[is_b, 'prob_mass'].mean():8.3f} "
        f"{j.loc[is_n, 'prob_mass'].mean():8.3f}"
    )
    print(f"  {'pair score':12s} {pair_b:8.3f} {pair_n:8.3f}")

    ones, cw = np.ones(len(j)), city_weights(j)
    print("\n  difference NEW - BASE, 95% interval")
    for col, label in [("top1", "top-1"), ("prob_mass", "prob-mass")]:
        y = j[col].to_numpy()
        pct = col == "top1"
        print(f"  {label}")
        show(pct, "pooled", difference(j, y, ones), "every judgment counts once")
        if both:
            units = f"{len(both)} cities in both runs, each counts once"
            show(pct, "city-paired", difference(j, y, cw), units)
    print("  pair score")
    show(
        False, "pooled", difference(p, p["y"].to_numpy(), np.ones(len(p))), "every pair counts once"
    )

    one_run_only = len(cities_b ^ cities_n)
    if one_run_only:
        print(
            f"\n  {one_run_only} cities were judged in only one run. They move the pooled rows but "
            f"not the city-paired rows; score every pair (--sample 0) in both runs to remove them."
        )


if __name__ == "__main__":
    main()
