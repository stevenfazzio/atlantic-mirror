"""Prototype: "police-lineup" caption metric -- score a label by how well it IDENTIFIES its cities.

For each displayed pair's shared label, build a lineup for each city (the true city + distractors
from the SAME group) and ask an LLM to score 0-100 how well the label fits each candidate, judging
ONLY from provided leads (so the metric measures the label, not the model's memory of famous cities).
The city's score = probability mass the label puts on the true city (1.0 = uniquely identifying,
~1/lineup = uninformative). A label's score = the WORSE city (min) -- and we also report harmonic
mean. High score => the label is specific, accurate, AND shared (a one-sided label fails the other
city; a bland label fails discrimination). Distractors are embedding-independent (rank-neighbours +
random) so the metric is fair across embedding/matching choices and non-circular.

Reads:  data/processed/<--captions>, data/processed/cities.parquet
Writes: data/raw/lineup_judgments/<label-key>/<eu>__<na>__<side>.json   cached per (label, city)
        data/processed/lineup_<label-key>.parquet                        analysis-ready
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import numpy as np
import pandas as pd

from _common import PROCESSED, RAW, cached_json, write_df

IDENTIFIER_MODEL = "claude-sonnet-5"
WORKERS = 8
K_RANK = 3  # distractors of similar prominence (embedding-independent, "same league")
K_RAND = 2  # uniformly-random distractors from the same group
LINEUP = 1 + K_RANK + K_RAND

SYSTEM = """You are given a LABEL -- a short phrase describing a city's character -- and several \
candidate cities, each with an encyclopedia lead. For EACH candidate, rate from 0 to 100 how well \
the LABEL describes that city, judging ONLY from its lead.

Be discriminating. Reserve high scores for cities the label fits genuinely and specifically; give low \
scores to cities it fits only in a generic way ("a city", "a place with some history") or not at all. \
A vague label that could describe many of the candidates should get similar middling scores for all \
of them; a sharp, specific label should single one out. Return one integer score per candidate, in \
the exact order presented."""

client = anthropic.Anthropic(max_retries=5)


SCHEMA = {  # API rejects minItems>1, so length is normalized in code instead
    "type": "object",
    "additionalProperties": False,
    "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
    "required": ["scores"],
}


def identify(label, leads, *, model, effort):
    body = "\n\n".join(f"{i + 1}. {ld}" for i, ld in enumerate(leads))
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"LABEL: {label}\n\nCANDIDATES:\n{body}"}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    scores = [max(0, min(100, int(s))) for s in json.loads(text)["scores"]]  # clamp to 0-100
    n = len(leads)
    return (scores + [0] * n)[:n]  # normalize to lineup length (pad short / truncate long)


def build_lineup(tq, meta, dctx):
    """Deterministic same-group lineup of size LINEUP. mode=rank: similar-prominence + random
    (easy, embedding-free). mode=nn: nearest neighbours by a fixed reference embedding (hard; forces
    the label to distinguish the true city from its look-alikes -- non-circular only when the matches
    being compared are identical, e.g. a caption A/B)."""
    rng = random.Random(int.from_bytes(hashlib.md5(tq.encode()).digest()[:8], "big"))
    if dctx["mode"] == "nn":
        pool = [q for q in dctx["g2q"][meta.loc[tq, "country"]] if q != tq and q in dctx["refvecs"]]
        v = dctx["refvecs"][tq]
        sims = np.array([dctx["refvecs"][q] @ v for q in pool])
        nbrs = [pool[i] for i in np.argsort(-sims)[: LINEUP - 1]]
    else:
        t = meta.loc[tq]
        pool = meta[(meta["country"] == t["country"]) & (meta.index != tq)]
        order = (pool["rank"] - t["rank"]).abs().to_numpy().argsort()[:K_RANK]
        rank_nbrs = list(pool.index[order])
        rest = [q for q in pool.index if q not in rank_nbrs]
        nbrs = rank_nbrs + rng.sample(rest, min(K_RAND, len(rest)))
    lineup = [tq] + nbrs
    rng.shuffle(lineup)  # true-city position varies but is deterministic
    return lineup


def score_city(tq, label, meta, leads, *, label_key, side, eu, na, model, effort, force, dctx):
    path = RAW / "lineup_judgments" / label_key / f"{eu}__{na}__{side}.json"

    def compute():
        lu = build_lineup(tq, meta, dctx)
        sc = identify(label, [leads[q] for q in lu], model=model, effort=effort)
        tpos = lu.index(tq)
        tot = sum(sc) or 1
        prob_mass = sc[tpos] / tot
        rank = 1 + sum(s > sc[tpos] for s in sc)  # 1 = best
        return {
            "prob_mass": prob_mass,
            "recip_rank": 1 / rank,
            "rank": rank,
            "lineup": lu,
            "scores": sc,
            "true_pos": tpos,
        }

    return cached_json(path, compute, force=force)


def norm_pair(q, grp, mq):
    return (q, mq) if grp == "Europe" else (mq, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="captioned matches json in data/processed")
    ap.add_argument(
        "--label-key", required=True, help="cache namespace + output tag (e.g. v3, v1, rawpca_v3)"
    )
    ap.add_argument("--sample", type=int, default=150, help="number of pairs (0 = all)")
    ap.add_argument("--identifier-model", default=IDENTIFIER_MODEL)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument(
        "--distractors",
        default="rank",
        choices=["rank", "nn"],
        help="rank=easy (prominence+random); nn=hard (nearest by --ref-reps, non-circular only for caption A/B)",
    )
    ap.add_argument(
        "--ref-reps",
        default="reps_qwen3_profile_haiku.parquet",
        help="reference reps for nn distractors",
    )
    ap.add_argument("--ref-method", default="centroid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    matches = json.loads((PROCESSED / args.captions).read_text())
    cdf = pd.read_parquet(PROCESSED / "cities.parquet").drop_duplicates("qid").set_index("qid")
    leads = cdf["lead_text"].to_dict()

    dctx = {"mode": args.distractors, "refvecs": None, "g2q": None}
    if args.distractors == "nn":
        reps = pd.read_parquet(PROCESSED / args.ref_reps)
        reps = reps[reps["method"] == args.ref_method]
        dctx["refvecs"] = {
            r.qid: (lambda a: a / (np.linalg.norm(a) + 1e-12))(
                np.asarray(r.embedding, dtype="float64")
            )
            for r in reps.itertuples(index=False)
        }
        dctx["g2q"] = cdf.groupby("country").groups
        dctx["g2q"] = {g: list(idx) for g, idx in dctx["g2q"].items()}

    pairs = {}
    for q, rec in matches.items():
        for m in rec["matches"]:
            pairs[norm_pair(q, rec["group"], m["qid"])] = m["caption"].strip()
    keys = sorted(pairs)
    if args.sample and args.sample < len(keys):
        keys = sorted(keys, key=lambda k: hashlib.md5(f"{k[0]}__{k[1]}".encode()).hexdigest())[
            : args.sample
        ]
        keys = sorted(keys)
    print(
        f"lineup eval: {len(keys)} pairs x2 cities, lineup={LINEUP}, chance={1 / LINEUP:.3f} "
        f"(model={args.identifier_model}, key={args.label_key})"
    )

    def work(eu, na):
        label = pairs[(eu, na)]
        e = score_city(
            eu,
            label,
            cdf,
            leads,
            label_key=args.label_key,
            side="eu",
            eu=eu,
            na=na,
            model=args.identifier_model,
            effort=args.effort,
            force=args.force,
            dctx=dctx,
        )
        n = score_city(
            na,
            label,
            cdf,
            leads,
            label_key=args.label_key,
            side="na",
            eu=eu,
            na=na,
            model=args.identifier_model,
            effort=args.effort,
            force=args.force,
            dctx=dctx,
        )
        return eu, na, label, e, n

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, eu, na) for eu, na in keys]
        for fut in as_completed(futs):
            eu, na, label, e, n = fut.result()
            pm_e, pm_n = e["prob_mass"], n["prob_mass"]
            rows.append(
                {
                    "eu_qid": eu,
                    "na_qid": na,
                    "eu_city": cdf.loc[eu, "city"],
                    "na_city": cdf.loc[na, "city"],
                    "caption": label,
                    "pm_eu": pm_e,
                    "pm_na": pm_n,
                    "rr_eu": e["recip_rank"],
                    "rr_na": n["recip_rank"],
                    "pair_min": min(pm_e, pm_n),
                    "pair_harm": 0.0 if pm_e + pm_n == 0 else 2 * pm_e * pm_n / (pm_e + pm_n),
                    "eu_rank": int(cdf.loc[eu, "rank"]),
                    "na_rank": int(cdf.loc[na, "rank"]),
                }
            )
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(keys)}")

    df = pd.DataFrame(rows)
    report(df, args.label_key)
    out = PROCESSED / f"lineup_{args.label_key}.parquet"
    write_df(df, out)
    print(f"Wrote {len(df)} -> {out.name}")


def report(df, key):
    chance = 1 / LINEUP
    pooled_pm = pd.concat([df["pm_eu"], df["pm_na"]])
    pooled_rank = pd.concat([df["eu_rank"], df["na_rank"]])
    print(f"\n########## lineup metric [{key}]  (chance prob-mass = {chance:.3f}) ##########")
    print(f"  pair score (min of 2 cities):      {df['pair_min'].mean():.3f}")
    print(f"  pair score (harmonic mean):        {df['pair_harm'].mean():.3f}")
    print(f"  per-city prob-mass (pooled):       {pooled_pm.mean():.3f}")
    print(
        f"  per-city top-1 identify rate:      {(pd.concat([df['rr_eu'], df['rr_na']]) == 1).mean():.1%}"
    )
    print(
        f"  fame confound corr(prob_mass, rank): {pooled_pm.corr(pooled_rank):+.3f}  "
        f"(rank: 1=most prominent; +ve => prominent cities score higher)"
    )
    lo = df.nsmallest(6, "pair_min")
    hi = df.nlargest(6, "pair_min")
    print("\n  lowest-scoring labels (bland/one-sided?):")
    for r in lo.itertuples(index=False):
        print(f"    [{r.pair_min:.2f}] {r.eu_city} <-> {r.na_city}: {r.caption}")
    print("\n  highest-scoring labels (specific+shared?):")
    for r in hi.itertuples(index=False):
        print(f"    [{r.pair_min:.2f}] {r.eu_city} <-> {r.na_city}: {r.caption}")


if __name__ == "__main__":
    main()
