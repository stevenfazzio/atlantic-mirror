"""Diagnostic: LLM-as-judge over the displayed (European, North American) caption pairs.

For each unique pair, an Opus judge reads the caption + BOTH cities' original leads and audits
how honestly the caption describes *shared* character. It decomposes the caption into atomic
claims and tags each `both / eu_only / na_only / neither`, then rates invention and specificity.
The judge is deliberately BLIND to the cosine similarity -- we join `similarity` back in during
analysis so we can tell whether vagueness is concentrated in genuinely-weak (low-sim) pairs, where
broad is fine, or in high-sim pairs, where it's inexcusable.

This is an offline diagnostic (like eval_profiles.py / analyze_name_collisions.py), NOT a pipeline
stage: it reports the failure-mode breakdown so we can decide caption model / prompt changes from
data. Per-pair judgments are cached (resumable) so re-runs are cheap and the harness is reusable
for future caption sets (model swaps, embedding-model experiments).

Reads:  data/processed/matches_<model>[_profile_<key>]_captioned.json, data/processed/cities.parquet
Writes: data/raw/caption_judgments/<eu_qid>__<na_qid>.json                 cached per pair
        data/processed/caption_judgments_<model>[_profile_<key>].parquet   flattened, analysis-ready
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from _common import PROCESSED, RAW, cached_json, write_df

JUDGE_MODEL = "claude-opus-4-8"
WORKERS = 8

SYSTEM = """You are auditing one-line captions for a two-way "sibling cities" map. Each caption is \
meant to describe the shared character of a European city and a North American city that an \
algorithm paired -- a SINGLE description that applies equally to BOTH cities. Captions are written \
from the two cities' encyclopedia leads.

You are given the caption and both cities' leads (marked EUROPEAN and NORTH AMERICAN). Judge how \
honestly the caption describes character the two cities genuinely SHARE.

Decompose the caption into its atomic descriptive claims -- each a distinct trait (an industry, an \
era like "medieval", a geographic feature, a scale/role, a cultural identity). For each claim, judge \
from the leads which city or cities it genuinely applies to:
- "both": clearly supported by BOTH leads
- "eu_only": true of the European city but not clearly the North American one
- "na_only": true of the North American city but not clearly the European one
- "neither": not clearly supported by either lead (invented or overstated)

Then judge:
- scale_role: does the caption assert a scale or regional role -- e.g. "hub", "anchors"/"anchoring a \
region", "major metropolis", "dominant center", "capital city"?
  - "none": no notable scale/role claim.
  - "consistent": the scale/role it asserts fits BOTH cities' actual scale as described in the leads.
  - "overclaimed": it inflates at least one city's scale/role beyond its lead -- e.g. calls a suburb \
or edge city a regional "hub", or a modest town a "major metropolis". Name which city in \
scale_role_note (otherwise "").
- invented: true if the caption asserts character not grounded in either lead; put the specifics in \
invented_note (otherwise "").
- specificity:
  - "concrete": names specific shared character (industry, era, geography, role) both leads support.
  - "vague_avoidable": stays generic (e.g. "vibrant regional hub undergoing transformation") when the \
leads actually share nameable concrete character it could have used.
  - "broad_appropriate": broad BECAUSE the two cities genuinely share little concrete character, so an \
honest caption can only be broad. Use this instead of penalizing honesty.
- note: one short sentence naming the main issue, or "clean" if there is none.

Judge ONLY from the leads. Do not use outside knowledge to rescue or condemn a claim. Base every \
judgment on what the leads actually say."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "applies_to": {
                        "type": "string",
                        "enum": ["both", "eu_only", "na_only", "neither"],
                    },
                },
                "required": ["text", "applies_to"],
            },
        },
        "scale_role": {
            "type": "string",
            "enum": ["none", "consistent", "overclaimed"],
        },
        "scale_role_note": {"type": "string"},
        "invented": {"type": "boolean"},
        "invented_note": {"type": "string"},
        "specificity": {
            "type": "string",
            "enum": ["concrete", "vague_avoidable", "broad_appropriate"],
        },
        "note": {"type": "string"},
    },
    "required": [
        "claims",
        "scale_role",
        "scale_role_note",
        "invented",
        "invented_note",
        "specificity",
        "note",
    ],
}

client = anthropic.Anthropic(max_retries=5)


def judge(caption: str, eu_lead: str, na_lead: str, *, model: str, effort: str) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    f"CAPTION:\n{caption}\n\n"
                    f"EUROPEAN city lead:\n{eu_lead}\n\n"
                    f"NORTH AMERICAN city lead:\n{na_lead}"
                ),
            }
        ],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def judge_pair(eu_qid, na_qid, caption, eu_lead, na_lead, *, model, effort, caption_key, force):
    base = RAW / "caption_judgments" / caption_key if caption_key else RAW / "caption_judgments"
    path = base / f"{eu_qid}__{na_qid}.json"
    verdict = cached_json(
        path,
        lambda: judge(caption, eu_lead, na_lead, model=model, effort=effort),
        force=force,
    )
    return (eu_qid, na_qid), verdict


def norm_pair(rec_qid: str, group: str, m_qid: str) -> tuple[str, str]:
    """Normalize a (city, match) pair to (european_qid, north_american_qid) regardless of direction."""
    return (rec_qid, m_qid) if group == "Europe" else (m_qid, rec_qid)


def flatten(eu_qid, na_qid, caption, similarity, cities, verdict) -> dict:
    claims = verdict["claims"]
    tally = {
        k: sum(c["applies_to"] == k for c in claims)
        for k in ("both", "eu_only", "na_only", "neither")
    }
    return {
        "eu_qid": eu_qid,
        "na_qid": na_qid,
        "eu_city": cities.get(eu_qid, ("?",))[0],
        "na_city": cities.get(na_qid, ("?",))[0],
        "similarity": similarity,
        "caption": caption,
        "word_count": len(caption.split()),
        "n_claims": len(claims),
        "n_both": tally["both"],
        "n_eu_only": tally["eu_only"],
        "n_na_only": tally["na_only"],
        "n_neither": tally["neither"],
        "one_sided": tally["eu_only"] + tally["na_only"] > 0,
        "unsupported": tally["neither"] > 0,
        "scale_role": verdict["scale_role"],
        "scale_overclaimed": verdict["scale_role"] == "overclaimed",
        "invented": verdict["invented"],
        "specificity": verdict["specificity"],
        "note": verdict["note"],
        "scale_role_note": verdict["scale_role_note"],
        "invented_note": verdict["invented_note"],
    }


def report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n########## caption judgment: {n} pairs ##########")
    print(f"  one-sided (>=1 eu-only/na-only claim): {df['one_sided'].mean():.1%}")
    print(f"  unsupported (>=1 'neither' claim):     {df['unsupported'].mean():.1%}")
    print(f"  scale/role overclaimed:                {df['scale_overclaimed'].mean():.1%}")
    print(f"  invented (judge flag):                 {df['invented'].mean():.1%}")
    eu_only, na_only = int(df["n_eu_only"].sum()), int(df["n_na_only"].sum())
    lean = eu_only - na_only
    print(
        f"  directional lean: {eu_only} eu-only vs {na_only} na-only claims "
        f"(net {lean:+d} toward {'Europe' if lean > 0 else 'North America'})"
    )

    print("\n  specificity:")
    for k, v in df["specificity"].value_counts().items():
        print(f"    {k:18} {v:4}  ({v / n:.1%})")

    print("\n  by similarity quartile (low sim = weak pair -> broad is OK):")
    q = pd.qcut(df["similarity"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    g = df.groupby(q, observed=True)
    print(f"    {'quartile':9} {'sim range':>15}  {'one-sided':>10}  {'vague_avoid':>12}")
    for name, sub in g:
        rng = f"{sub['similarity'].min():.2f}-{sub['similarity'].max():.2f}"
        vague = (sub["specificity"] == "vague_avoidable").mean()
        print(f"    {name:9} {rng:>15}  {sub['one_sided'].mean():>9.1%}  {vague:>11.1%}")

    worst = df[df["one_sided"]].sort_values(["n_eu_only", "n_na_only"], ascending=False).head(15)
    print(f"\n  worst one-sided captions (of {int(df['one_sided'].sum())}):")
    for r in worst.itertuples(index=False):
        print(
            f"    [{r.n_eu_only}eu/{r.n_na_only}na, sim {r.similarity:.2f}] "
            f"{r.eu_city} <-> {r.na_city}: {r.caption}"
        )
        print(f"        note: {r.note}")

    over = df[df["scale_overclaimed"]].head(10)
    print(f"\n  scale/role overclaims (of {int(df['scale_overclaimed'].sum())}):")
    for r in over.itertuples(index=False):
        print(f"    [sim {r.similarity:.2f}] {r.eu_city} <-> {r.na_city}: {r.caption}")
        print(f"        {r.scale_role_note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--source", choices=["lead", "profile"], default="profile")
    ap.add_argument("--profile-key", default="haiku")
    ap.add_argument(
        "--method",
        default="centroid",
        help="stage-05 method tag on the matches file (centroid = untagged)",
    )
    ap.add_argument(
        "--caption-key",
        default="",
        help="caption variant to judge (matches 07 --caption-key; baseline = empty)",
    )
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="judge only the first N pairs (0 = all); for a cheap dry run",
    )
    ap.add_argument("--force", action="store_true", help="re-judge even if cached")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    method_tag = "" if args.method == "centroid" else f"_{args.method}"
    cap = f"_{args.caption_key}" if args.caption_key else ""
    matches = json.loads(
        (PROCESSED / f"matches_{args.model}{suffix}{method_tag}_captioned{cap}.json").read_text()
    )
    cdf = pd.read_parquet(PROCESSED / "cities.parquet")
    lead = cdf.set_index("qid")["lead_text"].to_dict()
    cities = {r.qid: (r.city, r.country, r.country_name) for r in cdf.itertuples(index=False)}

    # Unique EU<->NA pairs, carrying the caption and (symmetric) cosine similarity.
    pairs: dict[tuple[str, str], tuple[str, float]] = {}
    for q, rec in matches.items():
        for m in rec["matches"]:
            pairs[norm_pair(q, rec["group"], m["qid"])] = (m["caption"], m["similarity"])
    keys = sorted(pairs)
    if args.limit:
        keys = keys[: args.limit]
    print(f"judging {len(keys)} pairs  (judge={args.judge_model}, effort={args.effort})")

    verdicts, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                judge_pair,
                euq,
                naq,
                pairs[(euq, naq)][0],
                lead[euq],
                lead[naq],
                model=args.judge_model,
                effort=args.effort,
                caption_key=args.caption_key,
                force=args.force,
            )
            for euq, naq in keys
        ]
        for fut in as_completed(futs):
            key, verdict = fut.result()
            verdicts[key] = verdict
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(keys)} judged")

    rows = [
        flatten(euq, naq, pairs[(euq, naq)][0], pairs[(euq, naq)][1], cities, verdicts[(euq, naq)])
        for euq, naq in keys
    ]
    df = pd.DataFrame(rows)
    report(df)

    out_path = PROCESSED / f"caption_judgments_{args.model}{suffix}{method_tag}{cap}.parquet"
    write_df(df, out_path)
    print(f"\nWrote {len(df)} judgments -> {out_path.name}")


if __name__ == "__main__":
    main()
