"""Stage 07: caption each displayed (UK city, US analog) pair with a shared-character phrase.

Grounded in both cities' original *leads* (named, concrete -- not the name-free profiles), an
LLM writes one descriptive phrase capturing the shared archetype (the kind of place both are).
Cached per pair (resumable); calls run concurrently. Augments the matches JSON with a `caption`
on each match.

Reads:  data/processed/matches_<model>[_profile_<key>].json, data/processed/cities.parquet
Writes: data/raw/captions/<uk_qid>__<us_qid>.txt                              cached per pair
        data/processed/matches_<model>[_profile_<key>]_captioned.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from _common import PROCESSED, RAW

CAPTION_MODEL = "claude-haiku-4-5"
WORKERS = 8

SYSTEM = """You are labeling a pair of "sibling cities" -- a British city and an American city \
an algorithm identified as cross-country analogs -- for a general audience.

You will be given both cities' encyclopedia leads. Write a single concise descriptive PHRASE \
(not a full sentence; ~10-20 words) capturing the shared character the two cities have -- the \
kind of place they both are. It must read as one description that applies equally to each.

Rules:
- Do NOT name either city, and do NOT start with "Both" or "Two".
- Write a noun phrase describing the shared type, e.g. "Port city that reinvented itself from a \
19th-century industrial hub into a globally influential music capital" or "Planned postwar new \
town built around modernist design and a major university".
- Use generic terms: call a capital "a capital city", never "a state capital"; avoid any \
country- or region-specific administrative labels.
- Be concrete about the shared archetype (industry, geography, scale/role, historical arc, \
culture); avoid vague filler like "vibrant, diverse city".
- Use ONLY character that both leads genuinely share; do not invent.
- If they share little, give the most honest shared descriptor you can, even if broad.
- Output only the phrase -- no preamble."""

client = anthropic.Anthropic(max_retries=5)


def make_caption(uk_lead: str, us_lead: str) -> str:
    resp = client.messages.create(
        model=CAPTION_MODEL,
        max_tokens=160,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": f"British city:\n{uk_lead}\n\nAmerican city:\n{us_lead}"}
        ],
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def caption_pair(uk_qid, us_qid, uk_lead, us_lead, *, force):
    path = RAW / "captions" / f"{uk_qid}__{us_qid}.txt"
    if path.exists() and not force:
        return (uk_qid, us_qid), path.read_text()
    text = make_caption(uk_lead, us_lead)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return (uk_qid, us_qid), text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic")
    ap.add_argument("--source", choices=["lead", "profile"], default="profile")
    ap.add_argument("--profile-key", default="haiku")
    ap.add_argument("--force", action="store_true", help="re-caption even if cached")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    matches = json.loads((PROCESSED / f"matches_{args.model}{suffix}.json").read_text())
    lead = pd.read_parquet(PROCESSED / "cities.parquet").set_index("qid")["lead_text"].to_dict()

    pairs = [(rec["qid"], m["qid"]) for rec in matches.values() for m in rec["matches"]]

    results, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(caption_pair, ukq, usq, lead[ukq], lead[usq], force=args.force)
            for ukq, usq in pairs
        ]
        for fut in as_completed(futs):
            key, text = fut.result()
            results[key] = text
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(pairs)} captions")

    for rec in matches.values():
        for m in rec["matches"]:
            m["caption"] = results[(rec["qid"], m["qid"])]

    out_path = PROCESSED / f"matches_{args.model}{suffix}_captioned.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(matches, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    print(f"Wrote {len(pairs)} captions -> {out_path.name}")


if __name__ == "__main__":
    main()
