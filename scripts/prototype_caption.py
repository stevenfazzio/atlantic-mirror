"""Prototype the pair-caption prompt on a handful of real matched pairs.

Captions are user-facing, so they're grounded in the original *leads* (named, concrete) -- not
the name-free profiles. Validate the prompt (specific? honest about weak pairs? not generic?)
before captioning all displayed pairs.
"""

from __future__ import annotations

import json

import anthropic
import pandas as pd

from _common import PROCESSED

MODEL = "claude-haiku-4-5"
MATCHES = "matches_nomic_profile_haiku.json"
SAMPLE = [
    "Manchester",
    "Oxford",
    "York",
    "Brighton and Hove",
    "Milton Keynes",
    "Liverpool",
    "Blackpool",
    "Edinburgh",
]

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
- Be concrete about the shared archetype (industry, geography, scale/role, historical arc, \
culture); avoid vague filler like "vibrant, diverse city".
- Use ONLY character that both leads genuinely share; do not invent.
- If they share little, give the most honest shared descriptor you can, even if broad.
- Output only the phrase -- no preamble."""

client = anthropic.Anthropic(max_retries=5)


def caption(uk_name: str, uk_lead: str, us_name: str, us_lead: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=160,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"British city: {uk_name}\n{uk_lead}\n\nAmerican city: {us_name}\n{us_lead}",
            }
        ],
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def main() -> None:
    matches = json.loads((PROCESSED / MATCHES).read_text())
    cities = pd.read_parquet(PROCESSED / "cities.parquet")
    us_lead = cities[cities["country"] == "US"].set_index("city")["lead_text"].to_dict()
    uk_lead = cities[cities["country"] == "UK"].set_index("qid")["lead_text"].to_dict()
    title_by_city = {v["city"]: t for t, v in matches.items()}

    for q in SAMPLE:
        if q not in title_by_city:
            continue
        rec = matches[title_by_city[q]]
        us = rec["matches"][0]["city"]
        cap = caption(q, uk_lead[rec["qid"]], us, us_lead.get(us, ""))
        print(f"=== {q} -> {us} ===\n  {cap}\n")


if __name__ == "__main__":
    main()
