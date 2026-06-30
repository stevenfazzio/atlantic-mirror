"""Prototype the LLM character-distillation prompt on a handful of cities.

Validate, before spending 300 calls, that the profiles are (a) name/country-free,
(b) character-rich, and (c) not homogenized into generic filler. Prints each city's
original lead (truncated) next to its distilled profile for comparison.
"""

from __future__ import annotations

import anthropic
import pandas as pd

from _common import PROCESSED

MODEL = "claude-opus-4-8"
SAMPLE = [
    ("Oxford", "UK"),  # persistent miss in the embedding approach
    ("Lincoln", "UK"),  # namesake (Lincoln NE)
    ("Birmingham", "UK"),  # namesake (Birmingham AL)
    ("Brighton and Hove", "UK"),  # good case -- should stay good
    ("Blackpool", "UK"),  # resort
    ("Manchester", "UK"),  # major industrial
    ("Chicago", "US"),  # US side, name-strip check
    ("Boston", "US"),  # US side
]

SYSTEM = """You distill an encyclopedia city description into a CHARACTER PROFILE used to \
compare cities ACROSS DIFFERENT COUNTRIES by what kind of place they are.

Given the article lead, write 3-5 sentences capturing the city's transferable character:
- economic base and main industries (heavy industry, finance, tech, tourism, agriculture, \
port/logistics, university/research, government, etc.)
- geographic and physical setting (coastal, riverside, mountainous, flat, climate, natural features)
- size and regional role (dominant national metropolis, major regional commercial center, modest \
regional hub, satellite/commuter town, small historic town, resort, etc.)
- historical character and the era that shaped it (ancient/medieval origins, industrial-revolution \
boom, post-industrial decline or regeneration, planned/new town, rapid modern growth)
- cultural identity (university town, arts/music scene, sports culture, leisure/tourism, religious \
or administrative significance)

CRITICAL RULES:
- Do NOT mention the city's name, its country, its region/state/county, or any demonym.
- Do NOT use ANY proper nouns that identify the specific place -- no names of landmarks, rivers, \
universities, companies, sports teams, people, or neighbouring places. Refer to them generically \
("a major river", "an ancient university", "a famous seaside pleasure pier", "a historic cathedral").
- Capture what makes this city DISTINCTIVE; avoid generic filler that would fit any city.
- Write plainly and concretely. Output ONLY the profile text -- no preamble, headings, or quotes."""

client = anthropic.Anthropic()


def distill(lead: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": lead}],
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def main() -> None:
    cities = pd.read_parquet(PROCESSED / "cities.parquet")
    for name, country in SAMPLE:
        row = cities[(cities["city"] == name) & (cities["country"] == country)]
        if not len(row):
            print(f"=== {name} ({country}) === NOT FOUND\n")
            continue
        lead = row.iloc[0]["lead_text"]
        profile = distill(lead)
        print(f"=== {name} ({country}) ===")
        print(f"  lead:    {lead[:160].strip()}...")
        print(f"  profile: {profile}\n")


if __name__ == "__main__":
    main()
