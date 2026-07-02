"""Stage 02b: distill each city's Wikipedia lead into a name-free character profile (LLM).

Claude rewrites each lead into 3-5 sentences of transferable character, stripping the city
name, country, region, and identifying proper nouns -- attacking name collisions and country
signal at the source. Profiles are cached per city per distillation key (resumable); calls run
concurrently. Use --model/--key to compare distillation models (e.g. opus vs haiku).

Reads:  data/processed/cities.parquet
Writes: data/raw/profiles/<key>/<qid>.txt        cached per city
        data/processed/profiles_<key>.parquet    identity cols + profile_text
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd
from analyze_name_collisions import core_words

from _common import PROCESSED, RAW, write_df

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_KEY = "opus"
WORKERS = 8
ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]

SYSTEM_V1 = """You distill an encyclopedia city description into a CHARACTER PROFILE used to \
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
("a major river", "an ancient university", "a famous seaside pleasure pier", "a historic \
cathedral").
- Capture what makes this city DISTINCTIVE; avoid generic filler that would fit any city.
- Write plainly and concretely. Output ONLY the profile text -- no preamble, headings, or quotes."""

# "rich" distillation (2026-07-02 experiment): the frontier-root probe. Same name-free rules, but
# asks for FULLER, more specific character across every dimension the lead supports, to test whether
# thinly-distilled profiles (not matching itself) are what limits shared-character in the pairs.
SYSTEM_RICH = """You distill an encyclopedia city description into a rich CHARACTER PROFILE used to \
compare cities ACROSS DIFFERENT COUNTRIES by what kind of place they are.

Given the article lead, write 6-9 sentences capturing the city's transferable character in specific \
detail. Cover every dimension the lead supports:
- economic base and main industries -- be SPECIFIC (not "industry" but "heavy steel and automotive \
manufacturing", "deep-water container port and logistics", "biotech and research universities", \
"government administration", "beach tourism and conventions")
- geographic and physical setting (coastal / riverside / lakeside / mountainous / plains, climate, \
notable natural features, relationship to water and terrain)
- size and regional role (dominant national metropolis / major regional commercial center / modest \
regional hub / satellite or commuter town / small historic town / resort) and its scale relative to \
its region
- the eras that shaped it (founding era, growth or boom periods, decline or reinvention, planned vs \
organic growth)
- cultural identity (university/college town, arts/music/film scene, sports culture, religious or \
administrative significance, distinctive demographics or subcultures)

Extract as MUCH transferable, distinctive character as the lead genuinely supports -- err toward more \
specific detail, not less -- but NEVER invent anything not in the lead.

CRITICAL RULES:
- Do NOT mention the city's name, its country, its region/state/county, or any demonym.
- Do NOT use ANY proper nouns that identify the specific place -- no names of landmarks, rivers, \
universities, companies, sports teams, people, or neighbouring places. Refer to them generically \
("a major river", "an ancient university", "a famous seaside pleasure pier").
- Capture what makes this city DISTINCTIVE; avoid generic filler that would fit any city.
- Write plainly and concretely. Output ONLY the profile text -- no preamble, headings, or quotes."""

PROMPTS = {"v1": SYSTEM_V1, "rich": SYSTEM_RICH}

client = anthropic.Anthropic(max_retries=5)


def distill(lead: str, model: str, system: str, max_tokens: int = 512) -> str:
    # Opus supports the effort parameter (low = fast/cheap for rewriting); Haiku 4.5 does not.
    kwargs = {} if "haiku" in model else {"output_config": {"effort": "low"}}
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": lead}],
        **kwargs,
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def profile_for(
    qid: str, lead: str, model: str, key: str, *, system: str, max_tokens: int, force: bool
) -> tuple[str, str]:
    path = RAW / "profiles" / key / f"{qid}.txt"
    if path.exists() and not force:
        return qid, path.read_text()
    text = distill(lead, model, system, max_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return qid, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default=DEFAULT_KEY, help="label for cache dir + output filename")
    ap.add_argument(
        "--prompt",
        choices=list(PROMPTS),
        default="v1",
        help="distillation prompt (v1 = shipped 3-5 sentence; rich = fuller 6-9 sentence)",
    )
    ap.add_argument("--force", action="store_true", help="re-distill even if cached")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    system = PROMPTS[args.prompt]
    max_tokens = 900 if args.prompt == "rich" else 512
    cities = pd.read_parquet(PROCESSED / "cities.parquet")
    results, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                profile_for,
                r.qid,
                r.lead_text,
                args.model,
                args.key,
                system=system,
                max_tokens=max_tokens,
                force=args.force,
            )
            for r in cities.itertuples(index=False)
        ]
        for fut in as_completed(futs):
            qid, text = fut.result()
            results[qid] = text
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(cities)} profiles")

    out = cities[ID_COLS].copy()
    out["profile_text"] = out["qid"].map(results)
    out["profile_chars"] = out["profile_text"].str.len()
    assert out["profile_text"].notna().all(), "some profiles missing"

    leaks = [
        r.city
        for r in out.itertuples(index=False)
        if any(
            re.search(rf"\b{re.escape(w)}\b", r.profile_text.lower()) for w in core_words(r.city)
        )
    ]
    if leaks:
        print(f"  name-leak candidates ({len(leaks)}): {', '.join(leaks[:15])}")

    out_path = PROCESSED / f"profiles_{args.key}.parquet"
    write_df(out, out_path)
    print(
        f"Wrote {len(out)} profiles -> {out_path.name}  "
        f"(model={args.model}, median {int(out['profile_chars'].median())} chars)"
    )


if __name__ == "__main__":
    main()
