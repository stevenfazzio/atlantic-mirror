"""Stage 07: caption each displayed (European city, North American city) pair with a shared-character phrase.

Grounded in both cities' original *leads* (named, concrete -- not the name-free profiles), an LLM
writes one descriptive phrase capturing the shared archetype (the kind of place both are). The
caption is direction-agnostic, so each European<->North American pair is captioned ONCE (normalized
to European-first) and attached to the match in both directions. Cached per pair (resumable).

Reads:  data/processed/matches_<model>[_profile_<key>].json, data/processed/cities.parquet
Writes: data/raw/captions/<eu_qid>__<na_qid>.txt                             cached per pair
        data/processed/matches_<model>[_profile_<key>]_captioned.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from _common import PROCESSED, RAW

CAPTION_MODEL = "claude-haiku-4-5"
WORKERS = 8

SYSTEM_V1 = """You are labeling a pair of "sibling cities" -- a European city and a North American city \
an algorithm identified as cross-country analogs -- for a general audience.

You will be given both cities' encyclopedia leads. Write a single concise descriptive PHRASE (not a \
full sentence; ~10-20 words) capturing the shared character the two cities have -- the kind of place \
they both are. It must read as one description that applies equally to each.

Rules:
- Do NOT name either city, and do NOT start with "Both" or "Two".
- Write a noun phrase describing the shared type, e.g. "Port city that reinvented itself from a \
19th-century industrial hub into a globally influential music capital" or "Planned postwar new town \
built around modernist design and a major university".
- Use generic terms: call a capital "a capital city", never "a state capital"; avoid any country- \
or region-specific administrative labels.
- Be concrete about the shared archetype (industry, geography, scale/role, historical arc, culture); \
avoid vague filler like "vibrant, diverse city".
- Use ONLY character that both leads genuinely share; do not invent.
- If they share little, give the most honest shared descriptor you can, even if broad.
- Output only the phrase -- no preamble."""

# V2 targets the failure modes the caption judge measured on V1 (scripts/judge_captions.py):
# 83% of captions were one-sided (a concrete trait true of only one city), 28% inflated a suburb into
# a "hub", and captions leaned toward the more richly-described city (usually the European one). The
# fixes: enforce symmetry against the WEAKER lead, forbid scale inflation, and prefer an honestly
# broad caption over a specific one-sided one.
SYSTEM_V2 = """You are labeling a pair of "sibling cities" -- a European city and a North American city \
an algorithm identified as cross-country analogs -- for a general audience.

You are given both cities' encyclopedia leads. Write a single concise descriptive PHRASE (not a full \
sentence; 20 words MAXIMUM) capturing character the two cities GENUINELY SHARE -- the kind of place \
they both are. It must read as one description that is TRUE OF EACH city on its own.

Hard rules:
- Do NOT name either city, and do NOT start with "Both" or "Two". Write a noun phrase describing the \
shared type.
- SYMMETRY (most important): every trait you include must be clearly supported by BOTH leads. If a \
trait -- an era ("medieval", "ancient"), an industry, a landmark, a role, a scale -- appears in only \
one lead, you must NOT use it. Before finalizing, check each word against the WEAKER (less-detailed) \
lead; if that city does not clearly have the trait, cut it.
- SCALE HONESTY: describe each city at the scale its own lead supports. Do NOT upgrade a suburb, edge \
city, satellite, or small town into a "hub", "major", "metropolis", or "regional center" it is not. \
Use generic terms: call a capital "a capital city", never "a state capital".
- SHARE LITTLE? STAY BROAD AND HONEST: if the leads reveal little genuine common character, give the \
most honest shared descriptor you can, even if broad ("a mid-sized regional city", "a coastal city \
shaped by tourism"). A broad but accurate caption is BETTER than a specific one that fits only one \
city. Never manufacture shared richness by importing one city's specifics.
- Do NOT default to the more richly-described city; the phrase must hold for the less-documented one too.
- Be concrete ONLY about character both leads truly share (industry, geography, scale/role, historical \
arc, culture); avoid vague filler like "vibrant, diverse city".
- Output only the phrase -- no preamble."""

# V3 keeps V2 and targets the two residuals the judge found on V2/Sonnet-5: (1) a new "capital-region /
# capital-adjacent / regional capital / regional seat" hedge the model used to smuggle one city's
# capital status into a fake shared framing, and (2) near-zero-overlap pairs where it still forced a
# concrete trait instead of going honestly broad.
SYSTEM_V3 = """You are labeling a pair of "sibling cities" -- a European city and a North American city \
an algorithm identified as cross-country analogs -- for a general audience.

You are given both cities' encyclopedia leads. Write a single concise descriptive PHRASE (not a full \
sentence; 20 words MAXIMUM) capturing character the two cities GENUINELY SHARE -- the kind of place \
they both are. It must read as one description that is TRUE OF EACH city on its own.

Hard rules:
- Do NOT name either city, and do NOT start with "Both" or "Two". Write a noun phrase describing the \
shared type.
- SYMMETRY (most important): every trait you include must be clearly supported by BOTH leads. If a \
trait -- an era ("medieval", "ancient"), an industry, a landmark, a role, a scale -- appears in only \
one lead, you must NOT use it. Before finalizing, check each word against the WEAKER (less-detailed) \
lead; if that city does not clearly have the trait, cut it.
- SCALE HONESTY: describe each city at the scale its own lead supports. Do NOT upgrade a suburb, edge \
city, satellite, or small town into a "hub", "major", "metropolis", or "regional center" it is not.
- NO CAPITAL/ADMINISTRATIVE HEDGING: describe the shared type as a "capital", "seat of government", or \
"administrative center" ONLY if BOTH leads clearly give that role. If only one city is a capital or \
administrative seat, drop that framing entirely. NEVER use bridging hedges like "capital-region", \
"capital-adjacent", "capital-like", "regional capital", or "regional seat" to make one city's status \
sound shared.
- SHARE LITTLE? STAY BROAD AND HONEST: after cutting one-sided traits, if little concrete character \
remains, that is fine -- write a short, plain, generic caption ("a mid-sized city on a river", "a \
growing suburban city", "a coastal city shaped by tourism"). A broad but accurate caption is BETTER \
than a specific one that fits only one city. Do NOT fill the gap with "hub", "capital", or \
administrative language, and never manufacture shared richness by importing one city's specifics.
- Do NOT default to the more richly-described city; the phrase must hold for the less-documented one too.
- Be concrete ONLY about character both leads truly share (industry, geography, scale/role, historical \
arc, culture); avoid vague filler like "vibrant, diverse city".
- Output only the phrase -- no preamble."""

# V4 keeps V3's honesty guardrails but fixes the blandness they caused: the judge showed V3 honest
# (~52% one-sided) but the diversity + lineup metrics showed it collapsed to templates ("on a river"
# 12%->41%). V4 replaces V3's eager "go broad/generic" fallback with a push for the MOST SPECIFIC
# genuinely-shared trait and an explicit anti-template rule -- honest AND identifying.
SYSTEM_V4 = """You are labeling a pair of "sibling cities" -- a European city and a North American \
city an algorithm identified as cross-country analogs -- for a general audience.

You are given both cities' encyclopedia leads. Write a single concise descriptive PHRASE (not a full \
sentence; 20 words MAXIMUM) capturing character the two cities GENUINELY SHARE -- the kind of place \
they both are. It must read as one description that is TRUE OF EACH city on its own.

Hard rules (honesty -- never break these):
- Do NOT name either city, and do NOT start with "Both" or "Two". Write a noun phrase describing the \
shared type.
- SYMMETRY (most important): every trait you include must be clearly supported by BOTH leads. If a \
trait -- an era ("medieval"), an industry, a landmark, a role, a scale -- appears in only one lead, \
you must NOT use it. Check each word against the WEAKER (less-detailed) lead; if that city does not \
clearly have the trait, cut it.
- SCALE HONESTY: describe each city at the scale its own lead supports. Do NOT upgrade a suburb, edge \
city, satellite, or small town into a "hub", "major", "metropolis", or "regional center" it is not.
- NO CAPITAL/ADMINISTRATIVE HEDGING: use "capital", "seat of government", or "administrative center" \
ONLY if BOTH leads clearly give that role. Never use bridging hedges like "capital-region", \
"regional capital", or "regional seat" to make one city's status sound shared.
- Never manufacture or overstate a shared trait to seem specific. Honesty wins over vividness.

Be SPECIFIC (this is what makes a caption good, once it is honest):
- Lead with the MOST SPECIFIC character the two GENUINELY share -- a particular industry, a defining \
historical era, a distinctive geographic or physical feature, an unusual economic or cultural role. A \
precise, even non-obvious shared trait (as long as BOTH leads clearly support it) is far better than \
a generic one.
- AVOID TEMPLATES: do NOT fall back on generic descriptors -- "a city on a river", "a coastal city", \
"a suburb near a larger metropolis", "a historic city with culture" -- unless that truly is the \
single most distinctive thing the two share. Such phrases fit hundreds of cities; aim for a caption \
that fits THESE TWO and few others.
- Only if the two genuinely share nothing specific, write a short honest plain phrase -- but treat \
that as the rare exception, not the default.
- Do NOT default to the more richly-described city; the phrase must hold for the less-documented one \
too.

Output only the phrase -- no preamble."""

# V4.1 keeps V4's specificity push but fixes its one regression: the judge showed V4 doubled
# scale/role overclaiming (11%->21%), almost all of it invented ADMINISTRATIVE-RANK or SIZE claims
# ("national capital", "imperial capital", "administrative seat", "second-largest") reached for as
# distinctive hooks. V4.1 forbids rank/size as the hook and redirects specificity to KIND of place.
SYSTEM_V4_1 = """You are labeling a pair of "sibling cities" -- a European city and a North American \
city an algorithm identified as cross-country analogs -- for a general audience.

You are given both cities' encyclopedia leads. Write a single concise descriptive PHRASE (not a full \
sentence; 20 words MAXIMUM) capturing character the two cities GENUINELY SHARE -- the kind of place \
they both are. It must read as one description that is TRUE OF EACH city on its own.

Hard rules (honesty -- never break these):
- Do NOT name either city, and do NOT start with "Both" or "Two". Write a noun phrase describing the \
shared type.
- SYMMETRY (most important): every trait you include must be clearly supported by BOTH leads. If a \
trait -- an era ("medieval"), an industry, a landmark, a role, a scale -- appears in only one lead, \
you must NOT use it. Check each word against the WEAKER (less-detailed) lead; if that city does not \
clearly have the trait, cut it.
- SCALE & RANK HONESTY (captions go wrong here most): a city's ADMINISTRATIVE STATUS and SIZE RANK \
are the most error-prone, least transferable traits. Do NOT claim a shared "national/imperial/\
provincial/state capital", "seat of government", "administrative center", "largest/second-largest \
city", "major metropolis", or "hub" unless BOTH leads state that role plainly. Never upgrade or \
invent rank to sound distinctive (a county seat is not a "provincial capital"; a suburb is not an \
"administrative center"; a state capital is not a "national capital"). Never use bridging hedges \
("capital-region", "regional capital", "just outside a capital") to make one city's status sound shared.
- Never manufacture or overstate a shared trait to seem specific. Honesty wins over vividness.

Be SPECIFIC -- but from the right material:
- Your distinctive hook must be the KIND of place the two share -- a particular industry, a defining \
historical era, a distinctive geographic or physical feature, an unusual economic or cultural role -- \
NEVER administrative rank or size. A precise, even non-obvious shared trait of this kind (as long as \
BOTH leads clearly support it) is far better than a generic one.
- AVOID TEMPLATES: do NOT fall back on generic descriptors -- "a city on a river", "a coastal city", \
"a suburb near a larger metropolis", "a historic city with culture" -- unless that truly is the \
single most distinctive thing the two share. Such phrases fit hundreds of cities; aim for a caption \
that fits THESE TWO and few others.
- Only if the two genuinely share nothing specific, write a short honest plain phrase -- but treat \
that as the rare exception, not the default.
- Do NOT default to the more richly-described city; the phrase must hold for the less-documented one \
too.

Output only the phrase -- no preamble."""

PROMPTS = {
    "v1": SYSTEM_V1,
    "v2": SYSTEM_V2,
    "v3": SYSTEM_V3,
    "v4": SYSTEM_V4,
    "v4.1": SYSTEM_V4_1,
}

client = anthropic.Anthropic(max_retries=5)


def make_caption(eu_lead: str, na_lead: str, *, model: str, system: str, effort: str | None) -> str:
    # Haiku 4.5 has no effort/adaptive-thinking; stronger models get a scratchpad to verify each
    # trait against BOTH leads (the symmetry check V2 asks for) before committing to the phrase.
    if "haiku" in model:
        kwargs = {"max_tokens": 160}
    else:
        kwargs = {
            "max_tokens": 1200,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort or "medium"},
        }
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"European city:\n{eu_lead}\n\nNorth American city:\n{na_lead}",
            }
        ],
        **kwargs,
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def caption_pair(eu_qid, na_qid, eu_lead, na_lead, *, model, system, effort, caption_key, force):
    base = RAW / "captions" / caption_key if caption_key else RAW / "captions"
    path = base / f"{eu_qid}__{na_qid}.txt"
    if path.exists() and not force:
        return (eu_qid, na_qid), path.read_text()
    text = make_caption(eu_lead, na_lead, model=model, system=system, effort=effort)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return (eu_qid, na_qid), text


def norm_pair(rec_qid: str, group: str, m_qid: str) -> tuple[str, str]:
    """Normalize a (city, match) pair to (european_qid, north_american_qid) regardless of direction."""
    return (rec_qid, m_qid) if group == "Europe" else (m_qid, rec_qid)


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
    ap.add_argument("--caption-model", default=CAPTION_MODEL)
    ap.add_argument("--caption-effort", default=None, help="effort for non-haiku caption models")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="v1")
    ap.add_argument(
        "--caption-key",
        default="",
        help="label; namespaces cache dir + output file (baseline = empty)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="caption only N hash-sampled pairs (prunes matches to them); for cheap prompt eval",
    )
    ap.add_argument("--force", action="store_true", help="re-caption even if cached")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    method_tag = "" if args.method == "centroid" else f"_{args.method}"
    matches = json.loads((PROCESSED / f"matches_{args.model}{suffix}{method_tag}.json").read_text())
    lead = pd.read_parquet(PROCESSED / "cities.parquet").set_index("qid")["lead_text"].to_dict()

    all_pairs = {
        norm_pair(q, rec["group"], m["qid"]) for q, rec in matches.items() for m in rec["matches"]
    }
    if args.sample and args.sample < len(all_pairs):
        # same hash-sample as lineup_eval, so a sampled caption set lines up with the sampled eval
        keep = set(
            sorted(all_pairs, key=lambda k: hashlib.md5(f"{k[0]}__{k[1]}".encode()).hexdigest())[
                : args.sample
            ]
        )
        for q, rec in list(matches.items()):
            rec["matches"] = [
                m for m in rec["matches"] if norm_pair(q, rec["group"], m["qid"]) in keep
            ]
        matches = {q: rec for q, rec in matches.items() if rec["matches"]}
    pairs = sorted(
        {norm_pair(q, rec["group"], m["qid"]) for q, rec in matches.items() for m in rec["matches"]}
    )
    system = PROMPTS[args.prompt]
    print(
        f"{len(pairs)} unique European<->North American pairs to caption "
        f"(model={args.caption_model}, prompt={args.prompt}, key={args.caption_key or '(baseline)'})"
    )

    results, done = {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                caption_pair,
                euq,
                naq,
                lead[euq],
                lead[naq],
                model=args.caption_model,
                system=system,
                effort=args.caption_effort,
                caption_key=args.caption_key,
                force=args.force,
            )
            for euq, naq in pairs
        ]
        for fut in as_completed(futs):
            key, text = fut.result()
            results[key] = text
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(pairs)} captions")

    for q, rec in matches.items():
        for m in rec["matches"]:
            m["caption"] = results[norm_pair(q, rec["group"], m["qid"])]

    cap = f"_{args.caption_key}" if args.caption_key else ""
    out_path = PROCESSED / f"matches_{args.model}{suffix}{method_tag}_captioned{cap}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(matches, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    print(f"Wrote {len(pairs)} captions -> {out_path.name}")


if __name__ == "__main__":
    main()
