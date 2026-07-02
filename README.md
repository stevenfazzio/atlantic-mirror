# Atlantic Mirror

**Live → [stevenfazzio.github.io/atlantic-mirror](https://stevenfazzio.github.io/atlantic-mirror/)**

A two-way character mirror between **North America** and **Europe**. Hover a city on either shore
and see the cities most like it *in character* on the other: what's the North American Turin?
(Pittsburgh). Which European city is the Milwaukee of the old world? (Munich, Plzeň). What's
Edinburgh's counterpart? (Ottawa, a fellow capital-and-university town).

250 North American cities (US + Canada + Mexico) and 250 European cities, each chosen by **global
prominence** — how many of the world's Wikipedias cover it — and matched on **character, not name**.
The key move: an LLM first distills each city's Wikipedia lead into a name-free, country-free
**character profile** (industry, geography, scale, history, culture). Embedding those profiles
instead of the raw text removes name collisions at the source (the UK and US "Birmingham" no longer
match on their shared name), focuses matching on what kind of place a city *is*, and dampens the
"which continent" signal. Matching runs **both directions** on the group-neutralized profile
embeddings via CSLS (which suppresses hub cities that are everyone's nearest neighbor); each city
gets its top-3 analogs on the other continent, and an LLM writes a one-phrase caption per pair
capturing the shared character.

## Pipeline

Numbered scripts under `scripts/`, each cached and idempotent — re-running skips work already on
disk. Outputs live under `data/` and `output/` (gitignored, regenerable).

| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 01 | `01_fetch_city_lists.py` | Wikidata SPARQL + wbgetentities | `data/interim/city_lists.{parquet,csv}` |
| 02 | `02_fetch_leads.py` | stage 01 + MediaWiki API | `data/processed/cities.parquet` |
| 02b | `02b_distill.py` | stage 02 + Claude API | `data/processed/profiles_<key>.parquet` |
| 03 | `03_embed.py` | stage 02 or 02b | `data/processed/embeddings_<model>[_profile_<key>].parquet` |
| 04 | `04_neutralize_country.py` | stage 03 + city_lists | `data/processed/reps_<model>[_profile_<key>].parquet` |
| 05 | `05_match.py` | stage 04 + cities + city_lists | `data/processed/matches_<model>[_profile_<key>].json` |
| 07 | `07_caption.py` | stage 05 + cities + Claude API | `…_captioned.json` (matches + a caption per pair) |
| 08 | `08_export_web.py` | stage 07 + city_lists | `docs/data/atlantic-mirror.json` (slim web JSON) |
| — | web map (`docs/`) | stage 08 | bespoke D3 two-panel map → GitHub Pages |

**Selection (01)** ranks by prominence — presence across the ~40 largest human-curated Wikipedia
language editions (bot-farms like Cebuano/Waray excluded) — above a 100k population floor, top-250
per side. North America = US (the "city in the United States" type) + Canada + Mexico (city / town /
municipality types); Europe = the 44 geographic-European sovereign states (continent = Europe, minus
the transcontinental five: Russia, Turkey, Kazakhstan, Georgia, Cyprus). By the same geographic
principle, off-continent territories are dropped despite being politically US/Spanish — Hawaii
(Pacific) and the Canary Islands (off Africa); Alaska and Iceland are kept.

**Matching (04–05)** treats the two continents as two groups, subtracts each group's centroid to
neutralize the coarse "which continent" offset (≈ LEACE; stage 04 also reports residual same-country
clustering as a tripwire), then ranks *both* directions from one symmetric CSLS matrix. Stage 05's
default representation is `centroid`. Stages 03–07 take `--source {lead,profile}` + `--profile-key`;
the **profile** track is primary, the **lead** track is a control.

Raw API responses are cached verbatim under `data/raw/` (SPARQL, sitelinks, leads, one file per
distilled profile, one per caption) so we never re-hit the network or re-pay an LLM call. Stages 02b
and 07 need `ANTHROPIC_API_KEY`.

## Running

```sh
uv sync
export ANTHROPIC_API_KEY=...                                    # for 02b + 07
uv run python scripts/01_fetch_city_lists.py
uv run python scripts/02_fetch_leads.py
uv run python scripts/02b_distill.py            --model claude-haiku-4-5 --key haiku
uv run python scripts/03_embed.py               --source profile --profile-key haiku
uv run python scripts/04_neutralize_country.py  --source profile --profile-key haiku
uv run python scripts/05_match.py               --source profile --profile-key haiku
uv run python scripts/07_caption.py             --source profile --profile-key haiku \
    --caption-model claude-sonnet-5 --caption-effort medium --prompt v3 --caption-key sonnet5v3
uv run python scripts/08_export_web.py          --source profile --profile-key haiku --caption-key sonnet5v3
```

Products: `data/processed/matches_nomic_profile_haiku_captioned_sonnet5v3.json` (full — every city
keyed by Wikidata QID with its group, real country, coordinates, and top-3 captioned analogs) and
stage 08's slim `docs/data/atlantic-mirror.json`, which the web map loads. Add `--force` to any stage
to ignore its cache.

Captions (07) run on **Sonnet 5** with the symmetry-enforcing **v3** prompt (`--caption-key
sonnet5v3`); the stage *defaults* (`--caption-model claude-haiku-4-5 --prompt v1`, no key) reproduce
the original Haiku baseline, kept for comparison — so pass the flags above, plus `--caption-key
sonnet5v3` to stage 08, to rebuild the shipped map, or a re-run silently reverts to the Haiku
captions. `scripts/judge_captions.py` is an Opus LLM-as-judge scoring caption honesty (one-sided
claims, scale inflation) that drove the v1→v3 iteration.

## Web map

`docs/` is a static, dependency-free site (vendored d3 + topojson + world-atlas) served straight from
GitHub Pages — no build step, no runtime keys. It renders **two independent, framed map-cards** (North
America and Europe), each pan/zoomable on its own; hover (or tap) a city and its three character counterparts
light up on the opposite card, with an arc to each and a card of captions — one sentence per pair,
written to fit *both* cities. Regenerate its data with stage 08, then preview locally with
`python3 -m http.server -d docs` (open `http://127.0.0.1:8000`).

## Evaluation

Grading a task with **no ground truth** needed its own toolkit — offline diagnostics (not pipeline
stages), all cached/resumable:

- `judge_captions.py` — Opus LLM-as-judge scoring caption honesty (one-sided claims, scale inflation);
  drove the v1→v3 caption iteration.
- `judge_pairs.py` — blind, order-randomized **head-to-head** twin-quality judge (a judge sees one home
  city and two candidate twins and picks the better); the primary metric for embedding/matching choices.
- `lineup_eval.py` — a **referring-expression** metric: score a caption by whether it can re-identify
  its own cities out of a look-alike lineup (leads provided, so it measures the caption, not city fame).

What they showed: text embedding + distillation drive most of the twin quality; neutralization and CSLS
trade a little per-pair quality to keep the map diverse (un-hubbed); the embedding *model* itself isn't
the lever; and a city's "twin" is really one representative of a *cloud* of comparable analogs (hence
three per city). Ablation flags on stages 02b/03/05/07 (`--prompt`, `--model`, `--method`, `--rank`,
`--sample`) drive these studies; their defaults reproduce the shipped map.

## Method notes & dropped approaches

- **Prominence, not population.** City-proper population is administratively inconsistent across
  countries (French communes tiny; German/Ukrainian/US units large); Wikipedia-language-edition
  count is boundary-independent and better captures "cities worth showing."
- **Distillation is validated** head-to-head vs. lead embeddings (kills namesake collisions, fixes
  character misses), and **Haiku matched Opus** at ~5× less cost — so Haiku is primary.
- Explored and rejected (`scripts/prototype_*.py`, `eval_profiles.py`, `analyze_name_collisions.py`):
  1:1 bijection, convex "build-a-city-from-parts" reconstruction, MMR diversification, and every
  output-surgery / masking / name-subspace approach to name collisions — distillation supersedes them.

`scripts/06_map.py` is the retired Plotly UK map, kept for reference and superseded by the D3 web map.
