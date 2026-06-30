# International Sibling Cities

Explaining cities in one country through *familiar* cities from another. The use case: a US
reader hovers over a UK city on a map and sees the US cities most like it.

The pipeline embeds a short description of each city, neutralizes the dominant "which
country" signal, and ranks cross-country analogs. The key move is **not** to embed the raw
Wikipedia text: an LLM first distills each lead into a name-free, country-free **character
profile** (industry, geography, scale, history, culture). Embedding those profiles instead
of the leads removes name collisions at the source (the UK and US "Birmingham" no longer
match on their shared name), focuses matching on city *character*, and softens the residual
country signal. Matching runs on the country-neutralized profile embeddings with CSLS (which
corrects for hub cities); each UK city gets its top-3 US analogs with cosine weights, and an
LLM writes a one-phrase caption per pair (grounded in the original leads) describing the
shared character.

## Pipeline

Numbered scripts under `scripts/`, each cached and idempotent — re-running skips work
already on disk. Outputs live under `data/` and `output/` (gitignored, regenerable).

| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 01 | `01_fetch_city_lists.py` | Wikidata SPARQL | `data/interim/city_lists.{parquet,csv}` |
| 02 | `02_fetch_leads.py` | stage 01 + MediaWiki API | `data/processed/cities.parquet` |
| 02b | `02b_distill.py` | stage 02 + Claude API | `data/processed/profiles_<key>.parquet` |
| 03 | `03_embed.py` | stage 02 or 02b | `data/processed/embeddings_<model>[_profile_<key>].parquet` |
| 04 | `04_neutralize_country.py` | stage 03 | `data/processed/reps_<model>[_profile_<key>].parquet` |
| 05 | `05_match.py` | stage 04 + cities | `data/processed/matches_<model>[_profile_<key>].json` |
| 07 | `07_caption.py` | stage 05 + cities + Claude API | `…_captioned.json` (matches + a caption per pair) |
| 06 | `06_map.py` | stage 05/07 + cities | `output/uk_map_<model>[_profile_<key>].html` |

Caption runs *after* matching and *before* the map: stage 07 augments stage 05's matches with a
shared-character phrase, and stage 06 renders the captioned file if present. Stages 03–07 take
`--source {lead,profile}` (and `--profile-key`, e.g. `haiku`/`opus`): the **profile** track is
primary; the **lead** track is kept as a control. Stage 02b takes `--model`/`--key` to distill
with different LLMs. Stage 04 builds three neutralized representations (`raw_pca`, `centroid`,
`leace`) with diagnostics; stage 05 reads one (default `centroid`).

Raw API responses are cached verbatim under `data/raw/` (SPARQL, leads, one file per distilled
profile, one per caption) so we never re-hit the network or re-pay an LLM call. Stages 02b and
07 need `ANTHROPIC_API_KEY`.

## Running

```sh
uv sync
uv run python scripts/01_fetch_city_lists.py
uv run python scripts/02_fetch_leads.py
uv run python scripts/02b_distill.py --model claude-haiku-4-5 --key haiku
# primary (profile) track:
uv run python scripts/03_embed.py --source profile
uv run python scripts/04_neutralize_country.py --source profile
uv run python scripts/05_match.py --source profile
uv run python scripts/07_caption.py --source profile
uv run python scripts/06_map.py --source profile
```

Drop `--source profile` from 03–07 for the lead-based control track. Add `--force` to ignore a
stage's cache.

View the map by serving the output dir (avoids `file://` issues):
`python3 -m http.server -d output`, then open `uk_map_nomic_profile_haiku.html`.

## Analysis

`eval_profiles.py` runs the head-to-head (lead vs profile-opus vs profile-haiku) on the
name-collision, country-residual, and face-validity metrics; `analyze_name_collisions.py`
quantifies the name-collision effect. The `prototype_*` scripts hold the explored-and-rejected
name-fix experiments (output surgery, masking, name-vector / subspace projection) and the
caption-prompt prototype, plus shared helpers used by `eval_profiles.py`.
