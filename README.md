# International Sibling Cities

Explaining cities in one country through *familiar* cities from another. The use case: a US
reader hovers over a UK city on a map and sees the US cities most like it.

The core trick is to neutralize the dominant "which country" signal in Wikipedia-lead
embeddings (per-country centroid subtraction; LEACE as a principled alternative), then rank
US cities for each UK city with CSLS (which corrects for hub cities) on that neutralized
space. Each UK city gets its top-n US analogs with cosine-similarity weights.

## Pipeline

Numbered scripts under `scripts/`, each cached and idempotent — re-running skips work
that's already on disk. All outputs live under `data/` (gitignored, regenerable).

| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 01 | `01_fetch_city_lists.py` | Wikidata SPARQL | `data/interim/city_lists.{parquet,csv}` |
| 02 | `02_fetch_leads.py` | stage 01 + MediaWiki API | `data/processed/cities.parquet` |
| 03 | `03_embed.py` | stage 02 | `data/processed/embeddings_<model>.parquet` |
| 04 | `04_neutralize_country.py` | stage 03 | `data/processed/reps_<model>.parquet` |
| 05 | `05_match.py` | stage 04 + stage 02 | `data/processed/matches_<model>.json` |
| 06 | `06_map.py` | stage 05 + stage 02 | `output/uk_map_<model>.html` |

Raw API responses are cached verbatim under `data/raw/` so we never re-hit the network
for something we've already fetched. Stage 04 builds three country-neutralized
representations (`raw_pca`, `centroid`, `leace`) and reports country-confound diagnostics;
stage 05 reads one of them (default `centroid`).

## Running

```sh
uv sync
uv run python scripts/01_fetch_city_lists.py
uv run python scripts/02_fetch_leads.py
uv run python scripts/03_embed.py
uv run python scripts/04_neutralize_country.py
uv run python scripts/05_match.py
uv run python scripts/06_map.py
```

View the map by serving the output dir (avoids `file://` issues):
`python3 -m http.server -d output` then open `http://127.0.0.1:8000/uk_map_nomic.html`.

Embeddings are written per-model (`embeddings_nomic.parquet`, …); swapping `MODEL_NAME` in
`03_embed.py` adds a new file without clobbering existing ones. Add `--force` to a stage to
ignore its cache.
