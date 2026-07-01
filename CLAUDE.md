# Atlantic Mirror — project guide

A two-way character mirror between North American and European cities (full overview in README.md).
Python data pipeline (`scripts/`, numbered stages) → bespoke D3 web map (**in progress**) → GitHub
Pages. Public repo `atlantic-mirror`; active work is on the `europe-pivot` branch.

## Running the pipeline
Stages run in order 01 → 02 → 02b → 03 → 04 → 05 → 07, each cached/idempotent (re-runs skip done
work; `--force` recomputes). Primary invocation uses `--source profile --profile-key haiku` for
03–07 and `--model claude-haiku-4-5 --key haiku` for 02b — exact commands in README.md § Running.
`ANTHROPIC_API_KEY` is required for 02b (distill) and 07 (caption). Product:
`data/processed/matches_nomic_profile_haiku_captioned.json`.

## Data safety — read before touching `data/`
`data/` and `output/` are gitignored. `data/raw/` caches cost real money and time to regenerate
(LLM distillations, captions, sitelink/SPARQL fetches). **Never delete or blow them away.** All data
writes are atomic — temp file → verify → rename (see `_common.write_df` / `cached_json`); follow that
pattern, never write-in-place. Caches are keyed by Wikidata QID, so a re-run only re-does genuinely
new cities. Don't pass `--force` without a reason.

## Data shapes
- `city_lists.parquet`: `country` = the GROUP (`"North America"` | `"Europe"`), `country_name` = the
  real country, plus `rank`, `city`, `population`, `wikipedia_title`, `qid`, `n_wikis`, `n_langs`.
  Stages 04/05 key on the `country` group; display uses `country_name`.
- **Output JSON** (`matches_..._captioned.json`, what the web map consumes) — keyed by QID:
  ```json
  "<qid>": {
    "city": "Munich", "group": "Europe", "country": "Germany", "lat": 48.13, "lon": 11.57,
    "matches": [{"qid": "...", "city": "Milwaukee", "similarity": 0.55, "caption": "Major industrial city with German heritage..."}, ...]
  }
  ```
  Both groups present; each city's `matches` are its top-3 in the *other* group. Bidirectional and
  **not necessarily mutual** (embraced asymmetry). Similarity is computed but **not displayed** on the map.

## Locked design — the web map (next task, not yet built)
Bespoke editorial **D3** (not Plotly). One **composite** map: North America (Albers/conic covering
US + Canada + Mexico — **not** `geoAlbersUsa`) + Europe (conic), **Atlantic cropped out**, under a
**single shared zoom/pan transform** (not two independent panels) — side-by-side on desktop, stacked
on mobile. **Nearest-city snap** (d3-quadtree) for tap/hover selection; **cross-highlight** the
matched cities in the other block; info as a **side panel (desktop hover) / bottom sheet (mobile
tap)**. **No arcs, no similarity numbers, no similarity color scale** — uniform dots. Emit a slim
static JSON from the matches file; static site → GitHub Pages, no runtime keys. Minimal tooling
(static + maybe Vite; no framework).

## Settled — don't re-explore
- Selection = prominence (Wikipedia sitelinks), not population. Scope = North America (US+CA+MX) ↔
  geographic Europe (44 states, transcontinental excluded), 250/side.
- Distillation supersedes every name-collision fix; Haiku ≈ Opus, so Haiku is primary. **Held off by
  the user:** Opus / full-Wikipedia-page / longer-profile distillation experiments.
- Dropped: 1:1 bijection, convex reconstruction, MMR, output-surgery / masking / name-subspace
  erasure (`scripts/prototype_*.py`).
