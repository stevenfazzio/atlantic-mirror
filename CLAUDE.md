# Atlantic Mirror — project guide

A two-way character mirror between North American and European cities (full overview in README.md).
Python data pipeline (`scripts/`, numbered stages) → bespoke D3 web map (`docs/`, built) → GitHub
Pages — **live** at https://stevenfazzio.github.io/atlantic-mirror/. Public repo `atlantic-mirror`;
the North America↔Europe pivot is merged (PR #1), so work on `main` now.

## Running the pipeline
Stages run in order 01 → 02 → 02b → 03 → 04 → 05 → 07 → 08, each cached/idempotent (re-runs skip done
work; `--force` recomputes). Primary invocation uses `--source profile --profile-key haiku` for
03–08 and `--model claude-haiku-4-5 --key haiku` for 02b — exact commands in README.md § Running.
**Shipped captions:** stage 07 runs on Sonnet 5 with the v3 prompt (`--caption-model claude-sonnet-5
--caption-effort medium --prompt v3 --caption-key sonnet5v3`), and stage 08 must get `--caption-key
sonnet5v3` to export them; 07/08 *defaults* (no key) still rebuild the original Haiku/v1 baseline, so
a re-run without those flags silently reverts the map to it.
`ANTHROPIC_API_KEY` is required for 02b (distill) and 07 (caption); 08 (export web JSON) needs no key.
Products: `data/processed/matches_nomic_profile_haiku_captioned_sonnet5v3.json` (full) and stage 08's
slim `docs/data/atlantic-mirror.json` (what the web map loads).

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
  **not necessarily mutual** (embraced asymmetry). `similarity` (cosine) stays in this file but the
  slim web JSON drops it — no raw similarity number or colour scale (those two Plotly encodings were
  distracting; a call about *those encodings*, not a ban on numbers/colour).
- **Slim web JSON** (`docs/data/atlantic-mirror.json`, stage 08, committed) — keyed by QID with
  `city, group, country, rank` (prominence → dot size), `wiki` (enwiki title → article link),
  `lat, lon`, and `matches: [{qid, caption}]` (each twin's name/coords/country resolved from the same
  table). Coordinates missing upstream are backfilled from Wikidata P625 (cached in `data/raw/`).

## Web map — `docs/` (built; the design we landed on)
Bespoke editorial **D3**, static → GitHub Pages, no runtime keys, no framework (d3 + topojson +
world-atlas vendored under `docs/vendor` & `docs/data`). **Two independent, framed map-cards** — North
America (Albers/conic over US+CA+MX; **not** `geoAlbersUsa`, which is US-only) and Europe (conic) —
side by side on desktop, **stacked on mobile**, each **independently pan/zoomable** (zoom out to 0.6×,
pan, per-card reset). No shared transform, no cropped-Atlantic seam: the fused "one interrupted map"
read as a *broken* map, so the deliberate call is two honest panels.
- **Dots sized by prominence** (`rank`). **No basemap labels** (country labels were too sparse in NA /
  too crowded in EU); only the selected city + its three twins get on-map labels.
- **Select**: hover (desktop, non-sticky preview) / tap (mobile); **click-to-pin** on desktop so you
  can move onto the card; **nearest-city snap** via a per-card d3-quadtree. → selected city vermilion,
  its three twins teal in the *other* card, an **arc** drawn to each across the gutter.
- **Info card**: desktop = a card that **tracks the selected dot** on its outer side (NA→left, EU→right,
  clamped on-screen so it never covers the arcs; a hover card is `pointer-events:none`, only a pinned
  one is interactive), with **Wikipedia links** for the city + twins. Mobile = a **peek sheet** in a
  reserved bottom strip (never covers the far map) with a **teaser** (the #1 twin's shared sentence) +
  a "See the other two" expander.
- Each caption is written to fit **both** the city and its twin; the card says so explicitly (readers
  were taking them as describing only the twin — the single most important non-obvious point).

> Scope note, so this doesn't get re-poisoned: "no arcs" was a shelved intra-Europe feature (the
> transatlantic arcs above are wanted); "no numbers / no colour scale" meant the specific Plotly cosine
> encodings, not a minimalist law. The **two-panel** layout is the chosen design, **not** a compromise
> to fix later — the user explicitly rejected fusing the continents into one map / re-adding the seam.
> A composite/inset projection is off the table unless the user revisits it.

## Settled — don't re-explore
- Selection = prominence (Wikipedia sitelinks), not population. Scope = North America (US+CA+MX) ↔
  geographic Europe (44 states, transcontinental excluded), 250/side. **Membership is geographic, not
  political:** off-continent territories are excluded like the transcontinental states — Hawaii and the
  Canary Islands dropped (Oceania / off Africa; `OFF_CONTINENT` in stage 01), Alaska and Iceland kept
  (genuinely on-continent — far but placed in situ, which is why no map insets are needed).
- Web map = **two independent panels** (see above): country labels dropped, hover = non-sticky preview
  + click-to-pin, mobile = reserved-strip peek with a teaser. Don't re-fuse the maps or re-add a seam.
- Distillation supersedes every name-collision fix; Haiku ≈ Opus, so Haiku is primary. **Held off by
  the user:** Opus / full-Wikipedia-page / longer-profile distillation experiments.
- **Captions (07) = Sonnet 5 + symmetry-enforcing v3 prompt (shipped 2026-07-01, commit `ac63eba`).**
  Haiku/v1 captions were one-sided (a concrete trait true of only one city of the pair) ~84% of the
  time → ~52% on Sonnet-5/v3, and shorter. `scripts/judge_captions.py` (Opus LLM-as-judge: per-claim
  both/eu-only/na-only tagging + scale/invention/specificity axes, blind to cosine) is the measurement
  rig, reusable for embedding-model experiments. The old 'call a capital "a capital city", never "a
  state capital"' rule was **intentionally dropped** in v3 — specific capital types are fine when
  accurate for both cities; don't re-add a blanket ban.
- Dropped: 1:1 bijection, convex reconstruction, MMR, output-surgery / masking / name-subspace
  erasure (`scripts/prototype_*.py`).
