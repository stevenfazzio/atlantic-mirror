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
  `lat, lon`, and `matches: [{qid, caption}]` (each counterpart's name/coords/country resolved from the same
  table). Coordinates missing upstream are backfilled from Wikidata P625 (cached in `data/raw/`).

## Web map — `docs/` (built; the design we landed on)
Bespoke editorial **D3**, static → GitHub Pages, no runtime keys, no framework (d3 + topojson +
world-atlas vendored under `docs/vendor` & `docs/data`). **Two independent, framed map-cards** — North
America (Albers/conic over US+CA+MX; **not** `geoAlbersUsa`, which is US-only) and Europe (conic) —
side by side on desktop, **stacked on mobile**, each **independently pan/zoomable** (zoom out to 0.6×,
pan, per-card reset). No shared transform, no cropped-Atlantic seam: the fused "one interrupted map"
read as a *broken* map, so the deliberate call is two honest panels.
- **Dots sized by prominence** (`rank`). **No basemap labels** (country labels were too sparse in NA /
  too crowded in EU); only the selected city + its three counterparts get on-map labels.
- **Select**: hover (desktop, non-sticky preview) / tap (mobile); **click-to-pin** on desktop so you
  can move onto the card; **nearest-city snap** via a per-card d3-quadtree. → selected city vermilion,
  its three counterparts teal in the *other* card, an **arc** drawn to each across the gutter.
- **Info card**: desktop = a card that **tracks the selected dot** on its outer side (NA→left, EU→right,
  clamped on-screen so it never covers the arcs; a hover card is `pointer-events:none`, only a pinned
  one is interactive), with **Wikipedia links** for the city + counterparts. Mobile = a **peek sheet** in a
  reserved bottom strip (never covers the far map) with a **teaser** (the #1 counterpart's shared sentence) +
  a "See the other two" expander.
- Each caption is written to fit **both** the city and its counterpart; the card says so explicitly (readers
  were taking them as describing only the counterpart — the single most important non-obvious point).
- **On-card term is "counterpart," not "twin"** (softened 2026-07-02). "Twin" overclaimed on two axes —
  uniqueness (three are shown) and closeness — against the **cloud-not-soulmate** finding, and a skeptic
  reading a loose pairing as a "twin" undercut perceived quality. Don't revert UI copy to "twin." (The
  sister-cities/town-twinning conceit stays; "twin" survives only as internal shorthand + the
  `judge_pairs` "twin-quality" metric name.)
- **City search** (added 2026-07-02): floating top-centre pill (collapses to a magnifier on mobile)
  over a diacritic-folded index of all 500 cities; results disambiguate by **country + continent**
  (e.g. Vancouver BC vs Vancouver WA). Choosing pins like a click, then **recenters that panel** so the
  dot clears the info card (`focusCity`: scale 1.7, biased toward the gutter since the card sits on the
  outer side; edge cities clamped to d3-zoom's own pan bounds via `clampTransform`) and **resets the
  opposite panel to fit** so all three counterparts + arcs stay visible. Because of the pill, the
  **panel titles moved to bottom-left** (don't move them back to the top). With four ways to pick a city
  now (hover/click/tap/search), UI copy is gesture-agnostic — **"Choose a city…"** (dek, hint, meta);
  the hint no longer branches on hover-capability.
- **Deploy cache-busting:** GitHub Pages serves `style.css`/`main.js` with `max-age=600`, and there's no
  build step to hash filenames — so `index.html` loads them with a `?v=<datetimestamp>` query. **Bump
  that `?v=` on both links whenever you change style.css or main.js**, or returning visitors get a stale
  JS/CSS against a fresh index.html (search unstyled, titles wrong, JS dead — happened 2026-07-02).

> Scope note, so this doesn't get re-poisoned: "no arcs" was a shelved intra-Europe feature (the
> transatlantic arcs above are wanted); "no numbers / no colour scale" meant the specific Plotly cosine
> encodings, not a minimalist law. The **two-panel** layout is the chosen design, **not** a compromise
> to fix later — the user explicitly rejected fusing the continents into one map / re-adding the seam.
> A composite/inset projection is off the table unless the user revisits it.

## Evaluation tooling (offline diagnostics, not pipeline stages)
Built to grade a no-ground-truth task; all cached/resumable, none change the shipped pipeline.
- `judge_captions.py` — Opus LLM-as-judge on caption honesty (per-claim one-sided / scale / invention).
- `judge_pairs.py` — blind, order-randomized head-to-head twin-quality judge (Sonnet + Opus spot-check),
  caption-free; the main metric for embedding/matching/method choices. `--config LABEL=FILE` (≥2).
- `lineup_eval.py` — "police-lineup" identifiability metric (feed leads → no fame confound; `--distractors
  rank|nn`) + caption-free target-diversity; per-label specificity signal.
- **Additive experiment flags (defaults reproduce the shipped map):** 03 `--model {nomic,bge,qwen3}`;
  05 `--method {centroid,leace,raw_pca}` + `--rank {csls,cosine}`; 07 `--method`/`--sample`/`--prompt
  {v1..v4.1}`; 02b `--prompt {v1,rich}`. Method/rank/caption-key tags keep ablation outputs from
  colliding with the shipped files.

## Settled — don't re-explore
- Selection = prominence (Wikipedia sitelinks), not population. Scope = North America (US+CA+MX) ↔
  geographic Europe (44 states, transcontinental excluded), 250/side. **Membership is geographic, not
  political:** off-continent territories are excluded like the transcontinental states — Hawaii and the
  Canary Islands dropped (Oceania / off Africa; `OFF_CONTINENT` in stage 01), Alaska and Iceland kept
  (genuinely on-continent — far but placed in situ, which is why no map insets are needed).
- Web map = **two independent panels** (see above): country labels dropped, hover = non-sticky preview
  + click-to-pin, mobile = reserved-strip peek with a teaser. Don't re-fuse the maps or re-add a seam.
- Distillation supersedes every name-collision fix; Haiku ≈ Opus, so Haiku is primary. **Richer
  distillation tested (2026-07, Opus + fuller `--prompt rich`):** a modest *matching*-only gain (more
  diverse, less identity leakage; per-pair edge not significant) that does NOT reach the lead-written
  captions — **not adopted**. Full-Wikipedia-page source is matching-only too, so shelved.
- **Investigated 2026-07 with the metric suite — shipped config held on every axis:** embedding model
  is NOT the lever (nomic ≈ bge ≈ qwen3 head-to-head); neutralization is a quality×diversity tradeoff
  (raw_pca hubby / leace over-spread / **centroid the balance**); the ablation ladder shows each stage
  earns its keep on a different axis (distill→character; neutralize/CSLS→un-hub the map). Matches are
  **cloud-not-soulmate** — swap the distiller and the #1 twin moves but stays in the old ranking's top
  ~8%, so the matcher identifies an analog *neighborhood*, not a unique twin (why the map shows three).
- **Captions (07) = Sonnet 5 + symmetry-enforcing v3 prompt (shipped 2026-07-01, commit `ac63eba`).**
  Haiku/v1 captions were one-sided (a concrete trait true of only one city of the pair) ~84% of the
  time → ~52% on Sonnet-5/v3, and shorter. `scripts/judge_captions.py` (Opus LLM-as-judge: per-claim
  both/eu-only/na-only tagging + scale/invention/specificity axes, blind to cosine) is the measurement
  rig, reusable for embedding-model experiments. The old 'call a capital "a capital city", never "a
  state capital"' rule was **intentionally dropped** in v3 — specific capital types are fine when
  accurate for both cities; don't re-add a blanket ban. **Caption prompt is exhausted (2026-07):**
  v4/v4.1 tried to also fix blandness (v3 leans on templates — "on a river" 12%→41% vs v1) but hit a
  **specificity↔honesty frontier** (single-prompt tweaks just shuffle the failure among {one-sided ↔
  scale-overclaim ↔ bland}), so **v3 stays**. Captions are written from *leads*, a ceiling separate
  from matching — the only remaining lever is a richer caption *source*, deliberately not pursued.
- Dropped: 1:1 bijection, convex reconstruction, MMR, output-surgery / masking / name-subspace
  erasure (`scripts/prototype_*.py`).
