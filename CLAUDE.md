# Atlantic Mirror — project guide

A two-way character mirror between North American and European cities (full overview in README.md).
Python data pipeline (`scripts/`, numbered stages) → bespoke D3 web map (`docs/`, built) → GitHub
Pages — **live** at https://stevenfazzio.github.io/atlantic-mirror/. Public repo `atlantic-mirror`;
the North America↔Europe pivot is merged (PR #1), so work on `main` now.

## Running the pipeline
Stages run in order 01 → 02 → 02b → 03 → 04 → 05 → 07 → 08, each cached/idempotent (re-runs skip done
work; `--force` recomputes). Primary invocation uses `--source profile --profile-key haiku` for
03–08 and `--model claude-haiku-4-5 --key haiku` for 02b — exact commands in README.md § Running.
The shipped **embedder is qwen3** (`Qwen3-Embedding-0.6B`; stage 03 `--model` default) — swapped from
nomic 2026-07-02 on the hardened-lineup evidence (see Settled).
**Shipped captions:** stage 07 runs on Sonnet 5 with the v3 prompt (`--caption-model claude-sonnet-5
--caption-effort medium --prompt v3 --caption-key sonnet5v3`), and stage 08 must get `--caption-key
sonnet5v3` to export them; 07/08 *defaults* (no key) still rebuild the original Haiku/v1 baseline, so
a re-run without those flags silently reverts the map to it.
`ANTHROPIC_API_KEY` is required for 02b (distill) and 07 (caption); 08 (export web JSON) needs no key.
Products: `data/processed/matches_qwen3_profile_haiku_captioned_sonnet5v3.json` (full) and stage 08's
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
    "city": "Venice", "group": "Europe", "country": "Italy", "lat": 45.44, "lon": 12.33,
    "matches": [{"qid": "...", "city": "New Orleans", "similarity": 0.55, "caption": "A historic waterfront city famed for culture and cuisine, shaped by water..."}, ...]
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
- **City search** (added 2026-07-02): a floating top-centre pill on desktop; on **mobile a persistent
  full-width bar in a reserved top strip** (`--search-h`) — no collapse (an animated collapse janked on
  real phones). Matches a diacritic-folded, **word-start** index of all 500 cities ("york" → New York,
  not Seattle) with results disambiguated by **country + continent** (Vancouver BC vs WA). Choosing pins
  like a click, then **recenters that panel** so the dot clears the info card (`focusCity`: scale 1.7,
  gutter-biased; edge cities clamped to d3-zoom's pan bounds via `clampTransform`) and **resets the
  opposite panel to fit** so all three counterparts + arcs stay visible. **Panel titles are bottom-left**
  (cleared the top bar — don't move them back). Copy is gesture-agnostic — **"Choose a city…"** (dek,
  hint, meta). An **× (shown only when there's text) and Escape** wipe the box **and** the selection and
  reset both panels — a clean slate for the next search.
- **Mobile launcher:** the otherwise-empty bottom strip (when nothing's selected) holds example-city
  **chips** (`buildLauncher`, currently Venice/Madrid/Detroit/Boston) — a one-tap way in on a phone where
  the dots are tiny; a chip pins + recenters like a search pick. Hidden on desktop and once a city is
  selected (the peek sheet takes the strip).
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
- `lineup_eval.py` — **the primary matching-quality metric.** A *referring-expression* eval framed as a
  **police lineup**: hand an LLM a caption + a lineup (true city + look-alikes) and see if it fingers the
  right one, **from Wikipedia leads only** (the *caption* is on trial, not the model's fame memory). It's
  **grounded** — a proxy task with a verifiable answer, so the model can't just emit a preference (unlike
  a taste-judge). **Hardening distractors to nearest-neighbours (`--distractors nn`, vs `rank`) gave it
  the sensitivity it lacked**; hardened, it tracks matching quality monotonically (random < population <
  shipped) and **caught the embedding lever the pair judge scored as a tie → qwen3** (see Settled).
  Metrics: per-pair prob-mass / top-1 rate / below-chance fraction + a caption-free target-diversity signal.
- `judge_pairs.py` — blind, order-randomized head-to-head twin-quality **preference** judge (Sonnet + Opus
  spot-check), caption-free. **Now the cross-check, not the arbiter:** it catches what the lineup can't (a
  specific-shared caption built on a *coincidental* trait), but as a subjective preference it's the weaker
  instrument where the two disagree. `--config LABEL=FILE` (≥2).
- `judge_captions.py` — Opus LLM-as-judge on caption honesty (per-claim one-sided / scale / invention).
- **Additive experiment flags (defaults now reproduce the qwen3 shipped map):** 03 `--model {nomic,bge,qwen3}` (default qwen3);
  05 `--method {centroid,leace,raw_pca}` + `--rank {csls,cosine}`; 07 `--method`/`--sample`/`--prompt
  {v1..v4.1}`; 02b `--prompt {v1,rich}`. Method/rank/caption-key tags keep ablation outputs from
  colliding with the shipped files.

## Research positioning (lit review 2026-07-02 → `blog-assets/RELATED_WORK.md`)
Per-claim novelty verdicts + the blog citation shortlist live there. Headlines: the lineup is novel
**as a composition** (its family is real — REG comprehension, self-retrieval captioning, Chang-2009
intrusion tests — cite it, don't hide it); no prior applied "preference judge ties, grounded metric
separates" case study found; closest artifact = Fitzpatrick & Dunn 2019's climate-analog maps
(nobody has done the character version). A workshop+ paper centred on the lineup + two-graders
finding is plausible — second domain = Toponymy topic-name eval ("wayfinding" Phase 4, sketched
2026-07-03 in `~/repos/toponymy` `experiments/label_quality/PLAN.md`).
**Active next (decided 2026-07-03): lineup-guided best-of-k caption reranking** (RELATED_WORK.md
§ improvement op 1) — selection, not prompt tweaks, to attack the specificity↔honesty frontier;
consider a pilot (~k=4 × ~30 pairs) first; Goodhart guard = validate on a held-out lineup config
(different grader model + distractor draw).

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
- **Embedder = qwen3 (`Qwen3-Embedding-0.6B`), swapped from nomic 2026-07-02.** The original bake-off
  called the embedders a tie — but that was on the *pair judge* (subjective preference). On the **hardened
  lineup** (grounded), **qwen3 clearly leads** (top-1 67.7% vs nomic 61.3% @300 pairs, ahead on every
  sub-metric, robust 150→300, not a fame artifact), and side-by-side its pairs read as more *specific*
  twins (Boston→Edinburgh, Detroit→Stuttgart/Ingolstadt/Wolfsburg, Venice→New Orleans). bge ≈ nomic. The
  swap lost Munich→Milwaukee (why the flagship moved to Venice→New Orleans; fine pre-launch). **So
  "embedding isn't the lever" is retired — it was a lever; the *subjective* metric just couldn't see it.**
  Richer profiles don't compound (opusrich flat on qwen3 too — no embedder×distillation synergy).
- Neutralization is a quality×diversity tradeoff (raw_pca hubby / leace over-spread / **centroid the
  balance**); the ablation ladder shows each stage earns its keep on a different axis (distill→character;
  neutralize/CSLS→un-hub the map). **But the grounded lineup grades that ladder differently (2026-07-02):**
  +distill is the pair judge's *peak* (72% win-rate) yet a lineup *dip* (55% top-1), while +neutralize/+CSLS
  are the lineup's *best* rungs (61%) — distill matches on abstract character (more generic shared captions →
  less discriminable), neutralize/CSLS spread to distinctive targets (sharper captions). The two metrics
  reward **different axes** (twin-similarity vs caption-discriminability), not just different sensitivities
  (`blog-assets/ablation_ladder_both.html`, built via `ladder_lineup.py`). Matches are **cloud-not-soulmate** — swap the distiller and the #1
  twin moves but stays in the old ranking's top ~8%, so the matcher identifies an analog *neighborhood*,
  not a unique twin (why the map shows three).
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
