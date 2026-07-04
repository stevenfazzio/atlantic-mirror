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
  too crowded in EU); only the selected city + its three counterparts get on-map labels. **Dots are
  the CONTENT and must out-rank the context lines (borders/coast):** base dots are sepia-brown
  `#7a5f48` (4.1:1 vs land — a *hue* step off the olive terrain family, not just lightness) with a
  0.8px paper casing, and grow **sub-linearly with zoom** (`k^0.3` in `dotR`: same size at fit view,
  ~2.3× by 16× — fully counter-scaled constant-size dots felt lost in a deep-zoomed panel).
- **Contrast floor (2026-07-04, low-vision feedback — "beige blob"):** strokes re-tuned so country
  borders sit **≥3:1 against land** (WCAG graphics floor; they were 1.74:1): border `#7d6f54` @1px,
  coast `#a2946f` @0.9px, land `#e1d6bd`. Don't re-lighten below 3:1. **Every map stroke is
  `non-scaling-stroke`, including the land/coast outline** — the coast was the one stroke that scaled
  with zoom and became ~13px bands at deep zoom.
- **Select**: hover (desktop, non-sticky preview) / tap (mobile); **click-to-pin** on desktop so you
  can move onto the card; **nearest-city snap** via a per-card d3-quadtree. → selected city vermilion,
  its three counterparts teal in the *other* card, an **arc** drawn to each across the gutter.
- **Info card**: desktop = a card that **tracks the selected dot** on its outer side (NA→left, EU→right;
  within card-width of the edge it **flips inward** rather than clamping — the clamp used to slide it
  over the dot/cursor, which is worse than overlapping an arc; feedback 2026-07-04). Vertically the
  **title line sits level with the dot** (fixed ~28px offset, not proportional — the name lands where
  the user is looking and doesn't jump with card height). A hover card is `pointer-events:none`, only
  a pinned one is interactive. **Link semantics (2026-07-04): one signifier per meaning** — **↗ +
  no-resting-underline always and only means "external link"** (the card title's Wikipedia article;
  the colophon's GitHub link; hover/focus restores an underline), and **underline-at-rest always and
  only means "selects that city in-app"** (teal `.p-goto` counterpart names — pin + `focusCity`, like
  a search pick). No per-counterpart wiki link (its article is one hop away via its own title); don't
  give external links a resting underline or the underline signifier goes ambiguous again.
  Mobile = a **peek sheet** in a
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
- **Usability pass (2026-07-04, HCI review after the feedback round):** stale SVG `aria-label` fixed
  (it still described the abandoned cropped-Atlantic composite; deeper BLV accessibility deliberately
  **deferred**, not declined); `--ink-faint` darkened `#9a9080`→`#726955` (as text it ran 2.1–2.8:1;
  now ≥4.5:1 AA on paper/surface/card, panel titles at full opacity); **pointer cursor** whenever the
  snap radius holds a city (`#map.can-pick` — the only hover signal while a card is pinned); hover
  cards carry a **"Click to pin this card"** hint (desktop only; mobile always pins); **deep links** —
  pinning writes `#city=<QID>` via `replaceState` (shareable, no history spam; hover never touches the
  URL), applied on load + `hashchange`. **Declined — don't re-propose:** zoom +/− buttons (deliberately
  removed earlier: global zoom is confusing, per-card buttons too busy; scroll/pinch suffices), larger
  card type (real estate at a premium; browser zoom is the path for users needing bigger text), desktop
  example chips, on-map 1/2/3 rank numerals, hover-snap hysteresis (selection disagreeing with
  proximity reads worse than transient dense-cluster flicker).
- **Colophon (simplified 2026-07-04):** two text segments + a right-aligned **GitHub ↗** link (external
  grammar). The old distill/neutralize clauses were merged — they read as redundant, "CSLS" was footer
  jargon, and "country-neutralized *before* embedding" was wrong (stage 04 operates on the vectors).
  On mobile the method clause is hidden (`.colophon__method`) — scale + GitHub only.
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
- `rerank_captions.py` — best-of-k caption-selection harness (generators: `--gen iid / diverse /
  diverse-rich`; lineup-argmax + honesty-gated policies via `--judge all`; built-in held-out
  Goodhart config = second grader model + fresh distractor draw). Experiment settled — see Settled.
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
**Best-of-k caption reranking (improvement op 1): run and settled 2026-07-03 — NOT adopted; see
Settled for the numbers.** Net gain for the paper/blog story: a real applied Goodhart catch by the
held-out lineup config, and the per-candidate honesty gate as the only escape from the
specificity↔honesty frontier — both strengthen the "grounded metric + guards" thesis.

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
- **Best-of-k caption reranking (RELATED_WORK op 1) — run 2026-07-03, NOT adopted; captions stay
  Sonnet-5/v3.** Three 30-pair pilots (`rerank_captions.py`, exp-keys `bok_pilot/bok_div/bok_div2`;
  held-out guard = Opus grader + fresh 5-of-top-9 nn draw): (1) iid k=4 samples are *paraphrases*
  (token-Jaccard 0.44; grader test–retest SD ≈0.011), so the +0.018 in-sample gain was
  lineup-draw-specific fit that didn't transfer — a textbook applied Goodhart catch; (2)
  trait-diverse candidates make selection genuine (+0.023–0.026 over the pool held-out, A→B
  transfer ρ≈0.5) but the pool sits at/below shipped; (3) rich multi-trait candidates close the
  pool gap while re-loading one-sidedness (raw argmax picks 70% one-sided — the frontier reappears
  *inside the pool*). No variant beat shipped discriminability (n=30, all deltas n.s.). The
  **honesty gate** (argmax among judge-clean candidates, fallback shipped) cut one-sided
  46.7%→10–23% under an independent second judge at flat discriminability — real, but
  reader-invisible (the reader-facing metric didn't move), judge-noisy (Opus↔Sonnet agree
  ~67–70%/caption), and complexity-heavy (~$130–260 at scale plus a style-rule layer: the gate
  passes truth-but-not-taste hooks like GaWC classes and population counts). Don't revisit without
  a new reason; a caption that bugs on the map is a one-off curation edit, not machinery.
- **Brussels dedup (2026-07-04):** Q240 (enwiki *Brussels*) and Q239 (*City of Brussels*, the 195k core
  commune) were both in — one city, two Wikidata entities with different enwiki titles, so stage 01's
  title dedup couldn't catch it (the only true duplicate; a 20 km same-group proximity scan cleared the
  other 499). Fixed in 01 via `EU_EXCLUDE_TITLES` (drop Q239) + `LABEL_FIX` (Q240's label
  "Brussels-Capital Region" → "Brussels"); Białystok promoted to EU #250. Full 01→08 re-run: D.C. —
  de-hubbed of the dupe (Q239 was its #1) — now surfaces as a counterpart for Paris/Vienna/Prague/
  Bucharest; ~29 cities saw a #3-slot change (cloud-not-soulmate churn), cached captions untouched.
- Dropped: 1:1 bijection, convex reconstruction, MMR, output-surgery / masking / name-subspace
  erasure (`scripts/prototype_*.py`).
