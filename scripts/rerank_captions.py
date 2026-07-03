"""Experiment: lineup-guided best-of-k caption reranking (RELATED_WORK.md improvement op 1).

Selection instead of prompt tweaks: sample k captions per pair from the SHIPPED caption config
(Sonnet 5 / v3 / effort medium -- Sonnet 5 takes no temperature, so diversity is sampling
stochasticity), score each candidate with the hardened lineup on BOTH cities, and take the argmax
of pair_min. Because pair_min is the worse of the two sides, a one-sided caption aces one lineup
and flunks the other -- selection pressure should favor symmetry, not fight it. Two guards:
  * Goodhart (Gao et al.): the gain is validated on a HELD-OUT lineup config -- different grader
    model (Opus) AND a different distractor draw (seeded 5-of-top-9 nn instead of top-5).
  * Honesty: the Opus per-claim judge (judge_captions.py) audits shipped vs selected for
    one-sided / scale-overclaim / invented -- specificity gains must not cost symmetry.

Conditions per pair: shipped (live caption), c0..c{k-1} (fresh samples), selected (argmax of
selection-config pair_min; ties -> pair_harm, then lowest index). All conditions are scored on
both configs; candidates on the held-out config give the candidate-mean control (selection gain
net of sampling luck) and a per-pair A<->B transfer correlation.

Two candidate generators (--gen). 'iid' = k independent same-config samples (pilot 1, 2026-07-03:
NULL -- same-config samples are paraphrases, within-pair spread ~0.02 vs grader test-retest SD
~0.011, so selection picked lineup-draw-specific fit that did not transfer). 'diverse' = ONE
structured call per pair returning k candidates each anchored on a DIFFERENT genuinely-shared
trait (v3 honesty rules apply to each) -- semantic variance instead of paraphrase variance.
With --judge all, every candidate is honesty-judged and a GATED policy is also computed: argmax
pair_min among candidates with a clean verdict (no one-sided / unsupported / scale-overclaimed /
invented claims), falling back to the shipped caption when none pass.

Reads:  data/processed/<--captions> (shipped captioned matches), cities.parquet, <--ref-reps>
Writes: data/raw/rerank/<exp-key>/candidates/<eu>__<na>__{c<i>.txt|div.json}   cached per pair
        data/raw/rerank/<exp-key>/lineup_{sel,val}/<eu>__<na>__<side>__<caphash>.json
        data/raw/caption_judgments/<--shipped-judge-key>/<eu>__<na>.json   (shared honesty cache)
        data/raw/caption_judgments/<exp-key>_cand/<eu>__<na>__<caphash>.json
        data/processed/rerank_<exp-key>.parquet           per (pair x condition) lineup scores
        data/processed/rerank_<exp-key>_honesty.parquet   per (pair x judged cond) judge rows
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import judge_captions
import lineup_eval
import numpy as np
import pandas as pd

from _common import PROCESSED, RAW, cached_json, write_df

LINEUP = lineup_eval.LINEUP  # 1 true + 5 distractors; chance prob-mass = 1/6

client = anthropic.Anthropic(max_retries=5)  # used only by the diverse candidate generator

DIVERSE_SCHEMA = {  # API rejects minItems>1, so candidate count is normalized in code
    "type": "object",
    "additionalProperties": False,
    "properties": {"captions": {"type": "array", "items": {"type": "string"}}},
    "required": ["captions"],
}


def load_stage07():
    """Stage 07's filename starts with a digit, so import the canonical caption prompts by path."""
    spec = importlib.util.spec_from_file_location(
        "stage07", Path(__file__).parent / "07_caption.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def caphash(caption: str) -> str:
    return hashlib.md5(caption.encode()).hexdigest()[:10]


def build_nn_lists(targets, meta, refvecs, g2q, n):
    """Top-n same-group nearest neighbours per target city, by the fixed reference embedding."""
    out = {}
    for tq in targets:
        pool = [q for q in g2q[meta.loc[tq, "country"]] if q != tq and q in refvecs]
        v = refvecs[tq]
        sims = np.array([refvecs[q] @ v for q in pool])
        out[tq] = [pool[i] for i in np.argsort(-sims)[:n]]
    return out


def build_lineup(tq, nn_lists, cfg, val_nn_top):
    """cfg='sel' reproduces lineup_eval --distractors nn byte-for-byte (top-5 nn, md5(tq) shuffle
    seed) so selection optimizes the actual primary metric. cfg='val' is the held-out draw: a
    seeded sample of 5 from the top-<val_nn_top> nn plus a fresh shuffle seed."""
    if cfg == "sel":
        rng = random.Random(int.from_bytes(hashlib.md5(tq.encode()).digest()[:8], "big"))
        nbrs = nn_lists[tq][: LINEUP - 1]
    else:
        rng = random.Random(int.from_bytes(hashlib.md5(f"{tq}__val".encode()).digest()[:8], "big"))
        nbrs = rng.sample(nn_lists[tq][:val_nn_top], LINEUP - 1)
    lu = [tq] + nbrs
    rng.shuffle(lu)
    return lu


def score_side(tq, caption, leads, nn_lists, *, cfg, path, model, effort, val_nn_top, legacy=None):
    """One grader call: caption vs the lineup for one city. Cached by caption hash; the shipped
    caption's selection-config judgments are read from the primary metric's own cache when
    available (identical lineup construction + grader config)."""
    if legacy is not None and legacy.exists():
        return json.loads(legacy.read_text())

    def compute():
        lu = build_lineup(tq, nn_lists, cfg, val_nn_top)
        sc = lineup_eval.identify(caption, [leads[q] for q in lu], model=model, effort=effort)
        tpos = lu.index(tq)
        tot = sum(sc) or 1
        return {
            "prob_mass": sc[tpos] / tot,
            "rank": 1 + sum(s > sc[tpos] for s in sc),
            "lineup": lu,
            "scores": sc,
            "true_pos": tpos,
        }

    return cached_json(path, compute)


def gen_candidate(euq, naq, i, leads, s07, *, cand_dir, model, system, effort):
    path = cand_dir / f"{euq}__{naq}__c{i}.txt"
    if path.exists():
        return path.read_text().strip()
    text = s07.make_caption(leads[euq], leads[naq], model=model, system=system, effort=effort)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return text.strip()


def diverse_system(v3: str, k: int, rich: bool = False) -> str:
    """Rewrite stage 07's single-caption prompt into a k-candidate generator: identical honesty
    rules, but candidates must differ in WHICH shared trait anchors them (pilot 1 showed
    same-config samples are paraphrases -- selection needs semantic variance). rich=True adds the
    pilot-2 fix: candidates came out as ~10-word single-trait fragments whose pool scored below
    the multi-trait shipped captions, so demand full-depth captions that differ only in the LEAD
    trait. Asserts so an upstream prompt edit fails loudly here instead of silently diverging."""
    single = "Write a single concise descriptive PHRASE (not a full sentence; 20 words MAXIMUM)"
    multi = (
        f"Write {k} DIFFERENT candidate captions -- each a single concise descriptive PHRASE "
        "(not a full sentence; 20 words MAXIMUM)"
    )
    tail = "- Output only the phrase -- no preamble."
    richness = (
        "- FULL DEPTH: each candidate must be a complete caption of the same depth you would "
        "write if it were the pair's only caption -- weave TWO or THREE genuinely-shared traits "
        "into one phrase (aim for 14-20 words), OPENING with that candidate's anchor trait. Do "
        "NOT write bare single-trait fragments. Supporting traits may recur across candidates; "
        "the OPENING anchor trait must not.\n"
        if rich
        else ""
    )
    diverse_tail = (
        "- DIVERSITY ACROSS CANDIDATES: each candidate must be anchored on a DIFFERENT "
        "genuinely-shared trait (a different industry, era, geographic or physical feature, or "
        "economic/cultural role). No two candidates may be paraphrases of the same idea. If the "
        f"leads support fewer than {k} distinct shared traits, make the remaining candidates "
        "honestly broader instead of stretching a one-sided trait.\n"
        + richness
        + "- Every rule above applies to EACH candidate independently; any candidate must be able "
        "to stand alone as the pair's single caption.\n"
        f'Return JSON: {{"captions": [...]}} with exactly {k} candidate phrases.'
    )
    assert single in v3 and tail in v3, "stage 07 v3 prompt changed; update diverse_system()"
    return v3.replace(single, multi).replace(tail, diverse_tail)


def gen_diverse(euq, naq, k, leads, *, cand_dir, model, system, effort):
    """One structured call -> k trait-diverse candidates; cached per pair. Short/long returns
    are normalized (cycle-pad / truncate) so downstream always sees exactly k."""
    path = cand_dir / f"{euq}__{naq}__div.json"

    def compute():
        resp = client.messages.create(
            model=model,
            max_tokens=4000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": DIVERSE_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"European city:\n{leads[euq]}\n\nNorth American city:\n{leads[naq]}"
                    ),
                }
            ],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    caps = [c.strip() for c in cached_json(path, compute)["captions"] if c.strip()]
    if not caps:
        raise RuntimeError(f"empty diverse candidate set for {euq}__{naq}")
    if len(caps) != k:
        print(f"  ! {euq}__{naq}: {len(caps)} candidates returned (want {k}); normalizing")
    return [(caps * k)[i] for i in range(k)]


def run_pool(jobs, workers, tag):
    """Run {key: thunk} in a thread pool; return {key: result}, raising at the end if any failed."""
    results, failed, done = {}, [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:  # cache keeps completed work; re-run resumes
                failed.append((key, exc))
            done += 1
            if done % 25 == 0:
                print(f"  {tag}: {done}/{len(jobs)}")
    if failed:
        print(f"  !! {tag}: {len(failed)} failures, e.g. {failed[0][0]}: {failed[0][1]}")
        raise RuntimeError(f"{tag}: {len(failed)}/{len(jobs)} jobs failed; re-run to resume")
    return results


def boot_ci(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-key", default="bok_pilot", help="namespaces every cache + output file")
    ap.add_argument("--captions", default="matches_qwen3_profile_haiku_captioned_sonnet5v3.json")
    ap.add_argument("--pairs", type=int, default=30, help="hash-sampled pair count (0 = all)")
    ap.add_argument("--k", type=int, default=4, help="fresh candidates per pair")
    ap.add_argument(
        "--gen",
        choices=["iid", "diverse", "diverse-rich"],
        default="iid",
        help="iid = k independent same-config samples; diverse[-rich] = one call, k trait-diverse "
        "(-rich demands full-depth multi-trait candidates)",
    )
    # candidate generation = the shipped stage-07 config
    ap.add_argument("--caption-model", default="claude-sonnet-5")
    ap.add_argument("--prompt", default="v3")
    ap.add_argument("--caption-effort", default="medium")
    # selection config (A) = the primary metric
    ap.add_argument("--sel-model", default=lineup_eval.IDENTIFIER_MODEL)
    ap.add_argument("--sel-effort", default="low")
    ap.add_argument(
        "--shipped-lineup-key",
        default="qwen3nn",
        help="reuse shipped-caption config-A judgments from this lineup_eval cache ('' = off)",
    )
    # held-out config (B): different grader + different distractor draw
    ap.add_argument("--val-model", default="claude-opus-4-8")
    ap.add_argument("--val-effort", default="low")
    ap.add_argument("--val-nn-top", type=int, default=9, help="held-out draw samples 5 of top-N nn")
    ap.add_argument("--ref-reps", default="reps_qwen3_profile_haiku.parquet")
    ap.add_argument("--ref-method", default="centroid")
    # honesty judge
    ap.add_argument(
        "--judge",
        choices=["selected", "all"],
        default="selected",
        help="judge only shipped + the lineup-argmax pick, or every candidate (enables the gate)",
    )
    ap.add_argument("--judge-model", default=judge_captions.JUDGE_MODEL)
    ap.add_argument("--judge-effort", default="medium")
    ap.add_argument(
        "--shipped-judge-key",
        default="sonnet5v3",
        help="shared caption_judgments cache namespace for the shipped captions",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    matches = json.loads((PROCESSED / args.captions).read_text())
    cdf = pd.read_parquet(PROCESSED / "cities.parquet").drop_duplicates("qid").set_index("qid")
    leads = cdf["lead_text"].to_dict()
    cities = {q: (r["city"], r["country"], r["country_name"]) for q, r in cdf.iterrows()}

    reps = pd.read_parquet(PROCESSED / args.ref_reps)
    reps = reps[reps["method"] == args.ref_method]
    refvecs = {
        r.qid: (lambda a: a / (np.linalg.norm(a) + 1e-12))(np.asarray(r.embedding, dtype="float64"))
        for r in reps.itertuples(index=False)
    }
    g2q = {g: list(idx) for g, idx in cdf.groupby("country").groups.items()}

    # Unique EU<->NA pairs with shipped caption + similarity; same md5 hash-sample as lineup_eval,
    # so the pilot pairs are a subset of the shipped 300-pair eval (cache reuse for shipped x A).
    pairs = {}
    for q, rec in matches.items():
        for m in rec["matches"]:
            pairs[lineup_eval.norm_pair(q, rec["group"], m["qid"])] = (
                m["caption"].strip(),
                m["similarity"],
            )
    keys = sorted(pairs)
    if args.pairs and args.pairs < len(keys):
        keys = sorted(keys, key=lambda p: hashlib.md5(f"{p[0]}__{p[1]}".encode()).hexdigest())[
            : args.pairs
        ]
        keys = sorted(keys)

    targets = sorted({q for p in keys for q in p})
    nn_lists = build_nn_lists(targets, cdf, refvecs, g2q, max(LINEUP - 1, args.val_nn_top))

    n_gen = len(keys) * (args.k if args.gen == "iid" else 1)
    n_score = len(keys) * (args.k + 1) * 2 * 2  # conds x sides x configs (upper bound, pre-cache)
    n_judge = len(keys) * (args.k + 1 if args.judge == "all" else 2)
    print(
        f"best-of-{args.k} rerank [{args.exp_key}] gen={args.gen}: {len(keys)} pairs, "
        f"lineup={LINEUP}, chance={1 / LINEUP:.3f}\n"
        f"  gen<= {n_gen} ({args.caption_model}/{args.prompt}/{args.caption_effort})  "
        f"score<= {n_score} (A={args.sel_model}, B={args.val_model})  "
        f"judge<= {n_judge} ({args.judge_model}, mode={args.judge})"
    )

    # ---- phase 1: k fresh candidates per pair (cached) --------------------------------------
    s07 = load_stage07()
    cand_dir = RAW / "rerank" / args.exp_key / "candidates"
    if args.gen != "iid":
        system = diverse_system(s07.PROMPTS[args.prompt], args.k, rich=args.gen == "diverse-rich")
        div_jobs = {
            (eu, na): (
                lambda eu=eu, na=na: gen_diverse(
                    eu,
                    na,
                    args.k,
                    leads,
                    cand_dir=cand_dir,
                    model=args.caption_model,
                    system=system,
                    effort=args.caption_effort,
                )
            )
            for eu, na in keys
        }
        div = run_pool(div_jobs, args.workers, "generate")
        cand_texts = {(eu, na, i): div[(eu, na)][i] for eu, na in keys for i in range(args.k)}
    else:
        system = s07.PROMPTS[args.prompt]
        gen_jobs = {
            (eu, na, i): (
                lambda eu=eu, na=na, i=i: gen_candidate(
                    eu,
                    na,
                    i,
                    leads,
                    s07,
                    cand_dir=cand_dir,
                    model=args.caption_model,
                    system=system,
                    effort=args.caption_effort,
                )
            )
            for eu, na in keys
            for i in range(args.k)
        }
        cand_texts = run_pool(gen_jobs, args.workers, "generate")

    conds = {}  # (eu, na) -> {cond_name: caption}
    for eu, na in keys:
        conds[(eu, na)] = {"shipped": pairs[(eu, na)][0]} | {
            f"c{i}": cand_texts[(eu, na, i)] for i in range(args.k)
        }

    # ---- phase 2: score every (pair, condition, side) on both configs (deduped by caption) --
    cfg_setup = {
        "sel": (args.sel_model, args.sel_effort),
        "val": (args.val_model, args.val_effort),
    }
    score_jobs, consumers = {}, []  # consumers: (pair, cond, cfg, side, jobkey)
    for eu, na in keys:
        for cond, caption in conds[(eu, na)].items():
            for cfg in ("sel", "val"):
                model, effort = cfg_setup[cfg]
                for side, tq in (("eu", eu), ("na", na)):
                    legacy = None
                    if cond == "shipped" and cfg == "sel" and args.shipped_lineup_key:
                        legacy = (
                            RAW
                            / "lineup_judgments"
                            / args.shipped_lineup_key
                            / f"{eu}__{na}__{side}.json"
                        )
                    path = (
                        RAW
                        / "rerank"
                        / args.exp_key
                        / f"lineup_{cfg}"
                        / f"{eu}__{na}__{side}__{caphash(caption)}.json"
                    )
                    jobkey = (cfg, eu, na, side, caphash(caption))
                    if jobkey not in score_jobs:
                        score_jobs[jobkey] = (
                            lambda tq=tq,
                            caption=caption,
                            cfg=cfg,
                            path=path,
                            model=model,
                            effort=effort,
                            legacy=legacy: score_side(
                                tq,
                                caption,
                                leads,
                                nn_lists,
                                cfg=cfg,
                                path=path,
                                model=model,
                                effort=effort,
                                val_nn_top=args.val_nn_top,
                                legacy=legacy,
                            )
                        )
                    consumers.append(((eu, na), cond, cfg, side, jobkey))
    scores = run_pool(score_jobs, args.workers, "score")

    # ---- assemble long rows + pick the argmax candidate per pair ----------------------------
    per = {}  # (pair, cond) -> {(cfg, side): rec}
    for pair, cond, cfg, side, jobkey in consumers:
        per.setdefault((pair, cond), {})[(cfg, side)] = scores[jobkey]

    rows, selected = [], {}
    for eu, na in keys:
        best = None
        for cond, caption in conds[(eu, na)].items():
            r = per[((eu, na), cond)]
            row = {
                "eu_qid": eu,
                "na_qid": na,
                "eu_city": cdf.loc[eu, "city"],
                "na_city": cdf.loc[na, "city"],
                "cond": cond,
                "caption": caption,
                "words": len(caption.split()),
                "similarity": pairs[(eu, na)][1],
            }
            for cfg in ("sel", "val"):
                pm_e, pm_n = r[(cfg, "eu")]["prob_mass"], r[(cfg, "na")]["prob_mass"]
                row[f"{cfg}_pm_eu"], row[f"{cfg}_pm_na"] = pm_e, pm_n
                row[f"{cfg}_pair_min"] = min(pm_e, pm_n)
                row[f"{cfg}_pair_harm"] = (
                    0.0 if pm_e + pm_n == 0 else 2 * pm_e * pm_n / (pm_e + pm_n)
                )
                row[f"{cfg}_rank_eu"] = r[(cfg, "eu")]["rank"]
                row[f"{cfg}_rank_na"] = r[(cfg, "na")]["rank"]
            rows.append(row)
            if cond != "shipped":
                key_ = (row["sel_pair_min"], row["sel_pair_harm"])
                if best is None or key_ > best[0]:
                    best = (key_, cond)
        selected[(eu, na)] = best[1]
    df = pd.DataFrame(rows)
    df["selected"] = [selected[(r.eu_qid, r.na_qid)] == r.cond for r in df.itertuples(index=False)]

    # ---- phase 3: honesty judge (shipped + argmax pick, or every candidate with --judge all) --
    judge_jobs = {}
    for eu, na in keys:
        ship_cap = conds[(eu, na)]["shipped"]
        judge_jobs[(eu, na, "shipped")] = (
            lambda eu=eu, na=na, cap=ship_cap: judge_captions.judge_pair(
                eu,
                na,
                cap,
                leads[eu],
                leads[na],
                model=args.judge_model,
                effort=args.judge_effort,
                caption_key=args.shipped_judge_key,
                force=False,
            )[1]
        )
        cand_conds = (
            [c for c in conds[(eu, na)] if c != "shipped"]
            if args.judge == "all"
            else [selected[(eu, na)]]
        )
        for cond in cand_conds:
            cap = conds[(eu, na)][cond]
            path = (
                RAW
                / "caption_judgments"
                / f"{args.exp_key}_cand"
                / f"{eu}__{na}__{caphash(cap)}.json"
            )
            judge_jobs[(eu, na, cond)] = lambda eu=eu, na=na, cap=cap, path=path: cached_json(
                path,
                lambda: judge_captions.judge(
                    cap, leads[eu], leads[na], model=args.judge_model, effort=args.judge_effort
                ),
            )
    verdicts = run_pool(judge_jobs, args.workers, "judge")

    hon_rows = []
    for (eu, na, cond), verdict in verdicts.items():
        cap = conds[(eu, na)][cond]
        hon_rows.append(
            judge_captions.flatten(eu, na, cap, pairs[(eu, na)][1], cities, verdict)
            | {"cond": cond}
        )
    hon = pd.DataFrame(hon_rows)

    # ---- gated policy (--judge all): best honest candidate, else keep the shipped caption ----
    if args.judge == "all":
        clean = hon.set_index(["eu_qid", "na_qid", "cond"])
        clean = ~(
            clean["one_sided"]
            | clean["unsupported"]
            | clean["scale_overclaimed"]
            | clean["invented"]
        )
        dfi = df.set_index(["eu_qid", "na_qid", "cond"])
        gated = {}
        for eu, na in keys:
            passing = [
                c for c in conds[(eu, na)] if c != "shipped" and bool(clean.get((eu, na, c), False))
            ]
            gated[(eu, na)] = (
                max(
                    passing,
                    key=lambda c: (
                        dfi.loc[(eu, na, c), "sel_pair_min"],
                        dfi.loc[(eu, na, c), "sel_pair_harm"],
                    ),
                )
                if passing
                else "shipped"
            )
        df["selected_gated"] = [
            gated[(r.eu_qid, r.na_qid)] == r.cond for r in df.itertuples(index=False)
        ]

    report(df, hon, args)
    out = PROCESSED / f"rerank_{args.exp_key}.parquet"
    write_df(df, out)
    hout = PROCESSED / f"rerank_{args.exp_key}_honesty.parquet"
    write_df(hon, hout)
    print(f"\nWrote {len(df)} rows -> {out.name} and {len(hon)} -> {hout.name}")


def cond_stats(sub, cfg):
    top1 = pd.concat([sub[f"{cfg}_rank_eu"], sub[f"{cfg}_rank_na"]]).eq(1).mean()
    return sub[f"{cfg}_pair_min"].mean(), sub[f"{cfg}_pair_harm"].mean(), top1


def token_jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb)


def report(df, hon, args):
    k = args.k
    npairs = df[["eu_qid", "na_qid"]].drop_duplicates().shape[0]
    cands = df[df["cond"] != "shipped"]
    shipped = df[df["cond"] == "shipped"].set_index(["eu_qid", "na_qid"])
    sel = df[df["selected"]].set_index(["eu_qid", "na_qid"])
    gsel = (
        df[df["selected_gated"]].set_index(["eu_qid", "na_qid"])
        if "selected_gated" in df.columns
        else None
    )
    cand_mean = cands.groupby(["eu_qid", "na_qid"])[
        [c for c in df.columns if c.startswith(("sel_", "val_"))]
    ].mean()

    print(
        f"\n########## best-of-{k} rerank [{args.exp_key}] gen={args.gen}  "
        f"chance={1 / LINEUP:.3f} ##########"
    )
    distinct = cands.groupby(["eu_qid", "na_qid"])["caption"].nunique()
    js = [
        token_jaccard(x, y)
        for _, g in cands.groupby(["eu_qid", "na_qid"])
        for i, x in enumerate(g["caption"].tolist())
        for y in g["caption"].tolist()[i + 1 :]
    ]
    print(
        f"  candidate diversity: mean {distinct.mean():.2f}/{k} distinct, "
        f"pairwise token-jaccard {np.mean(js):.2f}; "
        f"words shipped={shipped['words'].mean():.1f} cands={cands['words'].mean():.1f} "
        f"selected={sel['words'].mean():.1f}"
    )
    dup = sum(sel.loc[p, "caption"] == shipped.loc[p, "caption"] for p in shipped.index)
    picks = df[df["selected"]]["cond"].value_counts().sort_index()
    print(f"  lineup-argmax index dist: {picks.to_dict()}; == shipped text on {dup}/{npairs} pairs")

    hdr = f"{'A:min':>6} {'A:harm':>7} {'A:top1':>7} | {'B:min':>6} {'B:harm':>7} {'B:top1':>7}"
    print(f"\n  {'condition':22} {hdr}")
    conditions = [
        ("shipped", shipped.reset_index()),
        ("candidate mean", None),
        ("lineup-argmax", sel.reset_index()),
    ]
    if gsel is not None:
        conditions.append(("gated policy", gsel.reset_index()))
    for name, sub in conditions:
        if name == "candidate mean":
            a_min, a_harm = cand_mean["sel_pair_min"].mean(), cand_mean["sel_pair_harm"].mean()
            b_min, b_harm = cand_mean["val_pair_min"].mean(), cand_mean["val_pair_harm"].mean()
            a_t = pd.concat([cands["sel_rank_eu"], cands["sel_rank_na"]]).eq(1).mean()
            b_t = pd.concat([cands["val_rank_eu"], cands["val_rank_na"]]).eq(1).mean()
        else:
            a_min, a_harm, a_t = cond_stats(sub, "sel")
            b_min, b_harm, b_t = cond_stats(sub, "val")
        a_part = f"{a_min:6.3f} {a_harm:7.3f} {a_t:6.1%}"
        print(f"  {name:22} {a_part} | {b_min:6.3f} {b_harm:7.3f} {b_t:6.1%}")

    print("\n  held-out (B) paired deltas, pair_min [95% bootstrap CI]:")
    idx = shipped.index
    d_ship = (sel["val_pair_min"].reindex(idx) - shipped["val_pair_min"]).to_numpy()
    d_mean = (sel["val_pair_min"].reindex(idx) - cand_mean["val_pair_min"].reindex(idx)).to_numpy()
    d_base = (cand_mean["val_pair_min"].reindex(idx) - shipped["val_pair_min"]).to_numpy()
    deltas = [
        ("lineup-argmax - shipped ", d_ship),
        ("lineup-argmax - candmean", d_mean),
        ("cand mean - shipped     ", d_base),
    ]
    if gsel is not None:
        d_gate = (gsel["val_pair_min"].reindex(idx) - shipped["val_pair_min"]).to_numpy()
        deltas.insert(0, ("gated policy - shipped  ", d_gate))
    for label, d in deltas:
        lo, hi = boot_ci(d)
        print(f"    {label} = {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    a_gain = (sel["sel_pair_min"].reindex(idx) - shipped["sel_pair_min"]).mean()
    print(
        f"  in-sample (A) lineup-argmax - shipped = {a_gain:+.3f}  "
        f"(Goodhart gap A-B: {a_gain - d_ship.mean():+.3f})"
    )

    # does the selection signal transfer? within-pair rank corr over the k candidates
    corrs = []
    for _, g in cands.groupby(["eu_qid", "na_qid"]):
        if g["sel_pair_min"].nunique() > 1 and g["val_pair_min"].nunique() > 1:
            corrs.append(g["sel_pair_min"].corr(g["val_pair_min"], method="spearman"))
    print(
        f"  A->B transfer: mean within-pair spearman over candidates = "
        f"{np.mean(corrs):+.3f} (n={len(corrs)} pairs with variation)"
    )

    print(f"\n  honesty (judge={args.judge_model}, mode={args.judge}):")
    key3 = ["eu_qid", "na_qid", "cond"]
    blocks = [("shipped", hon[hon["cond"] == "shipped"])]
    if args.judge == "all":
        blocks.append(("candidates (all)", hon[hon["cond"] != "shipped"]))
    blocks.append(("lineup-argmax", hon.merge(df[df["selected"]][key3], on=key3)))
    if gsel is not None:
        blocks.append(("gated pick", hon.merge(df[df["selected_gated"]][key3], on=key3)))
    hdr2 = (
        f"{'one-sided':>10} {'unsupp':>7} {'scale-over':>11} {'invented':>9} "
        f"{'concrete':>9} {'vague':>7} {'broad-ok':>9}"
    )
    print(f"  {'cond':18} {hdr2}")
    for name, h in blocks:
        spec = h["specificity"].value_counts(normalize=True)
        print(
            f"  {name:18} {h['one_sided'].mean():>9.1%} {h['unsupported'].mean():>6.1%} "
            f"{h['scale_overclaimed'].mean():>10.1%} {h['invented'].mean():>8.1%} "
            f"{spec.get('concrete', 0):>8.1%} {spec.get('vague_avoidable', 0):>6.1%} "
            f"{spec.get('broad_appropriate', 0):>8.1%}"
        )
    if gsel is not None:
        cand_hon = hon[hon["cond"] != "shipped"]
        cand_ok = ~(
            cand_hon["one_sided"]
            | cand_hon["unsupported"]
            | cand_hon["scale_overclaimed"]
            | cand_hon["invented"]
        )
        n_fb = int((gsel["cond"] == "shipped").sum())
        print(
            f"  gate: {cand_ok.mean():.1%} of candidates pass; "
            f"fallback to shipped on {n_fb}/{npairs} pairs"
        )

    print("\n  sample candidate sets (A=selection config, B=held-out):")
    for p in list(shipped.index)[:2]:
        srow = shipped.loc[p]
        print(
            f"    {srow['eu_city']} <-> {srow['na_city']}   "
            f"shipped [A {srow['sel_pair_min']:.2f} B {srow['val_pair_min']:.2f}]: "
            f"{srow['caption']}"
        )
        g = df[(df["eu_qid"] == p[0]) & (df["na_qid"] == p[1]) & (df["cond"] != "shipped")]
        for r in g.itertuples(index=False):
            tag = " <-argmax" if r.selected else ""
            if "selected_gated" in df.columns and r.selected_gated:
                tag += " <-gated"
            print(
                f"      {r.cond} [A {r.sel_pair_min:.2f} B {r.val_pair_min:.2f}]{tag}: {r.caption}"
            )

    delta = (sel["val_pair_min"].reindex(idx) - shipped["val_pair_min"]).sort_values()
    print("\n  biggest held-out gains (lineup-argmax vs shipped):")
    for p in delta.index[-4:][::-1]:
        print(
            f"    [{delta[p]:+.2f}] {shipped.loc[p, 'eu_city']} <-> {shipped.loc[p, 'na_city']}\n"
            f"        shipped:  {shipped.loc[p, 'caption']}\n"
            f"        selected: {sel.loc[p, 'caption']}"
        )
    print("\n  biggest held-out losses:")
    for p in delta.index[:3]:
        print(
            f"    [{delta[p]:+.2f}] {shipped.loc[p, 'eu_city']} <-> {shipped.loc[p, 'na_city']}\n"
            f"        shipped:  {shipped.loc[p, 'caption']}\n"
            f"        selected: {sel.loc[p, 'caption']}"
        )


if __name__ == "__main__":
    main()
