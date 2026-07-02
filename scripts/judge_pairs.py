"""Diagnostic: blind head-to-head judge for the embedding bake-off.

Given two-or-more match sets (each a stage-05 matches_*.json produced by a different config -- a
different embedding model, or a different neutralization --method), this compares them on TWIN
QUALITY without captions. For each home city, it takes each config's top match (rank 0 by default)
and, for every pair of configs that DISAGREE, asks a judge which candidate is the better
character-twin. Presentation order is flipped deterministically per (home, config-pair) so the judge
is blind to which config produced which candidate; when two configs pick the SAME twin it is an
automatic tie and costs no API call.

Primary judge is Sonnet (preference is an easier task than the caption audit); a small Opus
spot-check re-judges a sample to confirm the cheap judge agrees. Per-comparison verdicts are cached
(resumable). Use --dry-run to see how many real judge calls a run would make, for free.

Reads:  data/processed/<each --config FILE>, data/processed/cities.parquet (leads + names)
Writes: data/raw/pair_judgments/<a>__<b>/<home_qid>.json          cached per comparison
        data/processed/pair_judgments_<labels>.parquet            flattened, analysis-ready
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from _common import PROCESSED, RAW, cached_json, write_df

JUDGE_MODEL = "claude-sonnet-5"
SPOT_MODEL = "claude-opus-4-8"
WORKERS = 8

SYSTEM = """You are judging a two-way "sibling cities" map that pairs a city with its closest \
character-twin on another continent. You are shown one HOME city and two CANDIDATE twin cities (each \
from the other continent). Pick which candidate is the better character-twin of the home city -- the \
one that shares more of its transferable CHARACTER.

Transferable character is what KIND of place a city is, independent of its name or country:
- economic base and main industries (heavy industry, finance, tech, tourism, agriculture, \
port/logistics, university/research, government...)
- geographic and physical setting (coastal, riverside, mountainous, climate, natural features)
- size and regional role (dominant national metropolis, major regional center, modest regional hub, \
satellite/commuter town, small historic town, resort...)
- the historical era that shaped it (ancient/medieval origins, industrial-revolution boom, \
post-industrial decline or regeneration, planned/new town, rapid modern growth)
- cultural identity (university town, arts/music scene, sports culture, religious or administrative \
significance)

Reward genuine shared character across SEVERAL of these axes. Penalize a candidate that resembles \
the home city only superficially (shares a single generic trait) or that mismatches badly on \
SCALE/ROLE -- a small town paired with a national capital is a poor twin even if both are "historic". \
Judge only from the three leads; do not use outside knowledge.

Answer "1" or "2" for the better twin. Use "tie" only when the two candidates are genuinely equal in \
twin quality -- avoid it whenever you can distinguish them. Give a one-sentence reason naming the \
deciding shared or missing character."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
}

client = anthropic.Anthropic(max_retries=5)


def judge(home: str, cand1: str, cand2: str, *, model: str, effort: str) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": (
                    f"HOME CITY:\n{home}\n\nCANDIDATE 1:\n{cand1}\n\nCANDIDATE 2:\n{cand2}"
                ),
            }
        ],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def presentation(qid: str, l0: str, l1: str) -> tuple[str, str]:
    """Deterministic, cache-stable order flip: return (option1_label, option2_label) for the sorted
    pair (l0, l1). Same (home, pair) always maps the same way, so blinding survives re-runs."""
    bit = hashlib.md5(f"{qid}|{l0}|{l1}".encode()).digest()[0] & 1
    return (l0, l1) if bit == 0 else (l1, l0)


def winner_label(verdict: dict, opt1: str, opt2: str) -> str:
    return {"1": opt1, "2": opt2, "tie": "tie"}[verdict["winner"]]


def compare(l0, l1, qid, opt1, opt2, leads, *, model, effort, force) -> dict:
    """Judge one (config-pair, home) comparison. opt1/opt2 are the match qids shown as candidate 1/2."""
    path = RAW / "pair_judgments" / f"{l0}__{l1}" / f"{qid}.json"
    verdict = cached_json(
        path,
        lambda: judge(leads[qid], leads[opt1], leads[opt2], model=model, effort=effort),
        force=force,
    )
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="LABEL=FILE",
        help="a match set to compare, e.g. nomic=matches_nomic_profile_haiku.json (repeat >=2x)",
    )
    ap.add_argument(
        "--rank", type=int, default=0, help="which twin to compare (0 = each config's top match)"
    )
    ap.add_argument(
        "--sample", type=int, default=0, help="judge only N random home cities (0 = all)"
    )
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument(
        "--spot-check",
        type=int,
        default=15,
        help="re-judge N decided comparisons with Opus (0 = off)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report how many real judge calls, spend nothing"
    )
    ap.add_argument("--force", action="store_true", help="re-judge even if cached")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    configs: dict[str, dict] = {}
    for spec in args.config:
        label, _, fname = spec.partition("=")
        assert fname, f"--config must be LABEL=FILE, got {spec!r}"
        configs[label] = json.loads((PROCESSED / fname).read_text())
    assert len(configs) >= 2, "need >=2 configs to compare"

    cdf = pd.read_parquet(PROCESSED / "cities.parquet")
    leads = cdf.set_index("qid")["lead_text"].to_dict()
    city = cdf.set_index("qid")["city"].to_dict()
    grp = cdf.set_index("qid")["country"].to_dict()

    def match_at(cfg: dict, qid: str) -> str | None:
        rec = cfg.get(qid)
        if not rec or len(rec.get("matches", [])) <= args.rank:
            return None
        return rec["matches"][args.rank]["qid"]

    # home cities present (with a rank-`args.rank` match) in EVERY config
    common = sorted(
        set.intersection(
            *[{q for q in cfg if match_at(cfg, q) is not None} for cfg in configs.values()]
        )
    )
    if args.sample and args.sample < len(common):
        # deterministic subsample (no RNG): stable hash order, take first N
        common = sorted(common, key=lambda q: hashlib.md5(f"sample|{q}".encode()).hexdigest())[
            : args.sample
        ]
        common = sorted(common)
    print(
        f"{len(configs)} configs {list(configs)}; comparing rank-{args.rank} twin over {len(common)} home cities"
    )

    # enumerate work per config pair, split auto-tie (same twin) vs needs-judging
    tasks, autoties, plan = [], [], {}
    for l0, l1 in itertools.combinations(sorted(configs), 2):
        same = diff = 0
        for qid in common:
            a, b = match_at(configs[l0], qid), match_at(configs[l1], qid)
            if a == b:
                autoties.append((l0, l1, qid))
                same += 1
            else:
                opt1_label, opt2_label = presentation(qid, l0, l1)
                opt1 = a if opt1_label == l0 else b
                opt2 = a if opt2_label == l0 else b
                tasks.append((l0, l1, qid, opt1, opt2, opt1_label, opt2_label))
                diff += 1
        plan[(l0, l1)] = (same, diff)
        print(
            f"  {l0:>10} vs {l1:<10}: {same:>3} same twin (auto-tie), {diff:>3} disagree -> judge"
        )

    if args.dry_run:
        total = sum(d for _, d in plan.values())
        print(
            f"\n[dry-run] {total} judge calls needed ({len(autoties)} auto-ties skipped). No API spend."
        )
        for l0, l1 in plan:
            ex = [t for t in tasks if t[0] == l0 and t[1] == l1][:4]
            for _, _, qid, o1, o2, ol1, ol2 in ex:
                print(
                    f"    [{l0} vs {l1}] {city.get(qid, '?')}: cand1={city.get(o1, '?')}({ol1}) "
                    f"cand2={city.get(o2, '?')}({ol2})"
                )
        return

    print(f"\njudging {len(tasks)} disagreements (judge={args.judge_model}, effort={args.effort})")
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                compare,
                l0,
                l1,
                qid,
                o1,
                o2,
                leads,
                model=args.judge_model,
                effort=args.effort,
                force=args.force,
            ): (l0, l1, qid, o1, o2, ol1, ol2)
            for (l0, l1, qid, o1, o2, ol1, ol2) in tasks
        }
        for fut in as_completed(futs):
            l0, l1, qid, o1, o2, ol1, ol2 = futs[fut]
            verdict = fut.result()
            rows.append(
                {
                    "pair": f"{l0}__{l1}",
                    "l0": l0,
                    "l1": l1,
                    "home_qid": qid,
                    "home_city": city.get(qid, "?"),
                    "home_group": grp.get(qid, "?"),
                    "opt1_label": ol1,
                    "opt1_city": city.get(o1, "?"),
                    "opt1_qid": o1,
                    "opt2_label": ol2,
                    "opt2_city": city.get(o2, "?"),
                    "opt2_qid": o2,
                    "winner": winner_label(verdict, ol1, ol2),
                    "reason": verdict["reason"],
                    "auto_tie": False,
                }
            )
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} judged")
    for l0, l1, qid in autoties:
        a = match_at(configs[l0], qid)
        rows.append(
            {
                "pair": f"{l0}__{l1}",
                "l0": l0,
                "l1": l1,
                "home_qid": qid,
                "home_city": city.get(qid, "?"),
                "home_group": grp.get(qid, "?"),
                "opt1_label": l0,
                "opt1_city": city.get(a, "?"),
                "opt1_qid": a,
                "opt2_label": l1,
                "opt2_city": city.get(a, "?"),
                "opt2_qid": a,
                "winner": "tie",
                "reason": "same twin",
                "auto_tie": True,
            }
        )

    df = pd.DataFrame(rows)
    report(df, list(configs))

    if args.spot_check:
        spot_check(df, leads, args.spot_check)

    labels = "_vs_".join(sorted(configs))
    out_path = PROCESSED / f"pair_judgments_{labels}.parquet"
    write_df(df, out_path)
    print(f"\nWrote {len(df)} comparisons -> {out_path.name}")


def report(df: pd.DataFrame, labels: list[str]) -> None:
    print(
        f"\n########## head-to-head: {len(df)} comparisons "
        f"({int((~df['auto_tie']).sum())} judged, {int(df['auto_tie'].sum())} auto-tie) ##########"
    )
    wins = {lab: 0 for lab in labels}
    decided = {lab: 0 for lab in labels}
    print("\n  pairwise (decided = non-tie):")
    for (l0, l1), sub in df.groupby(["l0", "l1"]):
        w0 = int((sub["winner"] == l0).sum())
        w1 = int((sub["winner"] == l1).sum())
        ties = int((sub["winner"] == "tie").sum())
        tot = w0 + w1
        rate = f"{w0}/{tot} ({w0 / tot:.0%})" if tot else "n/a"
        print(f"    {l0:>10} vs {l1:<10}: {l0} wins {rate}, {l1} wins {w1}, ties {ties}")
        wins[l0] += w0
        wins[l1] += w1
        decided[l0] += tot
        decided[l1] += tot
    print("\n  overall win-rate over all decided comparisons (higher = picks better twins):")
    rank = sorted(labels, key=lambda lab: -(wins[lab] / decided[lab] if decided[lab] else 0))
    for lab in rank:
        wr = wins[lab] / decided[lab] if decided[lab] else float("nan")
        print(f"    {lab:>10}: {wr:.0%}  ({wins[lab]}/{decided[lab]} decided)")

    print("\n  by home continent (does one model win more for EU vs NA sources?):")
    for g, sub in df[~df["auto_tie"]].groupby("home_group"):
        line = ", ".join(f"{lab} {int((sub['winner'] == lab).sum())}" for lab in labels)
        print(f"    {g:>14}: {line}  (ties {int((sub['winner'] == 'tie').sum())})")

    print("\n  sample decisions:")
    for r in df[~df["auto_tie"]].head(12).itertuples(index=False):
        print(
            f"    [{r.home_city}] won by {r.winner}: {r.opt1_city}({r.opt1_label}) vs "
            f"{r.opt2_city}({r.opt2_label}) -- {r.reason}"
        )


def spot_check(df: pd.DataFrame, leads: dict, k: int) -> None:
    """Re-judge a deterministic sample of DECIDED comparisons with Opus, same candidate order, and
    report how often the pricier judge agrees with the cheap one -- a cheap trust check on the gate."""
    decided = df[~df["auto_tie"]]
    if decided.empty:
        return
    samp = decided.sort_values(
        "home_qid", key=lambda s: s.map(lambda q: hashlib.md5(q.encode()).hexdigest())
    ).head(k)
    print(f"\n  Opus spot-check on {len(samp)} decided comparisons (agreement with {JUDGE_MODEL}):")
    agree = 0
    for r in samp.itertuples(index=False):
        v = judge(
            leads[r.home_qid],
            leads[r.opt1_qid],
            leads[r.opt2_qid],
            model=SPOT_MODEL,
            effort="medium",
        )
        opus = winner_label(v, r.opt1_label, r.opt2_label)
        ok = opus == r.winner
        agree += ok
        print(
            f"    {r.home_city}: {JUDGE_MODEL}->{r.winner}  {SPOT_MODEL}->{opus}  {'OK' if ok else 'DIFFER'}"
        )
    print(f"    agreement: {agree}/{len(samp)}")


if __name__ == "__main__":
    main()
