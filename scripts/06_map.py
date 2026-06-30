"""Stage 06: interactive UK map of US analogs (Plotly).

Each UK city is plotted at its coordinates; hovering shows its top US analogs (from stage 05)
with similarity, plus the shared-character caption (from stage 07) if available. Dots are sized
by population and colored by the top match's similarity.

--source lead | profile (+ --profile-key) selects which matches to plot. Prefers the captioned
matches file if present.
Reads:  data/processed/matches_<model>[_profile_<key>][_captioned].json + cities.parquet
Writes: output/uk_map_<model>[_profile_<key>].html  (self-contained)
"""

from __future__ import annotations

import argparse
import json
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from _common import PROCESSED, ROOT

MODEL_KEY = "nomic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_KEY)
    ap.add_argument(
        "--source",
        choices=["lead", "profile"],
        default="lead",
        help="lead embeddings or LLM character profiles",
    )
    ap.add_argument("--profile-key", default="haiku", help="distillation key for --source profile")
    args = ap.parse_args()

    suffix = "" if args.source == "lead" else f"_profile_{args.profile_key}"
    captioned = PROCESSED / f"matches_{args.model}{suffix}_captioned.json"
    plain = PROCESSED / f"matches_{args.model}{suffix}.json"
    src = captioned if captioned.exists() else plain
    matches = json.loads(src.read_text())
    pop = pd.read_parquet(PROCESSED / "cities.parquet").set_index("qid")["population"]

    rows, skipped = [], []
    for rec in matches.values():
        if rec["lat"] is None or rec["lon"] is None:
            skipped.append(rec["city"])
            continue
        lines = []
        for k, m in enumerate(rec["matches"]):
            lines.append(f"{k + 1}. {m['city']} ({m['similarity']:.2f})")
            if m.get("caption"):
                lines.append("<i>" + "<br>".join(textwrap.wrap(m["caption"], 46)) + "</i>")
        hover = f"<b>{rec['city']}</b><br>most like:<br>" + "<br>".join(lines)
        rows.append(
            {
                "city": rec["city"],
                "lat": rec["lat"],
                "lon": rec["lon"],
                "pop": int(pop.get(rec["qid"], 1000)),
                "top_sim": rec["matches"][0]["similarity"] if rec["matches"] else 0.0,
                "hover": hover,
            }
        )
    df = pd.DataFrame(rows)
    print(f"source: {src.name}")
    if skipped:
        print(f"skipped {len(skipped)} cities without coords: {', '.join(skipped)}")

    lp = np.log10(df["pop"].clip(lower=1000))
    sizes = 7 + 25 * (lp - lp.min()) / (lp.max() - lp.min() + 1e-9)

    fig = go.Figure(
        go.Scattergeo(
            lon=df["lon"],
            lat=df["lat"],
            text=df["hover"],
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=df["top_sim"],
                colorscale="Viridis",
                colorbar=dict(title="top match<br>similarity"),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
        )
    )
    fig.update_geos(
        fitbounds="locations",
        resolution=50,
        projection_type="mercator",
        showland=True,
        landcolor="rgb(246,246,243)",
        showocean=True,
        oceancolor="rgb(224,235,245)",
        showcountries=True,
        countrycolor="rgb(185,185,185)",
        showcoastlines=True,
        coastlinecolor="rgb(170,170,170)",
        showlakes=False,
    )
    fig.update_layout(
        title=f"UK cities & their most similar US cities (hover) — {args.model}{suffix}",
        height=820,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
    )

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"uk_map_{args.model}{suffix}.html"
    fig.write_html(out_path, include_plotlyjs=True)
    print(f"Wrote {len(df)} cities -> {out_path}")


if __name__ == "__main__":
    main()
