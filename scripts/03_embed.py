"""Stage 03: embed city text with a text-embedding model.

Default model: nomic-embed-text-v1.5 -- 8192-token context (no truncation), fully open
weights+data (reproducible), small and fast. The model is a single config swap
(MODEL_NAME / MODEL_KEY / DOC_PREFIX) so others (e.g. Qwen3) can be added as a robustness
axis; each model+source writes to its own file and never clobbers.

  --source lead     embed Wikipedia leads (cities.parquet)      -> embeddings_<key>.parquet
  --source profile  embed character profiles (profiles.parquet) -> embeddings_<key>_profile.parquet

Writes: identity cols + `model` + `embedding` (L2-normalized, one list[float] per row)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import PROCESSED, write_df

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_KEY = "nomic"
DOC_PREFIX = "search_document: "
ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["lead", "profile"],
        default="lead",
        help="embed Wikipedia leads or LLM character profiles",
    )
    ap.add_argument(
        "--profile-key",
        default="opus",
        help="distillation key for --source profile (e.g. opus, haiku)",
    )
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.source == "lead":
        in_path, text_col, suffix = PROCESSED / "cities.parquet", "lead_text", ""
    else:
        in_path = PROCESSED / f"profiles_{args.profile_key}.parquet"
        text_col, suffix = "profile_text", f"_profile_{args.profile_key}"
    out_path = PROCESSED / f"embeddings_{MODEL_KEY}{suffix}.parquet"

    df = pd.read_parquet(in_path)
    assert len(df) > 0, f"{in_path.name} is empty -- run upstream stages first"

    if out_path.exists() and not args.force:
        prev = pd.read_parquet(out_path)
        if set(prev["qid"]) == set(df["qid"]):
            print(f"{out_path.name} already covers all {len(df)} rows; --force to recompute.")
            return
        print("row set changed since last run; recomputing.")

    from sentence_transformers import SentenceTransformer  # heavy import, defer past cache check

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    texts = [DOC_PREFIX + t for t in df[text_col].tolist()]
    print(f"embedding {len(texts)} {args.source}s (prefix={DOC_PREFIX!r}) ...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    assert emb.shape[0] == len(df), f"row mismatch: {emb.shape[0]} vs {len(df)}"
    assert not np.isnan(emb).any(), "NaNs in embeddings"
    print(f"embeddings: shape={emb.shape}, mean L2 norm={np.linalg.norm(emb, axis=1).mean():.4f}")

    out = df[ID_COLS].copy()
    out["model"] = MODEL_KEY
    out["embedding"] = [row.tolist() for row in emb]
    write_df(out, out_path)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
