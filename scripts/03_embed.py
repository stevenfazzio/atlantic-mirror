"""Stage 03: embed each city's Wikipedia lead with a text-embedding model.

Default model: nomic-embed-text-v1.5 -- 8192-token context (no truncation of long leads),
fully open weights+data (reproducible), small and fast. The model is a single config swap
(MODEL_NAME / MODEL_KEY / DOC_PREFIX) so we can add others (e.g. Qwen3) as a robustness
axis; each model writes to its own file and never clobbers another's vectors.

Reads:  data/processed/cities.parquet
Writes: data/processed/embeddings_<key>.parquet
        identity cols + `model` + `embedding` (L2-normalized, one list[float] per row)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import PROCESSED, write_df

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_KEY = "nomic"
# nomic uses task-instruction prefixes; we treat every city as a document. Our downstream is
# similarity/clustering rather than search, so "clustering: " is worth an A/B -- swap the
# prefix and re-run (it lands in the same per-model file, so use --force or a new MODEL_KEY).
DOC_PREFIX = "search_document: "

ID_COLS = ["country", "rank", "city", "population", "wikipedia_title", "qid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out_path = PROCESSED / f"embeddings_{MODEL_KEY}.parquet"
    cities = pd.read_parquet(PROCESSED / "cities.parquet")
    assert len(cities) > 0, "cities.parquet is empty -- run stages 01 and 02 first"

    if out_path.exists() and not args.force:
        prev = pd.read_parquet(out_path)
        if set(prev["qid"]) == set(cities["qid"]):
            print(f"{out_path.name} already covers all {len(cities)} cities; --force to recompute.")
            return
        print("city set changed since last run; recomputing.")

    from sentence_transformers import SentenceTransformer  # heavy import, defer past cache check

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    texts = [DOC_PREFIX + t for t in cities["lead_text"].tolist()]
    print(f"embedding {len(texts)} leads (prefix={DOC_PREFIX!r}) ...")
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    assert emb.shape[0] == len(cities), f"row mismatch: {emb.shape[0]} vs {len(cities)}"
    assert not np.isnan(emb).any(), "NaNs in embeddings"
    print(f"embeddings: shape={emb.shape}, mean L2 norm={np.linalg.norm(emb, axis=1).mean():.4f}")

    out = cities[ID_COLS].copy()
    out["model"] = MODEL_KEY
    out["embedding"] = [row.tolist() for row in emb]
    write_df(out, out_path)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
