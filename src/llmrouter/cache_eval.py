"""Calibrate the semantic cache threshold on PAWS (Zhang et al., 2019; data from Google LLC).

PAWS pairs sentences with heavy word overlap; about half are paraphrases and half mean something
different ("flights from New York to Florida" vs "flights from Florida to New York"). That is the
failure a semantic cache must not make: serving one question's answer to a different question.
We pick the lowest threshold whose false-hit rate on non-paraphrases is at most the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llmrouter.embed import Embedder

URL = "https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/test-00000-of-00001.parquet"


def download(dest: Path) -> Path:
    import httpx

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(URL, follow_redirects=True, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def load(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(path).to_pydict()
    return t["sentence1"], t["sentence2"], np.array(t["label"], dtype=int)


def similarities(embedder: Embedder, a: list[str], b: list[str]) -> np.ndarray:
    ea, eb = embedder.embed(a), embedder.embed(b)
    return np.sum(ea * eb, axis=1)


def calibrate(sim: np.ndarray, label: np.ndarray, max_false_hit: float = 0.01) -> dict[str, Any]:
    rows = []
    grid = list(np.arange(0.80, 0.98, 0.02)) + list(np.arange(0.98, 1.0001, 0.002))
    for t in np.round(grid, 3):
        hit = sim >= t
        fh = float(hit[label == 0].mean())
        rec = float(hit[label == 1].mean())
        prec = float(label[hit].mean()) if hit.any() else 1.0
        rows.append({"threshold": float(t), "false_hit_rate": fh, "paraphrase_recall": rec, "precision": prec})
    ok = [r for r in rows if r["false_hit_rate"] <= max_false_hit]
    chosen = ok[0] if ok else rows[-1]
    return {"pairs": len(label), "paraphrases": int(label.sum()), "max_false_hit": max_false_hit, "chosen": chosen, "sweep": rows}


def markdown(r: dict[str, Any], embedder: str) -> str:
    c = r["chosen"]
    L = [
        "# Semantic cache calibration",
        "",
        f"PAWS test set (Zhang, Baldridge and He, 2019; data from Google LLC): {r['pairs']:,} sentence pairs with heavy word overlap, "
        f"{r['paraphrases']:,} of them paraphrases and the rest different in meaning. Embedder `{embedder}`, cosine similarity.",
        "",
        f"**Chosen threshold {c['threshold']:.3f}**: the lowest whose false-hit rate on pairs that mean different things is at most {r['max_false_hit']:.0%}. "
        f"At it, {c['false_hit_rate']:.2%} of different-meaning pairs would be served the other's answer, and {c['paraphrase_recall']:.1%} of true paraphrases hit.",
        "",
        "| Threshold | False hits on different meanings | Paraphrases that hit | Precision of hits |",
        "|---|---|---|---|",
    ]
    L += [
        f"| {row['threshold']:.3f} | {row['false_hit_rate']:.2%} | {row['paraphrase_recall']:.1%} | {row['precision']:.1%} |"
        for row in r["sweep"]
    ]
    L += [
        "",
        'What this means: a general-purpose sentence embedder scores "flights from Paris to Rome" and "flights from Rome to Paris" as near-identical. '
        "No threshold below exact equality keeps wrong answers rare on pairs like these, so **the semantic layer ships off**. The exact layer (the same "
        "request again) is on and safe. Turn the semantic layer on only for a tenant whose traffic is FAQ-like and where a near-miss answer is harmless, "
        "and calibrate on that tenant's own pairs first.",
    ]
    return "\n".join(L) + "\n"
