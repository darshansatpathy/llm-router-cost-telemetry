"""Train and evaluate the router on RouteLLM data, and write the router file and the report.

Split 70/10/20 (seeded): the router is trained on 70%, its threshold chosen on 10% for a target quality,
and every number in the report is measured on the last 20%, which nothing was tuned on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llmrouter import router as R
from llmrouter.data import Example, tokens
from llmrouter.embed import Embedder


@dataclass
class Prices:
    small_in: float = 0.60
    small_out: float = 0.60
    large_in: float = 10.00
    large_out: float = 30.00

    def small(self, e: Example) -> float:
        return (tokens(e.prompt) * self.small_in + tokens(e.small_answer) * self.small_out) / 1e6

    def large(self, e: Example) -> float:
        return (tokens(e.prompt) * self.large_in + tokens(e.large_answer) * self.large_out) / 1e6


def split(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.7 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def outcome(p: np.ndarray, small_ok: np.ndarray, c_small: np.ndarray, c_large: np.ndarray, t: float) -> dict[str, float]:
    to_small = p >= t
    quality = float(np.mean(np.where(to_small, small_ok, 1.0)))
    cost = float(np.sum(np.where(to_small, c_small, c_large)))
    return {"threshold": t, "share_small": float(to_small.mean()), "quality": quality, "cost": cost}


def curve(p: np.ndarray, small_ok: np.ndarray, c_small: np.ndarray, c_large: np.ndarray) -> list[dict[str, float]]:
    """Quality and cost when the top q% most-confident requests go to the small model, for q = 0..100."""
    order = np.argsort(-p, kind="stable")
    out = []
    for q in range(0, 101, 5):
        k = round(len(p) * q / 100)
        to_small = np.zeros(len(p), bool)
        to_small[order[:k]] = True
        out.append(
            {
                "share_small": q / 100,
                "quality": float(np.mean(np.where(to_small, small_ok, 1.0))),
                "cost": float(np.sum(np.where(to_small, c_small, c_large))),
            }
        )
    return out


def apgr(c: list[dict[str, float]], q_small: float) -> float:
    """Average performance gap recovered (RouteLLM): how much of the gap between all-small and all-large the router keeps, averaged over shares."""
    gaps = [(row["quality"] - q_small) / (1.0 - q_small) for row in c]
    # trapezoid over the share axis
    return float(sum((gaps[i] + gaps[i + 1]) / 2 for i in range(len(gaps) - 1)) / (len(gaps) - 1))


def run(
    examples: list[Example], embedder: Embedder, *, target_quality: float = 0.95, prices: Prices | None = None, seed: int = 7
) -> tuple[R.Router, dict[str, Any]]:
    prices = prices or Prices()
    prompts = [e.prompt for e in examples]
    x = R.features(embedder, prompts)
    y = np.array([e.small_ok for e in examples], dtype=np.float32)
    cs = np.array([prices.small(e) for e in examples])
    cl = np.array([prices.large(e) for e in examples])
    tr, va, te = split(len(examples), seed)

    best_c, best_ll = 1.0, math.inf
    for c in (0.1, 0.3, 1.0, 3.0):
        coef, b = R.train(x[tr], y[tr], c)
        p = 1 / (1 + np.exp(-(x[va] @ coef + b)))
        ll = float(-np.mean(y[va] * np.log(p + 1e-9) + (1 - y[va]) * np.log(1 - p + 1e-9)))
        if ll < best_ll:
            best_c, best_ll = c, ll
    coef, b = R.train(x[tr], y[tr], best_c)
    shape_coef, shape_b = R.train(x[tr][:, -len(R.SHAPE) :], y[tr], best_c)

    def prob(ix: np.ndarray) -> np.ndarray:
        out: np.ndarray = 1 / (1 + np.exp(-(x[ix] @ coef + b)))
        return out

    # Threshold: the lowest that keeps validation quality at the target.
    pv = prob(va)
    candidates = sorted(set(np.round(pv, 4)))
    threshold = 1.0
    for t in candidates:
        if outcome(pv, y[va], cs[va], cl[va], float(t))["quality"] >= target_quality:
            threshold = float(t)
            break

    pt = prob(te)
    rng = np.random.default_rng(seed)
    test = {
        "router": curve(pt, y[te], cs[te], cl[te]),
        "shape_only": curve(1 / (1 + np.exp(-(x[te][:, -len(R.SHAPE) :] @ shape_coef + shape_b))), y[te], cs[te], cl[te]),
        "shorter_first": curve(-np.array([len(examples[i].prompt) for i in te], float), y[te], cs[te], cl[te]),
        "random": curve(rng.random(len(te)), y[te], cs[te], cl[te]),
    }
    q_small = float(y[te].mean())
    chosen = outcome(pt, y[te], cs[te], cl[te], threshold)
    all_large = float(cl[te].sum())
    from sklearn.metrics import roc_auc_score

    results: dict[str, Any] = {
        "n": {"train": len(tr), "validation": len(va), "test": len(te)},
        "regularisation_c": best_c,
        "test_auc": float(roc_auc_score(y[te], pt)),
        "q_small_alone": q_small,
        "apgr": {k: apgr(v, q_small) for k, v in test.items()},
        "curves": test,
        "target_quality": target_quality,
        "threshold": threshold,
        "chosen": chosen | {"cost_all_large": all_large, "cost_all_small": float(cs[te].sum()), "saving": 1 - chosen["cost"] / all_large},
        "by_source": by_source(examples, te, pt >= threshold, y),
        "prices": prices.__dict__,
    }
    meta = {
        "trained_on": "routellm/gpt4_dataset valid.jsonl, 70% split",
        "accept_score": 4,
        "target_quality": target_quality,
        "test_auc": results["test_auc"],
    }
    return R.Router(coef, b, threshold, embedder.name, meta), results


def by_source(examples: list[Example], te: np.ndarray, to_small: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    srcs = np.array([examples[i].source for i in te])
    for s in sorted(set(srcs)):
        m = srcs == s
        if m.sum() < 40:
            continue
        out[s] = {
            "n": int(m.sum()),
            "small_ok": float(y[te][m].mean()),
            "routed_small": float(to_small[m].mean()),
            "quality": float(np.where(to_small[m], y[te][m], 1).mean()),
        }
    return out


def markdown(r: dict[str, Any]) -> str:
    c = r["chosen"]
    p = r["prices"]
    L = [
        "# Routing evaluation",
        "",
        f"RouteLLM `gpt4_dataset` (Apache-2.0): {sum(r['n'].values()):,} real prompts with a GPT-4 answer, a Mixtral 8x7B answer and a GPT-4 judge's 1-5 score "
        f"of the Mixtral answer. A score of 4 or more counts as good enough; GPT-4's answer counts as always good enough. "
        f"Trained on {r['n']['train']:,}, threshold chosen on {r['n']['validation']:,}, **every number below measured on the {r['n']['test']:,} held-out prompts**.",
        "",
        f"On its own the small model is good enough for **{r['q_small_alone']:.1%}** of prompts. The router's job is to find the rest.",
        "",
        "## At the chosen threshold",
        "",
        f"Target: keep quality at {r['target_quality']:.0%} of always using the large model. Threshold {r['threshold']:.3f} on the router's probability that the small model will do.",
        "",
        "| | Share to small model | Quality | Cost | Saving vs all-large |",
        "|---|---|---|---|---|",
        f"| Always large | 0% | 100.0% | ${c['cost_all_large']:.2f} | |",
        f"| **Router** | **{c['share_small']:.1%}** | **{c['quality']:.1%}** | **${c['cost']:.2f}** | **{c['saving']:.1%}** |",
        f"| Always small | 100% | {r['q_small_alone']:.1%} | ${c['cost_all_small']:.2f} | {1 - c['cost_all_small'] / c['cost_all_large']:.1%} |",
        "",
        f"Prices assumed (USD per million tokens): small {p['small_in']:.2f} in / {p['small_out']:.2f} out, large {p['large_in']:.2f} in / {p['large_out']:.2f} out; "
        "tokens estimated as characters / 4 from the actual prompts and answers.",
        "",
        "## Against simpler routers",
        "",
        f"Router test AUC {r['test_auc']:.3f}. APGR is the share of the quality gap between all-small and all-large that a router recovers, "
        "averaged over every split of traffic (RouteLLM's metric; random is 0.5).",
        "",
        "| Router | APGR | Quality with 50% to small | Quality with 80% to small |",
        "|---|---|---|---|",
    ]
    names = {
        "router": "Embedding + shape features (shipped)",
        "shape_only": "Shape features only",
        "shorter_first": "Shortest prompts to small",
        "random": "Random",
    }
    for k, v in r["curves"].items():
        at = {round(row["share_small"], 2): row["quality"] for row in v}
        L.append(f"| {names[k]} | {r['apgr'][k]:.3f} | {at[0.5]:.1%} | {at[0.8]:.1%} |")
    L += ["", "## Quality and cost across the whole range (test set)", "", "| Share to small | Quality | Cost |", "|---|---|---|"]
    L += [
        f"| {row['share_small']:.0%} | {row['quality']:.1%} | ${row['cost']:.2f} |"
        for row in r["curves"]["router"]
        if round(row["share_small"] * 100) % 10 == 0
    ]
    L += [
        "",
        "## By source of prompt (test set, sources with 40+ prompts)",
        "",
        "| Source | Prompts | Small model good enough | Routed to small | Quality |",
        "|---|---|---|---|---|",
    ]
    L += [
        f"| {s} | {v['n']} | {v['small_ok']:.0%} | {v['routed_small']:.0%} | {v['quality']:.1%} |"
        for s, v in sorted(r["by_source"].items(), key=lambda kv: -kv[1]["n"])
    ]
    L += [
        "",
        "## Limits",
        "",
        "- The judge is a model, not a person, and the labels are for one pair of models. Retrain on your own traffic and your own models' graded answers before trusting the threshold.",
        '- Quality here is "good enough by the judge", not task success. The proxy\'s cost per completed task metric is the production check.',
    ]
    return "\n".join(L) + "\n"


def save_results(r: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(r, indent=1))
