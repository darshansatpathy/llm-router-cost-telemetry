"""Decide, per request, whether the small model is likely to answer well enough.

The router is a logistic regression over the prompt's embedding and a few cheap shape features. It
predicts the probability that the small model's answer is acceptable; a request goes to the small model
when that probability clears a threshold. The threshold is the business decision: it is chosen on held-out
data for a target quality, and stored with the weights. Weights are plain JSON, never a pickle.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llmrouter.embed import Embedder

CODE = re.compile(r"```|\bdef |\bclass |#include|\bfunction\b|;\s*$|\bSELECT\b|\bimport\b", re.M | re.I)
MATH = re.compile(r"[=+\-*/^]\s*\d|\d\s*[=+\-*/^]|\b(prove|integral|derivative|equation|solve)\b", re.I)
REASONING = re.compile(r"\b(step by step|explain why|compare|analy[sz]e|design|optimi[sz]e|trade-?offs?|architecture)\b", re.I)
SHAPE = ["log_chars", "lines", "code", "math", "reasoning", "questions"]


def shape_features(prompt: str) -> np.ndarray:
    return np.array(
        [
            math.log1p(len(prompt)) / 10,
            min(prompt.count("\n"), 200) / 50,
            len(CODE.findall(prompt)) > 0,
            min(len(MATH.findall(prompt)), 20) / 5,
            len(REASONING.findall(prompt)) > 0,
            min(prompt.count("?"), 10) / 5,
        ],
        dtype=np.float32,
    )


def features(embedder: Embedder, prompts: list[str]) -> np.ndarray:
    e = embedder.embed([p[:4000] for p in prompts])
    s = np.vstack([shape_features(p) for p in prompts]) if prompts else np.zeros((0, len(SHAPE)), np.float32)
    return np.hstack([e, s]).astype(np.float32)


@dataclass
class Route:
    tier: str  # "small" or "large"
    p_small_ok: float
    threshold: float


@dataclass
class Router:
    coef: np.ndarray
    intercept: float
    threshold: float
    embedder_name: str
    meta: dict[str, Any]

    def probability(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.coef + self.intercept
        out: np.ndarray = 1 / (1 + np.exp(-z))
        return out

    def route(self, embedder: Embedder, prompt: str, threshold: float | None = None) -> Route:
        t = self.threshold if threshold is None else threshold
        p = float(self.probability(features(embedder, [prompt]))[0])
        return Route("small" if p >= t else "large", p, t)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "embedder": self.embedder_name,
                    "threshold": self.threshold,
                    "intercept": self.intercept,
                    "coef": [round(float(c), 6) for c in self.coef],
                    "meta": self.meta,
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: Path) -> Router:
        d = json.loads(path.read_text())
        return cls(np.array(d["coef"], dtype=np.float32), float(d["intercept"]), float(d["threshold"]), d["embedder"], d.get("meta", {}))


def train(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=c, max_iter=2000)
    m.fit(x, y)
    return m.coef_[0].astype(np.float32), float(m.intercept_[0])
