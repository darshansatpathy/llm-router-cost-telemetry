"""The training, calibration and replay code, on small synthetic data with the hashing embedder."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from llmrouter import cache_eval, data, replay, routing_eval
from llmrouter.data import Example
from llmrouter.embed import HashEmbedder

EASY = ["What is the capital of {}?", "Translate '{}' into French.", "Give me a synonym for {}."]
HARD = [
    "Prove step by step that the series for {} converges and analyze the trade-offs.",
    "Design an architecture to optimize {} and compare three designs.",
]
WORDS = ["Peru", "happy", "graph", "river", "token", "cache", "Kenya", "orbit", "maple", "solar"]


def synthetic(n: int = 400, seed: int = 0) -> list[Example]:
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        hard = rnd.random() < 0.3
        prompt = rnd.choice(HARD if hard else EASY).format(rnd.choice(WORDS)) + " " + str(rnd.randint(0, 10**6))
        ok = (rnd.random() < 0.35) if hard else (rnd.random() < 0.97)
        out.append(Example(prompt, "synthetic", 5 if ok else 2, "short answer", "a much longer and more careful answer " * 3))
    return out


def test_data_load_and_tokens(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [
        {"prompt": "hi", "source": ["x"], "mixtral_score": 4, "mixtral_response": "a", "gpt4_response": "b"},
        {"prompt": "", "source": ["x"], "mixtral_score": 4, "mixtral_response": "a", "gpt4_response": "b"},
        {"prompt": "q", "source": None, "mixtral_score": None, "mixtral_response": "a", "gpt4_response": "b"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    (e,) = data.load(p)
    assert e.small_ok and e.source == "x" and data.tokens("abcdefgh") == 2


def test_router_training_finds_the_hard_prompts():
    router, res = routing_eval.run(synthetic(), HashEmbedder(), target_quality=0.95)
    assert res["apgr"]["router"] > res["apgr"]["random"]
    assert res["chosen"]["quality"] >= 0.85 and 0 < res["chosen"]["share_small"] < 1 and res["chosen"]["saving"] > 0
    assert 0 < router.threshold <= 1 and len(router.coef) == 384 + 6
    text = routing_eval.markdown(res)
    assert "## At the chosen threshold" in text and "| Random |" in text


def test_curve_ends_at_all_small_and_all_large():
    p = np.array([0.9, 0.1, 0.5])
    ok = np.array([1.0, 0.0, 1.0])
    c = routing_eval.curve(p, ok, np.ones(3), np.full(3, 10.0))
    assert c[0]["quality"] == 1.0 and c[0]["cost"] == 30 and c[-1]["quality"] == pytest.approx(2 / 3) and c[-1]["cost"] == 3
    assert routing_eval.apgr(c, 2 / 3) > 0.5


def test_cache_calibration_picks_the_safe_threshold():
    rng = np.random.default_rng(0)
    label = np.array([1] * 500 + [0] * 500)
    sim = np.concatenate([rng.uniform(0.95, 1.0, 500), rng.uniform(0.80, 0.97, 500)])
    r = cache_eval.calibrate(sim, label, max_false_hit=0.01)
    assert r["chosen"]["false_hit_rate"] <= 0.01 and r["chosen"]["threshold"] >= 0.96
    assert "Chosen threshold" in cache_eval.markdown(r, "hash")
    sims = cache_eval.similarities(HashEmbedder(), ["a b c"], ["a b c"])
    assert sims[0] == pytest.approx(1.0)


@pytest.mark.anyio
async def test_replay_through_the_gateway(config, redis):
    ex = synthetic(60, seed=3)
    reps = replay.repeats_for(len(ex), 0.25)
    results = []
    for pol in (
        replay.Policy("always large", "large", False),
        replay.Policy("always small", "small", False),
        replay.Policy("router", "auto", False),
        replay.Policy("router + cache", "auto", True),
    ):
        results.append(await replay.run_policy(pol, config, ex, reps, HashEmbedder(), redis))
    large, small, router, cached = results
    assert all(r.tasks == 75 for r in results)
    assert large.completed_first_try == 75 and small.retries > 0
    assert cached.cache_hits > 0 and cached.cost <= router.cost
    text = replay.markdown(results, 60, 15, 0.5, 0.995)
    assert "| router + cache |" in text and "Wrong answers if nothing caught them" in text


def test_source_names():
    assert data.source_name("lmsys-chat-1m") == "lmsys-chat-1m" and data.source_name("x-hh") == "hh-rlhf"
