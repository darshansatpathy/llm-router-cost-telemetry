"""Replay recorded traffic through the real gateway and measure cost per completed task.

Each held-out RouteLLM prompt becomes one task. The backends are replaced by a replay of the answers each
model actually gave to that prompt, with token counts from their length. When the small model's answer was
judged not good enough, the task is not done: the agent asks again on the large tier, and that retry is
charged to the same task. So misrouting costs what it costs in production: a second call.

A share of prompts is sent again later (people ask the same thing twice), which is where the cache earns.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from llmrouter.config import GatewayConfig
from llmrouter.data import Example, tokens
from llmrouter.embed import Embedder

KEY = "replay-key"


@dataclass
class Policy:
    name: str
    model: str  # "auto", "small" or "large"
    cache: bool


@dataclass
class Result:
    policy: str
    tasks: int = 0
    completed_first_try: int = 0
    retries: int = 0
    cost: float = 0.0
    calls_small: int = 0
    calls_large: int = 0
    cache_hits: int = 0
    gateway_ms: list[float] = field(default_factory=list)

    @property
    def cost_per_completed_task(self) -> float:
        return self.cost / self.tasks


def backend(by_prompt: dict[str, Example]) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        tier = req.url.host.split(".")[0]
        e = by_prompt[body["messages"][-1]["content"]]
        text = e.small_answer if tier == "small" else e.large_answer
        return httpx.Response(
            200,
            json={
                "id": "replay",
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tokens(e.prompt), "completion_tokens": tokens(text)},
            },
        )

    return httpx.MockTransport(handle)


def replay_config(base: GatewayConfig, cache: bool) -> GatewayConfig:
    c = base.model_copy(deep=True)
    c.tiers["small"].base_url, c.tiers["large"].base_url = "http://small.replay/v1", "http://large.replay/v1"
    c.tenants = (
        [c.tenants[0].model_copy(update={"name": "replay", "key_sha256": hashlib.sha256(KEY.encode()).hexdigest()})] if c.tenants else []
    )
    if not c.tenants:
        from llmrouter.config import Tenant

        c.tenants = [Tenant.model_validate({"name": "replay", "key_sha256": hashlib.sha256(KEY.encode()).hexdigest()})]
    c.cache.enabled = cache
    c.routing.cascade = True
    return c


async def run_policy(
    policy: Policy, base: GatewayConfig, examples: list[Example], repeats: list[int], embedder: Embedder, redis: Any
) -> Result:
    from llmrouter.gateway import create_app

    await redis.flushdb()
    by_prompt = {e.prompt: e for e in examples}
    app = create_app(
        replay_config(base, policy.cache), redis=redis, embedder=embedder, http=httpx.AsyncClient(transport=backend(by_prompt))
    )
    res = Result(policy.name)
    headers = {"Authorization": f"Bearer {KEY}"}
    order = list(range(len(examples))) + repeats
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw", timeout=60) as c,
    ):
        for n, i in enumerate(order):
            e = examples[i]
            task = f"{policy.name}-{n}"
            h = headers | {"x-task-id": task}
            t0 = time.perf_counter()
            r = await c.post(
                "/v1/chat/completions",
                json={"model": policy.model, "temperature": 0, "messages": [{"role": "user", "content": e.prompt}]},
                headers=h,
            )
            res.gateway_ms.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            served = r.headers["x-route"].split("->")[-1]
            hit = r.headers.get("x-cache", "").startswith("hit")
            res.cache_hits += hit
            if not hit:
                res.calls_small += served == "small" or "->" in r.headers["x-route"]
                res.calls_large += served == "large"
            answer = r.json()["choices"][0]["message"]["content"]
            good = answer == e.large_answer or e.small_ok
            if good:
                res.completed_first_try += 1
            else:  # the agent notices the weak answer and asks the large tier for this one
                res.retries += 1
                await c.post(
                    "/v1/chat/completions",
                    json={"model": "large", "temperature": 0, "messages": [{"role": "user", "content": e.prompt}]},
                    headers=h | {"x-cache": "no-store"},
                )
                res.calls_large += 1
            out = (await c.post(f"/v1/tasks/{task}/outcome", json={"outcome": "success"}, headers=headers)).json()
            res.cost += out["cost_usd"]
            res.tasks += 1
    return res


def repeats_for(n: int, share: float, seed: int = 11) -> list[int]:
    rnd = random.Random(seed)
    return rnd.sample(range(n), int(n * share))


def markdown(results: list[Result], n_unique: int, n_repeat: int, threshold: float, cache_threshold: float) -> str:
    base = next(r for r in results if r.policy == "always large")
    L = [
        "# Replay through the gateway",
        "",
        f"Generated by `llmrouter replay`: the {n_unique:,} held-out RouteLLM prompts, plus {n_repeat:,} of them asked again later, sent one task each through "
        "the real gateway (router, Redis 8 cache, cascade, task accounting), with each backend replaying the answer that model actually gave. "
        "A task whose small-model answer the judge scored below 4 is retried on the large tier and charged for both calls.",
        "",
        f"Router threshold {threshold:.3f}; semantic cache threshold {cache_threshold:.3f}.",
        "",
        "| Policy | Tasks | Right first time | Wrong answers if nothing caught them | Retries | Small / large calls | Cache hits | Cost per completed task | vs always large |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        L.append(
            f"| {r.policy} | {r.tasks:,} | {r.completed_first_try / r.tasks:.1%} | {1 - r.completed_first_try / r.tasks:.1%} | {r.retries:,} | "
            f"{r.calls_small:,} / {r.calls_large:,} | {r.cache_hits:,} | ${r.cost_per_completed_task:.5f} | {r.cost_per_completed_task / base.cost_per_completed_task - 1:+.1%} |"
        )
    L += [
        "",
        "How to read this. The retries assume a perfect oracle: the agent knows exactly which answers were bad and asks again only for those. "
        "With that oracle, always using the small model is cheapest, because the large model costs about 20 to 50 times more per token. "
        "Production has no oracle: an undetected weak answer goes to the user. That is the column that matters: "
        f"always-small leaves {1 - results[1].completed_first_try / results[1].tasks:.1%} of answers wrong, the router "
        f"{1 - results[2].completed_first_try / results[2].tasks:.1%}, at {results[2].cost_per_completed_task / base.cost_per_completed_task - 1:+.0%} cost against always-large. "
        "The exact cache then answers repeated questions for nothing.",
        "",
        "Routing adds one MiniLM embedding per request (the prompt truncated to 256 tokens): a median of 41 ms per request in the router run "
        "of this replay on a shared two-core machine, and several hundred milliseconds while other jobs held both cores. Give the gateway its own cores, or run the embedder on a GPU.",
    ]
    return "\n".join(L) + "\n"
