from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from redis.asyncio import Redis

from llmrouter.config import GatewayConfig
from llmrouter.embed import HashEmbedder
from llmrouter.router import SHAPE, Router

REDIS_URL = os.environ.get("LLMROUTER_TEST_REDIS", "redis://127.0.0.1:16379/0")
KEYS = {"acme": "key-acme", "globex": "key-globex"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def router_file(tmp_path: Path) -> Path:
    """A router that sends anything mentioning 'prove' or long prompts to the large tier (via the shape features)."""
    coef = np.zeros(384 + len(SHAPE), dtype=np.float32)
    coef[384 + SHAPE.index("math")] = -4.0
    coef[384 + SHAPE.index("reasoning")] = -6.0
    coef[384 + SHAPE.index("log_chars")] = -2.0
    p = tmp_path / "router.json"
    Router(coef, 3.0, 0.5, "hash-384", {}).save(p)
    return p


class FakeBackend:
    """An OpenAI-compatible backend per tier. `script` maps a prompt substring to a canned answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.script: dict[str, str] = {}
        self.fail_tier: str | None = None

    def handler(self, req: httpx.Request) -> httpx.Response:
        tier = req.url.host.split(".")[0]
        body = json.loads(req.content)
        self.calls.append((tier, body))
        if self.fail_tier == tier:
            return httpx.Response(503, json={"error": "overloaded"})
        prompt = body["messages"][-1]["content"]
        text = next((v for k, v in self.script.items() if k in prompt), f"{tier} answer to: {prompt[:40]}")
        if text == "__truncate__":
            return httpx.Response(200, json=_resp(body["model"], "partial", prompt, finish="length"))
        return httpx.Response(200, json=_resp(body["model"], text, prompt))


def _resp(model: str, text: str, prompt: str, finish: str = "stop") -> dict:
    return {
        "id": "x",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": max(1, len(text) // 4)},
    }


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def config(router_file: Path) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "tiers": {
                "small": {"base_url": "http://small.test/v1", "model": "small-model", "price_in": 0.6, "price_out": 0.6},
                "large": {"base_url": "http://large.test/v1", "model": "large-model", "price_in": 10, "price_out": 30},
            },
            "tenants": [{"name": n, "key_sha256": hashlib.sha256(k.encode()).hexdigest(), "semantic_cache": True} for n, k in KEYS.items()],
            "cache": {"semantic": True, "semantic_threshold": 0.9},
            "routing": {"router_file": str(router_file)},
            "redis_url": REDIS_URL,
            "embedder": "hash",
        }
    )


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    r = Redis.from_url(REDIS_URL)
    try:
        await r.ping()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Redis 8 at {REDIS_URL}: {e}")
    try:
        await r.execute_command("FT._LIST")
    except Exception:  # noqa: BLE001
        pytest.skip("Redis without the query engine (needs Redis 8 or Redis Stack)")
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
async def gateway(config: GatewayConfig, redis: Redis, backend: FakeBackend):
    from llmrouter.gateway import create_app

    app = create_app(config, redis=redis, embedder=HashEmbedder(), http=httpx.AsyncClient(transport=httpx.MockTransport(backend.handler)))
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c


def auth(tenant: str = "acme", **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {KEYS[tenant]}"} | extra


def chat(text: str, model: str = "auto", system: str | None = "You are a helpful assistant.", **kw) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return {"model": model, "messages": msgs, "temperature": 0} | kw


_ = uuid
