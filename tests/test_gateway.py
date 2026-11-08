from __future__ import annotations

import pytest
from conftest import auth, chat

from llmrouter import telemetry

pytestmark = pytest.mark.anyio


async def test_routes_easy_to_small_and_hard_to_large(gateway, backend):
    r = await gateway.post("/v1/chat/completions", json=chat("What is the capital of France?"), headers=auth())
    assert r.status_code == 200 and r.headers["x-route"] == "small" and float(r.headers["x-router-p-small-ok"]) > 0.5
    r = await gateway.post(
        "/v1/chat/completions", json=chat("Prove step by step that the integral of x^2 = x^3/3 and analyze the trade-offs"), headers=auth()
    )
    assert r.headers["x-route"] == "large"
    assert [t for t, _ in backend.calls] == ["small", "large"]
    assert backend.calls[0][1]["model"] == "small-model"


async def test_pinned_tier_and_bad_model(gateway, backend):
    r = await gateway.post("/v1/chat/completions", json=chat("hi", model="large"), headers=auth())
    assert r.headers["x-route"] == "large" and "x-router-p-small-ok" not in r.headers
    assert (await gateway.post("/v1/chat/completions", json=chat("hi", model="gpt-x"), headers=auth())).status_code == 422


async def test_auth(gateway):
    assert (await gateway.post("/v1/chat/completions", json=chat("hi"))).status_code == 401
    assert (await gateway.post("/v1/chat/completions", json=chat("hi"), headers={"Authorization": "Bearer nope"})).status_code == 401


async def test_cascade_escalates_failed_small_answers(gateway, backend):
    backend.script = {"refuse me": "I'm sorry, I can't help with that.", "cut me": "__truncate__", "json please": "not json at all"}
    r = await gateway.post("/v1/chat/completions", json=chat("refuse me"), headers=auth())
    assert r.headers["x-route"] == "small->large"
    r = await gateway.post("/v1/chat/completions", json=chat("cut me"), headers=auth())
    assert r.headers["x-route"] == "small->large"
    r = await gateway.post("/v1/chat/completions", json=chat("json please", response_format={"type": "json_object"}), headers=auth())
    assert r.headers["x-route"] == "small->large"
    # Both calls are paid for.
    assert float(r.headers["x-cost-usd"]) > 0


async def test_backend_outage_on_small_falls_back_to_large(gateway, backend):
    backend.fail_tier = "small"
    r = await gateway.post("/v1/chat/completions", json=chat("hello there"), headers=auth())
    assert r.status_code == 200 and r.headers["x-route"] == "small->large"
    backend.fail_tier = "large"
    r = await gateway.post("/v1/chat/completions", json=chat("Prove that 2 + 2 = 4 step by step", model="large"), headers=auth())
    assert r.status_code == 502


async def test_exact_and_semantic_cache(gateway, backend):
    body = chat("What is the capital of France?")
    first = await gateway.post("/v1/chat/completions", json=body, headers=auth())
    assert first.headers["x-cache"] == "miss"
    again = await gateway.post("/v1/chat/completions", json=body, headers=auth())
    assert again.headers["x-cache"] == "hit-exact" and again.headers["x-cost-usd"] == "0"
    similar = await gateway.post("/v1/chat/completions", json=chat("what is the capital of france"), headers=auth())
    assert similar.headers["x-cache"] in ("hit-exact", "hit-semantic")
    assert len(backend.calls) == 1
    assert again.json() == first.json()


async def test_cache_is_scoped_and_narrow(gateway, backend):
    body = chat("Summarise our refund policy.")
    await gateway.post("/v1/chat/completions", json=body, headers=auth())
    # another tenant, another system prompt, a non-zero temperature, tools, multi-turn, or an opt-out: no hit
    variants = [
        (chat("Summarise our refund policy."), auth("globex")),
        (chat("Summarise our refund policy.", system="You are a pirate."), auth()),
        (chat("Summarise our refund policy.") | {"temperature": 0.7}, auth()),
        (chat("Summarise our refund policy.") | {"tools": [{"type": "function", "function": {"name": "f"}}]}, auth()),
        (body, auth(**{"x-cache": "no-store"})),
    ]
    for b, h in variants:
        r = await gateway.post("/v1/chat/completions", json=b, headers=h)
        assert not r.headers["x-cache"].startswith("hit"), (b, h)
    multi = {
        "model": "auto",
        "temperature": 0,
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "Summarise our refund policy."},
        ],
    }
    assert (await gateway.post("/v1/chat/completions", json=multi, headers=auth())).headers["x-cache"] == "miss"


async def test_failed_answers_are_not_cached(gateway, backend):
    backend.script = {"refuse me": "I'm sorry, I can't help with that."}
    await gateway.post("/v1/chat/completions", json=chat("refuse me", model="small"), headers=auth())
    backend.script = {}
    r = await gateway.post("/v1/chat/completions", json=chat("refuse me", model="small"), headers=auth())
    assert r.headers["x-cache"] == "miss"


async def test_purge_is_per_tenant(gateway):
    await gateway.post("/v1/chat/completions", json=chat("Tell me a fact."), headers=auth())
    await gateway.post("/v1/chat/completions", json=chat("Tell me a fact."), headers=auth("globex"))
    assert (await gateway.delete("/v1/cache", headers=auth())).json()["deleted"] >= 1
    assert (await gateway.post("/v1/chat/completions", json=chat("Tell me a fact."), headers=auth())).headers["x-cache"] == "miss"
    assert (await gateway.post("/v1/chat/completions", json=chat("Tell me a fact."), headers=auth("globex"))).headers[
        "x-cache"
    ] == "hit-exact"


async def test_cost_per_task(gateway):
    for q in ("What is DNS?", "Prove step by step that sqrt(2) is irrational and analyze it"):
        await gateway.post("/v1/chat/completions", json=chat(q, temperature=0.5), headers=auth(**{"x-task-id": "task-1"}))
    before = telemetry.TASKS.labels("acme", "success")._value.get()
    r = await gateway.post("/v1/tasks/task-1/outcome", json={"outcome": "success"}, headers=auth())
    out = r.json()
    assert out["calls"] == 2 and out["cost_usd"] > 0
    assert telemetry.TASKS.labels("acme", "success")._value.get() == before + 1
    assert (await gateway.post("/v1/tasks/task-1/outcome", json={"outcome": "failure"}, headers=auth())).status_code == 409
    assert (await gateway.post("/v1/tasks/task-2/outcome", json={"outcome": "nope"}, headers=auth())).status_code == 422


async def test_validation_and_ops(gateway):
    assert (await gateway.post("/v1/chat/completions", json={"model": "auto", "messages": []}, headers=auth())).status_code == 422
    assert (await gateway.post("/v1/chat/completions", json=chat("x", stream=True), headers=auth())).status_code == 422
    assert (await gateway.get("/healthz")).json() == {"status": "ok"}
    assert (await gateway.get("/readyz")).status_code == 200
    text = (await gateway.get("/metrics")).text
    assert "llm_requests_total" in text and "llm_counterfactual_cost_usd_total" in text


async def test_semantic_layer_is_off_unless_both_switches_are_on(config, redis, backend):
    import httpx

    from llmrouter.embed import HashEmbedder
    from llmrouter.gateway import create_app

    config.cache.semantic = False
    app = create_app(config, redis=redis, embedder=HashEmbedder(), http=httpx.AsyncClient(transport=httpx.MockTransport(backend.handler)))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        await c.post("/v1/chat/completions", json=chat("What is the capital of France?"), headers=auth())
        r = await c.post("/v1/chat/completions", json=chat("what is the capital of france?!"), headers=auth())
        assert r.headers["x-cache"] == "miss"
        assert await redis.keys("llmc:*") == []
