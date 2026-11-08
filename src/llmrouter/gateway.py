"""The gateway: an OpenAI-compatible `/v1/chat/completions` that routes, caches, escalates and accounts.

Send `model: "auto"` to let the router pick the tier, or `"small"` / `"large"` to pin one. Optional headers:
`x-task-id` to attribute spend to a task (then report its outcome to `/v1/tasks/{id}/outcome`), and
`x-cache: no-store` to bypass the cache. Responses carry `x-route`, `x-cache` and `x-cost-usd`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from redis.asyncio import Redis

from llmrouter import __version__, telemetry
from llmrouter.cache import SemanticCache, cacheable, scope
from llmrouter.config import GatewayConfig, Tenant, Tier
from llmrouter.embed import Embedder, get
from llmrouter.router import Router

log = logging.getLogger("llmrouter")
REFUSAL = re.compile(r"^\s*(I'm sorry|I am sorry|I can(?:no|')t (?:help|assist|answer)|As an AI)", re.I)


class Outcome(BaseModel):
    outcome: Literal["success", "failure", "escalated", "abandoned"]


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return str(m["content"])
    return ""


def routing_text(messages: list[dict[str, Any]]) -> str:
    """What the router reads: the last user turn, with a little system context."""
    system = next((m.get("content") for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str)), "")
    return (str(system)[:500] + "\n\n" if system else "") + last_user_text(messages)


def check(body: dict[str, Any], resp: dict[str, Any]) -> str | None:
    """Cheap checks on a small-tier answer; a failure sends the request to the large tier."""
    choice = (resp.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    if choice.get("finish_reason") == "length":
        return "truncated"
    if not text.strip() and not (choice.get("message") or {}).get("tool_calls"):
        return "empty"
    if REFUSAL.match(text):
        return "refusal"
    fmt = (body.get("response_format") or {}).get("type")
    if fmt in ("json_object", "json_schema"):
        try:
            json.loads(text)
        except ValueError:
            return "invalid_json"
    return None


def usage_of(resp: dict[str, Any]) -> tuple[int, int, int]:
    u = resp.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int(cached)


def create_app(
    config: GatewayConfig, *, redis: Redis | None = None, embedder: Embedder | None = None, http: httpx.AsyncClient | None = None
) -> FastAPI:
    emb = embedder or get(config.embedder)
    router = Router.load(Path(config.routing.router_file))
    threshold = config.routing.threshold if config.routing.threshold is not None else router.threshold
    r = redis or Redis.from_url(config.redis_url)
    cache = SemanticCache(r, emb, threshold=config.cache.semantic_threshold, ttl_seconds=config.cache.ttl_seconds)
    client = http or httpx.AsyncClient()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if config.cache.enabled:
            await cache.ensure_index()
        yield
        if http is None:
            await client.aclose()

    app = FastAPI(title="llm-router-cost-telemetry", version=__version__, lifespan=lifespan)
    app.state.cache, app.state.router = cache, router

    async def tenant_dep(authorization: Annotated[str | None, Header()] = None) -> Tenant:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing API key")
        t = config.tenant_for(authorization[7:].strip())
        if t is None:
            raise HTTPException(401, "invalid API key")
        return t

    TENANT = Depends(tenant_dep)

    async def call(tier_name: str, body: dict[str, Any]) -> dict[str, Any]:
        tier: Tier = config.tiers[tier_name]
        headers = {"content-type": "application/json"}
        key = tier.api_key()
        if key:
            headers["authorization"] = f"Bearer {key}"
        try:
            resp = await client.post(
                f"{tier.base_url.rstrip('/')}/chat/completions",
                json=body | {"model": tier.model},
                headers=headers,
                timeout=tier.timeout_seconds,
            )
        except httpx.HTTPError as e:
            telemetry.BACKEND_ERRORS.labels(tier_name, type(e).__name__).inc()
            raise
        if resp.status_code >= 400:
            telemetry.BACKEND_ERRORS.labels(tier_name, str(resp.status_code)).inc()
            resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def account(tenant: str, tier_name: str, resp: dict[str, Any], task_id: str | None) -> float:
        tier = config.tiers[tier_name]
        i, o, cached = usage_of(resp)
        cost = tier.cost(i, o, cached)
        telemetry.COST.labels(tenant, tier_name).inc(cost)
        telemetry.TOKENS.labels(tenant, tier_name, "input").inc(i)
        telemetry.TOKENS.labels(tenant, tier_name, "output").inc(o)
        if task_id:
            key = f"task:{tenant}:{task_id}"
            await r.hincrbyfloat(key, "cost", cost)
            await r.hincrby(key, "calls", 1)
            await r.expire(key, config.task_ttl_seconds)
        return cost

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        tenant: Tenant = TENANT,
        x_task_id: Annotated[str | None, Header()] = None,
        x_cache: Annotated[str | None, Header()] = None,
    ) -> Response:
        started = time.perf_counter()
        body: dict[str, Any] = await request.json()
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(422, "messages required")
        if body.get("stream"):
            raise HTTPException(422, "streaming is not supported by this gateway yet")
        requested = str(body.get("model", "auto"))
        if requested == "auto":
            route = router.route(emb, routing_text(messages), threshold)
            telemetry.ROUTER_P.observe(route.p_small_ok)
            tier_name, p_small = route.tier, route.p_small_ok
        elif requested in config.tiers:
            tier_name, p_small = requested, None
        else:
            raise HTTPException(422, f"model must be auto or one of {sorted(config.tiers)}")

        use_cache = config.cache.enabled and cacheable(body) and (x_cache or "").lower() != "no-store"
        sys_prompt = next((m.get("content") for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str)), "")
        sc = scope(tenant.name, tier_name, body, str(sys_prompt))
        headers = {"x-route": tier_name}
        if p_small is not None:
            headers["x-router-p-small-ok"] = f"{p_small:.3f}"
        if use_cache:
            hit = await cache.get(tenant.name, sc, messages, semantic=config.cache.semantic and tenant.semantic_cache)
            telemetry.CACHE.labels(tenant.name, hit.kind if hit else "miss").inc()
            if hit:
                telemetry.REQUESTS.labels(tenant.name, tier_name, "cache", hit.kind).inc()
                telemetry.COUNTERFACTUAL.labels(tenant.name).inc(config.tiers["large"].cost(*usage_of(hit.response)[:2]))
                headers |= {"x-cache": f"hit-{hit.kind}", "x-cache-similarity": f"{hit.similarity:.4f}", "x-cost-usd": "0"}
                return JSONResponse(hit.response, headers=headers)

        served = tier_name
        try:
            resp = await call(tier_name, body)
        except httpx.HTTPError:
            if tier_name == "small" and config.routing.cascade:
                telemetry.ESCALATIONS.labels(tenant.name, "backend_error").inc()
                served = "large"
                resp = await call("large", body)
            else:
                raise HTTPException(502, f"{tier_name} backend failed") from None
        cost = await account(tenant.name, served, resp, x_task_id)
        if served == "small" and config.routing.cascade:
            reason = check(body, resp)
            if reason:
                telemetry.ESCALATIONS.labels(tenant.name, reason).inc()
                served = "large"
                resp = await call("large", body)
                cost += await account(tenant.name, served, resp, x_task_id)
        i, o, _ = usage_of(resp)
        telemetry.COUNTERFACTUAL.labels(tenant.name).inc(config.tiers["large"].cost(i, o))
        telemetry.REQUESTS.labels(tenant.name, tier_name, served, "miss" if use_cache else "bypass").inc()
        telemetry.LATENCY.labels(served).observe(time.perf_counter() - started)
        if use_cache and check(body, resp) is None:
            await cache.put(tenant.name, sc, messages, resp, semantic=config.cache.semantic and tenant.semantic_cache)
        headers |= {
            "x-route": f"{tier_name}->{served}" if served != tier_name else tier_name,
            "x-cache": "miss" if use_cache else "bypass",
            "x-cost-usd": f"{cost:.8f}",
        }
        return JSONResponse(resp, headers=headers)

    @app.post("/v1/tasks/{task_id}/outcome")
    async def task_outcome(task_id: str, body: Outcome, tenant: Tenant = TENANT) -> dict[str, Any]:
        """Close a task: its accumulated spend is recorded against the outcome. Reporting twice counts once."""
        key = f"task:{tenant.name}:{task_id}"
        if not await r.set(f"{key}:closed", body.outcome, nx=True, ex=config.task_ttl_seconds):
            raise HTTPException(409, "task already closed")
        data = await r.hgetall(key)
        cost = float(data.get(b"cost", 0) or 0)
        calls = int(data.get(b"calls", 0) or 0)
        telemetry.TASKS.labels(tenant.name, body.outcome).inc()
        telemetry.TASK_COST.labels(tenant.name, body.outcome).observe(cost)
        await r.delete(key)
        return {"task_id": task_id, "outcome": body.outcome, "cost_usd": cost, "calls": calls}

    @app.delete("/v1/cache")
    async def purge(tenant: Tenant = TENANT) -> dict[str, int]:
        """Drop this tenant's cached answers, e.g. after changing a system prompt or model."""
        return {"deleted": await cache.purge(tenant.name)}

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> Response:
        try:
            await r.ping()
            return Response("ready")
        except Exception:  # noqa: BLE001
            return Response("redis unavailable", status_code=503)

    @app.get("/metrics", include_in_schema=False)
    async def prom() -> Response:
        return Response(generate_latest(telemetry.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


__all__ = ["create_app", "log"]
