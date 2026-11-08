"""A response cache in Redis with two layers: exact, then semantic.

A semantic cache returns a stored answer for a *different* prompt that means the same thing. Get the
threshold wrong and it confidently answers a question nobody asked, so it is narrow by design:

- only deterministic requests (temperature 0, one choice, no tools, not streamed);
- the semantic layer only for single-turn requests (one system prompt, one user message);
- every entry is scoped to the tenant, the model tier, the system prompt and the output settings;
- the similarity threshold is calibrated on adversarial paraphrase pairs (see docs/cache-report.md);
- entries expire, and callers can opt out per request.

Vector search uses the Redis Query Engine (Redis 8, or Redis Stack): an HNSW index over cosine distance.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from llmrouter.embed import Embedder

INDEX = "llmcache"
PREFIX = "llmc:"


@dataclass
class Hit:
    kind: Literal["exact", "semantic"]
    response: dict[str, Any]
    similarity: float


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cacheable(body: dict[str, Any]) -> bool:
    return (
        float(body.get("temperature", 1.0)) == 0.0
        and int(body.get("n", 1)) == 1
        and not body.get("tools")
        and not body.get("functions")
        and not body.get("stream")
        and body.get("seed") is None
    )


def single_turn(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    """(system prompt, user message) when the request is one user turn, else None."""
    roles = [m.get("role") for m in messages]
    if roles not in (["user"], ["system", "user"]):
        return None
    sys_prompt = messages[0]["content"] if roles[0] == "system" else ""
    user = messages[-1].get("content")
    if not isinstance(user, str) or not isinstance(sys_prompt, str):
        return None
    return sys_prompt, user


def scope(tenant: str, tier: str, body: dict[str, Any], system_prompt: str) -> str:
    """Everything that changes what a correct answer is, apart from the user's words."""
    keys = {k: body.get(k) for k in ("response_format", "max_tokens", "max_completion_tokens", "stop", "top_p")}
    raw = json.dumps({"t": tenant, "tier": tier, "sys": normalise(system_prompt), "k": keys}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class SemanticCache:
    def __init__(self, redis: Redis, embedder: Embedder, *, threshold: float, ttl_seconds: int = 86400):
        self.r, self.embedder, self.threshold, self.ttl = redis, embedder, threshold, ttl_seconds

    async def ensure_index(self) -> None:
        try:
            await self.r.ft(INDEX).info()
        except ResponseError:
            await self.r.ft(INDEX).create_index(
                [
                    TagField("scope"),
                    NumericField("created"),
                    VectorField(
                        "vec",
                        "HNSW",
                        {"TYPE": "FLOAT32", "DIM": self.embedder.dim, "DISTANCE_METRIC": "COSINE", "M": 16, "EF_CONSTRUCTION": 200},
                    ),
                ],
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
            )

    @staticmethod
    def _tenant(tenant: str) -> str:
        return hashlib.sha256(tenant.encode()).hexdigest()[:12]

    def _exact_key(self, tenant: str, sc: str, messages: list[dict[str, Any]]) -> str:
        h = hashlib.sha256(json.dumps([[m.get("role"), normalise(str(m.get("content")))] for m in messages]).encode()).hexdigest()
        return f"llmx:{self._tenant(tenant)}:{sc}:{h}"

    async def get(self, tenant: str, sc: str, messages: list[dict[str, Any]], *, semantic: bool = True) -> Hit | None:
        raw = await self.r.get(self._exact_key(tenant, sc, messages))
        if raw:
            return Hit("exact", json.loads(raw), 1.0)
        turn = single_turn(messages)
        if not semantic or turn is None:
            return None
        vec = self.embedder.embed([turn[1]])[0].astype(np.float32)
        q = Query(f"(@scope:{{{sc}}})=>[KNN 1 @vec $v AS dist]").sort_by("dist").return_fields("dist", "response").dialect(2)
        res = await self.r.ft(INDEX).search(q, query_params={"v": vec.tobytes()})
        if not res.docs:
            return None
        doc = res.docs[0]
        sim = 1 - float(doc.dist)
        if sim < self.threshold:
            return None
        return Hit("semantic", json.loads(doc.response), sim)

    async def put(self, tenant: str, sc: str, messages: list[dict[str, Any]], response: dict[str, Any], *, semantic: bool = True) -> None:
        body = json.dumps(response)
        await self.r.set(self._exact_key(tenant, sc, messages), body, ex=self.ttl)
        turn = single_turn(messages)
        if turn is None or not semantic:
            return
        vec = self.embedder.embed([turn[1]])[0].astype(np.float32)
        key = f"{PREFIX}{self._tenant(tenant)}:{sc}:" + hashlib.sha256(normalise(turn[1]).encode()).hexdigest()[:24]
        await self.r.hset(key, mapping={"scope": sc, "vec": vec.tobytes(), "response": body, "created": int(time.time())})
        await self.r.expire(key, self.ttl)

    async def purge(self, tenant: str | None = None) -> int:
        """Drop cached answers, for one tenant or all, e.g. after a prompt or model change."""
        n = 0
        t = self._tenant(tenant) if tenant else "*"
        for pattern in (f"llmx:{t}:*", f"{PREFIX}{t}:*"):
            async for key in self.r.scan_iter(match=pattern, count=500):
                n += await self.r.delete(key)
        return n
