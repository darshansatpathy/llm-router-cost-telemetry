"""`llmrouter`: train the router, calibrate the cache, replay traffic through the gateway, serve."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from llmrouter import __version__

DATA = Path("data")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llmrouter", description=__doc__)
    p.add_argument("--version", action="version", version=f"llmrouter {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the router on RouteLLM data and write models/router.json and docs/routing-report.md")
    t.add_argument("--target-quality", type=float, default=0.95)
    t.add_argument("--embedder", default="minilm")
    c = sub.add_parser("calibrate-cache", help="choose the semantic cache threshold on PAWS and write docs/cache-report.md")
    c.add_argument("--max-false-hit", type=float, default=0.01)
    c.add_argument("--embedder", default="minilm")
    r = sub.add_parser("replay", help="replay held-out prompts through the gateway under several policies")
    r.add_argument("--config", default="gateway.yaml")
    r.add_argument("--redis", default="redis://localhost:6379/0")
    r.add_argument("--limit", type=int, default=0, help="use only this many held-out prompts")
    r.add_argument("--repeat-share", type=float, default=0.25)
    r.add_argument("--out")
    s = sub.add_parser("serve")
    s.add_argument("--config", default="gateway.yaml")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)

    if a.cmd == "train":
        from llmrouter import data, routing_eval
        from llmrouter.embed import get

        examples = data.load(data.download(DATA / "routellm-valid.jsonl"))
        router, res = routing_eval.run(examples, get(a.embedder), target_quality=a.target_quality)
        Path("models").mkdir(exist_ok=True)
        router.save(Path("models/router.json"))
        Path("docs/routing-report.md").write_text(routing_eval.markdown(res))
        print(routing_eval.markdown(res))
    elif a.cmd == "calibrate-cache":
        from llmrouter import cache_eval
        from llmrouter.embed import get

        emb = get(a.embedder)
        s1, s2, y = cache_eval.load(cache_eval.download(DATA / "paws-test.parquet"))
        res = cache_eval.calibrate(cache_eval.similarities(emb, s1, s2), y, a.max_false_hit)
        Path("docs/cache-report.md").write_text(cache_eval.markdown(res, emb.name))
        print(json.dumps(res["chosen"]))
    elif a.cmd == "replay":
        return asyncio.run(_replay(a))
    elif a.cmd == "serve":
        import uvicorn

        from llmrouter.config import GatewayConfig
        from llmrouter.gateway import create_app

        uvicorn.run(create_app(GatewayConfig.load(Path(a.config))), host=a.host, port=a.port, log_level="warning", server_header=False)
    return 0


async def _replay(a: argparse.Namespace) -> int:
    from redis.asyncio import Redis

    from llmrouter import data, replay, routing_eval
    from llmrouter.config import GatewayConfig
    from llmrouter.embed import get
    from llmrouter.router import Router

    cfg = GatewayConfig.load(Path(a.config))
    examples = data.load(data.download(DATA / "routellm-valid.jsonl"))
    _, _, test = routing_eval.split(len(examples))
    held_out = [examples[i] for i in test]
    if a.limit:
        held_out = held_out[: a.limit]
    reps = replay.repeats_for(len(held_out), a.repeat_share)
    emb = get(cfg.embedder)
    redis = Redis.from_url(a.redis)
    policies = [
        replay.Policy("always large", "large", False),
        replay.Policy("always small", "small", False),
        replay.Policy("router", "auto", False),
        replay.Policy("router + cache", "auto", True),
    ]
    results = [await replay.run_policy(pol, cfg, held_out, reps, emb, redis) for pol in policies]
    await redis.flushdb()
    await redis.aclose()
    text = replay.markdown(results, len(held_out), len(reps), Router.load(cfg.routing.router_file).threshold, cfg.cache.semantic_threshold)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
