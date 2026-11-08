"""Prometheus metrics. Cost per completed task = rate(task cost sum) / rate(tasks with outcome success)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "llm_requests_total",
    "Chat requests by tenant, routed tier, served tier and cache result",
    ["tenant", "routed", "served", "cache"],
    registry=REGISTRY,
)
COST = Counter("llm_cost_usd_total", "Spend on backends", ["tenant", "tier"], registry=REGISTRY)
COUNTERFACTUAL = Counter(
    "llm_counterfactual_cost_usd_total",
    "What the same requests would have cost on the large tier, with no cache",
    ["tenant"],
    registry=REGISTRY,
)
TOKENS = Counter("llm_tokens_total", "Tokens by tier and type", ["tenant", "tier", "type"], registry=REGISTRY)
ESCALATIONS = Counter(
    "llm_escalations_total",
    "Small-tier answers that failed checks and were retried on the large tier",
    ["tenant", "reason"],
    registry=REGISTRY,
)
CACHE = Counter("llm_cache_lookups_total", "Cache lookups by result", ["tenant", "result"], registry=REGISTRY)
ROUTER_P = Histogram(
    "llm_router_p_small_ok", "Router's probability that the small tier will do", buckets=[i / 10 for i in range(11)], registry=REGISTRY
)
LATENCY = Histogram(
    "llm_request_seconds",
    "Gateway latency by served tier",
    ["served"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=REGISTRY,
)
BACKEND_ERRORS = Counter("llm_backend_errors_total", "Backend failures", ["tier", "kind"], registry=REGISTRY)
TASKS = Counter("agent_tasks_total", "Finished tasks by outcome", ["tenant", "outcome"], registry=REGISTRY)
TASK_COST = Histogram(
    "agent_task_cost_usd",
    "Spend per finished task",
    ["tenant", "outcome"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
    registry=REGISTRY,
)
