"""Generate the gateway dashboard. `--queries` prints every PromQL query (tests parse them with promtool)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
T = 'tenant=~"$tenant"'

PANELS: list[tuple[str, str, str, int, list[tuple[str, str]]]] = [
    ("Cost per completed task", "currencyUSD", "stat", 6, [(f"tenant:cost_per_completed_task_usd:1h{{{T}}}", "{{tenant}}")]),
    ("Saving vs always-large", "percentunit", "stat", 6, [(f"tenant:savings_ratio:1h{{{T}}}", "{{tenant}}")]),
    ("Share served by the small tier", "percentunit", "stat", 6, [(f"tenant:small_share:1h{{{T}}}", "{{tenant}}")]),
    ("Cache hit ratio", "percentunit", "stat", 6, [(f"tenant:cache_hit_ratio:1h{{{T}}}", "{{tenant}}")]),
    (
        "Spend vs always-large",
        "currencyUSD",
        "timeseries",
        12,
        [
            (f"sum(rate(llm_cost_usd_total{{{T}}}[$__rate_interval])) * 3600", "actual / h"),
            (f"sum(rate(llm_counterfactual_cost_usd_total{{{T}}}[$__rate_interval])) * 3600", "always large / h"),
        ],
    ),
    (
        "Cost per completed task over time",
        "currencyUSD",
        "timeseries",
        12,
        [(f"tenant:cost_per_completed_task_usd:1h{{{T}}}", "{{tenant}}")],
    ),
    (
        "Requests by route",
        "short",
        "timeseries",
        12,
        [(f"sum by (routed, served) (rate(llm_requests_total{{{T}}}[$__rate_interval])) * 60", "{{routed}} → {{served}} / min")],
    ),
    (
        "Escalations by reason",
        "short",
        "timeseries",
        12,
        [(f"sum by (reason) (rate(llm_escalations_total{{{T}}}[$__rate_interval])) * 60", "{{reason}} / min")],
    ),
    ("Router confidence", "short", "timeseries", 12, [("sum by (le) (rate(llm_router_p_small_ok_bucket[$__rate_interval]))", "{{le}}")]),
    (
        "Gateway latency p95 by served tier",
        "s",
        "timeseries",
        12,
        [("histogram_quantile(0.95, sum by (le, served) (rate(llm_request_seconds_bucket[$__rate_interval])))", "{{served}}")],
    ),
    (
        "Tokens by tier",
        "short",
        "timeseries",
        12,
        [(f"sum by (tier, type) (rate(llm_tokens_total{{{T}}}[$__rate_interval])) * 60", "{{tier}} {{type}} / min")],
    ),
    (
        "Backend errors",
        "short",
        "timeseries",
        12,
        [("sum by (tier, kind) (rate(llm_backend_errors_total[$__rate_interval])) * 60", "{{tier}} {{kind}} / min")],
    ),
]


def queries() -> list[str]:
    return [q for *_, targets in PANELS for q, _ in targets]


def build() -> dict:
    panels, x, y, row_h = [], 0, 0, 0
    for i, (title, unit, kind, w, targets) in enumerate(PANELS):
        h = 4 if kind == "stat" else 8
        if x + w > 24:
            x, y, row_h = 0, y + row_h, 0
        panels.append(
            {
                "id": i + 1,
                "type": kind,
                "title": title,
                "datasource": PROM,
                "gridPos": {"x": x, "y": y, "w": w, "h": h},
                "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                "targets": [
                    {"refId": chr(65 + j), "expr": q, "legendFormat": leg, "datasource": PROM} for j, (q, leg) in enumerate(targets)
                ],
            }
        )
        x += w
        row_h = max(row_h, h)
    return {
        "uid": "llm-gateway",
        "title": "LLM gateway: routing, cache and cost per completed task",
        "tags": ["llm", "cost", "routing"],
        "schemaVersion": 39,
        "time": {"from": "now-6h", "to": "now"},
        "refresh": "1m",
        "templating": {
            "list": [
                {
                    "name": "tenant",
                    "type": "query",
                    "datasource": PROM,
                    "query": {"query": "label_values(llm_requests_total, tenant)", "refId": "t"},
                    "includeAll": True,
                    "multi": True,
                    "allValue": ".*",
                    "current": {"text": "All", "value": "$__all"},
                }
            ]
        },
        "panels": panels,
    }


if __name__ == "__main__":
    if "--queries" in sys.argv:
        print("\n".join(queries()))
    else:
        out = Path(__file__).parent / "dashboards" / "llm-gateway.json"
        out.write_text(json.dumps(build(), indent=2) + "\n")
        print(f"wrote {out}")
