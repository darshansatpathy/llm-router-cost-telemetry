from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "grafana"))
import build  # noqa: E402

PROMTOOL = shutil.which("promtool") or shutil.which("/opt/bin/promtool")


def test_dashboard_is_generated():
    assert json.loads((ROOT / "deploy/grafana/dashboards/llm-gateway.json").read_text()) == build.build()


@pytest.mark.skipif(PROMTOOL is None, reason="promtool not installed")
def test_rules_and_queries(tmp_path):
    d = ROOT / "deploy/prometheus"
    assert subprocess.run([PROMTOOL, "test", "rules", "rules_test.yaml"], cwd=d, capture_output=True).returncode == 0
    qs = [q.replace("$__rate_interval", "5m").replace("$tenant", ".*") for q in build.queries()]
    f = tmp_path / "q.yaml"
    f.write_text(yaml.safe_dump({"groups": [{"name": "q", "rules": [{"record": f"q:{i}", "expr": q} for i, q in enumerate(qs)]}]}))
    out = subprocess.run([PROMTOOL, "check", "rules", str(f)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_committed_router_file_is_plain_json():
    path = ROOT / "models/router.json"
    if not path.exists():
        pytest.skip("router not trained yet")
    d = json.loads(path.read_text())
    assert len(d["coef"]) == 384 + 6 and 0 < d["threshold"] < 1 and d["embedder"] == "all-MiniLM-L6-v2"


def test_example_config_loads():
    from llmrouter.config import GatewayConfig

    cfg = GatewayConfig.load(ROOT / "gateway.yaml")
    assert cfg.tenant_for("secret").name == "demo" and cfg.tenant_for("nope") is None
    assert cfg.cache.enabled and not cfg.cache.semantic
    assert cfg.tiers["large"].cost(1_000_000, 0) == 10.0
