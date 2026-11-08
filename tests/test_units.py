from __future__ import annotations

import numpy as np
import pytest

from llmrouter import router as R
from llmrouter.cache import cacheable, scope, single_turn
from llmrouter.config import GatewayConfig
from llmrouter.embed import HashEmbedder
from llmrouter.gateway import check, routing_text


def test_router_round_trip_and_route(router_file):
    r = R.Router.load(router_file)
    e = HashEmbedder()
    assert r.route(e, "hi there").tier == "small"
    assert r.route(e, "Prove step by step the derivative of x^2 = 2x and analyze the trade-offs").tier == "large"
    assert r.route(e, "hi there", threshold=1.0).tier == "large"


def test_train_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    coef, b = R.train(x, y)
    assert coef[0] > 1 and abs(b) < 1


def test_shape_features():
    f = R.shape_features("```python\ndef f():\n  return 1\n```\nWhy?")
    assert f[R.SHAPE.index("code")] == 1 and f[R.SHAPE.index("questions")] > 0


def test_cacheable_and_turns():
    assert cacheable({"temperature": 0})
    assert not cacheable({})  # default temperature is 1
    assert not cacheable({"temperature": 0, "stream": True}) and not cacheable({"temperature": 0, "seed": 1})
    assert single_turn([{"role": "user", "content": "q"}]) == ("", "q")
    assert single_turn([{"role": "user", "content": [{"type": "text"}]}]) is None
    assert scope("a", "small", {}, "S") != scope("b", "small", {}, "S") != scope("a", "large", {}, "S")
    assert scope("a", "small", {"max_tokens": 5}, "S") != scope("a", "small", {}, "S")


def test_checks():
    def resp(text, finish="stop"):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}

    assert check({}, resp("Paris.")) is None
    assert check({}, resp("")) == "empty"
    assert check({}, resp("As an AI, I cannot")) == "refusal"
    assert check({}, resp("x", "length")) == "truncated"
    assert check({"response_format": {"type": "json_object"}}, resp("{bad")) == "invalid_json"
    assert check({"response_format": {"type": "json_object"}}, resp('{"a": 1}')) is None


def test_routing_text_uses_last_user_turn():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "new"},
    ]
    assert routing_text(msgs) == "SYS\n\nnew"


def test_config_requires_both_tiers(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tiers:\n  small: {base_url: http://s, model: s, price_in: 1, price_out: 1}\ntenants: []\n")
    with pytest.raises(ValueError, match="small and large"):
        GatewayConfig.load(p)
