"""Gateway configuration: model tiers and prices, tenants, cache and routing settings (YAML)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Tier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    api_key_env: str | None = Field(None, description="Environment variable holding the backend's API key")
    price_in: float = Field(ge=0, description="USD per million input tokens")
    price_out: float = Field(ge=0)
    price_cached_in: float | None = Field(None, ge=0)
    timeout_seconds: float = 60

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def cost(self, input_tokens: int, output_tokens: int, cached: int = 0) -> float:
        cached_price = self.price_in if self.price_cached_in is None else self.price_cached_in
        return ((input_tokens - cached) * self.price_in + cached * cached_price + output_tokens * self.price_out) / 1e6


class Tenant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    key_sha256: str
    semantic_cache: bool = Field(False, description="Opt this tenant in to the semantic layer (also needs cache.semantic)")


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    semantic: bool = Field(False, description="The semantic layer is off by default; see docs/cache-report.md for why")
    semantic_threshold: float = Field(0.995, ge=0.5, le=1.0)
    ttl_seconds: int = Field(86400, ge=60)


class RoutingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    router_file: Path = Path("models/router.json")
    threshold: float | None = Field(None, description="Override the router file's threshold")
    cascade: bool = Field(True, description="Retry on the large tier when the small tier's answer fails the checks")


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiers: dict[str, Tier]
    tenants: list[Tenant]
    cache: CacheSettings = Field(default_factory=lambda: CacheSettings.model_validate({}))
    routing: RoutingSettings = Field(default_factory=lambda: RoutingSettings.model_validate({}))
    redis_url: str = "redis://localhost:6379/0"
    embedder: str = "minilm"
    task_ttl_seconds: int = 86400

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        cfg = cls.model_validate(yaml.safe_load(path.read_text()))
        if {"small", "large"} - set(cfg.tiers):
            raise ValueError("tiers must define small and large")
        return cfg

    def tenant_for(self, key: str) -> Tenant | None:
        h = hashlib.sha256(key.encode()).hexdigest()
        return next((t for t in self.tenants if t.key_sha256 == h), None)
