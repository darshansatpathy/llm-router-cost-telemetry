# Changelog

## 0.1.0

- OpenAI-compatible gateway with `model: auto` routing between a small and a large tier, or pinned tiers.
- Router: logistic regression over MiniLM embeddings and shape features, trained on RouteLLM's graded data, threshold chosen for a quality target; weights in JSON.
- Two-layer cache in Redis (exact, then semantic via the Redis Query Engine), scoped per tenant, tier, system prompt and output settings. The semantic layer is off by default: calibration on PAWS found no safe threshold for MiniLM.
- Cascade: empty, refused, truncated or invalid-JSON small-tier answers and small-tier outages go to the large tier.
- Task accounting: cost per task across calls and escalations, closed once with an outcome.
- Prometheus metrics, recording and alert rules with promtool tests, a generated Grafana dashboard.
- `llmrouter train`, `calibrate-cache`, `replay` and `serve`; container with the embedding model baked in.
