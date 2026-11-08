# 2. The exact cache is on; the semantic cache is off unless a tenant opts in

Status: accepted

## Context

A semantic cache answers a new prompt with a stored answer to a similar one. When similar is not the same ("flights from Paris to Rome" and "from Rome to Paris"), it serves a confident wrong answer, with no model involved to notice. The failure is silent.

## Decision

- Two layers in Redis: exact (normalised request hash) first, then semantic (Redis Query Engine HNSW over MiniLM embeddings).
- Only deterministic requests are cached (temperature 0, one choice, no tools, not streamed); the semantic layer only for single-turn requests.
- Every entry is scoped to the tenant, the model tier, the system prompt and the output settings; purge is per tenant.
- The threshold is calibrated on PAWS, whose pairs share most words but half differ in meaning. With MiniLM, no threshold below exact equality keeps false hits at or below 1% (0.998 still gives 2%; 0.98 gives 45%). So the semantic layer is **off by default** (`cache.semantic: false`) and a tenant must also opt in (`semantic_cache: true`) after calibrating on its own traffic.
- Answers that failed the cascade checks are never cached. Callers can opt out per request (`x-cache: no-store`) and tenants can turn the semantic layer off.

## Consequences

The savings come from routing and the exact layer. A better embedder (one trained for paraphrase identification, or a cross-encoder check on candidate hits) is the way to make the semantic layer safe; the calibration command measures any replacement the same way.
