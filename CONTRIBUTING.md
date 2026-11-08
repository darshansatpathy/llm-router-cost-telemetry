# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
make install
make redis                                            # Redis 8 on port 16379, for the tests
export LLMROUTER_TEST_REDIS=redis://127.0.0.1:16379/0
make check
```

## Retraining the router

`llmrouter train` downloads RouteLLM's data (Apache-2.0) into `data/` (not committed), trains, picks the threshold and writes `models/router.json` and `docs/routing-report.md`. Commit both and say in the pull request what moved. `llmrouter replay` then checks the router end to end through the gateway.

## Changing the cache

Anything that changes what counts as a match reruns `llmrouter calibrate-cache` and commits `docs/cache-report.md`. Keep the scope: tenant, tier, system prompt, output settings.

## Commits

Small, focused commits with messages in the imperative. Update `CHANGELOG.md` for anything a user would notice.
