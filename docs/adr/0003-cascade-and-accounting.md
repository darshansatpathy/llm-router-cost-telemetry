# 3. Escalate failed small-model answers; charge every call to the task

Status: accepted

## Context

A router will sometimes send a request to the small model that it cannot handle. Some of those failures are visible without a judge: an empty answer, a refusal, a truncated answer, invalid JSON when JSON was asked for. And every misroute costs a second call somewhere, in the gateway or in the agent.

## Decision

- Cheap checks run on every small-tier answer; a failure re-sends the request to the large tier (`x-route: small->large`). A small-tier backend outage does the same.
- With `x-task-id`, every call's cost (both, for an escalation) accumulates against the task in Redis. `POST /v1/tasks/{id}/outcome` closes the task and records its total under its outcome, once.
- Dashboards lead with cost per completed task and the saving against a counterfactual of sending everything to the large tier with no cache.

## Consequences

The counterfactual uses the same token counts priced at the large tier; the large model's answers would differ in length, so the saving is an estimate. The replay report measures the real thing on recorded answers.
