# Security policy

Report a vulnerability privately through [GitHub security advisories](https://github.com/darshansatpathy/llm-router-cost-telemetry/security/advisories/new). Please do not open a public issue. You can expect a first reply within five working days.

## In scope

- One tenant receiving another tenant's cached answer, or purging another tenant's cache.
- A cache hit served for a request that should not be cached (non-zero temperature, tools, opted out).
- Backend API keys leaking into responses, logs or metrics; they are read from environment variables only.
- Loading a router file that runs code (router files are JSON only).

## Supported versions

The latest release receives fixes.
