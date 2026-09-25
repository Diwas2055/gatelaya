# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Use [GitHub Security Advisories](https://github.com/Diwas2055/gatelaya/security/advisories/new) ("Report a vulnerability") — you'll get a private thread with the maintainer.

If the advisory form is unavailable, email the maintainer via the address on the GitHub profile with:

- description of the issue and affected component (guardrail / router / dashboard / Docker image)
- steps to reproduce
- impact (e.g. bypass of a check, audit-log leak, RCE)

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅ |
| < 0.2   | ❌ (upgrade) |

## Scope notes

- GateLaya is a **fail-open** system by design (`GATELAYA_FAIL_OPEN=true` default): guardrail errors allow the request through. Bypassing a *detection* (e.g. crafting a prompt that evades the injection classifier) is a model-quality issue — report it, but it is not a security vulnerability. Bypassing an *enforced block* through a code path (masking bug, audit tampering, auth bypass on the dashboard) is.
- The dashboard ships with Bearer-token auth and is intended to run behind TLS; treat exposed unauthenticated deployments as unsupported.
- Eval fixtures in `evals/data/` intentionally contain synthetic PII/secret-shaped strings. They are not real credentials.
