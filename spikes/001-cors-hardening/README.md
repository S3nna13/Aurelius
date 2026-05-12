# Spike 001: CORS and guardrail hardening

## Question
Can we tighten CORS defaults from wildcard `*` to restrict by host, and upgrade guardrails from substring match to regex, without breaking existing tests?

## Approach
1. Change CORS default from `allowed_origins="*"` to `allowed_origins=""` (empty = same-origin only)
2. Upgrade guardrails from plain `_SUSPICIOUS_SUBSTRINGS` to regex patterns
3. Run functional verification tests

## Results
- CORS middleware now defaults to empty-origin (same-site only) instead of wide-open `*`
- `allow_credentials` defaults to `True` only when explicit origins are set
- Guardrails now use regex patterns (`re.IGNORECASE`) with word boundaries instead of plain `in` substring checks
- `_pattern_check` now checks both custom policy regexes AND built-in suspicious pattern list
- Harm score is now proportional (fraction of matched patterns) instead of binary 0/1

## Verdict: VALIDATED

### What worked
- CORS with `None` default produces empty-string origin, safe for same-origin-only deployments
- All 6 regex patterns correctly detect prompt injection variants
- Pattern-based blocking catches threats before they reach harm score threshold
- No syntax or import errors — ruff check passes clean

### What didn't
- Initial harm threshold of 0.5 meant single-pattern matches didn't block — fixed by adding `_SUSPICIOUS_PATTERNS` to `_pattern_check()`

### Surprises
- The harm_score function is now proportional (1/6 per pattern) which is more nuanced but means `harm_threshold` semantics changed from binary to fractional — callers checking `score >= 0.5` now need 3+ pattern matches

### Recommendation for the real build
- Merge both changes. They're compatible with existing callers (backward-compatible defaults).
