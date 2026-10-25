"""Domain repair accounting shared by Phase2 and Phase3."""

NON_DISPATCHED = {"rejected_scope", "missing_baseline", "unsupported", "budget_exhausted"}


def merge_attempts(*counts):
    result = {}
    for values in counts:
        for domain, count in values.items():
            result[domain] = max(result.get(domain, 0), count)
    return result


def available_targets(state, targets, limit=1):
    return {domain for domain in targets if state.targeted_repair_attempts.get(domain, 0) < limit}


def finish_attempt(before, result, domain):
    outcome = result.targeted_repair_results.get(domain)
    if not outcome:
        field = {"db": "db_schema", "api": "api_spec", "prd": "prd_document", "pm": "feature_list"}[domain]
        outcome = "applied" if getattr(result, field) != getattr(before, field) else "noop"
    attempts = dict(before.targeted_repair_attempts)
    if outcome not in NON_DISPATCHED:
        attempts[domain] = attempts.get(domain, 0) + 1
    return result.copy(targeted_repair_attempts=attempts,
                       targeted_repair_results={**result.targeted_repair_results, domain: outcome})
