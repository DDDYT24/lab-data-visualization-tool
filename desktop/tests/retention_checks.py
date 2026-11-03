"""Keep durable projects protected while accounting for declared demo expiry."""
from datetime import datetime, timezone


def assert_retained(expected, current, now=None):
    now = now or datetime.now(timezone.utc)
    missing = set(expected) - set(current)
    allowed = set()
    for identifier, row in expected.items():
        if row.get('storageMode') != 'temporary-cloud' or not row.get('expiresAt'):
            continue
        try:
            deadline = datetime.fromisoformat(row['expiresAt'].replace('Z', '+00:00'))
            if deadline.tzinfo is not None and deadline < now:
                allowed.add(identifier)
        except (TypeError, ValueError):
            pass
    assert not (set(current) - set(expected)), 'Unexpected scientific projects appeared'
    assert missing <= allowed, 'A durable or unexpired project was lost: '+str(sorted(missing-allowed))
    return sorted(missing)
