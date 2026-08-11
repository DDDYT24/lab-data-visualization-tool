"""Send one redacted SES acceptance probe through the production adapter."""

from __future__ import annotations

import json
import os
import secrets

from labviz_api.auth import SesV2EmailSender
from labviz_api.config import Settings

RECIPIENT_ENV = "LABVIZ_SES_PROBE_RECIPIENT"


def main() -> int:
    settings = Settings.from_env()
    if settings.auth_mode != "ses":
        print(json.dumps({"error": "ses-auth-mode-required", "status": "failed"}))
        return 2
    recipient = os.environ.get(RECIPIENT_ENV, "").strip()
    if not recipient:
        print(json.dumps({"error": f"{RECIPIENT_ENV}-required", "status": "failed"}))
        return 2
    try:
        receipt = SesV2EmailSender(settings).send_code(
            recipient,
            f"{secrets.randbelow(1_000_000):06d}",
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "errorType": type(exc).__name__,
                    "status": "failed",
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "messageId": receipt.message_id,
                "provider": receipt.provider,
                "status": "accepted",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
