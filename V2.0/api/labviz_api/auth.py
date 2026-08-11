"""Passwordless email-code authentication for the LabViz API."""

from __future__ import annotations

import hashlib
import logging
import secrets
import smtplib
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Any, Literal, Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

import boto3
from botocore.config import Config

from .config import Settings
from .repository import iso_at

LOGGER = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class AuthError(ValueError):
    """Expected authentication failure with a stable API error code."""

    def __init__(self, message: str, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class EmailDeliveryReceipt:
    provider: Literal["console", "memory", "smtp", "ses"]
    message_id: str | None


class EmailSender(Protocol):
    @property
    def delivery_mode(self) -> Literal["console", "email"]:
        """Report whether a real email or development console receives the code."""
        ...

    def send_code(self, email: str, code: str) -> EmailDeliveryReceipt:
        """Deliver one verification code."""
        ...


class AuthRepository(Protocol):
    """Operational authentication persistence selected with the project backend."""

    def latest_auth_challenge(self, email: str) -> dict[str, Any] | None: ...

    def create_auth_challenge(
        self,
        *,
        challenge_id: str,
        email: str,
        salt: str,
        code_digest: str,
        expires_at: str,
        resend_at: str,
    ) -> None: ...

    def get_auth_challenge(self, challenge_id: str) -> dict[str, Any] | None: ...

    def increment_auth_challenge_attempts(self, challenge_id: str) -> None: ...

    def delete_auth_challenge(self, challenge_id: str) -> None: ...

    def create_auth_session(
        self,
        *,
        token_digest: str,
        user_id: str,
        email: str,
        expires_at: str,
    ) -> None: ...

    def get_auth_session(self, token_digest: str) -> dict[str, Any] | None: ...

    def delete_auth_session(self, token_digest: str) -> None: ...

    def allow_auth_request(
        self,
        *,
        client_key: str,
        email: str,
        client_limit: int = 30,
        email_limit: int = 10,
        window_seconds: int = 3_600,
    ) -> bool: ...


class ConsoleEmailSender:
    """Development-only sender that prints the code to the API log."""

    delivery_mode: Literal["console"] = "console"

    def send_code(self, email: str, code: str) -> EmailDeliveryReceipt:
        LOGGER.warning("LabViz sign-in code for %s: %s", email, code)
        return EmailDeliveryReceipt(provider="console", message_id=None)


class SmtpEmailSender:
    delivery_mode: Literal["email"] = "email"

    def __init__(self, settings: Settings) -> None:
        if not settings.smtp_host:
            raise ValueError("LABVIZ_SMTP_HOST is required when SMTP auth mode is enabled.")
        self.settings = settings
        self.host = settings.smtp_host

    def send_code(self, email: str, code: str) -> EmailDeliveryReceipt:
        message = EmailMessage()
        message["Subject"] = "Your LabViz sign-in code"
        message["From"] = self.settings.smtp_from
        message["To"] = email
        message["Message-ID"] = make_msgid()
        message.set_content(
            f"Your LabViz verification code is {code}. "
            "It expires in 10 minutes. If you did not request this code, ignore this email."
        )

        with smtplib.SMTP(self.host, self.settings.smtp_port, timeout=15) as smtp:
            if self.settings.smtp_starttls:
                smtp.starttls(context=ssl.create_default_context())
            if self.settings.smtp_username and self.settings.smtp_password:
                smtp.login(self.settings.smtp_username, self.settings.smtp_password)
            refused = smtp.send_message(message)
            if refused:
                raise RuntimeError("SMTP refused the verification recipient.")
        return EmailDeliveryReceipt(provider="smtp", message_id=message["Message-ID"])


class SesV2Client(Protocol):
    def send_email(self, **kwargs: Any) -> dict[str, Any]: ...


class SesV2EmailSender:
    """Amazon SES v2 sender using only the standard AWS credential chain."""

    delivery_mode: Literal["email"] = "email"

    def __init__(self, settings: Settings, client: SesV2Client | None = None) -> None:
        if not settings.ses_region:
            raise ValueError("LABVIZ_SES_REGION is required when SES auth mode is enabled.")
        if not settings.ses_from:
            raise ValueError("LABVIZ_SES_FROM is required when SES auth mode is enabled.")
        if not settings.ses_configuration_set:
            raise ValueError(
                "LABVIZ_SES_CONFIGURATION_SET is required when SES auth mode is enabled."
            )
        self.region = settings.ses_region
        self.sender = settings.ses_from
        self.configuration_set = settings.ses_configuration_set
        self.environment = settings.environment
        self.client = client or cast(
            SesV2Client,
            boto3.client(
                "sesv2",
                region_name=self.region,
                config=Config(
                    connect_timeout=settings.ses_connect_timeout_seconds,
                    read_timeout=settings.ses_read_timeout_seconds,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            ),
        )

    def send_code(self, email: str, code: str) -> EmailDeliveryReceipt:
        response = self.client.send_email(
            FromEmailAddress=self.sender,
            Destination={"ToAddresses": [email]},
            Content={
                "Simple": {
                    "Subject": {
                        "Data": "Your LabViz sign-in code",
                        "Charset": "UTF-8",
                    },
                    "Body": {
                        "Text": {
                            "Data": (
                                f"Your LabViz verification code is {code}. "
                                "It expires in 10 minutes. If you did not request this code, "
                                "ignore this email."
                            ),
                            "Charset": "UTF-8",
                        }
                    },
                }
            },
            ConfigurationSetName=self.configuration_set,
            EmailTags=[
                {"Name": "purpose", "Value": "authentication-code"},
                {"Name": "environment", "Value": self.environment},
            ],
        )
        message_id = response.get("MessageId")
        if not isinstance(message_id, str) or not message_id:
            raise RuntimeError("Amazon SES accepted no message identifier.")
        return EmailDeliveryReceipt(provider="ses", message_id=message_id)


class MemoryEmailSender:
    """Test sender; it is never selected by environment configuration."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.delivery_mode: Literal["email"] = "email"

    def send_code(self, email: str, code: str) -> EmailDeliveryReceipt:
        self.messages.append((email, code))
        return EmailDeliveryReceipt(provider="memory", message_id=f"memory-{len(self.messages)}")


class AuthService:
    challenge_ttl_seconds = 600
    resend_after_seconds = 60
    maximum_attempts = 5
    code_hash_iterations = 210_000

    def __init__(
        self,
        sender: EmailSender,
        session_ttl_seconds: int,
        repository: AuthRepository,
    ) -> None:
        self.sender = sender
        self.session_ttl_seconds = session_ttl_seconds

        self.repository = repository

    @property
    def delivery_mode(self) -> Literal["console", "email"]:
        return self.sender.delivery_mode

    def request_code(self, email: str) -> tuple[str, int, int]:
        normalized = email.strip().lower()
        previous = self.repository.latest_auth_challenge(normalized)
        if previous and self._parse_time(previous["resend_at"]) > _now():
            remaining = max(
                1,
                int((self._parse_time(previous["resend_at"]) - _now()).total_seconds()),
            )
            raise AuthError(
                f"Wait {remaining} seconds before requesting another code.",
                "resend-too-soon",
                429,
            )

        code = f"{secrets.randbelow(1_000_000):06d}"
        salt = secrets.token_hex(16)
        now = _now()
        challenge_id = uuid4().hex
        self.repository.create_auth_challenge(
            challenge_id=challenge_id,
            email=normalized,
            salt=salt,
            code_digest=self._digest_code(salt, code),
            expires_at=iso_at(now + timedelta(seconds=self.challenge_ttl_seconds)),
            resend_at=iso_at(now + timedelta(seconds=self.resend_after_seconds)),
        )
        try:
            receipt = self.sender.send_code(normalized, code)
        except Exception:
            self.repository.delete_auth_challenge(challenge_id)
            raise
        LOGGER.info(
            "authentication-code-delivery-accepted provider=%s message_id=%s challenge_id=%s",
            receipt.provider,
            receipt.message_id or "none",
            challenge_id,
        )
        return challenge_id, self.challenge_ttl_seconds, self.resend_after_seconds

    def verify_code(self, challenge_id: str, code: str) -> tuple[dict[str, str], str]:
        challenge = self.repository.get_auth_challenge(challenge_id)
        if challenge is None:
            raise AuthError(
                "This verification challenge is invalid or has expired.",
                "invalid-challenge",
            )
        if challenge["failed_attempts"] >= self.maximum_attempts:
            self.repository.delete_auth_challenge(challenge_id)
            raise AuthError(
                "Too many incorrect attempts. Request a new code.",
                "challenge-locked",
                429,
            )
        if not secrets.compare_digest(
            challenge["code_digest"], self._digest_code(challenge["salt"], code)
        ):
            if challenge["failed_attempts"] + 1 >= self.maximum_attempts:
                self.repository.delete_auth_challenge(challenge_id)
                raise AuthError(
                    "Too many incorrect attempts. Request a new code.",
                    "challenge-locked",
                    429,
                )
            self.repository.increment_auth_challenge_attempts(challenge_id)
            raise AuthError("The verification code is incorrect.", "incorrect-code")

        self.repository.delete_auth_challenge(challenge_id)

        user_id = uuid5(NAMESPACE_URL, f"labviz:{challenge['email']}").hex
        raw_token = secrets.token_urlsafe(32)
        self.repository.create_auth_session(
            token_digest=self._digest_token(raw_token),
            user_id=user_id,
            email=challenge["email"],
            expires_at=iso_at(_now() + timedelta(seconds=self.session_ttl_seconds)),
        )
        return {"id": user_id, "email": challenge["email"]}, raw_token

    def get_user(self, raw_token: str | None) -> dict[str, str] | None:
        if not raw_token:
            return None
        session = self.repository.get_auth_session(self._digest_token(raw_token))
        if session is None:
            return None
        return {"id": session["user_id"], "email": session["email"]}

    def logout(self, raw_token: str | None) -> None:
        if raw_token:
            self.repository.delete_auth_session(self._digest_token(raw_token))

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _digest_code(salt: str, code: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            code.encode(),
            salt.encode(),
            AuthService.code_hash_iterations,
        ).hex()

    @staticmethod
    def _digest_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()


def build_auth_service(settings: Settings, repository: AuthRepository) -> AuthService:
    if settings.auth_mode == "ses":
        sender: EmailSender = SesV2EmailSender(settings)
    elif settings.auth_mode == "smtp":
        sender = SmtpEmailSender(settings)
    else:
        sender = ConsoleEmailSender()
    return AuthService(sender, settings.session_ttl_seconds, repository)
