from __future__ import annotations

import hashlib
import hmac
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from labviz_api.auth import AuthService, MemoryEmailSender
from labviz_api.client_identity import (
    ClientIdentityError,
    ClientIdentityResolver,
    normalize_client_address,
)
from labviz_api.config import Settings
from labviz_api.main import create_app
from labviz_api.repository import ProjectRepository

IDENTITY_KEY = "test-client-identity-key-that-is-at-least-32-bytes"


def settings(tmp_path: Path, **changes: object) -> Settings:
    values: dict[str, object] = {
        "database_path": tmp_path / "client-identity.db",
        "allowed_origins": ("http://localhost:3000",),
        "public_web_url": "http://localhost:3000",
        "trusted_proxy_cidrs": ("10.0.0.0/8",),
        "trusted_proxy_hops": 1,
        "client_identity_key": IDENTITY_KEY,
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("192.0.2.10", "192.0.2.10"),
        ("192.0.2.10:443", "192.0.2.10"),
        ("[2001:0db8:0:0::1]:8443", "2001:db8::1"),
        ("::ffff:192.0.2.10", "192.0.2.10"),
    ),
)
def test_normalizes_supported_proxy_address_forms(raw: str, expected: str) -> None:
    assert str(normalize_client_address(raw)) == expected


@pytest.mark.parametrize(
    "raw",
    ("", "not-an-ip", "192.0.2.1:0", "192.0.2.1:65536", "[2001:db8::1", "fe80::1%eth0"),
)
def test_rejects_malformed_or_ambiguous_addresses(raw: str) -> None:
    with pytest.raises(ClientIdentityError):
        normalize_client_address(raw)


def test_trusted_append_mode_uses_the_nearest_untrusted_address(tmp_path: Path) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path))

    actual = resolver.resolve("10.0.1.25", ["203.0.113.99, 198.51.100.20"])
    expected = hmac.new(
        IDENTITY_KEY.encode(),
        b"198.51.100.20",
        hashlib.sha256,
    ).hexdigest()

    assert actual == expected
    assert "198.51.100.20" not in actual


def test_configured_hops_ignore_forged_left_values_and_require_enough_entries(
    tmp_path: Path,
) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path, trusted_proxy_hops=2))

    expected = hmac.new(
        IDENTITY_KEY.encode(),
        b"198.51.100.20",
        hashlib.sha256,
    ).hexdigest()
    assert (
        resolver.resolve(
            "10.0.1.25",
            ["203.0.113.250, 198.51.100.20, 10.20.30.40"],
        )
        == expected
    )

    with pytest.raises(ClientIdentityError, match="insufficient"):
        resolver.resolve("10.0.1.25", ["198.51.100.20"])


def test_configured_hops_validate_the_trusted_suffix_and_selected_client(
    tmp_path: Path,
) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path, trusted_proxy_hops=2))

    with pytest.raises(ClientIdentityError, match="trusted proxy suffix"):
        resolver.resolve("10.0.1.25", ["198.51.100.20, 203.0.113.99"])
    with pytest.raises(ClientIdentityError, match="does not reach an untrusted client"):
        resolver.resolve("10.0.1.25", ["10.20.30.40, 10.30.40.50"])


def test_untrusted_peer_cannot_inject_forwarded_identity(tmp_path: Path) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path))

    direct = resolver.resolve("192.0.2.25", [])
    forged = resolver.resolve("192.0.2.25", ["203.0.113.99"])

    assert forged == direct


def test_trusted_peer_rejects_duplicate_or_malformed_headers(tmp_path: Path) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path))

    with pytest.raises(ClientIdentityError, match="Exactly one"):
        resolver.resolve("10.0.1.25", ["192.0.2.1", "192.0.2.2"])
    with pytest.raises(ClientIdentityError, match="malformed"):
        resolver.resolve("10.0.1.25", ["192.0.2.1,,192.0.2.2"])
    with pytest.raises(ClientIdentityError):
        resolver.resolve("10.0.1.25", ["192.0.2.1:not-a-port"])
    with pytest.raises(ClientIdentityError):
        resolver.resolve("10.0.1.25", ["not-an-ip, 192.0.2.1"])
    with pytest.raises(ClientIdentityError, match="too long"):
        resolver.resolve("10.0.1.25", [", ".join(["192.0.2.1"] * 33)])


def test_non_network_test_client_has_a_deterministic_nonproduction_path(tmp_path: Path) -> None:
    resolver = ClientIdentityResolver(settings(tmp_path))

    assert resolver.resolve("testclient", ["203.0.113.99"]) == resolver.resolve("127.0.0.1", [])


def test_api_persists_only_the_digest_and_rejects_an_unverifiable_path(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    repository = ProjectRepository(configured.database_path, configured.project_ttl_seconds)
    sender = MemoryEmailSender()
    auth = AuthService(sender, configured.session_ttl_seconds, repository)

    with TestClient(
        create_app(configured, repository, auth),
        client=("10.0.1.25", 50_000),
    ) as client:
        accepted = client.post(
            "/api/v1/auth/email-code",
            headers={"x-forwarded-for": "203.0.113.99"},
            json={"email": "Identity@Example.com"},
        )
        rejected = client.post(
            "/api/v1/auth/email-code",
            json={"email": "missing-path@example.com"},
        )
        duplicated = client.post(
            "/api/v1/auth/email-code",
            headers=[
                ("x-forwarded-for", "203.0.113.99"),
                ("x-forwarded-for", "198.51.100.20"),
            ],
            json={"email": "duplicate-path@example.com"},
        )

    with TestClient(
        create_app(configured, repository, auth),
        client=("192.0.2.25", 50_001),
    ) as direct_client:
        direct = direct_client.post(
            "/api/v1/auth/email-code",
            headers={"x-forwarded-for": "203.0.113.250"},
            json={"email": "direct@example.com"},
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "invalid-client-identity"
    assert duplicated.status_code == 400
    assert duplicated.json()["code"] == "invalid-client-identity"
    assert direct.status_code == 200

    expected = hmac.new(
        IDENTITY_KEY.encode(),
        b"203.0.113.99",
        hashlib.sha256,
    ).hexdigest()
    expected_direct = hmac.new(
        IDENTITY_KEY.encode(),
        b"192.0.2.25",
        hashlib.sha256,
    ).hexdigest()
    with sqlite3.connect(configured.database_path) as connection:
        stored = connection.execute(
            "SELECT client_key, email FROM auth_requests ORDER BY email"
        ).fetchall()

    assert stored == [
        (expected_direct, "direct@example.com"),
        (expected, "identity@example.com"),
    ]
    assert all("203.0.113" not in client_key for client_key, _email in stored)
