"""Versioned HMAC bearer tokens for immutable revision shares."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass
from uuid import UUID

TOKEN_FORMAT_VERSION = 1
TOKEN_PATTERN = re.compile(r"^s1\.([0-9a-f]{32})\.([A-Za-z0-9_-]{43})$")


@dataclass(frozen=True)
class ShareTokenCodec:
    """Issue reconstructable tokens while persisting only their SHA-256 digest."""

    keys: dict[int, bytes]
    current_key_version: int

    @classmethod
    def from_strings(
        cls,
        keys: tuple[tuple[int, str], ...],
        current_key_version: int,
    ) -> ShareTokenCodec:
        resolved = {version: value.encode("utf-8") for version, value in keys}
        if len(resolved) != len(keys):
            raise ValueError("Share token key versions must be unique.")
        if current_key_version not in resolved:
            raise ValueError("The current share token key version is missing from the key ring.")
        if any(version < 1 for version in resolved):
            raise ValueError("Share token key versions must be positive integers.")
        if any(len(value) < 32 for value in resolved.values()):
            raise ValueError("Every share token key must contain at least 32 UTF-8 bytes.")
        return cls(keys=resolved, current_key_version=current_key_version)

    @staticmethod
    def parse_public_id(token: str) -> UUID | None:
        if len(token) != 79:
            return None
        match = TOKEN_PATTERN.fullmatch(token)
        if match is None:
            return None
        return UUID(hex=match.group(1))

    def issue(self, public_id: UUID, key_version: int | None = None) -> str:
        resolved_version = key_version or self.current_key_version
        key = self.keys.get(resolved_version)
        if key is None:
            raise ValueError(f"Unknown share token key version: {resolved_version}")
        prefix = f"s{TOKEN_FORMAT_VERSION}.{public_id.hex}"
        mac = hmac.new(key, prefix.encode("ascii"), hashlib.sha256).digest()
        encoded_mac = base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")
        return f"{prefix}.{encoded_mac}"

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def verify(
        self,
        submitted_token: str,
        *,
        public_id: UUID,
        key_version: int,
        stored_digest: str,
    ) -> bool:
        try:
            expected_token = self.issue(public_id, key_version)
        except ValueError:
            expected_token = self.issue(public_id)
        expected_digest = self.digest(expected_token)
        try:
            token_matches = hmac.compare_digest(submitted_token, expected_token)
        except TypeError:
            token_matches = False
        digest_matches = hmac.compare_digest(stored_digest, expected_digest)
        return token_matches and digest_matches
