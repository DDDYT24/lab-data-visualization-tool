"""Trusted-proxy client identity for public abuse-control decisions."""

from __future__ import annotations

import hashlib
import hmac
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network

from fastapi import Request

from .config import Settings

MAX_FORWARDED_ENTRIES = 32


class ClientIdentityError(ValueError):
    """The network path cannot produce one trustworthy client identity."""


def _validate_port(value: str) -> None:
    if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 65_535:
        raise ClientIdentityError("The forwarded client port is invalid.")


def normalize_client_address(value: str) -> IPv4Address | IPv6Address:
    """Normalize supported proxy address forms without accepting ambiguous ports."""

    candidate = value.strip()
    if not candidate or any(character in candidate for character in ('"', "'", "%")):
        raise ClientIdentityError("The client address is malformed.")

    host = candidate
    if candidate.startswith("["):
        closing = candidate.find("]")
        if closing < 0:
            raise ClientIdentityError("The bracketed client address is malformed.")
        host = candidate[1:closing]
        suffix = candidate[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or len(suffix) == 1:
                raise ClientIdentityError("The bracketed client address suffix is invalid.")
            _validate_port(suffix[1:])
    else:
        try:
            address = ip_address(candidate)
        except ValueError:
            if candidate.count(":") != 1:
                raise ClientIdentityError("The client address is malformed.") from None
            host, port = candidate.rsplit(":", 1)
            _validate_port(port)
        else:
            if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
                return address.ipv4_mapped
            return address

    try:
        address = ip_address(host)
    except ValueError:
        raise ClientIdentityError("The client address is malformed.") from None
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


class ClientIdentityResolver:
    """Resolve and pseudonymize the nearest untrusted network address."""

    def __init__(self, settings: Settings) -> None:
        self.environment = settings.environment
        self.trusted_proxy_hops = settings.trusted_proxy_hops
        self.trusted_networks = tuple(
            ip_network(value, strict=False) for value in settings.trusted_proxy_cidrs
        )
        self.digest_key = settings.client_identity_key.encode("utf-8")

    def resolve_request(self, request: Request) -> str:
        peer_host = request.client.host if request.client is not None else ""
        return self.resolve(peer_host, request.headers.getlist("x-forwarded-for"))

    def resolve(self, peer_host: str, forwarded_for_values: list[str]) -> str:
        try:
            peer = normalize_client_address(peer_host)
        except ClientIdentityError:
            if self.environment == "production":
                raise
            peer = IPv4Address("127.0.0.1")

        if not self._is_trusted(peer):
            return self._digest(peer)

        if self.trusted_proxy_hops < 1:
            raise ClientIdentityError("A trusted proxy requires a positive proxy-hop count.")
        if len(forwarded_for_values) != 1:
            raise ClientIdentityError("Exactly one X-Forwarded-For header is required.")

        raw_entries = [item.strip() for item in forwarded_for_values[0].split(",")]
        if (
            not raw_entries
            or len(raw_entries) > MAX_FORWARDED_ENTRIES
            or any(not item for item in raw_entries)
        ):
            raise ClientIdentityError("The X-Forwarded-For chain is malformed or too long.")
        entries = [normalize_client_address(item) for item in raw_entries]
        if len(entries) < self.trusted_proxy_hops:
            raise ClientIdentityError("The X-Forwarded-For chain has insufficient proxy hops.")

        candidate_index = len(entries) - self.trusted_proxy_hops
        trusted_suffix = entries[candidate_index + 1 :]
        if any(not self._is_trusted(address) for address in trusted_suffix):
            raise ClientIdentityError("The X-Forwarded-For trusted proxy suffix is invalid.")

        candidate = entries[candidate_index]
        if self._is_trusted(candidate):
            raise ClientIdentityError(
                "The configured proxy-hop count does not reach an untrusted client."
            )
        return self._digest(candidate)

    def _is_trusted(self, address: IPv4Address | IPv6Address) -> bool:
        return any(
            address.version == network.version and address in network
            for network in self.trusted_networks
        )

    def _digest(self, address: IPv4Address | IPv6Address) -> str:
        return hmac.new(
            self.digest_key,
            str(address).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
