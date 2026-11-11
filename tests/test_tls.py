"""Tests for basir/api/tls.py: the phone listener's self-signed certificate."""

from __future__ import annotations

import ipaddress
from pathlib import Path

from cryptography import x509

from basir.api.tls import ensure_certificate


def test_certificate_covers_the_lan_address_and_is_reused(tmp_path: Path) -> None:
    cert, key = ensure_certificate(tmp_path, ["192.168.1.11", "10.42.0.1"])
    names = x509.load_pem_x509_certificate(cert.read_bytes()).extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    )
    ips = names.value.get_values_for_type(x509.IPAddress)
    assert ipaddress.ip_address("192.168.1.11") in ips and ipaddress.ip_address("10.42.0.1") in ips
    assert "localhost" in names.value.get_values_for_type(x509.DNSName)
    assert key.stat().st_mode & 0o077 == 0  # private key readable by the owner only
    stamp = cert.stat().st_mtime_ns
    assert ensure_certificate(tmp_path, ["10.42.0.1"])[0].stat().st_mtime_ns == stamp  # covered: reused
    ensure_certificate(tmp_path, ["10.0.0.5"])  # a new address: remade
    names = x509.load_pem_x509_certificate(cert.read_bytes()).extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    )
    assert ipaddress.ip_address("10.0.0.5") in names.value.get_values_for_type(x509.IPAddress)
