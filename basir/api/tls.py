"""A self-signed certificate for the phone listener, so a phone's browser may use its camera and AR.

Phone browsers only allow the camera and WebXR on secure (HTTPS) pages. The certificate is
made once, for this laptop's LAN addresses (router network, hotspot) and hostname, and remade
when an address is added or changes.
The phone shows a one-time warning ("connection not private"); accepting it is the only step.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
from pathlib import Path
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

VALID_DAYS = 825  # the longest validity browsers accept for a leaf certificate


def _covers(cert_path: Path, addresses: list[str]) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (OSError, ValueError, x509.ExtensionNotFound):
        return False
    still_valid = cert.not_valid_after_utc > dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    covered = names.get_values_for_type(x509.IPAddress)
    return still_valid and all(ipaddress.ip_address(a) in covered for a in addresses)


def ensure_certificate(directory: Path, addresses: list[str]) -> tuple[Path, Path]:
    """(certificate, key) PEM files valid for `addresses`, `localhost` and this hostname; made if needed."""
    cert_path, key_path = directory / "phone-listener.crt", directory / "phone-listener.key"
    if cert_path.is_file() and key_path.is_file() and _covers(cert_path, addresses):
        return cert_path, key_path
    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Basir on {host}")])
    now = dt.datetime.now(dt.UTC)
    alt_names: list[x509.GeneralName] = [x509.DNSName("localhost"), x509.DNSName(host)]
    for ip in sorted({*addresses, "127.0.0.1"}):
        alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    key_path.chmod(0o600)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path
