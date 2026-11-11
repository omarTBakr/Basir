"""This laptop's address on the local network."""

from __future__ import annotations

import socket


def lan_address() -> str:
    """The IPv4 address of the interface used for outbound traffic (127.0.0.1 when offline)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET address: nothing is sent, it only picks a route
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"
