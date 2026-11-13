"""This laptop's addresses on local networks (the router's LAN, a hotspot it shares, ...)."""

from __future__ import annotations

import fcntl
import socket
import struct

SIOCGIFADDR = 0x8915  # Linux ioctl: an interface's IPv4 address
VIRTUAL = ("lo", "docker", "br-", "veth", "virbr", "tun", "tap", "wg")


def lan_address() -> str:
    """The IPv4 address of the interface used for outbound traffic (127.0.0.1 when offline)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET address: nothing is sent, it only picks a route
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def lan_addresses() -> list[tuple[str, str]]:
    """(interface, IPv4) of every physical interface that has an address, outbound one first."""
    found: list[tuple[str, str]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            if name.startswith(VIRTUAL):
                continue
            try:
                packed = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, struct.pack("256s", name[:15].encode()))
            except OSError:
                continue  # no IPv4 address
            found.append((name, socket.inet_ntoa(packed[20:24])))
    primary = lan_address()
    return sorted(found, key=lambda item: item[1] != primary)
