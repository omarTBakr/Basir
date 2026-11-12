"""Security tests against XSS: no script writes HTML from strings, and templates escape server data."""

from __future__ import annotations

from pathlib import Path
import re

from fastapi.testclient import TestClient

API = Path(__file__).resolve().parent.parent / "basir" / "api"
HTML_SINKS = re.compile(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write")


def test_no_script_builds_html_from_strings() -> None:
    sources = [*API.glob("static/*.js"), *API.glob("phone/*.js"), *API.glob("phone/*.html")]
    sources += list(API.glob("templates/**/*.html"))
    assert sources
    offenders = [
        f"{path.relative_to(API)}:{number}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if HTML_SINKS.search(line)
    ]
    assert offenders == []


def test_device_names_are_escaped_on_the_devices_page(client: TestClient) -> None:
    client.app.state.ingest.tokens.create("x<img>")  # type: ignore[attr-defined]  # only the page can show it
    page = client.get("/partials/devices").text
    assert "x<img>" not in page and "x&lt;img&gt;" in page
