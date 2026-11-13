"""Reachability of the HTTP services the UI depends on: Grafana, Prometheus and its scrape of this server."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

TIMEOUT_S = 1.5
SCRAPE_JOB = "basir"  # the Prometheus job that scrapes this server's /metrics


@dataclass(frozen=True)
class ServiceStatus:
    name: str
    url: str
    up: bool
    detail: str


async def probe(name: str, base_url: str, path: str) -> ServiceStatus:
    """GET `base_url + path`; up means a 2xx answer within the timeout."""
    url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.get(url + path)
        return ServiceStatus(name, url, response.is_success, f"HTTP {response.status_code}")
    except httpx.HTTPError as e:
        return ServiceStatus(name, url, False, f"not reachable ({type(e).__name__})")


async def scrape_status(prometheus_url: str) -> ServiceStatus:
    """Whether Prometheus's `basir` job currently scrapes this server successfully."""
    url = prometheus_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.get(f"{url}/api/v1/targets", params={"state": "active"})
        targets = response.json()["data"]["activeTargets"]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return ServiceStatus("Prometheus scrape", url, False, f"not reachable ({type(e).__name__})")
    ours = [t for t in targets if t.get("labels", {}).get("job") == SCRAPE_JOB]
    if not ours:
        return ServiceStatus("Prometheus scrape", url, False, f"no '{SCRAPE_JOB}' job configured")
    target = ours[0]
    health = target.get("health", "unknown")
    detail = f"{target.get('scrapeUrl', '')}: {health}" + (
        f" ({target['lastError']})" if target.get("lastError") else ""
    )
    return ServiceStatus("Prometheus scrape", url, health == "up", detail)


async def gpu_utilization(prometheus_url: str) -> float | None:
    """GPU utilization in percent from the GPU exporter, via Prometheus; None when unknown."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.get(
                f"{prometheus_url.rstrip('/')}/api/v1/query", params={"query": "100 * nvidia_smi_utilization_gpu_ratio"}
            )
        result = response.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
