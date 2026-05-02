"""Wayback Machine fallback for failed live fetches."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from src.scrape._http import HOST_MIN_INTERVAL, USER_AGENT, get

logger = logging.getLogger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"


def latest_snapshot_url(original_url: str) -> Optional[str]:
    """Return the most recent Wayback snapshot URL for `original_url`, or None."""
    params = {
        "url": original_url,
        "output": "json",
        "limit": "-1",  # latest first
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
    }
    try:
        resp = requests.get(
            CDX_ENDPOINT,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
    except requests.RequestException as e:
        logger.warning("CDX request failed for %s: %s", original_url, e)
        return None

    if resp.status_code != 200:
        return None
    try:
        rows = resp.json()
    except ValueError:
        return None
    # First row is the header
    if len(rows) < 2:
        return None
    timestamp, original, _status = rows[1]
    return f"https://web.archive.org/web/{timestamp}/{original}"


def fetch_via_wayback(original_url: str) -> Optional[str]:
    """Fetch the latest Wayback snapshot HTML for `original_url`."""
    snap = latest_snapshot_url(original_url)
    if not snap:
        return None
    return get(snap, timeout=30.0)
