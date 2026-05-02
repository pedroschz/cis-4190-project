"""HTTP fetcher with polite throttling and retry."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

USER_AGENT = (
    "CIS5190-FinalProject/1.0 (academic research; Penn CIS 4190/5190 Spring 2026; "
    "contact via GitHub issue)"
)

# Per-host minimum interval between requests (seconds). Polite default = 1 req/sec.
HOST_MIN_INTERVAL = 1.0

_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
)

_last_request_at: dict[str, float] = defaultdict(float)
_lock = threading.Lock()


def _throttle(host: str) -> None:
    with _lock:
        now = time.time()
        delta = now - _last_request_at[host]
        if delta < HOST_MIN_INTERVAL:
            time.sleep(HOST_MIN_INTERVAL - delta)
        _last_request_at[host] = time.time()


class HttpError(Exception):
    """Raised when a request fails after retries or returns a non-2xx status."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def _do_get(url: str, timeout: float) -> requests.Response:
    return _session.get(url, timeout=timeout, allow_redirects=True)


def get(url: str, *, timeout: float = 20.0) -> Optional[str]:
    """Fetch a URL, returning the response text or None on failure.

    Returns None for non-2xx statuses or repeated network failures. Caller
    is responsible for downstream Wayback-fallback behavior.
    """
    host = requests.utils.urlparse(url).netloc
    _throttle(host)
    try:
        resp = _do_get(url, timeout)
    except (requests.RequestException, RetryError) as e:
        logger.warning("network error fetching %s: %s", url, e)
        return None

    if resp.status_code != 200:
        logger.info("non-200 status %d for %s", resp.status_code, url)
        return None
    return resp.text
