"""Who may call this API, and how often.

Two endpoints here spend real money or real time on each request:
``/evaluate`` calls an LLM, ``/sentiment`` runs FinBERT over freshly fetched
news. A third, ``/feedback/resolve``, writes. Exposing those without a door is
how a research service becomes someone else's free inference endpoint.

The default is chosen so that the safe thing happens when nothing is
configured:

- **Keys configured** (``EVALUATOR_API_KEYS``): every endpoint except the
  dashboard, its static files and ``/health`` needs ``X-API-Key``.
- **No keys configured**: the costly endpoints answer **only to loopback**, so
  a laptop stays usable and a service exposed to a network by accident is not.

Keys are compared with ``hmac.compare_digest`` -- a plain ``==`` on a secret
leaks its prefix through timing. Rate limits are token buckets held in memory,
which is per process: two workers mean two buckets, and a real deployment
should move this to the proxy or to Redis. It is documented rather than
pretended away.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
API_KEYS_ENV = "EVALUATOR_API_KEYS"

#: Paths anyone may reach: the dashboard itself, its assets, and liveness.
PUBLIC_PREFIXES = ("/static", "/docs", "/redoc", "/openapi.json")
PUBLIC_PATHS = ("/", "/health", "/favicon.ico")

#: Endpoints that cost money or time on every call.
COSTLY_PREFIXES = ("/evaluate", "/sentiment", "/feedback/resolve")

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


@dataclass
class Limit:
    """A token bucket: `capacity` requests, refilled over `per_seconds`."""

    capacity: int
    per_seconds: float

    @property
    def rate(self) -> float:
        return self.capacity / self.per_seconds


#: Per identity (API key, or client address when no keys are configured).
LIMITS = {
    "/evaluate": Limit(10, 3600),
    "/sentiment": Limit(30, 3600),
    "default": Limit(240, 60),
}


@dataclass
class _Bucket:
    tokens: float
    #: Set from the same clock reading that fills it. A default_factory here
    #: would capture time.monotonic at class-definition time, so a bucket and
    #: the code refilling it could disagree about what "now" is.
    updated: float


class RateLimiter:
    """Token buckets per (identity, bucket name), in this process only."""

    def __init__(self, limits: dict[str, Limit] | None = None) -> None:
        self.limits = limits or LIMITS
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def bucket_for(self, path: str) -> str:
        for prefix in self.limits:
            if prefix != "default" and path.startswith(prefix):
                return prefix
        return "default"

    def check(self, identity: str, path: str) -> tuple[bool, int]:
        """(allowed, seconds until a token is next available)."""
        name = self.bucket_for(path)
        limit = self.limits[name]
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get((identity, name))
            if bucket is None:
                bucket = _Bucket(tokens=float(limit.capacity), updated=now)
                self._buckets[(identity, name)] = bucket
            bucket.tokens = min(limit.capacity, bucket.tokens + (now - bucket.updated) * limit.rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            return False, max(1, int((1.0 - bucket.tokens) / limit.rate) + 1)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


limiter = RateLimiter()


def configured_keys() -> set[str]:
    raw = os.environ.get(API_KEYS_ENV, "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def is_costly(path: str) -> bool:
    return path.startswith(COSTLY_PREFIXES)


def key_is_valid(presented: str | None, keys: set[str]) -> bool:
    if not presented:
        return False
    # compare_digest against every key: a plain equality test would leak how
    # much of a key matched through how long the comparison took.
    return any(hmac.compare_digest(presented, key) for key in keys)


def identify(presented: str | None, client: str | None) -> str:
    """What the rate limit counts against: the key if there is one, else the caller."""
    if presented:
        return f"key:{presented[:8]}"
    return f"ip:{client or 'unknown'}"


def authorize(path: str, presented: str | None, client: str | None) -> tuple[bool, str]:
    """(allowed, reason). Reason is only meaningful when refused."""
    if is_public(path):
        return True, ""

    keys = configured_keys()
    if keys:
        if key_is_valid(presented, keys):
            return True, ""
        return False, f"missing or invalid {API_KEY_HEADER}"

    if is_costly(path) and (client or "") not in LOOPBACK:
        return False, (
            f"{path} is limited to local callers until {API_KEYS_ENV} is set. "
            "It spends an LLM call or a model run on every request."
        )
    return True, ""


__all__ = [
    "API_KEYS_ENV",
    "API_KEY_HEADER",
    "LIMITS",
    "Limit",
    "RateLimiter",
    "authorize",
    "configured_keys",
    "identify",
    "is_costly",
    "is_public",
    "key_is_valid",
    "limiter",
]
