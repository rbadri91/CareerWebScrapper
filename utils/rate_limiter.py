"""Async rate limiter with randomized delays to avoid bot detection."""

import asyncio
import random
import time
from collections import defaultdict


class RateLimiter:
    """
    Per-domain rate limiter. Enforces a minimum delay between requests
    to the same domain, with optional jitter to appear more human-like.
    """

    # Conservative defaults per domain (requests per minute)
    DOMAIN_LIMITS = {
        "linkedin.com": {"min_delay": 4.0, "max_delay": 9.0},
        "indeed.com":   {"min_delay": 2.0, "max_delay": 5.0},
        "wellfound.com":{"min_delay": 2.0, "max_delay": 4.0},
        "default":      {"min_delay": 1.5, "max_delay": 3.5},
    }

    def __init__(self):
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _get_limits(self, domain: str) -> dict:
        for key in self.DOMAIN_LIMITS:
            if key in domain:
                return self.DOMAIN_LIMITS[key]
        return self.DOMAIN_LIMITS["default"]

    async def wait(self, domain: str) -> None:
        """Wait the appropriate amount of time before the next request to domain."""
        async with self._locks[domain]:
            limits = self._get_limits(domain)
            delay = random.uniform(limits["min_delay"], limits["max_delay"])
            elapsed = time.monotonic() - self._last_request[domain]
            remaining = delay - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request[domain] = time.monotonic()

    @staticmethod
    def extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lstrip("www.")
        except Exception:
            return "default"


# Global singleton
limiter = RateLimiter()
