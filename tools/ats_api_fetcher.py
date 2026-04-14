"""
Direct JSON API fetchers for company career pages that block browser scraping.

Each fetcher hits the company's internal jobs API (discovered via DevTools),
returns structured job data without any browser overhead or Claude token cost.

Supported:
  - Greenhouse ATS   → DoorDash, Databricks, Airbnb (public, no auth)
  - Apple Jobs API   → jobs.apple.com/api/role/search
  - Netflix Jobs     → jobs.netflix.com/api/search
  - Meta GraphQL     → metacareers.com/graphql (doc_id captured via DevTools)
"""

import json
import logging
import os
from typing import Any
from urllib.parse import urlencode

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from models import JobListing, JobSource

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_SENIOR_TERMS = {
    "senior", "staff", "principal", "lead", "engineer", "developer",
    "architect", "sre", "platform", "backend", "infrastructure",
}


def _relevant(title: str, keywords: list[str]) -> bool:
    """True if the title contains any of the role keywords or senior-level terms."""
    lower = title.lower()
    return any(kw.lower() in lower for kw in keywords) or any(
        t in lower for t in _SENIOR_TERMS
    )


async def _get_json(url: str, params: dict | None = None) -> Any:
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)


# ──────────────────────────────────────────────
# Greenhouse ATS  (DoorDash, Databricks, Airbnb)
# ──────────────────────────────────────────────

GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
async def fetch_greenhouse_jobs(
    company_name: str,
    slug: str,
    keywords: list[str],
    known_h1b_sponsor: bool = True,
) -> list[JobListing]:
    """
    Greenhouse public jobs board API — no auth, returns all open roles as JSON.
    content=false skips the full HTML job description (saves bandwidth).
    """
    url = GREENHOUSE_BASE.format(slug=slug)
    logger.info("Greenhouse API: %s (%s)", company_name, slug)

    try:
        data = await _get_json(url, params={"content": "false"})
    except Exception as exc:
        logger.error("Greenhouse fetch failed for %s: %s", company_name, exc)
        return []

    listings: list[JobListing] = []
    for job in data.get("jobs", []):
        title = job.get("title", "").strip()
        if not title or not _relevant(title, keywords):
            continue
        location = job.get("location", {}).get("name", "")
        # Filter non-US locations
        if location and not any(
            us in location for us in ["United States", "US", "Remote", "CA,", "WA,", "NY,", "TX,"]
        ):
            continue
        listings.append(JobListing(
            title=title,
            company=company_name,
            location=location,
            url=job.get("absolute_url", ""),
            source=JobSource.CAREER_PAGE,
            known_h1b_sponsor=known_h1b_sponsor,
        ))

    logger.info("Greenhouse: %d matching jobs from %s", len(listings), company_name)
    return listings


# ──────────────────────────────────────────────
# Apple Jobs API
# ──────────────────────────────────────────────

APPLE_SEARCH_URL = "https://jobs.apple.com/api/role/search"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
async def fetch_apple_jobs(
    keywords: list[str],
    known_h1b_sponsor: bool = True,
) -> list[JobListing]:
    """
    Apple Jobs REST API. Filters for US locations and Senior level.
    Paginates up to 3 pages (60 results) to keep token cost low.
    Session cookie read from APPLE_COOKIE env var.
    """
    logger.info("Apple Jobs API")
    listings: list[JobListing] = []

    apple_headers = {
        **_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://jobs.apple.com/en-us/search",
        "Origin": "https://jobs.apple.com",
        "X-Requested-With": "XMLHttpRequest",
    }
    apple_cookie = os.getenv("APPLE_COOKIE", "")
    if apple_cookie:
        apple_headers["Cookie"] = apple_cookie

    for page in range(1, 4):
        params = {
            "filters[postingpostLocation][0]": "postLocation-USA",
            "filters[hierarchyLevel][0]": "level-Senior",
            "page": page,
            "locale": "en-US",
        }
        try:
            async with aiohttp.ClientSession(headers=apple_headers) as session:
                async with session.get(
                    APPLE_SEARCH_URL, params=params,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)

            results = data.get("searchResults", [])
            if not results:
                break

            for r in results:
                title = (r.get("postingTitle") or r.get("title") or "").strip()
                if not title or not _relevant(title, keywords):
                    continue

                locs = r.get("locations", [])
                location = locs[0].get("name", "") if locs else ""
                posting_id = r.get("positionId", "")
                job_url = (
                    f"https://jobs.apple.com/en-us/details/{posting_id}"
                    if posting_id else ""
                )
                listings.append(JobListing(
                    title=title,
                    company="Apple",
                    location=location,
                    url=job_url,
                    source=JobSource.CAREER_PAGE,
                    known_h1b_sponsor=known_h1b_sponsor,
                ))

            # Fewer results than a full page → no more pages
            if len(results) < 20:
                break

        except Exception as exc:
            logger.error("Apple Jobs API failed (page %d): %s", page, exc)
            break

    logger.info("Apple: %d matching jobs", len(listings))
    return listings


# ──────────────────────────────────────────────
# Netflix Jobs API
# ──────────────────────────────────────────────

NETFLIX_SEARCH_URL = "https://jobs.netflix.com/api/search"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
async def fetch_netflix_jobs(
    keywords: list[str],
    known_h1b_sponsor: bool = True,
) -> list[JobListing]:
    """
    Netflix Jobs search API. Returns structured JSON with postings.
    Tries both known response schemas (schema changed in 2024).
    """
    logger.info("Netflix Jobs API")

    try:
        data = await _get_json(
            NETFLIX_SEARCH_URL,
            params={"q": "senior software engineer", "team": "Engineering"},
        )
    except Exception as exc:
        logger.error("Netflix Jobs API failed: %s", exc)
        return []

    # Handle two known response shapes
    postings = (
        data.get("records", {}).get("postings")
        or data.get("postings")
        or data.get("jobs")
        or []
    )

    listings: list[JobListing] = []
    for p in postings:
        title = (
            p.get("text") or p.get("title") or p.get("name") or ""
        ).strip()
        if not title or not _relevant(title, keywords):
            continue

        cats = p.get("categories", {})
        location = (
            cats.get("location") or p.get("location", {}).get("name", "") or ""
        )
        job_url = p.get("hostedUrl") or p.get("absolute_url") or p.get("url") or ""

        listings.append(JobListing(
            title=title,
            company="Netflix",
            location=location,
            url=job_url,
            source=JobSource.CAREER_PAGE,
            known_h1b_sponsor=known_h1b_sponsor,
        ))

    logger.info("Netflix: %d matching jobs", len(listings))
    return listings


# ──────────────────────────────────────────────
# Meta GraphQL API
# ──────────────────────────────────────────────

META_GRAPHQL_URL = "https://www.metacareers.com/graphql"

# doc_id captured from DevTools → Network → Fetch/XHR on metacareers.com/jobs
# Update this value if Meta rotates the compiled query hash.
META_DOC_ID = "26446976041587120"

# US offices provided by user via DevTools inspection
META_US_OFFICES = [
    "Aurora, IL", "Sunnyvale, CA", "Seattle, WA", "Irvine, CA",
    "Newark, CA", "Fremont, CA", "Pasadena, CA", "San Mateo, CA",
    "Cambridge, MA", "Menlo Park, CA", "San Diego, CA", "Denver, CO",
    "New York, NY", "Boston, MA", "Sausalito, CA", "Los Angeles, CA",
    "Mountain View, CA", "Redmond, WA", "Bellevue, WA", "Vancouver, WA",
]

def _meta_headers() -> dict:
    """Build Meta request headers, injecting session cookie from env if set."""
    h = {
        **_HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.metacareers.com",
        "Referer": "https://www.metacareers.com/jobs/",
        "x-fb-friendly-name": "CareersJobSearchResultsQuery",
    }
    cookie = os.getenv("META_COOKIE", "")
    if cookie:
        h["Cookie"] = cookie
    return h


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=3, max=10))
async def fetch_meta_jobs(
    keywords: list[str],
    known_h1b_sponsor: bool = True,
    page: int = 1,
) -> list[JobListing]:
    """
    Meta careers GraphQL API.
    Variables and doc_id captured from DevTools on metacareers.com/jobs.
    Searches across all US offices for full-time engineering roles.
    """
    logger.info("Meta GraphQL API (page %d)", page)

    variables = {
        "search_input": {
            "q": None,
            "divisions": [],
            "offices": META_US_OFFICES,
            "roles": ["Full time employment"],
            "leadership_levels": [],
            "saved_jobs": [],
            "saved_searches": [],
            "sub_teams": [],
            "teams": [],
            "is_leadership": False,
            "is_remote_only": False,
            "sort_by_new": False,
            "page": page,
            "results_per_page": None,
        }
    }

    body = urlencode({
        "variables": json.dumps(variables),
        "doc_id": META_DOC_ID,
    })

    try:
        async with aiohttp.ClientSession(headers=_meta_headers()) as session:
            async with session.post(
                META_GRAPHQL_URL,
                data=body,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
    except Exception as exc:
        logger.error("Meta GraphQL failed: %s", exc)
        return []

    # Navigate response tree defensively
    results = (
        data.get("data", {})
            .get("job_search", {})
            .get("results", [])
    )
    if not results:
        # Some response shapes nest differently
        results = data.get("data", {}).get("results", [])

    listings: list[JobListing] = []
    for item in results:
        # Results can be nested under "job_posting" or directly as flat dict
        job = item.get("job_posting") or item
        title = (job.get("title") or "").strip()
        if not title or not _relevant(title, keywords):
            continue

        # Location: offices list or single location field
        offices = job.get("offices", [])
        location = offices[0].get("name", "") if offices else job.get("location", "")

        job_id = job.get("id", "")
        job_url = (
            f"https://www.metacareers.com/jobs/{job_id}/"
            if job_id else "https://www.metacareers.com/jobs/"
        )

        listings.append(JobListing(
            title=title,
            company="Meta",
            location=location,
            url=job_url,
            source=JobSource.CAREER_PAGE,
            known_h1b_sponsor=known_h1b_sponsor,
        ))

    logger.info("Meta: %d matching jobs (page %d)", len(listings), page)
    return listings
