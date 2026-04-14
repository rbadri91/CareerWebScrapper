"""
Scrapes public job boards: LinkedIn public search, Indeed, and Wellfound.
No authentication required. Uses Playwright + BeautifulSoup.
"""

import logging
import re
from urllib.parse import urlencode, quote_plus

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

from models import JobListing, JobSource
from utils.rate_limiter import limiter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# LinkedIn Public Job Search (no login needed)
# ──────────────────────────────────────────────

LINKEDIN_BASE = "https://www.linkedin.com/jobs/search/"

# Experience level codes: 4 = Senior, 5 = Director, 3 = Mid
LINKEDIN_SENIORITY = "4"


async def _get_page_html(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        try:
            await limiter.wait(limiter.extract_domain(url))
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(3000)
            html = await page.content()
        except PlaywrightTimeout:
            logger.warning("Timeout on %s", url)
            html = ""
        finally:
            await browser.close()
    return html


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=5, max=15))
async def search_linkedin_jobs(
    title: str,
    location: str = "United States",
    max_results: int = 30,
    extra_keywords: str = "visa sponsorship H1B",
) -> list[JobListing]:
    """
    Search LinkedIn public jobs. Adds visa sponsorship to keywords
    to surface roles more likely to sponsor H-1B.
    """
    params = {
        "keywords": f"{title} {extra_keywords}",
        "location": location,
        "f_E": LINKEDIN_SENIORITY,
        "f_JT": "F",          # Full-time only
        "f_TPR": "r604800",   # Posted in last 7 days
        "sortBy": "DD",       # Most recent
    }
    url = LINKEDIN_BASE + "?" + urlencode(params)
    logger.info("LinkedIn search: %s", url)

    html = await _get_page_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_=re.compile(r"job-search-card|base-card"))
    if not cards:
        # Fallback: try li elements
        cards = soup.find_all("li", class_=re.compile(r"jobs-search"))

    listings: list[JobListing] = []
    for card in cards[:max_results]:
        try:
            title_el = card.find(["h3", "h4"], class_=re.compile(r"title|job-title"))
            company_el = card.find(["h4", "a"], class_=re.compile(r"company|subtitle"))
            location_el = card.find(class_=re.compile(r"location|metadata-item"))
            link_el = card.find("a", href=True)

            job_title = title_el.get_text(strip=True) if title_el else ""
            company = company_el.get_text(strip=True) if company_el else ""
            job_location = location_el.get_text(strip=True) if location_el else ""
            job_url = link_el["href"].split("?")[0] if link_el else ""

            if not job_title:
                continue

            card_text = card.get_text(" ").lower()
            h1b = any(kw in card_text for kw in ["h-1b", "h1b", "visa sponsor", "work authorization"])

            listings.append(JobListing(
                title=job_title,
                company=company,
                location=job_location,
                url=job_url,
                source=JobSource.LINKEDIN,
                h1b_mentioned=h1b,
                known_h1b_sponsor=False,  # will be enriched by agent
            ))
        except Exception as exc:
            logger.debug("Skipping LinkedIn card: %s", exc)

    logger.info("LinkedIn returned %d jobs", len(listings))
    return listings


# ──────────────────────────────────────────────
# Indeed Scraper
# ──────────────────────────────────────────────

INDEED_BASE = "https://www.indeed.com/jobs"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=12))
async def search_indeed_jobs(
    title: str,
    location: str = "United States",
    max_results: int = 30,
) -> list[JobListing]:
    """Search Indeed for job listings."""
    params = {
        "q": f'"{title}" "visa sponsorship" OR "H1B" OR "H-1B"',
        "l": location,
        "sort": "date",
        "jt": "fulltime",
        "explvl": "experienced_level",
    }
    url = INDEED_BASE + "?" + urlencode(params)
    logger.info("Indeed search: %s", url)

    html = await _get_page_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_=re.compile(r"job_seen_beacon|jobCard|resultContent"))

    listings: list[JobListing] = []
    for card in cards[:max_results]:
        try:
            title_el = card.find(["h2", "span"], class_=re.compile(r"jobTitle|title"))
            company_el = card.find(class_=re.compile(r"companyName|company"))
            location_el = card.find(class_=re.compile(r"companyLocation|location"))
            link_el = card.find("a", href=True)

            job_title = title_el.get_text(strip=True) if title_el else ""
            company = company_el.get_text(strip=True) if company_el else ""
            job_location = location_el.get_text(strip=True) if location_el else ""

            href = link_el["href"] if link_el else ""
            job_url = f"https://www.indeed.com{href}" if href.startswith("/") else href

            if not job_title:
                continue

            card_text = card.get_text(" ").lower()
            h1b = any(kw in card_text for kw in ["h-1b", "h1b", "visa sponsor", "sponsorship"])

            listings.append(JobListing(
                title=job_title,
                company=company,
                location=job_location,
                url=job_url,
                source=JobSource.INDEED,
                h1b_mentioned=h1b,
            ))
        except Exception as exc:
            logger.debug("Skipping Indeed card: %s", exc)

    logger.info("Indeed returned %d jobs", len(listings))
    return listings


# ──────────────────────────────────────────────
# Wellfound (AngelList) Scraper
# ──────────────────────────────────────────────

WELLFOUND_BASE = "https://wellfound.com/jobs"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10))
async def search_wellfound_jobs(
    title: str,
    location: str = "United States",
    max_results: int = 20,
) -> list[JobListing]:
    """Search Wellfound (AngelList) for startup/tech jobs."""
    slug = quote_plus(title.lower().replace(" ", "-"))
    url = f"{WELLFOUND_BASE}?q={slug}&l={quote_plus(location)}&visa=true"
    logger.info("Wellfound search: %s", url)

    html = await _get_page_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_=re.compile(r"styles_component|JobListingCard"))

    listings: list[JobListing] = []
    for card in cards[:max_results]:
        try:
            title_el = card.find(class_=re.compile(r"title|role"))
            company_el = card.find(class_=re.compile(r"company|startup"))
            location_el = card.find(class_=re.compile(r"location|remote"))
            link_el = card.find("a", href=True)

            job_title = title_el.get_text(strip=True) if title_el else ""
            company = company_el.get_text(strip=True) if company_el else ""
            job_location = location_el.get_text(strip=True) if location_el else ""
            href = link_el["href"] if link_el else ""
            job_url = f"https://wellfound.com{href}" if href.startswith("/") else href

            if not job_title:
                continue

            listings.append(JobListing(
                title=job_title,
                company=company,
                location=job_location,
                url=job_url,
                source=JobSource.WELLFOUND,
                h1b_mentioned=True,  # We filtered for visa=true in URL
            ))
        except Exception as exc:
            logger.debug("Skipping Wellfound card: %s", exc)

    logger.info("Wellfound returned %d jobs", len(listings))
    return listings
