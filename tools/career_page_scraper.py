"""
Scrapes company career pages using Playwright for JS rendering,
then uses Claude Haiku to extract structured job listings from
pre-filtered page text (reducing tokens by ~70% vs raw HTML).
"""

import json
import logging
import re
from typing import Any

import anthropic
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

from models import JobListing, JobSource
from utils.rate_limiter import limiter

logger = logging.getLogger(__name__)

# Haiku is 25x cheaper than Opus and fast enough for structured extraction
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

# Words that signal a line is job-relevant — used to pre-filter page text
_JOB_SIGNALS = {
    "engineer", "developer", "software", "senior", "staff", "principal",
    "backend", "platform", "infrastructure", "architect", "scientist",
    "apply", "job", "role", "opening", "position", "full-time",
    "seattle", "san francisco", "sunnyvale", "mountain view", "new york",
    "remote", "california", "washington", "texas", "new york",
    "java", "python", "distributed", "cloud", "ai", "ml", "llm",
}

EXTRACTION_SYSTEM = (
    "Extract software engineering job listings from career page text. "
    "Return ONLY a JSON array. Each object: "
    "{\"title\": str, \"location\": str, \"url\": str, \"h1b_mentioned\": bool}. "
    "Empty array if none found. No explanation outside JSON."
)


def pre_filter_text(raw: str, max_chars: int = 6000) -> str:
    """
    Keep only lines that contain job-signal words.
    Reduces token usage by ~70% on typical career pages.
    """
    lines = raw.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue
        lower = stripped.lower()
        if any(word in lower for word in _JOB_SIGNALS):
            kept.append(stripped)
    filtered = "\n".join(kept)
    # Hard cap — never send more than max_chars to Claude
    return filtered[:max_chars]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=12))
async def _load_page_text(url: str) -> str:
    """
    Launch a visible (non-headless) Chromium window to avoid bot detection.
    Scrolls to trigger lazy-loaded job listings before extracting text.
    Running non-headless is intentional — sites like Microsoft and LinkedIn
    detect and block headless browsers via JS fingerprinting.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,       # Visible window bypasses bot detection
            slow_mo=80,           # Slight delay on each action — more human-like
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--window-size=1280,900",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()
        try:
            await limiter.wait(limiter.extract_domain(url))

            try:
                await page.goto(url, wait_until="networkidle", timeout=40000)
            except PlaywrightTimeout:
                logger.warning("networkidle timeout on %s — falling back to domcontentloaded", url)
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)

            # Extra settle time for SPAs and lazy-loaded content
            await page.wait_for_timeout(4000)

            # Scroll down in steps to trigger lazy-loading of job cards
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await page.wait_for_timeout(800)

            # Scroll back to top to capture any sticky/header content
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

            text = await page.inner_text("body")
        finally:
            await browser.close()

    return text


def _parse_json_response(raw: str) -> list[dict[str, Any]]:
    """Extract the JSON array from Claude's response."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse Claude response: %s", raw[:200])
    return []


async def scrape_career_page(
    company_name: str,
    career_url: str,
    role_keywords: list[str],
    known_h1b_sponsor: bool = True,
) -> list[JobListing]:
    """
    Scrape a company career page and return matching job listings.
    Page text is pre-filtered before sending to Claude to minimise token usage.
    """
    logger.info("Scraping: %s", company_name)

    try:
        raw_text = await _load_page_text(career_url)
    except Exception as exc:
        logger.error("Failed to load %s: %s", company_name, exc)
        return []

    filtered_text = pre_filter_text(raw_text)

    if len(filtered_text.strip()) < 80:
        logger.warning("Too little usable text from %s — skipping Claude call", company_name)
        return []

    prompt = (
        f"Company: {company_name}\n"
        f"Match keywords: {', '.join(role_keywords)}\n"
        f"Location: US or Remote only\n\n"
        f"{filtered_text}"
    )

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
    except Exception as exc:
        logger.error("Claude extraction failed for %s: %s", company_name, exc)
        return []

    listings: list[JobListing] = []
    for item in _parse_json_response(raw):
        try:
            listings.append(JobListing(
                title=item.get("title", ""),
                company=company_name,
                location=item.get("location", ""),
                url=item.get("url", career_url),
                description=item.get("description", ""),
                source=JobSource.CAREER_PAGE,
                h1b_mentioned=bool(item.get("h1b_mentioned", False)),
                known_h1b_sponsor=known_h1b_sponsor,
            ))
        except Exception as exc:
            logger.debug("Skipping malformed item: %s — %s", item, exc)

    logger.info("Extracted %d jobs from %s", len(listings), company_name)
    return listings
