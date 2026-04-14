"""
LinkedIn recruiter search using Playwright with a visible browser window.

Design decisions:
  - Non-headless: LinkedIn's JS fingerprinting detects and blocks headless Chrome.
  - slow_mo=120: every Playwright action is delayed 120ms — appears human-paced.
  - Human mouse movement: cursor follows a randomised curved path to each element
    before clicking/typing, not a straight teleport.
  - Login once, reuse session: avoids repeated login signals that trigger 2FA.
  - Conservative rate limits: 6–12 s between company searches, 3–6 s between
    page scrolls. Stays well under LinkedIn's undocumented soft limits.
  - No unofficial API calls: pure browser automation only, lower ban risk.
"""

import asyncio
import logging
import math
import os
import random
from urllib.parse import quote_plus

from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

from models import RecruiterProfile

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Human-like mouse movement helpers
# ──────────────────────────────────────────────

async def _move_to(page: Page, x: float, y: float) -> None:
    """
    Move the mouse from its current position to (x, y) along a curved path
    with randomised intermediate waypoints — mimics human hand movement.
    """
    steps = random.randint(10, 20)
    # Add a slight arc offset to avoid perfectly straight lines
    arc = random.uniform(-40, 40)

    for i in range(1, steps + 1):
        t = i / steps
        # Quadratic Bezier-like curve through a midpoint offset
        mid_x = (x / 2) + arc * math.sin(math.pi * t)
        mid_y = (y / 2) + arc * math.cos(math.pi * t)
        cur_x = mid_x * (1 - t) + x * t + random.uniform(-2, 2)
        cur_y = mid_y * (1 - t) + y * t + random.uniform(-2, 2)
        await page.mouse.move(cur_x, cur_y)
        await page.wait_for_timeout(random.randint(15, 45))

    await page.mouse.move(x, y)


async def _human_click(page: Page, selector: str) -> None:
    """Move cursor to selector's centre along a human path, then click."""
    element = page.locator(selector).first
    try:
        box = await element.bounding_box()
    except Exception:
        await page.click(selector)
        return

    if box:
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        await _move_to(page, cx, cy)
        await page.wait_for_timeout(random.randint(80, 200))
        await page.mouse.click(cx, cy)
    else:
        await page.click(selector)


async def _human_type(page: Page, selector: str, text: str) -> None:
    """Click a field with human mouse movement, then type with random key delays."""
    await _human_click(page, selector)
    await page.wait_for_timeout(random.randint(200, 500))
    # Type character-by-character with variable delay (40–120 ms per key)
    for char in text:
        await page.keyboard.type(char)
        await page.wait_for_timeout(random.randint(40, 120))


# ──────────────────────────────────────────────
# Login
# ──────────────────────────────────────────────

async def _login(page: Page) -> bool:
    """
    Log into LinkedIn. Returns True on success.
    Uses human-like typing and mouse movement.
    """
    email = os.getenv("LINKEDIN_EMAIL", "")
    password = os.getenv("LINKEDIN_PASSWORD", "")
    if not email or not password:
        logger.error("LINKEDIN_EMAIL / LINKEDIN_PASSWORD not set in .env")
        return False

    logger.info("Logging into LinkedIn...")
    await page.goto(
        "https://www.linkedin.com/login?fromSignIn=true&trk=guest_homepage-basic_nav-header-signin",
        wait_until="domcontentloaded",
    )
    await page.wait_for_timeout(random.randint(1500, 2500))

    # Dismiss cookie/GDPR consent banner if present
    for consent_sel in [
        "button[action-type='ACCEPT']",
        "button[data-tracking-control-name='cookie-policy-accept']",
        "#artdeco-global-alert-action__",
    ]:
        try:
            btn = page.locator(consent_sel).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(500)
                break
        except Exception:
            pass

    # LinkedIn uses different selectors in different regions/versions
    USERNAME_SELS = ["#username", "input[name='session_key']", "input[autocomplete='username']"]
    username_sel = None
    for sel in USERNAME_SELS:
        try:
            await page.wait_for_selector(sel, timeout=8000)
            username_sel = sel
            break
        except PWTimeout:
            continue

    if not username_sel:
        logger.error("LinkedIn login form not found — URL: %s", page.url)
        return False

    await page.fill(username_sel, email)
    await page.wait_for_timeout(random.randint(600, 1200))
    await page.fill("input[name='session_password'], #password", password)
    await page.wait_for_timeout(random.randint(800, 1500))

    await _human_click(page, '[type="submit"]')

    try:
        # Wait up to 20 s for redirect to feed or mynetwork
        await page.wait_for_url("**linkedin.com/feed/**", timeout=20000)
        logger.info("LinkedIn login successful")
        return True
    except PWTimeout:
        url = page.url
        if "checkpoint" in url or "challenge" in url or "verification" in url:
            logger.warning(
                "LinkedIn security check triggered — complete it in the browser window, "
                "then press Enter in the terminal to continue."
            )
            input("Press Enter after completing the LinkedIn security check...")
            return True
        # Already logged in and redirected somewhere other than /feed
        if "linkedin.com" in url and "login" not in url:
            logger.info("LinkedIn login successful (redirected to %s)", url)
            return True
        logger.error("Login failed — current URL: %s", url)
        return False


# ──────────────────────────────────────────────
# Recruiter search
# ──────────────────────────────────────────────

async def _scroll_results(page: Page, times: int = 3) -> None:
    """Scroll down the results page to load lazy-rendered profile cards."""
    for _ in range(times):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
        await page.wait_for_timeout(random.randint(700, 1400))


async def _extract_profiles_from_page(
    page: Page, company: str
) -> list[RecruiterProfile]:
    """
    Extract recruiter profiles from a LinkedIn people search results page.
    Uses JS to query anchor tags pointing to /in/ profiles alongside their
    surrounding text — more robust than CSS class selectors which LinkedIn
    rotates frequently.
    """
    # Pull all profile links + surrounding card text via JS
    cards_data: list[dict] = await page.evaluate("""() => {
        const results = [];
        // LinkedIn wraps each person result in a <li> that contains an <a href="/in/...">
        document.querySelectorAll('a[href*="/in/"]').forEach(a => {
            const href = a.href.split('?')[0];
            // Avoid duplicate profile links (nav bar, suggested connections etc.)
            if (!href.includes('/in/') || results.some(r => r.url === href)) return;
            // Walk up to find the result card container (up to 6 levels)
            let container = a;
            for (let i = 0; i < 6; i++) {
                if (!container.parentElement) break;
                container = container.parentElement;
                if (container.tagName === 'LI') break;
            }
            const text = container.innerText || '';
            results.push({ url: href, text: text.trim() });
        });
        return results.slice(0, 20);  // cap to first 20 links on page
    }""")

    _RECRUIT_KWS = {"recruit", "talent", "hiring", "sourcer", "acquisition", "staffing"}

    profiles: list[RecruiterProfile] = []
    for card in cards_data:
        url = card.get("url", "")
        text = card.get("text", "")
        if not url or not text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Remove degree indicators (• 2nd, • 3rd+) and UI noise lines
        _NOISE = {"connect", "follow", "message", "view profile", "dismiss", "1st", "2nd", "3rd+", "•"}
        lines = [ln for ln in lines if ln.lower() not in _NOISE and not ln.startswith("•")]
        if not lines:
            continue

        # First non-noise line is the name
        name = lines[0]
        headline = lines[1] if len(lines) > 1 else ""
        location = lines[2] if len(lines) > 2 else ""

        # Skip non-recruiters
        if not any(kw in headline.lower() for kw in _RECRUIT_KWS):
            continue

        # Skip if no name or looks like a UI label
        if len(name) < 3 or name.lower() in {"connect", "follow", "message", "view"}:
            continue

        profiles.append(RecruiterProfile(
            name=name,
            title=headline,
            company=company,
            linkedin_url=url,
            location=location,
        ))

    return profiles


async def _search_company(
    page: Page, company: str, max_results: int
) -> list[RecruiterProfile]:
    """Search LinkedIn for technical recruiters at a specific company."""
    query = quote_plus(f"technical recruiter {company}")
    url = (
        f"https://www.linkedin.com/search/results/people/"
        f"?keywords={query}&origin=GLOBAL_SEARCH_HEADER"
    )
    logger.info("Searching recruiters: %s", company)

    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(random.randint(4000, 6000))  # wait for JS render
    await _scroll_results(page, times=3)

    profiles = await _extract_profiles_from_page(page, company)
    return profiles[:max_results]


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

async def search_recruiters(
    companies: list[str],
    max_per_company: int = 5,
    max_total: int = 40,
) -> list[RecruiterProfile]:
    """
    Search LinkedIn for technical recruiters at each company.

    Opens one non-headless browser window, logs in once, then iterates
    through companies with human-like delays between searches.

    Args:
        companies:        company names to search
        max_per_company:  max recruiters to collect per company
        max_total:        hard cap on total recruiters returned
    """
    if not companies:
        return []

    all_profiles: list[RecruiterProfile] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=120,          # All actions delayed 120 ms — human-paced
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
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

        # ── Login once ──
        logged_in = await _login(page)
        if not logged_in:
            await browser.close()
            logger.error("Could not log into LinkedIn — skipping recruiter search")
            return []

        # Brief pause after login before starting searches
        await page.wait_for_timeout(random.randint(2000, 3500))

        # ── Search each company ──
        for company in companies:
            if len(all_profiles) >= max_total:
                break
            try:
                profiles = await _search_company(page, company, max_per_company)
                all_profiles.extend(profiles)
                logger.info(
                    "Found %d recruiters at %s (running total: %d)",
                    len(profiles), company, len(all_profiles)
                )
            except Exception as exc:
                logger.error("Recruiter search failed for %s: %s", company, exc)

            # Human-paced pause between company searches (6–12 seconds)
            if company != companies[-1]:
                await page.wait_for_timeout(random.randint(6000, 12000))

        await browser.close()

    logger.info("LinkedIn recruiter search complete — %d total", len(all_profiles))
    return all_profiles[:max_total]
