"""
Career Page Agent — fetches jobs from company career pages and scores
each listing against the user's profile using Claude Haiku.

Routing logic per company:
  api_type=greenhouse  → Greenhouse public API (no browser, instant)
  api_type=apple       → Apple Jobs REST API   (no browser, instant)
  api_type=netflix     → Netflix Jobs API      (no browser, instant)
  (none)               → Playwright browser scraping (fallback)

API-based companies run concurrently since they're just HTTP calls.
Browser companies run sequentially (one visible window at a time).
"""

import asyncio
import json
import logging

from agents.base_agent import BaseAgent
from models import JobListing
from tools.ats_api_fetcher import (
    fetch_apple_jobs,
    fetch_greenhouse_jobs,
    fetch_meta_jobs,
    fetch_netflix_jobs,
)
from tools.career_page_scraper import scrape_career_page

logger = logging.getLogger(__name__)

SCORING_SYSTEM = (
    "Score job relevance (0.0-1.0) for a 10yr Java/Python backend engineer "
    "(Goldman Sachs, Amazon) targeting Senior SWE at Big Tech. "
    "High score: distributed systems, platform, Java, Kubernetes, AI infra. "
    "Low score: frontend-only, mobile, defense, fintech-only. "
    "Return ONLY a JSON array: "
    '[{"title":str,"company":str,"relevance_score":float,"relevance_reason":str}]'
)


class CareerPageAgent(BaseAgent):
    """Fetches and scores job listings from company career pages."""

    async def run(
        self,
        companies: list[dict],
        role_keywords: list[str],
        max_companies: int = 10,
    ) -> list[JobListing]:
        target = companies[:max_companies]

        # Split into API-based vs browser-based
        api_companies = [c for c in target if c.get("api_type")]
        browser_companies = [c for c in target if not c.get("api_type")]

        self.logger.info(
            "Fetching %d via API, %d via browser",
            len(api_companies), len(browser_companies),
        )

        all_jobs: list[JobListing] = []

        # ── API companies: run all concurrently (no browser overhead) ──
        if api_companies:
            api_tasks = [
                self._fetch_via_api(c, role_keywords) for c in api_companies
            ]
            api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
            for company, result in zip(api_companies, api_results):
                if isinstance(result, Exception):
                    self.logger.error("API fetch failed for %s: %s", company["name"], result)
                else:
                    all_jobs.extend(result)

        # ── Browser companies: run sequentially (one visible window at a time) ──
        for company in browser_companies:
            self.logger.info("→ Browser: %s", company["name"])
            try:
                jobs = await scrape_career_page(
                    company_name=company["name"],
                    career_url=company["careers_url"],
                    role_keywords=role_keywords,
                    known_h1b_sponsor=company.get("known_h1b_sponsor", False),
                )
                all_jobs.extend(jobs)
            except Exception as exc:
                self.logger.error("Browser scrape failed for %s: %s", company["name"], exc)

        self.logger.info("Total jobs before scoring: %d", len(all_jobs))

        if not all_jobs:
            return []

        return await self._score_jobs(all_jobs)

    async def _fetch_via_api(
        self, company: dict, keywords: list[str]
    ) -> list[JobListing]:
        """Route to the correct API fetcher based on api_type."""
        name = company["name"]
        h1b = company.get("known_h1b_sponsor", True)
        api_type = company.get("api_type")

        self.logger.info("→ API (%s): %s", api_type, name)

        if api_type == "greenhouse":
            return await fetch_greenhouse_jobs(
                company_name=name,
                slug=company["api_slug"],
                keywords=keywords,
                known_h1b_sponsor=h1b,
            )
        elif api_type == "apple":
            return await fetch_apple_jobs(keywords=keywords, known_h1b_sponsor=h1b)
        elif api_type == "netflix":
            return await fetch_netflix_jobs(keywords=keywords, known_h1b_sponsor=h1b)
        elif api_type == "meta_graphql":
            return await fetch_meta_jobs(keywords=keywords, known_h1b_sponsor=h1b)
        else:
            self.logger.warning("Unknown api_type '%s' for %s — skipping", api_type, name)
            return []

    async def _score_jobs(self, jobs: list[JobListing]) -> list[JobListing]:
        """Use Claude Haiku to score each job's relevance. Batches of 20."""
        BATCH = 20
        scored: list[JobListing] = []

        for i in range(0, len(jobs), BATCH):
            batch = jobs[i : i + BATCH]
            slim = [{"title": j.title, "company": j.company} for j in batch]
            try:
                response = self.client.messages.create(
                    model=self.MODEL_HAIKU,
                    max_tokens=2048,
                    system=SCORING_SYSTEM,
                    messages=[{"role": "user", "content": json.dumps(slim)}],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1].lstrip("json").strip()
                scores = json.loads(raw)

                for job, score_data in zip(batch, scores):
                    job.relevance_score = float(score_data.get("relevance_score", 0.0))
                    job.relevance_reason = score_data.get("relevance_reason", "")
                    scored.append(job)

            except Exception as exc:
                self.logger.error("Scoring batch failed: %s", exc)
                scored.extend(batch)  # keep unscored rather than drop

        return scored
