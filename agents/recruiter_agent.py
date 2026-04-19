"""
Recruiter Agent — finds LinkedIn recruiters at target companies using
Playwright-based browser automation (non-headless, human-like behaviour).

Prioritises companies where high-scoring jobs were found so outreach
messages are as targeted as possible.
"""

import logging

from agents.base_agent import BaseAgent
from models import JobListing, RecruiterProfile
from tools.linkedin_tools import search_recruiters, scan_existing_connections

logger = logging.getLogger(__name__)


class RecruiterAgent(BaseAgent):
    """Searches LinkedIn for recruiters at target companies."""

    async def run(
        self,
        companies: list[str],
        top_jobs: list[JobListing] | None = None,
        max_per_company: int = 5,
        max_total: int = 40,
    ) -> list[RecruiterProfile]:
        """
        Find recruiters on LinkedIn.

        Runs two searches in parallel:
          1. Public people search for recruiters at target companies
          2. Scan of the user's existing 1st-degree connections for recruiters

        Deduplicates by LinkedIn profile URL. Existing connections are always
        kept when there is a conflict (they carry richer data including email).

        Args:
            companies:        company names to search (from orchestrator)
            top_jobs:         used to re-rank companies by job score
            max_per_company:  recruiter cap per company (public search)
            max_total:        hard cap on total results
        """
        if not companies:
            return []

        ordered = self._rank_companies(companies, top_jobs or [])
        self.logger.info("Recruiter search order: %s", ", ".join(ordered[:8]))

        # Run sequentially — both tasks open a browser and log into the same
        # LinkedIn account. Parallel sessions trigger LinkedIn's concurrent-login
        # detection, which throttles or degrades search results for one session.
        # Connection scan first (faster — scoped to your own network), then public.
        try:
            connection_results = await scan_existing_connections(
                companies=companies,
                max_results=50,
            )
        except Exception as exc:
            self.logger.error("Connection scan failed: %s", exc)
            connection_results = []

        try:
            public_results = await search_recruiters(
                companies=ordered,
                max_per_company=max_per_company,
                max_total=max_total,
            )
        except Exception as exc:
            self.logger.error("Public recruiter search failed: %s", exc)
            public_results = []

        # Merge: existing connections take priority on duplicate URLs
        seen_urls: dict[str, RecruiterProfile] = {}
        for r in public_results:
            key = r.linkedin_url.rstrip("/")
            seen_urls[key] = r
        for r in connection_results:
            key = r.linkedin_url.rstrip("/")
            seen_urls[key] = r  # overwrite public result with richer connection data

        merged = list(seen_urls.values())
        # Sort: existing connections first (they have emails), then by company
        merged.sort(key=lambda r: (not r.is_existing_connection, r.company))

        self.logger.info(
            "Recruiters: %d from public search, %d from connections, %d after dedup",
            len(public_results), len(connection_results), len(merged),
        )
        return merged[:max_total]

    @staticmethod
    def _rank_companies(
        companies: list[str], jobs: list[JobListing]
    ) -> list[str]:
        """
        Sort companies by the max relevance score of their job listings.
        Companies with no jobs found keep their original order at the end.
        """
        scores: dict[str, float] = {}
        for job in jobs:
            key = job.company.lower()
            scores[key] = max(scores.get(key, 0.0), job.relevance_score)

        return sorted(
            companies,
            key=lambda c: scores.get(c.lower(), 0.0),
            reverse=True,
        )
