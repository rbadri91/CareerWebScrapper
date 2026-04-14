"""
Recruiter Agent — finds LinkedIn recruiters at target companies using
Playwright-based browser automation (non-headless, human-like behaviour).

Prioritises companies where high-scoring jobs were found so outreach
messages are as targeted as possible.
"""

import logging

from agents.base_agent import BaseAgent
from models import JobListing, RecruiterProfile
from tools.linkedin_tools import search_recruiters

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

        Args:
            companies:        company names to search (from orchestrator)
            top_jobs:         used to re-rank companies by job score, so
                              the most promising companies are searched first
            max_per_company:  recruiter cap per company
            max_total:        hard cap on total results
        """
        if not companies:
            return []

        ordered = self._rank_companies(companies, top_jobs or [])
        self.logger.info(
            "Recruiter search order: %s", ", ".join(ordered[:8])
        )

        return await search_recruiters(
            companies=ordered,
            max_per_company=max_per_company,
            max_total=max_total,
        )

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
