"""
Job Search Agent — searches LinkedIn public jobs, Indeed, and Wellfound.
Filters results for H-1B relevance and scores them using Claude.
"""

import asyncio
import json
import logging

from agents.base_agent import BaseAgent
from models import JobListing, JobSource
from tools.job_board_scraper import (
    search_linkedin_jobs,
    search_indeed_jobs,
    search_wellfound_jobs,
)

logger = logging.getLogger(__name__)

# Known H-1B sponsors from our companies.json — used to auto-flag listings
KNOWN_H1B_SPONSORS: set[str] = {
    "google", "meta", "apple", "amazon", "microsoft", "netflix", "nvidia",
    "uber", "airbnb", "doordash", "databricks", "snowflake", "salesforce",
    "adobe", "linkedin", "lyft", "anthropic", "openai", "palantir", "stripe",
}

FILTER_SYSTEM = (
    "Filter job listings for a Senior SWE requiring H-1B sponsorship. "
    "KEEP: known big-tech sponsors, roles mentioning visa/H-1B, Java/backend/platform/AI roles. "
    "REMOVE: 'no sponsorship'/'US citizen only'/'clearance required', pure frontend, mobile, defense. "
    "Return ONLY JSON array: "
    "[{\"index\":int,\"keep\":bool,\"h1b_likely\":bool,\"relevance_score\":float,\"relevance_reason\":str}]"
)


class JobSearchAgent(BaseAgent):
    """Searches job boards and filters/scores results."""

    async def run(
        self,
        title: str = "Senior Software Engineer",
        location: str = "United States",
        max_per_board: int = 30,
    ) -> list[JobListing]:
        """Search all job boards concurrently and return filtered, scored results."""
        self.logger.info("Searching job boards for: %s in %s", title, location)

        linkedin_task = search_linkedin_jobs(title, location, max_per_board)
        indeed_task = search_indeed_jobs(title, location, max_per_board)
        wellfound_task = search_wellfound_jobs(title, location, max_per_board // 2)

        results = await asyncio.gather(
            linkedin_task, indeed_task, wellfound_task, return_exceptions=True
        )

        all_jobs: list[JobListing] = []
        labels = ["LinkedIn", "Indeed", "Wellfound"]
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                self.logger.error("%s search failed: %s", label, result)
            else:
                self.logger.info("%s: %d jobs", label, len(result))
                all_jobs.extend(result)

        self.logger.info("Total from job boards: %d", len(all_jobs))

        if not all_jobs:
            return []

        # Enrich known H-1B sponsors before sending to Claude
        for job in all_jobs:
            company_lower = job.company.lower()
            if any(sponsor in company_lower for sponsor in KNOWN_H1B_SPONSORS):
                job.known_h1b_sponsor = True

        return await self._filter_and_score(all_jobs)

    async def _filter_and_score(self, jobs: list[JobListing]) -> list[JobListing]:
        """Use Claude to filter out non-sponsoring roles and score relevance."""
        BATCH = 25
        kept: list[JobListing] = []

        for i in range(0, len(jobs), BATCH):
            batch = jobs[i : i + BATCH]
            payload = [
                {
                    "index": idx,
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "description": j.description[:300] if j.description else "",
                    "known_h1b_sponsor": j.known_h1b_sponsor,
                    "h1b_mentioned": j.h1b_mentioned,
                }
                for idx, j in enumerate(batch)
            ]

            try:
                response = self.client.messages.create(
                    model=self.MODEL_HAIKU,
                    max_tokens=1500,
                    system=FILTER_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": json.dumps(payload)
                    }],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                decisions = json.loads(raw)

                for decision in decisions:
                    idx = decision.get("index", 0)
                    if idx < len(batch) and decision.get("keep", False):
                        job = batch[idx]
                        job.h1b_mentioned = decision.get("h1b_likely", job.h1b_mentioned)
                        job.relevance_score = float(decision.get("relevance_score", 0.0))
                        job.relevance_reason = decision.get("relevance_reason", "")
                        kept.append(job)

            except Exception as exc:
                self.logger.error("Filter batch failed: %s", exc)
                # On error, keep known sponsors
                kept.extend(j for j in batch if j.known_h1b_sponsor)

        self.logger.info("After filtering: %d jobs kept", len(kept))
        return kept
