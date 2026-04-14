"""
Orchestrator Agent — the top-level Claude agent that plans and coordinates
all sub-agents using Claude's tool_use API.

Flow:
  1. Claude plans the search strategy
  2. Calls search_job_boards → JobSearchAgent
  3. Calls scrape_career_pages → CareerPageAgent
  4. Calls find_recruiters → RecruiterAgent (uses companies where jobs were found)
  5. Calls draft_outreach → OutreachAgent
  6. Aggregates into SearchResult and generates report
"""

import asyncio
import json
import logging
from typing import Any

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.base_agent import BaseAgent
from agents.career_page_agent import CareerPageAgent
from agents.job_search_agent import JobSearchAgent
from agents.outreach_agent import OutreachAgent
from agents.recruiter_agent import RecruiterAgent
from models import JobListing, RecruiterProfile, SearchResult
from utils.report_generator import generate_report

logger = logging.getLogger(__name__)
console = Console()

# ──────────────────────────────────────────────
# Tool schemas for the orchestrator
# ──────────────────────────────────────────────

ORCHESTRATOR_TOOLS = [
    {
        "name": "search_job_boards",
        "description": (
            "Search LinkedIn, Indeed, and Wellfound for Senior Software Engineer openings. "
            "Automatically filters for H-1B sponsorship relevance and scores by fit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Job title to search for",
                    "default": "Senior Software Engineer",
                },
                "location": {
                    "type": "string",
                    "description": "Location filter (e.g. 'United States', 'San Francisco')",
                    "default": "United States",
                },
                "max_per_board": {
                    "type": "integer",
                    "description": "Maximum results per job board",
                    "default": 30,
                },
            },
            "required": [],
        },
    },
    {
        "name": "scrape_career_pages",
        "description": (
            "Scrape the direct career pages of target Big Tech / FAANG companies. "
            "More reliable than job boards for finding all open roles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "integer",
                    "description": "1 = FAANG/top tier only, 2 = include all target companies",
                    "default": 2,
                },
                "max_companies": {
                    "type": "integer",
                    "description": "Maximum number of companies to scrape",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_recruiters",
        "description": (
            "Search LinkedIn for technical recruiters at the companies that have open roles. "
            "Returns recruiter profiles for direct outreach."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of company names to search recruiters at",
                },
                "max_total": {
                    "type": "integer",
                    "description": "Maximum total recruiters to return",
                    "default": 30,
                },
            },
            "required": ["companies"],
        },
    },
    {
        "name": "draft_outreach_messages",
        "description": (
            "Generate formal, personalized LinkedIn connection messages for recruiters. "
            "Each message is under 300 characters and tailored to the company's tech stack."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_messages": {
                    "type": "integer",
                    "description": "Maximum messages to generate",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
]

ORCHESTRATOR_SYSTEM = """\
You are a job search orchestrator for Badrinath Radhakrishnan, a Senior Software Engineer
with 10+ years experience (Goldman Sachs, Amazon). He requires H-1B sponsorship.

Your job is to coordinate specialized agents to find the best job opportunities and
relevant recruiters at Big Tech / FAANG companies.

Strategy:
1. First call search_job_boards to find recent openings across job boards
2. Then call scrape_career_pages to check direct company career pages (most comprehensive)
3. Then call find_recruiters using the companies where jobs were found
4. Finally call draft_outreach_messages to prepare personalized recruiter messages
5. When all data is gathered, summarize what was found

Be methodical. Call tools in the order above. Do not skip steps.
After all tool calls complete, provide a brief summary of: total jobs found, top 3 companies
with openings, number of recruiters found, and next recommended actions for the candidate.
"""


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class Orchestrator(BaseAgent):
    """Top-level agent that plans and coordinates all sub-agents."""

    def __init__(self, companies_config: list[dict]):
        super().__init__()
        self.companies_config = companies_config
        self.result = SearchResult()

        # Sub-agents
        self._job_agent = JobSearchAgent()
        self._career_agent = CareerPageAgent()
        self._recruiter_agent = RecruiterAgent()
        self._outreach_agent = OutreachAgent()

    async def run(self) -> SearchResult:
        """Run the full multi-agent job search pipeline."""
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Please run the full job search pipeline for me. "
                    "I am looking for Senior Software Engineer roles at Big Tech / FAANG companies "
                    "in the United States (West Coast preferred). I require H-1B sponsorship."
                ),
            }
        ]

        console.rule("[bold blue]Job Search Agent Starting[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Orchestrator thinking...", total=None)

            while True:
                progress.update(task, description="Claude is planning next step...")
                response = self.client.messages.create(
                    model=self.MODEL_SMART,
                    max_tokens=4096,
                    system=ORCHESTRATOR_SYSTEM,
                    tools=ORCHESTRATOR_TOOLS,
                    messages=messages,
                )

                # Collect tool calls and text blocks
                tool_results = []
                for block in response.content:
                    if block.type == "text":
                        console.print(f"\n[dim]Orchestrator:[/dim] {block.text}")
                    elif block.type == "tool_use":
                        progress.update(task, description=f"Running: {block.name}...")
                        console.print(f"\n[bold cyan]→ Tool call:[/bold cyan] {block.name}")
                        result = await self._dispatch(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                # Append assistant turn and tool results
                messages.append({"role": "assistant", "content": response.content})
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                if response.stop_reason == "end_turn":
                    break

        console.rule("[bold green]Pipeline Complete[/bold green]")
        return self.result

    async def _dispatch(self, tool_name: str, inputs: dict) -> dict[str, Any]:
        """Route orchestrator tool calls to the appropriate sub-agent."""
        try:
            if tool_name == "search_job_boards":
                jobs = await self._job_agent.run(
                    title=inputs.get("title", "Senior Software Engineer"),
                    location=inputs.get("location", "United States"),
                    max_per_board=inputs.get("max_per_board", 30),
                )
                self.result.jobs.extend(jobs)
                return {
                    "status": "success",
                    "jobs_found": len(jobs),
                    "sample": [{"title": j.title, "company": j.company, "score": j.relevance_score} for j in jobs[:5]],
                }

            elif tool_name == "scrape_career_pages":
                tier = inputs.get("tier", 2)
                companies = [
                    c for c in self.companies_config
                    if tier == 2 or c.get("tier", 2) <= tier
                ]
                role_keywords = ["senior software engineer", "backend engineer", "platform engineer", "java", "distributed"]
                jobs = await self._career_agent.run(
                    companies=companies,
                    role_keywords=role_keywords,
                    max_companies=inputs.get("max_companies", 10),
                )
                self.result.jobs.extend(jobs)
                return {
                    "status": "success",
                    "jobs_found": len(jobs),
                    "companies_scraped": len(companies[:inputs.get("max_companies", 10)]),
                    "sample": [{"title": j.title, "company": j.company, "score": j.relevance_score} for j in jobs[:5]],
                }

            elif tool_name == "find_recruiters":
                # Build company list from all jobs found so far if not provided
                companies = inputs.get("companies") or []
                if not companies:
                    seen: set[str] = set()
                    for j in self.result.top_jobs(30):
                        if j.company not in seen:
                            companies.append(j.company)
                            seen.add(j.company)
                recruiters = await self._recruiter_agent.run(
                    companies=companies,
                    top_jobs=self.result.top_jobs(20),
                    max_total=inputs.get("max_total", 30),
                )
                self.result.recruiters.extend(recruiters)
                return {
                    "status": "success",
                    "recruiters_found": len(recruiters),
                    "sample": [{"name": r.name, "company": r.company, "title": r.title} for r in recruiters[:5]],
                }

            elif tool_name == "draft_outreach_messages":
                messages_out = self._outreach_agent.run(
                    recruiters=self.result.recruiters,
                    top_jobs=self.result.top_jobs(20),
                    max_messages=inputs.get("max_messages", 20),
                )
                self.result.outreach_messages.extend(messages_out)
                return {
                    "status": "success",
                    "messages_drafted": len(messages_out),
                    "sample": messages_out[0].message if messages_out else "",
                }

            else:
                return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        except Exception as exc:
            error_msg = f"Tool {tool_name} failed: {exc}"
            logger.error(error_msg)
            self.result.errors.append(error_msg)
            return {"status": "error", "message": error_msg}
