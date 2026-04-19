"""
Entry point for the multi-agent job search system.

Usage:
    python main.py                      # Full pipeline (all agents)
    python main.py --jobs-only          # Job boards + career pages, no recruiter search
    python main.py --career-pages-only  # Only scrape company career pages
    python main.py --dry-run            # Print config and exit without running
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agents.orchestrator import Orchestrator
from utils.report_generator import generate_report
from utils.email_sender import send_report

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/run.log"),
    ],
)
# Quiet noisy third-party loggers
logging.getLogger("playwright").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

console = Console()


# ──────────────────────────────────────────────
# Config loaders
# ──────────────────────────────────────────────

def load_config() -> tuple[dict, list[dict]]:
    """Load user_profile.json and companies.json from config/."""
    config_dir = Path(__file__).parent / "config"
    with open(config_dir / "user_profile.json") as f:
        profile = json.load(f)
    with open(config_dir / "companies.json") as f:
        companies = json.load(f)["companies"]
    return profile, companies


def check_env() -> list[str]:
    """Return list of missing required environment variables."""
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

def print_summary(result, report_path: str) -> None:
    console.print()
    console.rule("[bold green]Results Summary[/bold green]")

    # Jobs table
    top_jobs = result.top_jobs(10)
    if top_jobs:
        table = Table(title="Top 10 Matching Jobs", show_header=True, header_style="bold magenta")
        table.add_column("#", width=3)
        table.add_column("Title", min_width=25)
        table.add_column("Company", min_width=15)
        table.add_column("Location", min_width=15)
        table.add_column("H-1B", width=5)
        table.add_column("Score", width=6)
        for i, job in enumerate(top_jobs, 1):
            h1b = "✓" if (job.h1b_mentioned or job.known_h1b_sponsor) else "?"
            table.add_row(
                str(i), job.title, job.company,
                job.location or "N/A", h1b,
                f"{job.relevance_score:.0%}"
            )
        console.print(table)

    # Stats
    console.print(f"\n[bold]Total jobs found:[/bold] {len(result.jobs)}")
    console.print(f"[bold]Recruiters found:[/bold] {len(result.recruiters)}")
    console.print(f"[bold]Outreach messages:[/bold] {len(result.outreach_messages)}")
    if result.errors:
        console.print(f"[bold red]Errors:[/bold red] {len(result.errors)}")
        for err in result.errors:
            console.print(f"  [red]•[/red] {err}")

    console.print(f"\n[bold green]Full report saved to:[/bold green] {report_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # Ensure outputs directory exists
    Path("outputs").mkdir(exist_ok=True)

    # Validate environment
    missing = check_env()
    if missing:
        console.print(f"[bold red]Missing required environment variables:[/bold red] {', '.join(missing)}")
        console.print("Copy .env.example to .env and fill in the values.")
        sys.exit(1)

    # Load config
    profile, companies = load_config()

    if args.dry_run:
        console.print("[bold]Dry run — config loaded successfully:[/bold]")
        console.print(f"  User: {profile['name']}")
        console.print(f"  Target role: {profile['target_role']}")
        console.print(f"  Companies configured: {len(companies)}")
        console.print(f"  Visa requirement: {profile['visa']['visa_type']}")
        return

    console.print(f"[bold blue]Starting job search for:[/bold blue] {profile['name']}")
    console.print(f"[bold blue]Target:[/bold blue] {profile['target_role']} @ Big Tech | H-1B required")
    console.print()

    if args.career_pages_only:
        # Run only the career page agent directly — no LinkedIn, no job boards
        from agents.career_page_agent import CareerPageAgent
        from models import SearchResult
        console.print("[bold cyan]Mode: Career pages only[/bold cyan]")
        agent = CareerPageAgent()
        role_keywords = ["senior software engineer", "backend engineer", "platform engineer", "java", "distributed"]
        jobs = await agent.run(companies=companies, role_keywords=role_keywords, max_companies=10)
        result = SearchResult(jobs=jobs)
    else:
        # Run full orchestrator pipeline
        orchestrator = Orchestrator(companies_config=companies)
        result = await orchestrator.run()

    # Generate report
    report_path = generate_report(result)

    # Email report (requires GMAIL_APP_PASSWORD in .env)
    send_report(report_path)

    # Print summary to console
    print_summary(result, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent job search system")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    parser.add_argument("--jobs-only", action="store_true", help="Skip recruiter search")
    parser.add_argument("--career-pages-only", dest="career_pages_only", action="store_true", help="Only scrape company career pages")
    args = parser.parse_args()

    asyncio.run(main(args))
