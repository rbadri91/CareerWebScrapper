"""Generates a structured Markdown report from search results."""

from datetime import datetime
from pathlib import Path
from models import SearchResult, JobListing, RecruiterProfile, OutreachMessage


def _job_table(jobs: list[JobListing]) -> str:
    if not jobs:
        return "_No jobs found for this source._\n"
    rows = ["| # | Title | Company | Location | H-1B | Score | Link |",
            "|---|-------|---------|----------|------|-------|------|"]
    for i, j in enumerate(jobs, 1):
        h1b = "✓" if (j.h1b_mentioned or j.known_h1b_sponsor) else "?"
        link = f"[Apply]({j.url})" if j.url else "N/A"
        rows.append(
            f"| {i} | {j.title} | {j.company} | {j.location or 'N/A'} | "
            f"{h1b} | {j.relevance_score:.0%} | {link} |"
        )
    return "\n".join(rows) + "\n"


def _recruiter_table(recruiters: list[RecruiterProfile]) -> str:
    if not recruiters:
        return "_No recruiters found._\n"
    rows = ["| Name | Title | Company | LinkedIn | Degree |",
            "|------|-------|---------|----------|--------|"]
    for r in recruiters:
        link = f"[Profile]({r.linkedin_url})" if r.linkedin_url else "N/A"
        rows.append(
            f"| {r.name} | {r.title or 'N/A'} | {r.company} | {link} | {r.connection_degree or 'N/A'} |"
        )
    return "\n".join(rows) + "\n"


def generate_report(result: SearchResult, output_dir: str = "outputs") -> str:
    """Write a Markdown report to outputs/ and return the file path."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = Path(output_dir) / f"job_search_{timestamp}.md"

    # Group jobs by source
    by_source: dict[str, list[JobListing]] = {}
    for job in result.jobs:
        by_source.setdefault(job.source.value, []).append(job)

    lines = [
        f"# Job Search Report — {datetime.now().strftime('%B %d, %Y')}",
        "",
        f"> Generated at {datetime.now().strftime('%H:%M')} | "
        f"{len(result.jobs)} jobs found | "
        f"{len(result.recruiters)} recruiters found",
        "",
        "---",
        "",
        "## Top 15 Matching Jobs (All Sources)",
        "",
        _job_table(result.top_jobs(15)),
        "",
        "---",
        "",
        "## Jobs by Source",
        "",
    ]

    source_labels = {
        "linkedin": "LinkedIn Jobs",
        "indeed": "Indeed",
        "wellfound": "Wellfound (AngelList)",
        "career_page": "Company Career Pages",
    }
    for source, jobs in by_source.items():
        label = source_labels.get(source, source.title())
        lines += [f"### {label} ({len(jobs)} jobs)", "", _job_table(jobs), ""]

    lines += [
        "---",
        "",
        "## Recruiters Found",
        "",
        _recruiter_table(result.recruiters),
        "",
        "---",
        "",
        "## Drafted Outreach Messages",
        "",
    ]

    if result.outreach_messages:
        for msg in result.outreach_messages:
            lines += [
                f"### To: {msg.recruiter.name} @ {msg.recruiter.company}",
                f"**Role:** {msg.job_title}  ",
                f"**LinkedIn:** {msg.recruiter.linkedin_url}  ",
                f"**Character count:** {msg.character_count}/300",
                "",
                f"> {msg.message}",
                "",
            ]
    else:
        lines.append("_No outreach messages generated._\n")

    if result.errors:
        lines += ["---", "", "## Errors / Warnings", ""]
        for err in result.errors:
            lines.append(f"- {err}")

    report_text = "\n".join(lines)
    path.write_text(report_text)
    return str(path)
