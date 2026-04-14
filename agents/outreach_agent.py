"""
Outreach Agent — generates formal, personalized LinkedIn connection messages
for each recruiter. Messages are kept under 300 characters (LinkedIn's limit).
"""

import logging

from agents.base_agent import BaseAgent
from models import JobListing, OutreachMessage, RecruiterProfile

logger = logging.getLogger(__name__)

OUTREACH_SYSTEM = """\
You are a career coach writing formal LinkedIn connection request messages for a senior software engineer.

Candidate: Badrinath Radhakrishnan
Background: 10+ years Java/Python backend engineer. Currently Senior SWE at Goldman Sachs where he
productionized an MCP gateway on Kubernetes enabling secure LLM tool orchestration — early enterprise
GenAI platform engineering. Previously at Amazon (distributed payment services, Spring Boot/AWS).
Visa: Requires H-1B sponsorship.

Rules for every message:
1. Formal, professional tone — no casual phrases or emojis
2. Under 300 characters total (LinkedIn connection request limit)
3. Mention ONE specific technical thing about the candidate relevant to the company's tech stack
4. Reference the recruiter's company by name
5. End with a clear, brief ask (e.g., "Would welcome the opportunity to connect.")
6. Do NOT mention visa/H-1B in the message — this is for initial connection only

Return ONLY the message text — no subject line, no labels, no explanation.
"""


class OutreachAgent(BaseAgent):
    """Generates personalized outreach messages for recruiters."""

    def run(
        self,
        recruiters: list[RecruiterProfile],
        top_jobs: list[JobListing],
        max_messages: int = 20,
    ) -> list[OutreachMessage]:
        """
        Generate outreach messages for a list of recruiters.

        Args:
            recruiters: list of RecruiterProfile objects
            top_jobs: used to match company → job title for context
            max_messages: cap on total messages generated
        """
        if not recruiters:
            return []

        # Build company → top job title lookup
        company_job: dict[str, str] = {}
        for job in sorted(top_jobs, key=lambda j: j.relevance_score, reverse=True):
            key = job.company.lower()
            if key not in company_job:
                company_job[key] = job.title

        messages: list[OutreachMessage] = []

        for recruiter in recruiters[:max_messages]:
            job_title = company_job.get(
                recruiter.company.lower(), "Senior Software Engineer"
            )
            message = self._generate_message(recruiter, job_title)
            if message:
                messages.append(message)

        self.logger.info("Generated %d outreach messages", len(messages))
        return messages

    def _generate_message(
        self, recruiter: RecruiterProfile, job_title: str
    ) -> OutreachMessage | None:
        prompt = (
            f"Recruiter name: {recruiter.name}\n"
            f"Recruiter title: {recruiter.title}\n"
            f"Company: {recruiter.company}\n"
            f"Company focus area: {self._company_focus(recruiter.company)}\n"
            f"Role I am targeting: {job_title}\n"
        )
        try:
            response = self.client.messages.create(
                model=self.MODEL_FAST,
                max_tokens=200,
                system=OUTREACH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Truncate hard at 300 chars to guarantee LinkedIn compliance
            if len(text) > 300:
                text = text[:297] + "..."

            return OutreachMessage(
                recruiter=recruiter,
                job_title=job_title,
                message=text,
            )
        except Exception as exc:
            self.logger.error(
                "Failed to generate message for %s @ %s: %s",
                recruiter.name, recruiter.company, exc
            )
            return None

    @staticmethod
    def _company_focus(company: str) -> str:
        """Return a brief tech focus hint to guide message personalization."""
        focus_map = {
            "google": "distributed systems, AI/ML infrastructure, Java/Go platform services",
            "meta": "large-scale backend infrastructure, AI platform, Java/C++ systems",
            "apple": "platform services, backend APIs, Java/Swift systems",
            "amazon": "AWS distributed services, Java microservices, high-availability systems",
            "microsoft": "Azure platform, developer tooling, Java/.NET distributed services",
            "netflix": "streaming microservices, Java/Spring Boot, chaos engineering",
            "nvidia": "AI infrastructure software, GPU compute platform, Java/C++ services",
            "uber": "real-time distributed systems, Java/Go backend, geospatial data",
            "airbnb": "Java backend platform, distributed services, data infrastructure",
            "doordash": "Java/Kotlin microservices, logistics platform, real-time systems",
            "databricks": "data platform engineering, Apache Spark, Java/Scala distributed compute",
            "snowflake": "cloud-native data platform, Java backend, distributed storage systems",
            "salesforce": "CRM platform, Java/Spring enterprise services, cloud infrastructure",
            "adobe": "creative cloud platform, Java services, AI/ML product integration",
            "linkedin": "Java/Scala distributed systems, social graph backend, data platform",
            "lyft": "Java/Python backend, real-time dispatch systems, distributed platform",
            "anthropic": "AI safety infrastructure, LLM platform engineering, Python/Java services",
            "openai": "AI platform engineering, developer APIs, distributed ML infrastructure",
            "palantir": "Java/Python data platform, distributed analytics, enterprise backend",
            "stripe": "payments infrastructure, Java/Ruby backend, distributed financial systems",
        }
        return focus_map.get(company.lower(), "distributed systems, backend platform, Java/Python services")
