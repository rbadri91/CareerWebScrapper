"""Shared Pydantic data models used across all agents and tools."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class JobSource(str, Enum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    WELLFOUND = "wellfound"
    CAREER_PAGE = "career_page"


class JobListing(BaseModel):
    title: str
    company: str
    location: str = ""
    url: str = ""
    description: str = ""
    posted_date: str = ""
    source: JobSource = JobSource.CAREER_PAGE
    h1b_mentioned: bool = False
    known_h1b_sponsor: bool = False
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_reason: str = ""
    scraped_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class RecruiterProfile(BaseModel):
    name: str
    title: str = ""
    company: str = ""
    linkedin_url: str = ""
    email: str = ""
    location: str = ""
    connection_degree: str = ""
    profile_summary: str = ""


class OutreachMessage(BaseModel):
    recruiter: RecruiterProfile
    job_title: str
    subject: str = ""
    message: str
    character_count: int = 0

    def model_post_init(self, __context):
        self.character_count = len(self.message)


class SearchResult(BaseModel):
    jobs: list[JobListing] = Field(default_factory=list)
    recruiters: list[RecruiterProfile] = Field(default_factory=list)
    outreach_messages: list[OutreachMessage] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def top_jobs(self, n: int = 20) -> list[JobListing]:
        return sorted(self.jobs, key=lambda j: j.relevance_score, reverse=True)[:n]
