"""Base agent class — provides a shared Anthropic client and logging."""

import logging
import anthropic


class BaseAgent:
    """All agents inherit from this. Provides a shared client and logger."""

    MODEL_HAIKU = "claude-haiku-4-5-20251001"  # Cheapest — extraction, scoring, filtering
    MODEL_FAST = "claude-sonnet-4-6"           # Mid-tier — outreach drafting
    MODEL_SMART = "claude-opus-4-6"            # Smartest — orchestration only

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.logger = logging.getLogger(self.__class__.__name__)
