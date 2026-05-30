"""
API cost tracker for Feature Intelligence pipeline.

Tracks input/output tokens per agent, computes estimated dollar costs,
and surfaces cost breakdowns in the UI and eval results.

Why this matters: agentic systems with 4+ agents and extended thinking
can easily burn $5-10 per run. Making cost visible is an operational
necessity, not a nice-to-have.

Pricing (as of 2025):
    claude-sonnet-4-6: $3/M input, $15/M output
    Extended thinking output tokens: $15/M (same as regular output)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Add other models as needed
}

DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


@dataclass
class APICallRecord:
    """Single API call record."""
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = ((self.output_tokens + self.thinking_tokens) / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    """
    Accumulates API call records across a pipeline run.

    Usage:
        tracker = CostTracker()
        tracker.record("ThemeAgent", response)  # pass anthropic response object
        ...
        summary = tracker.summary()
        # {'total_cost_usd': 0.42, 'by_agent': {...}, ...}
    """

    records: List[APICallRecord] = field(default_factory=list)
    _run_start: float = field(default_factory=time.time)

    def record(
        self,
        agent: str,
        response,  # anthropic.types.Message
        model: str = "claude-sonnet-4-6",
        duration_ms: float = 0.0,
    ) -> APICallRecord:
        """Record an API call from an Anthropic response object."""
        usage = getattr(response, "usage", None)

        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        # Extended thinking tokens are tracked in cache_creation for streaming
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage else 0

        record = APICallRecord(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            duration_ms=duration_ms,
        )

        self.records.append(record)
        return record

    def record_manual(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        model: str = "claude-sonnet-4-6",
    ) -> APICallRecord:
        """Record manually (when response object isn't available)."""
        record = APICallRecord(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
        )
        self.records.append(record)
        return record

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens + r.thinking_tokens for r in self.records)

    @property
    def total_api_calls(self) -> int:
        return len(self.records)

    def by_agent(self) -> Dict[str, Dict]:
        """Aggregate cost and token stats per agent."""
        agents: Dict[str, List[APICallRecord]] = {}
        for r in self.records:
            agents.setdefault(r.agent, []).append(r)

        return {
            agent: {
                "api_calls": len(records),
                "input_tokens": sum(r.input_tokens for r in records),
                "output_tokens": sum(r.output_tokens + r.thinking_tokens for r in records),
                "total_tokens": sum(r.total_tokens for r in records),
                "cost_usd": round(sum(r.cost_usd for r in records), 6),
                "avg_duration_ms": round(
                    sum(r.duration_ms for r in records) / len(records), 0
                ) if records else 0,
            }
            for agent, records in agents.items()
        }

    def summary(self) -> Dict:
        """Full cost summary for the pipeline run."""
        elapsed = time.time() - self._run_start

        return {
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_api_calls": self.total_api_calls,
            "elapsed_seconds": round(elapsed, 1),
            "cost_per_minute": round(
                (self.total_cost_usd / elapsed * 60) if elapsed > 0 else 0, 4
            ),
            "by_agent": self.by_agent(),
        }

