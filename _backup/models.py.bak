"""
Pydantic v2 models for Product Feature Intelligence.
All data flowing between pipeline stages is typed through these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Company context — the "who are we analysing" input
# ---------------------------------------------------------------------------

class CompanyContext(BaseModel):
    company_name: str                        # e.g. "Duolingo"
    product_description: str                 # e.g. "language learning app"
    subreddit: str = ""                      # e.g. "duolingo" — auto-detected if blank
    app_store_id: str = ""                   # e.g. "570060128" — searched if blank
    known_features: str = ""                 # optional paste of current feature list


# ---------------------------------------------------------------------------
# Raw feedback data
# ---------------------------------------------------------------------------

class FeedbackItem(BaseModel):
    source: Literal["reddit", "app_store", "google_play", "hacker_news", "seed"]
    text: str
    upvotes: int = 0
    date: str = ""
    url: str = ""


class FeedbackBundle(BaseModel):
    items: List[FeedbackItem]
    reddit_count: int = 0
    app_store_count: int = 0
    google_play_count: int = 0
    hacker_news_count: int = 0
    seed_count: int = 0
    fetch_timestamp: datetime = Field(default_factory=datetime.utcnow)
    company: Optional[CompanyContext] = None

    @property
    def total(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Agent 1: ThemeAgent outputs
# ---------------------------------------------------------------------------

class Theme(BaseModel):
    theme_name: str
    description: str
    example_quotes: List[str]
    frequency_estimate: Literal["very_high", "high", "medium", "low"]
    user_segment: str  # e.g. "intermediate learners", "beginners", "power users"


class ThemeAnalysis(BaseModel):
    themes: List[Theme]
    total_feedback_items: int
    dominant_frustration: str
    dominant_desire: str


# ---------------------------------------------------------------------------
# Agent 2: AIFeasibilityAgent outputs
# ---------------------------------------------------------------------------

class AIFeasibility(BaseModel):
    theme_name: str
    feasibility_score: float = Field(ge=0.0, le=1.0)
    ai_approach: str       # e.g. "conversational LLM with persona"
    why_ai_uniquely_suited: str
    technical_complexity: Literal["low", "medium", "high"]
    comparable_product: str  # e.g. "Speak app, Elsa"


class FeasibilityAnalysis(BaseModel):
    ratings: List[AIFeasibility]
    top_ai_opportunity: str
    hardest_to_solve: str


# ---------------------------------------------------------------------------
# Agent 3: GapAgent outputs
# ---------------------------------------------------------------------------

class GapFinding(BaseModel):
    theme_name: str
    gap_type: Literal["missing_entirely", "partial_solution", "poor_execution"]
    existing_feature: str  # "" if none
    gap_description: str
    market_evidence: str  # competitor or precedent
    confidence: Literal["high", "medium", "low"] = "medium"
    knowledge_basis: Literal["user_provided_features", "llm_knowledge", "user_feedback_only"] = "llm_knowledge"


class StrengthFinding(BaseModel):
    feature_name: str
    why_users_love_it: str
    quote: str


class GapAnalysis(BaseModel):
    gaps: List[GapFinding]
    strengths: List[StrengthFinding]
    biggest_gap: str
    quick_win: str


# ---------------------------------------------------------------------------
# Agent 4: QuoteAgent outputs
# ---------------------------------------------------------------------------

class PowerQuote(BaseModel):
    quote_text: str
    source: Literal["reddit", "app_store", "seed"]
    theme: str
    why_compelling: str


class QuoteAnalysis(BaseModel):
    quotes: List[PowerQuote]
    most_compelling_quote: str
    most_viral_potential_quote: str


# ---------------------------------------------------------------------------
# Orchestrator outputs — the Feature Brief
# ---------------------------------------------------------------------------

class RankedOpportunity(BaseModel):
    rank: int
    feature_name: str
    one_liner: str
    evidence_strength: Literal["strong", "moderate", "emerging"]
    ai_uniqueness: Literal["high", "medium", "low"]
    effort_estimate: Literal["quick_win", "medium_lift", "major_investment"]
    supporting_quote: str


class BriefSection(BaseModel):
    section_id: str
    title: str
    content: str  # markdown prose
    supporting_quotes: List[str] = Field(default_factory=list)


class FeatureBrief(BaseModel):
    title: str
    company_name: str = "Unknown"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    headline_insight: str
    top_3_insights: List[str]
    ranked_opportunities: List[RankedOpportunity]
    sections: List[BriefSection]
    call_to_action: str
    full_markdown: str
    # evidence counts for display
    reddit_count: int = 0
    app_store_count: int = 0
    seed_count: int = 0
    # full agent outputs — powering Evidence tab
    themes: List["Theme"] = Field(default_factory=list)
    power_quotes: List["PowerQuote"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE event envelope (reused pattern)
# ---------------------------------------------------------------------------

class SSEEvent(BaseModel):
    type: str
    agent: Optional[str] = None
    phase: Optional[str] = None
    message: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Extended thinking block
# ---------------------------------------------------------------------------

class ThinkingBlock(BaseModel):
    thinking_text: str
    token_count: int
