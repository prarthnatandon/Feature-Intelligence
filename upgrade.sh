#!/bin/bash
# Feature Intelligence Upgrade Script
# Run from your repo root: bash upgrade.sh
# Adds: RAG layer, eval framework, cost tracking
# Removes: dead code (graph.py, vision.py)

set -e

echo "🔧 Feature Intelligence Upgrade"
echo "================================"
echo ""

# Safety check
if [ ! -f "backend/main.py" ]; then
    echo "❌ Run this from your repo root (the folder with backend/)"
    exit 1
fi

echo "📁 Creating backup of modified files..."
mkdir -p _backup
cp backend/agents.py _backup/agents.py.bak 2>/dev/null || true
cp backend/main.py _backup/main.py.bak 2>/dev/null || true
cp backend/models.py _backup/models.py.bak 2>/dev/null || true
cp requirements.txt _backup/requirements.txt.bak 2>/dev/null || true

echo "🗑️  Removing dead code..."
rm -f backend/graph.py
rm -f backend/vision.py

echo "📝 Writing backend/rag.py (NEW)..."
cat > backend/rag.py << 'RAGEOF'
"""
Retrieval-Augmented Generation layer for Feature Intelligence.

Replaces naive context-stuffing (dumping 120 items into the prompt) with
semantic retrieval: each agent gets the *most relevant* feedback for its task.

Architecture:
    1. FeedbackVectorStore — TF-IDF vectorization + cosine similarity
    2. AgentRetriever     — per-agent retrieval strategies with token budgets
    3. RetrievalMetrics   — tracks retrieval quality for eval framework

Why TF-IDF over dense embeddings?
    - Zero additional API cost (no embedding model calls)
    - Sub-second indexing for <10K documents
    - Bigram features capture product-specific phrases ("streak anxiety", "AI tutor")
    - For this corpus size, TF-IDF + cosine rivals dense retrieval quality
    - Keeps the project self-contained (no external embedding service dependency)

Production upgrade path: swap TF-IDF for Voyage AI embeddings + Supabase pgvector
by implementing the VectorStore protocol with a different backend.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.models import CompanyContext, FeedbackBundle, FeedbackItem, Theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics — feeds into eval framework
# ---------------------------------------------------------------------------

@dataclass
class RetrievalEvent:
    """Single retrieval call — logged for eval analysis."""
    agent: str
    query: str
    top_k: int
    results_returned: int
    avg_score: float
    max_score: float
    min_score: float
    duration_ms: float
    source_distribution: Dict[str, int]


@dataclass
class RetrievalMetrics:
    """Accumulated metrics across all retrieval calls in a pipeline run."""
    events: List[RetrievalEvent] = field(default_factory=list)

    def record(self, event: RetrievalEvent) -> None:
        self.events.append(event)

    @property
    def total_retrievals(self) -> int:
        return len(self.events)

    @property
    def avg_relevance_score(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.avg_score for e in self.events) / len(self.events)

    @property
    def by_agent(self) -> Dict[str, List[RetrievalEvent]]:
        result: Dict[str, List[RetrievalEvent]] = {}
        for e in self.events:
            result.setdefault(e.agent, []).append(e)
        return result

    def summary(self) -> Dict:
        return {
            "total_retrievals": self.total_retrievals,
            "avg_relevance": round(self.avg_relevance_score, 4),
            "by_agent": {
                agent: {
                    "calls": len(events),
                    "avg_score": round(sum(e.avg_score for e in events) / len(events), 4),
                    "total_results": sum(e.results_returned for e in events),
                }
                for agent, events in self.by_agent.items()
            },
        }


# ---------------------------------------------------------------------------
# Vector store — TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

class FeedbackVectorStore:
    """
    In-memory vector store for feedback items.

    Uses TF-IDF with bigrams for feature extraction and cosine similarity
    for retrieval. Designed to be swapped for a pgvector backend in production.

    Usage:
        store = FeedbackVectorStore(bundle.items)
        results = store.search("AI conversation practice", top_k=20)
        # → [(FeedbackItem, 0.87), (FeedbackItem, 0.72), ...]
    """

    def __init__(
        self,
        items: List[FeedbackItem],
        max_features: int = 8000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.92,
    ):
        self.items = items
        self.texts = [item.text for item in items]
        self._item_hashes = [self._hash(t) for t in self.texts]

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df if len(items) > 10 else 1,
            max_df=max_df,
            sublinear_tf=True,       # log-normalize term frequencies
            strip_accents="unicode",
        )

        t0 = time.perf_counter()
        self.vectors = self.vectorizer.fit_transform(self.texts)
        self._index_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"FeedbackVectorStore indexed {len(items)} items "
            f"({self.vectors.shape[1]} features) in {self._index_time_ms:.0f}ms"
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()

    @property
    def size(self) -> int:
        return len(self.items)

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_filter: Optional[str] = None,
        min_score: float = 0.03,
        agent: str = "",
        metrics: Optional[RetrievalMetrics] = None,
    ) -> List[Tuple[FeedbackItem, float]]:
        """
        Retrieve top-k feedback items semantically similar to query.

        Args:
            query: Natural language search query
            top_k: Maximum results to return
            source_filter: Optional — restrict to "reddit", "app_store", etc.
            min_score: Minimum cosine similarity threshold
            agent: Agent name (for metrics tracking)
            metrics: Optional RetrievalMetrics to log this call
        """
        t0 = time.perf_counter()
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        # Apply source filter
        if source_filter:
            mask = np.array([item.source == source_filter for item in self.items])
            similarities = similarities * mask

        # Sort descending
        top_indices = np.argsort(similarities)[::-1]

        results: List[Tuple[FeedbackItem, float]] = []
        seen_hashes: set = set()

        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break

            # Deduplicate near-identical feedback
            h = self._item_hashes[idx]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            results.append((self.items[idx], score))
            if len(results) >= top_k:
                break

        duration_ms = (time.perf_counter() - t0) * 1000

        # Log metrics
        if metrics and results:
            scores = [s for _, s in results]
            source_dist: Dict[str, int] = {}
            for item, _ in results:
                source_dist[item.source] = source_dist.get(item.source, 0) + 1

            metrics.record(RetrievalEvent(
                agent=agent,
                query=query[:100],
                top_k=top_k,
                results_returned=len(results),
                avg_score=sum(scores) / len(scores),
                max_score=max(scores),
                min_score=min(scores),
                duration_ms=duration_ms,
                source_distribution=source_dist,
            ))

        return results

    def search_multi(
        self,
        queries: List[str],
        top_k: int = 30,
        min_score: float = 0.03,
        agent: str = "",
        metrics: Optional[RetrievalMetrics] = None,
    ) -> List[Tuple[FeedbackItem, float]]:
        """
        Multi-query retrieval — search with several queries, merge + deduplicate.
        Uses max-score fusion: if an item appears in multiple query results,
        keep the highest similarity score.
        """
        merged: Dict[str, Tuple[FeedbackItem, float]] = {}

        per_query_k = max(10, top_k // len(queries) + 5)

        for query in queries:
            for item, score in self.search(
                query, top_k=per_query_k, min_score=min_score,
                agent=agent, metrics=metrics,
            ):
                key = self._hash(item.text)
                if key not in merged or merged[key][1] < score:
                    merged[key] = (item, score)

        sorted_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_theme_feedback(
        self,
        theme: Theme,
        top_k: int = 25,
        agent: str = "",
        metrics: Optional[RetrievalMetrics] = None,
    ) -> List[FeedbackItem]:
        """
        Retrieve feedback most relevant to a specific theme.
        Builds multi-query from theme name + description + example quotes.
        """
        queries = [
            theme.theme_name,
            theme.description,
        ] + theme.example_quotes[:2]

        results = self.search_multi(
            queries, top_k=top_k, agent=agent, metrics=metrics,
        )
        return [item for item, _ in results]


# ---------------------------------------------------------------------------
# Per-agent retrieval strategies
# ---------------------------------------------------------------------------

class AgentRetriever:
    """
    Wraps FeedbackVectorStore with agent-specific retrieval strategies.

    Each agent has different needs:
    - ThemeAgent: broad, diverse sample to discover clusters
    - FeasibilityAgent: theme-specific feedback for each feasibility rating
    - GapAgent: feedback mentioning product features / competitors
    - QuoteAgent: emotionally resonant, high-signal quotes
    """

    def __init__(self, store: FeedbackVectorStore, metrics: Optional[RetrievalMetrics] = None):
        self.store = store
        self.metrics = metrics or RetrievalMetrics()

    def for_theme_agent(
        self,
        company: CompanyContext,
        max_items: int = 150,
    ) -> str:
        """
        Broad retrieval for theme discovery.
        Strategy: multiple diverse queries to surface different complaint clusters.
        Includes upvote-weighted sampling to prioritize high-signal feedback.
        """
        # Diverse seed queries to surface different feedback clusters
        queries = [
            f"{company.company_name} frustrating",
            f"{company.company_name} wish feature",
            f"{company.company_name} missing need",
            f"{company.company_name} love best",
            f"{company.company_name} competitor better alternative",
            f"AI {company.company_name} should",
            f"{company.company_name} subscription pricing worth",
            f"{company.company_name} advanced learner",
        ]

        # Get semantically diverse results
        retrieved = self.store.search_multi(
            queries, top_k=max_items,
            agent="ThemeAgent", metrics=self.metrics,
        )

        # Also include top upvoted items that might not match any query
        upvote_sorted = sorted(
            self.store.items, key=lambda x: x.upvotes, reverse=True
        )[:30]

        # Merge: retrieved items + high-upvote items, deduplicated
        seen = {self.store._hash(item.text) for item, _ in retrieved}
        all_items = [item for item, _ in retrieved]

        for item in upvote_sorted:
            h = self.store._hash(item.text)
            if h not in seen:
                seen.add(h)
                all_items.append(item)
            if len(all_items) >= max_items:
                break

        return self._format_items(all_items[:max_items])

    def for_feasibility_agent(
        self,
        themes: List[Theme],
        company: CompanyContext,
        per_theme: int = 8,
    ) -> str:
        """
        Theme-specific retrieval for feasibility rating.
        Strategy: for each theme, retrieve the most relevant feedback
        so the agent has concrete evidence when rating AI solvability.
        """
        sections = []
        for theme in themes:
            relevant = self.store.get_theme_feedback(
                theme, top_k=per_theme,
                agent="AIFeasibilityAgent", metrics=self.metrics,
            )
            if relevant:
                items_text = "\n".join(
                    f"  - [{i.source}] {i.text[:300]}" for i in relevant
                )
                sections.append(
                    f"### {theme.theme_name}\n"
                    f"Relevant feedback ({len(relevant)} items):\n{items_text}"
                )

        return "\n\n".join(sections)

    def for_quote_agent(
        self,
        company: CompanyContext,
        max_items: int = 80,
    ) -> str:
        """
        Quote-optimized retrieval.
        Strategy: queries targeting emotionally resonant, specific,
        actionable language — the kind that makes a product brief compelling.
        """
        queries = [
            f"{company.company_name} I wish I could",
            f"{company.company_name} frustrating because",
            f"{company.company_name} the problem is",
            f"{company.company_name} switched to alternative",
            f"{company.company_name} would pay for",
            f"{company.company_name} years using still can't",
            f"{company.company_name} love about it",
            f"{company.company_name} deal breaker",
        ]

        retrieved = self.store.search_multi(
            queries, top_k=max_items,
            agent="QuoteAgent", metrics=self.metrics,
        )

        items = [item for item, _ in retrieved]
        # Boost: longer items tend to be more quotable
        items.sort(key=lambda x: (x.upvotes * 0.3 + len(x.text) * 0.01), reverse=True)

        return self._format_items(items[:max_items])

    @staticmethod
    def _format_items(items: List[FeedbackItem], max_chars: int = 500) -> str:
        """Serialize feedback items into compact text for Claude context."""
        lines = []
        for item in items:
            prefix = f"[{item.source.upper()}"
            if item.upvotes:
                prefix += f" +{item.upvotes}"
            prefix += "] "
            lines.append(prefix + item.text[:max_chars])
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: build store from bundle
# ---------------------------------------------------------------------------

def build_vector_store(bundle: FeedbackBundle) -> FeedbackVectorStore:
    """Create a vector store from a feedback bundle."""
    if not bundle.items:
        raise ValueError("Cannot build vector store from empty bundle")
    return FeedbackVectorStore(bundle.items)

RAGEOF

echo "📝 Writing backend/evals.py (NEW)..."
cat > backend/evals.py << 'EVALSEOF'
"""
Evaluation framework for Feature Intelligence pipeline.

Measures quality across four dimensions:
    1. Theme Stability   — do repeated runs find the same themes?
    2. Quote Grounding   — are brief claims backed by real source data?
    3. Brief Quality     — LLM-as-judge scoring (actionability, evidence, specificity)
    4. Retrieval Quality — are RAG results relevant? (from RetrievalMetrics)

Design:
    - Each eval returns a typed EvalResult with score + details
    - Results persist to disk (JSON) for longitudinal tracking
    - CLI entry point: `python -m backend.evals --suite all`
    - Eval results render in the UI methodology tab

This is the layer that separates "demo" from "system" — it proves
the pipeline produces reliable, grounded, high-quality output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from backend.models import (
    CompanyContext,
    FeatureBrief,
    FeedbackBundle,
    Theme,
)
from backend.rag import RetrievalMetrics

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent.parent / "evals"
EVALS_DIR.mkdir(exist_ok=True)

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Eval result models
# ---------------------------------------------------------------------------

class EvalScore(BaseModel):
    """Individual evaluation score."""
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class EvalResult(BaseModel):
    """Complete evaluation result for a pipeline run."""
    run_id: str
    company: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scores: List[EvalScore] = Field(default_factory=list)
    overall_score: float = 0.0
    retrieval_metrics: Optional[Dict] = None
    duration_seconds: float = 0.0

    def compute_overall(self) -> None:
        """Weighted average of dimension scores."""
        if not self.scores:
            self.overall_score = 0.0
            return
        weights = {
            "theme_stability": 0.2,
            "quote_grounding": 0.3,
            "brief_quality": 0.35,
            "retrieval_quality": 0.15,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for s in self.scores:
            w = weights.get(s.dimension, 0.1)
            weighted_sum += s.score * w
            total_weight += w
        self.overall_score = round(weighted_sum / total_weight, 4) if total_weight else 0.0


# ---------------------------------------------------------------------------
# Eval 1: Theme Stability
# ---------------------------------------------------------------------------

async def eval_theme_stability(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    n_runs: int = 3,
    event_queue: Optional[asyncio.Queue] = None,
) -> EvalScore:
    """
    Run theme analysis N times and measure consistency.

    Metric: Jaccard similarity of theme name sets across runs.
    High stability = the system reliably finds the same themes.
    Low stability = themes are noisy / order-dependent.
    """
    from backend.agents import run_theme_agent

    theme_sets: List[set] = []

    for i in range(n_runs):
        logger.info(f"Theme stability eval: run {i+1}/{n_runs}")
        queue = event_queue or asyncio.Queue()
        try:
            analysis = await run_theme_agent(client, bundle, company, queue)
            # Normalize theme names for comparison
            names = {t.theme_name.lower().strip() for t in analysis.themes}
            theme_sets.append(names)
        except Exception as e:
            logger.warning(f"Theme stability run {i+1} failed: {e}")
            continue

    if len(theme_sets) < 2:
        return EvalScore(
            dimension="theme_stability",
            score=0.0,
            reasoning="Insufficient successful runs for stability measurement",
        )

    # Pairwise Jaccard similarity
    jaccard_scores = []
    for i in range(len(theme_sets)):
        for j in range(i + 1, len(theme_sets)):
            intersection = len(theme_sets[i] & theme_sets[j])
            union = len(theme_sets[i] | theme_sets[j])
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)

    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)

    # Also compute semantic overlap using fuzzy matching
    # (themes might be named differently but mean the same thing)
    fuzzy_score = _fuzzy_theme_overlap(theme_sets)

    combined = (avg_jaccard * 0.4) + (fuzzy_score * 0.6)

    return EvalScore(
        dimension="theme_stability",
        score=round(combined, 4),
        details={
            "n_runs": len(theme_sets),
            "exact_jaccard": round(avg_jaccard, 4),
            "fuzzy_overlap": round(fuzzy_score, 4),
            "theme_counts": [len(s) for s in theme_sets],
            "common_themes": list(set.intersection(*theme_sets)) if theme_sets else [],
        },
        reasoning=(
            f"Across {len(theme_sets)} runs, exact theme name overlap was "
            f"{avg_jaccard:.0%} and fuzzy semantic overlap was {fuzzy_score:.0%}. "
            f"Combined stability score: {combined:.0%}."
        ),
    )


def _fuzzy_theme_overlap(theme_sets: List[set]) -> float:
    """
    Fuzzy theme matching — two themes "match" if they share 2+ words.
    More forgiving than exact Jaccard for names like
    "AI Conversation Coach" vs "Conversational AI Practice".
    """
    if len(theme_sets) < 2:
        return 0.0

    scores = []
    for i in range(len(theme_sets)):
        for j in range(i + 1, len(theme_sets)):
            matched = 0
            set_a, set_b = theme_sets[i], theme_sets[j]
            for a in set_a:
                a_words = set(a.split())
                best_overlap = max(
                    (len(a_words & set(b.split())) / max(len(a_words), 1) for b in set_b),
                    default=0.0,
                )
                if best_overlap >= 0.4:  # 40%+ word overlap = fuzzy match
                    matched += 1
            total = max(len(set_a), len(set_b))
            scores.append(matched / total if total else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Eval 2: Quote Grounding
# ---------------------------------------------------------------------------

async def eval_quote_grounding(
    brief: FeatureBrief,
    bundle: FeedbackBundle,
) -> EvalScore:
    """
    Verify that quotes in the brief actually exist in the source data.

    For each quote referenced in the brief (opportunity cards, sections),
    search the original feedback for a close match. Grounding score =
    percentage of quotes that can be traced to real source data.

    This is a lightweight hallucination detector.
    """
    from backend.rag import FeedbackVectorStore

    store = FeedbackVectorStore(bundle.items)

    # Collect all quotes from the brief
    all_quotes: List[str] = []

    for opp in brief.ranked_opportunities:
        if opp.supporting_quote:
            all_quotes.append(opp.supporting_quote)

    for section in brief.sections:
        all_quotes.extend(section.supporting_quotes)

    for pq in brief.power_quotes:
        all_quotes.append(pq.quote_text)

    if not all_quotes:
        return EvalScore(
            dimension="quote_grounding",
            score=1.0,
            reasoning="No quotes to verify",
        )

    # Deduplicate
    unique_quotes = list(set(all_quotes))

    grounded = 0
    ungrounded: List[str] = []

    for quote in unique_quotes:
        results = store.search(quote, top_k=3, min_score=0.15)
        if results:
            # Check if any result is a close match
            best_score = results[0][1]
            if best_score >= 0.2:
                grounded += 1
            else:
                ungrounded.append(quote[:80])
        else:
            ungrounded.append(quote[:80])

    score = grounded / len(unique_quotes) if unique_quotes else 1.0

    return EvalScore(
        dimension="quote_grounding",
        score=round(score, 4),
        details={
            "total_quotes": len(unique_quotes),
            "grounded": grounded,
            "ungrounded_count": len(ungrounded),
            "ungrounded_samples": ungrounded[:5],
        },
        reasoning=(
            f"{grounded}/{len(unique_quotes)} quotes ({score:.0%}) could be traced "
            f"back to source feedback data. "
            f"{'All quotes grounded.' if not ungrounded else f'{len(ungrounded)} quotes could not be verified against source data.'}"
        ),
    )


# ---------------------------------------------------------------------------
# Eval 3: Brief Quality (LLM-as-Judge)
# ---------------------------------------------------------------------------

BRIEF_EVAL_RUBRIC = """You are evaluating a Product Feature Opportunity Brief.
Score each dimension from 0.0 to 1.0.

DIMENSIONS:
1. actionability (0-1): Are recommendations specific enough for a PM to act on? Does it say WHAT to build, not just what users want?
2. evidence_grounding (0-1): Is every claim supported by user quotes or data? Or are there unsupported assertions?
3. specificity (0-1): Are features described concretely (specific AI techniques, UX flows) vs vaguely ("improve the experience")?
4. coherence (0-1): Does the brief flow logically? Do sections build on each other? Is there internal consistency?
5. insight_depth (0-1): Does it surface non-obvious findings? Or just restate complaints? Would a PM learn something new?

Respond ONLY with JSON:
{
    "actionability": 0.X,
    "evidence_grounding": 0.X,
    "specificity": 0.X,
    "coherence": 0.X,
    "insight_depth": 0.X,
    "overall_reasoning": "2-3 sentences explaining the scores"
}"""


async def eval_brief_quality(
    client: anthropic.AsyncAnthropic,
    brief: FeatureBrief,
) -> EvalScore:
    """
    LLM-as-judge evaluation of brief quality across 5 dimensions.
    Uses a structured rubric to score actionability, evidence,
    specificity, coherence, and insight depth.
    """
    # Truncate brief to stay within reasonable token limits
    brief_text = brief.full_markdown[:6000] if brief.full_markdown else "No brief content."

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=BRIEF_EVAL_RUBRIC,
            messages=[{
                "role": "user",
                "content": f"Evaluate this brief:\n\n{brief_text}",
            }],
        )

        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        scores_data = json.loads(text)

        dimensions = ["actionability", "evidence_grounding", "specificity",
                       "coherence", "insight_depth"]
        dim_scores = {d: float(scores_data.get(d, 0.5)) for d in dimensions}
        avg_score = sum(dim_scores.values()) / len(dim_scores)

        return EvalScore(
            dimension="brief_quality",
            score=round(avg_score, 4),
            details=dim_scores,
            reasoning=scores_data.get("overall_reasoning", ""),
        )

    except Exception as e:
        logger.warning(f"Brief quality eval failed: {e}")
        return EvalScore(
            dimension="brief_quality",
            score=0.5,
            reasoning=f"Eval failed: {e}",
        )


# ---------------------------------------------------------------------------
# Eval 4: Retrieval Quality
# ---------------------------------------------------------------------------

def eval_retrieval_quality(
    retrieval_metrics: RetrievalMetrics,
) -> EvalScore:
    """
    Score RAG retrieval quality from accumulated metrics.

    Scoring:
    - avg relevance score > 0.15 = good
    - source diversity (not all from one source) = good
    - coverage (returned results close to top_k) = good
    """
    if not retrieval_metrics.events:
        return EvalScore(
            dimension="retrieval_quality",
            score=0.0,
            reasoning="No retrieval events recorded — RAG layer may not be active.",
        )

    events = retrieval_metrics.events

    # Relevance score (0-1, normalized from typical TF-IDF cosine range 0-0.5)
    avg_relevance = retrieval_metrics.avg_relevance_score
    relevance_normalized = min(1.0, avg_relevance / 0.25)

    # Coverage: ratio of results returned vs requested
    coverage_scores = [
        e.results_returned / e.top_k if e.top_k > 0 else 0.0
        for e in events
    ]
    avg_coverage = sum(coverage_scores) / len(coverage_scores)

    # Source diversity: average entropy of source distributions
    diversity_scores = []
    for e in events:
        total = sum(e.source_distribution.values())
        if total == 0:
            continue
        probs = [v / total for v in e.source_distribution.values()]
        n_sources = len(probs)
        if n_sources <= 1:
            diversity_scores.append(0.0)
        else:
            import math
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(n_sources)
            diversity_scores.append(entropy / max_entropy if max_entropy > 0 else 0.0)

    avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    combined = (relevance_normalized * 0.5) + (avg_coverage * 0.3) + (avg_diversity * 0.2)

    return EvalScore(
        dimension="retrieval_quality",
        score=round(combined, 4),
        details={
            "avg_relevance": round(avg_relevance, 4),
            "avg_coverage": round(avg_coverage, 4),
            "avg_diversity": round(avg_diversity, 4),
            "total_retrievals": len(events),
            "by_agent": retrieval_metrics.summary().get("by_agent", {}),
        },
        reasoning=(
            f"Across {len(events)} retrievals: avg relevance {avg_relevance:.3f}, "
            f"coverage {avg_coverage:.0%}, source diversity {avg_diversity:.0%}."
        ),
    )


# ---------------------------------------------------------------------------
# Full eval suite
# ---------------------------------------------------------------------------

class EvalSuite:
    """
    Runs all evaluations and persists results.

    Usage:
        suite = EvalSuite(client)
        result = await suite.run(brief, bundle, company, retrieval_metrics)
        # result.overall_score → 0.82
    """

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def run(
        self,
        brief: FeatureBrief,
        bundle: FeedbackBundle,
        company: CompanyContext,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        run_id: str = "",
        run_stability_eval: bool = False,
    ) -> EvalResult:
        """
        Run the full evaluation suite.

        Args:
            run_stability_eval: If True, runs theme stability eval (slow — 3x pipeline).
                                Default False for normal runs; True for eval-specific runs.
        """
        t0 = time.time()
        result = EvalResult(run_id=run_id, company=company.company_name)
        scores: List[EvalScore] = []

        # 1. Quote grounding (fast — no API calls)
        logger.info("Running quote grounding eval...")
        scores.append(await eval_quote_grounding(brief, bundle))

        # 2. Brief quality (1 API call)
        logger.info("Running brief quality eval...")
        scores.append(await eval_brief_quality(self.client, brief))

        # 3. Retrieval quality (no API calls — from metrics)
        if retrieval_metrics:
            logger.info("Running retrieval quality eval...")
            scores.append(eval_retrieval_quality(retrieval_metrics))
            result.retrieval_metrics = retrieval_metrics.summary()

        # 4. Theme stability (slow — N full theme agent runs)
        if run_stability_eval:
            logger.info("Running theme stability eval (this will take a while)...")
            scores.append(
                await eval_theme_stability(self.client, bundle, company, n_runs=3)
            )

        result.scores = scores
        result.duration_seconds = round(time.time() - t0, 2)
        result.compute_overall()

        # Persist
        self._save(result)

        return result

    def _save(self, result: EvalResult) -> None:
        """Save eval result to disk."""
        try:
            filename = f"{result.company}_{result.run_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            path = EVALS_DIR / filename
            path.write_text(result.model_dump_json(indent=2))
            logger.info(f"Eval saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save eval: {e}")

    @staticmethod
    def load_history(company: str = "", limit: int = 20) -> List[EvalResult]:
        """Load historical eval results for trend analysis."""
        results = []
        for path in sorted(EVALS_DIR.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text())
                result = EvalResult(**data)
                if not company or result.company.lower() == company.lower():
                    results.append(result)
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _cli_main():
    """Run evals from the command line using seed data."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Feature Intelligence evals")
    parser.add_argument("--company", default="Duolingo")
    parser.add_argument("--stability", action="store_true", help="Run slow theme stability eval")
    parser.add_argument("--history", action="store_true", help="Show eval history")
    args = parser.parse_args()

    if args.history:
        results = EvalSuite.load_history(args.company)
        for r in results:
            print(f"{r.timestamp.isoformat()} | {r.company} | overall={r.overall_score:.2f} | "
                  + " | ".join(f"{s.dimension}={s.score:.2f}" for s in r.scores))
        return

    # Full eval requires running the pipeline first
    print(f"To run evals, use the /eval/{{run_id}} API endpoint after completing an analysis.")
    print(f"Or run: python -m backend.evals --history to see past results.")


if __name__ == "__main__":
    asyncio.run(_cli_main())

EVALSEOF

echo "📝 Writing backend/cost_tracker.py (NEW)..."
cat > backend/cost_tracker.py << 'COSTEOF'
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

COSTEOF

echo "📝 Replacing backend/agents.py..."
cat > backend/agents.py << 'AGENTSEOF'
"""
Multi-agent pipeline for Product Feature Intelligence.

Phase 1 — 4 specialist agents run in TWO WAVES via asyncio.gather():
    Wave 1: ThemeAgent + QuoteAgent (independent — run in parallel)
    Wave 2: AIFeasibilityAgent + GapAgent (depend on theme output)

Phase 2 — Orchestrator with extended thinking synthesizes all 4 outputs
           into a Feature Opportunity Brief.

RAG Integration:
    All agents use AgentRetriever to get semantically relevant feedback
    instead of raw context-stuffing. Vector store is built once from the
    FeedbackBundle and shared across all agents.

Cost Tracking:
    Every API call is recorded via CostTracker. Token usage and estimated
    dollar costs are surfaced per agent and in aggregate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import anthropic

from backend.cost_tracker import CostTracker
from backend.models import (
    AIFeasibility,
    BriefSection,
    CompanyContext,
    FeasibilityAnalysis,
    FeedbackBundle,
    FeatureBrief,
    GapAnalysis,
    GapFinding,
    PowerQuote,
    QuoteAnalysis,
    RankedOpportunity,
    SSEEvent,
    StrengthFinding,
    Theme,
    ThemeAnalysis,
    ThinkingBlock,
)
from backend.prompts import (
    feasibility_agent_system,
    gap_agent_system,
    orchestrator_system,
    quote_agent_system,
    theme_agent_system,
)
from backend.rag import (
    AgentRetriever,
    FeedbackVectorStore,
    RetrievalMetrics,
    build_vector_store,
)

logger = logging.getLogger(__name__)
MODEL = "claude-sonnet-4-6"

# Limit concurrent Anthropic API calls to avoid rate-limit errors
_api_semaphore = asyncio.Semaphore(2)


# ===========================================================================
# Tool definitions (unchanged — same as original)
# ===========================================================================

THEME_TOOLS = [
    {
        "name": "record_theme",
        "description": "Record a distinct user need cluster (theme) identified in the feedback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "theme_name": {"type": "string", "description": "Precise product-opportunity name"},
                "description": {"type": "string", "description": "2-3 sentence description of the unmet need"},
                "example_quotes": {
                    "type": "array", "items": {"type": "string"},
                    "description": "2-3 direct user quotes that represent this theme",
                },
                "frequency_estimate": {
                    "type": "string", "enum": ["very_high", "high", "medium", "low"],
                    "description": "How often does this theme appear in the feedback?",
                },
                "user_segment": {
                    "type": "string",
                    "description": "Which user segment feels this most acutely",
                },
            },
            "required": ["theme_name", "description", "example_quotes", "frequency_estimate", "user_segment"],
        },
    },
    {
        "name": "emit_theme_summary",
        "description": "Called once after all themes are recorded. Provide aggregate summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "total_feedback_items": {"type": "integer"},
                "dominant_frustration": {"type": "string"},
                "dominant_desire": {"type": "string"},
            },
            "required": ["total_feedback_items", "dominant_frustration", "dominant_desire"],
        },
    },
]

FEASIBILITY_TOOLS = [
    {
        "name": "rate_ai_feasibility",
        "description": "Evaluate how well AI could address a specific user need theme.",
        "input_schema": {
            "type": "object",
            "properties": {
                "theme_name": {"type": "string"},
                "feasibility_score": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "0=AI can't help, 1=AI is the perfect solution",
                },
                "ai_approach": {"type": "string", "description": "Specific AI technique"},
                "why_ai_uniquely_suited": {"type": "string"},
                "technical_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                "comparable_product": {"type": "string"},
            },
            "required": ["theme_name", "feasibility_score", "ai_approach",
                         "why_ai_uniquely_suited", "technical_complexity", "comparable_product"],
        },
    },
    {
        "name": "emit_feasibility_summary",
        "description": "Called once after all themes are rated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_ai_opportunity": {"type": "string"},
                "hardest_to_solve": {"type": "string"},
            },
            "required": ["top_ai_opportunity", "hardest_to_solve"],
        },
    },
]

GAP_TOOLS = [
    {
        "name": "record_gap",
        "description": "Record a gap between what users need and what the company currently offers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "theme_name": {"type": "string"},
                "gap_type": {"type": "string", "enum": ["missing_entirely", "partial_solution", "poor_execution"]},
                "existing_feature": {"type": "string"},
                "gap_description": {"type": "string"},
                "market_evidence": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "knowledge_basis": {
                    "type": "string",
                    "enum": ["user_provided_features", "llm_knowledge", "user_feedback_only"],
                },
            },
            "required": ["theme_name", "gap_type", "existing_feature",
                         "gap_description", "market_evidence", "confidence", "knowledge_basis"],
        },
    },
    {
        "name": "record_strength",
        "description": "Record something the company currently does well that users love.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_name": {"type": "string"},
                "why_users_love_it": {"type": "string"},
                "quote": {"type": "string"},
            },
            "required": ["feature_name", "why_users_love_it", "quote"],
        },
    },
    {
        "name": "emit_gap_summary",
        "description": "Called once after all gaps and strengths are recorded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "biggest_gap": {"type": "string"},
                "quick_win": {"type": "string"},
            },
            "required": ["biggest_gap", "quick_win"],
        },
    },
]

QUOTE_TOOLS = [
    {
        "name": "record_power_quote",
        "description": "Record a compelling user quote for the evidence layer of the product brief.",
        "input_schema": {
            "type": "object",
            "properties": {
                "quote_text": {"type": "string"},
                "source": {"type": "string", "enum": ["reddit", "app_store", "seed"]},
                "theme": {"type": "string"},
                "why_compelling": {"type": "string"},
            },
            "required": ["quote_text", "source", "theme", "why_compelling"],
        },
    },
    {
        "name": "emit_quote_summary",
        "description": "Called once after all quotes are recorded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "most_compelling_quote": {"type": "string"},
                "most_viral_potential_quote": {"type": "string"},
            },
            "required": ["most_compelling_quote", "most_viral_potential_quote"],
        },
    },
]

ORCHESTRATOR_TOOLS = [
    {
        "name": "write_brief_section",
        "description": "Write one section of the Feature Opportunity Brief.",
        "input_schema": {
            "type": "object",
            "properties": {
                "section_id": {
                    "type": "string",
                    "enum": ["executive_summary", "market_signal", "top_opportunities",
                             "competitive_context", "recommended_next_feature", "methodology"],
                },
                "title": {"type": "string"},
                "content": {"type": "string", "description": "Polished markdown prose (200-500 words)"},
                "supporting_quotes": {
                    "type": "array", "items": {"type": "string"},
                    "description": "2-4 user quotes that support this section",
                },
            },
            "required": ["section_id", "title", "content", "supporting_quotes"],
        },
    },
    {
        "name": "rank_opportunities",
        "description": "Submit the final ranked list of AI feature opportunities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "feature_name": {"type": "string"},
                            "one_liner": {"type": "string"},
                            "evidence_strength": {"type": "string", "enum": ["strong", "moderate", "emerging"]},
                            "ai_uniqueness": {"type": "string", "enum": ["high", "medium", "low"]},
                            "effort_estimate": {"type": "string", "enum": ["quick_win", "medium_lift", "major_investment"]},
                            "supporting_quote": {"type": "string"},
                        },
                        "required": ["rank", "feature_name", "one_liner", "evidence_strength",
                                     "ai_uniqueness", "effort_estimate", "supporting_quote"],
                    },
                },
            },
            "required": ["opportunities"],
        },
    },
    {
        "name": "finalize_brief",
        "description": "Called once all sections and rankings are done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline_insight": {"type": "string"},
                "top_3_insights": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                "call_to_action": {"type": "string"},
            },
            "required": ["headline_insight", "top_3_insights", "call_to_action"],
        },
    },
]


# ===========================================================================
# Core agentic loop — now with cost tracking
# ===========================================================================

async def _run_agent_loop(
    client: anthropic.AsyncAnthropic,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    event_queue: asyncio.Queue,
    cost_tracker: Optional[CostTracker] = None,
    max_iterations: int = 20,
    on_tool_call: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Generic agentic tool-use loop with cost tracking.
    Returns dict of {tool_name: [list of inputs collected]}.
    """
    messages = [{"role": "user", "content": user_message}]
    collected: Dict[str, list] = defaultdict(list)
    iterations = 0

    await event_queue.put(SSEEvent(type="agent_start", agent=agent_name))

    while iterations < max_iterations:
        iterations += 1

        t0 = time.perf_counter()
        async with _api_semaphore:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                tool_choice={"type": "auto"},
                messages=messages,
            )
        duration_ms = (time.perf_counter() - t0) * 1000

        # Track cost
        if cost_tracker:
            cost_tracker.record(agent_name, response, MODEL, duration_ms)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                collected[block.name].append(block.input)
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent=agent_name,
                    tool=block.name, tool_input=block.input,
                ))
                if on_tool_call:
                    await on_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Recorded: {block.name}",
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    await event_queue.put(SSEEvent(type="agent_complete", agent=agent_name))
    return dict(collected)


# ===========================================================================
# Specialist agents — now with RAG retrieval
# ===========================================================================

async def run_theme_agent(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
    retriever: Optional[AgentRetriever] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> ThemeAnalysis:
    """
    ThemeAgent — identifies unmet need clusters.
    With RAG: uses semantically diverse retrieval instead of truncated dump.
    """
    if retriever:
        feedback_text = retriever.for_theme_agent(company, max_items=150)
        retrieval_note = " (RAG-retrieved: semantically diverse sample)"
    else:
        feedback_text = _feedback_to_text(bundle, max_items=120)
        retrieval_note = ""

    user_msg = (
        f"Analyze this batch of {bundle.total} user feedback items about "
        f"{company.company_name} ({company.product_description}).{retrieval_note}\n"
        f"Identify 8–12 distinct unmet need themes. For each, call record_theme.\n"
        f"End with emit_theme_summary.\n\n"
        f"FEEDBACK:\n{feedback_text}"
    )

    async def _on_tool(tool_name: str, tool_input: dict) -> None:
        if tool_name == "record_theme":
            await event_queue.put(SSEEvent(
                type="theme_discovered", agent="ThemeAgent", data=tool_input,
            ))

    raw = await _run_agent_loop(
        client, "ThemeAgent", theme_agent_system(company), user_msg,
        THEME_TOOLS, event_queue, cost_tracker=cost_tracker,
        on_tool_call=_on_tool,
    )

    themes = [Theme(**t) for t in raw.get("record_theme", [])]
    summary = (raw.get("emit_theme_summary") or [{}])[-1]

    return ThemeAnalysis(
        themes=themes,
        total_feedback_items=summary.get("total_feedback_items", bundle.total),
        dominant_frustration=summary.get("dominant_frustration", ""),
        dominant_desire=summary.get("dominant_desire", ""),
    )


async def run_feasibility_agent(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    themes: List[Theme],
    company: CompanyContext,
    event_queue: asyncio.Queue,
    retriever: Optional[AgentRetriever] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> FeasibilityAnalysis:
    """
    FeasibilityAgent — rates AI solvability per theme.
    With RAG: gets theme-specific feedback instead of random 40-item sample.
    """
    themes_text = json.dumps(
        [{"theme_name": t.theme_name, "description": t.description,
          "frequency": t.frequency_estimate, "segment": t.user_segment}
         for t in themes], indent=2,
    )

    if retriever:
        sample = retriever.for_feasibility_agent(themes, company, per_theme=8)
        retrieval_note = "\n\nBELOW: per-theme relevant feedback (RAG-retrieved):\n"
    else:
        sample = _feedback_to_text(bundle, max_items=40)
        retrieval_note = "\n\nSAMPLE FEEDBACK FOR CONTEXT:\n"

    user_msg = (
        f"Evaluate AI feasibility for each of the following {len(themes)} user need themes "
        f"from {company.company_name} feedback.\n"
        f"Call rate_ai_feasibility for each theme, then emit_feasibility_summary.\n\n"
        f"THEMES TO EVALUATE:\n{themes_text}"
        f"{retrieval_note}{sample}"
    )

    raw = await _run_agent_loop(
        client, "AIFeasibilityAgent", feasibility_agent_system(company), user_msg,
        FEASIBILITY_TOOLS, event_queue, cost_tracker=cost_tracker,
    )

    ratings = [AIFeasibility(**r) for r in raw.get("rate_ai_feasibility", [])]
    summary = (raw.get("emit_feasibility_summary") or [{}])[-1]

    return FeasibilityAnalysis(
        ratings=ratings,
        top_ai_opportunity=summary.get("top_ai_opportunity", ""),
        hardest_to_solve=summary.get("hardest_to_solve", ""),
    )


async def run_gap_agent(
    client: anthropic.AsyncAnthropic,
    themes: List[Theme],
    company: CompanyContext,
    event_queue: asyncio.Queue,
    cost_tracker: Optional[CostTracker] = None,
) -> GapAnalysis:
    themes_text = json.dumps(
        [{"theme_name": t.theme_name, "description": t.description,
          "example_quotes": t.example_quotes[:2]}
         for t in themes], indent=2,
    )

    user_msg = (
        f"Cross-reference these {len(themes)} user need themes against "
        f"{company.company_name}'s current feature set.\n"
        f"For each theme, call record_gap. Also call record_strength for 2-3 "
        f"things {company.company_name} does well.\nEnd with emit_gap_summary.\n\n"
        f"USER NEED THEMES:\n{themes_text}"
    )

    raw = await _run_agent_loop(
        client, "GapAgent", gap_agent_system(company), user_msg,
        GAP_TOOLS, event_queue, cost_tracker=cost_tracker,
    )

    gaps = [
        GapFinding(
            theme_name=g["theme_name"], gap_type=g["gap_type"],
            existing_feature=g.get("existing_feature", ""),
            gap_description=g["gap_description"],
            market_evidence=g.get("market_evidence", ""),
            confidence=g.get("confidence", "medium"),
            knowledge_basis=g.get("knowledge_basis", "llm_knowledge"),
        )
        for g in raw.get("record_gap", [])
    ]
    strengths = [StrengthFinding(**s) for s in raw.get("record_strength", [])]
    summary = (raw.get("emit_gap_summary") or [{}])[-1]

    return GapAnalysis(
        gaps=gaps, strengths=strengths,
        biggest_gap=summary.get("biggest_gap", ""),
        quick_win=summary.get("quick_win", ""),
    )


async def run_quote_agent(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
    retriever: Optional[AgentRetriever] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> QuoteAnalysis:
    """
    QuoteAgent — curates compelling user evidence.
    With RAG: retrieves emotionally resonant, high-signal quotes.
    """
    if retriever:
        feedback_text = retriever.for_quote_agent(company, max_items=80)
    else:
        # Fallback: manual selection
        reddit_top = sorted(
            [i for i in bundle.items if i.source == "reddit"],
            key=lambda x: x.upvotes, reverse=True
        )[:50]
        store_items = [i for i in bundle.items if i.source == "app_store"][:20]
        seed_items = [i for i in bundle.items if i.source == "seed"][:20]

        all_items = reddit_top + store_items + seed_items
        feedback_text = "\n\n".join(
            f"[{i.source.upper()}" + (f" +{i.upvotes}" if i.upvotes else "") + f"] {i.text}"
            for i in all_items
        )

    user_msg = (
        f"Mine these {company.company_name} user feedback items for the 15–20 most compelling, "
        f"quotable statements for a product strategy brief. Call record_power_quote for each, "
        f"then emit_quote_summary.\n\nFEEDBACK:\n{feedback_text}"
    )

    raw = await _run_agent_loop(
        client, "QuoteAgent", quote_agent_system(company), user_msg,
        QUOTE_TOOLS, event_queue, cost_tracker=cost_tracker,
    )

    quotes = [PowerQuote(**q) for q in raw.get("record_power_quote", [])]
    summary = (raw.get("emit_quote_summary") or [{}])[-1]

    return QuoteAnalysis(
        quotes=quotes,
        most_compelling_quote=summary.get("most_compelling_quote", ""),
        most_viral_potential_quote=summary.get("most_viral_potential_quote", ""),
    )


# ===========================================================================
# Orchestrator — extended thinking synthesis (unchanged core logic)
# ===========================================================================

async def run_orchestrator(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    theme_analysis: ThemeAnalysis,
    feasibility: FeasibilityAnalysis,
    gap_analysis: GapAnalysis,
    quote_analysis: QuoteAnalysis,
    event_queue: asyncio.Queue,
    cost_tracker: Optional[CostTracker] = None,
) -> FeatureBrief:

    await event_queue.put(SSEEvent(type="agent_start", agent="Orchestrator"))
    await event_queue.put(SSEEvent(
        type="phase_start", phase="synthesis",
        message="Orchestrator synthesizing with extended thinking...",
    ))

    def _short(text: str, max_chars: int = 120) -> str:
        return text[:max_chars] + "..." if len(text) > max_chars else text

    context = {
        "data": f"{bundle.total} items (Reddit:{bundle.reddit_count} AppStore:{bundle.app_store_count} Seed:{bundle.seed_count})",
        "dominant_frustration": _short(theme_analysis.dominant_frustration),
        "dominant_desire": _short(theme_analysis.dominant_desire),
        "top_ai_opportunity": _short(feasibility.top_ai_opportunity),
        "biggest_gap": _short(gap_analysis.biggest_gap),
        "quick_win": _short(gap_analysis.quick_win),
        "themes": [
            {"name": t.theme_name, "freq": t.frequency_estimate, "segment": t.user_segment,
             "desc": _short(t.description), "quote": t.example_quotes[0] if t.example_quotes else ""}
            for t in theme_analysis.themes
        ],
        "feasibility": [
            {"theme": r.theme_name, "score": r.feasibility_score, "approach": _short(r.ai_approach, 100),
             "complexity": r.technical_complexity, "comparable": r.comparable_product}
            for r in feasibility.ratings
        ],
        "gaps": [
            {"theme": g.theme_name, "type": g.gap_type, "gap": _short(g.gap_description, 150)}
            for g in gap_analysis.gaps
        ],
        "strengths": [
            {"feature": s.feature_name, "why": _short(s.why_users_love_it, 100)}
            for s in gap_analysis.strengths
        ],
        "top_quotes": [
            {"text": _short(q.quote_text, 200), "source": q.source, "theme": q.theme}
            for q in quote_analysis.quotes[:10]
        ],
    }

    user_msg = (
        f"Four research agents analyzed {company.company_name} user feedback. "
        f"Synthesize into a Feature Opportunity Brief.\n\n"
        f"Write all 6 sections (write_brief_section), then rank_opportunities, "
        f"then finalize_brief.\n\nRESEARCH SUMMARY:\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}"
    )

    sections: List[BriefSection] = []
    ranked_opportunities: List[RankedOpportunity] = []
    final_data: Dict = {}
    messages = [{"role": "user", "content": user_msg}]
    iterations = 0

    while iterations < 15:
        iterations += 1
        t0 = time.perf_counter()

        async with client.messages.stream(
            model=MODEL,
            max_tokens=8000,
            thinking={"type": "enabled", "budget_tokens": 5000},
            system=orchestrator_system(company),
            tools=ORCHESTRATOR_TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        ) as stream:
            async for event in stream:
                delta = getattr(event, "delta", None)
                if delta is None:
                    continue
                delta_type = getattr(delta, "type", "")
                if delta_type == "thinking_delta":
                    thinking_text = getattr(delta, "thinking", "")
                    if thinking_text:
                        await event_queue.put(SSEEvent(
                            type="thinking_delta", agent="Orchestrator", text=thinking_text,
                        ))
                elif delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        await event_queue.put(SSEEvent(
                            type="agent_stream", agent="Orchestrator", text=text,
                        ))

            final_message = await stream.get_final_message()

        duration_ms = (time.perf_counter() - t0) * 1000
        if cost_tracker:
            cost_tracker.record("Orchestrator", final_message, MODEL, duration_ms)

        messages.append({"role": "assistant", "content": final_message.content})

        if final_message.stop_reason == "end_turn":
            break

        tool_results = []
        for block in final_message.content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue

            if block.name == "write_brief_section":
                inp = block.input
                sq = inp.get("supporting_quotes", [])
                if isinstance(sq, str):
                    try:
                        sq = json.loads(sq)
                    except Exception:
                        sq = [sq] if sq else []
                sections.append(BriefSection(
                    section_id=inp["section_id"], title=inp["title"],
                    content=inp["content"],
                    supporting_quotes=sq if isinstance(sq, list) else [],
                ))
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="write_brief_section",
                    tool_input={"section_id": inp["section_id"], "title": inp["title"]},
                ))
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Section '{inp['section_id']}' recorded.",
                })

            elif block.name == "rank_opportunities":
                for opp in block.input.get("opportunities", []):
                    try:
                        if isinstance(opp, str):
                            opp = json.loads(opp)
                        if isinstance(opp, dict):
                            ranked_opportunities.append(RankedOpportunity(**opp))
                    except Exception as e:
                        logger.warning(f"Skipping malformed opportunity: {e}")
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="rank_opportunities",
                    tool_input={"count": len(block.input.get("opportunities", []))},
                ))
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": "Opportunities ranked.",
                })

            elif block.name == "finalize_brief":
                final_data = block.input
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="finalize_brief", tool_input={},
                ))
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": "Brief finalized.",
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    full_md = _assemble_markdown(
        sections, ranked_opportunities, final_data,
        bundle, theme_analysis, gap_analysis, company,
    )

    brief = FeatureBrief(
        title=f"{company.company_name} AI Feature Opportunity Brief",
        company_name=company.company_name,
        headline_insight=final_data.get("headline_insight", "") or "",
        top_3_insights=final_data.get("top_3_insights", []) if isinstance(final_data.get("top_3_insights"), list) else [],
        ranked_opportunities=ranked_opportunities,
        sections=sections,
        call_to_action=final_data.get("call_to_action", ""),
        full_markdown=full_md,
        reddit_count=bundle.reddit_count,
        app_store_count=bundle.app_store_count,
        seed_count=bundle.seed_count,
        themes=theme_analysis.themes,
        power_quotes=quote_analysis.quotes,
    )

    await event_queue.put(SSEEvent(type="agent_complete", agent="Orchestrator"))
    await event_queue.put(SSEEvent(
        type="brief_ready",
        data=brief.model_dump(mode="json"),
    ))

    return brief


# ===========================================================================
# Full pipeline entry point — now with RAG + cost tracking
# ===========================================================================

async def run_pipeline(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
) -> FeatureBrief:
    """
    Main pipeline entry point.

    New in v2: builds a FeedbackVectorStore from the bundle, creates an
    AgentRetriever, and passes it to all agents for RAG-based retrieval.
    Tracks costs per agent via CostTracker.
    """

    cost_tracker = CostTracker()
    retrieval_metrics = RetrievalMetrics()

    # ── Build vector store (RAG) ──────────────────────────────────────
    await event_queue.put(SSEEvent(
        type="progress",
        message=f"Building vector index over {bundle.total} feedback items...",
    ))

    try:
        vector_store = build_vector_store(bundle)
        retriever = AgentRetriever(vector_store, retrieval_metrics)
        await event_queue.put(SSEEvent(
            type="progress",
            message=f"Vector index ready: {vector_store.size} items, "
                    f"{vector_store.vectors.shape[1]} features "
                    f"({vector_store._index_time_ms:.0f}ms)",
        ))
    except Exception as e:
        logger.warning(f"RAG indexing failed, falling back to raw: {e}")
        retriever = None

    # ── Wave 1: ThemeAgent + QuoteAgent ───────────────────────────────
    await event_queue.put(SSEEvent(
        type="phase_start", phase="wave1",
        message=f"Launching Wave 1 agents on {bundle.total} {company.company_name} feedback items...",
    ))

    theme_analysis, quote_analysis = await asyncio.gather(
        run_theme_agent(client, bundle, company, event_queue, retriever, cost_tracker),
        run_quote_agent(client, bundle, company, event_queue, retriever, cost_tracker),
    )

    await event_queue.put(SSEEvent(
        type="phase_complete", phase="wave1",
        message=f"Wave 1 complete: {len(theme_analysis.themes)} themes, "
                f"{len(quote_analysis.quotes)} quotes. Launching Wave 2...",
        data={"theme_count": len(theme_analysis.themes),
              "quote_count": len(quote_analysis.quotes)},
    ))

    # ── Wave 2: FeasibilityAgent + GapAgent ───────────────────────────
    await event_queue.put(SSEEvent(
        type="phase_start", phase="wave2",
        message="Running AI Feasibility & Gap Analysis in parallel...",
    ))

    feasibility, gap_analysis = await asyncio.gather(
        run_feasibility_agent(
            client, bundle, theme_analysis.themes, company, event_queue,
            retriever, cost_tracker,
        ),
        run_gap_agent(
            client, theme_analysis.themes, company, event_queue, cost_tracker,
        ),
    )

    await event_queue.put(SSEEvent(
        type="phase_complete", phase="wave2",
        message="All agents complete. Starting orchestration...",
        data={
            "themes": len(theme_analysis.themes),
            "feasibility_ratings": len(feasibility.ratings),
            "gaps": len(gap_analysis.gaps),
            "quotes": len(quote_analysis.quotes),
        },
    ))

    # ── Orchestrator synthesis ────────────────────────────────────────
    brief = await run_orchestrator(
        client, bundle, company, theme_analysis, feasibility,
        gap_analysis, quote_analysis, event_queue, cost_tracker,
    )

    # ── Emit cost + retrieval metrics via SSE ─────────────────────────
    cost_summary = cost_tracker.summary()
    retrieval_summary = retrieval_metrics.summary()

    await event_queue.put(SSEEvent(
        type="pipeline_metrics",
        data={
            "cost": cost_summary,
            "retrieval": retrieval_summary,
        },
    ))

    # Attach metrics to brief for eval framework
    brief._cost_summary = cost_summary
    brief._retrieval_metrics = retrieval_metrics

    return brief


# ===========================================================================
# Legacy fallback (when RAG is not available)
# ===========================================================================

def _feedback_to_text(bundle: FeedbackBundle, max_items: int = 120) -> str:
    """Serialize feedback items into a compact string for Claude context."""
    lines = []
    sorted_items = sorted(bundle.items, key=lambda x: x.upvotes, reverse=True)
    for item in sorted_items[:max_items]:
        prefix = f"[{item.source.upper()}" + (f" +{item.upvotes}" if item.upvotes else "") + "] "
        lines.append(prefix + item.text)
    return "\n\n".join(lines)


# ===========================================================================
# Markdown assembly
# ===========================================================================

def _assemble_markdown(
    sections: List[BriefSection],
    opportunities: List[RankedOpportunity],
    final_data: Dict,
    bundle: FeedbackBundle,
    theme_analysis: ThemeAnalysis,
    gap_analysis: GapAnalysis,
    company: CompanyContext = None,
) -> str:
    title = f"{company.company_name} AI Feature Opportunity Brief" if company else "AI Feature Opportunity Brief"
    lines = [
        f"# {title}",
        f"> *Generated {datetime.utcnow().strftime('%B %d, %Y')} · "
        f"{bundle.total} feedback items analyzed "
        f"({bundle.reddit_count} Reddit · {bundle.app_store_count} App Store · {bundle.seed_count} seed)*",
        "",
    ]

    if final_data.get("headline_insight"):
        lines += ["## Headline Finding", "", f"> {final_data['headline_insight']}", ""]

    if final_data.get("top_3_insights"):
        lines += ["## Top 3 Insights", ""]
        for i, ins in enumerate(final_data["top_3_insights"], 1):
            lines.append(f"**{i}.** {ins}")
        lines.append("")

    if opportunities:
        lines += ["## Ranked AI Feature Opportunities", ""]
        for opp in opportunities:
            lines += [
                f"### #{opp.rank} — {opp.feature_name}",
                f"*{opp.one_liner}*", "",
                f"- **Evidence:** {opp.evidence_strength}",
                f"- **AI uniqueness:** {opp.ai_uniqueness}",
                f"- **Effort:** {opp.effort_estimate.replace('_', ' ')}", "",
                f"> \"{opp.supporting_quote}\"", "",
            ]

    for section in sections:
        lines += [f"## {section.title}", "", section.content, ""]
        if section.supporting_quotes:
            for q in section.supporting_quotes[:2]:
                lines.append(f"> \"{q}\"")
            lines.append("")

    if final_data.get("call_to_action"):
        lines += ["## Recommended Next Step", "", final_data["call_to_action"], ""]

    return "\n".join(lines)

AGENTSEOF

echo "📝 Replacing backend/main.py..."
cat > backend/main.py << 'MAINEOF'
"""
FastAPI backend — SSE streaming, pipeline orchestration, eval endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents import run_pipeline
from backend.evals import EvalSuite
from backend.models import CompanyContext, FeatureBrief, FeedbackBundle, SSEEvent

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Feature Intelligence", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

BRIEFS_DIR = Path(__file__).parent.parent / "briefs"
BRIEFS_DIR.mkdir(exist_ok=True)

# In-memory run store
runs: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    company_name: str = "Duolingo"
    product_description: str = "language learning app"
    subreddit: str = ""
    app_store_id: str = ""
    known_features: str = ""


class AnalyzeResponse(BaseModel):
    run_id: str
    company_name: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    path = FRONTEND_DIR / "index.html"
    if path.exists():
        return FileResponse(str(path))
    return {"message": "Feature Intelligence API — set ANTHROPIC_API_KEY and open /docs"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(request: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    company = CompanyContext(
        company_name=request.company_name.strip() or "Duolingo",
        product_description=request.product_description.strip() or "consumer app",
        subreddit=request.subreddit.strip().lstrip("r/"),
        app_store_id=request.app_store_id.strip(),
        known_features=request.known_features.strip(),
    )

    run_id = str(uuid.uuid4())
    queue: asyncio.Queue[Optional[SSEEvent]] = asyncio.Queue()

    runs[run_id] = {
        "queue": queue,
        "brief": None,
        "bundle": None,
        "status": "starting",
        "company": company,
    }

    asyncio.create_task(_pipeline_task(run_id, api_key, company))
    return AnalyzeResponse(run_id=run_id, company_name=company.company_name)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    queue = runs[run_id]["queue"]

    async def generator():
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"
        while True:
            try:
                event: Optional[SSEEvent] = await asyncio.wait_for(
                    queue.get(), timeout=120.0
                )
            except asyncio.TimeoutError:
                yield "data: " + json.dumps({"type": "keepalive"}) + "\n\n"
                continue

            if event is None:
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                break

            yield "data: " + event.model_dump_json() + "\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/brief/{run_id}")
async def get_brief(run_id: str):
    run = runs.get(run_id)
    if run and run.get("brief"):
        return run["brief"].model_dump(mode="json")
    brief_path = BRIEFS_DIR / f"{run_id}.json"
    if brief_path.exists():
        return json.loads(brief_path.read_text())
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    raise HTTPException(status_code=425, detail="Brief not ready yet")


@app.get("/brief/{run_id}/markdown")
async def get_brief_markdown(run_id: str):
    run = runs.get(run_id)
    if not run or not run.get("brief"):
        raise HTTPException(status_code=425, detail="Brief not ready yet")
    return {"markdown": run["brief"].full_markdown}


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": run["status"]}


# ---------------------------------------------------------------------------
# Cost endpoint
# ---------------------------------------------------------------------------

@app.get("/cost/{run_id}")
async def get_cost(run_id: str):
    """Return API cost breakdown for a completed run."""
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    brief = run.get("brief")
    if not brief:
        raise HTTPException(status_code=425, detail="Analysis not complete yet")

    cost_summary = getattr(brief, "_cost_summary", None)
    if not cost_summary:
        raise HTTPException(status_code=404, detail="Cost data not available")

    return cost_summary


# ---------------------------------------------------------------------------
# Eval endpoint
# ---------------------------------------------------------------------------

@app.post("/eval/{run_id}")
async def run_eval(run_id: str, stability: bool = False):
    """
    Run the evaluation suite on a completed analysis.

    Args:
        stability: If true, runs the slow theme stability eval (3x theme agent).
    """
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    brief = run.get("brief")
    bundle = run.get("bundle")
    company = run.get("company")
    if not brief or not bundle:
        raise HTTPException(status_code=425, detail="Analysis not complete yet")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    suite = EvalSuite(client)

    retrieval_metrics = getattr(brief, "_retrieval_metrics", None)

    result = await suite.run(
        brief=brief,
        bundle=bundle,
        company=company,
        retrieval_metrics=retrieval_metrics,
        run_id=run_id,
        run_stability_eval=stability,
    )

    return result.model_dump(mode="json")


@app.get("/eval/history")
async def eval_history(company: str = "", limit: int = 20):
    """Return historical eval results for trend analysis."""
    results = EvalSuite.load_history(company, limit)
    return [r.model_dump(mode="json") for r in results]


# ---------------------------------------------------------------------------
# Background pipeline task
# ---------------------------------------------------------------------------

async def _pipeline_task(run_id: str, api_key: str, company: CompanyContext):
    queue = runs[run_id]["queue"]
    client = anthropic.AsyncAnthropic(api_key=api_key)

    try:
        # Phase 1: Fetch data
        runs[run_id]["status"] = "fetching"
        await queue.put(SSEEvent(
            type="phase_start", phase="fetching",
            message=f"Fetching {company.company_name} feedback from Reddit and App Store...",
        ))

        from backend.scraper import fetch_bundle

        async def progress_cb(msg: str):
            await queue.put(SSEEvent(type="progress", message=msg))

        bundle: FeedbackBundle = await fetch_bundle(company=company, progress_cb=progress_cb)

        # Store bundle for eval endpoint
        runs[run_id]["bundle"] = bundle

        await queue.put(SSEEvent(
            type="phase_complete", phase="fetching",
            message=f"Loaded {bundle.total} {company.company_name} feedback items",
            data={
                "total": bundle.total,
                "reddit": bundle.reddit_count,
                "app_store": bundle.app_store_count,
                "google_play": bundle.google_play_count,
                "hacker_news": bundle.hacker_news_count,
                "seed": bundle.seed_count,
            },
        ))

        # Phase 2: Agent pipeline (now with RAG + cost tracking)
        runs[run_id]["status"] = "analyzing"
        brief = await run_pipeline(client, bundle, company, queue)
        runs[run_id]["brief"] = brief
        runs[run_id]["status"] = "complete"

        # Persist to disk
        try:
            brief_path = BRIEFS_DIR / f"{run_id}.json"
            brief_path.write_text(brief.model_dump_json())
        except Exception as e:
            logger.warning(f"Failed to persist brief {run_id}: {e}")

        await queue.put(SSEEvent(
            type="phase_complete", phase="pipeline",
            message="Analysis complete.",
        ))

    except Exception as e:
        logger.exception(f"Pipeline failed for run {run_id}")
        runs[run_id]["status"] = "error"
        await queue.put(SSEEvent(type="error", error=str(e)))
    finally:
        await queue.put(None)

MAINEOF

echo "📝 Replacing backend/models.py..."
cat > backend/models.py << 'MODELSEOF'
"""
Pydantic v2 models for Product Feature Intelligence.
All data flowing between pipeline stages is typed through these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CompanyContext(BaseModel):
    company_name: str
    product_description: str
    subreddit: str = ""
    app_store_id: str = ""
    known_features: str = ""


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


class Theme(BaseModel):
    theme_name: str
    description: str
    example_quotes: List[str]
    frequency_estimate: Literal["very_high", "high", "medium", "low"]
    user_segment: str


class ThemeAnalysis(BaseModel):
    themes: List[Theme]
    total_feedback_items: int
    dominant_frustration: str
    dominant_desire: str


class AIFeasibility(BaseModel):
    theme_name: str
    feasibility_score: float = Field(ge=0.0, le=1.0)
    ai_approach: str
    why_ai_uniquely_suited: str
    technical_complexity: Literal["low", "medium", "high"]
    comparable_product: str


class FeasibilityAnalysis(BaseModel):
    ratings: List[AIFeasibility]
    top_ai_opportunity: str
    hardest_to_solve: str


class GapFinding(BaseModel):
    theme_name: str
    gap_type: Literal["missing_entirely", "partial_solution", "poor_execution"]
    existing_feature: str = ""
    gap_description: str
    market_evidence: str = ""
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


class PowerQuote(BaseModel):
    quote_text: str
    source: Literal["reddit", "app_store", "seed"]
    theme: str
    why_compelling: str


class QuoteAnalysis(BaseModel):
    quotes: List[PowerQuote]
    most_compelling_quote: str
    most_viral_potential_quote: str


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
    content: str
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
    reddit_count: int = 0
    app_store_count: int = 0
    seed_count: int = 0
    themes: List["Theme"] = Field(default_factory=list)
    power_quotes: List["PowerQuote"] = Field(default_factory=list)


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


class ThinkingBlock(BaseModel):
    thinking_text: str
    token_count: int

MODELSEOF

echo "📝 Replacing backend/prompts.py..."
cat > backend/prompts.py << 'PROMPTSEOF'
"""Agent system prompts — all company-agnostic."""

from __future__ import annotations
from backend.models import CompanyContext


def theme_agent_system(company: CompanyContext) -> str:
    return f"""You are a senior product researcher specializing in consumer apps and digital products.
You are analyzing a batch of real user feedback — Reddit posts and App Store reviews — about {company.company_name} ({company.product_description}).

Your job is to identify the distinct unmet need clusters (themes) that users are expressing.
A theme is a recurring, specific user need — not a vague category.

Good theme: "Users want real-time AI pronunciation coaching that identifies which specific sounds they got wrong"
Bad theme: "Users want better features" (too vague)

For each theme:
- Name it precisely as a product opportunity, not a complaint
- Include 2-3 direct quotes that best represent it
- Estimate how frequently it appears (very_high / high / medium / low)
- Identify which user segment feels this most acutely

Aim for 8–12 distinct themes. Avoid overlap between themes.
Use record_theme for each theme, then call emit_theme_summary once."""


def feasibility_agent_system(company: CompanyContext) -> str:
    return f"""You are an AI product strategist with deep knowledge of large language models,
speech AI, and what's technically achievable in consumer AI products in 2024–2025.

You will receive a list of user need themes from {company.company_name} ({company.product_description}) feedback.
For each theme, evaluate how well AI (specifically LLMs, speech AI, or multimodal models) could address it.

Key questions to answer per theme:
1. What specific AI approach would address this?
2. Why is AI uniquely suited here vs. human support or static content?
3. How technically complex is building this?
4. What product already does something similar?

Be honest — some themes are better served by content work or UX improvements than AI. Flag those.
Use rate_ai_feasibility for each theme, then emit_feasibility_summary."""


def gap_agent_system(company: CompanyContext) -> str:
    known_features_block = ""
    if company.known_features.strip():
        known_features_block = f"""
The user has provided {company.company_name}'s current feature list:
---
{company.known_features.strip()}
---
Use this as your primary source of truth for what already exists.
"""
    else:
        known_features_block = f"""
Use your knowledge of {company.company_name}'s current product and feature set to assess gaps.
Be specific and accurate — if you're uncertain about a feature's existence, say so.
"""

    return f"""You are a product analyst who knows {company.company_name}'s current product deeply.
{company.company_name} is a {company.product_description}.
{known_features_block}
You will receive a list of user need themes. For each:
- Is there already a {company.company_name} feature addressing it?
- If partial — what exactly is missing from the existing solution?
- What's the market evidence that this gap is real?

Also identify 2-3 things {company.company_name} is genuinely doing well that users love.

Confidence rules (CRITICAL):
- Use confidence="high" ONLY when you are certain
- Use confidence="medium" when you believe a gap exists but have some uncertainty
- Use confidence="low" when users complain but you're unsure whether the company already addresses it
- Default to gap_type="partial_solution" when uncertain

Use record_gap for each gap, record_strength for each strength, then emit_gap_summary."""


def quote_agent_system(company: CompanyContext) -> str:
    return f"""You are a research analyst building the evidence layer for a product strategy brief.

You will receive raw user feedback about {company.company_name} ({company.product_description}).
Your job: find the 15–20 most powerful, specific, quotable statements that would make a compelling case
to a product team for building new AI features.

A powerful quote is:
- Specific (mentions a concrete need, not a vague complaint)
- Emotionally resonant (reveals genuine frustration or desire)
- Actionable (implies a clear product direction)
- Authentic (sounds like a real user, not a feature request)

For each quote, explain why it's compelling and which theme it supports.
Use record_power_quote for each, then emit_quote_summary."""


def orchestrator_system(company: CompanyContext) -> str:
    return f"""You are a Principal Product Manager writing a Feature Opportunity Brief for {company.company_name}'s
AI product strategy team. You have received outputs from four parallel research agents:
1. ThemeAgent — identified user need clusters from real feedback
2. AIFeasibilityAgent — evaluated AI solvability per theme
3. GapAgent — cross-referenced against existing features
4. QuoteAgent — curated the strongest user evidence

Principles:
- Ruthlessly prioritize. The #1 opportunity should be undeniable.
- Ground every claim in user evidence
- Be specific about what the feature IS
- Acknowledge what {company.company_name} does well
- Every section should read as if written by someone who deeply understands the business

Sections to write (in order):
1. executive_summary
2. market_signal
3. top_opportunities
4. competitive_context
5. recommended_next_feature
6. methodology

After writing sections, call rank_opportunities with the final ordered list.
End with finalize_brief."""

PROMPTSEOF

echo "📝 Updating requirements.txt..."
cat > requirements.txt << 'REQEOF'
anthropic>=0.40.0
pydantic>=2.5.0
fastapi>=0.109.0
uvicorn>=0.27.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
google-play-scraper>=1.2.4
scikit-learn>=1.4.0
numpy>=1.26.0

REQEOF


echo "📦 Creating evals directory..."
mkdir -p evals

echo ""
echo "✅ Upgrade complete!"
echo ""
echo "Files created:"
echo "  + backend/rag.py          (RAG retrieval layer)"
echo "  + backend/evals.py        (evaluation framework)"
echo "  + backend/cost_tracker.py (API cost tracking)"
echo ""
echo "Files replaced:"
echo "  ~ backend/agents.py       (RAG + cost tracking integrated)"
echo "  ~ backend/main.py         (eval + cost endpoints added)"
echo "  ~ backend/models.py       (dead model classes removed)"
echo "  ~ backend/prompts.py      (cleaned up)"
echo "  ~ requirements.txt        (scikit-learn + numpy added)"
echo ""
echo "Files deleted:"
echo "  - backend/graph.py        (dead code)"
echo "  - backend/vision.py       (dead code)"
echo ""
echo "Backups saved in: _backup/"
echo ""
echo "Next steps:"
echo "  1. pip install -r requirements.txt"
echo "  2. git add -A"
echo "  3. git commit -m 'feat: add RAG retrieval, eval framework, cost tracking'"
echo "  4. git push"
