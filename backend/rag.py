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

