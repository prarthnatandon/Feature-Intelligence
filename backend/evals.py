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

