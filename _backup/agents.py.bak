"""
Multi-agent pipeline for Product Feature Intelligence.

Phase 1 — 4 specialist agents run IN PARALLEL via asyncio.gather():
    ThemeAgent, AIFeasibilityAgent, GapAgent, QuoteAgent

Phase 2 — Orchestrator with extended thinking synthesizes all 4 outputs
           into a Feature Opportunity Brief.

All progress is streamed to the caller via an async queue (SSE events).
Works for any company — CompanyContext is threaded through all stages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import anthropic

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

logger = logging.getLogger(__name__)
MODEL = "claude-sonnet-4-6"

# Limit concurrent Anthropic API calls to avoid rate-limit errors
_api_semaphore = asyncio.Semaphore(2)


# ===========================================================================
# Tool definitions
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
                    "description": "Which user segment feels this most acutely (e.g. 'intermediate learners', 'Max subscribers')",
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
                "dominant_frustration": {"type": "string", "description": "The #1 pain point across all feedback"},
                "dominant_desire": {"type": "string", "description": "The #1 thing users most want"},
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
                "ai_approach": {
                    "type": "string",
                    "description": "Specific AI technique (e.g. 'conversational LLM with user history RAG')",
                },
                "why_ai_uniquely_suited": {
                    "type": "string",
                    "description": "Why AI specifically, vs. content work or human tutors?",
                },
                "technical_complexity": {
                    "type": "string", "enum": ["low", "medium", "high"],
                },
                "comparable_product": {
                    "type": "string",
                    "description": "Product that already does something similar (competitor or adjacent)",
                },
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
        "description": "Record a gap between what users need and what Duolingo currently offers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "theme_name": {"type": "string"},
                "gap_type": {
                    "type": "string",
                    "enum": ["missing_entirely", "partial_solution", "poor_execution"],
                },
                "existing_feature": {
                    "type": "string",
                    "description": "Name the closest existing product feature that addresses this, or empty string if none",
                },
                "gap_description": {
                    "type": "string",
                    "description": "Specifically what is missing from the current solution",
                },
                "market_evidence": {
                    "type": "string",
                    "description": "Competitor or precedent that proves this gap is real and solvable",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How confident are you this is a real gap? 'high' = you are certain the feature doesn't exist or is severely lacking; 'medium' = likely a gap but uncertain; 'low' = possible gap, unverified",
                },
                "knowledge_basis": {
                    "type": "string",
                    "enum": ["user_provided_features", "llm_knowledge", "user_feedback_only"],
                    "description": "What is your knowledge of this gap based on? 'user_provided_features' = user pasted a feature list; 'llm_knowledge' = your training data; 'user_feedback_only' = users complain but you don't know the current feature set",
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
                "quote": {"type": "string", "description": "A user quote expressing love for this feature"},
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
                "biggest_gap": {"type": "string", "description": "The most significant unaddressed gap"},
                "quick_win": {"type": "string", "description": "The gap that could be closed fastest"},
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
                "theme": {"type": "string", "description": "Which theme does this quote represent?"},
                "why_compelling": {
                    "type": "string",
                    "description": "Why is this quote particularly powerful for a product team?",
                },
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
                "content": {
                    "type": "string",
                    "description": "Polished markdown prose for this section (200-500 words)",
                },
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
                            "one_liner": {"type": "string", "description": "One sentence: what this feature does"},
                            "evidence_strength": {"type": "string", "enum": ["strong", "moderate", "emerging"]},
                            "ai_uniqueness": {"type": "string", "enum": ["high", "medium", "low"]},
                            "effort_estimate": {"type": "string", "enum": ["quick_win", "medium_lift", "major_investment"]},
                            "supporting_quote": {"type": "string", "description": "Best user quote for this opportunity"},
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
        "description": "Called once all sections and rankings are done. Provide the headline and call to action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline_insight": {
                    "type": "string",
                    "description": "The single most important finding — 1-2 sentences that would open a board presentation",
                },
                "top_3_insights": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 3, "maxItems": 3,
                },
                "call_to_action": {
                    "type": "string",
                    "description": "What should the product team do first, concretely?",
                },
            },
            "required": ["headline_insight", "top_3_insights", "call_to_action"],
        },
    },
]


# ===========================================================================
# Core agentic loop (reused pattern)
# ===========================================================================

async def _run_agent_loop(
    client: anthropic.AsyncAnthropic,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    event_queue: asyncio.Queue,
    max_iterations: int = 20,
    on_tool_call: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Generic agentic tool-use loop.
    Returns dict of {tool_name: [list of inputs collected]}.
    Emits SSE events throughout.
    on_tool_call: optional async callable(tool_name, tool_input) for custom side-effects.
    """
    messages = [{"role": "user", "content": user_message}]
    collected: Dict[str, list] = defaultdict(list)
    iterations = 0

    await event_queue.put(SSEEvent(type="agent_start", agent=agent_name))

    while iterations < max_iterations:
        iterations += 1

        async with _api_semaphore:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                tool_choice={"type": "auto"},
                messages=messages,
            )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                collected[block.name].append(block.input)
                await event_queue.put(SSEEvent(
                    type="agent_tool_call",
                    agent=agent_name,
                    tool=block.name,
                    tool_input=block.input,
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
# Specialist agents
# ===========================================================================

def _feedback_to_text(bundle: FeedbackBundle, max_items: int = 120) -> str:
    """Serialize feedback items into a compact string for Claude context."""
    lines = []
    # Sort by upvotes descending so most-voted appear first
    sorted_items = sorted(bundle.items, key=lambda x: x.upvotes, reverse=True)
    for item in sorted_items[:max_items]:
        prefix = f"[{item.source.upper()}" + (f" +{item.upvotes}" if item.upvotes else "") + "] "
        lines.append(prefix + item.text)
    return "\n\n".join(lines)


async def run_theme_agent(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
) -> ThemeAnalysis:
    feedback_text = _feedback_to_text(bundle, max_items=120)

    user_msg = f"""Analyze this batch of {bundle.total} user feedback items about {company.company_name} ({company.product_description}).
Identify 8–12 distinct unmet need themes. For each, call record_theme.
End with emit_theme_summary.

FEEDBACK:
{feedback_text}"""

    async def _on_tool(tool_name: str, tool_input: dict) -> None:
        if tool_name == "record_theme":
            await event_queue.put(SSEEvent(
                type="theme_discovered",
                agent="ThemeAgent",
                data=tool_input,
            ))

    raw = await _run_agent_loop(
        client, "ThemeAgent", theme_agent_system(company), user_msg, THEME_TOOLS, event_queue,
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
) -> FeasibilityAnalysis:
    themes_text = json.dumps(
        [{"theme_name": t.theme_name, "description": t.description,
          "frequency": t.frequency_estimate, "segment": t.user_segment}
         for t in themes],
        indent=2,
    )

    sample = _feedback_to_text(bundle, max_items=40)

    user_msg = f"""Evaluate AI feasibility for each of the following {len(themes)} user need themes from {company.company_name} feedback.
Call rate_ai_feasibility for each theme, then emit_feasibility_summary.

THEMES TO EVALUATE:
{themes_text}

SAMPLE FEEDBACK FOR CONTEXT:
{sample}"""

    raw = await _run_agent_loop(
        client, "AIFeasibilityAgent", feasibility_agent_system(company), user_msg, FEASIBILITY_TOOLS, event_queue
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
) -> GapAnalysis:
    themes_text = json.dumps(
        [{"theme_name": t.theme_name, "description": t.description,
          "example_quotes": t.example_quotes[:2]}
         for t in themes],
        indent=2,
    )

    user_msg = f"""Cross-reference these {len(themes)} user need themes against {company.company_name}'s current feature set.
For each theme, call record_gap. Also call record_strength for 2-3 things {company.company_name} does well.
End with emit_gap_summary.

USER NEED THEMES:
{themes_text}"""

    raw = await _run_agent_loop(
        client, "GapAgent", gap_agent_system(company), user_msg, GAP_TOOLS, event_queue
    )

    gaps = [
        GapFinding(
            theme_name=g["theme_name"],
            gap_type=g["gap_type"],
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
        gaps=gaps,
        strengths=strengths,
        biggest_gap=summary.get("biggest_gap", ""),
        quick_win=summary.get("quick_win", ""),
    )


async def run_quote_agent(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
) -> QuoteAnalysis:
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

    user_msg = f"""Mine these {company.company_name} user feedback items for the 15–20 most compelling, quotable statements
for a product strategy brief. Call record_power_quote for each, then emit_quote_summary.

FEEDBACK:
{feedback_text}"""

    raw = await _run_agent_loop(
        client, "QuoteAgent", quote_agent_system(company), user_msg, QUOTE_TOOLS, event_queue
    )

    quotes = [PowerQuote(**q) for q in raw.get("record_power_quote", [])]
    summary = (raw.get("emit_quote_summary") or [{}])[-1]

    return QuoteAnalysis(
        quotes=quotes,
        most_compelling_quote=summary.get("most_compelling_quote", ""),
        most_viral_potential_quote=summary.get("most_viral_potential_quote", ""),
    )


# ===========================================================================
# Orchestrator — extended thinking synthesis
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
) -> FeatureBrief:

    await event_queue.put(SSEEvent(
        type="agent_start", agent="Orchestrator",
    ))
    await event_queue.put(SSEEvent(
        type="phase_start", phase="synthesis",
        message="Orchestrator synthesizing with extended thinking...",
    ))

    # Build a compact context to stay well under rate limits (~8k input tokens max)
    # Trim descriptions to 1 sentence, limit quotes, cap power_quotes at 10
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
            {
                "name": t.theme_name,
                "freq": t.frequency_estimate,
                "segment": t.user_segment,
                "desc": _short(t.description),
                "quote": t.example_quotes[0] if t.example_quotes else "",
            }
            for t in theme_analysis.themes
        ],
        "feasibility": [
            {
                "theme": r.theme_name,
                "score": r.feasibility_score,
                "approach": _short(r.ai_approach, 100),
                "complexity": r.technical_complexity,
                "comparable": r.comparable_product,
            }
            for r in feasibility.ratings
        ],
        "gaps": [
            {
                "theme": g.theme_name,
                "type": g.gap_type,
                "gap": _short(g.gap_description, 150),
            }
            for g in gap_analysis.gaps
        ],
        "strengths": [
            {"feature": s.feature_name, "why": _short(s.why_users_love_it, 100)}
            for s in gap_analysis.strengths
        ],
        "top_quotes": [
            {"text": _short(q.quote_text, 200), "source": q.source, "theme": q.theme}
            for q in quote_analysis.quotes[:10]  # cap at 10 most compelling
        ],
    }

    user_msg = f"""Four research agents analyzed {company.company_name} user feedback. Synthesize into a Feature Opportunity Brief.

Write all 6 sections (write_brief_section), then rank_opportunities, then finalize_brief.

RESEARCH SUMMARY:
{json.dumps(context, indent=2, ensure_ascii=False)}"""

    sections: List[BriefSection] = []
    ranked_opportunities: List[RankedOpportunity] = []
    final_data: Dict = {}
    messages = [{"role": "user", "content": user_msg}]
    iterations = 0

    while iterations < 15:
        iterations += 1

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
                            type="thinking_delta",
                            agent="Orchestrator",
                            text=thinking_text,
                        ))
                elif delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        await event_queue.put(SSEEvent(
                            type="agent_stream", agent="Orchestrator", text=text,
                        ))

            final_message = await stream.get_final_message()

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
                        import json as _json; sq = _json.loads(sq)
                    except Exception:
                        sq = [sq] if sq else []
                sections.append(BriefSection(
                    section_id=inp["section_id"],
                    title=inp["title"],
                    content=inp["content"],
                    supporting_quotes=sq if isinstance(sq, list) else [],
                ))
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="write_brief_section",
                    tool_input={"section_id": inp["section_id"], "title": inp["title"]},
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Section '{inp['section_id']}' recorded.",
                })

            elif block.name == "rank_opportunities":
                for opp in block.input.get("opportunities", []):
                    try:
                        if isinstance(opp, str):
                            import json as _json
                            opp = _json.loads(opp)
                        if isinstance(opp, dict):
                            ranked_opportunities.append(RankedOpportunity(**opp))
                    except Exception as e:
                        logger.warning(f"Skipping malformed opportunity: {e} — {opp!r}")
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="rank_opportunities",
                    tool_input={"count": len(block.input.get("opportunities", []))},
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Opportunities ranked.",
                })

            elif block.name == "finalize_brief":
                final_data = block.input
                await event_queue.put(SSEEvent(
                    type="agent_tool_call", agent="Orchestrator",
                    tool="finalize_brief", tool_input={},
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Brief finalized.",
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    full_md = _assemble_markdown(
        sections, ranked_opportunities, final_data,
        bundle, theme_analysis, gap_analysis
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
# Full pipeline entry point
# ===========================================================================

async def run_pipeline(
    client: anthropic.AsyncAnthropic,
    bundle: FeedbackBundle,
    company: CompanyContext,
    event_queue: asyncio.Queue,
) -> FeatureBrief:

    await event_queue.put(SSEEvent(
        type="phase_start", phase="wave1",
        message=f"Launching Wave 1 agents on {bundle.total} {company.company_name} feedback items...",
    ))

    # Wave 1: ThemeAgent + QuoteAgent run in parallel — neither depends on the other
    theme_analysis, quote_analysis = await asyncio.gather(
        run_theme_agent(client, bundle, company, event_queue),
        run_quote_agent(client, bundle, company, event_queue),
    )

    await event_queue.put(SSEEvent(
        type="phase_complete", phase="wave1",
        message=f"Wave 1 complete: {len(theme_analysis.themes)} themes, {len(quote_analysis.quotes)} quotes. Launching Wave 2...",
        data={"theme_count": len(theme_analysis.themes), "quote_count": len(quote_analysis.quotes)},
    ))

    await event_queue.put(SSEEvent(
        type="phase_start", phase="wave2",
        message="Running AI Feasibility & Gap Analysis in parallel...",
    ))

    # Wave 2: FeasibilityAgent + GapAgent — both require theme output from Wave 1
    feasibility, gap_analysis = await asyncio.gather(
        run_feasibility_agent(client, bundle, theme_analysis.themes, company, event_queue),
        run_gap_agent(client, theme_analysis.themes, company, event_queue),
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

    brief = await run_orchestrator(
        client, bundle, company, theme_analysis, feasibility, gap_analysis, quote_analysis, event_queue
    )

    return brief


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
) -> str:
    lines = [
        "# Duolingo AI Feature Opportunity Brief",
        f"> *Generated {datetime.utcnow().strftime('%B %d, %Y')} · "
        f"{bundle.total} feedback items analyzed "
        f"({bundle.reddit_count} Reddit · {bundle.app_store_count} App Store · {bundle.seed_count} seed)*",
        "",
    ]

    if final_data.get("headline_insight"):
        lines += [
            "## Headline Finding",
            "",
            f"> {final_data['headline_insight']}",
            "",
        ]

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
                f"*{opp.one_liner}*",
                "",
                f"- **Evidence:** {opp.evidence_strength}",
                f"- **AI uniqueness:** {opp.ai_uniqueness}",
                f"- **Effort:** {opp.effort_estimate.replace('_', ' ')}",
                "",
                f"> \"{opp.supporting_quote}\"",
                "",
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
