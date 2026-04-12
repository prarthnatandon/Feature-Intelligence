"""
NetworkX skill dependency graph builder.
Computes centrality metrics and exports D3-ready JSON for visualization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logging.warning("networkx not installed — graph metrics will be skipped")

from backend.models import DuolingoDataBundle, GraphLink, GraphNode, SkillGraph, SkillNode

logger = logging.getLogger(__name__)

# Category → D3 color group (integer, used by visualization.js for coloring)
CATEGORY_GROUP = {
    "core_vocabulary": 1,
    "thematic_vocabulary": 2,
    "grammar_introduction": 3,
    "grammar_practice": 4,
    "pronunciation_phonetics": 5,
    "cultural_context": 6,
    "review_consolidation": 7,
    "compound_mixed": 8,
}


def build_skill_graph(
    bundle: DuolingoDataBundle,
    retention_scores: Optional[Dict[str, float]] = None,
) -> SkillGraph:
    """
    Build a directed graph of skill dependencies.
    retention_scores: dict mapping skill_title → float 0–1 (from RetentionAgent output).
    Falls back to position-based score estimate if not provided.
    """
    skills = bundle.skills
    score_map = retention_scores or {}

    # Build lookup by id
    skill_by_id: Dict[str, SkillNode] = {s.id: s for s in skills}

    # --- NetworkX graph (if available) ---
    centrality: Dict[str, float] = {}
    if NX_AVAILABLE:
        G = nx.DiGraph()
        for skill in skills:
            G.add_node(skill.id, title=skill.title)
        for skill in skills:
            for dep_id in skill.dependencies:
                if dep_id in skill_by_id:
                    G.add_edge(dep_id, skill.id)  # dependency → skill

        # Betweenness centrality: identifies "gateway" skills
        try:
            bc = nx.betweenness_centrality(G, normalized=True)
            centrality = {skill_by_id[nid].title: v for nid, v in bc.items() if nid in skill_by_id}
            logger.info(f"Computed betweenness centrality for {len(centrality)} skills")
        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")

    # --- Build GraphNodes ---
    nodes: List[GraphNode] = []
    for skill in skills:
        title = skill.title
        ret_score = score_map.get(title, _position_based_score(skill))
        category = skill.category_hint

        nodes.append(GraphNode(
            id=skill.id,
            title=title,
            category=category,
            retention_score=round(ret_score, 3),
            vocab_load=len(skill.words),
            grammar_concepts=_infer_grammar_concepts(skill),
            coords_x=skill.coords_x,
            coords_y=skill.coords_y,
            group=CATEGORY_GROUP.get(category, 8),
        ))

    # --- Build GraphLinks ---
    links: List[GraphLink] = []
    for skill in skills:
        for dep_id in skill.dependencies:
            if dep_id in skill_by_id:
                links.append(GraphLink(
                    source=dep_id,
                    target=skill.id,
                    strength=1.0,
                ))

    # --- Metadata for D3 / frontend ---
    metadata: Dict[str, Any] = {
        "total_skills": len(skills),
        "total_links": len(links),
        "category_counts": _category_counts(skills),
        "gateway_skills": _top_gateway_skills(centrality, n=5),
        "centrality": centrality,
        "avg_retention_score": round(
            sum(n.retention_score for n in nodes) / len(nodes) if nodes else 0, 3
        ),
    }

    return SkillGraph(nodes=nodes, links=links, metadata=metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _position_based_score(skill: SkillNode) -> float:
    """
    Estimate a retention score when no agent data is available.
    Heuristic: skills at mid-tree depth tend to have higher retention
    because they combine known vocabulary with new grammar (optimal challenge).
    """
    depth = skill.coords_y
    category = skill.category_hint

    base = {
        "core_vocabulary": 0.62,
        "grammar_introduction": 0.71,
        "grammar_practice": 0.68,
        "thematic_vocabulary": 0.58,
        "pronunciation_phonetics": 0.64,
        "cultural_context": 0.55,
        "review_consolidation": 0.80,   # high: pure retrieval practice
        "compound_mixed": 0.66,
    }.get(category, 0.60)

    # Slight peak at mid-depth (rows 3–5)
    if 3 <= depth <= 5:
        base += 0.05
    elif depth >= 7:
        base -= 0.05

    return min(1.0, max(0.0, base))


def _infer_grammar_concepts(skill: SkillNode) -> List[str]:
    """Infer grammar concepts from skill title and category hint."""
    title_lower = skill.title.lower()
    concepts = []

    concept_keywords = {
        "present": "present tense conjugation",
        "past": "preterite formation",
        "imperfect": "imperfect aspect",
        "future": "simple future tense",
        "conditional": "conditional mood",
        "subjunctive": "subjunctive mood",
        "ser": "ser vs estar distinction",
        "estar": "ser vs estar distinction",
        "adjective": "adjective agreement",
        "plural": "plural formation",
        "object": "object pronoun placement",
        "reflexive": "reflexive verb construction",
        "por": "por vs para distinction",
        "para": "por vs para distinction",
        "relative": "relative clause construction",
        "passive": "passive voice",
        "question": "interrogative formation",
        "preposition": "prepositional usage",
    }

    for keyword, concept in concept_keywords.items():
        if keyword in title_lower and concept not in concepts:
            concepts.append(concept)

    if not concepts and skill.category_hint in ("grammar_introduction", "grammar_practice"):
        concepts.append("grammar structure")

    return concepts


def _category_counts(skills: List[SkillNode]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in skills:
        counts[s.category_hint] = counts.get(s.category_hint, 0) + 1
    return counts


def _top_gateway_skills(centrality: Dict[str, float], n: int = 5) -> List[str]:
    """Return the top-N skills by betweenness centrality (the structural gateways)."""
    if not centrality:
        return []
    sorted_skills = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [title for title, _ in sorted_skills[:n]]
