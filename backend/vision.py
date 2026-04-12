"""
Claude Vision — multi-modal extraction from Duolingo screenshots.
Sends captured images to Claude and extracts structured lesson data.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

import anthropic

from backend.models import ExtractedExercise, SkillVisualData

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

VISION_SYSTEM_PROMPT = """You are an expert educational UX researcher analyzing screenshots
of the Duolingo language learning app. Your job is to extract structured data about:

1. Exercise types visible (translation, listening, multiple choice, word-order, fill-blank, matching, speaking)
2. Vocabulary words shown (Spanish words, phrases, or sentences)
3. Grammar patterns visible (verb conjugations, noun-adjective agreement, tense markers, etc.)
4. UI/gamification elements present (progress bar, hearts, streak, XP display, correct/wrong feedback, character animations)
5. Estimated difficulty (1=beginner, 5=advanced)

Be precise and evidence-based — only report what is actually visible in the screenshot.
Always respond with valid JSON matching the requested schema."""


EXTRACTION_TOOL = {
    "name": "extract_lesson_data",
    "description": (
        "Extract structured data from a Duolingo screenshot. "
        "Call this once per screenshot with all observed lesson elements."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exercise_type": {
                "type": "string",
                "enum": ["translation", "listening", "speaking", "multiple_choice",
                         "matching", "fill_blank", "word_order", "reading", "unknown"],
                "description": "The primary exercise type visible in the screenshot"
            },
            "visible_words": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Spanish words, phrases, or sentences visible on screen"
            },
            "grammar_cues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Grammar patterns observed (e.g. 'verb conjugation -ar ending', 'noun gender agreement', 'preterite tense')"
            },
            "ui_elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "UI and gamification elements visible (e.g. 'progress bar', 'hearts display', 'streak fire icon', 'green correct feedback', 'XP counter')"
            },
            "estimated_difficulty": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Estimated difficulty: 1=absolute beginner, 5=advanced"
            },
            "notes": {
                "type": "string",
                "description": "Any additional observations about the lesson design, pedagogy, or UX"
            }
        },
        "required": ["exercise_type", "visible_words", "grammar_cues",
                     "ui_elements", "estimated_difficulty", "notes"]
    }
}


async def extract_from_screenshots(
    client: anthropic.AsyncAnthropic,
    screenshots: List[Tuple[str, str]],  # (label, base64_png)
    progress_cb=None,
) -> List[SkillVisualData]:
    """
    Send each screenshot to Claude Vision and extract structured lesson data.
    Returns one SkillVisualData per screenshot label.
    """
    if not screenshots:
        logger.info("No screenshots to process — skipping vision extraction")
        return []

    results: List[SkillVisualData] = []

    for label, b64_png in screenshots:
        if progress_cb:
            await progress_cb(f"Vision: analyzing {label}...")

        extracted = await _extract_single_screenshot(client, label, b64_png)
        if extracted:
            visual_data = _aggregate_to_skill(label, [extracted])
            results.append(visual_data)
            logger.info(f"Vision extraction complete for: {label}")

    return results


async def _extract_single_screenshot(
    client: anthropic.AsyncAnthropic,
    label: str,
    b64_png: str,
) -> Optional[ExtractedExercise]:
    """Send one screenshot to Claude Vision with tool use."""
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=VISION_SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "required"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_png,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a screenshot labeled '{label}' from the Duolingo app. "
                                "Extract all lesson data using the extract_lesson_data tool."
                            ),
                        },
                    ],
                }
            ],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_lesson_data":
                inp = block.input
                return ExtractedExercise(
                    exercise_type=inp["exercise_type"],
                    visible_words=inp["visible_words"],
                    grammar_cues=inp["grammar_cues"],
                    ui_elements=inp["ui_elements"],
                    estimated_difficulty=inp["estimated_difficulty"],
                    notes=inp["notes"],
                )

    except Exception as e:
        logger.warning(f"Vision extraction failed for {label}: {e}")

    return None


def _aggregate_to_skill(label: str, exercises: List[ExtractedExercise]) -> SkillVisualData:
    """Aggregate multiple extracted exercises into a single SkillVisualData."""
    all_words: list[str] = []
    all_grammar: list[str] = []
    all_ui: list[str] = []
    exercise_types: list[str] = []

    for ex in exercises:
        all_words.extend(ex.visible_words)
        all_grammar.extend(ex.grammar_cues)
        all_ui.extend(ex.ui_elements)
        exercise_types.append(ex.exercise_type)

    # Dominant exercise type by frequency
    dominant = max(set(exercise_types), key=exercise_types.count) if exercise_types else "unknown"

    return SkillVisualData(
        skill_title=label,
        screenshots_analyzed=len(exercises),
        extracted_exercises=exercises,
        dominant_exercise_type=dominant,
        observed_vocabulary=list(dict.fromkeys(all_words)),   # deduplicated, order preserved
        observed_grammar_patterns=list(dict.fromkeys(all_grammar)),
        gamification_elements=list(dict.fromkeys(all_ui)),
    )


def build_synthetic_visual_data(skills_sample: list) -> List[SkillVisualData]:
    """
    Build synthetic SkillVisualData for skills when screenshots aren't available.
    Used to give agents visual-dimension context even in fallback mode.
    """
    category_to_exercises = {
        "core_vocabulary": [
            ("translation", ["el hombre", "la mujer", "bebe"], ["subject-verb agreement"], ["progress bar", "hearts", "XP counter"]),
            ("multiple_choice", ["agua", "pan", "leche"], ["noun gender"], ["green correct feedback", "streak fire"]),
        ],
        "grammar_introduction": [
            ("fill_blank", ["soy", "eres", "es"], ["ser conjugation", "1st/2nd person"], ["progress bar", "orange incorrect feedback"]),
            ("word_order", ["grande el perro"], ["adjective position", "noun phrase order"], ["hearts", "green correct feedback"]),
        ],
        "grammar_practice": [
            ("translation", ["hablé", "comí", "viví"], ["preterite -é ending", "regular -ar/-er/-ir"], ["progress bar", "character animation"]),
            ("fill_blank", ["hablaba", "comía"], ["imperfect -aba ending", "ongoing past state"], ["hearts", "XP counter"]),
        ],
        "thematic_vocabulary": [
            ("matching", ["el avión", "el tren", "el hotel"], ["travel noun cluster"], ["progress bar", "streak fire"]),
            ("listening", ["la madre", "el padre"], ["family vocabulary"], ["audio waveform", "hearts"]),
        ],
        "pronunciation_phonetics": [
            ("speaking", ["rojo", "perro", "gente"], ["r/rr distinction", "g/j sound"], ["microphone icon", "sound waveform"]),
            ("listening", ["café", "árbol"], ["stress pattern recognition", "written accents"], ["audio waveform", "progress bar"]),
        ],
        "cultural_context": [
            ("translation", ["usted", "vosotros"], ["formal vs informal register"], ["cultural note card", "progress bar"]),
            ("multiple_choice", ["buenos días", "buenas noches"], ["time-appropriate greeting"], ["progress bar", "hearts"]),
        ],
        "review_consolidation": [
            ("translation", ["mixed vocabulary"], ["multiple tense interleaving"], ["review crown icon", "XP counter"]),
            ("word_order", ["mixed grammar"], ["cumulative grammar review"], ["progress bar", "streak fire"]),
        ],
        "compound_mixed": [
            ("translation", ["¿cuánto cuesta?", "me gustaría"], ["conditional polite form", "question inversion"], ["progress bar", "green correct feedback"]),
            ("fill_blank", ["pedir", "recomendar"], ["irregular verb in context"], ["hearts", "character animation"]),
        ],
    }

    result = []
    for skill in skills_sample[:20]:  # cap at 20 for synthetic data
        cat = getattr(skill, "category_hint", "compound_mixed")
        templates = category_to_exercises.get(cat, category_to_exercises["compound_mixed"])

        exercises = []
        for i, (ex_type, words, grammar, ui) in enumerate(templates):
            exercises.append(ExtractedExercise(
                exercise_type=ex_type,
                visible_words=words,
                grammar_cues=grammar,
                ui_elements=ui,
                estimated_difficulty=min(5, (skill.coords_y // 2) + 1),
                notes=f"Synthetic data for {skill.title} — category: {cat}",
            ))

        result.append(SkillVisualData(
            skill_title=skill.title,
            screenshots_analyzed=len(exercises),
            extracted_exercises=exercises,
            dominant_exercise_type=exercises[0].exercise_type if exercises else "translation",
            observed_vocabulary=skill.words[:8],
            observed_grammar_patterns=exercises[0].grammar_cues if exercises else [],
            gamification_elements=["progress bar", "hearts", "streak fire", "XP counter"],
        ))
    return result
