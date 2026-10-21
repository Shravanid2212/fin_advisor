"""Risk profile engine and scoring implementation.

Source of truth: docs/PRD.md Section 11 and Section 15.5.
Computes risk capacity (Q1-Q6) with investment horizon weighted 2x,
risk tolerance (Q7-Q10), normalises both to 0-100, maps to levels 1-5,
applies the lower-of-the-two rule, and enforces protective guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import AppConfig, TargetAllocationLevel, load_config


@dataclass(frozen=True)
class Option:
    """Option for a risk profiling question."""

    text: str
    score: int = 0


@dataclass(frozen=True)
class Question:
    """Question definition in the risk questionnaire."""

    id: str
    number: int
    category: str  # 'capacity', 'tolerance', or 'preference'
    weight: int
    prompt: str
    options: list[Option]
    multi_select: bool = False


@dataclass(frozen=True)
class ProfileResult:
    """Output of risk profile assessment."""

    capacity_score: float  # 0 to 100
    tolerance_score: float  # 0 to 100
    capacity_level: int  # 1 to 5
    tolerance_level: int  # 1 to 5
    preliminary_level: int  # min(capacity_level, tolerance_level)
    final_level: int  # after applying guardrails (1 to 5)
    label: str  # Profile label, e.g. 'Moderate'
    limiting_factor: str  # 'capacity', 'tolerance', or 'both'
    guardrail_messages: list[str]  # Explanations of any guardrails applied
    target_allocation: TargetAllocationLevel  # Target asset allocation from config
    raw_scores: dict[str, int] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)


def load_questions(yaml_path: str | Path | None = None) -> list[Question]:
    """Load the 14 risk questionnaire questions from YAML.

    Args:
        yaml_path: Optional path to questions.yaml. Defaults to 'src/profile/questions.yaml'.

    Returns:
        List of 14 validated Question objects.
    """
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent / "questions.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Questions configuration file not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("Invalid questions.yaml structure: 'questions' root key required")

    questions: list[Question] = []
    for item in data["questions"]:
        options = [
            Option(text=str(opt.get("text", "")), score=int(opt.get("score", 0)))
            for opt in item.get("options", [])
        ]
        q = Question(
            id=str(item["id"]).strip().lower(),
            number=int(item["number"]),
            category=str(item["category"]).strip().lower(),
            weight=int(item.get("weight", 1)),
            prompt=str(item["prompt"]),
            options=options,
            multi_select=bool(item.get("multi_select", False)),
        )
        questions.append(q)

    if len(questions) != 14:
        raise ValueError(f"Expected exactly 14 questions, found {len(questions)}")

    return questions


def map_score_to_level(score: float) -> int:
    """Map a normalised 0-100 score to risk level 1-5.

    Formula:
        Level 1 (Conservative): score <= 20
        Level 2 (Moderately Conservative): 20 < score <= 40
        Level 3 (Moderate): 40 < score <= 60
        Level 4 (Moderately Aggressive): 60 < score <= 80
        Level 5 (Aggressive): score > 80

    Args:
        score: Normalised score in [0, 100].

    Returns:
        Integer level from 1 to 5.
    """
    if score <= 20.0:
        return 1
    elif score <= 40.0:
        return 2
    elif score <= 60.0:
        return 3
    elif score <= 80.0:
        return 4
    else:
        return 5


def _resolve_answer_score(question: Question, answer: Any) -> int:
    """Resolve an answer value into a 1-4 score for capacity and tolerance questions.

    Supports integer scores (1-4) or string matching option text.
    """
    if isinstance(answer, (int, float)):
        val = int(answer)
        if 1 <= val <= 4:
            return val
        raise ValueError(
            f"Score for question {question.id} ({question.prompt}) must be between 1 and 4, got {val}"
        )

    if isinstance(answer, str):
        cleaned = answer.strip().lower()
        for opt in question.options:
            if opt.text.strip().lower() == cleaned:
                return opt.score
        # Check substring match if exact match fails
        for opt in question.options:
            if cleaned in opt.text.strip().lower() or opt.text.strip().lower() in cleaned:
                return opt.score

    raise ValueError(
        f"Invalid answer '{answer}' for question {question.id} ({question.prompt}). "
        f"Valid options: {[opt.text for opt in question.options]}"
    )


def calculate_profile(
    answers: dict[str, Any],
    config: AppConfig | None = None,
    questions: list[Question] | None = None,
) -> ProfileResult:
    """Calculate the investor risk profile from questionnaire answers.

    Formulas and rules:
        Raw Capacity:
            R_c = sum(w_i * s_i) for i in Q1..Q6
            where w_1..w_5 = 1, w_6 = 2 (horizon counted double).
            min R_c = 1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 2*1 = 7.
            max R_c = 1*4 + 1*4 + 1*4 + 1*4 + 1*4 + 2*4 = 28.
            Capacity Score = ((R_c - 7) / 21) * 100 in [0, 100].

        Raw Tolerance:
            R_t = sum(s_j) for j in Q7..Q10
            where w_7..w_10 = 1.
            min R_t = 4 * 1 = 4.
            max R_t = 4 * 4 = 16.
            Tolerance Score = ((R_t - 4) / 12) * 100 in [0, 100].

        Levels:
            Capacity Level = map_score_to_level(Capacity Score) in {1..5}
            Tolerance Level = map_score_to_level(Tolerance Score) in {1..5}
            Preliminary Level = min(Capacity Level, Tolerance Level)

        Guardrails:
            1. Horizon under 1 year (Q6 score == 1): final level <= 2.
            2. Emergency fund under 3 months (Q4 score == 1): final level <= 3.

        Preferences (Q11-Q14):
            Stored in preferences dictionary with 0 weight; never alter scores or levels.

    Args:
        answers: Dictionary mapping question IDs (e.g. 'q1'..'q14', or '1'..'14')
                 to selected option string or score.
        config: Optional AppConfig instance. If None, loaded via load_config().
        questions: Optional pre-loaded list of Question objects.

    Returns:
        ProfileResult containing scores, levels, limiting factor, guardrails, and target allocation.
    """
    if config is None:
        config = load_config()

    if questions is None:
        questions = load_questions()

    # Normalize answer keys (e.g. '1' -> 'q1', 'Q1' -> 'q1')
    normalized_answers: dict[str, Any] = {}
    for k, v in answers.items():
        k_str = str(k).strip().lower()
        if not k_str.startswith("q") and k_str.isdigit():
            k_str = f"q{k_str}"
        normalized_answers[k_str] = v

    raw_scores: dict[str, int] = {}
    preferences: dict[str, Any] = {}

    # 1. Capacity scoring (Q1-Q6)
    raw_capacity = 0
    min_capacity = 7
    max_capacity = 28

    for q in questions:
        if q.category == "capacity":
            ans = normalized_answers.get(q.id)
            if ans is None:
                raise ValueError(f"Missing required answer for capacity question {q.id} ({q.prompt})")
            score = _resolve_answer_score(q, ans)
            raw_scores[q.id] = score
            raw_capacity += score * q.weight

    capacity_score = round(((raw_capacity - min_capacity) / (max_capacity - min_capacity)) * 100.0, 4)
    capacity_level = map_score_to_level(capacity_score)

    # 2. Tolerance scoring (Q7-Q10)
    raw_tolerance = 0
    min_tolerance = 4
    max_tolerance = 16

    for q in questions:
        if q.category == "tolerance":
            ans = normalized_answers.get(q.id)
            if ans is None:
                raise ValueError(f"Missing required answer for tolerance question {q.id} ({q.prompt})")
            score = _resolve_answer_score(q, ans)
            raw_scores[q.id] = score
            raw_tolerance += score * q.weight

    tolerance_score = round(((raw_tolerance - min_tolerance) / (max_tolerance - min_tolerance)) * 100.0, 4)
    tolerance_level = map_score_to_level(tolerance_score)

    # 3. Preferences (Q11-Q14)
    for q in questions:
        if q.category == "preference":
            ans = normalized_answers.get(q.id)
            if ans is not None:
                preferences[q.id] = ans

    # 4. Preliminary Level (lower of the two)
    preliminary_level = min(capacity_level, tolerance_level)
    if capacity_level < tolerance_level:
        limiting_factor = "capacity"
    elif tolerance_level < capacity_level:
        limiting_factor = "tolerance"
    else:
        limiting_factor = "both"

    # 5. Guardrails
    # Guardrail 1: Horizon under 1 year (Q6 score == 1) caps level at 2
    # Guardrail 2: Emergency fund under 3 months (Q4 score == 1) caps level at 3
    final_level = preliminary_level
    guardrail_messages: list[str] = []

    # Horizon guardrail check: caps level at 2
    if raw_scores.get("q6") == 1 and final_level > 2:
        final_level = 2
        guardrail_messages.append(
            "Investment horizon under 1 year caps risk profile at Level 2 (Moderately Conservative)."
        )

    # Emergency fund guardrail check: caps level at 3
    if raw_scores.get("q4") == 1 and final_level > 3:
        final_level = 3
        guardrail_messages.append(
            "Emergency fund under 3 months caps risk profile at Level 3 (Moderate)."
        )

    target_allocation = config.target_allocations[final_level]

    return ProfileResult(
        capacity_score=capacity_score,
        tolerance_score=tolerance_score,
        capacity_level=capacity_level,
        tolerance_level=tolerance_level,
        preliminary_level=preliminary_level,
        final_level=final_level,
        label=target_allocation.label,
        limiting_factor=limiting_factor,
        guardrail_messages=guardrail_messages,
        target_allocation=target_allocation,
        raw_scores=raw_scores,
        preferences=preferences,
    )
