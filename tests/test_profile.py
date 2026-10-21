"""Unit tests for the risk profile engine (src/profile/scoring.py).

Verifies:
1. All-lowest answers produce Level 1 (Conservative, 0 score).
2. All-highest answers produce Level 5 (Aggressive, 100 score).
3. The lower-of-the-two rule holds (min(capacity, tolerance)).
4. Horizon under 1 year guardrail caps final level at 2.
5. Emergency fund under 3 months guardrail caps final level at 3.
6. Preference answers (Q11-Q14) have strictly zero impact on scores or levels.
7. Both text-based and numeric answer inputs work identically.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.profile.scoring import (
    calculate_profile,
    load_questions,
    map_score_to_level,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def questions():
    return load_questions()


def test_load_questions_schema(questions):
    """Verify all 14 questions are parsed with proper categories and weights."""
    assert len(questions) == 14
    capacity_qs = [q for q in questions if q.category == "capacity"]
    tolerance_qs = [q for q in questions if q.category == "tolerance"]
    preference_qs = [q for q in questions if q.category == "preference"]

    assert len(capacity_qs) == 6
    assert len(tolerance_qs) == 4
    assert len(preference_qs) == 4

    # Horizon question (Q6) must have weight 2
    q6 = next(q for q in capacity_qs if q.id == "q6")
    assert q6.weight == 2


def test_map_score_to_level():
    """Verify cut-offs at 20, 40, 60, 80."""
    assert map_score_to_level(0.0) == 1
    assert map_score_to_level(20.0) == 1
    assert map_score_to_level(20.001) == 2
    assert map_score_to_level(40.0) == 2
    assert map_score_to_level(40.001) == 3
    assert map_score_to_level(60.0) == 3
    assert map_score_to_level(60.001) == 4
    assert map_score_to_level(80.0) == 4
    assert map_score_to_level(80.001) == 5
    assert map_score_to_level(100.0) == 5


def test_all_lowest_scores_produce_level_1(config, questions):
    """All lowest answers (1s) must produce score 0 and Level 1 Conservative."""
    answers = {f"q{i}": 1 for i in range(1, 11)}
    result = calculate_profile(answers, config=config, questions=questions)

    assert result.capacity_score == 0.0
    assert result.tolerance_score == 0.0
    assert result.capacity_level == 1
    assert result.tolerance_level == 1
    assert result.preliminary_level == 1
    assert result.final_level == 1
    assert result.label == "Conservative"
    assert result.limiting_factor == "both"
    assert len(result.guardrail_messages) == 0
    assert result.target_allocation.equity == 0.20
    assert result.target_allocation.debt == 0.70


def test_all_highest_scores_produce_level_5(config, questions):
    """All highest answers (4s) must produce score 100 and Level 5 Aggressive."""
    answers = {f"q{i}": 4 for i in range(1, 11)}
    result = calculate_profile(answers, config=config, questions=questions)

    assert result.capacity_score == 100.0
    assert result.tolerance_score == 100.0
    assert result.capacity_level == 5
    assert result.tolerance_level == 5
    assert result.preliminary_level == 5
    assert result.final_level == 5
    assert result.label == "Aggressive"
    assert result.limiting_factor == "both"
    assert len(result.guardrail_messages) == 0
    assert result.target_allocation.equity == 0.85
    assert result.target_allocation.debt == 0.10


def test_lower_of_two_rule(config, questions):
    """Profile level must be the minimum of capacity and tolerance."""
    # Case A: High capacity (Level 5) + Low tolerance (Level 2)
    # Tolerance raw: 2 + 2 + 2 + 2 = 8 -> ((8-4)/12)*100 = 33.33% -> Level 2
    answers_a = {f"q{i}": 4 for i in range(1, 7)}
    answers_a.update({f"q{i}": 2 for i in range(7, 11)})
    res_a = calculate_profile(answers_a, config=config, questions=questions)
    assert res_a.capacity_level == 5
    assert res_a.tolerance_level == 2
    assert res_a.preliminary_level == 2
    assert res_a.final_level == 2
    assert res_a.limiting_factor == "tolerance"
    assert res_a.label == "Moderately Conservative"

    # Case B: Low capacity (Level 2) + High tolerance (Level 5)
    # Capacity raw: 2 + 2 + 2 + 2 + 2 + 2*2 = 14 -> ((14-7)/21)*100 = 33.33% -> Level 2
    answers_b = {f"q{i}": 2 for i in range(1, 7)}
    answers_b.update({f"q{i}": 4 for i in range(7, 11)})
    res_b = calculate_profile(answers_b, config=config, questions=questions)
    assert res_b.capacity_level == 2
    assert res_b.tolerance_level == 5
    assert res_b.preliminary_level == 2
    assert res_b.final_level == 2
    assert res_b.limiting_factor == "capacity"
    assert res_b.label == "Moderately Conservative"


def test_guardrail_horizon_under_1_year(config, questions):
    """Investment horizon under 1 year (Q6=1) must cap level at 2."""
    # Q1-Q5 are high (4), Q6 is 'under 1 yr' (1)
    # Raw capacity = 4*5 + 2*1 = 22 -> ((22-7)/21)*100 = 71.43% -> Level 4
    # Tolerance is high (Level 4, all 3s -> raw 12 -> ((12-4)/12)*100 = 66.67% -> Level 4)
    answers = {
        "q1": 4,
        "q2": 4,
        "q3": 4,
        "q4": 4,
        "q5": 4,
        "q6": "under 1 yr",  # score 1
        "q7": 3,
        "q8": 3,
        "q9": 3,
        "q10": 3,
    }
    result = calculate_profile(answers, config=config, questions=questions)
    assert result.capacity_level == 4
    assert result.tolerance_level == 4
    assert result.preliminary_level == 4
    assert result.final_level == 2  # Capped at 2 by horizon guardrail
    assert result.label == "Moderately Conservative"
    assert any("horizon under 1 year" in msg.lower() for msg in result.guardrail_messages)


def test_guardrail_emergency_fund_under_3_months(config, questions):
    """Emergency fund under 3 months (Q4=1) must cap level at 3."""
    # Q1-Q3=4, Q4='under 3 months' (1), Q5=4, Q6=4 (7+ yrs)
    # Raw capacity = 4 + 4 + 4 + 1 + 4 + 2*4 = 25 -> ((25-7)/21)*100 = 85.71% -> Level 5
    # Tolerance is Level 5 (all 4s)
    answers = {
        "q1": 4,
        "q2": 4,
        "q3": 4,
        "q4": "under 3 months",  # score 1
        "q5": 4,
        "q6": 4,
        "q7": 4,
        "q8": 4,
        "q9": 4,
        "q10": 4,
    }
    result = calculate_profile(answers, config=config, questions=questions)
    assert result.capacity_level == 5
    assert result.tolerance_level == 5
    assert result.preliminary_level == 5
    assert result.final_level == 3  # Capped at 3 by emergency fund guardrail
    assert result.label == "Moderate"
    assert any("emergency fund under 3 months" in msg.lower() for msg in result.guardrail_messages)


def test_both_guardrail_conditions_handled(config, questions):
    """When both guardrail answers are lowest, horizon caps preliminary Level 3 down to Level 2."""
    answers = {
        "q1": 4,
        "q2": 4,
        "q3": 4,
        "q4": "under 3 months",  # score 1 (caps at 3)
        "q5": 4,
        "q6": "under 1 yr",  # score 1 (caps at 2)
        "q7": 4,
        "q8": 4,
        "q9": 4,
        "q10": 4,
    }
    result = calculate_profile(answers, config=config, questions=questions)
    # Capacity score = ((19 - 7) / 21) * 100 = 57.14% -> preliminary level = 3
    assert result.preliminary_level == 3
    assert result.final_level == 2
    assert len(result.guardrail_messages) == 1
    assert "horizon under 1 year" in result.guardrail_messages[0].lower()


def test_preference_answers_never_change_score(config, questions):
    """Preferences (Q11-Q14) are informational filters and have zero scoring weight."""
    base_answers = {f"q{i}": 3 for i in range(1, 11)}

    pref_variation_1 = {
        **base_answers,
        "q11": "retirement",
        "q12": "large only",
        "q13": ["IT", "FMCG"],
        "q14": "yes",
    }
    pref_variation_2 = {
        **base_answers,
        "q11": "wealth growth",
        "q12": "include small",
        "q13": ["Metals", "Auto"],
        "q14": "no",
    }
    pref_variation_3 = {
        **base_answers,
        "q11": "short-term",
        "q12": "large + mid",
        "q13": [],
        "q14": "no preference",
    }

    res_base = calculate_profile(base_answers, config=config, questions=questions)
    res_1 = calculate_profile(pref_variation_1, config=config, questions=questions)
    res_2 = calculate_profile(pref_variation_2, config=config, questions=questions)
    res_3 = calculate_profile(pref_variation_3, config=config, questions=questions)

    for res in [res_1, res_2, res_3]:
        assert res.capacity_score == res_base.capacity_score
        assert res.tolerance_score == res_base.tolerance_score
        assert res.capacity_level == res_base.capacity_level
        assert res.tolerance_level == res_base.tolerance_level
        assert res.preliminary_level == res_base.preliminary_level
        assert res.final_level == res_base.final_level
        assert res.limiting_factor == res_base.limiting_factor


def test_text_and_numeric_equivalence(config, questions):
    """Verify answering with option strings yields the exact same scores as numeric scores."""
    numeric_answers = {
        "q1": 4,  # under 25
        "q2": 4,  # fixed salary
        "q3": 4,  # none
        "q4": 3,  # 6-12
        "q5": 3,  # 10-30%
        "q6": 3,  # 3-7
        "q7": 3,  # hold
        "q8": 2,  # +10/-8
        "q9": 3,  # 2-5
        "q10": 4,  # maximum growth
    }

    text_answers = {
        "q1": "under 25",
        "q2": "fixed salary",
        "q3": "none",
        "q4": "6-12",
        "q5": "10-30%",
        "q6": "3-7",
        "q7": "hold",
        "q8": "+10/-8",
        "q9": "2-5",
        "q10": "maximum growth",
    }

    res_num = calculate_profile(numeric_answers, config=config, questions=questions)
    res_text = calculate_profile(text_answers, config=config, questions=questions)

    assert res_num.capacity_score == res_text.capacity_score
    assert res_num.tolerance_score == res_text.tolerance_score
    assert res_num.final_level == res_text.final_level
    assert res_num.limiting_factor == res_text.limiting_factor
