import pytest

from app.prompt_planner import clean_prompt, rules_plan


def test_clean_prompt_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        clean_prompt("   ")


def test_rules_plan_preserves_intent_and_adds_constraints() -> None:
    planned = rules_plan(" make the jacket red  ")

    assert planned.startswith("make the jacket red")
    assert "preserve composition" in planned
    # Must fit within CLIP's 77-token budget (~500 chars is a safe upper bound)
    assert len(planned) < 500


def test_rules_plan_adds_face_recognition_constraints() -> None:
    planned = rules_plan("brighten the eyes and smooth the face")

    assert "brighten the eyes and smooth the face" in planned
    assert "preserve identity" in planned
    assert "facial features" in planned
    # Compact enough that CLIP won't truncate the face guidance
    assert len(planned) < 200


def test_rules_plan_adds_body_part_constraints() -> None:
    planned = rules_plan("make the left hand sharper")

    assert "make the left hand sharper" in planned
    assert "proportions" in planned
    assert "anatomy" in planned
    assert len(planned) < 200
