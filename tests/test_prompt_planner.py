import pytest

from app.prompt_planner import clean_prompt, rules_plan


def test_clean_prompt_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        clean_prompt("   ")


def test_rules_plan_preserves_intent_and_adds_constraints() -> None:
    planned = rules_plan(" make the jacket red  ")

    assert planned.startswith("make the jacket red.")
    assert "Preserve the original composition" in planned
    assert "photorealistic" in planned
