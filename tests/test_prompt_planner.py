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


def test_rules_plan_adds_face_recognition_constraints() -> None:
    planned = rules_plan("brighten the eyes and smooth the face")

    assert "Recognize people, faces, and body parts as structured anatomy" in planned
    assert "Apply the requested change only to the named face or body part" in planned
    assert "Preserve the person's identity, expression, gaze, and facial symmetry" in planned


def test_rules_plan_adds_body_part_constraints() -> None:
    planned = rules_plan("make the left hand sharper")

    assert "limbs, hands, skin texture, pose, proportions, and clothing unchanged" in planned
    assert "Preserve natural joints, fingers, limb count, posture, and body proportions" in planned
