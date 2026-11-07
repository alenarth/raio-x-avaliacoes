import pytest

from reviews.llm import pick_flash_model


def test_picks_newest_stable_flash():
    names = [
        "models/gemini-2.5-flash", "models/gemini-3-flash", "models/gemini-3.6-flash",
        "models/gemini-3.8-flash-preview", "models/gemini-3.6-flash-lite", "models/gemini-3.6-pro",
        "models/gemini-3.6-flash-image", "models/text-embedding-004",
    ]
    assert pick_flash_model(names) == "gemini-3.6-flash"


def test_falls_back_to_preview_when_no_stable():
    assert pick_flash_model(["models/gemini-4-flash-preview", "models/gemini-4-pro"]) == "gemini-4-flash-preview"


def test_errors_when_no_flash():
    with pytest.raises(RuntimeError):
        pick_flash_model(["models/gemini-3-pro"])
