"""Tests for civitai_loader.py — no network calls, mock data only."""

from __future__ import annotations

import pytest
from aurelius.data.civitai_loader import (
    CivitaiImage,
    CivitaiImageMeta,
    extract_prompts,
    filter_sfw_images,
    image_to_instruction_sample,
    mock_civitai_images,
    mock_civitai_models,
    parse_civitai_image,
    parse_civitai_model,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_models():
    return mock_civitai_models(4)


@pytest.fixture()
def raw_images():
    return mock_civitai_images(4)


@pytest.fixture()
def parsed_models(raw_models):
    return [parse_civitai_model(r) for r in raw_models]


@pytest.fixture()
def parsed_images(raw_images):
    return [parse_civitai_image(r) for r in raw_images]


# ---------------------------------------------------------------------------
# Test 1 – mock_civitai_models has correct top-level fields
# ---------------------------------------------------------------------------


def test_mock_models_correct_fields(raw_models):
    required_keys = {"id", "name", "type", "nsfw", "tags", "stats", "modelVersions"}
    for model in raw_models:
        assert required_keys.issubset(model.keys()), (
            f"Missing keys {required_keys - model.keys()} in mock model"
        )


# ---------------------------------------------------------------------------
# Test 2 – parse_civitai_model extracts name correctly
# ---------------------------------------------------------------------------


def test_parse_model_name(raw_models, parsed_models):
    for raw, model in zip(raw_models, parsed_models):
        assert model.name == raw["name"]


# ---------------------------------------------------------------------------
# Test 3 – parse_civitai_model extracts trained_words from modelVersions
# ---------------------------------------------------------------------------


def test_parse_model_trained_words(raw_models, parsed_models):
    for raw, model in zip(raw_models, parsed_models):
        expected = []
        for version in raw["modelVersions"]:
            expected.extend(version.get("trainedWords", []))
        assert model.trained_words == expected


# ---------------------------------------------------------------------------
# Test 4 – parse_civitai_model reads rating from stats
# ---------------------------------------------------------------------------


def test_parse_model_rating(raw_models, parsed_models):
    for raw, model in zip(raw_models, parsed_models):
        assert model.rating == pytest.approx(raw["stats"]["rating"])


# ---------------------------------------------------------------------------
# Test 5 – mock_civitai_images has correct top-level fields
# ---------------------------------------------------------------------------


def test_mock_images_correct_fields(raw_images):
    required_keys = {"id", "url", "meta", "stats"}
    for image in raw_images:
        assert required_keys.issubset(image.keys()), (
            f"Missing keys {required_keys - image.keys()} in mock image"
        )


# ---------------------------------------------------------------------------
# Test 6 – parse_civitai_image extracts prompt from meta
# ---------------------------------------------------------------------------


def test_parse_image_prompt(raw_images, parsed_images):
    for raw, image in zip(raw_images, parsed_images):
        assert image.meta.prompt == raw["meta"]["prompt"]


# ---------------------------------------------------------------------------
# Test 7 – parse_civitai_image handles missing / None meta gracefully
# ---------------------------------------------------------------------------


def test_parse_image_none_meta():
    raw = {
        "id": 99999,
        "url": "https://example.com/img.jpg",
        "width": 512,
        "height": 512,
        "nsfw": False,
        "nsfwLevel": 0,
        "stats": {"likeCount": 0},
        "meta": None,
        "username": "tester",
    }
    image = parse_civitai_image(raw)
    assert image.meta.prompt == ""
    assert image.meta.negative_prompt == ""
    assert image.meta.steps == 20


def test_parse_image_missing_meta():
    raw = {
        "id": 99998,
        "url": "https://example.com/img2.jpg",
        "width": 256,
        "height": 256,
        "nsfw": False,
        "nsfwLevel": 0,
        "stats": {"likeCount": 5},
        "username": "tester2",
        # 'meta' key absent entirely
    }
    image = parse_civitai_image(raw)
    assert image.meta.prompt == ""


# ---------------------------------------------------------------------------
# Test 8 – extract_prompts returns list of non-empty strings
# ---------------------------------------------------------------------------


def test_extract_prompts_non_empty(parsed_images):
    prompts = extract_prompts(parsed_images)
    assert isinstance(prompts, list)
    assert all(isinstance(p, str) and p for p in prompts)


def test_extract_prompts_skips_empty():
    """Images with empty prompt should be excluded."""
    images = [
        CivitaiImage(
            id=1,
            url="",
            width=512,
            height=512,
            nsfw=False,
            nsfw_level=0,
            like_count=0,
            meta=CivitaiImageMeta(prompt="hello"),
        ),
        CivitaiImage(
            id=2,
            url="",
            width=512,
            height=512,
            nsfw=False,
            nsfw_level=0,
            like_count=0,
            meta=CivitaiImageMeta(prompt=""),
        ),
    ]
    prompts = extract_prompts(images)
    assert len(prompts) == 1
    assert prompts[0] == "hello"


# ---------------------------------------------------------------------------
# Test 9 – image_to_instruction_sample returns dict with required keys
# ---------------------------------------------------------------------------


def test_image_to_instruction_sample_keys(parsed_images):
    sample = image_to_instruction_sample(parsed_images[0])
    assert isinstance(sample, dict)
    assert "instruction" in sample
    assert "input" in sample
    assert "output" in sample


def test_image_to_instruction_sample_values(parsed_images):
    image = parsed_images[0]
    sample = image_to_instruction_sample(image)
    assert sample["input"] == image.meta.prompt
    assert "negative:" in sample["output"]
    assert "steps:" in sample["output"]
    assert "cfg:" in sample["output"]


# ---------------------------------------------------------------------------
# Test 10 – filter_sfw_images removes nsfw=True images
# ---------------------------------------------------------------------------


def test_filter_sfw_removes_nsfw(parsed_images):
    sfw = filter_sfw_images(parsed_images)
    for img in sfw:
        assert not img.nsfw


# ---------------------------------------------------------------------------
# Test 11 – filter_sfw_images keeps nsfw=False images with low nsfw_level
# ---------------------------------------------------------------------------


def test_filter_sfw_keeps_sfw():
    sfw_image = CivitaiImage(
        id=1,
        url="",
        width=512,
        height=512,
        nsfw=False,
        nsfw_level=1,
        like_count=0,
        meta=CivitaiImageMeta(prompt="safe"),
    )
    nsfw_image = CivitaiImage(
        id=2,
        url="",
        width=512,
        height=512,
        nsfw=True,
        nsfw_level=3,
        like_count=0,
        meta=CivitaiImageMeta(prompt="unsafe"),
    )
    result = filter_sfw_images([sfw_image, nsfw_image])
    assert len(result) == 1
    assert result[0].id == 1


# ---------------------------------------------------------------------------
# Test 12 – mock includes varied nsfw levels for filtering test
# ---------------------------------------------------------------------------


def test_mock_images_varied_nsfw_levels(raw_images):
    nsfw_levels = {img["nsfwLevel"] for img in raw_images}
    # Expect at least 2 distinct nsfw levels across 4 mock images
    assert len(nsfw_levels) >= 2, f"Expected varied nsfwLevel values, got {nsfw_levels}"


# ---------------------------------------------------------------------------
# Test 13 – model type field parsed correctly
# ---------------------------------------------------------------------------


def test_parse_model_type(raw_models, parsed_models):
    for raw, model in zip(raw_models, parsed_models):
        assert model.model_type == raw["type"]


# ---------------------------------------------------------------------------
# Test 14 – trained_words list correctly flattened from all versions
# ---------------------------------------------------------------------------


def test_trained_words_flattened_multi_version():
    raw = {
        "id": 1,
        "name": "Multi-Version Model",
        "type": "LORA",
        "nsfw": False,
        "tags": [],
        "description": "",
        "stats": {"downloadCount": 0, "rating": 4.0},
        "modelVersions": [
            {"id": 1, "name": "v1", "trainedWords": ["alpha", "beta"], "files": []},
            {"id": 2, "name": "v2", "trainedWords": ["gamma"], "files": []},
        ],
    }
    model = parse_civitai_model(raw)
    assert model.trained_words == ["alpha", "beta", "gamma"]


def test_trained_words_empty_when_no_versions():
    raw = {
        "id": 2,
        "name": "No Versions",
        "type": "Checkpoint",
        "nsfw": False,
        "tags": [],
        "description": "",
        "stats": {"downloadCount": 0, "rating": 3.0},
        "modelVersions": [],
    }
    model = parse_civitai_model(raw)
    assert model.trained_words == []
