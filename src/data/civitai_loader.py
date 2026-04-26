from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CivitaiModelVersion:
    id: int
    name: str
    trained_words: list[str] = field(default_factory=list)
    base_model: str = ""


@dataclass
class CivitaiModel:
    id: int
    name: str
    model_type: str  # "LORA", "Checkpoint", "TextualInversion", etc.
    nsfw: bool
    tags: list[str]
    rating: float
    download_count: int
    trained_words: list[str]  # flattened from all versions
    description: str = ""


@dataclass
class CivitaiImageMeta:
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = 0
    model_name: str = ""
    size: str = "512x512"


@dataclass
class CivitaiImage:
    id: int
    url: str
    width: int
    height: int
    nsfw: bool
    nsfw_level: int
    like_count: int
    meta: CivitaiImageMeta
    username: str = ""


def parse_civitai_model(raw: dict) -> CivitaiModel:
    """Parse raw model API response into CivitaiModel."""
    stats = raw.get("stats", {})
    versions = raw.get("modelVersions", [])

    trained_words: list[str] = []
    for version in versions:
        trained_words.extend(version.get("trainedWords", []))

    return CivitaiModel(
        id=raw["id"],
        name=raw["name"],
        model_type=raw.get("type", ""),
        nsfw=raw.get("nsfw", False),
        tags=raw.get("tags", []),
        rating=float(stats.get("rating", 0.0)),
        download_count=int(stats.get("downloadCount", 0)),
        trained_words=trained_words,
        description=raw.get("description", ""),
    )


def parse_civitai_image(raw: dict) -> CivitaiImage:
    """Parse raw image API response. meta may be None."""
    stats = raw.get("stats", {})
    meta_raw = raw.get("meta") or {}

    meta = CivitaiImageMeta(
        prompt=meta_raw.get("prompt", ""),
        negative_prompt=meta_raw.get("negativePrompt", ""),
        steps=int(meta_raw.get("Steps", 20)),
        cfg_scale=float(meta_raw.get("CFG scale", 7.0)),
        seed=int(meta_raw.get("Seed", 0)),
        model_name=meta_raw.get("Model", ""),
        size=meta_raw.get("Size", "512x512"),
    )

    return CivitaiImage(
        id=raw["id"],
        url=raw.get("url", ""),
        width=raw.get("width", 0),
        height=raw.get("height", 0),
        nsfw=raw.get("nsfw", False),
        nsfw_level=raw.get("nsfwLevel", 0),
        like_count=int(stats.get("likeCount", 0)),
        meta=meta,
        username=raw.get("username", ""),
    )


def extract_prompts(images: list[CivitaiImage]) -> list[str]:
    """Extract all non-empty prompts from image metadata."""
    return [img.meta.prompt for img in images if img.meta.prompt]


def image_to_instruction_sample(image: CivitaiImage) -> dict:
    """Convert image metadata to instruction format."""
    meta = image.meta
    return {
        "instruction": "Generate an image with the following description",
        "input": meta.prompt,
        "output": (f"negative: {meta.negative_prompt}, steps: {meta.steps}, cfg: {meta.cfg_scale}"),
    }


def filter_sfw_images(images: list[CivitaiImage]) -> list[CivitaiImage]:
    """Return only images where nsfw=False and nsfw_level <= 1."""
    return [img for img in images if not img.nsfw and img.nsfw_level <= 1]


def mock_civitai_models(n: int = 4) -> list[dict]:
    """Generate mock model responses matching real schema."""
    model_types = ["LORA", "Checkpoint", "TextualInversion", "Hypernetwork"]
    results = []
    for i in range(n):
        results.append(
            {
                "id": 10000 + i,
                "name": f"Mock Model {i}",
                "description": f"A mock model description {i}",
                "type": model_types[i % len(model_types)],
                "nsfw": i % 3 == 0,
                "tags": ["style", f"tag{i}"],
                "creator": {"username": f"user{i}", "image": f"https://example.com/img{i}.jpg"},
                "stats": {
                    "downloadCount": 1000 + i * 100,
                    "favoriteCount": 200 + i * 10,
                    "commentCount": 50 + i,
                    "ratingCount": 100 + i,
                    "rating": round(3.5 + (i % 3) * 0.5, 1),
                },
                "modelVersions": [
                    {
                        "id": 100 + i,
                        "name": f"v{i}.0",
                        "trainedWords": [f"word{i}a", f"word{i}b"],
                        "files": [],
                    }
                ],
            }
        )
    return results


def mock_civitai_images(n: int = 4) -> list[dict]:
    """Generate mock image responses with meta field populated."""
    results = []
    for i in range(n):
        # Vary nsfw status so filtering tests have both SFW and NSFW samples
        is_nsfw = i % 3 == 0
        nsfw_level = 3 if is_nsfw else (i % 2)
        results.append(
            {
                "id": 60000 + i,
                "url": f"https://image.civitai.com/mock/{i}.jpg",
                "hash": f"abcdef{i:04d}",
                "width": 512,
                "height": 512 + i * 64,
                "nsfw": is_nsfw,
                "nsfwLevel": nsfw_level,
                "createdAt": "2024-01-01T00:00:00Z",
                "postId": 200 + i,
                "stats": {
                    "cryCount": 0,
                    "laughCount": 0,
                    "likeCount": 100 + i * 5,
                    "heartCount": 50 + i,
                    "commentCount": i,
                },
                "meta": {
                    "prompt": f"a beautiful landscape scene {i}",
                    "negativePrompt": "blurry, low quality",
                    "Model": f"MockCheckpoint{i}",
                    "Steps": 20 + i,
                    "CFG scale": 7.0 + i * 0.5,
                    "Seed": 42 + i * 1000,
                    "Size": "512x512",
                },
                "username": f"mockuser{i}",
            }
        )
    return results
