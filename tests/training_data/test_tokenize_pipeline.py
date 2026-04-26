from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from training_data.tokenize_pipeline import TokenizePipeline


@pytest.fixture
def pipeline():
    return TokenizePipeline(
        {
            "tokenize": {
                "shard_size": 8,
                "max_length": 16,
                "num_workers": 1,
                "verify_integrity": False,
            },
        }
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestTokenizePipeline:
    def test_load_tokenizer_fallback(self, pipeline):
        tok = pipeline.load_tokenizer()
        assert hasattr(tok, "encode")
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_tokenize_texts_creates_shards(self, pipeline, temp_dir):
        texts = ["hello world"] * 20
        shards = pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        assert len(shards) == 3
        for s in shards:
            assert os.path.exists(s)
        assert shards[0].endswith(".npy")

    def test_tokenize_texts_uint16_dtype(self, pipeline, temp_dir):
        texts = ["test"] * 5
        shards = pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        arr = np.load(shards[0])
        assert arr.dtype == np.uint16

    def test_tokenize_texts_max_length_truncation(self, pipeline, temp_dir):
        pipeline._max_length = 4
        texts = ["a" * 100]
        shards = pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        arr = np.load(shards[0])
        assert arr.shape[1] <= 4

    def test_tokenize_texts_id_clamping(self, pipeline, temp_dir):
        texts = ["a"]
        shards = pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        arr = np.load(shards[0])
        assert arr.max() < pipeline._vocab_size
        assert arr.min() >= 0

    def test_tokenize_jsonl(self, pipeline, temp_dir):
        jsonl_path = os.path.join(temp_dir, "input.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"document {i}"}) + "\n")

        output_dir = os.path.join(temp_dir, "shards")
        count = pipeline.tokenize_jsonl(jsonl_path, output_dir, text_field="text")
        assert count == 10
        shard_files = sorted(os.listdir(output_dir))
        assert len(shard_files) > 0
        assert all(f.endswith(".npy") for f in shard_files)

    def test_create_shard_manifest(self, pipeline, temp_dir):
        texts = ["hello"] * 10
        pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        manifest_path = os.path.join(temp_dir, "manifest.json")
        manifest = pipeline.create_shard_manifest(temp_dir, manifest_path)
        assert manifest["shard_count"] > 0
        assert "total_tokens" in manifest
        assert len(manifest["shards"]) == manifest["shard_count"]
        assert os.path.exists(manifest_path)

    def test_split_train_val(self, pipeline):
        shards = [f"shard_{i:06d}.npy" for i in range(20)]
        train, val = pipeline.split_train_val(shards, val_ratio=0.1, seed=42)
        assert len(train) + len(val) == 20
        assert len(val) >= 1
        assert len(train) >= len(val)
        # deterministic seed
        train2, val2 = pipeline.split_train_val(shards, val_ratio=0.1, seed=42)
        assert train == train2
        assert val == val2

    def test_validate_shard_valid(self, pipeline, temp_dir):
        arr = np.zeros((4, 8), dtype=np.uint16)
        path = os.path.join(temp_dir, "valid.npy")
        np.save(path, arr)
        assert pipeline.validate_shard(path) is True

    def test_validate_shard_wrong_dtype(self, pipeline, temp_dir):
        arr = np.zeros((4, 8), dtype=np.float32)
        path = os.path.join(temp_dir, "bad.npy")
        np.save(path, arr)
        assert pipeline.validate_shard(path) is False

    def test_validate_shard_zero_samples(self, pipeline, temp_dir):
        arr = np.zeros((0, 8), dtype=np.uint16)
        path = os.path.join(temp_dir, "empty.npy")
        np.save(path, arr)
        assert pipeline.validate_shard(path) is False

    def test_validate_shard_wrong_ndim(self, pipeline, temp_dir):
        arr = np.zeros(10, dtype=np.uint16)
        path = os.path.join(temp_dir, "flat.npy")
        np.save(path, arr)
        assert pipeline.validate_shard(path) is False

    def test_compute_hash(self, pipeline, temp_dir):
        arr = np.zeros((2, 4), dtype=np.uint16)
        path = os.path.join(temp_dir, "hash_test.npy")
        np.save(path, arr)
        h = pipeline.compute_hash(path)
        assert isinstance(h, str)
        assert len(h) == 64
        # deterministic
        assert pipeline.compute_hash(path) == h

    def test_validate_shard_nan_detection(self, pipeline, temp_dir):
        arr = np.zeros((2, 4), dtype=np.uint16)
        path = os.path.join(temp_dir, "no_nan.npy")
        np.save(path, arr)
        assert pipeline.validate_shard(path) is True
        # uint16 cannot hold NaN, so this tests the float-conversion path is safe

    def test_load_tokenizer_returns_same_instance(self, pipeline):
        tok1 = pipeline.load_tokenizer()
        tok2 = pipeline.load_tokenizer()
        assert tok1 is tok2

    def test_tokenize_jsonl_missing_field(self, pipeline, temp_dir):
        jsonl_path = os.path.join(temp_dir, "bad_input.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"title": "no text field"}) + "\n")
        output_dir = os.path.join(temp_dir, "shards_empty")
        count = pipeline.tokenize_jsonl(jsonl_path, output_dir, text_field="text")
        assert count == 0

    def test_tokenize_jsonl_file_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.tokenize_jsonl("/nonexistent/path.jsonl", "/tmp/out")

    def test_interleave_datasets(self, pipeline, temp_dir):
        dir1 = os.path.join(temp_dir, "ds1")
        dir2 = os.path.join(temp_dir, "ds2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        arr1 = np.ones((2, 4), dtype=np.uint16)
        arr2 = np.full((3, 4), 2, dtype=np.uint16)
        np.save(os.path.join(dir1, "shard_000000.npy"), arr1)
        np.save(os.path.join(dir2, "shard_000000.npy"), arr2)

        out_dir = os.path.join(temp_dir, "mixed")
        pipeline.interleave_datasets([dir1, dir2], out_dir, [0.5, 0.5])
        mixed_shards = sorted(os.listdir(out_dir))
        assert len(mixed_shards) == 2

    def test_create_shard_manifest_hash(self, pipeline, temp_dir):
        texts = ["hello"] * 10
        pipeline.tokenize_texts(texts, temp_dir, shard_size=8)
        manifest_path = os.path.join(temp_dir, "manifest.json")
        manifest = pipeline.create_shard_manifest(temp_dir, manifest_path)
        assert isinstance(manifest["hash"], str)
        assert len(manifest["hash"]) == 64
