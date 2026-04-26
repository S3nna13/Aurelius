"""
Tests for src/computer_use/screen_recorder.py
"""

import unittest

from src.computer_use.screen_recorder import (
    SCREEN_RECORDER_REGISTRY,
    ScreenRecorder,
    ScreenSnapshot,
)


class TestScreenSnapshot(unittest.TestCase):
    def test_snapshot_fields(self):
        snap = ScreenSnapshot(timestamp=1.0, width=1920, height=1080, content="hello", frame_id=0)
        self.assertEqual(snap.timestamp, 1.0)
        self.assertEqual(snap.width, 1920)
        self.assertEqual(snap.height, 1080)
        self.assertEqual(snap.content, "hello")
        self.assertEqual(snap.frame_id, 0)

    def test_snapshot_frozen(self):
        snap = ScreenSnapshot(timestamp=1.0, width=800, height=600, content="test", frame_id=0)
        with self.assertRaises(Exception):
            snap.content = "modified"  # type: ignore[misc]

    def test_snapshot_equality(self):
        snap1 = ScreenSnapshot(timestamp=1.0, width=800, height=600, content="abc", frame_id=0)
        snap2 = ScreenSnapshot(timestamp=1.0, width=800, height=600, content="abc", frame_id=0)
        self.assertEqual(snap1, snap2)


class TestScreenRecorder(unittest.TestCase):
    def setUp(self):
        self.recorder = ScreenRecorder()

    # --- capture ---
    def test_capture_returns_snapshot(self):
        snap = self.recorder.capture(800, 600, "content")
        self.assertIsInstance(snap, ScreenSnapshot)

    def test_capture_sets_width_height_content(self):
        snap = self.recorder.capture(1024, 768, "my content")
        self.assertEqual(snap.width, 1024)
        self.assertEqual(snap.height, 768)
        self.assertEqual(snap.content, "my content")

    def test_capture_frame_id_starts_at_zero(self):
        snap = self.recorder.capture(800, 600, "a")
        self.assertEqual(snap.frame_id, 0)

    def test_capture_frame_id_increments(self):
        snap0 = self.recorder.capture(800, 600, "a")
        snap1 = self.recorder.capture(800, 600, "b")
        snap2 = self.recorder.capture(800, 600, "c")
        self.assertEqual(snap0.frame_id, 0)
        self.assertEqual(snap1.frame_id, 1)
        self.assertEqual(snap2.frame_id, 2)

    def test_capture_timestamp_is_float(self):
        snap = self.recorder.capture(800, 600, "t")
        self.assertIsInstance(snap.timestamp, float)

    def test_capture_timestamp_positive(self):
        snap = self.recorder.capture(800, 600, "t")
        self.assertGreater(snap.timestamp, 0)

    def test_capture_does_not_store(self):
        self.recorder.capture(800, 600, "no store")
        self.assertEqual(len(self.recorder.frames()), 0)

    # --- record ---
    def test_record_stores_snapshot(self):
        snap = self.recorder.capture(800, 600, "abc")
        self.recorder.record(snap)
        self.assertEqual(len(self.recorder.frames()), 1)

    def test_record_multiple(self):
        for i in range(5):
            snap = self.recorder.capture(800, 600, f"frame {i}")
            self.recorder.record(snap)
        self.assertEqual(len(self.recorder.frames()), 5)

    # --- capture_and_record ---
    def test_capture_and_record_returns_snapshot(self):
        snap = self.recorder.capture_and_record(800, 600, "combined")
        self.assertIsInstance(snap, ScreenSnapshot)

    def test_capture_and_record_stores(self):
        snap = self.recorder.capture_and_record(800, 600, "stored")
        self.assertEqual(len(self.recorder.frames()), 1)
        self.assertEqual(self.recorder.frames()[0], snap)

    # --- latest ---
    def test_latest_returns_last(self):
        self.recorder.capture_and_record(800, 600, "first")
        snap2 = self.recorder.capture_and_record(800, 600, "second")
        self.assertEqual(self.recorder.latest(), snap2)

    def test_latest_empty_returns_none(self):
        self.assertIsNone(self.recorder.latest())

    def test_latest_single(self):
        snap = self.recorder.capture_and_record(1920, 1080, "only")
        self.assertEqual(self.recorder.latest(), snap)

    # --- diff ---
    def test_diff_same_content_changed_false(self):
        snap_a = ScreenSnapshot(1.0, 800, 600, "hello", 0)
        snap_b = ScreenSnapshot(2.0, 800, 600, "hello", 1)
        result = self.recorder.diff(snap_a, snap_b)
        self.assertFalse(result["content_changed"])
        self.assertFalse(result["changed"])

    def test_diff_different_content_changed_true(self):
        snap_a = ScreenSnapshot(1.0, 800, 600, "hello", 0)
        snap_b = ScreenSnapshot(2.0, 800, 600, "world", 1)
        result = self.recorder.diff(snap_a, snap_b)
        self.assertTrue(result["content_changed"])
        self.assertTrue(result["changed"])

    def test_diff_size_changed(self):
        snap_a = ScreenSnapshot(1.0, 800, 600, "same", 0)
        snap_b = ScreenSnapshot(2.0, 1920, 1080, "same", 1)
        result = self.recorder.diff(snap_a, snap_b)
        self.assertTrue(result["size_changed"])
        self.assertTrue(result["changed"])

    def test_diff_no_change(self):
        snap_a = ScreenSnapshot(1.0, 800, 600, "abc", 0)
        snap_b = ScreenSnapshot(2.0, 800, 600, "abc", 1)
        result = self.recorder.diff(snap_a, snap_b)
        self.assertFalse(result["changed"])
        self.assertFalse(result["size_changed"])
        self.assertFalse(result["content_changed"])

    def test_diff_returns_dict_with_required_keys(self):
        snap_a = ScreenSnapshot(1.0, 800, 600, "x", 0)
        snap_b = ScreenSnapshot(2.0, 800, 600, "y", 1)
        result = self.recorder.diff(snap_a, snap_b)
        self.assertIn("changed", result)
        self.assertIn("content_changed", result)
        self.assertIn("size_changed", result)

    # --- export_log ---
    def test_export_log_format(self):
        self.recorder.capture_and_record(800, 600, "log test")
        log = self.recorder.export_log()
        self.assertEqual(len(log), 1)
        entry = log[0]
        self.assertIn("frame_id", entry)
        self.assertIn("timestamp", entry)
        self.assertIn("width", entry)
        self.assertIn("height", entry)
        self.assertIn("content_len", entry)

    def test_export_log_content_len_correct(self):
        content = "hello world"
        self.recorder.capture_and_record(800, 600, content)
        log = self.recorder.export_log()
        self.assertEqual(log[0]["content_len"], len(content))

    def test_export_log_empty(self):
        self.assertEqual(self.recorder.export_log(), [])

    # --- max_frames eviction ---
    def test_max_frames_eviction(self):
        recorder = ScreenRecorder(max_frames=3)
        for i in range(5):
            recorder.capture_and_record(800, 600, f"frame {i}")
        frames = recorder.frames()
        self.assertEqual(len(frames), 3)
        # Oldest frames should have been evicted — last 3 should remain
        self.assertEqual(frames[0].frame_id, 2)
        self.assertEqual(frames[1].frame_id, 3)
        self.assertEqual(frames[2].frame_id, 4)

    def test_max_frames_default(self):
        recorder = ScreenRecorder()
        for i in range(100):
            recorder.capture_and_record(1, 1, str(i))
        self.assertEqual(len(recorder.frames()), 100)

    def test_max_frames_overflow_drops_oldest(self):
        recorder = ScreenRecorder(max_frames=2)
        recorder.capture_and_record(800, 600, "A")
        recorder.capture_and_record(800, 600, "B")
        recorder.capture_and_record(800, 600, "C")
        frames = recorder.frames()
        self.assertEqual(frames[0].content, "B")
        self.assertEqual(frames[1].content, "C")

    # --- REGISTRY ---
    def test_registry_exists(self):
        self.assertIn("default", SCREEN_RECORDER_REGISTRY)

    def test_registry_default_is_class(self):
        self.assertIs(SCREEN_RECORDER_REGISTRY["default"], ScreenRecorder)

    def test_registry_default_instantiable(self):
        cls = SCREEN_RECORDER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, ScreenRecorder)


if __name__ == "__main__":
    unittest.main()
