import unittest
import numpy as np
from camera_viewer import display_frame, CaptureTimer, save_viewer_capture, annotate_and_save
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json


class ViewerTests(unittest.TestCase):
    def test_annotation_pause_restarts_full_interval(self):
        timer = CaptureTimer(10)
        timer.due(100)
        self.assertTrue(timer.due(110))
        timer.restart(150)
        self.assertFalse(timer.due(159.9))
        self.assertTrue(timer.due(160))

    def test_cancel_annotation_saves_nothing(self):
        with patch("cv2.selectROI", return_value=(0, 0, 0, 0)), \
                patch("camera_viewer.save_viewer_capture") as save:
            self.assertIsNone(annotate_and_save(np.zeros((20, 20, 3)), "test", item="plate"))
        save.assert_not_called()

    def test_annotation_saves_frozen_frame_and_label_together(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        def select(window, frozen, **kwargs):
            frame[:] = 255  # Incoming data changes while the user annotates.
            return (2, 3, 10, 12)
        with TemporaryDirectory() as tmp, patch("cv2.selectROI", side_effect=select), \
                patch("capture_storage.CAPTURES_DIR", Path(tmp)):
            path = annotate_and_save(frame, "test", item="plate", label="clean", plate_id="blue_01")
            data = json.loads(path.with_suffix(".json").read_text())
            import cv2
            self.assertFalse(np.any(cv2.imread(str(path))))
            self.assertEqual(data["bounding_box"]["coordinates"], [2, 3, 10, 12])
            self.assertEqual(data["image_width"], 30)
            self.assertEqual(data["label"], "clean")
            self.assertEqual(data["plate_id"], "blue_01")
            self.assertEqual(path.parent.name, "clean")
            self.assertEqual(data["class_name"], "clean_plate")
            preview = path.parent / "previews" / f"{path.stem}_boxed.png"
            self.assertTrue(preview.exists())
            self.assertTrue(np.any(cv2.imread(str(preview))))

    def test_timer_waits_and_does_not_catch_up_in_bursts(self):
        timer = CaptureTimer(10)
        self.assertFalse(timer.due(100))
        self.assertFalse(timer.due(109.9))
        self.assertTrue(timer.due(110))
        self.assertFalse(timer.due(110.1))
        self.assertTrue(timer.due(150))
        self.assertFalse(timer.due(150.1))
        self.assertFalse(CaptureTimer(None).due(100))

    def test_human_label_and_item_id_saved_beside_image(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plate_1.png"
            with patch("camera_viewer.save_capture", return_value=path):
                save_viewer_capture(None, "plate", "clean", "blue_01", "session1", "/camera")
            data = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(data["label"], "clean")
            self.assertEqual(data["plate_id"], "blue_01")
            self.assertEqual(data["session"], "session1")

    def test_guides_do_not_modify_capture(self):
        original = np.zeros((120, 200, 3), dtype=np.uint8)
        for stale in (False, True):
            preview = display_frame(original, stale=stale)
            self.assertTrue(np.any(preview))
            self.assertFalse(np.any(original))
            self.assertFalse(np.shares_memory(preview, original))


if __name__ == "__main__":
    unittest.main()
