"""Camera failure and isolation checks; run with unittest discovery from the project root."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import main_real


class CameraTests(unittest.TestCase):
    def test_failed_open_releases_camera(self):
        camera = Mock()
        camera.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(RuntimeError, "could not be opened"):
                main_real.get_frame(warmup=0)
        camera.release.assert_called_once()

    def test_no_frame_releases_camera(self):
        camera = Mock()
        camera.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(RuntimeError, "returned no image"):
                main_real.get_frame(warmup=0)
        camera.release.assert_called_once()

    def test_default_capture_does_not_load_model_or_robot(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        with patch.object(main_real, "get_frame", return_value=frame), \
                patch.object(main_real, "save_capture", return_value="plate_1.png"), \
                patch.object(main_real, "load_model") as model, \
                patch.object(main_real.importlib, "import_module") as robot:
            main_real.main([])
        model.assert_not_called()
        robot.assert_not_called()

    def test_save_failure_is_reported(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        with patch.object(main_real, "get_frame", return_value=frame), \
                patch.object(main_real, "save_capture", side_effect=RuntimeError("Could not save capture")):
            with self.assertRaisesRegex(RuntimeError, "Could not save capture"):
                main_real.main(["--camera-only"])


if __name__ == "__main__":
    unittest.main()
