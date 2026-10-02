import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import numpy as np
import main_real
from prediction_log import log_prediction, FIELDS


class PredictionLogTests(unittest.TestCase):
    def test_append_match_mismatch_and_unknown(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "captures" / "results.csv"
            image = Path(tmp) / "plate,1.png"
            for expected in ("clean", "dirty", None):
                log_prediction(image, {"label": "clean", "score": 0.2}, expected, path)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, list(FIELDS))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["match"] for row in rows], ["true", "false", ""])
            self.assertEqual(rows[-1]["expected"], "")
            self.assertEqual(rows[0]["image"], str(image.resolve()))
            self.assertEqual(float(rows[0]["dirty_score"]), 0.2)

    def test_existing_image_logs_expected_without_camera(self):
        with patch("classifier_client.classify_image", return_value={"label": "clean", "score": 0.1}), \
                patch.object(main_real, "log_prediction") as log, \
                patch.object(main_real, "get_frame") as camera:
            main_real.main(["--image", "plate.png", "--expected", "clean"])
        camera.assert_not_called()
        log.assert_called_once_with(Path("plate.png"), {"label": "clean", "score": 0.1}, "clean")

    def test_dirty_score_threshold_preserved(self):
        model = Mock()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for score, label in ((0.5, "clean"), (0.5001, "dirty")):
            model.predict.return_value = np.array([[score]])
            self.assertEqual(main_real.predict_details(frame, model), {"label": label, "score": score})


if __name__ == "__main__":
    unittest.main()
