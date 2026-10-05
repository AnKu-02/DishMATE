import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np
from train_classifier import load_records


class TrainingDataTests(unittest.TestCase):
    def test_split_is_by_plate_not_photo_and_mismatch_is_excluded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter = 0
            for plate in ("plate_01", "plate_02", "plate_03"):
                for label in ("clean", "dirty"):
                    counter += 1
                    image = root / f"sample_{counter}.png"
                    cv2.imwrite(str(image), np.full((20, 30, 3), counter, dtype=np.uint8))
                    data = {"image": image.name, "label": label, "plate_id": plate,
                            "image_width": 30, "image_height": 20,
                            "bounding_box": {"format": "xywh", "units": "pixels", "coordinates": [0, 0, 30, 20]}}
                    image.with_suffix(".json").write_text(json.dumps(data))
            (root / "bad.json").write_text(json.dumps({"image": "sample_1.png"}))
            records, excluded = load_records(root, ["plate_02"], ["plate_03"])
            self.assertEqual(len(records), 6)
            self.assertEqual(len(excluded), 1)
            for row in records:
                self.assertEqual(row["split"], {"plate_01": "train", "plate_02": "val", "plate_03": "test"}[row["plate_id"]])

    def test_overlapping_holdouts_rejected(self):
        with self.assertRaises(ValueError):
            load_records(Path("unused"), ["plate_01"], ["plate_01"])
