import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np
from train_classifier import load_folder_records, load_records


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

    def test_exclusion_manifest_skips_image_and_records_reason(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter = 0
            for plate in ("plate_01", "plate_02", "plate_03", "plate_04"):
                for label in ("clean", "dirty"):
                    counter += 1
                    image = root / f"sample_{counter}.png"
                    cv2.imwrite(str(image), np.full((20, 30, 3), counter, dtype=np.uint8))
                    data = {"image": image.name, "label": label, "plate_id": plate,
                            "image_width": 30, "image_height": 20,
                            "bounding_box": {"format": "xywh", "units": "pixels",
                                             "coordinates": [0, 0, 30, 20]}}
                    image.with_suffix(".json").write_text(json.dumps(data))
            manifest = root / "exclusions.csv"
            manifest.write_text("image,reason\nsample_1.png,partially obscured\n")

            records, excluded = load_records(root, ["plate_02"], ["plate_03"],
                                             exclusion_manifest=manifest)

            self.assertEqual(len(records), 7)
            self.assertEqual(excluded, [{"image": str((root / "sample_1.png").resolve()),
                                         "metadata": str((root / "sample_1.json").resolve()),
                                         "reason": "partially obscured"}])

    def test_folder_dataset_has_reproducible_stratified_splits(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_index, folder in enumerate(("cleaned", "dirty")):
                directory = root / "train" / folder
                directory.mkdir(parents=True)
                for image_index in range(10):
                    value = 1 + class_index * 20 + image_index
                    image = np.full((20, 30, 3), value, dtype=np.uint8)
                    cv2.imwrite(str(directory / f"{image_index:04d}.jpg"), image)

            records, excluded = load_folder_records(root)
            repeated, _ = load_folder_records(root)
            counts = {(split, label): count for (split, label), count in
                      Counter((row["split"], row["label"]) for row in records).items()}
            signature = [(row["image"], row["label"], row["split"]) for row in records]
            repeated_signature = [(row["image"], row["label"], row["split"]) for row in repeated]

            self.assertEqual(excluded, [])
            self.assertEqual(counts, {("train", "clean"): 6, ("train", "dirty"): 6,
                                      ("val", "clean"): 2, ("val", "dirty"): 2,
                                      ("test", "clean"): 2, ("test", "dirty"): 2})
            self.assertEqual(signature, repeated_signature)

    def test_multidish_capture_expands_to_box_labeled_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, split in enumerate(("train", "val", "test")):
                image = np.full((20, 30, 3), 10 + index, dtype=np.uint8)
                image[:, :10] = 50 + index
                image[:, 15:] = 150 + index
                path = root / f"{split}.png"
                cv2.imwrite(str(path), image)
                data = {"image": path.name, "image_width": 30, "image_height": 20,
                        "objects": [
                            {"plate_id": f"{split}_clean", "label": "clean",
                             "bounding_box": {"format": "xywh", "units": "pixels",
                                              "coordinates": [0, 0, 10, 20]}},
                            {"plate_id": f"{split}_dirty", "label": "dirty",
                             "bounding_box": {"format": "xywh", "units": "pixels",
                                              "coordinates": [15, 0, 15, 20]}},
                        ]}
                path.with_suffix(".json").write_text(json.dumps(data))

            records, excluded = load_records(root, ["val_clean", "val_dirty"],
                                             ["test_clean", "test_dirty"], crop_to_box=True)
            counts = Counter((row["split"], row["label"]) for row in records)

            self.assertEqual(len(records), 6)
            self.assertEqual(excluded, [])
            self.assertEqual(counts, Counter({("train", "clean"): 1, ("train", "dirty"): 1,
                                              ("val", "clean"): 1, ("val", "dirty"): 1,
                                              ("test", "clean"): 1, ("test", "dirty"): 1}))
