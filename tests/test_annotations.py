import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from annotate_images import save_annotation, validate_box


class AnnotationTests(unittest.TestCase):
    def test_preserves_labels_and_image_when_saving_or_replacing_box(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "plate_1.png"
            image.write_bytes(b"original-image")
            image.with_suffix(".json").write_text(json.dumps({
                "label": "clean", "plate_id": "blue_01", "session": "session1"}))
            for box in ([10, 20, 100, 120], [20, 25, 110, 125]):
                path = save_annotation(image, box, 640, 480, "plate")
                data = json.loads(path.read_text())
                self.assertEqual(data["label"], "clean")
                self.assertEqual(data["plate_id"], "blue_01")
                self.assertEqual(data["session"], "session1")
                self.assertEqual(data["bounding_box"]["coordinates"], box)
                self.assertEqual(image.read_bytes(), b"original-image")

    def test_invalid_boxes_rejected(self):
        for box in ([0, 0, 0, 10], [-1, 0, 10, 10], [0, 0, 641, 480], [1.5, 0, 10, 10]):
            with self.assertRaises(ValueError):
                validate_box(box, 640, 480)

    def test_unlabelled_image_does_not_gain_an_invented_label(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "cup_1.png"
            path = save_annotation(image, [0, 0, 20, 20], 100, 100, "cup")
            data = json.loads(path.read_text())
            self.assertNotIn("label", data)
            self.assertEqual(data["bounding_box"]["object_type"], "cup")

    def test_malformed_metadata_is_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "plate_1.png"
            metadata = image.with_suffix(".json")
            metadata.write_text("broken json")
            with self.assertRaises(ValueError):
                save_annotation(image, [0, 0, 20, 20], 100, 100, "plate")
            self.assertEqual(metadata.read_text(), "broken json")
