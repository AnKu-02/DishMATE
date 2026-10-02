"""Capture storage checks using temporary folders."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from capture_storage import save_capture


class StorageTests(unittest.TestCase):
    def test_numbering_dates_and_image_roundtrip(self):
        frame = np.full((12, 16, 3), 80, dtype=np.uint8)
        with TemporaryDirectory() as tmp, patch("capture_storage.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 1)
            first = save_capture(frame, root=tmp)
            second = save_capture(frame, root=tmp)
            cup = save_capture(frame, item="cup", root=tmp)
            self.assertEqual(first.relative_to(tmp), Path("2026-10-01/plate_1.png"))
            self.assertEqual(second.name, "plate_2.png")
            self.assertEqual(cup.name, "cup_1.png")
            np.testing.assert_array_equal(cv2.imread(str(first)), frame)
            clock.now.return_value = datetime(2026, 10, 2)
            tomorrow = save_capture(frame, root=tmp)
            self.assertEqual(tomorrow.relative_to(tmp), Path("2026-10-02/plate_1.png"))
            self.assertTrue(first.exists())

    def test_explicit_output_creates_parents_and_refuses_overwrite(self):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "example.png"
            self.assertEqual(save_capture(frame, output=output), output.resolve())
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                save_capture(frame + 1, output=output)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
