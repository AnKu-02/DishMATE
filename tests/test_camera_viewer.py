import unittest
import numpy as np
from camera_viewer import display_frame


class ViewerTests(unittest.TestCase):
    def test_guides_do_not_modify_capture(self):
        original = np.zeros((120, 200, 3), dtype=np.uint8)
        for stale in (False, True):
            preview = display_frame(original, stale=stale)
            self.assertTrue(np.any(preview))
            self.assertFalse(np.any(original))
            self.assertFalse(np.shares_memory(preview, original))


if __name__ == "__main__":
    unittest.main()
