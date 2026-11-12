import unittest

import numpy as np

import rack_classifier


class RackClassifierTest(unittest.TestCase):
    def test_crop_keeps_the_plate_area(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = rack_classifier.crop_plate(image, (0.2, 0.95, 0.0, 0.72))
        self.assertEqual(crop.shape, (345, 480, 3))

    def test_head_separates_classes_and_weights_them_equally(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(-1, 0.3, (5, 8))
        dirty = rng.normal(1, 0.3, (20, 8))   # four times as many dirty examples
        X = np.vstack([clean, dirty])
        y = np.array([0] * 5 + [1] * 20)
        w, bias = rack_classifier.train_head(X, y)
        p = rack_classifier.dirty_probability(X, w, bias)
        self.assertTrue(np.all(p[:5] < 0.5) and np.all(p[5:] > 0.5))
        self.assertLess(abs(rack_classifier.dirty_probability(np.zeros((1, 8)), w, bias)[0] - 0.5), 0.15)


if __name__ == "__main__":
    unittest.main()
