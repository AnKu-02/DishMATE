import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import cv2
import numpy as np
from dish_annotations import save_objects, load_objects
from export_detection_data import yolo_lines, export_dataset


def dish(identifier, label, box):
    return {'plate_id': identifier, 'label': label, 'object_type': 'plate',
            'bounding_box': {'format': 'xywh', 'units': 'pixels', 'coordinates': box}}


class MultiDishTests(unittest.TestCase):
    def test_mixed_labels_roundtrip_and_original_preserved(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp)/'plate_1.png'
            cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
            original = image.read_bytes()
            image.with_suffix('.json').write_text(json.dumps({'label':'clean', 'plate_id':'old', 'session':'keep'}))
            objects = [dish('plate_01','clean',[0,0,100,100]), dish('plate_02','dirty',[100,0,100,100])]
            data = json.loads(save_objects(image, objects, 200, 100).read_text())
            self.assertNotIn('label', data)
            self.assertEqual(data['session'], 'keep')
            self.assertEqual(data['legacy_single_object']['label'], 'clean')
            self.assertEqual([o['class_name'] for o in load_objects(data,200,100)], ['clean_plate','dirty_plate'])
            self.assertEqual(image.read_bytes(), original)
            rows = export_dataset(Path(tmp), Path(tmp)/'export')
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['plate_ids'], ['plate_01','plate_02'])

    def test_box_normalization_and_separate_classes(self):
        lines = yolo_lines([dish('a','clean',[0,0,100,100]),dish('b','dirty',[100,0,100,100])],200,100)
        self.assertEqual(lines.splitlines(), ['0 0.25000000 0.50000000 0.50000000 1.00000000',
                                            '1 0.75000000 0.50000000 0.50000000 1.00000000'])

    def test_duplicate_physical_id_rejected(self):
        with self.assertRaises(ValueError):
            yolo_lines([dish('a','clean',[0,0,10,10]),dish('a','dirty',[10,0,10,10])],100,100)

    def test_unlabelled_images_block_export(self):
        with TemporaryDirectory() as tmp:
            cv2.imwrite(str(Path(tmp)/'plate.png'), np.zeros((20,20,3), dtype=np.uint8))
            output=Path(tmp)/'export'
            with self.assertRaises(ValueError):
                export_dataset(Path(tmp),output)
            self.assertFalse(output.exists())
