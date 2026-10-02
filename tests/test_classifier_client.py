import json
import os
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import main_real
from classifier_client import classify_image
from settings import configure_ros


class ClassifierTests(unittest.TestCase):
    def test_subprocess_isolated_from_ros_python(self):
        response = subprocess.CompletedProcess([], 0, json.dumps({"label": "clean"}), "")
        with patch.dict(os.environ, {"PYTHONPATH": "/ros/python38", "VIRTUAL_ENV": "/ros"}), \
                patch("classifier_client.subprocess.run", return_value=response) as run:
            self.assertEqual(classify_image("example.png")["label"], "clean")
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertNotIn("VIRTUAL_ENV", run.call_args.kwargs["env"])

    def test_capture_classify_does_not_import_motion(self):
        with patch.object(main_real, "get_frame", return_value=np.zeros((10, 10, 3))), \
                patch.object(main_real, "save_capture", return_value="plate_1.png"), \
                patch("classifier_client.classify_image", return_value={"label": "dirty"}) as classify, \
                patch.object(main_real.importlib, "import_module") as motion:
            main_real.main(["--classify"])
        classify.assert_called_once_with("plate_1.png")
        motion.assert_not_called()

    def test_localhost_default_replaced_but_explicit_master_preserved(self):
        with patch.dict(os.environ, {"ROS_MASTER_URI": "http://localhost:11311"}):
            config = configure_ros()
            self.assertEqual(os.environ["ROS_MASTER_URI"], config["ros_master_uri"])
        with patch.dict(os.environ, {"ROS_MASTER_URI": "http://example:11311"}):
            configure_ros()
            self.assertEqual(os.environ["ROS_MASTER_URI"], "http://example:11311")
