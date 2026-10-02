"""Run the model in its own interpreter without importing TensorFlow into ROS."""
import json
import math
import os
from pathlib import Path
import subprocess
from settings import ROOT, load_settings


def classify_image(path):
    interpreter = load_settings()["classifier_python"]
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    try:
        completed = subprocess.run(
            [interpreter, str(ROOT / "classify_image.py"), str(Path(path).resolve())],
            env=env, capture_output=True, text=True, timeout=120, check=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Classifier timed out after 120 seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Classifier failed: {exc.stderr.strip()}") from exc
    try:
        result = json.loads(completed.stdout)
        if result["label"] not in ("clean", "dirty"):
            raise ValueError("Unexpected label")
        score = result["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid score")
        if result["label"] != ("dirty" if score > 0.5 else "clean"):
            raise ValueError("Label does not match score")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Classifier returned an invalid response.") from exc
    return result
