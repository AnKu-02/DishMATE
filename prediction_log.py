"""Append classification evaluations; expected labels are supplied by the user."""
import csv
from datetime import datetime
from pathlib import Path
from settings import ROOT

RESULTS_PATH = ROOT / "captures" / "results.csv"
FIELDS = ("timestamp", "image", "predicted", "dirty_score", "expected", "match")


def log_prediction(image, result, expected=None, log_path=None):
    if expected not in (None, "clean", "dirty"):
        raise ValueError("Expected label must be clean or dirty")
    path = Path(log_path) if log_path is not None else RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Path(image).resolve()
    try:
        image_name = image.relative_to(ROOT).as_posix()
    except ValueError:
        image_name = str(image)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        handle.seek(0)
        header = next(csv.reader(handle), None)
        if header is not None and header != list(FIELDS):
            raise RuntimeError(f"Unexpected CSV columns in {path}; existing data preserved")
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if header is None:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "image": image_name,
            "predicted": result["label"],
            "dirty_score": result["score"],
            "expected": expected or "",
            "match": "" if expected is None else str(result["label"] == expected).lower(),
        })
    return path.resolve()
