"""Save captures with daily, per-item numbering without overwriting images."""
from datetime import datetime
from pathlib import Path
import re

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"


def save_capture(frame, item="plate", output=None, root=None, label=None):
    import cv2

    if item not in ("plate", "cup"):
        raise ValueError("Item must be plate or cup")
    if label not in (None, "clean", "dirty"):
        raise ValueError("Label must be clean or dirty")
    explicit = output is not None
    if explicit:
        path = Path(output).resolve()
    else:
        folder = Path(root if root is not None else CAPTURES_DIR) / datetime.now().strftime("%Y-%m-%d")
        if label is not None:
            folder = folder / label
        folder.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(rf"{item}_(\d+)\.png")
        numbers = [int(match.group(1)) for entry in folder.iterdir()
                   if (match := pattern.fullmatch(entry.name))]
        number = max(numbers, default=0) + 1
        path = folder / f"{item}_{number}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, encoded = cv2.imencode(path.suffix, frame)
    except cv2.error as exc:
        raise RuntimeError(f"Could not encode capture: {path}") from exc
    if not ok:
        raise RuntimeError(f"Could not encode capture: {path}")
    while True:
        try:
            handle = path.open("xb")
        except FileExistsError:
            if explicit:
                raise FileExistsError(f"Capture already exists; choose a new --output: {path}")
            number += 1
            path = folder / f"{item}_{number}.png"
            continue
        try:
            with handle:
                handle.write(encoded.tobytes())
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path.resolve()
