"""Per-object annotations for scenes containing several dishes."""
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from annotate_images import read_metadata, validate_box

CLASSES = ("clean_plate", "dirty_plate", "clean_cup", "dirty_cup")


def validate_objects(objects, width, height):
    if not isinstance(objects, list) or not objects:
        raise ValueError("At least one labelled dish is required")
    result, identifiers = [], set()
    for original in objects:
        obj = copy.deepcopy(original)
        if obj.get("label") not in ("clean", "dirty") or obj.get("object_type") not in ("plate", "cup"):
            raise ValueError("Every dish needs an object type and clean/dirty label")
        identifier = obj.get("plate_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every dish needs a physical item ID")
        if identifier in identifiers:
            raise ValueError(f"Physical ID appears twice in one image: {identifier}")
        identifiers.add(identifier)
        box = obj.get("bounding_box", {})
        if box.get("format") != "xywh" or box.get("units") != "pixels":
            raise ValueError("Bounding boxes must use xywh pixels")
        box["coordinates"] = validate_box(box.get("coordinates"), width, height)
        obj["class_name"] = obj["label"] + "_" + obj["object_type"]
        result.append(obj)
    return result


def load_objects(metadata, width, height):
    if "objects" in metadata:
        return validate_objects(metadata["objects"], width, height)
    # Bring a labelled legacy single-object box into the editor for review.
    if metadata.get("bounding_box") and metadata.get("label") and metadata.get("plate_id"):
        return validate_objects([{
            "label": metadata["label"], "plate_id": metadata["plate_id"],
            "object_type": metadata.get("object_type", metadata["bounding_box"].get("object_type", "plate")),
            "bounding_box": metadata["bounding_box"],
        }], width, height)
    return []


def save_objects(image, objects, width, height):
    image = Path(image)
    objects = validate_objects(objects, width, height)
    data = read_metadata(image)
    legacy = {key: data.pop(key) for key in ("label", "class_name", "object_type", "plate_id", "bounding_box") if key in data}
    if legacy:
        data["legacy_single_object"] = legacy
    data.update(schema_version=2, image=image.name, image_width=width, image_height=height,
                objects=objects, all_dishes_reviewed=True,
                annotated_at=datetime.now().astimezone().isoformat())
    target = image.with_suffix(".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(str(temporary), str(target))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def draw_objects(frame, objects):
    import cv2
    preview = frame.copy()
    for index, obj in enumerate(objects, 1):
        x, y, w, h = obj["bounding_box"]["coordinates"]
        color = (0, 200, 0) if obj["label"] == "clean" else (0, 80, 255)
        cv2.rectangle(preview, (x, y), (x+w-1, y+h-1), color, 2)
        label = f"{index}: {obj['class_name']} {obj['plate_id']}"
        cv2.putText(preview, label, (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    return preview


def save_preview(image, frame, objects):
    import cv2
    path = Path(image).parent / "previews" / (Path(image).stem + "_boxed.png")
    path.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(path), draw_objects(frame, objects)):
        raise RuntimeError(f"Could not save preview: {path}")
    return path
