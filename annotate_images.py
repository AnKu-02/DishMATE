"""Review saved images and annotate one plate or cup per image; no ROS required."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile


def read_metadata(image):
    path = Path(image).with_suffix(".json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def validate_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("Box must contain x, y, width, height")
    if any(type(value) is not int for value in box):
        raise ValueError("Box coordinates must be integer pixels")
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("Box must have positive area and fit within the image")
    return list(box)


def save_annotation(image, box, width, height, item):
    if item not in ("plate", "cup"):
        raise ValueError("Item must be plate or cup")
    box = validate_box(box, width, height)
    image = Path(image)
    data = read_metadata(image)
    data.update(image=image.name, image_width=width, image_height=height,
                bounding_box={"format": "xywh", "units": "pixels", "coordinates": box,
                              "object_type": item},
                annotated_at=datetime.now().astimezone().isoformat())
    path = image.with_suffix(".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save_box_preview(image, frame, box, class_name, plate_id=None):
    """Derived visual reference only; keep originals for training."""
    import cv2
    image = Path(image)
    height, width = frame.shape[:2]
    x, y, w, h = validate_box(box, width, height)
    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    text = f"{class_name} | {plate_id}" if plate_id else class_name
    cv2.putText(preview, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    path = image.parent / "previews" / f"{image.stem}_boxed.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Could not save bounding-box preview: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="one image or a folder of images (searched recursively)")
    parser.add_argument("--item", choices=("plate", "cup"),
                        help="override object type; otherwise use metadata or filename prefix")
    args = parser.parse_args(argv)
    extensions = {".png", ".jpg", ".jpeg"}
    images = ([args.path] if args.path.is_file() else
              sorted(p for p in args.path.rglob("*") if p.suffix.lower() in extensions
                     and "previews" not in p.relative_to(args.path).parts))
    if not images:
        raise RuntimeError(f"No images found: {args.path}")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("This tool needs a GUI display, such as WSLg.")
    import cv2

    window = "Dish annotations"
    print("B: draw/replace box | S: save and next | N: skip without saving | Q/Esc: quit")
    print("While drawing: drag around the whole dish, Enter confirms, C cancels.")
    print("One object per image. Coordinates refer to the original image; originals are not modified.")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for index, path in enumerate(images, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Skipping unreadable image: {path}")
                continue
            height, width = frame.shape[:2]
            metadata = read_metadata(path)
            if "objects" in metadata:
                raise ValueError(f"{path.name} has multiple-object annotations; use annotate_dishes.py to edit safely")
            annotation = metadata.get("bounding_box", {})
            item = args.item or annotation.get("object_type") or path.stem.split("_")[0]
            if item not in ("plate", "cup"):
                raise ValueError(f"Cannot infer item for {path.name}; supply --item plate or --item cup")
            box = annotation.get("coordinates")
            if box is not None:
                if annotation.get("format") != "xywh" or annotation.get("units") != "pixels":
                    raise ValueError(f"Unsupported bounding box format: {path.with_suffix('.json')}")
                box = validate_box(box, width, height)
            print(f"[{index}/{len(images)}] {path} | {item} | label: {metadata.get('label', 'unlabelled')}")
            while True:
                preview = frame.copy()
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
                cv2.putText(preview, "B: box | S: save | N: skip | Q: quit", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key in (ord("n"), ord("N")):
                    break
                if key in (ord("b"), ord("B")):
                    candidate = tuple(int(v) for v in cv2.selectROI(window, frame, showCrosshair=True))
                    if candidate[2] and candidate[3]:
                        box = validate_box(candidate, width, height)
                if key in (ord("s"), ord("S")):
                    if box is None:
                        print("Draw a box with B first; nothing saved.")
                        continue
                    print(f"Saved: {save_annotation(path, box, width, height, item)}")
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
