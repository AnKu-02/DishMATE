"""Label all dishes in saved scenes. Uses the ROS environment's OpenCV, but no ROS."""
import argparse
import copy
import os
from pathlib import Path
import re
import sys
from annotate_images import read_metadata, validate_box
from dish_annotations import draw_objects, load_objects, save_objects, save_preview


def enter_id(window, frame, objects):
    import cv2
    text = ""
    while True:
        preview = draw_objects(frame, objects)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(preview, "Physical ID (e.g. plate_01), Enter accepts, Esc cancels:",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(preview, text + "_", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key in (10, 13) and text:
            return text
        if key in (8, 127):
            text = text[:-1]
        elif 32 <= key < 127 and (chr(key).isalnum() or chr(key) in "_-.") and len(text) < 64:
            text += chr(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--recursive", action="store_true", help="also include subfolders (previews always excluded)")
    args = parser.parse_args(argv)
    source = args.path
    images = [source] if source.is_file() else [
        p for p in (source.rglob("*") if args.recursive else source.glob("*"))
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and "previews" not in p.relative_to(source).parts]
    images.sort(key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))])
    if not images:
        raise RuntimeError(f"No images found: {source}")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("Open a WSL terminal with GUI/WSLg support")
    import cv2
    window = "Multiple dish annotations"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("1: clean, 2: dirty | P: plate, C: cup | B: add box | U: undo | S: confirm ALL dishes/save | N: skip | Q: quit")
    print("After drawing each box, type its physical ID in the image window. IDs persist across photos.")
    try:
        for index, path in enumerate(images, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot read image: {path}")
            height, width = frame.shape[:2]
            objects = load_objects(read_metadata(path), width, height)
            undo = []
            selected_label, selected_type = None, "plate"
            print(f"[{index}/{len(images)}] {path}: {len(objects)} existing boxes", flush=True)
            while True:
                preview = draw_objects(frame, objects)
                cv2.rectangle(preview, (0, 0), (width, 45), (0, 0, 0), -1)
                cv2.putText(preview, f"1 clean | 2 dirty | P plate | C cup | B add | U undo | S save | N next | Q quit",
                            (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
                cv2.putText(preview, f"Next: {selected_label or 'CHOOSE LABEL'} {selected_type} | boxes: {len(objects)}",
                            (5, 36), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.imshow(window, preview)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key in (ord('n'), ord('N')):
                    break
                if key == ord('1'):
                    selected_label = 'clean'
                if key == ord('2'):
                    selected_label = 'dirty'
                if key in (ord('p'), ord('P')):
                    selected_type = 'plate'
                if key in (ord('c'), ord('C')):
                    selected_type = 'cup'
                if key in (ord('u'), ord('U')):
                    if undo:
                        objects = undo.pop()
                    elif objects:
                        objects.pop()
                if key in (ord('b'), ord('B')):
                    if selected_label is None:
                        print("Choose 1 (clean) or 2 (dirty) before adding a dish.")
                        continue
                    box = [int(v) for v in cv2.selectROI(window, frame, showCrosshair=True)]
                    if not box[2] or not box[3]:
                        continue
                    box = validate_box(box, width, height)
                    identifier = enter_id(window, frame, objects)
                    if identifier is None:
                        continue
                    if any(obj['plate_id'] == identifier for obj in objects):
                        print("That ID already has a box in this image. Use U to remove an incorrect box.")
                        continue
                    undo.append(copy.deepcopy(objects))
                    objects.append({"plate_id": identifier, "label": selected_label, "object_type": selected_type,
                                    "class_name": selected_label + '_' + selected_type,
                                    "bounding_box": {"format": "xywh", "units": "pixels", "coordinates": box}})
                if key in (ord('s'), ord('S')):
                    if not objects:
                        print("Add a box for every visible dish first.")
                        continue
                    print(f"Saved: {save_objects(path, objects, width, height)}", flush=True)
                    print(f"Preview: {save_preview(path, frame, objects)}", flush=True)
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
