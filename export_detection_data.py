"""Export reviewed multi-dish annotations to YOLO labels; no model training here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from dish_annotations import CLASSES, validate_objects


def yolo_lines(objects, width, height):
    objects = validate_objects(objects, width, height)
    rows = []
    for obj in objects:
        x, y, w, h = obj['bounding_box']['coordinates']
        rows.append(f"{CLASSES.index(obj['class_name'])} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
    return '\n'.join(rows) + '\n'


def export_dataset(source, output, recursive=False):
    import cv2
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    records, problems = [], []
    images = sorted(p for p in (source.rglob('*') if recursive else source.glob('*')) if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                    and 'previews' not in p.relative_to(source).parts)
    for image in images:
        path = image.with_suffix('.json')
        if not path.exists():
            problems.append(f"Missing annotations: {image}")
            continue
        data = json.loads(path.read_text())
        if data.get('schema_version') != 2 or data.get('all_dishes_reviewed') is not True:
            problems.append(f"Review every dish with annotate_dishes.py: {image}")
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Unreadable image: {image}")
        h, w = frame.shape[:2]
        if (w, h) != (data.get('image_width'), data.get('image_height')) or data.get('image') != image.name:
            raise ValueError(f"Annotation image/dimensions mismatch: {path}")
        objects = validate_objects(data['objects'], w, h)
        relative = image.relative_to(source).as_posix()
        name = hashlib.sha256(relative.encode()).hexdigest()[:16] + image.suffix.lower()
        records.append({'source_image': str(image), 'source_metadata': str(path), 'export_name': name,
                        'plate_ids': sorted(obj['plate_id'] for obj in objects),
                        'classes': [obj['class_name'] for obj in objects],
                        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                        'yolo': yolo_lines(objects, w, h)})
    if problems:
        raise ValueError("Export stopped; no images should be treated as unlabelled background:\n" + '\n'.join(problems))
    if not records:
        raise ValueError("No reviewed images found")
    (output/'images').mkdir(parents=True)
    (output/'labels').mkdir()
    for row in records:
        shutil.copy2(row['source_image'], output/'images'/row['export_name'])
        (output/'labels'/Path(row['export_name']).with_suffix('.txt')).write_text(row.pop('yolo'))
    (output/'classes.json').write_text(json.dumps(dict(enumerate(CLASSES)), indent=2))
    (output/'manifest.json').write_text(json.dumps(records, indent=2))
    print(f"Exported {len(records)} images and {sum(len(r['classes']) for r in records)} dish boxes to {output}")
    print("No train/validation split was made. Shared physical IDs and repeated scenes must stay in one split.")
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='folder containing reviewed images')
    parser.add_argument('output', type=Path, help='new export directory')
    parser.add_argument('--recursive', action='store_true', help='also include subfolders except previews')
    args = parser.parse_args()
    try:
        export_dataset(args.source, args.output, args.recursive)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
