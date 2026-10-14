"""Fine-tune MobileNetV2 on grouped captures or a labeled class-folder dataset."""
import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent


def load_exclusions(manifest, source):
    if manifest is None:
        return {}
    source = source.resolve()
    exclusions = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "reason"}.issubset(reader.fieldnames):
            raise ValueError("Exclusion manifest needs image,reason columns")
        for row in reader:
            image = (row.get("image") or "").strip()
            reason = (row.get("reason") or "").strip()
            if not image or not reason:
                raise ValueError("Every exclusion row needs an image and reason")
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = source / image_path
            image_path = image_path.resolve()
            try:
                image_path.relative_to(source)
            except ValueError as exc:
                raise ValueError(f"Excluded image must be inside source folder: {image}") from exc
            key = str(image_path)
            if key in exclusions:
                raise ValueError(f"Duplicate exclusion entry: {image}")
            exclusions[key] = reason
    return exclusions


def load_records(source, validation_ids, test_ids, crop_to_box=False, exclusion_manifest=None):
    import cv2
    from annotate_images import validate_box
    source = source.resolve()
    if set(validation_ids) & set(test_ids):
        raise ValueError("Validation and test plate IDs overlap")
    exclusions = load_exclusions(exclusion_manifest, source)
    records, hashes, excluded = [], {}, []
    for path in sorted(source.rglob("*.json")):
        if "previews" in path.relative_to(source).parts:
            continue
        data = json.loads(path.read_text())
        if Path(data.get("image", "")).stem != path.stem:
            excluded.append({"metadata": str(path), "reason": "image field differs from sidecar filename"})
            print("Excluding mismatched image reference:", path, flush=True)
            continue
        image = path.parent / data["image"]
        if image.resolve().parent != path.parent.resolve():
            raise ValueError(f"Invalid image path: {path}")
        image_key = str(image.resolve())
        if image_key in exclusions:
            excluded.append({"image": image_key, "metadata": str(path.resolve()),
                             "reason": exclusions[image_key]})
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Missing/unreadable image: {image}")
        h, w = frame.shape[:2]
        if (w, h) != (data.get("image_width"), data.get("image_height")):
            raise ValueError(f"Image dimensions disagree: {image}")
        objects = data.get("objects")
        if objects is not None:
            if not crop_to_box:
                excluded.append({"metadata": str(path),
                                 "reason": "multi-object image requires --crop-to-box"})
                continue
            entries = objects
        else:
            entries = [data]
        for entry in entries:
            label, plate_id = entry.get("label"), entry.get("plate_id")
            if label not in ("clean", "dirty") or not plate_id:
                raise ValueError(f"Missing human label/plate ID: {path}")
            box = entry.get("bounding_box")
            if not isinstance(box, dict) or box.get("format") != "xywh" or box.get("units") != "pixels":
                raise ValueError(f"Unsupported bounding box: {path}")
            coordinates = validate_box(box["coordinates"], w, h)
            split = "val" if plate_id in validation_ids else "test" if plate_id in test_ids else "train"
            x, y, box_width, box_height = coordinates
            pixels = frame[y:y + box_height, x:x + box_width] if crop_to_box else frame
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image content: {image} and {hashes[digest]}")
            hashes[digest] = str(image)
            records.append({"image": str(image.resolve()), "metadata": str(path.resolve()),
                            "plate_id": plate_id, "label": label, "split": split,
                            "bounding_box": coordinates,
                            "sha256_pixels": digest})
    for split in ("train", "val", "test"):
        labels = {r["label"] for r in records if r["split"] == split}
        if labels != {"clean", "dirty"}:
            raise ValueError(f"{split} must contain both classes, got {labels}")
    return records, excluded


def load_folder_records(source, validation_fraction=0.2, test_fraction=0.2, seed=42):
    import cv2

    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    train_root = source / "train"
    class_dirs = {"cleaned": "clean", "dirty": "dirty"}
    records, hashes = [], {}
    rng = random.Random(seed)
    for folder, label in class_dirs.items():
        paths = sorted((train_root / folder).glob("*"))
        paths = [path for path in paths if path.is_file() and path.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        if len(paths) < 3:
            raise ValueError(f"Expected at least three labeled images in {train_root / folder}")
        rng.shuffle(paths)
        val_count = max(1, round(len(paths) * validation_fraction))
        test_count = max(1, round(len(paths) * test_fraction))
        if val_count + test_count >= len(paths):
            raise ValueError(f"Not enough images for train/validation/test splits in {train_root / folder}")
        split_paths = (("train", paths[val_count + test_count:]),
                       ("val", paths[:val_count]),
                       ("test", paths[val_count:val_count + test_count]))
        for split, images in split_paths:
            for path in images:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"Unreadable image: {path}")
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"Duplicate image content: {path} and {hashes[digest]}")
                hashes[digest] = str(path)
                records.append({"image": str(path.resolve()), "metadata": None,
                                "plate_id": None, "label": label, "split": split,
                                "bounding_box": None,
                                "sha256_pixels": digest})
    return records, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source", type=Path, default=ROOT / "captures/2026-10-05",
                         help="JSON-annotated capture folder")
    sources.add_argument("--folder-dataset", type=Path,
                         help="dataset root with train/cleaned and train/dirty folders")
    parser.add_argument("--exclusion-manifest", type=Path,
                        help="CSV of images to omit, with image,reason columns")
    parser.add_argument("--crop-to-box", action="store_true",
                        help="crop annotated images to the ground-truth plate box before resizing")
    parser.add_argument("--validation-plates", nargs="+", default=["plate_02"])
    parser.add_argument("--test-plates", nargs="+", default=["plate_03"])
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-epochs", type=int, default=18)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import tensorflow as tf
    from main_real import load_model, WEIGHTS_PATH

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    if args.folder_dataset:
        if args.exclusion_manifest:
            parser.error("--exclusion-manifest requires JSON-annotated --source data")
        source = args.folder_dataset.resolve()
        records, excluded = load_folder_records(source)
        split_method = "stratified random image split; physical plate IDs unavailable"
    else:
        source = args.source.resolve()
        records, excluded = load_records(source, args.validation_plates, args.test_plates,
                                         crop_to_box=args.crop_to_box,
                                         exclusion_manifest=args.exclusion_manifest)
        split_method = "grouped by physical plate ID"
    if args.crop_to_box and any(record["bounding_box"] is None for record in records):
        parser.error("--crop-to-box requires annotated capture data with bounding boxes")
    run = ROOT / "training_runs" / ("cnn_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(records, indent=2))
    summary = {split: dict(Counter(r["label"] for r in records if r["split"] == split))
               for split in ("train", "val", "test")}
    print("Split counts (%s): %s" % (split_method, summary), flush=True)
    arrays = {}
    for split in summary:
        rows = [r for r in records if r["split"] == split]
        frames = []
        for row in rows:
            frame = cv2.imread(row["image"])
            if frame is None:
                raise ValueError(f"Unreadable image: {row['image']}")
            if args.crop_to_box:
                x0, y0, width, height = row["bounding_box"]
                frame = frame[y0:y0 + height, x0:x0 + width]
            frames.append(cv2.resize(frame, (224, 224)))
        x = np.asarray(frames, dtype=np.float32) / 255.0
        y = np.asarray([r["label"] == "dirty" for r in rows], dtype=np.float32)
        arrays[split] = (x, y)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=42),
        tf.keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=43),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect", seed=44),
        tf.keras.layers.RandomZoom(0.08, 0.08, fill_mode="reflect", seed=45),
    ])

    def dataset(split, training=False):
        ds = tf.data.Dataset.from_tensor_slices(arrays[split])
        if training:
            ds = ds.shuffle(len(arrays[split][1]), seed=42)
        ds = ds.batch(8)
        if training:
            ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=1)
        return ds.with_options(options).prefetch(1)

    train, val = dataset("train", True), dataset("val")
    model = load_model()
    original_hash = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
    best = run / "best.weights.h5"

    def callbacks(phase, threshold=None):
        return [tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True,
                    save_weights_only=True, initial_value_threshold=threshold),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                tf.keras.callbacks.CSVLogger(str(run / f"{phase}_history.csv"))]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    print("Phase 1: train classification head", flush=True)
    history = model.fit(train, validation_data=val, epochs=args.head_epochs, callbacks=callbacks("head"), verbose=2)
    head_best = min(history.history["val_loss"])
    model.load_weights(str(best))
    for layer in model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    print("Phase 2: fine-tune last layers; BatchNormalization stays frozen", flush=True)
    model.fit(train, validation_data=val, epochs=args.fine_epochs,
              callbacks=callbacks("fine", head_best), verbose=2)
    model.load_weights(str(best))

    def evaluate(candidate, split):
        x, y = arrays[split]
        scores = np.concatenate([candidate(x[i:i+8], training=False).numpy().reshape(-1)
                                 for i in range(0, len(x), 8)])
        predicted = scores > 0.5
        actual = y.astype(bool)
        confusion = [[int(np.sum((actual == i) & (predicted == j))) for j in (False, True)]
                     for i in (False, True)]
        rows = [r for r in records if r["split"] == split]
        return {"accuracy": float(np.mean(predicted == actual)), "count": len(y),
                "confusion_rows_actual_cols_predicted_clean_dirty": confusion,
                "predictions": [{"image": r["image"], "expected": r["label"],
                                 "score": float(s), "predicted": "dirty" if s > 0.5 else "clean"}
                                for r, s in zip(rows, scores)]}

    report = {"source": str(source), "split_counts": summary, "excluded": excluded,
              "exclusion_manifest": (str(args.exclusion_manifest.resolve())
                                     if args.exclusion_manifest else None),
              "split_method": split_method,
              "validation_plate_ids": args.validation_plates if not args.folder_dataset else None,
              "test_plate_ids": args.test_plates if not args.folder_dataset else None,
              "original_weights_sha256": original_hash,
              "preprocessing": ("ground-truth bounding-box crop, BGR, resize 224x224, divide by 255"
                                if args.crop_to_box else
                                "full image, BGR, resize 224x224, divide by 255; no bounding-box crop"),
              "crop_to_box": args.crop_to_box,
              "augmentation": "training only: horizontal flip, rotation 0.06, translation 0.04, zoom 0.08; reflect fill",
              "selection": "lowest validation binary crossentropy across both phases; fixed score threshold 0.5",
              "tensorflow": tf.__version__, "seed": 42,
              "finetuned": {split: evaluate(model, split) for split in ("train", "val", "test")}}
    original = load_model()
    report["original"] = {split: evaluate(original, split) for split in ("val", "test")}
    # Verify the artifact loads with the exact existing inference architecture.
    original.load_weights(str(best))
    np.testing.assert_allclose(original(arrays["test"][0][:1], training=False).numpy(),
                               model(arrays["test"][0][:1], training=False).numpy(), atol=1e-6)
    assert hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest() == original_hash
    destination = ROOT / "models" / f"plate_cleanliness_{run.name}.weights.h5"
    import shutil
    shutil.copy2(best, destination)
    report["weights"] = str(destination)
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print("RESULT:", json.dumps({"weights": str(destination), "report": str(run / "report.json"),
          "original_test_accuracy": report["original"]["test"]["accuracy"],
          "finetuned_test_accuracy": report["finetuned"]["test"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
