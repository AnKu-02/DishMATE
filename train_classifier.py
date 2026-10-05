"""Fine-tune the existing MobileNetV2 using grouped, human-labelled captures."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent


def load_records(source, validation_ids, test_ids):
    import cv2
    from annotate_images import validate_box
    if set(validation_ids) & set(test_ids):
        raise ValueError("Validation and test plate IDs overlap")
    records, hashes, excluded = [], {}, []
    for path in sorted(source.rglob("*.json")):
        if "previews" in path.relative_to(source).parts:
            continue
        data = json.loads(path.read_text())
        if Path(data.get("image", "")).stem != path.stem:
            excluded.append({"metadata": str(path), "reason": "image field differs from sidecar filename"})
            print("Excluding mismatched image reference:", path, flush=True)
            continue
        label, plate_id = data.get("label"), data.get("plate_id")
        if label not in ("clean", "dirty") or not plate_id:
            raise ValueError(f"Missing human label/plate ID: {path}")
        image = path.parent / data["image"]
        if image.resolve().parent != path.parent.resolve():
            raise ValueError(f"Invalid image path: {path}")
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Missing/unreadable image: {image}")
        h, w = frame.shape[:2]
        if (w, h) != (data.get("image_width"), data.get("image_height")):
            raise ValueError(f"Image dimensions disagree: {image}")
        box = data["bounding_box"]
        if box.get("format") != "xywh" or box.get("units") != "pixels":
            raise ValueError(f"Unsupported box: {path}")
        validate_box(box["coordinates"], w, h)
        split = "val" if plate_id in validation_ids else "test" if plate_id in test_ids else "train"
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image content: {image} and {hashes[digest]}")
        hashes[digest] = str(image)
        records.append({"image": str(image.resolve()), "metadata": str(path.resolve()),
                        "plate_id": plate_id, "label": label, "split": split,
                        "sha256_pixels": digest})
    for split in ("train", "val", "test"):
        labels = {r["label"] for r in records if r["split"] == split}
        if labels != {"clean", "dirty"}:
            raise ValueError(f"{split} must contain both classes, got {labels}")
    return records, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "captures/2026-10-05")
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
    records, excluded = load_records(args.source.resolve(), args.validation_plates, args.test_plates)
    run = ROOT / "training_runs" / ("cnn_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(records, indent=2))
    summary = {split: dict(Counter(r["label"] for r in records if r["split"] == split))
               for split in ("train", "val", "test")}
    print("Split by physical plate:", summary, flush=True)
    arrays = {}
    for split in summary:
        rows = [r for r in records if r["split"] == split]
        # Preserve original inference preprocessing: full image, OpenCV BGR, /255.
        x = np.asarray([cv2.resize(cv2.imread(r["image"]), (224, 224)) for r in rows], dtype=np.float32) / 255.0
        y = np.asarray([r["label"] == "dirty" for r in rows], dtype=np.float32)
        arrays[split] = (x, y)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=42),
        tf.keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=43),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect", seed=44),
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

    report = {"source": str(args.source.resolve()), "split_counts": summary, "excluded": excluded,
              "validation_plate_ids": args.validation_plates, "test_plate_ids": args.test_plates,
              "original_weights_sha256": original_hash,
              "preprocessing": "full image, BGR, resize 224x224, divide by 255; no bounding-box crop",
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
