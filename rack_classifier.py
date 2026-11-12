"""Clean/dirty classifier for plates photographed in the dish rack (robot observe pose GP156).

The plate area of the photo is cropped, turned into a 1280-number feature vector by an
ImageNet-pretrained MobileNetV2, and a small logistic-regression layer trained on labelled rack
photos (datasets/rack_labels.csv) gives the dirty probability. Runs in the classifier
interpreter (Python 3.10, TensorFlow).

Retrain after adding labelled photos to datasets/rack_labels.csv:
    /root/.venvs/dishwashing-classifier/bin/python rack_classifier.py train
This prints a leave-one-day-out check and writes models/plate_rack_head.npz.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HEAD_PATH = ROOT / "models" / "plate_rack_head.npz"
LABELS_PATH = ROOT / "datasets" / "rack_labels.csv"
ROI = (0.20, 0.95, 0.0, 0.72)   # x0, x1, y0, y1 fractions of the photo: plate area at GP156
IMG_SIZE = 224
THRESHOLD = 0.5


def crop_plate(image, roi=ROI):
    h, w = image.shape[:2]
    x0, x1, y0, y1 = roi
    return image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def load_backbone():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))


def features(backbone, images, roi=ROI):
    """BGR photos -> L2-normalised 1280-number feature vectors of the plate area."""
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    batch = [preprocess_input(cv2.cvtColor(cv2.resize(crop_plate(im, roi), (IMG_SIZE, IMG_SIZE)),
                                           cv2.COLOR_BGR2RGB).astype("float32")) for im in images]
    feats = backbone.predict(np.array(batch), verbose=0)
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def train_head(X, y, steps=3000, rate=0.5, l2=1e-3):
    """Logistic regression with clean and dirty weighted equally; returns (weights, bias)."""
    w, bias = np.zeros(X.shape[1]), 0.0
    dirty_share = y.mean()
    sample_weight = np.where(y == 1, 0.5 / dirty_share, 0.5 / (1 - dirty_share))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + bias)))
        g = (p - y) * sample_weight
        w -= rate * (X.T @ g / len(y) + l2 * w)
        bias -= rate * g.mean()
    return w, bias


def dirty_probability(X, w, bias):
    return 1 / (1 + np.exp(-(X @ w + bias)))


def load_head(path=HEAD_PATH):
    data = np.load(path)
    return data["weights"], float(data["bias"]), tuple(data["roi"])


def predict(frame, backbone=None, head_path=HEAD_PATH):
    """One BGR photo -> {"label", "score"} with score = dirty probability."""
    w, bias, roi = load_head(head_path)
    score = float(dirty_probability(features(backbone or load_backbone(), [frame], roi), w, bias)[0])
    return {"label": "dirty" if score > THRESHOLD else "clean", "score": score, "model": "rack_head"}


def train(labels_path=LABELS_PATH, head_path=HEAD_PATH):
    import cv2
    rows = list(csv.DictReader(open(labels_path)))
    images = [cv2.imread(str(ROOT / r["image"])) for r in rows]
    missing = [r["image"] for r, im in zip(rows, images) if im is None]
    if missing:
        raise SystemExit(f"Cannot read: {missing}")
    y = np.array([1 if r["label"] == "dirty" else 0 for r in rows])
    days = np.array([Path(r["image"]).parts[1] for r in rows])   # captures/<day>/plate_N.png
    X = features(load_backbone(), images)

    # Check on unseen days before training the final layer on everything.
    pred = np.zeros_like(y)
    for day in np.unique(days):
        test = days == day
        if len(np.unique(y[~test])) < 2:
            continue
        w, bias = train_head(X[~test], y[~test])
        pred[test] = (dirty_probability(X[test], w, bias) > THRESHOLD).astype(int)
    wrong = [rows[i]["image"] for i in np.where(pred != y)[0]]
    print(f"Leave-one-day-out: {np.mean(pred == y):.0%} correct ({len(wrong)} wrong of {len(y)}; "
          f"{int(y.sum())} dirty, {int(len(y) - y.sum())} clean)")
    for name in wrong:
        print(f"  wrong: {name}")

    w, bias = train_head(X, y)
    head_path.parent.mkdir(exist_ok=True)
    np.savez(head_path, weights=w, bias=bias, roi=np.array(ROI), images=len(y))
    print(f"Saved {head_path}")
    return {"accuracy_unseen_days": float(np.mean(pred == y)), "wrong": wrong, "head": str(head_path)}


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        print(json.dumps(train()))
    else:
        raise SystemExit(__doc__)
