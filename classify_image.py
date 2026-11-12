"""Classifier worker: reads one saved image, returns one JSON result."""
import argparse
from contextlib import redirect_stdout
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--weights", help="use this full-image CNN instead of the rack classifier")
    parser.add_argument("--cnn", action="store_true", help="use the original full-image CNN instead of the rack classifier")
    args = parser.parse_args()
    import cv2
    import rack_classifier

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    with redirect_stdout(sys.stderr):
        if args.cnn or args.weights or not rack_classifier.HEAD_PATH.exists():
            from main_real import load_model, predict_details
            result = dict(predict_details(frame, load_model(args.weights)), model="cnn")
        else:
            result = rack_classifier.predict(frame)
    print(json.dumps({**result, "image": args.image}))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
