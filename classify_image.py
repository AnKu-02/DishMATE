"""Classifier worker: reads one saved image, returns one JSON result."""
import argparse
from contextlib import redirect_stdout
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    args = parser.parse_args()
    import cv2
    from main_real import load_model, predict_details

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    with redirect_stdout(sys.stderr):
        result = predict_details(frame, load_model())
    print(json.dumps({**result, "image": args.image}))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
