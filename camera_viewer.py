"""Live ROS camera viewer. No classifier or robot motion clients are loaded."""
import argparse
import math
import json
from datetime import datetime
from uuid import uuid4
import os
import sys
import time
from threading import Lock

from capture_storage import save_capture
from settings import configure_ros, load_settings
from annotate_images import validate_box, save_box_preview


class CaptureTimer:
    """Start on the first frame; skip missed slots rather than saving bursts."""
    def __init__(self, interval):
        self.interval = interval
        self.next_capture = None

    def due(self, now):
        if self.interval is None:
            return False
        if self.next_capture is None:
            self.next_capture = now + self.interval
        if now < self.next_capture:
            return False
        self.next_capture = now + self.interval
        return True

    def restart(self, now):
        self.next_capture = None if self.interval is None else now + self.interval


def save_viewer_capture(frame, item, label=None, plate_id=None, session=None, topic=None,
                        box=None, captured_at=None):
    if box is not None:
        height, width = frame.shape[:2]
        box = validate_box(box, width, height)
    path = save_capture(frame, item=item, label=label)
    if label is not None or box is not None:
        metadata = {"image": path.name, "label": label, "plate_id": plate_id,
                    "object_type": item, "class_name": f"{label}_{item}" if label else item,
                    "session": session, "topic": topic,
                    "captured_at": captured_at or datetime.now().astimezone().isoformat()}
        if box is not None:
            metadata.update(image_width=width, image_height=height,
                            bounding_box={"format": "xywh", "units": "pixels",
                                          "coordinates": box, "object_type": item},
                            annotated_at=datetime.now().astimezone().isoformat())
        with path.with_suffix(".json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if box is not None:
            preview = save_box_preview(path, frame, box, metadata["class_name"], plate_id)
            print(f"Box preview: {preview}", flush=True)
    return path


def annotate_and_save(frame, window, **capture_options):
    """Freeze exactly this frame; cancel saves neither image nor metadata."""
    import cv2
    frozen = frame.copy()
    captured_at = datetime.now().astimezone().isoformat()
    print("Draw a box around the whole dish. Enter/Space: save; C/Esc: skip. Timer paused.", flush=True)
    box = tuple(int(value) for value in cv2.selectROI(window, frozen, showCrosshair=True))
    if not box[2] or not box[3]:
        return None
    return save_viewer_capture(frozen, box=box, captured_at=captured_at, **capture_options)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def display_frame(frame, stale=False):
    """Draw guides on a copy so saved images remain unmodified."""
    import cv2

    preview = frame.copy()
    height, width = preview.shape[:2]
    center = (width // 2, height // 2)
    color = (0, 0, 255) if stale else (0, 255, 0)
    cv2.drawMarker(preview, center, color, cv2.MARKER_CROSS, 30, 2)
    label = "STALE - saving disabled" if stale else "S: save | Q/Esc: quit"
    cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=load_settings()["camera_topic"],
                        help="ROS image topic; /compressed suffix selects CompressedImage")
    parser.add_argument("--item", choices=("plate", "cup"), default="plate")
    parser.add_argument("--interval", type=positive_seconds,
                        help="automatically save every N seconds after the first frame")
    parser.add_argument("--label", choices=("clean", "dirty"),
                        help="human-provided label for all saved images in this session")
    parser.add_argument("--plate-id", help="physical item identifier, e.g. plate_01")
    parser.add_argument("--annotate", action="store_true",
                        help="pause each capture to draw a box; restart interval after saving/skipping")
    parser.add_argument("--timeout", type=positive_seconds, default=10,
                        help="seconds to wait for the first image after ROS initialization")
    parser.add_argument("--stale-after", type=positive_seconds, default=2,
                        help="disable saving if no new frame arrives for this many seconds")
    args = parser.parse_args(argv)
    if args.label and not args.plate_id:
        parser.error("--label requires --plate-id to identify the physical item")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("No DISPLAY is configured. Open a WSL terminal with GUI/WSLg support.")
    try:
        import cv2
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, CompressedImage
    except ImportError as exc:
        raise RuntimeError("Run in the Python 3.8 ROS environment: source ./activate_wsl.sh") from exc

    configure_ros()
    print(f"ROS master: {os.environ['ROS_MASTER_URI']}", flush=True)
    print(f"Topic: {args.topic} | Saving as: {args.item}_N.png", flush=True)
    print("S saves an image without overlays; Q/Esc or closing the window exits. Ctrl+C also exits.", flush=True)
    if args.interval:
        print(f"Automatic capture every {args.interval:g}s. Stale frames are skipped.", flush=True)
    if args.label:
        print(f"All captures labelled {args.label}; item ID: {args.plate_id}. Restart to change labels.", flush=True)
    if args.annotate:
        print("Each capture freezes for box annotation. Enter saves; C/Esc skips. Q exits in live view.", flush=True)
    rospy.init_node("dishwashing_camera_viewer", anonymous=True)
    bridge = CvBridge()
    lock = Lock()
    latest = {"frame": None, "received": 0.0, "error": None}
    compressed = args.topic.rstrip("/").endswith("/compressed")

    def receive(message):
        try:
            frame = (bridge.compressed_imgmsg_to_cv2(message, desired_encoding="bgr8")
                     if compressed else bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
            if frame is None or not frame.size:
                raise ValueError("Empty camera frame")
            with lock:
                latest.update(frame=frame, received=time.monotonic(), error=None)
        except Exception as exc:
            with lock:
                latest["error"] = str(exc)

    subscriber = rospy.Subscriber(args.topic, CompressedImage if compressed else Image,
                                  receive, queue_size=1, buff_size=32 * 1024 * 1024)
    window = "MAiRA camera"
    opened = False
    started = time.monotonic()
    timer = CaptureTimer(args.interval)
    session = uuid4().hex
    try:
        while not rospy.is_shutdown():
            with lock:
                frame, received, error = latest["frame"], latest["received"], latest["error"]
            if frame is None:
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError(f"No image on {args.topic}. Check ROS networking and topic name. "
                                       f"{('Decode error: ' + error) if error else ''}")
                time.sleep(0.05)
                continue
            stale = error is not None or time.monotonic() - received > args.stale_after
            if not opened:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                opened = True
                print(f"Receiving {frame.shape[1]} x {frame.shape[0]} images.", flush=True)
            cv2.imshow(window, display_frame(frame, stale))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            automatic = timer.due(time.monotonic())
            if key in (ord("s"), ord("S")) or automatic:
                if stale:
                    print("No recent valid image; capture skipped.", flush=True)
                else:
                    try:
                        options = dict(item=args.item, label=args.label, plate_id=args.plate_id,
                                       session=session, topic=args.topic)
                        path = (annotate_and_save(frame, window, **options) if args.annotate
                                else save_viewer_capture(frame, **options))
                        print(f"Saved: {path}" if path is not None else "Skipped; nothing saved.", flush=True)
                    except (OSError, RuntimeError) as exc:
                        print(f"Save failed: {exc}", flush=True)
                    finally:
                        if args.annotate:
                            timer.restart(time.monotonic())
    finally:
        subscriber.unregister()
        if opened:
            cv2.destroyAllWindows()
        rospy.signal_shutdown("Camera viewer closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
