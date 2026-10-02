"""Live ROS camera viewer. No classifier or robot motion clients are loaded."""
import argparse
import math
import os
import sys
import time
from threading import Lock

from capture_storage import save_capture
from settings import configure_ros, load_settings


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
    parser.add_argument("--timeout", type=positive_seconds, default=10,
                        help="seconds to wait for the first image after ROS initialization")
    parser.add_argument("--stale-after", type=positive_seconds, default=2,
                        help="disable saving if no new frame arrives for this many seconds")
    args = parser.parse_args(argv)
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
    print("S saves a clean image; Q/Esc or closing the window exits. Ctrl+C also exits.", flush=True)
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
            if key in (ord("s"), ord("S")):
                if stale:
                    print("No recent valid image; capture skipped.", flush=True)
                else:
                    try:
                        print(f"Saved: {save_capture(frame, item=args.item)}", flush=True)
                    except (OSError, RuntimeError) as exc:
                        print(f"Save failed: {exc}", flush=True)
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
