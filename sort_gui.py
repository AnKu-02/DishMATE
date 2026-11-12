"""Plate sorting GUI: live camera, clean/dirty status light and counts.

With --robot, Space runs one sorting cycle per pick point in settings.json:
observe point (open gripper, photo, classify) -> pick (close, wait 2 s) -> lift ->
clean_place or dirty_place (open, wait 5 s) -> observe point.
Without --robot, C classifies the current camera frame and nothing moves.
"""
import argparse
import os
import sys
import threading
import time

from capture_storage import save_capture
from settings import configure_ros, load_settings

PANEL_WIDTH = 340
GREEN, RED, GREY, AMBER = (60, 180, 75), (40, 40, 220), (120, 120, 120), (0, 190, 255)


class SortState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.received = 0.0
        self.label = None
        self.score = None
        self.counts = {"clean": 0, "dirty": 0}
        self.step = "Idle"
        self.busy = False
        self.stop_requested = False

    def update(self, **values):
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self):
        with self.lock:
            return (self.frame, self.label, self.score, dict(self.counts), self.step, self.busy)


def fresh_frame(state, after, timeout=10.0):
    """Wait for a frame received after `after` so the photo is not motion-blurred."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with state.lock:
            if state.frame is not None and state.received > after:
                return state.frame.copy()
        time.sleep(0.05)
    raise RuntimeError("No new camera frame after the robot stopped")


def classify_frame(state, frame):
    from classifier_client import classify_image
    from prediction_log import log_prediction

    state.update(step="Classifying...", label=None, score=None)
    path = save_capture(frame, item="plate")
    result = classify_image(path)
    log_prediction(path, result)
    with state.lock:
        state.label, state.score = result["label"], result["score"]
        state.counts[result["label"]] += 1
    print(f"{path.name}: {result['label']} ({result['score']:.3f})", flush=True)
    return result["label"]


def run_cycles(state, arm, points):
    try:
        for pick in points["picks"]:
            if state.stop_requested:
                break
            state.update(step=f"Moving to {points['observe']}")
            arm.move(points["observe"])
            arm.open_gripper()
            time.sleep(0.5)
            label = classify_frame(state, fresh_frame(state, time.monotonic()))
            place = points[f"{label}_place"]
            for step, action in (
                (f"Picking at {pick}", lambda: arm.move(pick)),
                ("Closing gripper", lambda: (arm.close_gripper(), time.sleep(2.0))),
                (f"Lifting ({points['lift']})", lambda: arm.move(points["lift"])),
                (f"Placing {label} plate at {place}", lambda: arm.move(place)),
                ("Releasing", lambda: (arm.open_gripper(), time.sleep(5.0))),
                (f"Returning to {points['observe']}", lambda: arm.move(points["observe"])),
            ):
                state.update(step=step)
                action()
        state.update(step="Stopped by user" if state.stop_requested else "Cycle complete")
    except Exception as exc:
        state.update(step=f"ERROR: {exc}")
        print(f"Robot error: {exc}", flush=True)
    finally:
        state.update(busy=False)


def subscribe_camera(state, topic):
    """Keep state.frame updated from the ROS camera; requires rospy.init_node() first."""
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    bridge = CvBridge()

    def receive(message):
        try:
            state.update(frame=bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"),
                         received=time.monotonic())
        except Exception as exc:
            print(f"Camera decode error: {exc}", flush=True)

    return rospy.Subscriber(topic, Image, receive, queue_size=1, buff_size=32 * 1024 * 1024)


def render(state, robot_mode, keys=None):
    import cv2
    import numpy as np

    frame, label, score, counts, step, busy = state.snapshot()
    height = max(480, frame.shape[0] if frame is not None else 0)
    camera = frame if frame is not None else np.zeros((480, 640, 3), np.uint8)
    canvas = np.full((height, camera.shape[1] + PANEL_WIDTH, 3), 30, np.uint8)
    canvas[:camera.shape[0], :camera.shape[1]] = camera
    x = camera.shape[1] + 20

    def text(y, value, scale=0.6, color=(235, 235, 235), thickness=1):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    text(35, "Plate sorter", 0.8, thickness=2)
    light = AMBER if busy and label is None else {"clean": GREEN, "dirty": RED}.get(label, GREY)
    cv2.circle(canvas, (x + 60, 120), 55, light, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 60, 120), 55, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, (label or "--").upper(), (x + 135, 130), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, light, 2, cv2.LINE_AA)
    if score is not None:
        text(200, f"Dirty score: {score:.3f}")
    text(250, "Mixed rack count", 0.65, thickness=2)
    text(285, f"Clean: {counts['clean']}", 0.7, GREEN, 2)
    text(320, f"Dirty: {counts['dirty']}", 0.7, RED, 2)
    text(355, f"Total: {counts['clean'] + counts['dirty']}", 0.7)
    for i, line in enumerate([step[j:j + 30] for j in range(0, len(step), 30)][:3]):
        text(400 + 22 * i, line, 0.5, AMBER if busy else (200, 200, 200))
    keys = keys or ("Space: start  R: reset  Q: quit" if robot_mode else "C: classify  R: reset  Q: quit")
    text(height - 15, keys, 0.5, (170, 170, 170))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", action="store_true", help="enable robot sorting cycles (Space)")
    parser.add_argument("--speed", type=float, default=20.0, help="joint speed/acceleration percent")
    args = parser.parse_args()
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("No DISPLAY is configured. Open a WSL terminal with GUI/WSLg support.")

    settings = load_settings()
    points = settings["sort_points"]
    if args.robot:
        missing = [k for k in ("observe", "lift", "clean_place", "dirty_place") if not points.get(k)]
        if missing or not points.get("picks"):
            raise RuntimeError(f"Set sort_points {missing or ['picks']} in settings.json before using --robot")

    import cv2
    import rospy

    configure_ros()
    rospy.init_node("plate_sort_gui", anonymous=True, disable_signals=True)
    state = SortState()
    subscriber = subscribe_camera(state, settings["camera_topic"])
    arm = None
    if args.robot:
        from robot_motion import MairaArm
        names = {points["observe"], points["lift"], points["clean_place"], points["dirty_place"], *points["picks"]}
        arm = MairaArm(sorted(names), speed=args.speed)
        print(f"Robot ready. Points: {', '.join(sorted(names))}", flush=True)

    window = "Plate sorter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    worker = None
    try:
        while True:
            cv2.imshow(window, render(state, args.robot))
            key = cv2.waitKey(30) & 0xFF
            closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), ord("Q"), 27) or closed:
                if worker is not None and worker.is_alive():
                    state.update(stop_requested=True, step="Stopping after this cycle...")
                    if closed:
                        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    continue
                break
            if state.busy:
                continue
            if key in (ord("r"), ord("R")):
                state.update(counts={"clean": 0, "dirty": 0}, label=None, score=None, step="Counts reset")
            elif key == ord(" ") and args.robot:
                state.update(busy=True, stop_requested=False, label=None, score=None)
                worker = threading.Thread(target=run_cycles, args=(state, arm, points), daemon=True)
                worker.start()
            elif key in (ord("c"), ord("C")) and not args.robot:
                frame = state.snapshot()[0]
                if frame is None:
                    state.update(step="No camera frame yet")
                    continue
                state.update(busy=True)

                def classify_only(image=frame.copy()):
                    try:
                        classify_frame(state, image)
                        state.update(step="Classified")
                    except Exception as exc:
                        state.update(step=f"ERROR: {exc}")
                    finally:
                        state.update(busy=False)

                worker = threading.Thread(target=classify_only, daemon=True)
                worker.start()
    finally:
        subscriber.unregister()
        if arm is not None:
            arm.stop()
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Plate sorter closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
