"""Pick-and-place cycles with classification; every plate goes to one drop point (no sorting yet).

For each pick point: observe (open gripper, photo, classify) -> pick (close, wait 2 s)
-> lift -> via -> drop (open, wait 5 s) -> via -> observe.

Run from the project folder with the ROS environment active (source ./activate_wsl.sh):
    python pick_cycles.py              # moves the robot (Automatic mode, e-stop in reach)
    python pick_cycles.py --gui        # same, with the live sorting window (Q stops after this cycle)
    python pick_cycles.py --dry-run    # print planned joint turns only
"""
import argparse
import math
import os
import sys
import threading
import time

from capture_storage import save_capture
from main_real import classify_and_log, get_ros_frame
from robot_motion import plan_target
from settings import configure_ros, load_settings


def run_cycles(arm, args, settings, state=None):
    """Robot sequence; with a GUI state, report steps and results to the window."""
    def step(text):
        print(text, flush=True)
        if state is not None:
            state.update(step=text)

    for pick in args.picks:
        if state is not None and state.stop_requested:
            step("Stopped by user")
            return
        print(f"=== Cycle: pick at {pick} ===", flush=True)
        step(f"Moving to {args.observe}")
        arm.move(args.observe)
        arm.open_gripper()
        if state is None:
            path = save_capture(get_ros_frame(settings["camera_topic"], 10.0), item="plate")
            print(f"Saved: {path}", flush=True)
            classify_and_log(path)
        else:
            from sort_gui import classify_frame, fresh_frame
            time.sleep(0.5)
            classify_frame(state, fresh_frame(state, time.monotonic()))
        step(f"Picking at {pick}")
        arm.move(pick)
        arm.close_gripper()
        time.sleep(2.0)
        step(f"Lifting ({args.lift})")
        arm.move(args.lift)
        step(f"Placing at {args.drop}")
        arm.move(args.via)
        arm.move(args.drop)
        arm.open_gripper()
        time.sleep(5.0)
        step(f"Returning to {args.observe}")
        arm.move(args.via)
        arm.move(args.observe)
    step("Sequence complete")


def run_with_gui(arm, args, settings):
    import cv2
    from sort_gui import SortState, render, subscribe_camera

    state = SortState()
    state.update(busy=True)
    subscriber = subscribe_camera(state, settings["camera_topic"])

    def worker():
        try:
            run_cycles(arm, args, settings, state)
        except Exception as exc:
            state.update(step=f"ERROR: {exc}")
            print(f"Robot error: {exc}", flush=True)
        finally:
            state.update(busy=False)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    window = "Plate sorter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            keys = "Q: stop after this cycle" if thread.is_alive() else "Q: close window"
            cv2.imshow(window, render(state, True, keys))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                if not thread.is_alive():
                    break
                state.update(stop_requested=True)
    finally:
        thread.join()
        subscriber.unregister()
        cv2.destroyAllWindows()


def main():
    points = load_settings()["sort_points"]  # defaults shared with the web UI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observe", default=points["observe"])
    parser.add_argument("--picks", nargs="+", default=points["picks"])
    parser.add_argument("--lift", default=points["lift"])
    parser.add_argument("--drop", default=points["drop"])
    parser.add_argument("--via", default=points["via"], help="waypoint before and after the drop point")
    parser.add_argument("--speed", type=float, default=20.0, help="joint speed/acceleration percent")
    parser.add_argument("--gui", action="store_true", help="show the live sorting window while running")
    parser.add_argument("--dry-run", action="store_true", help="plan only; the robot does not move")
    args = parser.parse_args()
    if args.gui and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("No DISPLAY is configured. Open a WSL terminal with GUI/WSLg support.")

    import rospy
    from robot_motion import MairaArm

    settings = configure_ros()
    rospy.init_node("pick_cycles", anonymous=True, disable_signals=True)
    arm = MairaArm(sorted({args.observe, args.lift, args.via, args.drop, *args.picks}), speed=args.speed)

    if args.dry_run:
        pose = arm.robot.get_current_joint_angles()
        route = [n for pick in args.picks for n in (args.observe, pick, args.lift, args.via, args.drop, args.via)] + [args.observe]
        for name in route:
            pose, turns = plan_target(arm.points[name], pose)
            print(f"{name}: joint turns (deg) " + ", ".join(f"{math.degrees(t):.0f}" for t in turns))
        return

    try:
        if args.gui:
            run_with_gui(arm, args, settings)
        else:
            run_cycles(arm, args, settings)
    finally:
        arm.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
