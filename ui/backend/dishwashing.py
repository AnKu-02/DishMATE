"""Dishwashing use case: classify each plate, then pick it and place it at the drop point.

Per pick point: observe (open gripper, photo, classify) -> pick (close, wait 2 s) -> lift ->
via -> drop (open, wait 5 s) -> via -> observe. Every plate goes to the same drop point.

Pause holds the robot still before its next move. Stop while running finishes the current
plate (drop and return to the observe pose) so a plate is never left in the gripper; Stop
while paused ends at once and leaves the robot where it is.
"""
import threading
import time
from datetime import datetime

THRESHOLD = 0.5
HISTORY_LIMIT = 50


class Stopped(Exception):
    """Raised inside the worker when the operator ends the run."""


def classify_and_record(frame):
    """Save the photo, classify it in the classifier interpreter and log it to results.csv."""
    from capture_storage import save_capture
    from classifier_client import classify_image
    from prediction_log import log_prediction

    path = save_capture(frame, item="plate")
    result = classify_image(path)
    log_prediction(path, result)
    return {"label": result["label"], "score": result["score"], "image": path.name}


class DishwashingSession:
    def __init__(self, camera, points, arm_factory, classify=classify_and_record, sleep=time.sleep):
        self.camera = camera
        self.points = points
        self.arm_factory = arm_factory
        self.classify = classify
        self.sleep = sleep
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._resume.set()
        self._stop = False       # end at the next point where no plate is held
        self._stop_now = False   # end at the next move, wherever the robot is (Stop while paused)
        self._thread = None
        self._status = "idle"
        self._step = "Ready"
        self._message = ""
        self._cycle = 0
        self._counts = {"clean": 0, "dirty": 0}
        self._history = []

    # ---- operator commands -------------------------------------------------
    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Already running")
            self._stop = self._stop_now = False
            self._resume.set()
            self._status, self._step, self._message, self._cycle = "starting", "Connecting to robot", "", 0
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def pause(self):
        with self._lock:
            if self._status != "running":
                raise RuntimeError("Nothing to pause")
            self._resume.clear()
            self._status = "paused"

    def resume(self):
        with self._lock:
            if self._status != "paused":
                raise RuntimeError("Not paused")
            self._status = "running"
            self._resume.set()

    def stop(self):
        with self._lock:
            if self._status not in ("starting", "running", "paused"):
                raise RuntimeError("Not running")
            self._stop = True
            if self._status == "paused":
                self._stop_now = True
                self._resume.set()  # wake the worker so it can end without moving
            self._status = "stopping"

    def reset_counts(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Reset is available when the robot is not running")
            self._counts = {"clean": 0, "dirty": 0}
            self._history = []

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    # ---- state for the frontend ------------------------------------------
    def state(self):
        with self._lock:
            return {
                "status": self._status,
                "step": self._step,
                "message": self._message,
                "cycle": self._cycle,
                "cycles_total": len(self.points["picks"]),
                "counts": dict(self._counts),
                "latest": self._history[0] if self._history else None,
                "history": list(self._history),
                "threshold": THRESHOLD,
                "camera_online": self.camera.online,
                "sequence": {k: self.points[k] for k in ("observe", "picks", "lift", "via", "drop")},
            }

    # ---- worker ------------------------------------------------------------
    def _set(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, f"_{key}", value)

    def _checkpoint(self, finishing):
        """Block while paused; end the run here unless the current plate must be finished."""
        self._resume.wait()
        with self._lock:
            if self._stop_now or (self._stop and not finishing):
                raise Stopped()

    def _move(self, arm, name, step, finishing=False):
        self._checkpoint(finishing)
        self._set(step=step)
        arm.move(name)

    def _cycle_plate(self, arm, pick):
        p = self.points
        self._move(arm, p["observe"], f"Moving to {p['observe']}")
        arm.open_gripper()
        self._set(step="Taking photo")
        self.sleep(0.5)
        frame = self.camera.fresh_frame(time.monotonic())
        self._set(step="Classifying")
        result = self.classify(frame)
        result.update(time=datetime.now().strftime("%H:%M:%S"), pick=pick)
        with self._lock:
            self._counts[result["label"]] += 1
            self._history = ([result] + self._history)[:HISTORY_LIMIT]
        self._move(arm, pick, f"Picking at {pick}")
        self._checkpoint(finishing=False)  # last chance to stop or pause without a plate in the gripper
        # From here the plate is in the gripper: a Stop request finishes the plate first.
        self._set(step="Gripping plate")
        arm.close_gripper()
        self.sleep(2.0)
        self._move(arm, p["lift"], f"Lifting ({p['lift']})", finishing=True)
        self._move(arm, p["via"], f"Moving to drop-off ({p['via']})", finishing=True)
        self._move(arm, p["drop"], f"Placing at {p['drop']}", finishing=True)
        arm.open_gripper()
        self.sleep(5.0)
        self._move(arm, p["via"], f"Leaving drop-off ({p['via']})", finishing=True)
        self._move(arm, p["observe"], f"Returning to {p['observe']}", finishing=True)

    def _run(self):
        arm = None
        try:
            arm = self.arm_factory()
            self._set(status="running")
            for index, pick in enumerate(self.points["picks"], start=1):
                self._set(cycle=index)
                self._cycle_plate(arm, pick)
                with self._lock:
                    if self._stop:
                        raise Stopped()
            self._set(status="finished", step="All plates done")
        except Stopped:
            with self._lock:
                self._status = "stopped"
                self._message = ("Stopped while paused: the robot was left where it is. Check the gripper "
                                 "before the next start." if self._stop_now else "")
                self._step = "Stopped by operator"
        except Exception as exc:
            self._set(status="error", step="Stopped with an error", message=str(exc))
            print(f"Robot error: {exc}", flush=True)
        finally:
            self._resume.set()
            if arm is not None:
                try:
                    arm.stop()
                except Exception as exc:
                    print(f"Robot stop failed: {exc}", flush=True)
