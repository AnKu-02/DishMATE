import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ui.backend.dishwashing import DishwashingSession
from ui.backend.server import make_handler

POINTS = {"observe": "OBS", "picks": ["P1", "P2"], "lift": "LIFT", "via": "VIA", "drop": "DROP"}


class FakeFrame:
    def copy(self):
        return self


class FakeCamera:
    online = True

    def fresh_frame(self, after, timeout=10.0):
        return FakeFrame()

    def jpeg(self):
        return b"\xff\xd8fake\xff\xd9"


class FakeArm:
    def __init__(self, gate=None):
        self.calls = []
        self.gate = gate  # when set, each move waits for the test to release it

    def move(self, name):
        if self.gate is not None:
            self.gate.acquire()
        self.calls.append(name)

    def open_gripper(self):
        self.calls.append("open")

    def close_gripper(self):
        self.calls.append("close")

    def stop(self):
        self.calls.append("stop")


def scores(*values):
    it = iter(values)

    def classify(frame):
        score = next(it)
        return {"label": "dirty" if score > 0.5 else "clean", "score": score, "image": "plate.png"}
    return classify


def session(arm, classify=None):
    return DishwashingSession(FakeCamera(), POINTS, lambda: arm, classify or scores(0.2, 0.8),
                              sleep=lambda s: None)


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


CYCLE = ["OBS", "open", "{pick}", "close", "LIFT", "VIA", "DROP", "open", "VIA", "OBS"]


def cycle(pick):
    return [step.format(pick=pick) for step in CYCLE]


class DishwashingSessionTest(unittest.TestCase):
    def test_full_run_moves_every_plate_and_counts_results(self):
        arm = FakeArm()
        s = session(arm)
        s.start()
        s.join(2)
        state = s.state()
        self.assertEqual(state["status"], "finished")
        self.assertEqual(arm.calls, cycle("P1") + cycle("P2") + ["stop"])
        self.assertEqual(state["counts"], {"clean": 1, "dirty": 1})
        self.assertEqual([h["pick"] for h in state["history"]], ["P2", "P1"])
        self.assertEqual(state["latest"]["label"], "dirty")

    def test_pause_holds_before_next_move_and_resume_continues(self):
        gate = threading.Semaphore(0)
        arm = FakeArm(gate)
        s = session(arm)
        s.start()
        gate.release()  # first move to OBS
        self.assertTrue(wait_for(lambda: arm.calls == ["OBS", "open"] and s.state()["step"] == f"Picking at P1"))
        s.pause()
        gate.release()  # the move already under way finishes
        self.assertTrue(wait_for(lambda: arm.calls[-1] == "P1"))
        for _ in range(20):
            gate.release()
        time.sleep(0.1)
        self.assertEqual(arm.calls[-1], "P1")  # held before the grasp
        self.assertEqual(s.state()["status"], "paused")
        s.resume()
        s.join(2)
        self.assertEqual(s.state()["status"], "finished")

    def test_stop_while_running_finishes_the_plate_in_the_gripper(self):
        gate = threading.Semaphore(0)
        arm = FakeArm(gate)
        s = session(arm)
        s.start()
        gate.release()
        gate.release()
        self.assertTrue(wait_for(lambda: "close" in arm.calls))
        s.stop()
        for _ in range(20):
            gate.release()
        s.join(2)
        self.assertEqual(s.state()["status"], "stopped")
        self.assertEqual(arm.calls, cycle("P1") + ["stop"])  # P2 never started

    def test_stop_before_the_grasp_ends_without_picking(self):
        gate = threading.Semaphore(0)
        arm = FakeArm(gate)
        s = session(arm)
        s.start()
        gate.release()
        self.assertTrue(wait_for(lambda: arm.calls == ["OBS", "open"]))
        s.stop()
        for _ in range(20):
            gate.release()
        s.join(2)
        self.assertNotIn("close", arm.calls)
        self.assertEqual(s.state()["status"], "stopped")

    def test_stop_while_paused_ends_without_moving(self):
        gate = threading.Semaphore(0)
        arm = FakeArm(gate)
        s = session(arm)
        s.start()
        gate.release()  # OBS; the worker then waits inside the move to P1
        self.assertTrue(wait_for(lambda: s.state()["step"] == "Picking at P1"))
        s.pause()
        gate.release()  # P1 arrives, then the worker holds before the grasp
        self.assertTrue(wait_for(lambda: arm.calls[-1] == "P1"))
        s.stop()
        for _ in range(20):
            gate.release()
        s.join(2)
        state = s.state()
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(arm.calls, ["OBS", "open", "P1", "stop"])  # no grasp, no further move
        self.assertIn("left where it is", state["message"])

    def test_stop_while_paused_with_a_plate_in_the_gripper_ends_without_moving(self):
        gate = threading.Semaphore(0)
        arm = FakeArm(gate)
        s = session(arm)
        s.start()
        gate.release()
        gate.release()
        self.assertTrue(wait_for(lambda: s.state()["step"] == "Lifting (LIFT)"))  # worker waits inside LIFT
        s.pause()
        gate.release()  # LIFT arrives; the worker holds before VIA
        self.assertTrue(wait_for(lambda: arm.calls[-1] == "LIFT"))
        s.stop()
        for _ in range(20):
            gate.release()
        s.join(2)
        self.assertEqual(arm.calls[-2:], ["LIFT", "stop"])
        self.assertEqual(s.state()["status"], "stopped")

    def test_robot_error_is_reported(self):
        class BrokenArm(FakeArm):
            def move(self, name):
                raise RuntimeError("Automatic Robot movement is possible only in Automatic/SemiAutomatic modes")
        s = session(BrokenArm())
        s.start()
        s.join(2)
        state = s.state()
        self.assertEqual(state["status"], "error")
        self.assertIn("Automatic", state["message"])

    def test_commands_are_rejected_in_the_wrong_state(self):
        s = session(FakeArm())
        for command in (s.pause, s.resume, s.stop):
            with self.assertRaises(RuntimeError):
                command()


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.session = session(FakeArm())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeCamera(), {"dishwashing": self.session}))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def get(self, path):
        with urlopen(self.base + path, timeout=5) as response:
            return response.status, response.headers.get("Content-Type"), response.read()

    def post(self, path):
        with urlopen(Request(self.base + path, method="POST"), timeout=5) as response:
            return json.loads(response.read())

    def test_frontend_and_use_cases(self):
        status, kind, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", kind)
        self.assertIn(b"MAiRA Workcell", body)
        self.assertEqual(self.get("/app.js")[0], 200)
        use_cases = json.loads(self.get("/api/usecases")[2])
        self.assertEqual([u["id"] for u in use_cases], ["dishwashing"])

    def test_static_files_cannot_escape_the_frontend_folder(self):
        with self.assertRaises(HTTPError) as error:
            self.get("/../backend/server.py")
        self.assertEqual(error.exception.code, 404)

    def test_start_runs_and_state_reports_results(self):
        self.post("/api/dishwashing/start")
        self.session.join(2)
        state = json.loads(self.get("/api/dishwashing/state")[2])
        self.assertEqual(state["status"], "finished")
        self.assertEqual(state["counts"]["clean"] + state["counts"]["dirty"], 2)

    def test_invalid_command_returns_conflict(self):
        with self.assertRaises(HTTPError) as error:
            self.post("/api/dishwashing/pause")
        self.assertEqual(error.exception.code, 409)

    def test_camera_stream_sends_jpeg_parts(self):
        response = urlopen(self.base + "/api/camera/stream.mjpg", timeout=5)
        self.assertIn("multipart/x-mixed-replace", response.headers.get("Content-Type"))
        self.assertIn(b"Content-Type: image/jpeg", response.read(200))
        response.close()


if __name__ == "__main__":
    unittest.main()
