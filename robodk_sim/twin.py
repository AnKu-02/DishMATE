"""RoboDK digital twin: mirror the real MAiRA from the workcell UI backend into RoboDK.

The RoboDK robot follows the live /joint_states; plates attach when the real gripper closes,
detach when it opens, and return to the rack when a new run starts. RoboDK stays in
simulation mode: nothing is ever sent from RoboDK to the real robot.

Run with RoboDK's Python on Windows while the UI backend is running (run_dishwashing.py
starts everything):
    C:\\RoboDK\\Python-Embedded\\python.exe robodk_sim\\twin.py
"""
import json
import math
import time
from urllib.error import URLError
from urllib.request import urlopen

from robodk.robolink import (Robolink, ITEM_TYPE_FRAME, ITEM_TYPE_PROGRAM_PYTHON, ITEM_TYPE_ROBOT,
                             ITEM_TYPE_TOOL, RUNMODE_SIMULATE)

from build_station import (FINGERS_CLOSED, FINGERS_OPEN, FRAME_NAME, PLATE_DIAMETER, SIM_GRIPPER, TOOL_NAME,
                           find_maira)

TWIN_URL = "http://localhost:8080/api/twin"
PERIOD = 0.05          # 20 updates per second
MIN_CHANGE_DEG = 0.02  # skip redraws while the robot stands still


def fetch(url=TWIN_URL, timeout=1.0):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


class Twin:
    def __init__(self, RDK):
        if RDK.RunMode() != RUNMODE_SIMULATE:
            raise RuntimeError("RoboDK is not in simulation mode; the twin only drives the simulated robot")
        self.RDK = RDK
        self.robot = find_maira(RDK)
        self.tool = RDK.Item(TOOL_NAME, ITEM_TYPE_TOOL)
        self.frame = RDK.Item(FRAME_NAME, ITEM_TYPE_FRAME)
        self.reset = RDK.Item("Reset_Plates", ITEM_TYPE_PROGRAM_PYTHON)
        self.fingers = RDK.Item(SIM_GRIPPER, ITEM_TYPE_ROBOT)  # visual stand-in for the RG6 (optional)
        missing = [name for name, item in (("robot", self.robot), (TOOL_NAME, self.tool), (FRAME_NAME, self.frame),
                                           ("Reset_Plates", self.reset)) if not item.Valid()]
        if missing:
            raise RuntimeError(f"Station is missing {missing}; run build_station.py on it first")
        self.robot.setTool(self.tool)
        # Start from a known scene even if the station was saved mid-run (plates on the table).
        self.tool.DetachAll(self.frame)
        self.reset.RunProgram()
        self.last_joints = None
        self.gripper = None
        self.run_id = None

    def update(self, state):
        joints = state.get("joints_rad")
        if joints:
            degrees = [math.degrees(a) for a in joints]
            if self.last_joints is None or max(abs(a - b) for a, b in zip(degrees, self.last_joints)) > MIN_CHANGE_DEG:
                self.robot.setJoints(degrees)
                self.last_joints = degrees
        if self.run_id is not None and state["run_id"] != self.run_id:
            self.tool.DetachAll(self.frame)
            self.reset.RunProgram()
            print("New run: plates back in the rack", flush=True)
        self.run_id = state["run_id"]
        if state["gripper"] != self.gripper:
            if self.fingers.Valid() and state["gripper"] in ("open", "closed"):
                self.fingers.setJoints([FINGERS_CLOSED if state["gripper"] == "closed" else FINGERS_OPEN])
            if state["gripper"] == "closed":
                plate = self.tool.AttachClosest(keyword="Plate", tolerance_mm=PLATE_DIAMETER / 2)
                print(f"Gripper closed: holding {plate.Name() if plate.Valid() else 'nothing'}", flush=True)
            elif state["gripper"] == "open" and self.gripper == "closed":
                self.tool.DetachAll(self.frame)
                print("Gripper opened: plate released", flush=True)
            self.gripper = state["gripper"]


def run(RDK, url=TWIN_URL, stop=lambda: False):
    twin = Twin(RDK)
    print(f"Twin running: RoboDK '{RDK.ActiveStation().Name()}' mirrors {url}", flush=True)
    offline = False
    while not stop():
        started = time.monotonic()
        try:
            twin.update(fetch(url))
            if offline:
                print("Backend reachable again", flush=True)
            offline = False
        except (URLError, OSError, ValueError):
            if not offline:
                print("Waiting for the UI backend...", flush=True)
            offline = True
        time.sleep(max(0.0, PERIOD - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        run(Robolink())
    except KeyboardInterrupt:
        pass
