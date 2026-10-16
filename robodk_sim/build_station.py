"""Build the real dishwashing pick-and-place in a RoboDK station, in the NEURA robot base frame.

Uses robodk_sim/gp_points.json (from export_gp_points.py): joint angles, flange and TCP poses
of the taught GP points, read from the real MAiRA. Everything is placed under a frame that
coincides with the robot base frame, so RoboDK coordinates equal NEURA coordinates.

Creates (and replaces on re-run):
  * tool "RG6 (real TCP)" with the controller's flange->TCP offset
  * joint targets GP156, GP157, GP163, GP161, GP165, GP164 (real joint values)
  * a dish rack with two plates at the real grasp poses, and a table at the drop-off
  * scripts Gripper_Open / Gripper_Close / Take_Photo / Reset_Plates
  * program "Dishwashing_Sim": per plate GP156 (open, photo) -> pick (close, 2 s) -> GP161 ->
    GP165 -> GP164 (open, 5 s) -> GP165 -> GP156

Run either way:
  * inside RoboDK, with the station open: Tools > Run Script (or drag this file into RoboDK);
    then save the station.
  * from a terminal, with no other RoboDK open:
        C:\\RoboDK\\Python-Embedded\\python.exe robodk_sim\\build_station.py IN.rdk OUT.rdk
"""
import json
import math
import os
import sys

from robodk.robolink import (Robolink, ITEM_TYPE_FRAME, ITEM_TYPE_PROGRAM, ITEM_TYPE_PROGRAM_PYTHON,
                             ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, INSTRUCTION_CALL_PROGRAM)
from robodk.robomath import Mat, invH, pose_2_xyzrpw, transl, xyzrpw_2_pose

HERE = os.path.dirname(os.path.abspath(__file__))
POINTS_FILE = os.path.join(HERE, "gp_points.json")
GENERATED = os.path.join(HERE, "generated")

FRAME_NAME = "Dishwashing (NEURA base)"
TOOL_NAME = "RG6 (real TCP)"
PROGRAM_NAME = "Dishwashing_Sim"
SCRIPTS = ("Gripper_Open", "Gripper_Close", "Take_Photo", "Reset_Plates")

# Visual gripper: RoboDK has no RG6 model, so the station's OnRobot 2FG7 stands in for it. Its
# fingers open/close with the real gripper; plates are still held at the real RG6 TCP (180 mm).
SIM_GRIPPER = "OnRobot 2FG7 Gripper"
HIDDEN_GRIPPERS = ("RobotiQ 2F-85 Gripper (Mechanism)",)
FINGERS_OPEN = 55.0
FINGERS_CLOSED = 5.0

# Plate model: paper plate held by its top rim, fingers closing along tool Y (measured: at both
# pick poses tool Y is parallel to the 90 mm spacing between the plates).
PLATE_DIAMETER = 230.0
PLATE_THICKNESS = 6.0
GRIP_DEPTH = 20.0          # how far the rim sits inside the fingers
RACK_SIZE = (320.0, 420.0, 110.0)   # along plate spacing, along plate plane, height (mm)
TABLE_SIZE = (700.0, 600.0, 40.0)


# ---------------------------------------------------------------- geometry helpers
def neura_pose(values):
    """[x, y, z, a, b, c] in m/rad (NEURA RPY: Rz(c) Ry(b) Rx(a)) -> RoboDK pose in mm."""
    x, y, z, a, b, c = values
    return xyzrpw_2_pose([x * 1000, y * 1000, z * 1000, math.degrees(a), math.degrees(b), math.degrees(c)])


def column(pose, i):
    return [pose[0, i], pose[1, i], pose[2, i]]


def plate_center_pose(tcp_pose):
    """Plate hangs below the grasp point along tool Z; its plane is the tool XZ plane."""
    return tcp_pose * transl(0, 0, PLATE_DIAMETER / 2 - GRIP_DEPTH)


def lowest_z(center_pose):
    """Lowest point of the plate disc (plane normal = local Y)."""
    n = column(center_pose, 1)
    horizontal = math.sqrt(max(0.0, 1.0 - n[2] ** 2))  # length of -Z projected into the plate plane
    return center_pose[2, 3] - PLATE_DIAMETER / 2 * horizontal


def disc_triangles(diameter, thickness, segments=48):
    """Closed disc centred on the origin, plane XZ, normal Y."""
    r, h = diameter / 2, thickness / 2
    ring = [(r * math.cos(2 * math.pi * k / segments), r * math.sin(2 * math.pi * k / segments))
            for k in range(segments)]
    tris = []
    for k in range(segments):
        (x0, z0), (x1, z1) = ring[k], ring[(k + 1) % segments]
        tris += [[0, h, 0], [x1, h, z1], [x0, h, z0]]          # top face
        tris += [[0, -h, 0], [x0, -h, z0], [x1, -h, z1]]       # bottom face
        tris += [[x0, h, z0], [x1, h, z1], [x1, -h, z1]]       # rim
        tris += [[x0, h, z0], [x1, -h, z1], [x0, -h, z0]]
    return tris


def box_triangles(sx, sy, sz):
    """Box centred in X/Y with its bottom at z=0."""
    x, y = sx / 2, sy / 2
    v = [[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], [-x, -y, sz], [x, -y, sz], [x, y, sz], [-x, y, sz]]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [v[i] for face in faces for i in face]


def add_shape(RDK, triangles, frame):
    """AddShape's second argument merges into an existing object, so add first, then re-parent."""
    item = RDK.AddShape(to_mat(triangles))
    item.setParent(frame)
    return item


def to_mat(triangles):
    """AddShape converts a plain list of [x, y, z] vertices itself (a prebuilt Mat is rejected)."""
    return [[float(v) for v in point] for point in triangles]


# ---------------------------------------------------------------- station building
def find_maira(RDK):
    """The 7-axis arm; gripper mechanisms are robot items too, so pick by axis count."""
    for item in RDK.ItemList(filter=ITEM_TYPE_ROBOT):
        if len(item.Joints().list()) == 7:
            return item
    raise RuntimeError("No 7-axis MAiRA in the station")


def remove_previous(RDK):
    for name, kind in [(FRAME_NAME, ITEM_TYPE_FRAME), (PROGRAM_NAME, ITEM_TYPE_PROGRAM), (TOOL_NAME, ITEM_TYPE_TOOL)]:
        item = RDK.Item(name, kind)
        while item.Valid():
            item.Delete()
            item = RDK.Item(name, kind)
    for name in SCRIPTS:
        item = RDK.Item(name, ITEM_TYPE_PROGRAM_PYTHON)
        while item.Valid():
            item.Delete()
            item = RDK.Item(name, ITEM_TYPE_PROGRAM_PYTHON)


def check_frames(robot, points):
    """Compare RoboDK's flange FK with the controller's for every point."""
    worst_mm, worst_deg = 0.0, 0.0
    for name, p in points.items():
        rdk = robot.SolveFK(p["joints_deg"])
        real = neura_pose(p["flange_m_rad"])
        diff = invH(real) * rdk
        worst_mm = max(worst_mm, math.sqrt(sum(diff[i, 3] ** 2 for i in range(3))))
        trace = diff[0, 0] + diff[1, 1] + diff[2, 2]
        worst_deg = max(worst_deg, math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))))
    return worst_mm, worst_deg


def write_scripts(tool_name, plate_homes):
    os.makedirs(GENERATED, exist_ok=True)
    homes = {name: [round(v, 4) for v in pose_2_xyzrpw(pose)] for name, pose in plate_homes.items()}
    sources = {
        "Gripper_Close": f'''from robodk.robolink import Robolink, ITEM_TYPE_TOOL, ITEM_TYPE_ROBOT
RDK = Robolink()
fingers = RDK.Item({SIM_GRIPPER!r}, ITEM_TYPE_ROBOT)
if fingers.Valid():
    fingers.MoveJ([{FINGERS_CLOSED}])
tool = RDK.Item({tool_name!r}, ITEM_TYPE_TOOL)
tool.AttachClosest(keyword="Plate", tolerance_mm={PLATE_DIAMETER / 2:.0f})
''',
        "Gripper_Open": f'''from robodk.robolink import Robolink, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_ROBOT
RDK = Robolink()
tool = RDK.Item({tool_name!r}, ITEM_TYPE_TOOL)
tool.DetachAll(RDK.Item({FRAME_NAME!r}, ITEM_TYPE_FRAME))
fingers = RDK.Item({SIM_GRIPPER!r}, ITEM_TYPE_ROBOT)
if fingers.Valid():
    fingers.MoveJ([{FINGERS_OPEN}])
''',
        "Take_Photo": '''from robodk.robolink import Robolink
RDK = Robolink()
RDK.ShowMessage("Camera: photo taken, plate classified (clean/dirty)", False)
''',
        "Reset_Plates": f'''from robodk.robolink import Robolink, ITEM_TYPE_OBJECT, ITEM_TYPE_FRAME, ITEM_TYPE_TOOL
from robodk.robomath import xyzrpw_2_pose
RDK = Robolink()
frame = RDK.Item({FRAME_NAME!r}, ITEM_TYPE_FRAME)
RDK.Item({tool_name!r}, ITEM_TYPE_TOOL).DetachAll(frame)
for name, xyzrpw in {homes!r}.items():
    plate = RDK.Item(name, ITEM_TYPE_OBJECT)
    plate.setParentStatic(frame)
    plate.setPose(xyzrpw_2_pose(xyzrpw))
''',
    }
    paths = {}
    for name, code in sources.items():
        path = os.path.join(GENERATED, name + ".py")
        with open(path, "w") as handle:
            handle.write(code)
        paths[name] = path
    return paths


def build(RDK):
    with open(POINTS_FILE) as handle:
        data = json.load(handle)
    points = data["points"]
    seq = data["sequence"]

    robot = find_maira(RDK)
    print(f"Robot: {robot.Name()}")

    worst_mm, worst_deg = check_frames(robot, points)
    print(f"Frame check (RoboDK FK vs NEURA controller FK): max {worst_mm:.2f} mm, {worst_deg:.3f} deg")
    if worst_mm > 10 or worst_deg > 1:
        print("WARNING: the RoboDK robot model differs from the real robot by more than 10 mm.")
    else:
        print("OK: RoboDK's nominal MAiRA model matches the calibrated controller within a few mm.")

    remove_previous(RDK)
    base = robot.Parent()
    frame = RDK.AddFrame(FRAME_NAME, base)
    frame.setPose(transl(0, 0, 0))   # identical to the robot base = NEURA base frame

    live = data["live_sample"]
    tool_pose = invH(neura_pose(live["flange_m_rad"])) * neura_pose(live["tcp_m_rad"])
    tool = robot.AddTool(tool_pose, TOOL_NAME)
    robot.setTool(tool)
    for name in HIDDEN_GRIPPERS:
        hidden = RDK.Item(name, ITEM_TYPE_ROBOT)
        if hidden.Valid():
            hidden.setVisible(False)
    fingers = RDK.Item(SIM_GRIPPER, ITEM_TYPE_ROBOT)
    if fingers.Valid():
        fingers.setVisible(True)
        fingers.setJoints([FINGERS_OPEN])
        print(f"Visual gripper: {SIM_GRIPPER} (hidden: {', '.join(HIDDEN_GRIPPERS)})")
    else:
        print(f"Note: no {SIM_GRIPPER!r} in the station; plates still attach to {TOOL_NAME}")
    print(f"Tool {TOOL_NAME}: TCP {[round(v, 1) for v in pose_2_xyzrpw(tool_pose)]} (mm, deg)")

    targets = {}
    for name in [seq["observe"], *seq["picks"], seq["lift"], seq["via"], seq["drop"]]:
        target = RDK.AddTarget(name, frame, robot)
        target.setAsJointTarget()
        target.setJoints(points[name]["joints_deg"])
        targets[name] = target

    # Plates at the real grasp poses, rack around them, table under the drop-off.
    plate_mesh = disc_triangles(PLATE_DIAMETER, PLATE_THICKNESS)
    plate_homes = {}
    for i, pick in enumerate(seq["picks"], start=1):
        plate = add_shape(RDK, plate_mesh, frame)
        plate.setName(f"Plate {i}")
        plate.setColor([0.96, 0.95, 0.9, 1.0])
        plate_homes[plate.Name()] = plate_center_pose(neura_pose(points[pick]["tcp_m_rad"]))
        plate.setPose(plate_homes[plate.Name()])

    centers = list(plate_homes.values())
    cx = sum(p[0, 3] for p in centers) / len(centers)
    cy = sum(p[1, 3] for p in centers) / len(centers)
    rack_bottom = min(lowest_z(p) for p in centers) - 5
    spacing = column(centers[0], 1)                       # plate normal = row direction of the rack
    yaw = math.degrees(math.atan2(spacing[1], spacing[0]))
    rack = add_shape(RDK, box_triangles(*RACK_SIZE), frame)
    rack.setName("Dish rack")
    rack.setColor([0.45, 0.47, 0.5, 0.45])
    rack.setPose(xyzrpw_2_pose([cx, cy, rack_bottom, 0, 0, yaw]))

    drop_center = plate_center_pose(neura_pose(points[seq["drop"]]["tcp_m_rad"]))
    table_top = lowest_z(drop_center)
    table = add_shape(RDK, box_triangles(*TABLE_SIZE), frame)
    table.setName("Table")
    table.setColor([0.62, 0.48, 0.34, 1.0])
    table.setPose(xyzrpw_2_pose([drop_center[0, 3], drop_center[1, 3], table_top - TABLE_SIZE[2], 0, 0, 0]))
    print(f"Rack bottom z={rack_bottom:.0f} mm, table top z={table_top:.0f} mm (robot base frame)")

    for name, path in write_scripts(TOOL_NAME, plate_homes).items():
        RDK.AddFile(path)

    prog = RDK.AddProgram(PROGRAM_NAME, robot)
    prog.setPoseFrame(frame)
    prog.setPoseTool(tool)
    prog.RunInstruction("Reset_Plates", INSTRUCTION_CALL_PROGRAM)
    for pick in seq["picks"]:
        prog.MoveJ(targets[seq["observe"]])
        prog.RunInstruction("Gripper_Open", INSTRUCTION_CALL_PROGRAM)
        prog.RunInstruction("Take_Photo", INSTRUCTION_CALL_PROGRAM)
        prog.MoveJ(targets[pick])
        prog.RunInstruction("Gripper_Close", INSTRUCTION_CALL_PROGRAM)
        prog.Pause(2000)
        for name in (seq["lift"], seq["via"], seq["drop"]):
            prog.MoveJ(targets[name])
        prog.RunInstruction("Gripper_Open", INSTRUCTION_CALL_PROGRAM)
        prog.Pause(5000)
        prog.MoveJ(targets[seq["via"]])
        prog.MoveJ(targets[seq["observe"]])
    print(f"Program {PROGRAM_NAME}: {prog.InstructionCount()} instructions")
    return prog


def main():
    if len(sys.argv) == 3:
        RDK = Robolink(args=["-NOSPLASH", "-NOSHOW"], port=20610)
        RDK.AddFile(os.path.abspath(sys.argv[1]))
        build(RDK)
        RDK.Save(os.path.abspath(sys.argv[2]))
        print(f"Saved {sys.argv[2]}")
        RDK.CloseRoboDK()
    else:
        build(Robolink())


if __name__ == "__main__":
    main()
