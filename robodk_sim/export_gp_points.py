"""Export the taught GP points from the real MAiRA for the RoboDK station. Read-only: no motion.

For every point it saves the joint angles from the robot's point database and the TCP pose
the robot's own forward kinematics gives for them (robot base frame, active tool). Using
these in RoboDK keeps the simulation in the same coordinate frame as the real robot.

Run in the ROS environment with the robot reachable (source ./activate_wsl.sh):
    python robodk_sim/export_gp_points.py
Writes robodk_sim/gp_points.json.
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = Path(__file__).resolve().parent / "gp_points.json"


def main():
    import rospy
    from neurapy.robot import Robot
    from neurapy_ai.clients.database_client import DatabaseClient
    from settings import configure_ros

    settings = configure_ros()
    p = settings["sort_points"]
    names = [p["observe"], *p["picks"], p["lift"], p["via"], p["drop"]]
    rospy.init_node("export_gp_points", anonymous=True, disable_signals=True)
    db = DatabaseClient()
    robot = Robot()

    points = {}
    for name in dict.fromkeys(names):
        res, joints = db.get_joint_positions(name)
        if res.value < 0:
            raise SystemExit(f"{name} not found in the robot point database")
        flange = robot.compute_forward_kinematics(list(joints))  # flange pose, no tool
        saved = robot.get_pose_in_world_frame_from_DB(name)       # TCP pose stored when taught
        tcp = [saved["position"][k] for k in "xyz"] + [saved["euler"][k] for k in "abc"]
        points[name] = {"joints_rad": list(joints),
                        "joints_deg": [math.degrees(a) for a in joints],
                        "flange_m_rad": list(flange),
                        "tcp_m_rad": tcp,
                        "reference_frame_id": saved.get("referenceFrame")}
        print(f"{name}: joints(deg)={[round(a, 1) for a in points[name]['joints_deg']]}  "
              f"tcp(mm)={[round(v * 1000, 1) for v in tcp[:3]]}")

    # One live flange/TCP pair gives the controller's flange->TCP transform for the active tool.
    data = {"exported": datetime.now().isoformat(timespec="seconds"),
            "tool": robot.get_selected_tool_name(),
            "live_sample": {"joints_rad": list(robot.get_current_joint_angles()),
                            "flange_m_rad": list(robot.get_flange_pose()),
                            "tcp_m_rad": list(robot.get_tcp_pose())},
            "frame": ("robot base frame, metres and radians [x, y, z, a, b, c]; flange from forward "
                      "kinematics, tcp as stored in the point database with the RG6 tool"),
            "sequence": {k: p[k] for k in ("observe", "picks", "lift", "via", "drop")},
            "points": points}
    OUTPUT.write_text(json.dumps(data, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
