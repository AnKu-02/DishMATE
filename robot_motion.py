"""MAiRA joint moves to taught GUI points and RG6 gripper control (Python 3.8 ROS env)."""
import math

# Joint 1 is limited to +/-180 deg; refuse anything that would need a longer turn.
MAX_JOINT_TURN = 2 * math.pi
ARRIVAL_TOLERANCE = math.radians(1.0)


def plan_target(target, current):
    """Pick the reachable joint-1 equivalent closest to the current angle."""
    target = list(target)
    options = [target[0] + k * 2 * math.pi for k in (-1, 0, 1)]
    options = [a for a in options if -math.pi - 1e-6 <= a <= math.pi + 1e-6]
    target[0] = min(options, key=lambda a: abs(a - current[0]))
    deltas = [abs(t - c) for t, c in zip(target, current)]
    worst = max(range(len(deltas)), key=deltas.__getitem__)
    if deltas[worst] > MAX_JOINT_TURN:
        raise RuntimeError(f"Refusing move: joint {worst + 1} would turn {math.degrees(deltas[worst]):.0f} deg")
    return target, deltas


class MairaArm:
    def __init__(self, point_names, speed=20.0):
        """Requires rospy.init_node() first; loads every point before any motion."""
        from neurapy.robot import Robot
        from neurapy_ai.clients.database_client import DatabaseClient

        db = DatabaseClient()
        self.points = {}
        for name in point_names:
            res, joints = db.get_joint_positions(name)
            if res.value < 0:
                raise RuntimeError(f"{name} not found in robot point database")
            self.points[name] = joints
        self.speed = speed
        self.robot = Robot()

    def move(self, name):
        current = self.robot.get_current_joint_angles()
        target, _ = plan_target(self.points[name], current)
        self.robot.move_joint(target_joint=[target], current_joint_angles=current,
                              speed=self.speed, acceleration=self.speed, enable_blending=False)
        reached = self.robot.get_current_joint_angles()
        error = max(abs(a - b) for a, b in zip(reached, target))
        if error > ARRIVAL_TOLERANCE:
            raise RuntimeError(f"Did not reach {name} (off by {math.degrees(error):.1f} deg)")

    # On this RG6 setup "on" opens and "off" closes (observed 2026-10-09).
    def open_gripper(self):
        self.robot.gripper("on")

    def close_gripper(self):
        self.robot.gripper("off")

    def stop(self):
        self.robot.stop()
