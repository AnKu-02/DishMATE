"""Latest robot joint angles from the controller's /joint_states topic (read-only)."""
import threading
import time

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


class JointMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._positions = None
        self._received = 0.0

    def subscribe(self, topic="/joint_states"):
        """Requires rospy.init_node() first."""
        import rospy
        from sensor_msgs.msg import JointState

        def receive(message):
            order = {name: i for i, name in enumerate(message.name)}
            if all(name in order for name in JOINT_NAMES):
                self.put([message.position[order[name]] for name in JOINT_NAMES])

        return rospy.Subscriber(topic, JointState, receive, queue_size=1)

    def put(self, positions):
        with self._lock:
            self._positions = list(positions)
            self._received = time.monotonic()

    def latest(self):
        """(joint angles in rad or None, seconds since the last message or None)."""
        with self._lock:
            if self._positions is None:
                return None, None
            return list(self._positions), time.monotonic() - self._received
