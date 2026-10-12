"""Latest ROS camera frame, shared by the live stream and the classifier."""
import threading
import time

ONLINE_AFTER = 2.0  # seconds without a frame before the camera counts as offline


class Camera:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._received = 0.0
        self._seq = 0
        self._jpeg = (None, b"")

    def subscribe(self, topic):
        """Feed frames from a ROS Image topic; requires rospy.init_node() first."""
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        bridge = CvBridge()

        def receive(message):
            try:
                self.put(bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
            except Exception as exc:
                print(f"Camera decode error: {exc}", flush=True)

        return rospy.Subscriber(topic, Image, receive, queue_size=1, buff_size=32 * 1024 * 1024)

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._received = time.monotonic()
            self._seq += 1

    @property
    def online(self):
        with self._lock:
            return self._frame is not None and time.monotonic() - self._received < ONLINE_AFTER

    def fresh_frame(self, after, timeout=10.0):
        """A copy of the first frame received after `after`, so the photo is not motion-blurred."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._received > after:
                    return self._frame.copy()
            time.sleep(0.05)
        raise RuntimeError("No new camera frame after the robot stopped")

    def jpeg(self, quality=80):
        """JPEG of the latest frame, encoded once per new frame; None before the first frame."""
        import cv2

        with self._lock:
            frame, seq = self._frame, self._seq
            cached_seq, data = self._jpeg
        if frame is None:
            return None
        if cached_seq != seq:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            data = encoded.tobytes()
            with self._lock:
                self._jpeg = (seq, data)
        return data
