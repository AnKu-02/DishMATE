"""Workcell web UI: serves the frontend, the live camera stream and the use-case API.

Run from the project folder with the ROS environment active (source ./activate_wsl.sh):
    python -m ui.backend.server
then open http://localhost:8080 in a browser on this computer.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
STREAM_FPS = 15

USE_CASES = [
    {"id": "dishwashing", "name": "Dishwashing",
     "description": "Check each plate for dirt with the camera, then pick it from the rack and place it at the drop-off."},
]


def make_handler(camera, sessions):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal for robot messages
            pass

        def _send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/usecases":
                return self._send(200, USE_CASES)
            if path.startswith("/api/") and path.endswith("/state"):
                session = sessions.get(path.split("/")[2])
                return self._send(200, session.state()) if session else self._send(404, {"error": "Unknown use case"})
            if path == "/api/camera/stream.mjpg":
                return self._stream()
            return self._static(path)

        def do_POST(self):
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) != 3 or parts[0] != "api" or parts[1] not in sessions:
                return self._send(404, {"error": "Unknown endpoint"})
            session = sessions[parts[1]]
            action = {"start": session.start, "pause": session.pause, "resume": session.resume,
                      "stop": session.stop, "reset": session.reset_counts}.get(parts[2])
            if action is None:
                return self._send(404, {"error": "Unknown action"})
            try:
                action()
            except RuntimeError as exc:
                return self._send(409, {"error": str(exc), "state": session.state()})
            return self._send(200, session.state())

        def _static(self, path):
            target = (FRONTEND / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
            if FRONTEND not in target.parents or not target.is_file():
                return self._send(404, b"Not found", "text/plain")
            return self._send(200, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    jpeg = camera.jpeg()
                    if jpeg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                    time.sleep(1.0 / STREAM_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=20.0, help="joint speed/acceleration percent")
    args = parser.parse_args()

    import rospy
    from robot_motion import MairaArm
    from settings import configure_ros

    from ui.backend.camera import Camera
    from ui.backend.dishwashing import DishwashingSession

    settings = configure_ros()
    points = settings["sort_points"]
    print(f"Connecting to ROS at {settings['ros_master_uri']} ...", flush=True)
    rospy.init_node("workcell_ui", anonymous=True, disable_signals=True)
    camera = Camera()
    camera.subscribe(settings["camera_topic"])

    names = sorted({points["observe"], points["lift"], points["via"], points["drop"], *points["picks"]})
    sessions = {"dishwashing": DishwashingSession(camera, points, lambda: MairaArm(names, speed=args.speed))}

    # Localhost only: anyone who can reach this page can move the robot.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(camera, sessions))
    server.daemon_threads = True
    print(f"Workcell UI ready: open http://localhost:{args.port}  (Ctrl+C to quit)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for session in sessions.values():
            if session.state()["status"] in ("starting", "running", "paused", "stopping"):
                print("Robot run in progress: stopping after the current plate...", flush=True)
                try:
                    session.stop()
                except RuntimeError:
                    pass
                session.join()
        server.server_close()
        rospy.signal_shutdown("Workcell UI closed")


if __name__ == "__main__":
    main()
