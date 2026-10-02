"""Shared project settings; no camera or model imports."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_settings():
    with (ROOT / "settings.json").open() as handle:
        return json.load(handle)


def configure_ros():
    settings = load_settings()
    # Project configuration replaces ROS's incidental localhost default.
    master = os.environ.get("ROS_MASTER_URI", "")
    if not master or master.rstrip("/") in ("http://localhost:11311", "http://127.0.0.1:11311"):
        os.environ["ROS_MASTER_URI"] = settings["ros_master_uri"]
    os.environ.setdefault("ROS_IP", settings["ros_ip"])
    os.environ.setdefault("ROS_HOSTNAME", os.environ["ROS_IP"])
    return settings


if __name__ == "__main__":
    import shlex
    configure_ros()
    for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOSTNAME"):
        print("export " + key + "=" + shlex.quote(os.environ[key]))
