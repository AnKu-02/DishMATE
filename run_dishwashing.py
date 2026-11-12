"""Start the dishwashing workcell: UI backend (WSL), browser UI and RoboDK digital twin.

Run on Windows with RoboDK's Python (or double-click run_dishwashing.bat):
    C:\\RoboDK\\Python-Embedded\\python.exe run_dishwashing.py

1. starts `python -m ui.backend.server` in Ubuntu-20.04 (reuses one that is already running)
2. opens RoboDK with the dishwashing station (settings.json "robodk_station") if needed
3. opens http://localhost:8080 in the browser
4. mirrors the real robot into RoboDK until Ctrl+C, then shuts the backend down

The real robot only moves when Start is pressed in the browser (Automatic mode, e-stop in reach).
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from urllib.request import urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "robodk_sim"))

DISTRO = "Ubuntu-20.04"
WSL_PROJECT = f"/mnt/{HERE[0].lower()}/" + HERE[3:].replace("\\", "/")  # Windows path -> WSL path
UI_URL = "http://localhost:8080"
ROBODK_EXE = r"C:\RoboDK\bin\RoboDK.exe"
ROBODK_PORT = 20500


def backend_up():
    try:
        with urlopen(UI_URL + "/api/usecases", timeout=1):
            return True
    except OSError:
        return False


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_backend():
    command = ["wsl", "-d", DISTRO, "-u", "root", "--cd", WSL_PROJECT, "--", "bash", "-c",
               "source ./activate_wsl.sh >/dev/null && exec python -u -m ui.backend.server"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace")

    def relay():
        for line in process.stdout:
            print("[backend] " + line.rstrip(), flush=True)
    threading.Thread(target=relay, daemon=True).start()
    return process


def stop_backend(process):
    print("Stopping the UI backend (a running plate is finished first)...", flush=True)
    subprocess.run(["wsl", "-d", DISTRO, "-u", "root", "--", "pkill", "-INT", "-f", "ui.backend.server"],
                   capture_output=True)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for(check, what, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        if process is not None and process.poll() is not None:
            raise SystemExit(f"{what} exited; see the [backend] messages above")
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for {what}")


def main():
    with open(os.path.join(HERE, "settings.json")) as handle:
        station = json.load(handle).get("robodk_station")

    backend = None
    if backend_up():
        print("UI backend already running", flush=True)
    else:
        print("Starting the UI backend in WSL (connects to the robot's ROS)...", flush=True)
        backend = start_backend()
        wait_for(backend_up, "the UI backend", 120, backend)

    if not port_open(ROBODK_PORT):
        if not station or not os.path.isfile(station):
            raise SystemExit(f"RoboDK is not open and the station file is missing: {station}")
        print(f"Opening RoboDK with {station}", flush=True)
        subprocess.Popen([ROBODK_EXE, station])
        wait_for(lambda: port_open(ROBODK_PORT), "RoboDK", 90)
        time.sleep(3)  # let the station finish loading

    webbrowser.open(UI_URL + "/#/dishwashing")
    print(f"UI: {UI_URL}   |   Ctrl+C here stops the twin and the backend", flush=True)

    from robodk.robolink import Robolink
    import twin
    try:
        twin.run(Robolink(robodk_path="", args=[], port=ROBODK_PORT, quit_on_close=False))
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            stop_backend(backend)


if __name__ == "__main__":
    main()
