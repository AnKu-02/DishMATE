# Camera and model checks

## Ubuntu-20.04 development workspace

Fresh standalone terminals start at `/root` (`root@AIMatters:~#`) without an
active Python environment or automatic ROS setup. VS Code terminals retain their
project working directory. Existing terminals must be closed and reopened to
discard their old environment settings; there is no need to restart WSL.

Activate this project's environment explicitly:

```bash
cd /mnt/c/Users/aimat/Desktop/Dishwashing_Maira
source ./activate_wsl.sh
which python
```

Expected interpreter: `/root/.venvs/dishwashing-maira/bin/python` (Python 3.8).
The environment is stored on the Linux filesystem for performance. It reads the
shared system packages required by ROS Noetic through `--system-site-packages`,
but new pip installs go into its own environment. It does not include packages
from the existing Kanak/NEURA virtual environments. ROS, cv_bridge, OpenCV, NumPy,
and NeuraPy imports have been verified. Do not install `requirements.txt` into this
Python 3.8 environment. Classifier packages use the separate Python 3.10 environment
at `/root/.venvs/dishwashing-classifier`; run `bash tests/check_classifier.sh` from this
project to check it. That helper clears inherited Python package paths and leaves
the active ROS shell unchanged. Use `python main_real.py --ros --classify` to capture and classify across both
interpreters without motion. Real robot sorting is still unvalidated.

In VS Code connected to Ubuntu-20.04, select that Linux interpreter. The existing
project `.venv` is Windows-only and remains untouched.

For the original Kanak workflow, run `kanak` in a fresh terminal. This selects the
original `neura_new_sw_version/.venv` and enters the Kanak folder. `nsw` selects
the same original environment in its own project folder. Existing project code
and environments were not modified.

Each student can keep a separately named environment under `/root/.venvs` and
activate it explicitly from their project. Use a fresh terminal when switching
ROS projects: `deactivate` resets Python activation, but does not undo ROS exports.

Original shell settings are backed up at
`/root/.shell-backups/dishwashing-20261001-150120/`. Only `.bashrc` and
`.bash_aliases` were edited; `.profile` and `.bash_neura_aliases` were also backed
up for reference. `setup/setup_wsl_workspace.py` records the one-time migration and
should not be rerun. `tests/verify_wsl_workspace.py` checks startup and manual activation.

## MAiRA built-in camera (ROS)

### Live positioning viewer

In the activated Python 3.8 ROS environment, run:

```bash
python camera_viewer.py --item plate
```

Choose `--item cup` for cup filenames. Press S to save, Q/Escape to exit, or close
the window. The green center crosshair helps framing; it is not a robot coordinate
or grasp target. Saved images contain no crosshair or text. Saving uses the daily
numbered captures folder. A stalled stream displays a red warning and disables
saving until images resume. The viewer never imports a robot motion client.

The default topic is `/camera/color/image_raw`; use `--topic` to select the actual
camera topic. Topics ending in `/compressed` are read as ROS CompressedImage.
For example, the old Kanak startup settings referenced
`/base_camera_1/color/image_rect_color/compressed`; this still needs live verification.
The first-frame timeout is 10 seconds after ROS initialization. Ctrl+C cancels
waiting for a ROS master. The window requires working WSL GUI/WSLg support.

### Single-image capture

The provided NeuraAISDK 3.3.12 wheel uses ROS `sensor_msgs/Image` on
`/camera/color/image_raw` by default (see
`neurapy_ai/utils/sensor_data_acquisition.py` inside `neurapy_ai-0.7.2`).
The actual robot topic must be checked on the connected ROS system.
This camera is not a Windows USB camera index.

On this PC, Ubuntu-20.04 has ROS Noetic and working camera imports in
`/root/.venvs/dishwashing-maira`. The saved SDK settings are master `192.168.2.14:11311`
and client `192.168.2.99`. The helper uses those defaults, permits environment
overrides, and limits the whole capture process to 25 seconds. From PowerShell:

```powershell
wsl -d Ubuntu-20.04 -e bash /mnt/c/Users/aimat/Desktop/Dishwashing_Maira/capture_maira.sh
```

Live viewing and saving were confirmed by the user on 2026-10-02. Shared defaults
are stored in `settings.json`; update its topic if your working command uses another one.

Run capture using the project Ubuntu/ROS environment with `rospy`,
`sensor_msgs`, `cv_bridge`, and OpenCV available. Preserve the working SDK network
configuration (`ROS_MASTER_URI` and reachable ROS_IP/ROS_HOSTNAME); the controller
address defaults are in `settings.json` and can be overridden through environment
variables. In a terminal where `activate_wsl.sh` has been sourced:

```bash
rostopic list
python3 /mnt/c/Users/aimat/Desktop/Dishwashing_Maira/main_real.py --camera-only --ros-topic /camera/color/image_raw --timeout 10
```

This saves `captures/YYYY-MM-DD/plate_N.png` into this project and does not import the motion SDK.
Use `--item cup` for `cup_N.png`. Numbering is separate per item and per date.
TensorFlow is not needed for capture. ROS images are converted to BGR for OpenCV.
The timeout bounds waiting for an image after ROS initialization; master connection
and initialization can take additional time. ROS capture uses the publisher's
resolution; USB preview/resolution/backend options are not supported in this mode.
The Windows `.venv` is for local camera/model checks, not the ROS installation.

## PC USB camera and model checks

Select `.venv\Scripts\python.exe` as your IDE interpreter.
To recreate this environment from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check package versions, model weights, and inference with a synthetic image:

```powershell
.\.venv\Scripts\python.exe main_real.py --check-model
```

This checks execution, not classification accuracy. It needs no camera or robot.

Capture from camera 0 without TensorFlow or a robot connection:

```powershell
.\.venv\Scripts\python.exe main_real.py --camera-only --camera 0
```

Open the printed image path under `captures/YYYY-MM-DD/` to check framing, focus,
and lighting. Each capture gets the next available item number. For live preview,
press S to save or Q/Escape to cancel:

```powershell
.\.venv\Scripts\python.exe main_real.py --camera-only --camera 0 --preview
```

When the final camera arrives, change `--camera` to its index (for example 1) or
a supported network stream URL. Indices can change after reconnecting devices.
If a Windows USB camera fails, try `--backend dshow` or `--backend msmf`, close
other camera apps, and allow desktop apps access in Windows camera settings.

Optional settings: `--width 1280 --height 720 --warmup 3 --output camera_test.png`.
Resolution requests depend on camera support; actual image dimensions are printed.
Warmup allows exposure to settle, but is not a network read timeout. Network streams
may block inside the driver. Custom output folders are created automatically;
an existing output file is never overwritten.

Running without arguments defaults to camera-only mode. Sorting requires explicit
`--sort` and movement confirmation. Real movement targets still need validation,
and the exported routine still needs physical gripper commands before deployment.
