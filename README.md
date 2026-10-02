# Dishwashing MAiRA

## Activate the WSL environment

Open Ubuntu-20.04 and run:

```bash
cd /mnt/c/Users/aimat/Desktop/Dishwashing_Maira
source ./activate_wsl.sh
```

The prompt should begin with `(dishwashing-maira)`. Verify the interpreter with:

```bash
which python
```

Expected path: `/root/.venvs/dishwashing-maira/bin/python`.

## Check the classifier (Python 3.10)

The classifier has a separate interpreter:
`/root/.venvs/dishwashing-classifier/bin/python`.
The ROS environment above remains on Python 3.8.

From the project folder, check the model using:

```bash
bash tests/check_classifier.sh
```

This runs the Python 3.10 model check without changing your active ROS environment
or accessing the robot/camera. The helper removes ROS's Python 3.8 package paths
from its subprocess. It checks model loading and synthetic inference, not accuracy.
Folder auto-activation continues to select the Python 3.8 ROS environment.

To leave the Python environment, run `deactivate`. Use a fresh terminal when
switching ROS projects, since deactivation does not undo ROS environment settings.

See [Camera setup](docs/CAMERA_SETUP.md) for camera checks and environment details.

## Live camera viewer

From the project folder with the Python 3.8 ROS environment active, run:

```bash
python camera_viewer.py --item plate
```

- The center crosshair helps position objects in the camera view.
- Press **S** to save an image without the crosshair or text overlays.
- Press **Q** or **Esc** to exit.
- Use `--item cup` for cup filenames.

Images save under `captures/YYYY-MM-DD/` as `plate_1.png`, `plate_2.png`, or
`cup_1.png`, with separate numbering per item and date. The viewer performs no
classification or robot movement.

Camera and network defaults are in `settings.json`. Use `--topic` to override
the camera topic. Live viewing and image saving were confirmed on 2026-10-02.
Displaying the window requires working WSL GUI/WSLg support.

## Capture and classify (no robot movement)

From the activated ROS environment:

```bash
python main_real.py --ros --classify
```

To classify an existing image without accessing the camera:

```bash
python main_real.py --image captures/2026-10-02/plate_2.png
```

The ROS process uses Python 3.8; classification runs separately in Python 3.10.
These commands print clean/dirty and never request robot movement. Accuracy still
needs evaluation with representative plates. The existing `--sort` mode has not
been migrated to this two-environment workflow or validated on the real robot.

## Evaluate predictions

Supply the actual label when testing a plate:

```bash
python main_real.py --ros --classify --expected clean
python main_real.py --image captures/2026-10-02/plate_2.png --expected dirty
```

Each successful `--classify` or `--image` run appends a row to
`captures/results.csv`: timestamp, image path, prediction, dirty score, expected
label, and whether they match. Without `--expected`, the expected and match fields
stay blank. Rechecking an image adds a new row. The score is the model's raw
sigmoid output: above 0.5 means dirty, otherwise clean; it is not a validated
confidence percentage. The expected label never changes the prediction.
The CSV is local and excluded from Git along with captures.

## Project folders

- `captures/`: camera images.
- `tests/`: automated tests and diagnostic scripts.
- `models/`: trained weights.
- `docs/`: guides and deployment PDF.
- `setup/`: one-time environment setup scripts.

Run automated tests from the project folder: `python -m unittest discover -s tests -v`.
