# Dishwashing MAiRA

## Multiple dishes in one image

The CNN still produces one label for a whole image. It cannot locate or separately
classify multiple dishes. Multi-dish detector training requires one human-labelled
box for **every visible dish**, including overlapping dishes.

Annotate the 16 images directly inside the October 5 folder:

```bash
python annotate_dishes.py captures/2026-10-05
```

This command does not enter the existing clean/dirty subfolders. For each dish:

1. Press **1** for clean or **2** for dirty. Occupied plates (packets/wrappers) count as dirty for this project's sorting task.
2. Press **P** for plate (default), or **C** for cup.
3. Press **B**, draw its bounding box, then Enter. C/Esc cancels drawing.
4. Type its physical ID (for example `plate_01`) in the image window and press Enter.
5. Repeat for every dish. **U** undoes/removes the last box if you made a mistake.
6. Press **S** only when all dishes have correct boxes and labels. This saves and advances.

**N** skips without saving; **Q/Esc** exits in the main view. Do not give two
different dishes the same ID. Reuse an ID when the same dish appears in another
photo. Keep boxes tight, and annotate each overlapping dish separately. Skip
scenes where you cannot determine a dish's condition instead of guessing labels.

Images remain unchanged. JSON now contains an `objects` list with each dish's
class, ID and pixel bounding box. Existing single-object metadata is retained
under `legacy_single_object` when converted. Boxed previews show all objects.
The older single-object editor refuses these records so it cannot silently replace
them with one box. Live `camera_viewer.py --annotate` is still single-object;
use this saved-image editor for multi-dish scenes.

After review, a YOLO-format exporter is available:

```bash
python export_detection_data.py captures/2026-10-05 datasets/dishes_v1
```

Export reads the selected folder only (add `--recursive` for subfolders), excludes previews, and stops if any image lacks a
complete multi-object annotation. Review legacy folders with `annotate_dishes.py`
separately (or `--recursive`) before including them. Export does not train a model
or choose a split. Group images sharing plates/scenes into the same split; collect
independent plate groups for validation and testing. Cropped-image CNN performance
is also unverified; the fine-tuned CNN was trained on full frames.

An initial fine-tuned version of the existing CNN is available. See
[CNN training results](docs/CNN_TRAINING_2026-10-05.md) for weights, evaluation,
limitations, and a command to test it. Default inference still uses the original model.

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
from its subprocess and exposes the venv's CUDA libraries. It checks model loading
and synthetic inference, not accuracy. To install or repair TensorFlow GPU support,
then validate it on the NVIDIA GPU, run:

```bash
bash setup/enable_tensorflow_cuda.sh
```

For direct classifier or training commands in WSL, use
`bash setup/run_classifier_python.sh SCRIPT [ARGS...]` so CUDA remains available.
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

With `--label dirty` or `--label clean`, the viewer saves into that date's `dirty/`
or `clean/` subfolder, with separate numbering in each. Images and matching JSON
files stay together. Earlier captures are not moved.
Annotated live captures also save `previews/plate_N_boxed.png` inside that class
folder, showing the box, class name and plate ID. These previews are for review
only; use the original PNG and JSON for training. Folder annotation skips previews.

To collect dirty plates with boxes first:

```bash
python camera_viewer.py --interval 10 --annotate --item plate --label dirty --plate-id plate_01
```

Wait for the frozen frame, then left-click and drag from one corner to the opposite
corner around the entire plate. Release the mouse and press Enter/Space to save
the image and annotation. C/Esc skips the frozen capture. The next 10-second wait
begins after saving/skipping. Press Q in the live view to quit (if currently drawing,
press C first to return to live view). S starts a capture early from the live view.

Stop and restart with `--plate-id plate_02`, etc. when switching physical plates.
For clean examples, restart with `--label clean`, keeping the same ID for the
same physical plate. The filename counter is not the physical plate ID. Each JSON
records the supplied cleanliness label, object type, class name (`dirty_plate` or
`clean_plate`), physical plate ID, and bounding box. This is manual labelling, not
a prediction. Capture one plate per image.

Camera and network defaults are in `settings.json`. Use `--topic` to override
the camera topic. Live viewing and image saving were confirmed on 2026-10-02.
Displaying the window requires working WSL GUI/WSLg support.

## Capture and classify (no robot movement)

For dataset collection, automatically save every 10 seconds:

```bash
python camera_viewer.py --interval 10 --item plate --label clean --plate-id blue_printed_01
```

The first automatic capture occurs about 10 seconds after the feed appears.
Keep the plate still at capture time and move your hands out of view. Press Q/Esc
to stop. Stale frames are skipped; missed captures are not saved in a burst.
Manual S capture still works without resetting the automatic timer.
The label applies to every image saved during this run. Stop and restart with
`--label dirty` when changing to dirty examples, keeping the same physical plate ID.
Omit `--label` and `--plate-id` for unlabelled capture.

Labelled images have a matching JSON file containing the human label, physical
plate ID, session ID, timestamp and topic. This does not run classification or
write prediction results to the CSV. Review blurry, empty, or hand-obscured frames
before training. Keep photos of the same physical plate together when splitting
training and testing to avoid testing on nearly duplicate training examples.

### Single-image classification

To annotate incoming captures immediately, add `--annotate`:

```bash
python camera_viewer.py --interval 10 --annotate --label clean --plate-id blue_printed_01
```

Every capture freezes the image. Drag a box around the whole plate and press
Enter/Space to save the image and its labelled bounding box, or C/Esc to skip
without saving. The next 10-second wait starts after saving or skipping, so
annotation time never causes queued captures. S also starts a manual capture.
Q/Esc exits when back in the live view. Stop and restart with `--label dirty`
when the plate's condition changes. Original images are saved without overlays.

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

## Workcell web UI

`python -m ui.backend.server`, then open <http://localhost:8080>: live camera, Start /
Pause / Stop, and clean/dirty results for the dishwashing use case. See
[ui/README.md](ui/README.md).

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

## Annotate saved images

In the project environment, open a day's captures:

```bash
python annotate_images.py captures/2026-10-03
```

You can also supply one image path. Press **B** and drag a box around the whole
plate or cup, then **Enter** to confirm (C cancels drawing). Press **S** to save
and advance, **N** to skip without saving, or **Q/Esc** to quit. Unsaved changes
are discarded when skipping or quitting. Existing boxes are displayed for review;
B replaces the box. This initial tool supports one object per image.

Annotations are written into the image's matching JSON file. Existing clean/dirty
labels, plate IDs, and session information are preserved; unlabelled images stay
unlabelled. The box records object type and `[x, y, width, height]` in original-image
pixels, measured from the top-left corner. Original images are never modified.
Object type comes from the existing annotation or plate/cup filename prefix;
`--item plate` or `--item cup` overrides it. The tool requires GUI/WSLg support,
but does not connect to ROS, the camera, or the robot. Boxes are saved annotations
only; the classifier still uses full images until crop preprocessing is added.

## Folder layout

- `captures/`: camera images.
- `tests/`: automated tests and diagnostic scripts.
- `models/`: trained weights.
- `docs/`: guides and deployment PDF.
- `setup/`: one-time environment setup scripts.

Run automated tests from the project folder: `python -m unittest discover -s tests -v`.
