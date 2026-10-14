# CNN fine-tuning — 2026-10-05

Fine-tuned the existing MobileNetV2 classifier, starting from
`models/plate_cleanliness.weights.h5`. The original file and default inference
selection remain unchanged. This is image classification, not object detection.

New model: `models/plate_cleanliness_cnn_20261005_132253.weights.h5`.

## Data and split

Source: `captures/2026-10-05`, original PNGs with human labels only. Previews are
excluded. Image dimensions and bounding boxes were checked; no boxes were used to
crop the training images. Preprocessing matches existing inference: full image,
OpenCV BGR, resize to 224x224, divide by 255. Threshold remains 0.5.

| Split | Physical plate IDs | Clean | Dirty |
|---|---|---:|---:|
| Train | plate_01, plate_04, plate_05 | 30 | 22 |
| Validation | plate_02 | 15 | 12 |
| Test | plate_03 | 12 | 16 |

One record was excluded: `clean/plate_1.json` references `plate_6.png` rather than
`plate_1.png`. Source annotations and images were not modified. No exact duplicate
image content was found among the retained records. Plate ID consistency across
clean/dirty captures is assumed from the supplied annotations.

To omit reviewed bad images without deleting or moving source files, pass a CSV
with `image,reason` columns. Image paths are relative to `--source`; unlisted
images keep the existing behavior. For example:

```csv
image,reason
dirty/plate_12.png,partially obscured
clean/plate_18.png,blurred
```

```powershell
.\.venv\Scripts\python.exe train_classifier.py --source captures\2026-10-05 --exclusion-manifest captures\2026-10-05\exclusions.csv
```

Excluded paths and reasons are recorded in the run report. Original images and
annotations remain untouched.

## Training

Ran on the Windows project's existing TensorFlow 2.21.0 CPU environment. Seed 42,
batch size 8. First trained the classification head at learning rate 0.0001 (up to
12 epochs, stopped after 11); then unfroze non-BatchNormalization layers among
the final 30 model layers for 18 epochs at learning rate 0.00001. Augmentation:
horizontal flips, modest rotations and translations, on training images only.
Selected the lowest validation loss across both phases (fine-tuning epoch 13).
No test results were used to select the checkpoint or change the threshold.

| Model | Validation correct | Test correct |
|---|---:|---:|
| Original | 15/27 (55.6%) | 12/28 (42.9%) |
| Fine-tuned | 17/27 (63.0%) | 20/28 (71.4%) |

Fine-tuned test outcomes: 11/12 clean images correctly classified, 9/16 dirty
images correctly classified. Seven dirty images were called clean; one clean
image was called dirty. Training accuracy was 50/52 (96.2%).

This is a small pilot, not proof of deployment reliability. Validation and test
each contain only one physical plate, with correlated views; training includes
only one dirty plate ID. Collect dirty counterparts for more plate types and
independent capture sessions. The gap between training and held-out performance
indicates limited generalization. Do not use these results to authorize robot
sorting automatically.

Detailed manifest, epoch CSVs, per-image scores, and confusion matrices are in
`training_runs/cnn_20261005_132253/` (local, ignored by Git). The saved weights were
reloaded into the original inference architecture and checked for matching output;
the original weights' SHA-256 was checked unchanged.

## Additional folder dataset experiment — 2026-10-09

The folder dataset at `Downloads/plates/plates` contained 20 images in each of
`train/cleaned` and `train/dirty`; its 744 unlabeled `test` images were not used.
Run this layout with:

```powershell
.\.venv\Scripts\python.exe train_classifier.py --folder-dataset "C:\Users\aimat\Downloads\plates\plates"
```

Because the images have no physical plate IDs, this experiment used a seeded,
stratified image-level split (12 per class for training, 4 per class for
validation, 4 per class for test). The split may contain correlated views of the
same physical plate, so its metrics are only a small pilot. Fine-tuned accuracy
was 5/8 (62.5%) on the held-out split, below the original model's 6/8 (75%). On
the known dirty GP145 capture, the original and new model both predicted clean;
the new model's dirty score was lower (0.139 vs 0.285). Do not use this
checkpoint for deployment or replace the default weights.

Weights: `models/plate_cleanliness_cnn_20261009_100656.weights.h5`.
Full split manifest and predictions: `training_runs/cnn_20261009_100656/report.json`.

### Zoom/crop augmentation follow-up

A second run added training-only random zoom (0.08, reflected edge fill) to the
existing flips, rotations, and translations. Preprocessing and inference stayed
unchanged: full-frame BGR, resize to 224x224, divide by 255. This simulates small
crop/scale variation without changing the validation or test images. It again
scored 5/8 (62.5%) on the same held-out split, and still called the known dirty
GP145 image clean (dirty score 0.275; original score 0.285). The augmentation
did not improve this failure case; do not promote this checkpoint either.

Weights: `models/plate_cleanliness_cnn_20261009_165406.weights.h5`.
Full report: `training_runs/cnn_20261009_165406/report.json`.

### Cross-check on the known GP145 plate

Offline inference on `captures/2026-10-09/plate_1.png` gave these raw dirty
scores: original weights 0.285 (clean), October 5 project-capture fine-tune
0.526 (dirty), October 9 folder fine-tune 0.139 (clean), and October 9 zoom
fine-tune 0.275 (clean). The October 5 checkpoint is the strongest candidate
for further evaluation on this camera domain. This is one known plate and does
not justify changing the default model or using it for autonomous sorting.

### Ground-truth box-crop experiment — 2026-10-10

Trained with `--crop-to-box`, using reviewed bounding boxes for single- and
multi-dish annotated images. The physical-plate split was preserved. The cropped
checkpoint scored 8/28 (28.6%) on the held-out project-camera test set, below the
original weights evaluated on the same box-cropped test inputs at 12/28 (42.9%).
On the known GP145 plate crop, it still predicted clean (dirty score 0.406). This checkpoint is not an improvement. It
also requires a matching box at inference, which the current live classifier
does not provide; leave it experimental.

Weights: `models/plate_cleanliness_cnn_20261010_121221.weights.h5`.
Full report: `training_runs/cnn_20261010_121221/report.json`.

## Try the new model on a saved image

From the project folder in WSL (no camera or robot connection):

```bash
env -u PYTHONPATH -u PYTHONHOME /root/.venvs/dishwashing-classifier/bin/python classify_image.py captures/2026-10-05/clean/plate_2.png --weights models/plate_cleanliness_cnn_20261005_132253.weights.h5
```

This prints JSON with the label and dirty score. It does not append to results.csv.
The normal `main_real.py --classify` flow still uses the original model.

To reproduce training in the classifier environment:

```bash
env -u PYTHONPATH -u PYTHONHOME /root/.venvs/dishwashing-classifier/bin/python train_classifier.py --source captures/2026-10-05
```

Every run creates a new timestamped output. Numerical results may vary between
platforms; exact input images and metadata must match the recorded manifest.

Method reference: [Keras transfer learning and fine-tuning](https://keras.io/guides/transfer_learning/).
