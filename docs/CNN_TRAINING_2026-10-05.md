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
