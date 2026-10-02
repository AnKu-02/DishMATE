import cv2
import numpy as np
import time
import tempfile
import os

from robodk.robolink import Robolink

# ---------------------------
# LOAD MODEL
# ---------------------------
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

IMG_SIZE = 224
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "models", "plate_cleanliness.weights.h5")

# ---------------------------
# RECREATE MODEL
# ---------------------------
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

# Same as training
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ---------------------------
# LOAD TRAINED WEIGHTS
# ---------------------------
model.load_weights(WEIGHTS_PATH)

print("✅ AI model loaded successfully.")


def predict_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img, verbose=0)[0][0]
    print(f"Prediction value = {prediction:.4f}")

    if prediction > 0.5:
        return "dirty"
    else:
        return "clean"


# ---------------------------
# ROBO DK SETUP
# ---------------------------
RDK = Robolink()
robot = RDK.Item('Neura MAiRA M')

# Targets (names EXACTLY as in RoboDK)
Home  = RDK.Item('Home')
home2 = RDK.Item('home 2')
home3 = RDK.Item('home 3')
home4 = RDK.Item('home 4')

# Programs
C1 = RDK.Item('C1')
D1 = RDK.Item('D1')
C2 = RDK.Item('C2')
D2 = RDK.Item('D2')

# Camera
camera = RDK.Item('Camera 1')

# ---------------------------
# VALIDATION
# ---------------------------
items = [robot, Home, home2, home3, home4, C1, D1, C2, D2, camera]
names = ["Robot", "Home", "home 2", "home 3", "home 4", "C1", "D1", "C2", "D2", "Camera 1"]

for item, name in zip(items, names):
    if not item.Valid():
        print(f"❌ {name} not found in RoboDK")
        exit()

print("✅ All RoboDK items found.")


# ---------------------------
# CAMERA FUNCTION (ROBO DK)
# ---------------------------
def get_frame_from_robodk():
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img_path = temp_file.name
    temp_file.close()

    # Take snapshot from RoboDK camera
    RDK.Cam2D_Snapshot(img_path, camera)

    frame = cv2.imread(img_path)

    return frame


# ---------------------------
# MAIN FLOW
# ---------------------------
# 1️⃣ HOME
robot.MoveJ(Home)
time.sleep(2)

frame = get_frame_from_robodk()
if frame is not None:
    result = predict_frame(frame)
    print("Home →", result)

    if result == "dirty":
        D1.RunProgram()
    else:
        print("========== STEP 1: Moving to home 2 ==========")

time.sleep(5)

# 2️⃣ home 2
robot.MoveJ(home2)
time.sleep(2)

frame = get_frame_from_robodk()
if frame is not None:
    result = predict_frame(frame)
    print("home 2 →", result)

    if result == "clean":
        C1.RunProgram()
    else:
        print("========== STEP 2: Moving to home 3 ==========")

time.sleep(5)

# 3️⃣ home 3
robot.MoveJ(home3)
time.sleep(2)

frame = get_frame_from_robodk()
if frame is not None:
    result = predict_frame(frame)
    print("home 3 →", result)

    if result == "clean":
        C2.RunProgram()
    else:
        D2.RunProgram()

time.sleep(5)

# 4️⃣ home 4
robot.MoveJ(home4)
time.sleep(2)

frame = get_frame_from_robodk()
if frame is not None:
    result = predict_frame(frame)
    print("home 4 →", result)

    if result == "clean":
        C2.RunProgram()
    else:
        D2.RunProgram()

time.sleep(5)

# Return to HOME
robot.MoveJ(Home)

# ---------------------------
# CLEANUP
# ---------------------------
cv2.destroyAllWindows()
