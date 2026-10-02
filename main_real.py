"""MAiRA camera checks and sorting. Default mode captures only; --sort enables sorting."""
import argparse
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
from capture_storage import save_capture
from settings import configure_ros

IMG_SIZE = 224
HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "models" / "plate_cleanliness.weights.h5"
CAMERA_SOURCE = 0
CONFIRM_BEFORE_MOVE = True


def load_model():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.models import Model

    base = MobileNetV2(weights=None, include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    model = Model(inputs=base.input, outputs=Dense(1, activation="sigmoid")(x))
    model.load_weights(str(WEIGHTS_PATH))
    print(f"AI model loaded: {WEIGHTS_PATH}")
    return model


def check_model():
    import sys

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for package in ("tensorflow", "tensorflow-cpu", "keras", "numpy", "opencv-python"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed under this distribution name")
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError(
            "TensorFlow is not installed in this interpreter. The project's pinned "
            "TensorFlow 2.21.0 and Keras 3.12.1 require Python 3.10 or newer. "
            "Keep the ROS Python 3.8 environment intact; classifier setup is still pending."
        )
    import numpy as np

    model = load_model()
    output = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32),
                           verbose=0)
    if output.shape != (1, 1) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected model output: {output}")
    print("Model check passed: weights loaded and inference completed.")
    print("This checks execution only, not classification accuracy.")


def predict_frame(frame, model):
    import cv2
    import numpy as np

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    p = float(model.predict(img, verbose=0)[0][0])
    print(f"Prediction value = {p:.4f}")
    return "dirty" if p > 0.5 else "clean"


def camera_source(value):
    return int(value) if value.isdecimal() else value


def get_frame(source=CAMERA_SOURCE, backend="auto", warmup=2.0,
              width=None, height=None, preview=False):
    import cv2

    backends = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF}
    cap = cv2.VideoCapture(source, backends[backend])
    try:
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened. Check --camera, close other "
                               "camera apps, and check Windows camera permissions.")
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Camera opened using {cap.getBackendName()}.")
        deadline = time.monotonic() + warmup
        frame = None
        while True:
            ok, current = cap.read()
            if ok and current is not None and current.size:
                frame = current
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if frame is None:
            raise RuntimeError("Camera opened but returned no image. Try another "
                               "--camera or --backend, or increase --warmup.")
        if preview:
            print("Preview: S saves the current frame; Q or Escape cancels.")
            while True:
                cv2.imshow("MAiRA camera - S: save, Q: cancel", frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    return None
                if key == ord("s"):
                    break
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    raise RuntimeError("Camera stopped returning images during preview.")
        print(f"Captured image: {frame.shape[1]} x {frame.shape[0]} pixels.")
        return frame
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def get_ros_frame(topic, timeout=10.0):
    """Read the AI SDK's ROS color stream without creating a motion client."""
    try:
        import rospy
        from sensor_msgs.msg import Image, CompressedImage
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError("ROS camera capture needs rospy, sensor_msgs and cv_bridge. "
                           "Run in the configured Ubuntu/ROS SDK environment, "
                           "not the Windows .venv.") from exc
    configure_ros()
    compressed = topic.rstrip("/").endswith("/compressed")
    if not rospy.core.is_initialized():
        rospy.init_node("dishwashing_camera", anonymous=True, disable_signals=True)
    print(f"Waiting for ROS color image on {topic} (timeout {timeout}s)...")
    try:
        message = rospy.wait_for_message(topic, CompressedImage if compressed else Image, timeout=timeout)
    except rospy.ROSException as exc:
        raise RuntimeError(f"No ROS image received on {topic}: {exc}") from exc
    # Match OpenCV's BGR convention used by USB capture and saved PNG files.
    bridge = CvBridge()
    frame = (bridge.compressed_imgmsg_to_cv2(message, desired_encoding="bgr8") if compressed
             else bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
    if frame is None or not frame.size:
        raise RuntimeError("ROS camera returned an empty image.")
    print(f"Captured ROS image: {frame.shape[1]} x {frame.shape[0]} pixels.")
    return frame


def positive_seconds(value):
    number = warmup_seconds(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def warmup_seconds(value):
    import math

    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--camera-only", action="store_true", help="capture only (default)")
    modes.add_argument("--check-model", action="store_true", help="check dependencies and weights without a camera")
    modes.add_argument("--sort", action="store_true", help="classify and request real robot motion")
    modes.add_argument("--classify", action="store_true", help="capture and classify in Python 3.10; no robot motion")
    modes.add_argument("--image", type=Path, help="classify an existing image without camera or robot access")
    parser.add_argument("--ros", action="store_true", help="use the camera topic in settings.json")
    parser.add_argument("--camera", type=camera_source, default=CAMERA_SOURCE,
                        help="camera index (0, 1, ...) or network stream URL")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--warmup", type=warmup_seconds, default=2.0)
    parser.add_argument("--width", type=positive_int)
    parser.add_argument("--height", type=positive_int)
    parser.add_argument("--preview", action="store_true", help="live preview before saving")
    parser.add_argument("--ros-topic", help="ROS color Image topic, typically /camera/color/image_raw")
    parser.add_argument("--timeout", type=positive_seconds, default=10.0,
                        help="ROS image wait timeout in seconds")
    parser.add_argument("--item", choices=("plate", "cup"), default="plate",
                        help="capture filename prefix (default: plate); not automatic object detection")
    parser.add_argument("--output", type=Path,
                        help="optional image path; default: captures/YYYY-MM-DD/<item>_<number>.png")
    args = parser.parse_args(argv)
    if args.ros and not args.ros_topic:
        args.ros_topic = configure_ros()["camera_topic"]
    if args.image:
        from classifier_client import classify_image
        print("Result:", classify_image(args.image)["label"])
        return
    if args.ros_topic and (args.preview or args.width or args.height or args.backend != "auto"):
        parser.error("--ros-topic does not support preview, resolution or USB backend options; "
                     "configure the ROS camera publisher instead.")
    if args.check_model:
        check_model()
        return

    import cv2

    frame = (get_ros_frame(args.ros_topic, args.timeout) if args.ros_topic else
             get_frame(args.camera, args.backend, args.warmup,
                       args.width, args.height, args.preview))
    if frame is None:
        print("Capture cancelled.")
        return
    output = save_capture(frame, item=args.item, output=args.output)
    print(f"Saved: {output}")
    if args.classify:
        from classifier_client import classify_image
        print("Result:", classify_image(output)["label"])
        return
    if not args.sort:
        print("Camera-only check complete. No classification or motion SDK connection.")
        return

    result = predict_frame(frame, load_model())
    print("Result:", result)
    if result == "clean" and not (HERE / "C1_program.py").is_file():
        print("Clean-rack routine unavailable: add C1_program.py defining C1(). No movement requested.")
        return
    if CONFIRM_BEFORE_MOVE:
        input(f"Plate is '{result}'. Emergency stop within reach? Press Enter to move the robot...")
    # Exported programs create Robot() on import. Import only after confirmation.
    module_name, function_name = (("Dishwasher_Rack_Code", "D1") if result == "dirty"
                                  else ("C1_program", "C1"))
    program = importlib.import_module(module_name)
    getattr(program, function_name)()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
