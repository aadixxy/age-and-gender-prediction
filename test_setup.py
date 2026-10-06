from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

REQUIRED = [
    "opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb",
    "age_deploy.prototxt",
    "age_net.caffemodel",
]

for name in REQUIRED:
    path = MODEL_DIR / name
    assert path.exists(), f"Missing {name}. Run python setup_models.py"
    assert path.stat().st_size > 0, f"Empty model file: {name}"

face_net = cv2.dnn.readNetFromTensorflow(
    str(MODEL_DIR / "opencv_face_detector_uint8.pb"),
    str(MODEL_DIR / "opencv_face_detector.pbtxt"),
)
age_net = cv2.dnn.readNetFromCaffe(
    str(MODEL_DIR / "age_deploy.prototxt"),
    str(MODEL_DIR / "age_net.caffemodel"),
)

print("Model setup test passed.")
print("Face detector and age classifier loaded successfully.")
