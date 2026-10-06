import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

FACE_PROTO = MODEL_DIR / "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR / "opencv_face_detector_uint8.pb"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"

AGE_BUCKETS = [
    "0-2", "4-6", "8-12", "15-20",
    "25-32", "38-43", "48-53", "60-100"
]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def validate_models() -> None:
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing) +
            ". Run: python setup_models.py"
        )


def load_models():
    validate_models()
    face_net = cv2.dnn.readNetFromTensorflow(str(FACE_MODEL), str(FACE_PROTO))
    age_net = cv2.dnn.readNetFromCaffe(str(AGE_PROTO), str(AGE_MODEL))
    return face_net, age_net


def detect_and_predict(image: np.ndarray, face_net, age_net, confidence_threshold=0.50):
    h, w = image.shape[:2]
    annotated = image.copy()
    results = []

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        age_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MEAN_VALUES, swapRB=False
        )
        age_net.setInput(age_blob)
        age_scores = age_net.forward()[0]
        age_index = int(np.argmax(age_scores))
        age_group = AGE_BUCKETS[age_index]
        age_confidence = float(age_scores[age_index])

        label = f"Age: {age_group} ({age_confidence * 100:.1f}%)"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(25, y1 - 10)
        cv2.putText(
            annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 255, 0), 2, cv2.LINE_AA
        )

        results.append({
            "box": (x1, y1, x2, y2),
            "face_confidence": confidence,
            "age_group": age_group,
            "age_confidence": age_confidence,
        })

    return annotated, results


def run_image(input_path: Path, output_path: Path, threshold: float):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    face_net, age_net = load_models()
    start = time.perf_counter()
    result, detections = detect_and_predict(image, face_net, age_net, threshold)
    elapsed = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not save output image: {output_path}")

    print(f"Faces detected : {len(detections)}")
    print(f"Inference time : {elapsed:.3f} sec")
    print(f"Saved result   : {output_path}")
    for i, item in enumerate(detections, 1):
        print(
            f"Face {i}: age group {item['age_group']}, "
            f"face confidence {item['face_confidence']:.2f}, "
            f"age confidence {item['age_confidence']:.2f}"
        )

    cv2.imshow("Face & Age Prediction", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(camera_index: int, threshold: float):
    face_net, age_net = load_models()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try --camera 1 or close other apps using the webcam."
        )

    prev = time.perf_counter()
    fps = 0.0

    print("Webcam started. Press Q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            start = time.perf_counter()
            annotated, detections = detect_and_predict(frame, face_net, age_net, threshold)
            elapsed = time.perf_counter() - start

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps = 0.85 * fps + 0.15 * instant_fps if fps else instant_fps

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | Faces: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Face & Age Prediction - Webcam", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Face detection and approximate age-group prediction using OpenCV DNN."
    )
    parser.add_argument("--image", type=Path, help="Path to an input image")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "result.jpg",
        help="Output path for image mode"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for webcam mode (default: 0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Face confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Use webcam mode instead of image mode"
    )
    args = parser.parse_args()

    if args.webcam or args.image is None:
        run_webcam(args.camera, args.threshold)
    else:
        run_image(args.image, args.output, args.threshold)


if __name__ == "__main__":
    main()
