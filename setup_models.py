from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FILES = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v1.0/age_net.caffemodel",
}


def download(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name} ...")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded / total * 100:6.1f}%", end="\r")
    print(f"  Saved to {destination}")


for filename, url in FILES.items():
    path = MODEL_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        print(f"Already exists: {filename}")
        continue
    try:
        download(url, path)
    except Exception as exc:
        if path.exists():
            path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {filename}: {exc}") from exc

print("\nAll model files are ready in the models/ folder.")
