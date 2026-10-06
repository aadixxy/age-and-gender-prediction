# Task 3 - Face Detection & Age Prediction

A beginner-friendly computer vision project using Python, OpenCV DNN, NumPy, and pretrained deep-learning models.

## What it does

- Detects faces in images.
- Predicts an **approximate age group** for each detected face.
- Supports multiple faces in one image.
- Supports real-time webcam inference.
- Shows face confidence and age-model confidence.
- Saves annotated image results.

> Note: The model predicts an age **range**, not an exact age. Results can be affected by lighting, pose, image quality, occlusion, and other factors.

## Project structure

```text
Task3_Age_Prediction/
│
├── models/
│   ├── README.md
│   └── (downloaded model files are ignored by Git)
│
├── images/
│   └── test.jpg                # put your own image here
│
├── output/
│   └── .gitkeep
│
├── age_prediction.py           # main program
├── setup_models.py              # downloads pretrained models
├── test_setup.py                # checks that models load
├── run_image.bat                # Windows image launcher
├── run_webcam.bat               # Windows webcam launcher
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Open the project

In PowerShell:

```powershell
cd path\to\Task3_Age_Prediction
```

## 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the program with the global Python installation or use Command Prompt with:

```cmd
.venv\Scripts\activate.bat
```

## 3. Install libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download the pretrained models

```powershell
python setup_models.py
```

On Windows PowerShell, the equivalent is:

```powershell
.\setup_models.ps1
```

Then verify the installation:

```powershell
python test_setup.py
```

Expected ending:

```text
Model setup test passed.
Face detector and age classifier loaded successfully.
```

## 5. Run on an image

Put a clear face image at:

```text
images\test.jpg
```

Run:

```powershell
python age_prediction.py --image images\test.jpg --output output\result.jpg
```

The annotated result is written to `output\result.jpg`.

You can also use:

```powershell
run_image.bat
```

## 6. Run with webcam

```powershell
python age_prediction.py --webcam
```

or:

```powershell
run_webcam.bat
```

Press **Q** or **Esc** to exit.

If your camera is not index 0:

```powershell
python age_prediction.py --webcam --camera 1
```

## 7. Change face detection sensitivity

Default threshold is `0.50`.

Lower it slightly when faces are hard to detect:

```powershell
python age_prediction.py --image images\test.jpg --threshold 0.40
```

Raise it when you are getting too many false detections:

```powershell
python age_prediction.py --image images\test.jpg --threshold 0.60
```

## Age classes

The pretrained classifier uses these eight broad age ranges:

```text
0-2
4-6
8-12
15-20
25-32
38-43
48-53
60-100
```

## GitHub upload

The large binary model weights are ignored so the repository remains GitHub-friendly. Upload the source code and configuration files, then users run:

```powershell
python setup_models.py
```

to fetch the models locally.

### Recommended Git commands

```powershell
git init
git add .
git commit -m "Add face detection and age prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Task3-Age-Prediction.git
git push -u origin main
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

```powershell
pip install -r requirements.txt
```

### Missing model files

```powershell
python setup_models.py
```

### Webcam does not open

Try another camera index:

```powershell
python age_prediction.py --webcam --camera 1
```

Also close Teams, Zoom, browser tabs, or other apps that may be using the webcam.

### OpenCV window does not appear

Run from a normal local desktop Python session rather than a headless server.

## Academic explanation

The pipeline is:

```text
Input image / webcam frame
        ↓
OpenCV DNN face detector
        ↓
Face bounding boxes
        ↓
Crop each face
        ↓
Resize to 227 × 227
        ↓
Mean subtraction
        ↓
Pretrained Caffe age CNN
        ↓
8-class softmax prediction
        ↓
Draw age range + confidence
```

The implementation demonstrates face detection, preprocessing, pretrained CNN inference, confidence scores, and result visualization.

## Important limitation

This project estimates age groups from facial appearance. It should not be treated as a reliable identification or exact-age system, and predictions may be inaccurate for particular people or conditions.
