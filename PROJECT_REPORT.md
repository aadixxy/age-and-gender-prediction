# Task 3 Project Report - Face Detection & Age Prediction

## 1. Objective

Develop a computer vision application that detects human faces and estimates an approximate age group from each detected face using pretrained deep-learning models.

## 2. Technologies

- Python
- OpenCV
- OpenCV DNN module
- NumPy
- Pretrained TensorFlow face detector
- Pretrained Caffe age classifier

## 3. Methodology

1. Read an image or webcam frame.
2. Convert the frame into a DNN input blob.
3. Detect faces using a pretrained SSD-style OpenCV DNN face detector.
4. Crop each face from the original image.
5. Resize the crop to 227 × 227 pixels.
6. Apply the training-time mean values.
7. Run the pretrained age CNN.
8. Select the age bucket with the highest softmax score.
9. Draw a bounding box, predicted age group, and confidence on the frame.
10. Save the result in image mode or display it in real time in webcam mode.

## 4. Age categories

The classifier uses eight approximate categories:

- 0-2
- 4-6
- 8-12
- 15-20
- 25-32
- 38-43
- 48-53
- 60-100

## 5. Expected output

For each detected face the application reports:

- face bounding box
- face detection confidence
- predicted age range
- age-model confidence

## 6. Limitations

Age prediction is an estimate and should not be interpreted as an exact measurement. Performance can change with lighting, face angle, occlusion, image resolution, expressions, demographic variation, and other factors.

## 7. Future improvements

- Add video-file processing.
- Add performance logging.
- Compare multiple age-estimation models.
- Build a simple Streamlit interface for image uploads.
- Evaluate the model on a held-out labeled dataset and report metrics.
