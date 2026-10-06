$ErrorActionPreference = "Stop"
$ModelDir = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

$files = @{
  "opencv_face_detector.pbtxt" = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt"
  "opencv_face_detector_uint8.pb" = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb"
  "age_deploy.prototxt" = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
  "age_net.caffemodel" = "https://github.com/GilLevi/AgeGenderDeepLearning/releases/download/v1.0/age_net.caffemodel"
}

foreach ($name in $files.Keys) {
  $path = Join-Path $ModelDir $name
  if (Test-Path $path) {
    Write-Host "Already exists: $name"
    continue
  }
  Write-Host "Downloading: $name"
  Invoke-WebRequest -Uri $files[$name] -OutFile $path
}

Write-Host "All model files are ready in models/."
