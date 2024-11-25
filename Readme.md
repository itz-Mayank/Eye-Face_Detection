# Installation

## Tenserflow setup

![image](https://github.com/user-attachments/assets/948f95d8-2636-46fd-aedd-96a6976a6120)

Download and install the NVIDIA CUDA Toolkit --  https://developer.nvidia.com/cuda-toolkit
Download and install cuDNN --  https://developer.nvidia.com/cudnn

## Upgrade pip (optional but recommended):
pip install --upgrade pip

## Install OpenCV:
pip install opencv-python
pip install opencv-contrib-python

## Install TensorFlow:
pip install tensorflow

## Install NumPy (used for numerical operations in TensorFlow and OpenCV):
pip install numpy

## Audio
pip install opencv-python tensorflow numpy speechrecognition pyaudio pvrecorder

## Requirements
pip install -r requirements.txt

## Additional Installation
pip install gaze-tracking
pip install opencv-python dlib imutils

# Create a virtual environment(------If needed-----)
python -m venv venv
venv\Scripts\activate

## To Run the model
python FaceEye_Train.py

































































<!-- # To create dataset Run :
python data.py

# To Train and Extract the model Run :
python train.py

# To predict the realtime face liveness
python prediction.py -->
