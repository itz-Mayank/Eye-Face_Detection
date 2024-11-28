# Installation

## Tenserflow setup

![image](https://github.com/user-attachments/assets/948f95d8-2636-46fd-aedd-96a6976a6120)

Download and install the NVIDIA CUDA Toolkit --  https://developer.nvidia.com/cuda-toolkit
Download and install cuDNN --  https://developer.nvidia.com/cudnn

## Install VS with C++
![image].(https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSFeaturesPage&passive=true&tailored=cplus&cid=2031#cplusplus)

## Necessary To set the Model
Dir. : Face live/gaze_tracking
Create a folder named as "trained_models" inside "Dir. : Face live/gaze_tracking".
Download this file -- (https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
& move it in "trained_models" folder.

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
