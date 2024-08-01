## Facial Expression Recognition

This repository contains a Python project that performs facial expression recognition using a custom-built convolutional neural network (CNN) model and OpenCV. The project processes video input to detect faces and classify their expressions into one of seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

https://github.com/user-attachments/assets/5aa0ecb2-d23b-458b-b804-0d5cdd1ad4d9

### Features

- Custom-built CNN model for facial expression recognition.
- Detects faces in video frames using Haar Cascade Classifier.
- Displays real-time facial expression predictions on video frames.
- Supports both video file input and real-time camera feed.
- Saves the processed video with bounding boxes and predicted expressions.

### Requirements

- Python 3.9 or above
- OpenCV
- NumPy
- Keras
- TensorFlow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HussainNasirKhan/Facial-Expression-Recognition.git
   cd Facial-Expression-Recognition

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Download the dataset from the following link: https://www.kaggle.com/msambare/fer2013 and place the training and testing images in the dataset/train and dataset/test directories, respectively.

4. Train the model:
   ```bash
   python train.py

### Usage

To run the facial expression recognition on a video file, use the following command:

  ```bash
  python test.py
