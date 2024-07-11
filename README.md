# SecureVision : Suspicious Activity Detection using YOLO and Keras

This project is an intelligent video surveillance system that detects suspicious activities in a video feed, specifically focusing on weapon detection and abnormal activity detection using YOLO and a trained Keras model.

## Features

- Detects the presence of weapons in video frames.
- Identifies abnormal activities in the video feed.
- Utilizes YOLO for real-time object detection.
- Uses a pre-trained Keras model for abnormal activity detection.
- Provides a graphical user interface (GUI) for easy video file selection and detection results display.


## Requirements

- Python 3.6 or higher
- The following Python libraries:
  - `tkinter`
  - `PIL`
  - `cv2`
  - `numpy`
  - `keras`
  - `imutils`
  - `psutil`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/omkarpawar201/Secure-vision.git
   cd Secure-vision

2. Ensure you have the YOLO weights and configuration files:

 - yolov3_training_2000.weights
 - yolov3_testing.cfg

3. Ensure you have the pre-trained Keras model file:

 - saved_model.keras

## Usage

1. **Run the Python script:**
    ```sh
    python reco.py
    ```

2. In the GUI:

 - Click "Browse" to select a video file.
 - Click "Detect" to start the detection process.

3. The system will analyze the video and display the detection results. If a weapon is detected, it will be highlighted in the video frame. If abnormal activity is detected, a corresponding message will be displayed.