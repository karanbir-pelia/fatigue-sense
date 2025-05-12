# Fatigue Sense - Driver Monitoring System

## Overview

Fatigue Sense is a real-time driver monitoring system designed to detect driver fatigue, improper posture, and unsafe hand positions. The system uses computer vision and machine learning to analyze video input and provide alerts when potentially dangerous driving behaviors are detected.

## Features

-   **Fatigue Detection**: Monitors driver's face for signs of drowsiness using a trained deep learning model
-   **Posture Analysis**: Detects improper driving posture using MediaPipe pose estimation
-   **Hand Position Monitoring**: Ensures driver's hands remain on the steering wheel
-   **Real-time Alerts**: Visual and audible warnings when unsafe conditions are detected
-   **Comprehensive Visualization**: On-screen display of detection results and alerts

## Requirements

-   Python 3.8+
-   Webcam or video input source
-   Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/karanbir-pelia/fatigue-sense.git
    cd fatigue-sense
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Ensure the model and sound files are in the correct directories:
    - The fatigue detection model should be in the `models/` directory
    - The alert sound file should be in the `sounds/` directory

## Usage

Run the main script to start the monitoring system:

```
python main.py
```

The system will use your default webcam (camera index 0). To use a different camera or video file, modify the `video_source` parameter in the main script.

### Controls

-   Press 'q' to quit the application

## Project Structure

```
fatigue-sense/
├── main.py                 # Main application entry point
├── models/                 # Directory for ML models
│   └── best_fatigue_model.keras  # Trained fatigue detection model
├── modules/                # System components
│   ├── __init__.py
│   ├── alert_system.py     # Audio alert functionality
│   ├── fatigue_detector.py # Fatigue detection using ML model
│   ├── hand_detector.py    # Hand position detection
│   ├── posture_analyzer.py # Driver posture analysis
│   └── visualizer.py       # On-screen visualization
├── sounds/                 # Audio resources
│   └── alert_sound.wav     # Alert sound file
└── requirements.txt        # Project dependencies
```

## How It Works

1. **Fatigue Detection**: Uses a Keras model to analyze facial features and detect signs of fatigue
2. **Posture Analysis**: Uses MediaPipe Pose to track upper body landmarks and detect improper posture
3. **Hand Detection**: Uses MediaPipe Hands to track hand positions relative to the steering wheel
4. **Alert System**: Triggers audio alerts when unsafe conditions are detected
5. **Visualization**: Displays status information and alerts on the video feed

## Customization

-   Adjust detection thresholds in the respective module files
-   Replace the alert sound with your own WAV file
-   Fine-tune the fatigue detection model for better performance

## Acknowledgements

-   TensorFlow and Keras for the deep learning framework
-   MediaPipe for the pose and hand detection capabilities
-   OpenCV for computer vision functionality
-   Pygame for audio playback
