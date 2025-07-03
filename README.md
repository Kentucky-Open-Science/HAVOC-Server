# HAVOC-Server: Temi Monitoring System

This project is a monitoring system for the Healthcare Assistant with Video, Olfaction, and Conversation (HAVOC) program running on the Temi robot. It uses computer vision and machine learning to detect falls, classify smells, and generate daily reports. It provides real-time metrics and visualizations through a web interface, making it ideal for applications in healthcare, safety monitoring, or robotics research.

## Features

- **Fall Detection**: Uses YOLO (You Only Look Once) for real-time detection of falls with multiple methods (bounding box, pose keypoints, and bottom fraction).
- **Smell Classification**: Employs a K-Nearest Neighbors (KNN) model to classify smells based on sensor data.
- **Real-Time Streaming**: Streams live video and sensor data from the Temi robot via WebRTC.
- **Daily Reports**: Generates and sends daily reports with metrics and t-SNE visualizations of smell data.
- **Web Interface**: Provides an interactive dashboard for monitoring, recording, and visualizing data.

## Installation

1. **Clone the repository**:

   ```bash
   git clone [insert link]
   cd HAVOC-Server
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Additional Setup**:

   - Download the YOLO model weights (`yolo11n-pose.pt`) and place them in the `yolo_weights/` directory.

   - Set up environment variables in a `.env` file:

     ```plaintext
     TEMI_EMAIL_SENDER=your-email@gmail.com
     TEMI_EMAIL_PASSWORD=your-app-specific-password
     TEMI_EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
     TEMI_REPORT_TIME=20:00
     ```

   - Ensure the `static/newSensor_training.csv` file exists for smell classification training data.

## Running the Application

To start the server, run:

```bash
python server.py
```

- The Flask server will start on `http://0.0.0.0:8133`, serving the web interface.
- The aiohttp signaling server for WebRTC will start on `http://0.0.0.0:5432`.

Open your web browser and navigate to `http://localhost:8133` to access the web interface.

## Usage

The web interface (`index.html`) provides several interactive sections:

- **Stream**: Displays the live video feed from the Temi robot, with fall detection overlays in four quadrants (Box, Pose, Bottom, Combined).
- **Recording**: Start/stop video recording using the buttons. Recordings are saved in `Temi_VODs/`.
- **Vision Modes**: Toggle "Glasses" (adds a fun glasses/mustache overlay) or "Fullscreen" modes.
- **Map**: Shows the Temi robot's location on a suite map, with dots indicating smell detections (requires the map image from Temi).
- **Metrics**: Displays real-time metrics, including:
  - People detected today
  - Falls detected (Box, Pose, Bottom, Full Consensus)
  - Frames processed
  - Smell classifications
- **Smell Data**: Shows the last two hours of smell data with timestamps, updated via Server-Sent Events (SSE).


## System Architecture

The system is composed of several key modules:

- `yolo_fall_detection.py`: Implements fall detection using YOLO, with methods for bounding box, pose keypoints, and bottom fraction analysis.
- `report_visualizer.py`: Generates t-SNE visualizations comparing daily sensor data with training data.
- `plot_points_pixel.py`: Plots the robot’s position and smell detections on a map image.
- `smell_classifier.py`: Classifies smells using a KNN model trained on sensor data.
- `server.py`: Manages WebRTC streaming, Flask web server, and real-time data processing.
- `fall_tracking.py`: Tracks fall events over time, maintaining unique faller counts.
- `daily_reports.py`: Generates and sends daily reports via email, including metrics and visualizations.
- `index.html`: The frontend interface providing a user-friendly dashboard.