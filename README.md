# Temi Fall Detection & Monitoring System

---

## 📦 Features

### Fall Detection
- Detects falls using 3 methods:
  - Bounding box (aspect ratio)
  - Pose keypoint verticality
  - Bottom-half-only keypoints
  - Full consensus if all agree
- Per-person tracking using centroids
- Fall cooldown + persistence logic
- Tracks unique fallers by ID

### Video Streaming
- Receives live stream via WebRTC
- Fallback placeholder when offline
- Record stream to MP4 on demand
- Timestamps video files

### Sensor Data
- Receives JSON from Temi DataChannel or REST
- Plots sensor values on live bar chart
- Records to CSV if should_record=True

### Daily Metrics Reporting
- Metrics tracked:
  - Sensor triggers, CSV rows
  - All fall types + frames processed
  - Unique fallers and people seen
  - Stream state uptime (live/frozen/offline)
  - API & WebRTC connection counts
- Generates styled HTML report
- Sends email via SMTP at configured time
- Supports manual `/send-report-now` route

### Web UI
- Clean, responsive dashboard
- Tailwind CSS layout with live video
- Real-time metric updates
- Sensor chart via Chart.js
- Recording buttons

---

## 🚀 Quick Start

### 1. Clone Repo & Install Requirements
```bash
git clone https://github.com/your-org/temi-monitoring.git
cd temi-monitoring
pip install -r requirements.txt
```

### 2. Set Up `.env`
```env
TEMI_EMAIL_SENDER=your_email@gmail.com
TEMI_EMAIL_PASSWORD=your_app_password
TEMI_EMAIL_RECIPIENTS=you@example.com,team@example.com
TEMI_REPORT_TIME=20:00
```

### 3. Start the Server
```bash
python server.py
```

- Flask app on `http://localhost:8133`
- WebRTC signaling server on port `5432`

---

## 📁 Project Structure
```
.
├── server.py              # Main WebRTC + Flask server
├── daily_reports.py       # Metric tracking and email reporting
├── yolo_fall_detection.py # FallDetector logic with YOLO + pose
├── fall_tracking.py       # Centroid tracking with cooldowns
├── templates/
│   └── index.html         # Tailwind UI
├── Temi_Sensor_Data/      # Recorded CSVs
├── Temi_VODs/             # Recorded videos
└── static/                # Offline placeholder image
```

## 📐 System Architecture

```text
                        +-------------------+
                        |   Temi Robot      |
                        | (Camera + Sensors)|
                        +--------+----------+
                                 |
                          WebRTC / Sensor Stream
                                 |
              +------------------v-------------------+
              |              server.py               |
              |--------------------------------------|
              | - WebRTC video receiver              |
              | - Flask web API + video feed         |
              | - Calls FallDetector + CSV writer    |
              | - Exposes: /offer, /status, /metrics |
              +------------------+-------------------+
                                 |
                   +-------------v-------------+
                   |      FallDetector         |
                   |---------------------------|
                   | - Handles 3 fall methods  |
                   | - Uses YOLO detections    |
                   | - Calls FallTracker       |
                   +-------------+-------------+
                                 |
                     +-----------v------------+
                     |      FallTracker       |
                     |------------------------|
                     | - Tracks centroids     |
                     | - Assigns temp IDs     |
                     | - Tracks cooldowns     |
                     | - Persists fall states |
                     +-----------+------------+
                                 |
           +---------------------v---------------------+
           |            daily_reports.py               |
           |-------------------------------------------|
           | - Metrics dictionary                      |
           | - Tracks API usage, fall events, uptime   |
           | - Generates + emails daily HTML report    |
           | - Scheduled via threading.Timer           |
           +---------------------+---------------------+
                                 |
                      +----------v---------+
                      |  index.html (UI)   |
                      |--------------------|
                      | - TailwindCSS UI   |
                      | - Live stream view |
                      | - Real-time charts |
                      | - Metric dashboard |
                      +--------------------+
```
