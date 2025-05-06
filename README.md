
---
# Temi Fall Detection & Monitoring System

### By: Zach Bernard, Intern @CAAI from Nov 2024 - May 2025

---

## 📦 Features

### Fall Detection

* Detects falls using 3 YOLO-based methods:

  * Bounding box (aspect ratio)
  * Pose keypoint vertical alignment
  * Bottom-half keypoint analysis
  * Full consensus if all agree
* Centroid-based person tracking
* Fall cooldown + persistence state tracking
* Unique faller counter

### Video Streaming

* WebRTC-based live video feed from Temi
* Fallback image when offline
* Record MP4 video on demand with timestamped filenames
* Toggle glasses button for the memes

### Sensor Data Collection

* Receives JSON sensor payloads from Temi (DataChannel or REST)
* Live bar chart of sensor values in browser
* Saves to CSV when `should_record=True`is received from Temi

### Embedding & Training Pipeline

* Autoencoder-based smell embedding system
* Daily ambient + target embedding generation
* Fine-tunes on today's ambient data if available
* Generates t-SNE plot visualizing clustering
* Silhouette score and reconstruction loss calculation
* Optional `--skip-save` mode for testing without file writes
* Combines historical embeddings for progressive model tracking

### Daily Metrics Reporting

* Tracks:

  * Sensor triggers, CSV row count
  * All fall detection types
  * Frames processed, people detected
  * Unique fallers
  * Stream uptime (live/frozen/offline)
  * API/WebRTC connection metrics
* HTML report emailed daily via SMTP
* Manually trigger via `/send-report-now` route, SENDS TO ALL ON `TEMI_EMAIL_RECIPIENTS`.
* E-mail list in `.env`

### Web UI

* Tailwind CSS dashboard
* Live stream viewer + recording controls
* Real-time fall & metric updates
* Sensor data chart (Chart.js)
* Additional frontend integration in TV-react repo

---

## 🚀 Quick Start

### 1. Clone Repo & Install Requirements

```bash
git clone https://github.com/innovationcore/temiStreamCatch.git
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

---

## 🔁 Scheduled Tasks

Both the embedding pipeline and report generation are scheduled daily:

* Embedding pipeline runs 5 minutes before `TEMI_REPORT_TIME`
* Report generates and sends at `TEMI_REPORT_TIME`

### Manual Trigger (SENDS TO ALL ON `TEMI_EMAIL_RECIPIENTS`)

To manually test pipeline + report without saving:

```bash
http://localhost:8133/send-report-now
```

To run and save output:

```bash
http://localhost:8133/send-report-now?skip_save=false
```

---

## 🧪 Training Pipeline CLI

You can also run the embedding pipeline from the command line:

```bash
python training_pipeline.py --test-mode       # Save output locally but show debug output
python training_pipeline.py --skip-save       # Skip saving, useful for testing
```

---

## 🗂 Project Structure

```
.
├── server.py              # Flask server + WebRTC receiver
├── daily_reports.py       # Metrics tracking + report email logic
├── training_pipeline.py   # Autoencoder training and t-SNE generation
├── record_smells.py       # Receives smell data over WebRTC or REST
├── yolo_fall_detection.py # Pose and box-based fall detection
├── fall_tracking.py       # Centroid tracker for fall state
├── templates/index.html   # Frontend dashboard (Tailwind UI)
├── Temi_Sensor_Data/      # CSV logs of sensor data
├── Temi_VODs/             # Recorded video files
├── embeddings/            # Daily + master embedding vectors and metadata
├── visualizations/        # t-SNE plots
├── models/                # Trained autoencoder weights
└── static/                # Offline fallback images
```

---

## 📐 System Architecture

```text
                      +----------------------+
                      |     Temi Robot       |
                      |  (Video + Sensors)   |
                      +----------+-----------+
                                 |
                       WebRTC Video + Sensor Stream
                                 |
                 +---------------v---------------+
                 |            server.py          |
                 |-------------------------------|
                 | - WebRTC video handling       |
                 | - REST API for sensor input   |
                 | - Fall detection + CSV writer |
                 | - Routes: /offer, /status...  |
                 +---------------+---------------+
                                 |
                 +---------------v----------------+
                 |       yolo_fall_detection.py   |
                 |--------------------------------|
                 | - 3-mode fall detection        |
                 | - Pose, box, and bottom logic  |
                 +---------------+----------------+
                                 |
        +------------------------v-------------------------+
        |             training_pipeline.py                 |
        |--------------------------------------------------|
        | - Autoencoder model for embeddings               |
        | - Combines daily and historical embeddings       |
        | - Generates visualizations + metrics             |
        +------------------------+-------------------------+
                                 |
        +------------------------v-------------------------+
        |               daily_reports.py                   |
        |--------------------------------------------------|
        | - Tracks API/fall metrics                        |
        | - HTML report generation                         |
        | - Sends via email daily or on demand             |
        +------------------------+-------------------------+
```

## TODO
* Ensure training pipeline accuracy day to day
  
  Make sure that when finetuning with a new days data that the report ebing sent includes the cumulative training data when makingthe t-sne.

* Record better and more useful target data
  
  The current target data is from a smell sensor placed right above a coffee cup so this might not be the best data to test against (although getting some promising results so far).

* Front end graph adjustments

  The graph on the frontend could use some adjustments, sometimes throughout the day the values are off the charts so it might be better to decrease the `    const sensitivityFactor = 100` (100x).

* Create single feed toggle

  Create a front end toggle mode to turn the video render to only display one clean video stream, similar to glasses mode. this allows for the best recording/viewing experience for Demos and such.

* Video recording FPS

  The FPS of the recorded videos isnt quite right, when I tested it was always either too fast or too slow, so some finetuning by way of frame recieved averaging might be nice.

* E-mail list Form

  Currently the only way to add users to the email list is to manually go into the `.env` file and append the new user to the `TEMI_EMAIL_RECIPIENTS` list. This is fine but for demo purposes it might be nice for people to add themselves from the TV (Make sure to account for people wanting to remove themselves).