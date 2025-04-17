import os
import csv
import smtplib
import shutil
import threading
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, jsonify
from dotenv import load_dotenv

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

# === CONFIGURATION ===
SENSOR_CSV_PATH = "Temi_Sensor_Data/sensor_data_master.csv"
VIDEO_DIR = "Temi_VODs"
EMAIL_SENDER = os.getenv("TEMI_EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("TEMI_EMAIL_PASSWORD")
EMAIL_RECIPIENTS = os.getenv("TEMI_EMAIL_RECIPIENTS", "").split(",")
REPORT_TIME = os.getenv("TEMI_REPORT_TIME", "20:00")  # Default to 8 PM

# === METRIC TRACKERS ===
metrics = {
    "record_triggers_today": 0,
    "new_csv_rows_today": 0,
    "total_csv_rows": 0,
    "frames_processed": 0,
    "falls_box": 0,
    "falls_pose": 0,
    "falls_bottom": 0,
    "falls_full": 0,
    "people_detected_today": 0,
    "stream_offline_count": 0,
    "stream_frozen_count": 0,
    "stream_live_seconds": 0,
    "stream_frozen_seconds": 0,
    "stream_offline_seconds": 0,
    "webrtc_connections": 0,
    "http_api_calls": 0
}

# === TRACKER UPDATE FUNCTIONS (called from server.py) ===
def increment(key):
    if key in metrics:
        metrics[key] += 1

def add_time(key, seconds):
    if key in metrics:
        metrics[key] += seconds

def set_total_csv_rows(count):
    metrics["total_csv_rows"] = count

def reset_daily_metrics():
    for key in metrics:
        if key.endswith("today") or key in ["frames_processed", "falls_box", "falls_pose", "falls_bottom", "falls_full",
                                             "people_detected_today", "stream_offline_count", "stream_frozen_count",
                                             "stream_live_seconds", "stream_frozen_seconds", "stream_offline_seconds",
                                             "webrtc_connections", "http_api_calls"]:
            metrics[key] = 0

# === METRIC EXTRACTION ===
def calculate_file_size(path):
    return round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else 0

def get_disk_space():
    total, used, free = shutil.disk_usage("/")
    return round(free / (1024 ** 3), 2)  # in GB

def update_csv_metrics():
    today = datetime.now().date()
    row_count = 0
    today_count = 0
    if os.path.exists(SENSOR_CSV_PATH):
        with open(SENSOR_CSV_PATH, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                if 'timestamp' in row:
                    ts = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
                    if ts.date() == today:
                        today_count += 1
    metrics["new_csv_rows_today"] = today_count
    metrics["total_csv_rows"] = row_count

def get_video_metrics():
    count_total = 0
    count_today = 0
    total_size = 0
    today_str = datetime.now().strftime("%Y%m%d")

    if os.path.exists(VIDEO_DIR):
        for f in os.listdir(VIDEO_DIR):
            path = os.path.join(VIDEO_DIR, f)
            if f.endswith(".mp4") and os.path.isfile(path):
                count_total += 1
                total_size += os.path.getsize(path)

                # Check if the filename has today's date
                if f.startswith("recorded_video_"):
                    parts = f.replace("recorded_video_", "").split("_")
                    if parts and parts[0] == today_str:
                        count_today += 1

    total_size_mb = round(total_size / (1024 * 1024), 2)  # in MB
    return count_total, count_today, total_size_mb

def format_seconds(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


# === EMAIL REPORT GENERATION ===
def generate_html_report():
    update_csv_metrics()
    count_total, count_today, total_size = get_video_metrics()
    disk_free = get_disk_space()

    html = f"""
    <html>
    <body style='font-family:Arial; background-color:#f4f4f4; padding:20px;'>
        <h2 style='color:#2c3e50;'>🤖 Temi Server Daily Report - {datetime.now().strftime('%Y-%m-%d')}</h2>

        <h3>📊 Data Collection</h3>
        <ul>
            <li><strong>Sensor recordings triggered:</strong> {metrics['record_triggers_today']}</li>
            <li><strong>New rows added today:</strong> {metrics['new_csv_rows_today']}</li>
            <li><strong>Total rows in CSV:</strong> {metrics['total_csv_rows']}</li>
        </ul>

        <h3>🎥 Video & Fall Detection</h3>
        <ul>
        <li><strong>Frames processed:</strong> {metrics['frames_processed']}</li>
        <li><strong>Falls (Box):</strong> {metrics['falls_box']}</li>
        <li><strong>Falls (Pose):</strong> {metrics['falls_pose']}</li>
        <li><strong>Falls (Bottom):</strong> {metrics['falls_bottom']}</li>
        <li><strong>Falls (Full Consensus):</strong> {metrics['falls_full']}</li>
        <li><strong>People detected today:</strong> {metrics['people_detected_today']}</li>
        </ul>

        <h3>📶 Stream/Uptime</h3>
        <ul>
            <li><strong>Stream offline events:</strong> {metrics['stream_offline_count']}</li>
            <li><strong>Stream frozen events:</strong> {metrics['stream_frozen_count']}</li>
            <li><strong>Live time:</strong> {format_seconds(metrics['stream_live_seconds'])}</li>
            <li><strong>Frozen time:</strong> {format_seconds(metrics['stream_frozen_seconds'])}</li>
            <li><strong>Offline time:</strong> {format_seconds(metrics['stream_offline_seconds'])}</li>
        </ul>


        <h3>🌐 API Activity</h3>
        <ul>
            <li><strong>WebRTC connections:</strong> {metrics['webrtc_connections']}</li>
            <li><strong>HTTP API calls:</strong> {metrics['http_api_calls']}</li>
        </ul>

        <h3>🗂 File & Storage</h3>
        <ul>
            <li><strong>CSV size:</strong> {calculate_file_size(SENSOR_CSV_PATH)} MB</li>
            <li><strong>Videos saved today:</strong> {count_today}</li>
            <li><strong>Total videos saved:</strong> {count_total}</li>
            <li><strong>Total video size:</strong> {total_size} MB</li>
            <li><strong>Disk space remaining:</strong> {disk_free} GB</li>
        </ul>


        <p style='color:#95a5a6;'>Report generated automatically by Temi server at {datetime.now().strftime('%H:%M:%S')}.</p>
    </body>
    </html>
    """
    return html

def send_email_report():
    subject = f"Temi Server Report - {datetime.now().strftime('%Y-%m-%d')}"
    html = generate_html_report()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(EMAIL_RECIPIENTS)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
        print("✅ Daily report sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send daily report: {e}")
        return False

# === SCHEDULER ===
def schedule_daily_report():
    now = datetime.now()
    hour, minute = map(int, REPORT_TIME.split(":"))
    run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if run_time < now:
        run_time += timedelta(days=1)

    delay = (run_time - now).total_seconds()
    threading.Timer(delay, run_and_reschedule).start()
    print(f"📅 Daily report scheduled in {round(delay / 60)} minutes.")

def run_and_reschedule():
    send_email_report()
    reset_daily_metrics()
    schedule_daily_report()
