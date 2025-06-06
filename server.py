import asyncio
import cv2
import logging
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from flask import Flask, Response, render_template, jsonify, request
from flask import Flask
from flask_cors import CORS

import threading
import time
from datetime import datetime, timedelta
import numpy as np
import os
import json
import time
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp, SessionDescription as SDPDescription
from daily_reports import (
    send_email_report,
    schedule_daily_report,
    increment,
    add_time,
    update_csv_metrics,
    metrics,
)
from training_pipeline import schedule_embedding_pipeline, run_embedding_pipeline
from yolo_fall_detection import FallDetector  # Import the FallDetector class
from smell_classifier import SmellClassifier
from queue import Queue
import schedule

sse_queue = Queue()

# Set logging to INFO level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("temi-stream")

# WebRTC globals
relay = MediaRelay()
pcs = set()
frame_holder = {'frame': None}

# Frame processing globals
last_pts = None
freeze_detected_time = None
duplicate_frame_count = 0
duplicate_threshold = 5
freeze_threshold = 5.0
last_frame_time = "N/A"
last_should_record = False
vision_mode = {"glasses": False, "fullscreen": False, "last_toggle": 0}


# Load offline placeholder image
filler_image_path = os.path.join('static', 'temiFace_screen_saver.png')
if not os.path.exists(filler_image_path):
    logger.error(f"Filler image not found at {filler_image_path}")
    offline_bytes = b"..."
else:
    filler_img = cv2.imread(filler_image_path)
    if filler_img is None:
        logger.error(f"Failed to load filler image from {filler_image_path}")
        offline_bytes = b"..."
    else:
        ret, buffer = cv2.imencode('.jpg', filler_img)
        if not ret:
            logger.error("Failed to encode filler image to JPEG")
            offline_bytes = b"..."
        else:
            offline_bytes = buffer.tobytes()
            logger.info(f"Loaded filler image from {filler_image_path}, size={len(offline_bytes)} bytes")

# Instantiate FallDetector and SmellClassifier globally
fall_detector = FallDetector()
smell_classifier = SmellClassifier()

# Store classified smell data
classified_data = []

# --- SSE CHANGE: Helper function to push metric updates ---
def push_metrics_update():
    """Puts the current metrics onto the SSE queue."""
    metrics_payload = {
        "event": "metrics_update",
        "data": metrics
    }
    sse_queue.put(json.dumps(metrics_payload))

# -------- aiohttp WebRTC server ----------
app = web.Application()

class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        frame_holder['frame'] = frame
        if last_pts is None:
            logger.info(f"WebRTC stream started, first frame received, pts={frame.pts}")
        return frame

async def create_peer_connection():
    peer = RTCPeerConnection()

    @peer.on("track")
    async def on_track(track):
        if track.kind == "video":
            codec = getattr(track, 'codec', None)
            codec_mime = codec.mimeType.lower() if codec else ''
            if codec_mime == 'video/rtx':
                logger.info("Ignoring RTX (retransmission) track.")
                return

            video_track = VideoProcessorTrack(relay.subscribe(track))
            peer.addTrack(video_track)

            async def consume_track():
                try:
                    while True:
                        frame = await video_track.recv()
                        logger.debug("Frame consumed successfully.")
                except Exception as e:
                    logger.error(f"Error consuming video track: {e}")

            video_track = VideoProcessorTrack(relay.subscribe(track))
            peer.addTrack(video_track)
            asyncio.create_task(consume_track())

    @peer.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"📡 DataChannel received: {channel.label}")

        @channel.on("message")
        def on_message(message_str):
            global last_should_record
            global latest_sensor_data
            global classified_data

            try:
                data_payload = json.loads(message_str)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                x_position, y_position = None, None  # Default to None

                # CHANGE: Handle position updates first, regardless of other data
                if 'current_position' in data_payload and isinstance(data_payload.get('current_position'), dict):
                    pos_data = data_payload['current_position']
                    x_pos = pos_data.get('x')
                    y_pos = pos_data.get('y')

                    # Ensure coordinates are valid numbers before processing
                    if isinstance(x_pos, (int, float)) and isinstance(y_pos, (int, float)):
                        x_position, y_position = x_pos, y_pos  # Store valid coordinates
                        latest_sensor_data['current_position'] = pos_data

                        # Send the real-time position update for the moving dot
                        position_event_payload = {
                            "event": "robot_position_update",
                            "data": {"x": x_position, "y": y_position}
                        }
                        sse_queue.put(json.dumps(position_event_payload))
                        logger.info(f"Pushed position update: x={x_position}, y={y_position}")

                # CHANGE: Handle sensor value updates separately
                if 'values' in data_payload:
                    sensor_values = data_payload.get('values')
                    should_record_from_payload = data_payload.get('should_record', False)
                    formatted_values = format_data(sensor_values)

                    logging.info(f"[{timestamp}] DataChannel: Received sensor values: {sensor_values}")
                    logging.info(f"[{timestamp}] DataChannel: formatted sensor values: {formatted_values}")
                    # Update global store
                    latest_sensor_data.update({
                        'data': formatted_values,
                        'should_record': should_record_from_payload,
                        'timestamp': timestamp
                    })
                    logger.info(
                        f"[{timestamp}] DataChannel: Received sensor data. Record flag: {should_record_from_payload}")

                    # Push sensor data to the line graph
                    if formatted_values:
                        sensor_event_payload = {
                            "event": "sensor_update",
                            "data": {
                                "timestamp": timestamp,
                                "values": formatted_values
                            }
                        }
                        sse_queue.put(json.dumps(sensor_event_payload))

                    # Logic for recording data and classified dots (which requires sensor values)
                    frame_filename = None
                    if should_record_from_payload:
                        if not last_should_record:
                            increment("record_triggers_today")
                            push_metrics_update()

                        # Save frame if available
                        if frame_holder['frame'] is not None and not isinstance(frame_holder['frame'], bytes):
                            image_dir = os.path.join("Temi_Sensor_Data", "frames")
                            os.makedirs(image_dir, exist_ok=True)
                            frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                            frame_filename = f"frame_{frame_timestamp}.jpg"
                            frame_path = os.path.join(image_dir, frame_filename)
                            try:
                                img_to_save = frame_holder['frame'].to_ndarray(format="bgr24")
                                cv2.imwrite(frame_path, img_to_save)
                                logger.info(f"Saved frame: {frame_path}")
                            except Exception as e:
                                logger.error(f"Error saving frame: {e}")
                                frame_filename = None

                        latest_sensor_data['frame_filename'] = frame_filename
                        record_sensor_data_to_csv(sensor_values, timestamp, x_position, y_position, frame_filename)
                        update_csv_metrics()

                        # Classify and push map dot event (requires position AND values)
                        if x_position is not None and y_position is not None and formatted_values:
                            classification = smell_classifier.classify_sensor_data(formatted_values)
                            logger.info(f"[{timestamp}] DataChannel: Classified smell data: {classification}")

                            map_dot_payload = {
                                "event": "map_dot_update",
                                "data": {
                                    'x': x_position,
                                    'y': y_position,
                                    'class': classification,
                                    'timestamp': timestamp
                                }
                            }
                            sse_queue.put(json.dumps(map_dot_payload))

                    last_should_record = should_record_from_payload

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode DataChannel JSON message: {message_str}. Error: {e}")
            except Exception as e:
                logger.error(f"Failed to process DataChannel message: {e}")

    pcs.add(peer)
    increment("webrtc_connections")
    push_metrics_update()
    return peer

# get the fifteen channels from the raw sensor data
def format_data(data):
    # check if data is wrong length or contains any non-numeric values
    if (len(data) != 66) or any(not isinstance(x, (int, float)) for x in data):
        return None

    # Define the channel map
    channel_map = [999, 999, 11, 11, 11, 3, 3, 3, 2, 2, 2, 1, 1, 1, 999, 999,
                   999, 999, 999, 999, 999, 9, 9, 9, 6, 6, 6, 5, 5, 5, 999, 999,
                   999, 999, 14, 14, 14, 10, 10, 10, 7, 7, 7, 4, 4, 4, 999, 999,
                   999, 999, 15, 15, 15, 13, 13, 13, 12, 12, 12, 8, 8, 8, 999, 999,
                   16, 17]

    # Filter out entries in data where the corresponding channel_map value is 999
    filtered_data = [x for x, ch in zip(data, channel_map) if ch != 999]
    num_groups = (len(filtered_data) - 2) // 3  # Calculate number of groups of 3, excluding temp and humidity
    averaged_data = []
    # Average every three values together, excluding the last two (temp and humidity)
    for i in range(num_groups):
        start = i * 3
        end = start + 3
        group_average = sum(filtered_data[start:end]) / 3
        averaged_data.append(group_average)

    # Add the last two values (temperature and humidity) without averaging
    averaged_data.append(filtered_data[-2])  # Temperature
    averaged_data.append(filtered_data[-1])  # Humidity
    return averaged_data

async def offer(request):
    params = await request.json()
    original_offer_sdp = params["sdp"]
    offer_type = params["type"]

    # Manually remove all RTX lines from SDP
    sdp_lines = original_offer_sdp.splitlines()
    filtered_sdp_lines = []
    skip_payload_types = set()

    for line in sdp_lines:
        if 'a=rtpmap' in line and 'rtx' in line.lower():
            # Extract RTX payload type to skip it later
            payload_type = line.split(' ')[0].split(':')[1]
            skip_payload_types.add(payload_type)
            continue  # skip adding RTX lines directly
        if line.startswith('a=fmtp:'):
            fmtp_payload_type = line.split(' ')[0].split(':')[1]
            if fmtp_payload_type in skip_payload_types:
                continue  # skip RTX fmtp lines
        filtered_sdp_lines.append(line)

    # Remove RTX payload types from m=video line payload types
    final_sdp_lines = []
    for line in filtered_sdp_lines:
        if line.startswith('m=video'):
            parts = line.split(' ')
            # Keep payload types that are not in skip_payload_types
            m_line_payload_types = [pt for pt in parts[3:] if pt not in skip_payload_types]
            new_m_line = ' '.join(parts[:3] + m_line_payload_types)
            final_sdp_lines.append(new_m_line)
        else:
            final_sdp_lines.append(line)

    cleaned_offer_sdp = '\r\n'.join(final_sdp_lines) + '\r\n'

    # Use cleaned offer without RTX payloads
    offer = RTCSessionDescription(sdp=cleaned_offer_sdp, type=offer_type)

    peer = await create_peer_connection()
    await peer.setRemoteDescription(offer)
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    #REPORT: increment every api call
    increment("http_api_calls")
    push_metrics_update()

    return web.json_response({
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })

app.router.add_post("/offer", offer)
# -------- Flask video feed server ----------
flask_app = Flask(__name__)
CORS(flask_app)  # <-- Enable CORS for all routes

@flask_app.route('/')
def index():
    frame = frame_holder.get('frame', offline_bytes)
    current_time = time.time()
    if isinstance(frame, bytes):
        stream_status = "Offline"
    else:
        if last_pts is None:
            stream_status = "Live"
        elif freeze_detected_time and (duplicate_frame_count > duplicate_threshold or
                                      current_time - freeze_detected_time > freeze_threshold):
            stream_status = "Frozen"
        else:
            stream_status = "Live"
    return render_template('index.html', stream_status=stream_status, last_frame_time=last_frame_time)

@flask_app.route('/set-vision-mode', methods=['POST'])
def set_vision_mode():
    global vision_mode
    data = request.get_json()
    mode = data.get('mode')  # 'glasses' or 'fullscreen'

    if mode == 'glasses':
        vision_mode['glasses'] = True
        vision_mode['fullscreen'] = False
    elif mode == 'fullscreen':
        vision_mode['fullscreen'] = True
        vision_mode['glasses'] = False
    else:
        vision_mode['fullscreen'] = False
        vision_mode['glasses'] = False

    vision_mode['last_toggle'] = time.time()
    return jsonify({"status": "ok", "vision_mode": vision_mode})


# --- SSE CHANGE: Create the new streaming endpoint ---
@flask_app.route('/stream-updates')
def stream_updates():
    def event_stream():
        while True:
            # Block until a message is available
            message = sse_queue.get()
            # The message should be a JSON string with 'event' and 'data' keys
            # We need to parse it to format the SSE message correctly
            try:
                payload = json.loads(message)
                event_type = payload.get('event', 'message')
                event_data = json.dumps(payload.get('data'))
                yield f"event: {event_type}\ndata: {event_data}\n\n"
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Could not format SSE message from queue item: {message}")

    return Response(event_stream(), mimetype='text/event-stream')

@flask_app.route('/status')
def get_status():
    frame = frame_holder.get('frame', offline_bytes)
    current_time = time.time()
    if isinstance(frame, bytes):
        stream_status = "Offline"
    else:
        if last_pts is None:
            stream_status = "Live"
        elif freeze_detected_time and (duplicate_frame_count > duplicate_threshold or
                                      current_time - freeze_detected_time > freeze_threshold):
            stream_status = "Frozen"
        else:
            stream_status = "Live"

    #REPORT: increment every api call
    increment("http_api_calls")

    return jsonify({'stream_status': stream_status, 'last_frame_time': last_frame_time})

@flask_app.route('/video_feed')
def video_feed():
    #REPORT: increment every api call
    increment("http_api_calls")

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Global variable to hold latest sensor data
latest_sensor_data = {"data": None, "timestamp": None, "should_record": False, "current_position": None, "frame_filename": None}

# Alternative endpoint to receive sensor data from the Temi robot, not used in the current setup
# @flask_app.route('/sensor-data', methods=['POST'])
# def sensor_data():
#     global latest_sensor_data
#     data = request.json.get('sensor_data')
#     should_record = request.json.get('should_record', False)
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     current_position = request.json.get('current_position', None)
#
#     x_position = None
#     y_position = None
#     if current_position and isinstance(current_position, dict):
#         x_position = current_position.get('x')
#         y_position = current_position.get('y')
#
#     frame_filename = None
#     if should_record and frame_holder['frame'] is not None and not isinstance(frame_holder['frame'], bytes):
#         # Save the most recent frame
#         image_dir = os.path.join("Temi_Sensor_Data", "frames")
#         os.makedirs(image_dir, exist_ok=True)
#         frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Milliseconds
#         frame_filename = f"frame_{frame_timestamp}.jpg"
#         frame_path = os.path.join(image_dir, frame_filename)
#         try:
#             img_to_save = frame_holder['frame'].to_ndarray(format="bgr24")
#             cv2.imwrite(frame_path, img_to_save)
#             logger.info(f"Saved frame: {frame_path}")
#         except Exception as e:
#             logger.error(f"Error saving frame: {e}")
#             frame_filename = None  # Reset filename if saving failed
#
#
#     latest_sensor_data = {
#         "data": data,
#         "timestamp": timestamp,
#         "should_record": should_record,
#         "current_position": current_position,
#         "frame_filename": frame_filename
#     }
#
#     # # Record to CSV if flag is True
#     # if should_record:                                 <------ Rest API version to record data from another source
#     #     record_sensor_data_to_csv(data, timestamp)
#
#     #REPORT: increment every api call
#     increment("http_api_calls")
#
#     return jsonify({"status": "Sensor data received"}), 200

# @flask_app.route('/get-latest-sensor-data', methods=['GET'])
# def get_latest_sensor_data():
#     #REPORT: increment every api call
#     increment("http_api_calls")
#     return jsonify(latest_sensor_data), 200
#
# Additional imports for recording
video_writer = None
recording = False
record_lock = threading.Lock()

@flask_app.route('/start-recording', methods=['POST'])
def start_recording():
    global video_writer, recording

    with record_lock:
        if recording:
            return jsonify({'status': 'already recording'}), 200

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Temi_VODs/recorded_video_{timestamp}.mp4"
        video_writer = cv2.VideoWriter(filename, fourcc, 12.0, (640, 480)) # changed to 12 fps
        recording = True

    #REPORT: increment every api call
    increment("http_api_calls")

    return jsonify({'status': 'recording started', 'filename': filename}), 200

@flask_app.route('/stop-recording', methods=['POST'])
def stop_recording():
    global video_writer, recording

    with record_lock:
        if not recording:
            return jsonify({'status': 'not recording'}), 200

        recording = False
        video_writer.release()
        video_writer = None

    #REPORT: increment every api call
    increment("http_api_calls")

    return jsonify({'status': 'recording stopped'}), 200

# === MANUAL TRIGGER ROUTE ===
@flask_app.route("/send-report-now", methods=["GET"])
def trigger_report():
    skip_save = request.args.get("skip_save", "true").lower() != "false"
    print(f"🚀 Running embedding pipeline before report (skip_save={skip_save})...")

    data = run_embedding_pipeline(skip_save=skip_save)
    print("📧 Sending report email...")

    # If skip_save is True, pass in-memory data to report
    if skip_save:
        success = send_email_report(data)
    else:
        success = send_email_report()

    increment("http_api_calls")

    return jsonify({"status": "sent" if success else "failed"})


# @flask_app.route('/get-classified-data', methods=['GET'])
# def get_classified_data():
#     increment("http_api_calls")
#     return jsonify(classified_data), 200

@flask_app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

def gen_frames():
    global last_pts, freeze_detected_time, duplicate_frame_count, last_frame_time, video_writer, recording

    last_state = None
    last_state_change_time = time.time()
    last_person_count = 0
    last_person_increment_time = 0
    fall_cooldown = .5
    person_cooldown = .5
    fall_persistence_time = 1.0

    prev_falls = {"box": False, "pose": False, "bottom": False, "full": False}
    last_fall_times = {"box": 0, "pose": 0, "bottom": 0, "full": 0}
    last_seen_fallen = {"box": 0, "pose": 0, "bottom": 0, "full": 0}

    while True:
        time.sleep(0.02)
        frame = frame_holder.get('frame', offline_bytes)

        current_state = "live"
        if isinstance(frame, bytes) or frame is None:
            current_state = "frozen"
        elif freeze_detected_time and (
            duplicate_frame_count > duplicate_threshold or
            time.time() - freeze_detected_time > freeze_threshold):
            current_state = "offline"

        if current_state != last_state:
            if last_state is not None:
                elapsed = int(time.time() - last_state_change_time)
                add_time(f"stream_{last_state}_seconds", elapsed)
            if current_state in ["frozen", "offline"]:
                increment(f"stream_{current_state}_count")
                push_metrics_update()
            last_state = current_state
            last_state_change_time = time.time()
        else:
            elapsed = int(time.time() - last_state_change_time)
            if elapsed >= 1:
                add_time(f"stream_{current_state}_seconds", elapsed)
                last_state_change_time = time.time()

        if isinstance(frame, bytes):
            frame_bytes = frame
            if last_pts is not None:
                logger.info("Stream switched to offline, yielding placeholder")
        elif frame is None:
            frame_bytes = offline_bytes
        else:
            if last_pts is None or frame.pts != last_pts:
                last_pts = frame.pts
                last_frame_time = datetime.now().strftime('%H:%M:%S')
                freeze_detected_time = None
                duplicate_frame_count = 0
                img = frame.to_ndarray(format="bgr24")

                if vision_mode['glasses']:
                    grid_img = fall_detector.draw_glasses_mustache(cv2.resize(img.copy(), (640, 480)))
                    if time.time() - vision_mode['last_toggle'] > 30:
                        vision_mode['glasses'] = False
                elif vision_mode['fullscreen']:
                    grid_img = cv2.resize(img.copy(), (640, 480))
                    if time.time() - vision_mode['last_toggle'] > 60:
                        vision_mode['fullscreen'] = False
                else:
                    height, width = img.shape[:2]
                    half_w, half_h = width // 2, height // 2

                    box_img, box_fallen, person_count, unique_fallers = fall_detector.test_process_frame_box(cv2.resize(img.copy(), (half_w, half_h)))
                    pose_img, pose_fallen = fall_detector.test_process_frame_pose_fall(cv2.resize(img.copy(), (half_w, half_h)))
                    bottom_img, bottom_fallen = fall_detector.bottom_frac_fall_detection(cv2.resize(img.copy(), (half_w, half_h)))
                    combined_img, combined_fallen = fall_detector.combined_frame(cv2.resize(img.copy(), (half_w, half_h)))

                    now = time.time()
                    if person_count > last_person_count and now - last_person_increment_time > person_cooldown:
                        increment("people_detected_today")
                        last_person_increment_time = now
                        push_metrics_update()
                    last_person_count = person_count

                    if box_fallen and now - last_fall_times["box"] > fall_cooldown:
                        increment("falls_box")
                        last_fall_times["box"] = now
                        push_metrics_update()
                    if pose_fallen and not prev_falls["pose"] and now - last_fall_times["pose"] > fall_cooldown:
                        increment("falls_pose")
                        last_fall_times["pose"] = now
                        push_metrics_update()
                    if bottom_fallen and not prev_falls["bottom"] and now - last_fall_times["bottom"] > fall_cooldown:
                        increment("falls_bottom")
                        last_fall_times["bottom"] = now
                        push_metrics_update()
                    if combined_fallen and not prev_falls["full"] and now - last_fall_times["full"] > fall_cooldown:
                        increment("falls_full")
                        last_fall_times["full"] = now
                        push_metrics_update()

                    prev_falls["box"] = box_fallen
                    prev_falls["pose"] = pose_fallen
                    prev_falls["bottom"] = bottom_fallen
                    prev_falls["full"] = combined_fallen

                    top_row = np.hstack((box_img, pose_img))
                    bottom_row = np.hstack((bottom_img, combined_img))
                    grid_img = np.vstack((top_row, bottom_row))

                increment("frames_processed")
                push_metrics_update()

                if recording and video_writer is not None:
                    resized_frame = cv2.resize(grid_img, (640, 480))
                    video_writer.write(resized_frame)

                ret, buffer = cv2.imencode('.jpg', grid_img)
                frame_bytes = buffer.tobytes()
            else:
                if freeze_detected_time is None:
                    freeze_detected_time = time.time()
                elif duplicate_frame_count > duplicate_threshold or time.time() - freeze_detected_time > freeze_threshold:
                    frame_bytes = offline_bytes
                    logger.info(f"Stream frozen, switching to placeholder after {duplicate_frame_count} duplicates, time elapsed={time.time() - freeze_detected_time:.2f}s")
                else:
                    duplicate_frame_count += 1
                    if duplicate_frame_count == duplicate_threshold:
                        logger.warning(f"Duplicate frames detected, count={duplicate_frame_count}")

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



import csv
import os
from datetime import datetime

def record_sensor_data_to_csv(sensor_data, timestamp, x_position=None, y_position=None, frame_filename=None):
    """Record sensor data to a master CSV file for long-term accumulation"""
    if not sensor_data:
        return

    # Create directory if it doesn't exist
    csv_dir = "Temi_Sensor_Data"
    os.makedirs(csv_dir, exist_ok=True)

    # Master file path
    csv_path = os.path.join(csv_dir, "sensor_data_master.csv")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        if isinstance(sensor_data, list):
            if len(sensor_data) != 66:
                logger.warning(f"Invalid sensor data length: {len(sensor_data)} (expected 66). Data not saved.")
                return  # Skip saving invalid data

            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(['timestamp'] + [f'value_{i}' for i in range(66)] + ['x_position', 'y_position', 'frame_filename'])

            writer.writerow([timestamp] + sensor_data + [x_position, y_position, frame_filename])

            # increment("new_csv_rows_today")
            update_csv_metrics()

        elif isinstance(sensor_data, dict):
            if len(sensor_data) != 66:
                logger.warning(f"Invalid sensor data length (dict): {len(sensor_data)} (expected 66). Data not saved.")
                return  # Skip saving invalid data

            fieldnames = ['timestamp'] + list(sensor_data.keys()) + ['x_position', 'y_position', 'frame_filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row_data = sensor_data.copy()
            row_data['timestamp'] = timestamp
            row_data['x_position'] = x_position
            row_data['y_position'] = y_position
            row_data['frame_filename'] = frame_filename
            writer.writerow(row_data)

            # increment("new_csv_rows_today")
            update_csv_metrics()

        else:
            logger.warning(f"Unexpected sensor data format: {type(sensor_data)}. Data not saved.")

def trigger_daily_map_clear():
    """Puts a clear map event onto the SSE queue for all clients."""
    clear_event_payload = {
        "event": "clear_map_dots",
        "data": {"message": "Clearing dots for the new day."}
    }
    sse_queue.put(json.dumps(clear_event_payload))
    logger.info("Triggered daily map clear event for all clients.")


# -------- Main Execution ----------
if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=8133, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # Initial metrics update on startup
    update_csv_metrics()


    # CHANGE: Consolidate all scheduled tasks into one background thread
    def run_scheduled_tasks():
        # --- Schedule all your daily tasks here ---
        schedule_daily_report()  # This is your existing function call

        # Schedule the new map clearing task to run every day at midnight
        schedule.every().day.at("00:00").do(trigger_daily_map_clear)

        logger.info("Daily tasks scheduled: Email report and map clearing.")

        while True:
            schedule.run_pending()
            time.sleep(1)


    # Run the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    scheduler_thread.start()


    async def aiohttp_main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host='0.0.0.0', port=5432)
        await site.start()
        logger.info("🚀 aiohttp signaling server started on http://0.0.0.0:5432")
        while True:
            await asyncio.sleep(3600)


    # You can remove the old schedule_daily_report() call from here if it exists
    # as it's now handled in the dedicated scheduler thread.
    asyncio.run(aiohttp_main())