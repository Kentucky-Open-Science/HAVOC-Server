import asyncio
import cv2
import logging
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from flask import Flask, Response, render_template, jsonify
import threading
import time
from datetime import datetime
import numpy as np
import os

# Set logging to INFO level to reduce verbosity (DEBUG can be re-enabled for troubleshooting)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("temi-stream")

relay = MediaRelay()
pcs = set()
frame_holder = {'frame': None}

# Global variables
last_pts = None
freeze_detected_time = None
duplicate_frame_count = 0
duplicate_threshold = 5
freeze_threshold = 5.0

# Load the filler image at startup
filler_image_path = os.path.join('static', 'temiFace_screen_saver.png')
if not os.path.exists(filler_image_path):
    logger.error(f"Filler image not found at {filler_image_path}")
    offline_bytes = b"..."  # Fallback if image is missing
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

last_frame_time = "N/A"

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
        # Log only the first frame to confirm stream start
        if last_pts is None:
            logger.info(f"WebRTC stream started, first frame received, pts={frame.pts}")
        return frame

async def create_peer_connection():
    peer = RTCPeerConnection()

    @peer.on("track")
    async def on_track(track):
        if track.kind == "video":
            processor = VideoProcessorTrack(relay.subscribe(track))

            async def consume_track():
                try:
                    while True:
                        frame = await processor.recv()
                        frame_holder['frame'] = frame
                except Exception as e:
                    logger.error(f"Error consuming track: {e}")

            asyncio.create_task(consume_track())

    pcs.add(peer)
    return peer

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    peer = await create_peer_connection()
    await peer.setRemoteDescription(offer)
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    return web.json_response({"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})

app.router.add_post("/offer", offer)

# -------- Flask video feed server ----------
flask_app = Flask(__name__)

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
    
    # Removed repetitive debug logs
    return render_template('index.html', stream_status=stream_status, last_frame_time=last_frame_time)

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
    
    return jsonify({'stream_status': stream_status, 'last_frame_time': last_frame_time})

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global last_pts, freeze_detected_time, duplicate_frame_count, last_frame_time

    while True:
        time.sleep(0.02)
        frame = frame_holder.get('frame', offline_bytes)
        
        if isinstance(frame, bytes):
            frame_bytes = frame
            # Log only on state change to Offline
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
                ret, buffer = cv2.imencode('.jpg', img)
                frame_bytes = buffer.tobytes()
            else:
                if freeze_detected_time is None:
                    freeze_detected_time = time.time()
                elif duplicate_frame_count > duplicate_threshold or time.time() - freeze_detected_time > freeze_threshold:
                    frame_bytes = offline_bytes
                    logger.info(f"Stream frozen, switching to placeholder after {duplicate_frame_count} duplicates, "
                                f"time elapsed={time.time() - freeze_detected_time:.2f}s")
                else:
                    duplicate_frame_count += 1
                    # Log only on significant duplicate threshold
                    if duplicate_frame_count == duplicate_threshold:
                        logger.warning(f"Duplicate frames detected, count={duplicate_frame_count}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------- Main Execution ----------
if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=8133, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    async def aiohttp_main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host='0.0.0.0', port=5432)
        await site.start()
        logger.info("🚀 aiohttp signaling server started on http://0.0.0.0:5432")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(aiohttp_main())