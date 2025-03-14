import asyncio
import cv2
import logging
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from flask import Flask, Response, render_template
import threading
import time
import numpy as np

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for detailed logs
logger = logging.getLogger("temi-stream")

relay = MediaRelay()
pcs = set()
frame_holder = {'frame': None}

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
        logger.debug(f"WebRTC recv: frame received, pts={frame.pts}, size={frame.width}x{frame.height}")
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

async def start_aiohttp():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host='0.0.0.0', port=5000)
    await site.start()
    logger.info("🚀 aiohttp signaling server started on http://0.0.0.0:5000")

# -------- Flask video feed server ----------
flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    frame = frame_holder.get('frame', offline_bytes)
    logger.debug(f"index(): frame={'placeholder' if frame == offline_bytes else 'live'}, last_pts={last_pts}, "
                 f"duplicate_frame_count={duplicate_frame_count}, freeze_detected_time={freeze_detected_time}")
    if frame == offline_bytes:
        status = "Disconnected" if last_pts is None else "Frozen"
    else:
        status = "Live"
    
    last_frame_time = time.ctime(time.time()) if last_pts else "N/A"
    logger.debug(f"index(): status={status}, last_frame_time={last_frame_time}")
    return render_template('index.html', stream_status=status, last_frame_time=last_frame_time)

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

last_pts = None
freeze_detected_time = None
freeze_threshold = 10  # Seconds before switching to a placeholder
duplicate_threshold = 20  # Number of duplicate frames before switching to a placeholder
duplicate_frame_count = 0

# read in the offline image
offline_img = cv2.imread("static/temiFace_screen_saver.png")
ret, placeholder_buffer = cv2.imencode('.png', offline_img)
offline_bytes = placeholder_buffer.tobytes()

frame_holder['frame'] = offline_bytes  # Store bytes instead of None
def gen_frames():
    global last_pts, freeze_detected_time, duplicate_frame_count

    while True:
        time.sleep(0.02)
        frame = frame_holder.get('frame', offline_bytes)
        logger.debug(f"gen_frames(): frame={'placeholder' if frame == offline_bytes else 'live'}, last_pts={last_pts}")
        
        if isinstance(frame, bytes):
            frame_bytes = frame
            logger.debug("gen_frames(): Yielding placeholder image")
        else:
            logger.debug(f"gen_frames(): Processing live frame, pts={frame.pts}")
            if last_pts is None or frame.pts != last_pts:
                last_pts = frame.pts
                freeze_detected_time = None
                duplicate_frame_count = 0
                logger.debug(f"gen_frames(): New frame, last_pts updated to {last_pts}")
                img = frame.to_ndarray(format="bgr24")
                ret, buffer = cv2.imencode('.jpg', img)
                frame_bytes = buffer.tobytes()
            else:
                if freeze_detected_time is None:
                    freeze_detected_time = time.time()
                    logger.debug(f"gen_frames(): Duplicate frame detected, freeze_detected_time set to {freeze_detected_time}")
                elif duplicate_frame_count > duplicate_threshold or time.time() - freeze_detected_time > freeze_threshold:
                    frame_bytes = offline_bytes
                    logger.debug(f"gen_frames(): Switching to placeholder, duplicates={duplicate_frame_count}, "
                                 f"time elapsed={time.time() - freeze_detected_time}")
                else:
                    duplicate_frame_count += 1
                    logger.debug(f"gen_frames(): Duplicate frame, count={duplicate_frame_count}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def start_flask():
    flask_app.logger.setLevel(logging.INFO)
    flask_app.logger.info("Starting Flask server on http://0.0.0.0:8133")
    flask_app.run(host='0.0.0.0', port=8133, debug=False, use_reloader=False, threaded=True)

# -------- Main Execution ----------
if __name__ == "__main__":
    import threading

    # Flask server
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=8133, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # aiohttp signaling server
    async def aiohttp_main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host='0.0.0.0', port=5432)
        await site.start()
        logger.info("🚀 aiohttp signaling server started on http://0.0.0.0:5432")

        # Run indefinitely
        while True:
            await asyncio.sleep(3600)

    asyncio.run(aiohttp_main())