import asyncio
import cv2
import logging
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from flask import Flask, Response
import threading
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
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
        # logger.info(f"✅ Frame received at recv(): pts={frame.pts}, size={frame.width}x{frame.height}, time={time.time()}")
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
                        # logger.info(f"✅ Frame processed: pts={frame.pts}, size={frame.width}x{frame.height}")
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
    return '<html><body><img src="/video_feed"></body></html>'

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


last_pts = None
freeze_detected_time = None
freeze_threshold = 10  # Seconds before switching to a placeholder
duplicate_threshold = 20  # Number of duplicate frames before switching to a placeholder
duplicate_frame_count = 0

# read in the offline image
offline_img = cv2.imread("temiFace for TV.png")
ret, placeholder_buffer = cv2.imencode('.png', offline_img)
offline_bytes = placeholder_buffer.tobytes()


def gen_frames():
    global last_pts, freeze_detected_time, duplicate_frame_count

    while True:
        time.sleep(0.02)
        frame = frame_holder.get('frame', None)

        if frame is not None:
            if last_pts is None or frame.pts != last_pts:
                # New frame detected, reset freeze timer and duplicate count
                last_pts = frame.pts
                freeze_detected_time = None
                duplicate_frame_count = 0
                # logger.info("✅ Streaming live frame")

                img = frame.to_ndarray(format="bgr24")
                ret, buffer = cv2.imencode('.jpg', img)
                frame_bytes = buffer.tobytes()
            else:
                if freeze_detected_time is None:
                    freeze_detected_time = time.time()
                elif duplicate_frame_count > duplicate_threshold or time.time() - freeze_detected_time > freeze_threshold:
                    # logger.warning("🚨 Stream frozen, displaying placeholder image")
                    frame_bytes = offline_bytes
                else:
                    duplicate_frame_count += 1
                    # logger.info(f"⚠️ Duplicate frame detected: {duplicate_frame_count}")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            logger.debug("Waiting for first frame...")
            time.sleep(0.01)


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
