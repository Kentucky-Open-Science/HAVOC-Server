import asyncio
import cv2
import logging
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from flask import Flask, Response
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fall_detection")

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
        logger.info(f"✅ Frame received at recv(): pts={frame.pts}, size={frame.width}x{frame.height}, time={time.time()}")
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
                        logger.info(f"✅ Frame processed: pts={frame.pts}, size={frame.width}x{frame.height}")
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
    return '<html><body>hello<img src="/video_feed"></body></html>'

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        frame = frame_holder.get('frame', None)
        if frame is not None:
            logger.info("✅ Got frame from frame_holder for streaming")
            img = frame.to_ndarray(format="bgr24")
            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            logger.debug("Waiting for first frame...")
            time.sleep(0.01)


# webcam test
# def gen_frames():
#     cap = cv2.VideoCapture(0)  # local webcam test
#     while True:
#         success, frame = cap.read()
#         if not success:
#             continue
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



def start_flask():
    flask_app.logger.setLevel(logging.INFO)
    flask_app.logger.info("Starting Flask server on http://0.0.0.0:8080")
    flask_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False, threaded=True)

# -------- Main Execution ----------
if __name__ == "__main__":
    import threading

    # Flask server
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    # aiohttp signaling server
    async def aiohttp_main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host='0.0.0.0', port=5000)
        await site.start()
        logger.info("🚀 aiohttp signaling server started on http://0.0.0.0:5000")

        # Run indefinitely
        while True:
            await asyncio.sleep(3600)

    asyncio.run(aiohttp_main())

