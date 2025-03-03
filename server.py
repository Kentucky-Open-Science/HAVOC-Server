import asyncio
import cv2
import logging
import aiohttp_cors
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from yolo_fall_detection import FallDetector

logging.basicConfig(level=logging.INFO)

app = web.Application()
pcs = set()
relay = MediaRelay()
fall_detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3)

class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0

    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1

        if self.frame_count % 3 == 0:  # Process every third frame
            img = frame.to_ndarray(format="bgr24")
            processed_img = fall_detector.process_frame(img)
            cv2.imshow("Fall Detection", processed_img)
            cv2.waitKey(1)
            logging.info(f"✅ Processed frame {self.frame_count}")
        else:
            logging.info(f"⏭ Skipped frame {self.frame_count}")

        return frame

async def offer(request):
    params = await request.json()
    logging.info("📩 Received SDP offer")

    peer = RTCPeerConnection()
    pcs.add(peer)

    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            logging.info("🎥 Received video track")
            relayed_track = relay.subscribe(track)
            video_processor = VideoProcessorTrack(relayed_track)
            peer.addTrack(video_processor)

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state is {peer.connectionState}")
        if peer.connectionState == "failed":
            await peer.close()
            pcs.discard(peer)

    try:
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer.setRemoteDescription(offer)
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
    except Exception as e:
        logging.error(f"❌ Error in WebRTC setup: {e}")
        return web.json_response({"error": str(e)}, status=500)

    logging.info("📤 Sending SDP answer")
    return web.json_response({
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })

cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

app.router.add_post("/offer", offer)

async def on_shutdown(app):
    coros = [peer.close() for peer in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, port=5000)