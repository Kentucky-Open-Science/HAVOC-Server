import asyncio
import cv2
import aiohttp_cors
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCRtpSender
from aiortc.contrib.media import MediaRelay
from yolo_fall_detection import FallDetector

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
        self.last_restart = asyncio.get_event_loop().time()

    async def recv(self):
        try:
            frame = await self.track.recv()
            self.frame_count += 1

            if self.frame_count % 3 == 0:  # Process every third frame
                img = frame.to_ndarray(format="bgr24")
                processed_img = fall_detector.process_frame(img)
                cv2.imshow("Fall Detection", processed_img)
                cv2.waitKey(1)
                print(f"✅ Processed frame {self.frame_count}")
            else:
                print(f"⏭ Skipped frame {self.frame_count}")

            # Periodic restart of video processing
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_restart > 300:  # Restart every 5 minutes
                print("Restarting video processing...")
                self.frame_count = 0
                self.last_restart = current_time
                cv2.destroyAllWindows()
                fall_detector.reset()

            return frame

        except Exception as e:
            print(f"Error in video processing: {e}")
            await asyncio.sleep(1)  # Wait a bit before trying again
            return await self.recv()  # Recursive call to try again

async def create_peer_connection():
    peer = RTCPeerConnection()
    
    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            print("🎥 Received video track")
            relayed_track = relay.subscribe(track)
            video_processor = VideoProcessorTrack(relayed_track)
            peer.addTrack(video_processor)

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {peer.connectionState}")
        if peer.connectionState == "failed":
            await peer.close()
            pcs.discard(peer)
            # Attempt to reconnect
            await asyncio.sleep(5)
            new_peer = await create_peer_connection()
            pcs.add(new_peer)

    return peer

async def offer(request):
    params = await request.json()
    print("📩 Received SDP offer")

    peer = await create_peer_connection()
    pcs.add(peer)

    try:
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer.setRemoteDescription(offer)

        # Disable RTX to avoid decoder issues
        for transceiver in peer.getTransceivers():
            if transceiver.kind == "video":
                codecs = RTCRtpSender.getCapabilities("video").codecs
                codecs = [codec for codec in codecs if codec.mimeType.lower() != "video/rtx"]
                transceiver.setCodecPreferences(codecs)

        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
    except Exception as e:
        print(f"❌ Error in WebRTC setup: {e}")
        return web.json_response({"error": str(e)}, status=500)

    print("📤 Sending SDP answer")
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
    print("Starting server on http://0.0.0.0:5000")
    web.run_app(app, host='0.0.0.0', port=5000)