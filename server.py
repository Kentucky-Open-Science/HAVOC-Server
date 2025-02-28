import cv2
import asyncio
import json
import aiohttp_cors
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

app = web.Application()
pcs = set()

# Enable CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

class VideoProcessorTrack(MediaStreamTrack):
    """Receives a WebRTC video stream and processes it in real-time."""
    
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0  # Count received frames

    async def recv(self):
        """Receives and processes a video frame from WebRTC."""
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        print(f"✅ Frame {self.frame_count} received - Shape: {img.shape}")

        # Draw a red rectangle for visibility (debugging)
        cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 5)
        cv2.putText(img, "Processing Video", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame using OpenCV
        cv2.imshow("WebRTC Stream", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return frame

async def offer(request):
    """Handles incoming WebRTC offer and sets up peer connection."""
    params = await request.json()
    print("\n📩 Received SDP offer:")
    print(params["sdp"])

    peer = RTCPeerConnection()
    pcs.add(peer)

    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            print("🎥 Received video track, processing frames...")
            video_processor = VideoProcessorTrack(track)
            peer.addTrack(video_processor)

    # Set remote SDP **before** creating an answer
    try:
        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer.setRemoteDescription(offer_sdp)  # ✅ Must await here
        print("✅ Successfully set remote description")
    except Exception as e:
        print(f"❌ Error setting remote description: {e}")
        return web.json_response({"error": str(e)}, status=500)  # ✅ Proper JSON response

    # Create and send SDP answer
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    print("\n📤 Sending SDP answer:")
    print(peer.localDescription.sdp)

    return web.json_response({  # ✅ Return a properly formatted JSON response
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })

# Setup CORS for the /offer route
app.router.add_post("/offer", offer)

web.run_app(app, port=5000)
