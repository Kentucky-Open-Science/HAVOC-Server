import cv2
import asyncio
import json
import aiohttp_cors
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder
from yolo_fall_detection import FallDetector  # Import the FallDetector class

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

# Initialize YOLO Fall Detector
fall_detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3)


class VideoProcessorTrack(MediaStreamTrack):
    """Receives a WebRTC video stream and processes it using YOLO for fall detection."""

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

        # Run YOLO fall detection
        processed_img = fall_detector.process_frame(img)  # Uses FallDetector to process the frame

        # Display the processed frame with bounding boxes and fall labels
        cv2.imshow("WebRTC Stream - Fall Detection", processed_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

        return frame


class AudioProcessorTrack(MediaStreamTrack):
    """Receives a WebRTC audio stream and processes it."""

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0  # Count received audio frames

    async def recv(self):
        """Receives and processes an audio frame from WebRTC."""
        frame = await self.track.recv()
        self.frame_count += 1
        print(f"🎤 Audio Frame {self.frame_count} received")

        return frame


async def offer(request):
    """Handles incoming WebRTC offer and sets up peer connection."""
    params = await request.json()
    print("\n📩 Received SDP offer:")
    print(params["sdp"])

    peer = RTCPeerConnection()
    pcs.add(peer)

    # Setup media recorder (optional: save to file)
    media_recorder = MediaRecorder("output.mp4")  # Record incoming streams

    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            print("🎥 Received video track, processing frames with YOLO...")
            video_processor = VideoProcessorTrack(track)
            peer.addTrack(video_processor)
            media_recorder.addTrack(track)  # Save video
        elif track.kind == "audio":
            print("🎤 Received audio track, processing frames...")
            audio_processor = AudioProcessorTrack(track)
            peer.addTrack(audio_processor)
            media_recorder.addTrack(track)  # Save audio

    await media_recorder.start()  # Start recording

    # Set remote SDP **before** creating an answer
    try:
        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer.setRemoteDescription(offer_sdp)
        print("✅ Successfully set remote description")
    except Exception as e:
        print(f"❌ Error setting remote description: {e}")
        return web.json_response({"error": str(e)}, status=500)

    # Create and send SDP answer
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    print("\n📤 Sending SDP answer:")
    print(peer.localDescription.sdp)

    return web.json_response({
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })


# Setup CORS for the /offer route
app.router.add_post("/offer", offer)

web.run_app(app, port=5000)
