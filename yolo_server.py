import cv2
import asyncio
import json
import aiohttp_cors
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from ultralytics import YOLO
import math

# Initialize aiohttp web application
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

# Load YOLO model
model = YOLO("yolo_weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


class VideoProcessorTrack(MediaStreamTrack):
    """Receives a WebRTC video stream and processes it in real-time using YOLOv8."""
    
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0  
        self.last_frame = None  # Store the latest frame
        self.processing = False  # Track if YOLO is running

    async def recv(self):
        """Receives and processes a video frame from WebRTC."""
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Store the latest frame but don't process it immediately
        self.last_frame = img.copy()

        # Avoid frame processing backlog
        if not self.processing:
            self.processing = True
            asyncio.create_task(self.process_frame())

        return frame

    async def process_frame(self):
        """Processes the latest frame asynchronously to prevent lag."""
        if self.last_frame is None:
            self.processing = False
            return

        img = self.last_frame
        self.frame_count += 1
        print(f"✅ Processing frame {self.frame_count} - Shape: {img.shape}")

        # Run YOLO object detection (async)
        results = await asyncio.to_thread(model, img)

        # Process YOLO results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

                # Confidence score
                confidence = round(float(box.conf[0]) * 100, 2)

                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                print(f"🔍 Detected {class_name} with {confidence}% confidence")

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Label the object
                cv2.putText(img, f"{class_name} {confidence}%", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the processed frame
        cv2.imshow("YOLOv8 WebRTC Stream", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        self.processing = False  # Allow next frame processing


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
            print("🎥 Received video track, processing frames with YOLO...")
            video_processor = VideoProcessorTrack(track)
            peer.addTrack(video_processor)

    # Set remote SDP before creating an answer
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

# Start the aiohttp web server
web.run_app(app, port=5000)
