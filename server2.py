import asyncio
import cv2
import aiohttp_cors
import os
import platform
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCRtpSender
from aiortc.contrib.media import MediaRelay
from yolo_fall_detection import FallDetector

app = web.Application()
pcs = set()
relay = MediaRelay()
fall_detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3)

# Check if display is available
has_display = True
if platform.system() == "Linux":
    # Check if DISPLAY environment variable is set
    has_display = "DISPLAY" in os.environ

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
                
                # Only show window if display is available
                if has_display:
                    cv2.imshow("Fall Detection", processed_img)
                    cv2.waitKey(1)
                
                print(f"✅ Processed frame {self.frame_count}")
            else:
                print(f"⏭  Skipped frame {self.frame_count}")

            # Periodic restart of video processing
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_restart > 300:  # Restart every 5 minutes
                print("Restarting video processing...")
                self.frame_count = 0
                self.last_restart = current_time
                if has_display:
                    cv2.destroyAllWindows()
                fall_detector.reset()

            return frame

        except Exception as e:
            print(f"Error in video processing: {e}")
            await asyncio.sleep(1)  # Wait a bit before trying again
            return await self.recv()  # Recursive call to try again