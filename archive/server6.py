import asyncio
import cv2
import aiohttp_cors
import os
import platform
import logging
import json
import time
import sys
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCRtpSender
from aiortc.contrib.media import MediaRelay
from yolo_fall_detection import FallDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fall_detection")

app = web.Application()
pcs = set()
relay = MediaRelay()
fall_detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3)

# Check if display is available
has_display = True
if platform.system() == "Linux":
    has_display = "DISPLAY" in os.environ

class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.last_restart = time.time()
        self.consecutive_errors = 0
        self.max_errors = 5
        self.last_frame_time = time.time()
        self.buffer_size = 0
        self.skip_factor = 2
        self.max_latency = 0.3  # Reduced max latency threshold
        self.frame_queue = asyncio.Queue(maxsize=5)  # Limit queue size
        self.processing_task = asyncio.create_task(self.process_frames())

    async def process_frames(self):
        while True:
            try:
                frame = await self.frame_queue.get()
                current_time = time.time()
                
                if self.frame_count % self.skip_factor == 0:
                    img = frame.to_ndarray(format="bgr24")
                    # Resize frame for faster processing
                    img = cv2.resize(img, (640, 360))
                    processed_img = await asyncio.get_event_loop().run_in_executor(
                        None, fall_detector.process_frame, img
                    )
                    
                    if has_display:
                        cv2.imshow("Fall Detection", processed_img)
                        cv2.waitKey(1)
                    
                    logger.info(f"✅ Processed frame {self.frame_count} (latency: {time.time() - current_time:.2f}s, skip: {self.skip_factor})")
                else:
                    logger.debug(f"⏭ Skipped frame {self.frame_count}")

                self.frame_queue.task_done()
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                await asyncio.sleep(0.1)

    async def recv(self):
        try:
            current_time = time.time()
            frame = await self.track.recv()
            
            frame_delay = time.time() - current_time
            self.buffer_size = frame_delay
            
            # Adaptive frame skipping
            if frame_delay > self.max_latency:
                self.skip_factor = min(10, self.skip_factor + 1)
                if self.skip_factor > 2:
                    logger.warning(f"High latency ({frame_delay:.2f}s), increasing skip factor to {self.skip_factor}")
            elif frame_delay < self.max_latency / 2 and self.skip_factor > 2:
                self.skip_factor = max(2, self.skip_factor - 1)
                logger.info(f"Latency normal ({frame_delay:.2f}s), decreasing skip factor to {self.skip_factor}")
            
            self.consecutive_errors = 0
            self.frame_count += 1

            # Add frame to queue for processing
            if not self.frame_queue.full():
                await self.frame_queue.put(frame)
            else:
                logger.warning("Frame queue full, dropping frame")

            # Periodic reset
            current_time = time.time()
            if current_time - self.last_restart > 60:
                logger.info("⏰ Periodic reset of video processing...")
                self.frame_count = 0
                self.last_restart = current_time
                if has_display:
                    cv2.destroyAllWindows()
                fall_detector.reset()

            return frame

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error in video processing: {e}")
            
            if self.consecutive_errors >= self.max_errors:
                logger.critical(f"Too many consecutive errors ({self.consecutive_errors}), connection needs reset")
                raise RuntimeError("Video processing failed repeatedly, connection needs reset")
                
            await asyncio.sleep(0.05)  # Reduced sleep time
            return await self.recv()

    async def stop(self):
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass

async def create_peer_connection():
    peer = RTCPeerConnection()
    
    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            logger.info("🎥 Received video track")
            relayed_track = relay.subscribe(track)
            video_processor = VideoProcessorTrack(relayed_track)
            peer.addTrack(video_processor)

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {peer.connectionState}")
        if peer.connectionState == "failed":
            await peer.close()
            pcs.discard(peer)
            await asyncio.sleep(1)
            logger.info("Attempting to recreate peer connection")
            new_peer = await create_peer_connection()
            pcs.add(new_peer)

    return peer

async def offer(request):
    params = await request.json()
    logger.info("📩 Received SDP offer")

    peer = await create_peer_connection()
    pcs.add(peer)

    try:
        offer_sdp = params["sdp"]
        # Optimize SDP for low latency
        lines = offer_sdp.split("\r\n")
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            if line.startswith("m=video"):
                # Set low latency parameters
                new_lines.extend([
                    "b=AS:2000",  # Limit bandwidth
                    "a=rtcp-fb:* nack pli",  # Enable NACK and PLI for faster recovery
                    "a=rtcp-fb:* ccm fir",   # Enable FIR for full intra-frame requests
                    "a=fmtp:96 max-fs=12288;max-fr=30;x-google-start-bitrate=1500"  # Add VP8 parameters here
                ])
        offer_sdp = "\r\n".join(new_lines)
        
        offer = RTCSessionDescription(sdp=offer_sdp, type=params["type"])
        await peer.setRemoteDescription(offer)

        # Configure codecs for low latency
        for transceiver in peer.getTransceivers():
            if transceiver.kind == "video":
                codecs = RTCRtpSender.getCapabilities("video").codecs
                preferred_codecs = [
                    codec for codec in codecs 
                    if codec.mimeType.lower() == "video/vp8"
                ]
                other_codecs = [
                    codec for codec in codecs 
                    if codec.mimeType.lower() != "video/rtx" and codec.mimeType.lower() != "video/vp8"
                ]
                transceiver.setCodecPreferences(preferred_codecs + other_codecs)

        answer = await peer.createAnswer()
        
        # Optimize answer SDP
        answer_sdp = answer.sdp
        lines = answer_sdp.split("\r\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if "a=rtpmap:96 VP8/90000" in line:
                new_lines.append("a=fmtp:96 max-fs=12288;max-fr=30;x-google-start-bitrate=1500")
        answer_sdp = "\r\n".join(new_lines)
        
        modified_answer = RTCSessionDescription(sdp=answer_sdp, type=answer.type)
        await peer.setLocalDescription(modified_answer)
        
    except Exception as e:
        logger.error(f"❌ Error in WebRTC setup: {e}")
        await peer.close()
        pcs.discard(peer)
        return web.json_response({"error": str(e)}, status=500)

    logger.info("📤 Sending SDP answer")
    return web.json_response({
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })

async def cleanup_resources():
    logger.info("Cleaning up resources...")
    coros = [peer.close() for peer in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    if has_display:
        cv2.destroyAllWindows()
    import gc
    gc.collect()
    logger.info("Cleanup complete")

async def on_shutdown(app):
    logger.info("Shutting down server, cleaning resources...")
    await cleanup_resources()

app.on_shutdown.append(on_shutdown)

# Configure CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

# Register routes with CORS
route = cors.add(app.router.add_resource("/offer"))
cors.add(route.add_route("POST", offer))

if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:5000")
    
    try:
        web.run_app(app, host='0.0.0.0', port=5000)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        loop = asyncio.new_event_loop()
        asyncio