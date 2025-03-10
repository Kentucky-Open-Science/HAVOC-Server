import asyncio
import cv2
import aiohttp_cors
import os
import platform
import logging
import json
import time
import sys
import signal
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
        self.running = True

    async def process_frames(self):
        while self.running:
            try:
                # Use a timeout to prevent blocking forever
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                current_time = time.time()
                
                if self.frame_count % self.skip_factor == 0:
                    img = frame.to_ndarray(format="bgr24")
                    # Resize frame for faster processing
                    img = cv2.resize(img, (640, 360))
                    
                    # Use a timeout for the frame processing to prevent blocking
                    processed_img = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, fall_detector.process_frame, img
                        ),
                        timeout=2.0  # Timeout after 2 seconds
                    )
                    
                    # Always try to display the frame, but handle exceptions
                    try:
                        cv2.imshow("Fall Detection", processed_img)
                        cv2.waitKey(1)
                    except Exception as e:
                        logger.warning(f"Display error (non-critical): {e}")
                    
                    logger.info(f"✅ Processed frame {self.frame_count} (latency: {time.time() - current_time:.2f}s, skip: {self.skip_factor})")
                else:
                    logger.debug(f"⏭ Skipped frame {self.frame_count}")

                self.frame_queue.task_done()
            except asyncio.TimeoutError:
                # This is expected, just continue
                continue
            except asyncio.CancelledError:
                # Task was cancelled, exit cleanly
                logger.info("Frame processing task cancelled")
                break
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

            # Add frame to queue for processing - use try with no wait to avoid blocking
            try:
                self.frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("Frame queue full, dropping frame")

            # Periodic reset
            current_time = time.time()
            if current_time - self.last_restart > 60:
                logger.info("⏰ Periodic reset of video processing...")
                self.frame_count = 0
                self.last_restart = current_time
                
                # Close windows safely
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    logger.warning(f"Error closing windows: {e}")
                    
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
        self.running = False
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass
        
        # Safely close any OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Error closing windows during stop: {e}")

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
        
        # Filter out RTX from the offer SDP to prevent the error
        lines = offer_sdp.split("\r\n")
        rtx_pt = None
        rtx_apt = None
        new_offer_lines = []
        
        # First pass - identify RTX payload type and its associated payload type
        for line in lines:
            if "a=rtpmap:" in line and "rtx/90000" in line:
                rtx_pt = line.split("a=rtpmap:")[1].split(" ")[0]
            if rtx_pt and "a=fmtp:" + rtx_pt in line:
                rtx_apt = line.split("apt=")[1]
        
        # Second pass - filter out RTX-related lines
        for line in lines:
            if rtx_pt and ("a=rtpmap:" + rtx_pt in line or 
                          "a=fmtp:" + rtx_pt in line or
                          "a=rtcp-fb:" + rtx_pt in line):
                logger.info(f"Filtering out RTX line: {line}")
                continue
            
            if line.startswith("m=video"):
                parts = line.split(" ")
                filtered_parts = [p for p in parts if p != rtx_pt]
                new_line = " ".join(filtered_parts)
                new_offer_lines.append(new_line)
                continue
                
            new_offer_lines.append(line)
        
        # Now create optimized SDP with low latency parameters
        optimized_lines = []
        for i, line in enumerate(new_offer_lines):
            optimized_lines.append(line)
            if line.startswith("m=video"):
                # Set low latency parameters
                optimized_lines.extend([
                    "b=AS:2000",  # Limit bandwidth
                    "a=rtcp-fb:* nack pli",  # Enable NACK and PLI for faster recovery
                    "a=rtcp-fb:* ccm fir"    # Enable FIR for full intra-frame requests
                ])
                
        modified_offer_sdp = "\r\n".join(optimized_lines)
        
        offer = RTCSessionDescription(sdp=modified_offer_sdp, type=params["type"])
        await peer.setRemoteDescription(offer)

        # Configure codecs for low latency - explicitly filter out RTX
        for transceiver in peer.getTransceivers():
            if transceiver.kind == "video":
                codecs = RTCRtpSender.getCapabilities("video").codecs
                # Filter out RTX codecs completely
                filtered_codecs = [
                    codec for codec in codecs 
                    if "rtx" not in codec.mimeType.lower()
                ]
                
                # Prefer VP8 over other codecs
                preferred_codecs = [
                    codec for codec in filtered_codecs 
                    if codec.mimeType.lower() == "video/vp8"
                ]
                other_codecs = [
                    codec for codec in filtered_codecs 
                    if codec.mimeType.lower() != "video/vp8"
                ]
                
                logger.info(f"Setting codec preferences: {[c.mimeType for c in preferred_codecs + other_codecs]}")
                transceiver.setCodecPreferences(preferred_codecs + other_codecs)

        answer = await peer.createAnswer()
        
        # Optimize answer SDP
        answer_sdp = answer.sdp
        lines = answer_sdp.split("\r\n")
        new_lines = []
        for line in lines:
            # Skip any RTX-related lines in the answer as well
            if "a=rtpmap:" in line and "rtx/90000" in line:
                continue
            if "a=fmtp:" in line and "apt=" in line:
                continue
                
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
    
    # First, stop all video processor tracks
    for peer in pcs:
        for sender in peer.getSenders():
            if hasattr(sender.track, "stop") and callable(sender.track.stop):
                try:
                    await sender.track.stop()
                except Exception as e:
                    logger.warning(f"Error stopping track: {e}")
    
    # Close peer connections
    coros = [peer.close() for peer in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
    # Close OpenCV windows
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning(f"Error closing windows: {e}")
        
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("Cleanup complete")

async def on_shutdown(app):
    logger.info("Shutting down server, cleaning resources...")
    await cleanup_resources()

def handle_exit_signals():
    """Set up proper handling of exit signals for both Linux and Windows"""
    loop = asyncio.get_event_loop()
    
    # Register signal handlers for graceful shutdown
    signals = (signal.SIGINT, signal.SIGTERM)
    for sig in signals:
        try:
            loop.add_signal_handler(
                sig,
                lambda sig=sig: asyncio.create_task(shutdown(sig, loop))
            )
            logger.info(f"Registered signal handler for {sig}")
        except NotImplementedError:
            # Windows doesn't support SIGTERM properly, so we'll handle it differently
            if platform.system() == "Windows":
                logger.info("Running on Windows, using different signal handling approach")
                break

async def shutdown(sig, loop):
    """Gracefully shut down the application"""
    logger.info(f"Received exit signal {sig.name}, shutting down...")
    await cleanup_resources()
    
    # Stop the web server
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    
    loop.stop()

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
    
    # Set up signal handlers for clean shutdown
    handle_exit_signals()
    
    try:
        web.run_app(app, host='0.0.0.0', port=5000)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        loop = asyncio.get_event_loop()
        asyncio.run(cleanup_resources())