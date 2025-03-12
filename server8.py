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
import traceback
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCRtpSender
from aiortc.contrib.media import MediaRelay, MediaBlackhole
from yolo_fall_detection import FallDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fall_detection")
codec_logger = logging.getLogger("codec_enforcer")
codec_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("codec_errors.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
codec_logger.addHandler(file_handler)

app = web.Application()
pcs = set()
relay = MediaRelay()
audio_sink = MediaBlackhole()
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
        self.max_latency = 0.3
        self.frame_queue = asyncio.Queue(maxsize=3)  # Reduced queue size
        self.processing_task = asyncio.create_task(self.process_frames())
        self.running = True
        self.codec_info_logged = False

    async def process_frames(self):
        while self.running:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                current_time = time.time()
                
                if self.frame_count % self.skip_factor == 0:
                    img = frame.to_ndarray(format="bgr24")
                    
                    if os.environ.get('DISPLAY'):
                        try:
                            cv2.imshow("Fall Detection", img)
                            cv2.waitKey(1)
                        except Exception as e:
                            logger.error(f"cv2.imshow failed: {e}")
                    else:
                        logger.info("No DISPLAY environment variable, skipping cv2.imshow().")
                    
                    logger.info(f"✅ Processed frame {self.frame_count} (latency: {time.time() - current_time:.2f}s)")
                self.frame_count += 1
                
            except asyncio.CancelledError:
                logger.info("Frame processing task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(0.1)

    async def recv(self):
        try:
            frame = await self.track.recv()
            frame_delay = time.time() - self.last_frame_time
            self.buffer_size = frame_delay

            if frame_delay > self.max_latency:
                self.skip_factor = min(10, self.skip_factor + 1)
                logger.warning(f"High latency ({frame_delay:.2f}s), increasing skip factor to {self.skip_factor}")
            elif frame_delay < self.max_latency / 2 and self.skip_factor > 2:
                self.skip_factor = max(2, self.skip_factor - 1)
                logger.info(f"Latency normal ({frame_delay:.2f}s), decreasing skip factor to {self.skip_factor}")

            self.consecutive_errors = 0
            self.frame_count += 1
            
            try:
                self.frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("Frame queue full, dropping frame")
            
            return frame
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error in video processing: {e}")
            if self.consecutive_errors >= self.max_errors:
                logger.critical(f"Too many consecutive errors ({self.consecutive_errors}), resetting connection")
                raise RuntimeError("Video processing failed repeatedly")
            await asyncio.sleep(0.05)
            return None

async def create_peer_connection():
    peer = RTCPeerConnection()
    
    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {peer.connectionState}")
        if peer.connectionState == "failed":
            await peer.close()
            pcs.discard(peer)
    
    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            logger.info("🎥 Received video track")
            relayed_track = relay.subscribe(track)
            video_processor = VideoProcessorTrack(relayed_track)
            peer.addTrack(video_processor)
        elif track.kind == "audio":
            logger.info("🔊 Received audio track")
            relayed_audio = relay.subscribe(track)
            peer.addTrack(relayed_audio)
            audio_sink.addTrack(relayed_audio)

    return peer

async def offer(request):
    params = await request.json()
    logger.info("📩 Received SDP offer")
    
    peer = await create_peer_connection()
    pcs.add(peer)
    try:
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer.setRemoteDescription(offer)
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
    except Exception as e:
        logger.error(f"❌ WebRTC setup failed: {e}")
        await peer.close()
        pcs.discard(peer)
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type
    })

async def cleanup_resources():
    logger.info("Cleaning up resources...")
    for peer in pcs:
        await peer.close()
    pcs.clear()
    await audio_sink.stop()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    import gc
    gc.collect()
    logger.info("Cleanup complete")

async def on_shutdown(app):
    logger.info("Shutting down server...")
    await cleanup_resources()

def handle_exit_signals():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))
            logger.info(f"Registered signal handler for {sig}")
        except NotImplementedError:
            break

async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}, shutting down...")
    await cleanup_resources()
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
    loop.stop()

app.on_shutdown.append(on_shutdown)
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    logger.info("🚀 Starting WebRTC server on http://0.0.0.0:5000")
    handle_exit_signals()
    web.run_app(app, host='0.0.0.0', port=5000)
