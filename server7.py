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
from aiortc.contrib.media import MediaRelay, MediaBlackhole
from yolo_fall_detection import FallDetector

# Setup enhanced logging with codec information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fall_detection")
codec_logger = logging.getLogger("codec_enforcer")
codec_logger.setLevel(logging.DEBUG)

# Add file handler for codec-specific logging
file_handler = logging.FileHandler("codec_errors.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
codec_logger.addHandler(file_handler)

app = web.Application()
pcs = set()
relay = MediaRelay()
# Create audio sink that doesn't require playback
audio_sink = MediaBlackhole()
fall_detector = FallDetector(model_path="yolo_weights/yolo11n-pose.pt", conf_threshold=0.3)

class CodecEnforcer:
    """Helper class to monitor and enforce VP8 video and Opus audio codec usage"""
    
    @staticmethod
    def log_codec_info(sdp, direction="unknown"):
        """Parse SDP to identify and log codec information"""
        lines = sdp.split("\r\n")
        codec_map = {}
        video_payload_types = []
        audio_payload_types = []
        current_section = None
        
        # First extract the media section payload types
        for line in lines:
            if line.startswith("m=video "):
                current_section = "video"
                parts = line.split(" ")
                if len(parts) > 3:
                    # Skip the m=video port proto and get payload types
                    video_payload_types = parts[3:]
                    codec_logger.debug(f"Video payload types in {direction} SDP: {video_payload_types}")
            elif line.startswith("m=audio "):
                current_section = "audio"
                parts = line.split(" ")
                if len(parts) > 3:
                    # Skip the m=audio port proto and get payload types
                    audio_payload_types = parts[3:]
                    codec_logger.debug(f"Audio payload types in {direction} SDP: {audio_payload_types}")
        
        # Then map payload types to codec names
        for line in lines:
            if line.startswith("a=rtpmap:"):
                parts = line.split(" ")
                if len(parts) >= 2:
                    pt_parts = parts[0].split(":")
                    if len(pt_parts) >= 2:
                        pt = pt_parts[1]
                        codec_info = parts[1]
                        codec = codec_info.split("/")[0]
                        codec_map[pt] = codec_info
                        
                        # Log non-VP8 video codecs
                        if pt in video_payload_types and codec.upper() != "VP8":
                            codec_logger.warning(f"⚠️ Non-VP8 video codec detected in {direction} SDP: PT={pt}, Codec={codec}")
                        
                        # Log non-OPUS audio codecs
                        if pt in audio_payload_types and codec.upper() != "OPUS":
                            codec_logger.warning(f"⚠️ Non-OPUS audio codec detected in {direction} SDP: PT={pt}, Codec={codec}")
        
        # Log the full codec mapping
        codec_logger.info(f"Codec mapping in {direction} SDP: {codec_map}")
        return codec_map, video_payload_types, audio_payload_types
    
    @staticmethod
    def enforce_codecs(sdp, direction="unknown"):
        """Modify SDP to ensure only VP8 is used for video and OPUS for audio"""
        lines = sdp.split("\r\n")
        new_lines = []
        
        # First identify the VP8 and OPUS payload types
        vp8_pt = None
        opus_pt = None
        payload_to_codec = {}
        current_section = None
        
        for line in lines:
            if line.startswith("m=video "):
                current_section = "video"
            elif line.startswith("m=audio "):
                current_section = "audio"
            
            if line.startswith("a=rtpmap:"):
                parts = line.split(" ")
                if len(parts) >= 2:
                    pt = parts[0].split(":")[1]
                    codec_info = parts[1]
                    codec = codec_info.split("/")[0]
                    
                    if codec.upper() == "VP8":
                        vp8_pt = pt
                        codec_logger.info(f"✅ Found VP8 payload type: {vp8_pt}")
                    elif codec.upper() == "OPUS":
                        opus_pt = pt
                        codec_logger.info(f"✅ Found OPUS payload type: {opus_pt}")
                    
                    payload_to_codec[pt] = codec_info
        
        if not vp8_pt:
            codec_logger.error(f"❌ No VP8 codec found in {direction} SDP!")
            raise ValueError(f"VP8 codec not found in {direction} SDP")
            
        if not opus_pt:
            codec_logger.warning(f"⚠️ No OPUS codec found in {direction} SDP. Audio may not be available.")
        
        # Modify the SDP to enforce codec restrictions
        current_section = None
        m_video_modified = False
        m_audio_modified = False
        
        for line in lines:
            # Track current media section
            if line.startswith("m=video "):
                current_section = "video"
                # Modify the m=video line to only include VP8 payload type
                if not m_video_modified and vp8_pt:
                    parts = line.split(" ")
                    if len(parts) > 3:
                        new_line = " ".join(parts[:3]) + " " + vp8_pt
                        new_lines.append(new_line)
                        codec_logger.info(f"✏️ Modified m=video line to VP8 only: {new_line}")
                        m_video_modified = True
                        continue
            elif line.startswith("m=audio "):
                current_section = "audio"
                # Modify the m=audio line to only include OPUS payload type if available
                if not m_audio_modified and opus_pt:
                    parts = line.split(" ")
                    if len(parts) > 3:
                        new_line = " ".join(parts[:3]) + " " + opus_pt
                        new_lines.append(new_line)
                        codec_logger.info(f"✏️ Modified m=audio line to OPUS only: {new_line}")
                        m_audio_modified = True
                        continue
            
            # Skip lines for non-preferred codecs
            if line.startswith("a=rtpmap:") or line.startswith("a=rtcp-fb:") or line.startswith("a=fmtp:"):
                pt = line.split(":")[1].split(" ")[0]
                
                # Skip non-VP8 video codec lines
                if current_section == "video" and pt in payload_to_codec:
                    codec_info = payload_to_codec[pt]
                    if "VP8" not in codec_info and pt != vp8_pt:
                        codec_logger.debug(f"🗑️ Removing non-VP8 video codec line: {line}")
                        continue
                
                # Skip non-OPUS audio codec lines if OPUS is available
                if current_section == "audio" and opus_pt and pt in payload_to_codec:
                    codec_info = payload_to_codec[pt]
                    if "opus" not in codec_info.lower() and pt != opus_pt:
                        codec_logger.debug(f"🗑️ Removing non-OPUS audio codec line: {line}")
                        continue
            
            new_lines.append(line)
        
        # Add VP8-specific optimization parameters if not already present
        vp8_fmtp_found = False
        for line in new_lines:
            if line.startswith(f"a=fmtp:{vp8_pt} "):
                vp8_fmtp_found = True
                break
        
        if not vp8_fmtp_found and vp8_pt:
            # Insert VP8 parameters after the rtpmap line
            for i, line in enumerate(new_lines):
                if line.startswith(f"a=rtpmap:{vp8_pt} VP8"):
                    new_lines.insert(i+1, f"a=fmtp:{vp8_pt} max-fs=12288;max-fr=30;x-google-start-bitrate=1500")
                    codec_logger.info(f"➕ Added VP8 parameters for payload type {vp8_pt}")
                    break
        
        modified_sdp = "\r\n".join(new_lines)
        
        # Log a comparison summary
        orig_codec_map, orig_video_pts, orig_audio_pts = CodecEnforcer.log_codec_info(sdp, f"original {direction}")
        new_codec_map, new_video_pts, new_audio_pts = CodecEnforcer.log_codec_info(modified_sdp, f"modified {direction}")
        
        codec_logger.info(f"SDP modification summary ({direction}):")
        codec_logger.info(f"  - Original video codecs: {[orig_codec_map.get(pt) for pt in orig_video_pts]}")
        codec_logger.info(f"  - Modified video codecs: {[new_codec_map.get(pt) for pt in new_video_pts]}")
        codec_logger.info(f"  - Original audio codecs: {[orig_codec_map.get(pt) for pt in orig_audio_pts]}")
        codec_logger.info(f"  - Modified audio codecs: {[new_codec_map.get(pt) for pt in new_audio_pts]}")
        
        return modified_sdp

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
        self.codec_info_logged = False

    async def process_frames(self):
        while self.running:
            try:
                # Use a timeout to prevent blocking forever
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                current_time = time.time()
                
                if self.frame_count % self.skip_factor == 0:
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Check and log codec information on first frame
                    if not self.codec_info_logged and hasattr(frame, 'codec') and frame.codec:
                        codec_name = getattr(frame.codec, 'name', 'unknown')
                        if codec_name.upper() != 'VP8':
                            codec_logger.error(f"❌ Non-VP8 codec detected in frame: {codec_name}")
                        else:
                            codec_logger.info(f"✅ VP8 codec confirmed in frame: {codec_name}")
                        self.codec_info_logged = True
                    
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
            
            # Log codec information if available
            if not self.codec_info_logged and hasattr(self.track, 'codec') and self.track.codec:
                codec_name = getattr(self.track.codec, 'name', 'unknown')
                if codec_name.upper() != 'VP8':
                    codec_logger.error(f"❌ Non-VP8 codec detected in track: {codec_name}")
                    codec_logger.error(f"Track details: {self.track}")
                else:
                    codec_logger.info(f"✅ VP8 codec confirmed in track: {codec_name}")
                self.codec_info_logged = True
            
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
            
            # Check if error might be codec-related
            error_str = str(e).lower()
            if "codec" in error_str or "decode" in error_str or "format" in error_str:
                codec_logger.error(f"❌ Possible codec-related error: {e}")
            
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

class AudioPassthroughTrack(MediaStreamTrack):
    """Simple audio track that passes through the audio without modification"""
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.codec_info_logged = False
        codec_logger.info(f"🔊 Audio passthrough track created")
        
    async def recv(self):
        try:
            frame = await self.track.recv()
            
            # Log codec information if available (only once)
            if not self.codec_info_logged and hasattr(self.track, 'codec') and self.track.codec:
                codec_name = getattr(self.track.codec, 'name', 'unknown')
                if codec_name.upper() != 'OPUS':
                    codec_logger.warning(f"⚠️ Non-OPUS audio codec detected in track: {codec_name}")
                else:
                    codec_logger.info(f"✅ OPUS audio codec confirmed in track: {codec_name}")
                self.codec_info_logged = True
                
            return frame
        except Exception as e:
            codec_logger.error(f"Error in audio processing: {e}")
            # Just rethrow the exception - audio errors are less critical and don't need complex handling
            raise

async def create_peer_connection():
    peer = RTCPeerConnection()
    
    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            logger.info("🎥 Received video track")
            
            # Log track info
            track_info = {
                "id": getattr(track, "id", "unknown"),
                "kind": track.kind,
                "codec": getattr(track, "codec", "unknown"),
            }
            codec_logger.info(f"Video track info: {track_info}")
            
            # Check codec directly if available
            if hasattr(track, 'codec') and track.codec:
                codec_name = getattr(track.codec, 'name', 'unknown')
                if codec_name.upper() != 'VP8':
                    codec_logger.error(f"❌ Non-VP8 codec detected in incoming video track: {codec_name}")
                else:
                    codec_logger.info(f"✅ Incoming video track using VP8 codec: {codec_name}")
            
            relayed_track = relay.subscribe(track)
            video_processor = VideoProcessorTrack(relayed_track)
            peer.addTrack(video_processor)
            
        elif track.kind == "audio":
            logger.info("🔊 Received audio track")
            
            # Log track info
            track_info = {
                "id": getattr(track, "id", "unknown"),
                "kind": track.kind,
                "codec": getattr(track, "codec", "unknown"),
            }
            codec_logger.info(f"Audio track info: {track_info}")
            
            # Check codec directly if available
            if hasattr(track, 'codec') and track.codec:
                codec_name = getattr(track.codec, 'name', 'unknown')
                if codec_name.upper() != 'OPUS':
                    codec_logger.warning(f"⚠️ Non-OPUS codec detected in incoming audio track: {codec_name}")
                else:
                    codec_logger.info(f"✅ Incoming audio track using OPUS codec: {codec_name}")
            
            # Process and echo back audio
            relayed_audio = relay.subscribe(track)
            audio_track = AudioPassthroughTrack(relayed_audio)
            peer.addTrack(audio_track)
            
            # Also send audio to a sink (no playback needed for server)
            audio_sink.addTrack(relayed_audio)

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {peer.connectionState}")
        if peer.connectionState == "failed":
            codec_logger.warning("WebRTC connection failed - possible codec mismatch")
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
        
        # Log original codec information
        codec_logger.info("Analyzing original offer SDP...")
        CodecEnforcer.log_codec_info(offer_sdp, "offer")
        
        # Enforce VP8 for video and OPUS for audio in the offer
        codec_logger.info("Enforcing VP8 video and OPUS audio in offer SDP...")
        modified_offer_sdp = CodecEnforcer.enforce_codecs(offer_sdp, "offer")
        
        # Create and set the remote description with the modified offer
        offer = RTCSessionDescription(sdp=modified_offer_sdp, type=params["type"])
        await peer.setRemoteDescription(offer)

        # Configure codec preferences to strictly use VP8 for video and OPUS for audio
        for transceiver in peer.getTransceivers():
            if transceiver.kind == "video":
                codecs = RTCRtpSender.getCapabilities("video").codecs
                
                # Only keep VP8 codec for video
                vp8_codecs = [codec for codec in codecs if codec.mimeType.lower() == "video/vp8"]
                
                if not vp8_codecs:
                    codec_logger.error("❌ VP8 codec not found in video capabilities!")
                    for codec in codecs:
                        codec_logger.debug(f"Available video codec: {codec.mimeType}")
                    raise ValueError("VP8 codec not available")
                
                codec_logger.info(f"✅ Setting video codec preferences to VP8 only: {[c.mimeType for c in vp8_codecs]}")
                transceiver.setCodecPreferences(vp8_codecs)
                
            elif transceiver.kind == "audio":
                codecs = RTCRtpSender.getCapabilities("audio").codecs
                
                # Prefer OPUS codec for audio
                opus_codecs = [codec for codec in codecs if codec.mimeType.lower() == "audio/opus"]
                other_codecs = [codec for codec in codecs if codec.mimeType.lower() != "audio/opus"]
                
                if not opus_codecs:
                    codec_logger.warning("⚠️ OPUS codec not found in audio capabilities! Using available codecs.")
                    codec_logger.debug(f"Available audio codecs: {[c.mimeType for c in codecs]}")
                    # Don't raise an error, just use whatever codecs are available
                else:
                    preferred_codecs = opus_codecs + other_codecs
                    codec_logger.info(f"✅ Setting audio codec preferences to prefer OPUS: {[c.mimeType for c in preferred_codecs]}")
                    transceiver.setCodecPreferences(preferred_codecs)

        # Create the answer
        answer = await peer.createAnswer()
        
        # Enforce VP8 for video and OPUS for audio in the answer
        codec_logger.info("Enforcing VP8 video and OPUS audio in answer SDP...")
        modified_answer_sdp = CodecEnforcer.enforce_codecs(answer.sdp, "answer")
        
        # Set the modified answer as the local description
        modified_answer = RTCSessionDescription(sdp=modified_answer_sdp, type=answer.type)
        await peer.setLocalDescription(modified_answer)
        
        # Final codec verification in answer
        CodecEnforcer.log_codec_info(modified_answer_sdp, "final answer")
        
    except Exception as e:
        logger.error(f"❌ Error in WebRTC setup: {e}")
        codec_logger.error(f"WebRTC setup failed: {e}", exc_info=True)
        await peer.close()
        pcs.discard(peer)
        return web.json_response({"error": str(e), "details": "Possible codec compatibility issue"}, status=500)

    logger.info("📤 Sending SDP answer (VP8 video + OPUS audio)")
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
    
    # Close audio sink
    await audio_sink.stop()
    
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

# Add an endpoint to check codec status
async def codec_status(request):
    """Return the current codec status and logs"""
    try:
        # Read the most recent codec logs
        if os.path.exists("codec_errors.log"):
            with open("codec_errors.log", "r") as f:
                logs = f.readlines()
                # Get last 20 lines
                recent_logs = logs[-20:] if len(logs) > 20 else logs
        else:
            recent_logs = ["No codec logs available yet"]
        
        # Count active connections and track codec usage
        vp8_video_tracks = 0
        other_video_tracks = 0
        opus_audio_tracks = 0
        other_audio_tracks = 0
        
        for peer in pcs:
            for receiver in peer.getReceivers():
                if receiver.track:
                    if receiver.track.kind == "video":
                        if hasattr(receiver.track, 'codec') and receiver.track.codec:
                            codec_name = getattr(receiver.track.codec, 'name', '').upper()
                            if codec_name == 'VP8':
                                vp8_video_tracks += 1
                            else:
                                other_video_tracks += 1
                    elif receiver.track.kind == "audio":
                        if hasattr(receiver.track, 'codec') and receiver.track.codec:
                            codec_name = getattr(receiver.track.codec, 'name', '').upper()
                            if codec_name == 'OPUS':
                                opus_audio_tracks += 1
                            else:
                                other_audio_tracks += 1
        
        return web.json_response({
            "status": "active",
            "connections": len(pcs),
            "video_tracks": {
                "vp8": vp8_video_tracks,
                "other": other_video_tracks
            },
            "audio_tracks": {
                "opus": opus_audio_tracks,
                "other": other_audio_tracks
            },
            "recent_logs": recent_logs
        })
    except Exception as e:
        logger.error(f"Error in codec status endpoint: {e}")
        return web.json_response({"error": str(e)}, status=500)

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
offer_route = cors.add(app.router.add_resource("/offer"))
cors.add(offer_route.add_route("POST", offer))

# Add codec status endpoint
status_route = cors.add(app.router.add_resource("/codec-status"))
cors.add(status_route.add_route("GET", codec_status))

if __name__ == "__main__":
    logger.info("🚀 Starting WebRTC server (VP8 video + OPUS audio) on http://0.0.0.0:5000")
    codec_logger.info("VP8 video and OPUS audio codec enforcement active")
    
    # Set up signal handlers for clean shutdown
    handle_exit_signals()
    
    try:
        web.run_app(app, host='0.0.0.0', port=5000)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        loop = asyncio.get_event_loop()
        asyncio.run(cleanup_resources())