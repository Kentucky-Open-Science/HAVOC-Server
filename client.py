import asyncio
import json
import logging
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRelay

logging.basicConfig(level=logging.INFO)

VIDEO_SOURCE = "video=Logi Webcam C920e"
SERVER_URL = "http://localhost:5000/offer"

ICE_SERVERS = [
    RTCIceServer("stun:stun.l.google.com:19302"),
]

async def start_stream():
    async with aiohttp.ClientSession() as session:
        config = RTCConfiguration(iceServers=ICE_SERVERS)
        peer = RTCPeerConnection(configuration=config)
        relay = MediaRelay()

        try:
            player = MediaPlayer(
                VIDEO_SOURCE, 
                format="dshow", 
                options={
                    "video_size": "640x480",
                    "rtbufsize": "2M",  # Further reduced buffer size
                    "framerate": "20",  # Reduced framerate
                    "fflags": "nobuffer",
                    "probesize": "32",
                    "analyzeduration": "0",
                    "videobitrate": "1.5M",  # Adjusted bitrate
                }
            )
            video_track = relay.subscribe(player.video)

            peer.addTrack(video_track)
            logging.info("🎥 Added video track to WebRTC")

            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)

            logging.info("📡 Sending offer to server...")
            async with session.post(SERVER_URL, json={
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type
            }) as response:
                if response.status != 200:
                    logging.error(f"❌ Server error: {response.status} {await response.text()}")
                    return

                answer = await response.json()
                await peer.setRemoteDescription(RTCSessionDescription(
                    sdp=answer["sdp"], type=answer["type"]
                ))

            logging.info("✅ WebRTC connection established")

            while True:
                await asyncio.sleep(1)
                if peer.connectionState == "failed":
                    logging.warning("Connection failed, restarting...")
                    await peer.close()
                    return await start_stream()

        except Exception as e:
            logging.error(f"❌ Error: {e}")
        finally:
            await peer.close()

if __name__ == "__main__":
    asyncio.run(start_stream())