import asyncio
import json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

# Use an MP4 file instead of a camera
video_source = "output.mp4"

async def start_stream():
    """Streams an MP4 file to the WebRTC server."""
    async with aiohttp.ClientSession() as session:
        peer = RTCPeerConnection()

        try:
            # Play the MP4 file in a loop
            player = MediaPlayer(video_source, loop=True)
            peer.addTrack(player.video)

            print("🎥 Adding track to WebRTC: video")

            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)

            print("📡 Sending offer to server...")
            async with session.post("http://localhost:5000/offer", json={
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type
            }) as response:

                if response.status != 200:
                    print(f"❌ Server returned error: {response.status} {await response.text()}")
                    return

                answer = await response.json()  # ✅ Properly decode JSON
                await peer.setRemoteDescription(RTCSessionDescription(
                    sdp=answer["sdp"], type=answer["type"]
                ))

            print("✅ MP4 video stream started!")
            await asyncio.sleep(3600)  # Keep the stream running

        except Exception as e:
            print(f"❌ Error accessing video file: {e}")

if __name__ == "__main__":
    asyncio.run(start_stream())
