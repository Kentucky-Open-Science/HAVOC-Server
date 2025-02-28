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

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Example processing (convert to grayscale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Processed Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return frame
    
def fix_sdp(sdp):
    """Ensures SDP is properly formatted to avoid parsing errors while preserving required attributes."""
    if not sdp:
        print("❌ Error: Empty SDP received.")
        return ""

    lines = sdp.splitlines()
    new_lines = []
    found_setup = False  # Track if we have set `a=setup`

    for line in lines:
        if line.startswith("a=setup:"):
            if not found_setup:
                print("✅ Keeping first setup line:", line)
                new_lines.append("a=setup:actpass")  # Ensure correct WebRTC behavior
                found_setup = True
            else:
                print("⚠️ Removing duplicate setup line:", line)  # Prevent multiple setups
            continue

        new_lines.append(line)

    # Ensure we have at least one `m=` line (Media description)
    has_media = any(line.startswith("m=") for line in new_lines)
    if not has_media:
        print("❌ Error: No media section found in SDP!")
        return None  # Return `None` to indicate failure

    final_sdp = "\n".join(new_lines)
    print("\n✅ Final Fixed SDP:\n", final_sdp)  # Debugging output
    return final_sdp



async def offer(request):
    params = await request.json()
    print("\n🔹 Received Offer SDP:")
    print(params["sdp"])  # Debugging: Show offer SDP

    peer = RTCPeerConnection()
    
    @peer.on("track")
    def on_track(track):
        if track.kind == "video":
            peer.addTrack(VideoProcessorTrack(track))

    # Fix SDP before setting remote description
    offer_sdp = fix_sdp(params["sdp"])
    if not offer_sdp:
        return web.Response(status=400, text=json.dumps({"error": "Invalid SDP: No media section found"}))

    try:
        offer_sdp = RTCSessionDescription(sdp=offer_sdp, type=params["type"])
        await peer.setRemoteDescription(offer_sdp)
    except Exception as e:
        print("❌ Error setting remote description:", str(e))
        return web.Response(status=500, text=json.dumps({"error": str(e)}))

    # Create an answer SDP
    answer = await peer.createAnswer()
    await peer.setLocalDescription(answer)

    # Print the raw Answer SDP before fixing it
    print("\n🔹 Raw Generated Answer SDP:")
    print(peer.localDescription.sdp)

    # Fix SDP and return
    fixed_sdp = fix_sdp(peer.localDescription.sdp)
    if not fixed_sdp:
        return web.Response(status=500, text=json.dumps({"error": "Failed to fix SDP"}))

    print("\n✅ Final Fixed Answer SDP:")
    print(fixed_sdp)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": fixed_sdp, "type": peer.localDescription.type})
    )


# Setup CORS for the /offer route
offer_resource = cors.add(app.router.add_post("/offer", offer))

web.run_app(app, port=5000)