# Temi Robot WebRTC Streaming Server

This repository contains a WebRTC streaming server that enables video streaming from a Temi robot within the University of Kentucky Center for Applied AI (CAAI) office environment.

## Overview

The server consists of two main components:
- An aiohttp-based WebRTC signaling server that handles peer connections
- A Flask web server that provides a simple HTTP interface for viewing the video stream

The system handles video frames from the Temi robot, processes them, and makes them available through a web interface. It includes fail-safe mechanisms like freeze detection and fallback to a placeholder image when the stream is interrupted.

## Requirements

- Python 3.8+
- OpenCV
- aiortc
- aiohttp
- Flask
- numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/uky-caai/temi-stream.git
cd temi-stream

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Ensure you have a placeholder image named "temiFace for TV.png" in the root directory
2. The default ports are:
   - 5432 for the WebRTC signaling server
   - 8133 for the Flask video stream server

## Running the Server

```bash
# Start the server
python app.py
```

The server will start both components:
- WebRTC signaling server at http://0.0.0.0:5432
- Flask video feed server at http://0.0.0.0:8133

## Usage

1. Access the video stream by navigating to `http://[server-ip]:8133` in a web browser
2. For WebRTC connections, connect to the signaling server at `http://[server-ip]:5432/offer`

## Troubleshooting

- If stream freezes, the system will automatically switch to a placeholder image after 10 seconds
- Check logs for debugging information (logging level is set to INFO by default)

## License

This project is property of the University of Kentucky Center for Applied AI (CAAI).