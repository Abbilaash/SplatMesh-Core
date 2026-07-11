#!/usr/bin/env python3
import os
import time
import struct
import asyncio
from pathlib import Path
from aiohttp import web
import aiohttp
from colmap_wrapper import run_colmap_reconstruction

PORT = 3000
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# Resolve absolute paths relative to python_relay script location
BASE_DATA_DIR = (script_dir / ".." / "data").resolve()
MOBILE_DIR = (script_dir / ".." / ".." / "mobile").resolve()

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# State for current session
session_state = {
    "recording": False,
    "current_dir": None,
    "frame_index": 0
}

def save_frame(filepath: Path, data: bytes):
    """Write binary data to disk (helper for thread executor)."""
    try:
        with open(filepath, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"[SERVER] Error saving frame to {filepath}: {e}")

def detect_hardware():
    """Detect available hardware accelerator and suggest frame width."""
    hardware = "CPU"
    suggested_width = 640
    try:
        # pyrefly: ignore [missing-import]
        import onnxruntime as ort
        if "QNNExecutionProvider" in ort.get_available_providers():
            hardware = "NPU"
            suggested_width = 1280
            return hardware, suggested_width
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            hardware = "CUDA"
            suggested_width = 1280
            return hardware, suggested_width
    except ImportError:
        pass

    return hardware, suggested_width

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    peername = request.transport.get_extra_info('peername')
    client_ip = peername[0] if peername else "unknown"
    print(f"[SERVER] Phone connected from {client_ip}")

    # Send hardware configuration handshake to the phone client
    hardware_acc, suggested_width = detect_hardware()
    print(f"[SERVER] Detected acceleration: {hardware_acc}. Suggesting {suggested_width}px width.")
    await ws.send_json({
        "hardware_acceleration": hardware_acc,
        "suggested_width": suggested_width
    })

    loop = asyncio.get_event_loop()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                cmd = msg.data.strip()
                if cmd == "START":
                    session_name = f"session_{int(time.time())}"
                    session_state["current_dir"] = BASE_DATA_DIR / session_name
                    session_state["current_dir"].mkdir(parents=True, exist_ok=True)
                    # Create images subfolder for clean frame storage
                    (session_state["current_dir"] / "images").mkdir(parents=True, exist_ok=True)
                    session_state["recording"] = True
                    session_state["frame_index"] = 0
                    print(f"[SERVER] Started new session: {session_name}")
                    
                elif cmd == "STOP":
                    print("[SERVER] Stopped session. Initiating COLMAP reconstruction...")
                    session_state["recording"] = False
                    
                    if session_state["current_dir"] and session_state["current_dir"].exists():
                        session_dir = session_state["current_dir"]
                        # Run COLMAP reconstruction asynchronously in a background thread to prevent blocking
                        loop.run_in_executor(None, run_colmap_reconstruction, session_dir)
                    
                    session_state["current_dir"] = None

            elif msg.type == aiohttp.WSMsgType.BINARY:
                if not session_state["recording"] or not session_state["current_dir"]:
                    continue

                data = msg.data
                if len(data) < 32:
                    continue  # Invalid packet

                # Parse the 32-byte header:
                # - timestamp (8 bytes, int64, little-endian)
                # - pitch (4 bytes, float, little-endian)
                # - roll (4 bytes, float, little-endian)
                # - yaw (4 bytes, float, little-endian)
                try:
                    header = data[:32]
                    ts = struct.unpack('<q', header[0:8])[0]
                    pitch = struct.unpack('<f', header[8:12])[0]
                    roll = struct.unpack('<f', header[12:16])[0]
                    yaw = struct.unpack('<f', header[16:20])[0]
                    
                    image_data = data[32:]
                    session_state["frame_index"] += 1
                    idx = session_state["frame_index"]

                    filename = f"frame_{ts}_{idx:06d}_P{pitch:.1f}_R{roll:.1f}_Y{yaw:.1f}.jpg"
                    filepath = session_state["current_dir"] / "images" / filename

                    # Offload file I/O to execution thread pool to avoid blocking asyncio event loop
                    await loop.run_in_executor(None, save_frame, filepath, image_data)
                except Exception as e:
                    print(f"[SERVER] Error parsing binary frame: {e}")

    except Exception as e:
        print(f"[SERVER] Connection error with client: {e}")
    finally:
        print(f"[SERVER] Phone disconnected ({client_ip})")
        if session_state["recording"]:
            print("[SERVER] Connection lost during recording. Stopping session.")
            session_state["recording"] = False
            session_state["current_dir"] = None
            
    return ws

async def serve_root(request):
    """Serve the phone_stream.html file directly at root."""
    html_path = MOBILE_DIR / "phone_stream.html"
    if html_path.exists():
        return web.FileResponse(html_path)
    return web.Response(text="phone_stream.html not found", status=404)

def make_app():
    app = web.Application()
    app.router.add_get('/', serve_root)
    app.router.add_get('/ws', handle_websocket)
    # Serve static assets (js, css, etc.) from the mobile directory
    app.router.add_static('/', path=MOBILE_DIR)
    return app

if __name__ == '__main__':
    app = make_app()
    print(f"[SERVER] Running Python Relay Server on http://0.0.0.0:{PORT}")
    web.run_app(app, host='0.0.0.0', port=PORT)
