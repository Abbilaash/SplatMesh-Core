#!/usr/bin/env python3
import os
import time
import struct
import asyncio
import subprocess
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
import aiohttp

# Import your existing COLMAP wrapper
from colmap_wrapper import run_colmap_reconstruction

PORT = 3000
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

BASE_DATA_DIR = (script_dir / "data").resolve()
MOBILE_DIR = (script_dir / "mobile").resolve()
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

BLUR_THRESHOLD = 20.0

# Session state
session_state = {
    "recording": False,
    "current_dir": None,
    "frame_index": 0,
    "sharp_frames_saved": 0
}

def process_and_save_frame(filepath: Path, image_bytes: bytes, pitch: float, roll: float, yaw: float):
    """
    Decodes the JPEG in RAM, checks sharpness, and saves to disk only if it passes.
    Runs in a ThreadPoolExecutor to prevent blocking the async WebSocket loop.
    """
    try:
        # Decode JPEG bytes directly from memory
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False

        # Calculate Laplacian Variance (Sharpness)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Dynamic Culling
        if variance >= BLUR_THRESHOLD:
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return True
        return False
        
    except Exception as e:
        print(f"[SERVER] Error processing frame: {e}")
        return False

def run_reconstruction_and_convert(session_dir: Path) -> None:
    """Runs COLMAP and immediately converts to PLY."""
    print("\n[PIPELINE] Starting GPU Sparse Reconstruction...")
    
    # 1. Run COLMAP Pipeline
    success = run_colmap_reconstruction(session_dir)
    
    if not success:
        print("[ERROR] COLMAP pipeline failed. Aborting PLY conversion.")
        return

    # 2. Convert to PLY directly using PyCOLMAP
    sparse_dir = session_dir / "sparse" / "0"
    ply_path = sparse_dir / "model.ply"
    
    if sparse_dir.exists() and (sparse_dir / "cameras.bin").exists():
        print(f"\n[PIPELINE] Converting output to PLY...")
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(str(sparse_dir))
            reconstruction.export_PLY(str(ply_path))
            print(f"\n[SUCCESS] Final point cloud generated: {ply_path}")
            print("Ready to open in Blender!")
        except ImportError:
            print("\n[PIPELINE] pycolmap not found. Falling back to COLMAP CLI for conversion...")
            convert_cmd = [
                "C:/colmap/COLMAP.bat", "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(ply_path),
                "--output_type", "PLY"
            ]
            subprocess.run(convert_cmd, stdout=subprocess.DEVNULL)
            print(f"\n[SUCCESS] Final point cloud generated: {ply_path}")
    else:
        print("\n[ERROR] Sparse model binaries not found.")

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_ip = request.transport.get_extra_info('peername')[0]
    print(f"[SERVER] Phone connected from {client_ip}")

    await ws.send_json({"hardware_acceleration": "CUDA", "suggested_width": 1280})
    loop = asyncio.get_event_loop()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                cmd = msg.data.strip()
                
                if cmd == "START":
                    session_name = f"session_{int(time.time())}"
                    session_state["current_dir"] = BASE_DATA_DIR / session_name
                    (session_state["current_dir"] / "images").mkdir(parents=True, exist_ok=True)
                    session_state["recording"] = True
                    session_state["frame_index"] = 0
                    session_state["sharp_frames_saved"] = 0
                    print(f"\n[SERVER] Streaming started. Saving to: {session_name}")
                    
                elif cmd == "STOP":
                    print(f"\n[SERVER] Stream stopped. Total sharp frames captured: {session_state['sharp_frames_saved']}")
                    session_state["recording"] = False
                    
                    if session_state["current_dir"] and session_state["sharp_frames_saved"] > 10:
                        session_dir = session_state["current_dir"]
                        # Trigger the reconstruction on a background thread so the server doesn't freeze
                        loop.run_in_executor(None, run_reconstruction_and_convert, session_dir)
                    else:
                        print("[ABORT] Not enough frames to reconstruct.")
                    session_state["current_dir"] = None

            elif msg.type == aiohttp.WSMsgType.BINARY:
                if not session_state["recording"] or not session_state["current_dir"]:
                    continue

                data = msg.data
                if len(data) < 32: continue

                try:
                    # Unpack IMU header telemetry 
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

                    # Send to background thread for real-time sharpness check
                    def callback(future):
                        if future.result():
                            session_state["sharp_frames_saved"] += 1
                            print(f"\rCaptured Sharp Frames: {session_state['sharp_frames_saved']}", end="", flush=True)

                    task = loop.run_in_executor(None, process_and_save_frame, filepath, image_data, pitch, roll, yaw)
                    task.add_done_callback(callback)

                except Exception as e:
                    print(f"[SERVER] Error parsing binary frame: {e}")

    except Exception as e:
        print(f"[SERVER] Connection error: {e}")
    finally:
        print(f"\n[SERVER] Phone disconnected ({client_ip})")
        session_state["recording"] = False
            
    return ws

async def serve_root(request):
    html_path = MOBILE_DIR / "phone_stream.html"
    if html_path.exists(): return web.FileResponse(html_path)
    return web.Response(text="phone_stream.html not found", status=404)

def make_app():
    app = web.Application()
    app.router.add_get('/', serve_root)
    app.router.add_get('/ws', handle_websocket)
    return app

if __name__ == '__main__':
    app = make_app()
    print(f"=====================================================")
    print(f" SplatMesh-Core Engine running on http://0.0.0.0:{PORT}")
    print(f"=====================================================")
    web.run_app(app, host='0.0.0.0', port=PORT)