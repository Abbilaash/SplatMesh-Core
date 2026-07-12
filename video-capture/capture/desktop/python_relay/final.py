#!/usr/bin/env python3
import os
import time
import struct
import asyncio
import subprocess
import sys
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

BLUR_THRESHOLD = 0.0

session_state = {
    "recording": False,
    "current_dir": None,
    "frame_index": 0,
    "raw_frames_saved": 0
}

def save_frame_raw(filepath: Path, image_bytes: bytes):
    """Phase 1: I/O Bound. Blindly write bytes to disk as fast as possible."""
    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return True
    except Exception:
        return False

def filter_and_reconstruct(session_dir: Path):
    """Phase 2: CPU Bound. Runs only AFTER you hit Stop or the connection drops."""
    
    # Allow the OS I/O buffer to finish writing the final frames to disk
    time.sleep(1.5)
    print("\n") # Clear the carriage return from the raw frames counter
    
    images_dir = session_dir / "images"
    all_images = list(images_dir.glob("*.jpg"))
    
    print(f"[PIPELINE] Filtering {len(all_images)} captured frames...")
    
    sharp_count = 0
    blurry_count = 0
    
    # 1. Filter out the blurry frames
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            img_path.unlink()
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if variance < BLUR_THRESHOLD:
            img_path.unlink()
            blurry_count += 1
        else:
            sharp_count += 1
            
    print(f"[PIPELINE] Filtering complete. Kept {sharp_count} sharp frames. Deleted {blurry_count} blurry frames.")
    
    if sharp_count < 10:
        print("[ERROR] Not enough sharp frames left to run COLMAP. Aborting pipeline.")
        return

    # 2. Run COLMAP
    print("\n[PIPELINE] Starting GPU Sparse Reconstruction...")
    success = run_colmap_reconstruction(session_dir)
    
    if not success:
        print("[ERROR] COLMAP pipeline failed.")
        return

    # 3. Convert to PLY
    sparse_dir = session_dir / "sparse" / "0"
    ply_path = sparse_dir / "model.ply"
    
    if sparse_dir.exists() and (sparse_dir / "cameras.bin").exists():
        print(f"\n[PIPELINE] Converting output to PLY...")
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(str(sparse_dir))
            reconstruction.export_PLY(str(ply_path))
            print(f"\n[SUCCESS] Final point cloud generated: {ply_path}")
        except ImportError:
            convert_cmd = [
                "C:/colmap/COLMAP.bat", "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(ply_path),
                "--output_type", "PLY"
            ]
            subprocess.run(convert_cmd, stdout=subprocess.DEVNULL)
            print(f"\n[SUCCESS] Final point cloud generated: {ply_path}")

async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    client_ip = request.transport.get_extra_info('peername')[0]
    print(f"\n[SERVER] Phone connected from {client_ip}")

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
                    session_state["raw_frames_saved"] = 0
                    print(f"\n[SERVER] Streaming started. Saving to: {session_name}")
                    
                elif cmd == "STOP":
                    print(f"\n[SERVER] Stream stopped cleanly by user. Total raw frames: {session_state['raw_frames_saved']}")
                    session_state["recording"] = False
                    
                    if session_state["current_dir"]:
                        session_dir = session_state["current_dir"]
                        loop.run_in_executor(None, filter_and_reconstruct, session_dir)
                    session_state["current_dir"] = None

            elif msg.type == aiohttp.WSMsgType.BINARY:
                if not session_state["recording"] or not session_state["current_dir"]:
                    continue

                data = msg.data
                if len(data) < 32: continue

                try:
                    header = data[:32]
                    ts = struct.unpack('<q', header[0:8])[0]
                    
                    image_data = data[32:]
                    session_state["frame_index"] += 1
                    idx = session_state["frame_index"]

                    filename = f"frame_{ts}_{idx:06d}.jpg"
                    filepath = session_state["current_dir"] / "images" / filename

                    # THE FIX: Awaiting the executor creates natural network backpressure!
                    # The server will not accept the next frame until this one is safely on the disk.
                    success = await loop.run_in_executor(None, save_frame_raw, filepath, image_data)
                    
                    if success:
                        session_state["raw_frames_saved"] += 1
                        sys.stdout.write(f"\rCaptured Raw Frames: {session_state['raw_frames_saved']}  ")
                        sys.stdout.flush()

                except Exception as e:
                    print(f"[SERVER] Error parsing binary frame: {e}")

    except Exception as e:
        print(f"\n[SERVER] Connection error: {e}")
    finally:
        print(f"\n[SERVER] Phone disconnected ({client_ip})")
        
        # SAFETY NET: If the socket severed before the STOP packet arrived
        if session_state["recording"] and session_state["current_dir"]:
            print("[SERVER] Stream interrupted abruptly. Recovering session and triggering COLMAP...")
            session_dir = session_state["current_dir"]
            loop.run_in_executor(None, filter_and_reconstruct, session_dir)
            
        session_state["recording"] = False
        session_state["current_dir"] = None
            
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