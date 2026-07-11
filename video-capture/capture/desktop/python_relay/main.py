#!/usr/bin/env python3
import os
import time
import json
import re
import struct
import asyncio
import shutil
import subprocess
import sys
import tempfile
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
KEYFRAME_SCRIPT = (script_dir / ".." / ".." / ".." / "extract_keyframes.py").resolve()

KEYFRAME_FPS = os.environ.get("SPLATMESH_KEYFRAME_FPS", "4")
KEYFRAME_BLUR_THRESHOLD = os.environ.get("SPLATMESH_KEYFRAME_BLUR_THRESHOLD", "100")
KEYFRAME_SIM_THRESHOLD = os.environ.get("SPLATMESH_KEYFRAME_SIM_THRESHOLD", "0.92")

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

def _frame_sort_key(path: Path):
    match = re.search(r"_(\d{6})_P", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)

def _build_session_video(images_dir: Path, video_path: Path) -> bool:
    image_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=_frame_sort_key,
    )
    if not image_files:
        print("[SERVER] No captured images found. Skipping keyframe extraction.")
        return False

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        print("[SERVER] FFmpeg not available on PATH. Skipping keyframe extraction.")
        return False

    with tempfile.TemporaryDirectory(prefix="relay_seq_", dir=str(images_dir.parent)) as seq_dir:
        seq_path = Path(seq_dir)
        for idx, source in enumerate(image_files, start=1):
            destination = seq_path / f"img_{idx:06d}{source.suffix.lower()}"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        first_suffix = image_files[0].suffix.lower()
        pattern = seq_path / f"img_%06d{first_suffix}"
        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(f"[SERVER] Failed to create session video for keyframe extraction: {completed.stderr.strip()}")
            return False

    return video_path.exists()

def _replace_images_with_keyframes(images_dir: Path, keyframes_dir: Path, manifest_path: Path) -> bool:
    selected_files = sorted(
        [
            p for p in keyframes_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not selected_files:
        return False

    backup_dir = images_dir.parent / "images_raw"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    images_dir.rename(backup_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, source in enumerate(selected_files, start=1):
        destination = images_dir / f"keyframe_{idx:06d}{source.suffix.lower()}"
        shutil.move(str(source), str(destination))

    if manifest_path.exists():
        shutil.move(str(manifest_path), str(images_dir.parent / "keyframes.json"))

    shutil.rmtree(keyframes_dir, ignore_errors=True)
    print(f"[SERVER] Replaced raw frames with {len(selected_files)} selected keyframes for COLMAP.")
    return True

def prepare_keyframes_for_colmap(session_dir: Path) -> None:
    images_dir = session_dir / "images"
    if not images_dir.exists():
        print("[SERVER] Images directory missing. Skipping keyframe preparation.")
        return

    if not KEYFRAME_SCRIPT.exists():
        print(f"[SERVER] Keyframe script not found at {KEYFRAME_SCRIPT}. Skipping keyframe preparation.")
        return

    video_path = session_dir / "session_capture.mp4"
    if not _build_session_video(images_dir, video_path):
        return

    keyframes_dir = session_dir / "images_keyframes"
    if keyframes_dir.exists():
        shutil.rmtree(keyframes_dir)

    command = [
        sys.executable,
        str(KEYFRAME_SCRIPT),
        "--video-path",
        str(video_path),
        "--output-dir",
        str(keyframes_dir),
        "--fps",
        KEYFRAME_FPS,
        "--blur-threshold",
        KEYFRAME_BLUR_THRESHOLD,
        "--keyframe-similarity-threshold",
        KEYFRAME_SIM_THRESHOLD,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[SERVER] Keyframe extraction failed; continuing with raw frames.\n{completed.stderr.strip()}")
        return

    manifest_path = keyframes_dir / "keyframes.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected_count = int(payload.get("selected_count", 0))
            print(f"[SERVER] Keyframe extraction completed. Selected {selected_count} frames.")
        except Exception:
            print("[SERVER] Keyframe extraction completed.")

    if not _replace_images_with_keyframes(images_dir, keyframes_dir, manifest_path):
        print("[SERVER] No keyframes selected; keeping original frames for COLMAP.")

def run_reconstruction_pipeline(session_dir: Path) -> None:
    try:
        prepare_keyframes_for_colmap(session_dir)
    except Exception as exc:
        print(f"[SERVER] Keyframe preparation failed unexpectedly: {exc}")
    run_colmap_reconstruction(session_dir)

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
                        # Build keyframes from captured frames first, then run COLMAP in a worker thread.
                        loop.run_in_executor(None, run_reconstruction_pipeline, session_dir)
                    
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
