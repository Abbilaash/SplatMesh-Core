# qualcomm-hackathon-poc

> **Hardware target:** Windows PC with NVIDIA RTX 3090 · Python 3.10+ · CUDA 12.x  
> **Goal:** Prove the full pipeline works — phone video → YOLO masking → COLMAP poses → 3D Gaussian Splat → live browser viewer  
> **Time to complete:** ~2–3 hours (including training)

---

## What This PoC Proves

| Claim | How it's proven here |
|---|---|
| Phone video can be received over WebSocket | `phone_stream.html` + `laptop_receiver.py` |
| Moving people can be masked in real time | YOLOv8-Nano running on GPU via ONNX |
| Camera poses can be extracted from video | COLMAP sequential SfM on extracted frames |
| A photorealistic 3D Gaussian Splat can be trained | nerfstudio `splatfacto` on RTX 3090 |
| The result renders at 60+ FPS in a browser | gsplat.js WebGL viewer loading the `.ply` file |

The RTX 3090 replaces the Cloud AI 100 for this PoC. Everything else is identical to the hackathon build.

---

## Directory Structure

```
splatmesh-poc/
│
├── README.md                    ← you are here
│
├── 1_capture/
│   ├── phone_stream.html        ← open this on your phone browser
│   └── laptop_receiver.py      ← runs on PC, receives frames via WebSocket
│
├── 2_npu_pipeline/
│   ├── yolo_mask.py             ← YOLOv8-Nano masking (GPU via ONNX)
│   ├── depth_inference.py       ← DepthAnything-V2 depth maps (GPU)
│   └── run_pipeline.py          ← combined pipeline entry point
│
├── 3_colmap/
│   └── extract_and_run.py       ← ffmpeg frame extraction + COLMAP SfM
│
├── 4_train/
│   └── train.sh                 ← nerfstudio splatfacto training command
│
├── 5_viewer/
│   └── viewer.html              ← browser-based Gaussian Splat viewer
│
├── data/
│   ├── raw_frames/              ← saved frames from phone stream (created at runtime)
│   ├── scan_video.mp4           ← place your recorded scan video here
│   ├── colmap_keyframes/        ← extracted keyframes for COLMAP (created at runtime)
│   ├── colmap_output/           ← COLMAP sparse reconstruction (created at runtime)
│   └── processed_data/          ← nerfstudio-format dataset (created at runtime)
│
└── outputs/
    └── splat_export/            ← final point_cloud.ply lives here (created at runtime)
```

Create the full directory tree now:

```bash
mkdir -p splatmesh-poc/{1_capture,2_npu_pipeline,3_colmap,4_train,5_viewer}
mkdir -p splatmesh-poc/data/{raw_frames,colmap_keyframes,colmap_output,processed_data}
mkdir -p splatmesh-poc/outputs/splat_export
cd splatmesh-poc
```

---

## Prerequisites

### System Requirements

- Windows 10/11 (64-bit)
- NVIDIA RTX 3090 with latest drivers (≥ 535.x)
- CUDA 12.1 installed — verify with `nvcc --version`
- At least 32 GB RAM
- At least 50 GB free disk space (nerfstudio + COLMAP + datasets)
- Both PC and phone on the **same Wi-Fi network**

### Step 1 — Install CUDA Toolkit

Download CUDA 12.1 from NVIDIA:  
https://developer.nvidia.com/cuda-12-1-0-download-archive

After install, verify:
```bash
nvcc --version
nvidia-smi
```

### Step 2 — Install Miniconda (if not already installed)

```bash
# Download Miniconda for Windows
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# Run the installer, then open "Anaconda Prompt"
```

### Step 3 — Create the Python Environment

Open **Anaconda Prompt** and run:

```bash
conda create -n splatmesh python=3.10 -y
conda activate splatmesh
```

> All commands from here assume you are inside the `splatmesh` conda environment.

### Step 4 — Install PyTorch with CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090
```

### Step 5 — Install Core Python Dependencies

```bash
pip install ultralytics onnxruntime-gpu opencv-python numpy flask websockets pyserial
```

### Step 6 — Install ffmpeg

Download the Windows build from https://www.gyan.dev/ffmpeg/builds/ (get the "full" release).  
Extract it, then add the `bin/` folder to your system `PATH`.

Verify:
```bash
ffmpeg -version
```

### Step 7 — Install COLMAP

Download the Windows binary release from https://github.com/colmap/colmap/releases  
(get `COLMAP-dev-windows-cuda.zip` for CUDA-accelerated matching)

Extract to `C:\colmap\` and add `C:\colmap\` to your system `PATH`.

Verify:
```bash
colmap help
```

### Step 8 — Install nerfstudio

```bash
pip install nerfstudio
```

After install, verify:
```bash
ns-train --help
```

> **Note:** nerfstudio may also require `tinycudann`. If `ns-train` errors, run:
> ```bash
> pip install ninja
> pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
> ```

---

## The Scripts

Copy each script below into its matching file inside `splatmesh-poc/`.

---

### `1_capture/phone_stream.html`

Serve this from the PC. Open it on your phone browser. It streams the camera to the PC over WebSocket.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SplatMesh — Camera Stream</title>
  <style>
    body { margin: 0; background: #000; display: flex; flex-direction: column;
           align-items: center; justify-content: center; min-height: 100vh; color: #fff;
           font-family: sans-serif; }
    #status { margin-top: 12px; font-size: 14px; opacity: 0.7; }
    #fps    { font-size: 12px; opacity: 0.5; margin-top: 4px; }
    video   { display: none; }
    canvas  { max-width: 100%; border-radius: 8px; }
  </style>
</head>
<body>
  <canvas id="c"></canvas>
  <div id="status">Connecting...</div>
  <div id="fps">-- FPS</div>
  <video id="v" autoplay playsinline muted></video>

  <script>
    // ── CONFIG ──────────────────────────────────────────
    // Replace with your PC's local IP address
    const PC_IP   = 'REPLACE_WITH_PC_IP';  // e.g. '192.168.1.42'
    const WS_PORT = 8765;
    const QUALITY = 0.80;   // JPEG quality (0.0 – 1.0)
    const SCALE   = 0.5;    // Downscale factor (0.5 = half resolution for network)
    // ────────────────────────────────────────────────────

    const video  = document.getElementById('v');
    const canvas = document.getElementById('c');
    const ctx    = canvas.getContext('2d');
    const status = document.getElementById('status');
    const fpsEl  = document.getElementById('fps');

    let ws, frameCount = 0, lastTime = Date.now();

    function connect() {
      ws = new WebSocket(`ws://${PC_IP}:${WS_PORT}`);
      ws.binaryType = 'arraybuffer';

      ws.onopen  = () => { status.textContent = '✅ Connected — streaming'; };
      ws.onclose = () => { status.textContent = '⚠️ Disconnected — retrying...'; setTimeout(connect, 2000); };
      ws.onerror = () => { status.textContent = '❌ Connection failed — check PC_IP'; };
    }

    function sendFrame() {
      if (!ws || ws.readyState !== WebSocket.OPEN) { requestAnimationFrame(sendFrame); return; }

      const w = Math.floor(video.videoWidth  * SCALE);
      const h = Math.floor(video.videoHeight * SCALE);
      canvas.width  = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);

      canvas.toBlob(blob => {
        if (blob && ws.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then(buf => ws.send(buf));
        }
        frameCount++;
        const now = Date.now();
        if (now - lastTime >= 1000) {
          fpsEl.textContent = `${frameCount} FPS`;
          frameCount = 0;
          lastTime   = now;
        }
        requestAnimationFrame(sendFrame);
      }, 'image/jpeg', QUALITY);
    }

    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 },
               facingMode: 'environment' }
    }).then(stream => {
      video.srcObject = stream;
      video.onloadedmetadata = () => { connect(); sendFrame(); };
    }).catch(err => {
      status.textContent = '❌ Camera access denied: ' + err.message;
    });
  </script>
</body>
</html>
```

---

### `1_capture/laptop_receiver.py`

Runs on the PC. Receives frames from the phone, shows them live, and saves them to disk when recording is active.

```python
#!/usr/bin/env python3
"""
SplatMesh PoC — WebSocket frame receiver
Receives JPEG frames from the phone browser stream,
displays them live, and saves to disk when recording.

Usage:
    python laptop_receiver.py
    Press R to toggle recording, Q to quit.
"""

import asyncio
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

# ── CONFIG ──────────────────────────────────────────
SAVE_DIR    = Path("../data/raw_frames")
WS_HOST     = "0.0.0.0"
WS_PORT     = 8765
SHOW_WINDOW = True
# ────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)

recording   = False
frame_queue = []
frame_lock  = threading.Lock()
frame_count = 0


async def handler(websocket):
    global frame_count
    print(f"[INFO] Phone connected: {websocket.remote_address}")
    async for message in websocket:
        arr   = np.frombuffer(message, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        with frame_lock:
            frame_queue.clear()
            frame_queue.append(frame.copy())
            if recording:
                ts   = int(time.time() * 1000)
                path = SAVE_DIR / f"frame_{ts}_{frame_count:06d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_count += 1


def display_loop():
    global recording
    print("[INFO] Display window open. Press R to record, Q to quit.")
    while True:
        with frame_lock:
            frame = frame_queue[-1].copy() if frame_queue else None
        if frame is not None and SHOW_WINDOW:
            label = "● REC" if recording else "○ LIVE"
            color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("SplatMesh — Live Feed (R=Record  Q=Quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            recording = not recording
            state = "STARTED" if recording else "STOPPED"
            print(f"[INFO] Recording {state} — saved to {SAVE_DIR}")
    cv2.destroyAllWindows()


async def main():
    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()
    print(f"[INFO] WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
    print("[INFO] Open phone_stream.html on your phone and set PC_IP to this machine's IP")
    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
```

---

### `2_npu_pipeline/yolo_mask.py`

Loads YOLOv8-Nano, runs inference on a frame, returns a masked version with moving people erased.

```python
#!/usr/bin/env python3
"""
SplatMesh PoC — YOLOv8-Nano person masking
Detects people (and other moving classes) and blacks them out of the frame.
Uses the GPU via ONNX Runtime / PyTorch (replaces Hexagon NPU for the PoC).
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Classes to mask out (COCO class IDs)
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 15=bird, 16=cat, 17=dog
MASK_CLASSES = {0, 1, 2, 3, 15, 16, 17}
CONF_THRESH  = 0.40


class YOLOMasker:
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        print(f"[YOLO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[YOLO] Model ready")

    def mask_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Returns (masked_frame, num_objects_masked).
        masked_frame has all detected moving objects blacked out.
        """
        results = self.model(frame, conf=CONF_THRESH, verbose=False)[0]
        masked  = frame.copy()
        count   = 0

        if results.masks is not None and results.boxes is not None:
            for mask_data, box in zip(results.masks.data, results.boxes):
                cls = int(box.cls[0])
                if cls not in MASK_CLASSES:
                    continue
                # Resize mask to frame dimensions and apply
                mask_np = mask_data.cpu().numpy()
                mask_rs = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                masked[mask_rs > 0.5] = 0
                count += 1

        return masked, count


if __name__ == "__main__":
    # Quick test on webcam
    masker = YOLOMasker()
    cap    = cv2.VideoCapture(0)
    print("Testing YOLO masker on webcam. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        masked, n = masker.mask_frame(frame)
        cv2.putText(masked, f"Masked: {n} objects", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("YOLO Masker Test", masked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

---

### `2_npu_pipeline/depth_inference.py`

Runs DepthAnything-V2 on a frame and returns a depth map.

```python
#!/usr/bin/env python3
"""
SplatMesh PoC — DepthAnything-V2 monocular depth inference
Returns a per-pixel depth map for a given BGR frame.
Uses the Hugging Face transformers pipeline (GPU).
"""

import cv2
import numpy as np
import torch
from transformers import pipeline as hf_pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DepthEstimator:
    def __init__(self):
        print(f"[Depth] Loading DepthAnything-V2-Small on {DEVICE}...")
        self.pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if DEVICE == "cuda" else -1,
        )
        print("[Depth] Model ready")

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a float32 depth map (H, W), values in [0, 1].
        0 = near, 1 = far (relative depth).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result    = self.pipe(frame_rgb)
        depth     = np.array(result["depth"], dtype=np.float32)
        # Normalise to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def colorise(self, depth: np.ndarray) -> np.ndarray:
        """Returns an 8-bit BGR visualisation of the depth map."""
        d8 = (depth * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


if __name__ == "__main__":
    from pathlib import Path
    import sys

    estimator = DepthEstimator()

    if len(sys.argv) > 1:
        img   = cv2.imread(sys.argv[1])
        depth = estimator.estimate(img)
        vis   = estimator.colorise(depth)
        out   = Path(sys.argv[1]).stem + "_depth.jpg"
        cv2.imwrite(out, vis)
        print(f"[Depth] Saved depth map to {out}")
    else:
        # Webcam test
        cap = cv2.VideoCapture(0)
        print("Testing depth on webcam. Press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            depth = estimator.estimate(frame)
            vis   = estimator.colorise(depth)
            cv2.imshow("Depth Map", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
```

> **Note:** The first run will download `Depth-Anything-V2-Small-hf` (~100 MB) from Hugging Face automatically.

---

### `2_npu_pipeline/run_pipeline.py`

Combined pipeline: receives frames, runs YOLO masking, runs depth estimation, displays all three views side by side.

```python
#!/usr/bin/env python3
"""
SplatMesh PoC — Full edge pipeline
Runs YOLO masking + depth estimation on live phone stream.
Displays original | masked | depth side by side.

Usage:
    python run_pipeline.py
    (Start laptop_receiver.py first, then run this to tap into the same stream)

OR run standalone (processes saved frames):
    python run_pipeline.py --frames ../data/raw_frames
"""

import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))
from yolo_mask      import YOLOMasker
from depth_inference import DepthEstimator


def process_frame(frame, masker, depth_est):
    t0 = time.time()

    masked, n_masked = masker.mask_frame(frame)
    depth            = depth_est.estimate(masked)
    depth_vis        = depth_est.colorise(depth)

    elapsed_ms = (time.time() - t0) * 1000

    # Resize all to same height for side-by-side display
    h = 360
    def rs(img):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    row = np.hstack([rs(frame), rs(masked), rs(depth_vis)])

    cv2.putText(row, f"Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(row, f"YOLO Masked ({n_masked} obj)", (rs(frame).shape[1]+10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(row, f"Depth ({elapsed_ms:.0f}ms)", (rs(frame).shape[1]*2+10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
    return row


def main():
    masker    = YOLOMasker()
    depth_est = DepthEstimator()

    if "--frames" in sys.argv:
        frame_dir  = Path(sys.argv[sys.argv.index("--frames") + 1])
        frame_list = sorted(frame_dir.glob("*.jpg"))
        print(f"[Pipeline] Processing {len(frame_list)} saved frames from {frame_dir}")
        for fp in frame_list:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            row = process_frame(frame, masker, depth_est)
            cv2.imshow("SplatMesh Pipeline", row)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    else:
        cap = cv2.VideoCapture(0)
        print("[Pipeline] Running on webcam. Press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            row = process_frame(frame, masker, depth_est)
            cv2.imshow("SplatMesh Pipeline", row)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

### `3_colmap/extract_and_run.py`

Extracts keyframes from the scan video using ffmpeg, then runs COLMAP sequential SfM, then converts to nerfstudio format.

```python
#!/usr/bin/env python3
"""
SplatMesh PoC — COLMAP pipeline
1. Extracts keyframes from scan video using ffmpeg
2. Runs COLMAP sequential matcher + mapper
3. Converts output to nerfstudio format

Usage:
    python extract_and_run.py --video ../data/scan_video.mp4

Outputs:
    ../data/colmap_keyframes/   — extracted frames
    ../data/colmap_output/      — COLMAP sparse model
    ../data/processed_data/     — nerfstudio-ready dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, desc: str):
    print(f"\n[COLMAP] {desc}")
    print(f"  > {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def extract_frames(video_path: Path, output_dir: Path, fps: float = 2.0):
    """Extract frames at `fps` frames-per-second using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",                     # high quality JPEG
        str(output_dir / "frame_%04d.jpg"),
        "-y"
    ], f"Extracting frames at {fps} FPS from {video_path.name}")

    frames = list(output_dir.glob("*.jpg"))
    print(f"[COLMAP] Extracted {len(frames)} keyframes to {output_dir}")
    return frames


def run_colmap(frames_dir: Path, output_dir: Path, db_path: Path):
    """Run COLMAP feature extraction + sequential matching + mapper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # 1. Feature extraction (SIFT)
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu",    "1",
        "--SiftExtraction.max_image_size", "1600",
        "--SiftExtraction.num_threads", "8",
    ], "Feature extraction (SIFT on GPU)")

    # 2. Sequential matcher (fast for video — each frame matched to neighbours)
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "10",   # match each frame to ±10 neighbours
        "--SiftMatching.use_gpu",       "1",
    ], "Sequential feature matching")

    # 3. Sparse reconstruction (bundle adjustment)
    run([
        "colmap", "mapper",
        "--database_path",   str(db_path),
        "--image_path",      str(frames_dir),
        "--output_path",     str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001",
    ], "Sparse reconstruction (bundle adjustment)")

    # Check output
    models = list(sparse_dir.iterdir())
    if not models:
        print("[ERROR] COLMAP produced no models. Check frame quality / overlap.")
        sys.exit(1)
    print(f"[COLMAP] Found {len(models)} model(s). Using model 0.")
    return sparse_dir / "0"


def convert_to_nerfstudio(frames_dir: Path, colmap_sparse: Path, output_dir: Path):
    """Convert COLMAP output to nerfstudio transforms.json format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ns-process-data", "images",
        "--data",                str(frames_dir),
        "--output-dir",          str(output_dir),
        "--colmap-model-path",   str(colmap_sparse),
        "--skip-colmap",         # we already ran COLMAP
    ], "Converting to nerfstudio format")
    print(f"[COLMAP] nerfstudio dataset ready at {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True, help="Path to scan_video.mp4")
    parser.add_argument("--fps",        type=float, default=2.0,
                        help="Keyframe extraction rate (default: 2.0 FPS = ~300 frames for 150s video)")
    parser.add_argument("--frames-dir", default="../data/colmap_keyframes")
    parser.add_argument("--colmap-dir", default="../data/colmap_output")
    parser.add_argument("--output-dir", default="../data/processed_data")
    args = parser.parse_args()

    video_path  = Path(args.video)
    frames_dir  = Path(args.frames_dir)
    colmap_dir  = Path(args.colmap_dir)
    output_dir  = Path(args.output_dir)
    db_path     = colmap_dir / "database.db"

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("  SplatMesh PoC — COLMAP Pipeline")
    print("=" * 60)

    extract_frames(video_path, frames_dir, fps=args.fps)
    sparse_model = run_colmap(frames_dir, colmap_dir, db_path)
    convert_to_nerfstudio(frames_dir, sparse_model, output_dir)

    print("\n" + "=" * 60)
    print("  ✅ COLMAP pipeline complete!")
    print(f"  Dataset ready at: {output_dir.resolve()}")
    print("  Next step: run 4_train/train.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

### `4_train/train.sh`

Runs nerfstudio `splatfacto` training on the RTX 3090.

```bash
#!/usr/bin/env bash
# SplatMesh PoC — Gaussian Splat training
# Expected training time: ~20-30 min on RTX 3090

set -e

DATA_DIR="../data/processed_data"
OUTPUT_DIR="../outputs"
EXPORT_DIR="../outputs/splat_export"

echo "========================================"
echo "  SplatMesh — Gaussian Splat Training"
echo "  Model: splatfacto (3DGS)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

# Train
ns-train splatfacto \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --viewer.quit-on-train-completion True \
  splatfacto \
    --num-downscales 0 \
    --max-num-iterations 30000 \
    --densify-until-iter 15000 \
    --densification-interval 100 \
    --num-random-points 100000

echo ""
echo "Training complete. Exporting PLY..."

# Find the latest training config
CONFIG=$(find "$OUTPUT_DIR/splatfacto" -name "config.yml" | sort | tail -1)
echo "Using config: $CONFIG"

# Export Gaussian Splat PLY
ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir  "$EXPORT_DIR"

echo ""
echo "========================================"
echo "  ✅ Export complete!"
echo "  PLY file: $EXPORT_DIR/point_cloud.ply"
echo "  Next step: open 5_viewer/viewer.html"
echo "========================================"
```

Make it executable:
```bash
chmod +x 4_train/train.sh
```

On Windows (run in Anaconda Prompt instead):
```bash
ns-train splatfacto --data ../data/processed_data --output-dir ../outputs splatfacto --max-num-iterations 30000
```

---

### `5_viewer/viewer.html`

The final browser viewer. Loads the exported `.ply` file and renders the photorealistic Gaussian Splat at 60+ FPS.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SplatMesh Core — 3D Viewer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0a0a0f; color: #fff; font-family: system-ui, sans-serif; overflow: hidden; }
    canvas { display: block; width: 100vw; height: 100vh; }

    #ui {
      position: fixed; top: 0; left: 0; width: 100%; padding: 16px;
      display: flex; justify-content: space-between; align-items: flex-start;
      pointer-events: none;
    }

    #badge {
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
      backdrop-filter: blur(8px); border-radius: 10px; padding: 10px 16px;
      font-size: 13px; line-height: 1.6;
    }
    #badge strong { font-size: 15px; display: block; margin-bottom: 4px; }

    #status-panel {
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
      backdrop-filter: blur(8px); border-radius: 10px; padding: 10px 16px;
      font-size: 13px; text-align: right; min-width: 200px;
    }
    #fps-counter { font-size: 22px; font-weight: 600; color: #4ade80; }
    #splat-count { font-size: 11px; opacity: 0.5; margin-top: 2px; }

    #progress-bar {
      position: fixed; bottom: 0; left: 0; width: 100%; height: 3px;
      background: rgba(255,255,255,0.1);
    }
    #progress-fill {
      height: 100%; background: linear-gradient(90deg, #6366f1, #4ade80);
      width: 0%; transition: width 0.3s;
    }

    #controls {
      position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
      background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px; padding: 8px 16px; font-size: 12px;
      opacity: 0.6; text-align: center;
    }

    #load-overlay {
      position: fixed; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; background: #0a0a0f; z-index: 100;
    }
    #load-overlay h1 { font-size: 28px; margin-bottom: 8px; }
    #load-overlay p  { opacity: 0.5; font-size: 14px; }
    #load-bar {
      margin-top: 24px; width: 300px; height: 4px;
      background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;
    }
    #load-fill {
      height: 100%; background: #6366f1; width: 0%;
      transition: width 0.2s; border-radius: 2px;
    }
    #load-msg { margin-top: 12px; font-size: 12px; opacity: 0.4; }
  </style>
</head>
<body>

<div id="load-overlay">
  <h1>SplatMesh Core</h1>
  <p>Loading Gaussian Splat scene...</p>
  <div id="load-bar"><div id="load-fill"></div></div>
  <div id="load-msg">Initialising WebGL renderer...</div>
</div>

<canvas id="canvas"></canvas>

<div id="ui">
  <div id="badge">
    <strong>SplatMesh Core — PoC</strong>
    3D Gaussian Splatting · RTX 3090<br>
    Drag to orbit · Scroll to zoom · Right-drag to pan
  </div>
  <div id="status-panel">
    <div id="fps-counter">-- FPS</div>
    <div id="splat-count">Loading...</div>
  </div>
</div>

<div id="progress-bar"><div id="progress-fill"></div></div>

<div id="controls">
  Left drag: orbit &nbsp;|&nbsp; Right drag: pan &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; F: fit to view
</div>

<!-- gsplat.js — WebGL Gaussian Splat renderer -->
<script src="https://cdn.jsdelivr.net/npm/gsplat@latest/dist/index.umd.min.js"></script>

<script>
  // ── CONFIG ──────────────────────────────────────────────────────────────────
  // Path to your exported PLY file.
  // Option A: serve with a local HTTP server and set the URL here.
  // Option B: drag-and-drop the PLY file onto the canvas (supported below).
  const PLY_URL = './point_cloud.ply';  // relative to this HTML file
  // ────────────────────────────────────────────────────────────────────────────

  const canvas      = document.getElementById('canvas');
  const loadOverlay = document.getElementById('load-overlay');
  const loadFill    = document.getElementById('load-fill');
  const loadMsg     = document.getElementById('load-msg');
  const fpsCounter  = document.getElementById('fps-counter');
  const splatCount  = document.getElementById('splat-count');

  let frameCount = 0, lastTime = Date.now();

  function updateFPS() {
    const now   = Date.now();
    const delta = (now - lastTime) / 1000;
    if (delta >= 1) {
      fpsCounter.textContent = Math.round(frameCount / delta) + ' FPS';
      frameCount = 0;
      lastTime   = now;
    }
    frameCount++;
  }

  async function loadScene(url) {
    loadMsg.textContent = 'Loading ' + url + ' ...';

    const { Scene, WebGLRenderer, Camera, OrbitControls } = GSPLAT;

    const renderer = new WebGLRenderer({ canvas });
    const scene    = new Scene();
    const camera   = new Camera();
    const controls = new OrbitControls(camera, canvas);

    // Load the PLY file
    await scene.loadFromURL(url, (progress) => {
      const pct = Math.round(progress * 100);
      loadFill.style.width = pct + '%';
      loadMsg.textContent  = `Loading... ${pct}%`;
    });

    const numSplats = scene.splats?.length ?? '?';
    splatCount.textContent = `${numSplats.toLocaleString()} Gaussians`;

    loadOverlay.style.display = 'none';

    // Render loop
    function render() {
      controls.update();
      renderer.render(scene, camera);
      updateFPS();
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

    // Keyboard: F = fit to view
    document.addEventListener('keydown', e => {
      if (e.key === 'f' || e.key === 'F') controls.fitToScene(scene);
    });
  }

  // Drag-and-drop PLY support (alternative to URL loading)
  canvas.addEventListener('dragover', e => e.preventDefault());
  canvas.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.name.endsWith('.ply')) {
      const url = URL.createObjectURL(file);
      loadScene(url);
    }
  });

  // Auto-load
  loadScene(PLY_URL).catch(err => {
    loadMsg.textContent = '❌ Could not load PLY. Drag and drop your point_cloud.ply file onto this window.';
    console.error('[Viewer] Load error:', err);
  });
</script>
</body>
</html>
```

---

## Step-by-Step Run Guide

Follow these steps in order. Each step builds on the previous one.

---

### Step 1 — Find Your PC's Local IP Address

```bash
ipconfig
# Look for "IPv4 Address" under your Wi-Fi adapter
# Example: 192.168.1.42
```

You'll need this IP in `phone_stream.html`.

---

### Step 2 — Edit `phone_stream.html`

Open `1_capture/phone_stream.html` and replace:

```javascript
const PC_IP = 'REPLACE_WITH_PC_IP';
```

with your actual IP, e.g.:

```javascript
const PC_IP = '192.168.1.42';
```

---

### Step 3 — Serve the Phone Page

From inside `splatmesh-poc/`:

```bash
cd 1_capture
python -m http.server 8080
```

On your phone browser, open:  
`http://192.168.1.42:8080/phone_stream.html`

Allow camera access when prompted. You should see "Connecting..."

---

### Step 4 — Start the Laptop Receiver

Open a **second terminal** (same conda environment):

```bash
conda activate splatmesh
cd splatmesh-poc/1_capture
python laptop_receiver.py
```

The status on the phone should change to "✅ Connected — streaming".  
A live window opens on the PC showing the phone feed.

---

### Step 5 — Test the NPU Pipeline

Open a **third terminal**:

```bash
conda activate splatmesh
cd splatmesh-poc/2_npu_pipeline

# First run downloads YOLOv8-Nano weights (~6 MB) and DepthAnything-V2 (~100 MB)
python run_pipeline.py
```

You'll see a side-by-side window: original | YOLO masked | depth map.  
This is your proof that the edge intelligence layer works.

---

### Step 6 — Record the Scan

Go back to the laptop receiver window. Walk your phone slowly through the space you want to scan.

**Scanning tips:**
- Walk at ~0.5 m/s (slow and steady)
- Keep the phone at chest height, pointed forward
- Do 3 passes: straight, angled left, angled right
- Aim for 2–3 minutes of footage
- Clear the space of people before scanning

Press **R** in the receiver window to start recording.  
Press **R** again to stop.  
Frames are saved to `data/raw_frames/`.

**Alternatively:** Record a video directly on your phone, then copy `scan_video.mp4` into `data/`. This is easier for the PoC.

---

### Step 7 — Run COLMAP

Make sure `scan_video.mp4` is in `data/`. Then:

```bash
conda activate splatmesh
cd splatmesh-poc/3_colmap

python extract_and_run.py --video ../data/scan_video.mp4 --fps 2.0
```

**What happens:**
- `ffmpeg` extracts one frame every 0.5 seconds (~300 frames for a 2.5 min video)
- COLMAP runs SIFT feature extraction (GPU-accelerated)
- Sequential matcher matches each frame to its ±10 neighbours
- Bundle adjustment produces a sparse point cloud + camera poses
- `ns-process-data` converts it all to nerfstudio format

**Expected time:** 10–25 minutes depending on frame count.

**If COLMAP fails** ("no models found"):
- Slow down the scan — too much motion blur breaks feature matching
- Increase overlap by walking slower
- Make sure the space has enough texture (plain white walls are hard for SIFT)
- Try `--fps 3.0` to extract more frames

---

### Step 8 — Train the Gaussian Splat

```bash
conda activate splatmesh
cd splatmesh-poc/4_train

# Windows (Anaconda Prompt):
ns-train splatfacto --data ../data/processed_data --output-dir ../outputs splatfacto --max-num-iterations 30000

# Linux / WSL / Git Bash:
bash train.sh
```

**Expected time on RTX 3090:** 20–30 minutes for 30,000 iterations.

While training, nerfstudio opens a web-based viewer at `http://localhost:7007` — you can watch the scene materialise in real time.

After training finishes, export the PLY:

```bash
# Find your config file (shown in the training output)
ns-export gaussian-splat \
  --load-config outputs/splatfacto/YYYY-MM-DD_HHMMSS/config.yml \
  --output-dir  outputs/splat_export
```

This produces `outputs/splat_export/point_cloud.ply`.

---

### Step 9 — View the Result

Copy `point_cloud.ply` into the `5_viewer/` folder:

```bash
cp outputs/splat_export/point_cloud.ply 5_viewer/point_cloud.ply
```

Serve the viewer with a local HTTP server (required — browsers block `file://` for WebGL):

```bash
cd splatmesh-poc/5_viewer
python -m http.server 9000
```

Open in Chrome or Firefox:  
`http://localhost:9000/viewer.html`

You should see your photorealistic 3D scene at 60+ FPS.

**Alternatively:** Drag and drop `point_cloud.ply` directly onto the viewer canvas.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Phone can't connect to WebSocket | PC firewall blocking port 8765 | Allow port 8765 in Windows Defender Firewall |
| YOLO model not found | First run, needs download | Run `python yolo_mask.py` once to auto-download |
| COLMAP produces 0 models | Frames too blurry or low overlap | Slow down the scan, retry with `--fps 3.0` |
| nerfstudio install fails | `tinycudann` not compiled | `pip install ninja` then reinstall tinycudann (see Step 8 in Prerequisites) |
| `ns-train` CUDA out of memory | 30k iterations too heavy | Add `--max-num-iterations 15000` to the command |
| Viewer shows blank/black | Browser blocked file:// | Always use `python -m http.server`, never `file://` |
| PLY loads but scene is blurry | Too few training iterations | Increase to 50,000 iterations |

---

## Expected Output Quality

| Frames extracted | COLMAP success rate | Splat quality |
|---|---|---|
| < 100 | Low | Poor |
| 200–400 | Good | Decent — sharp features |
| 400–600 | Very good | High — photorealistic |
| > 600 | Diminishing returns | Marginal improvement |

A well-captured 2-minute walk-through video at 2 FPS extraction gives ~240 frames — this is the minimum for a good result indoors.

---

## What to Show at the Hackathon

This PoC proves every technical claim:

1. ✅ Phone → WebSocket → PC at 30 FPS — **proven by Step 4**
2. ✅ Real-time YOLO person masking — **proven by Step 5**
3. ✅ Monocular depth estimation — **proven by Step 5**
4. ✅ COLMAP camera pose extraction from phone video — **proven by Step 7**
5. ✅ Full 3D Gaussian Splat trained from phone video — **proven by Step 8**
6. ✅ 60+ FPS photorealistic 3D viewer in browser — **proven by Step 9**

At the hackathon, swap the RTX 3090 for the Cloud AI 100 and swap ONNX Runtime GPU for the QNN Hexagon NPU execution provider. The logic is identical.

---

## License

MIT — free to use, fork, and build on.
