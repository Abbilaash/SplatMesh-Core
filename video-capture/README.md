# SplatMesh Capture and Ingestion Pipeline

This directory contains the real-time video streaming and sensor telemetry capture system for the SplatMesh pipeline. It consists of a mobile web client and a desktop Python relay server that ingests frames and automates Structure-from-Motion (SfM) pose estimation.

---

## Architecture Overview

```
[Mobile Phone Client] 
   - Captures 1080p Video + Gyroscope (Pitch, Roll, Yaw)
   - Resizes frames dynamically based on server acceleration
   - Streams packaged binary (32-byte header + JPEG) over WebSockets
          │
          ▼
[Python Ingestion Server (Port 3000)]
   - Hosts static mobile HTML pages
   - Auto-detects hardware acceleration (NPU, CUDA, or CPU)
   - Negotiates stream resolution with phone
   - Saves frames asynchronously in images/ subfolders
          │
          ▼ (Upon stopping recording)
[Automated COLMAP Pipeline]
   - Triggers CPU-only SIFT extraction, sequential matching, and mapping
   - Saves final database and sparse model files inside the session directory
```

---

## Setup and Installation

### 1. Prerequisites

Ensure the following packages are installed in your conda environment on the desktop:
```bash
pip install aiohttp websockets opencv-python onnxruntime torch
```

Ensure COLMAP is installed on the laptop. The wrapper will check for `colmap` in the system path, with an automatic fallback to `C:\colmap\bin\colmap.exe`.

### 2. Run the Relay Server

Open PowerShell in the `python_relay` directory (run natively on Windows, do not use WSL):
```powershell
cd capture/desktop/python_relay
python main.py
```

The server will initialize on `http://0.0.0.0:3000`.

### 3. Mobile Browser Configuration

Mobile browsers block camera, orientation sensor APIs, and WebSockets on insecure local IPs by default. To bypass this for development:

1. Open Chrome on your mobile device.
2. Navigate to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
3. Enable the flag and add your laptop's local IP and port (for example: `http://10.91.53.25:3000`).
4. Tap "Relaunch" to restart Chrome.

---

## Ingestion Workflow

### Hardware Auto-Detection & Resolution Negotiation
Upon WebSocket connection, the Python server queries available runtime acceleration backends:
* If **QNN (Hexagon NPU)** or **CUDA (NVIDIA GPU)** is available, it suggests a stream resolution width of **1280px** to maximize pose triangulation accuracy.
* If only **CPU** fallback is available, it suggests a resolution width of **640px** to keep latency low.

The phone client parses this handshake and configures its canvas scale at runtime.

### Automated Mapping
When you tap the Record button to stop, the WebSocket handler signals a STOP event and triggers `run_colmap_reconstruction` on a background execution thread:
1. **SIFT Feature Extraction**: Extracts keypoints on the CPU using SIFT.
2. **Sequential Feature Matching**: Correlates features between sequential frames.
3. **Sparse Mapper**: Calculates camera trajectories and constructs 3D point cloud structures.

---

## Directory Output Structure

The server organizes datasets inside the `capture/desktop/data/` folder, keeping images isolated from databases and logs to avoid feature extraction contamination:

```
capture/desktop/data/session_<timestamp>/
├── images/
│   ├── frame_<timestamp>_000001_P<pitch>_R<roll>_Y<yaw>.jpg
│   └── ...
├── database.db                   (COLMAP SQLite keypoints/matches)
├── colmap_reconstruction.log     (Detailed COLMAP logging)
└── sparse/
    └── 0/
        ├── cameras.bin           (Camera intrinsics)
        ├── images.bin            (Camera poses)
        └── points3D.bin          (3D point coordinates)
```

---

## Troubleshooting

* **Address Already in Use (Errno 10048)**: A python process did not close cleanly and is still listening on port 3000. Run the following command in PowerShell to release the socket:
  ```powershell
  Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force -ErrorAction SilentlyContinue
  ```
* **Reconstruction Fails**: Ensure you rotate the camera slowly during recording, maintain sufficient overlap between frames, and avoid featureless regions (such as blank white walls).
