# SplatMesh-Core — Real-Time Edge Capture & 3D Reconstruction Pipeline

SplatMesh-Core is an edge-optimized ingestion and processing pipeline designed for real-time 3D Gaussian Splatting and Structure-from-Motion (SfM). It dynamically manages network backpressure, performs intelligent edge-culling on the fly, and automates GPU-accelerated sparse point cloud generation.

## Screenshots & Outputs

**Example .ply model files generated from our COLMAP architecture pipelines**

<img src="image.png" width="500" alt="Ply model files" />

**The streaming edge device: OnePlus 15 streaming video to the Python server via LAN**

<img src="image-1.png" width="300" alt="Streaming edge device" />

**Trial with Depth Anything — Quantized using Qualcomm AI Hub**

<img src="image-2.png" width="600" alt="Depth Anything Trial" />

**Single frame of a computer mouse: Original vs. Depth Analyzed Sample**<br>
<table>
  <tr>
    <td align="center"><b>Original Image</b></td>
    <td align="center"><b>Depth Map</b></td>
  </tr>
  <tr>
    <td><img src="frame_1783788351257_000452_P15.3_R-9.5_Y249.6.jpg" width="300" alt="Original Mouse"></td>
    <td><img src="frame_1783788351257_000452_P15.3_R-9.5_Y249.6.png" width="300" alt="Depth Mouse"></td>
  </tr>
</table>

**Spatial field map of a chair from COLMAP wrapping, viewed in Blender**<br>
<img src="image-3.png" width="700" alt="Spatial field map">

---

## 1. System Components

The repository is organized into a streamlined, decoupled architecture to ensure real-time performance without network buffer bloat:

1. **Mobile Web Client (`mobile/phone_stream.html`)**: The edge ingestion interface running in a standard mobile browser. It captures high-resolution video and IMU telemetry (Pitch, Roll, Yaw). It implements application-layer congestion control (via `ws.bufferedAmount`) to gracefully drop frames if Wi-Fi bandwidth degrades, ensuring the WebSocket tunnel never crashes.
2. **Python Edge Orchestrator (`final.py`)**: The centralized backend engine running on the local compute node. It operates in two phases:
* **Phase 1 (I/O-Bound)**: Rapid, asynchronous disk buffering of incoming binary payloads, enforcing TCP backpressure to match network speeds.
* **Phase 2 (CPU-Bound)**: Triggered safely upon session completion or network disconnect. It executes intelligent edge-culling (Laplacian variance filtering) to discard motion blur before handing data to the GPU.


3. **Automated Pipeline (`colmap_wrapper.py`)**: A subprocess wrapper that orchestrates SIFT feature extraction, sequential matching, Bundle Adjustment mapping, and automated `.ply` format conversion for immediate rendering in engines like `gsplat.js` or Blender.

---

## 2. Ingestion & Processing Flow

$$\text{Mobile (HTML5)} \xrightarrow{\text{WebSocket (Congestion Controlled)}} \text{Edge Server (final.py)} \xrightarrow{\text{Laplacian Culling}} \text{GPU COLMAP} \xrightarrow{\text{Export}} \text{model.ply}$$

1. **Capture**: The mobile client streams timestamped JPEG payloads packed with a 32-byte binary IMU header to the local server.
2. **Edge Ingestion**: The server dynamically negotiates stream resolution (e.g., 640px) to optimize packet size and writes data to disk using strictly `await`ed asynchronous loops to prevent memory overflow.
3. **Smart Filtering**: Once the stream terminates (or abruptly drops), the server calculates the Laplacian variance matrix of every frame in RAM, instantly purging blurry images to save compute time.
4. **Pose Reconstruction**: The system seamlessly triggers COLMAP on the GPU to compute camera path matrices, generating the sparse `points3D.bin` and `images.bin`.
5. **Asset Generation**: The sparse binary graph is automatically converted into a standard `.ply` file, ready for Cloud AI 100 training or immediate WebGL visualization.

---

## 3. Setup and Installation

### Prerequisites

Ensure you have **Python 3.9+** and **COLMAP** installed on your host machine. The wrapper will automatically search your system path, with a fallback to `C:\colmap\COLMAP.bat`.

Create a conda or virtual environment, then install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt

```

### Running the Edge Orchestrator

Open a terminal in the project `video-capture\capture\desktop\python_relay` directory and start the unified server:

```bash
python final.py

```

The server will initialize and listen on `[http://0.0.0.0:3000](http://0.0.0.0:3000)`.

### Mobile Browser Configuration

Modern mobile browsers block camera access and DeviceOrientation sensors on insecure local network IPs by default. To bypass this for local deployment:

1. Open Chrome on your mobile device.
2. Navigate to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
3. Enable the flag and add your host machine's local IP and port (e.g., `[http://10.91.56.127:3000](http://10.91.56.127:3000)`).
4. Tap "Relaunch" to restart Chrome.

---

## 4. Directory Output Structure

The orchestrator dynamically isolates datasets inside the `data/` folder based on session timestamps to prevent feature contamination across different scans:

```text
SplatMesh-Core/video-capture/capture/desktop/python_relay
├── final.py
├── colmap_wrapper.py
├── requirements.txt
├── mobile/
│   └── phone_stream.html
└── data/
    └── session_<timestamp>/
        ├── images/
        │   ├── frame_<timestamp>_000001.jpg
        │   └── ... (Only sharp frames remain post-culling)
        ├── database.db                   (COLMAP SQLite feature matches)
        ├── colmap_reconstruction.log     
        └── sparse/
            └── 0/
                ├── cameras.bin           
                ├── images.bin            
                ├── points3D.bin          
                └── model.ply             <-- Final 3D point cloud asset

```

---

## 5. Troubleshooting

* **Address Already in Use (Errno 10048)**: A previous server instance did not close cleanly. Run the following command in PowerShell to release the socket:
```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force -ErrorAction SilentlyContinue

```


* **Reconstruction Fails/Aborts**: Ensure you rotate the camera slowly during recording and maintain overlap. If you are scanning a completely smooth or textureless object, the Laplacian variance filter may delete good frames. Open `final.py` and set `BLUR_THRESHOLD = 0.0` to disable culling.

---
