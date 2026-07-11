# SplatMesh Infrastructure & Dashboard Plan (High-Detail Version)

This document outlines the detailed implementation plan for setting up the backend Flask server (deployed on the Qualcomm Cloud AI 100 Linux machine) and the frontend dashboard app (React + Vite + TypeScript + Tailwind CSS + gsplat.js).

---

### 1. System Architecture & Component Mapping

```
+---------------------------------------------------------------------------------------------------+
|                                     EDGE INGESTION & PIPELINE (Laptop)                             |
|                                                                                                   |
|  [Flutter Mobile App] --(WiFi / WebSocket:8765)--> [laptop_receiver.py] (Saves RAW images)        |
|  (Camera + Sensor IMU)                                   |                                        |
|                                                          |                                        |
|  [run_pipeline.py] <-------------------------------------+                                        |
|        |                                                                                          |
|        +--> [yolo_mask.py] (Hexagon NPU: Person Segmentation)                                     |
|        +--> [depth_inference.py] (Hexagon NPU: Relative Depth Map)                                |
|        +--> (Optional) [imu_reader.py] (Arduino Serial Interface: Alternative Pitch/Roll)         |
|                                                                                                   |
|  [extract_and_run.py] (COLMAP SfM: Camera Poses -> processed_data/transforms.json)                |
+--------------------------------------------------+------------------------------------------------+
                                                   |
                                                   | Upload (rsync / zip-upload)
                                                   v
+---------------------------------------------------------------------------------------------------+
|                                     CLOUD ORCHESTRATION (Cloud AI 100)                             |
|                                                                                                   |
|  [Flask Server (cloud_server.py)] (Port 5000)                                                     |
|        |                                                                                          |
|        +--> (Triggers) --> [nerfstudio splatfacto] (Background Training loop)                     |
|        |                          |                                                               |
|        |                          +--> writes logs to outputs/train_log.txt                       |
|        |                          +--> writes checkpoints to outputs/splatfacto/.../*.ckpt        |
|        |                                                                                          |
|        +--> (Auto-export) -> [ns-export] -> Generates outputs/splat_export/point_cloud.ply        |
|        |                                                                                          |
|        +--> (Describe) ----> [Llama 3.1-8B-Instruct Pipeline] (NPU / GPU Inference)              |
+--------------------------------------------------+------------------------------------------------+
                                                   |
                                                   | CORS Enabled API Traffic (Port 5000)
                                                   v
+---------------------------------------------------------------------------------------------------+
|                                     FRONTEND VISUALIZATION (Vite React Client)                    |
|                                                                                                   |
|  [Vite Dev Server / Static Hosting]                                                               |
|        |                                                                                          |
|        +--> [App.tsx] (Main Grid Controller & Global Connection State)                            |
|              |                                                                                    |
|              +--> [SplatViewer.tsx] (Binds WebGL canvas, stream-loads point_cloud.ply via gsplat)  |
|              +--> [TelemetryHUD.tsx] (Displays iteration stats, FPS rates, loss curves)           |
|              +--> [LogTerminal.tsx] (Monitors step outputs in scrolling monospace container)      |
|              +--> [SceneDescription.tsx] (Glassmorphic panel displaying Llama scene reports)      |
+--------------------------------------------------+------------------------------------------------+
```

---

## 2. API Routes & Endpoint Schema Details

All endpoints must be served by the Flask app running on port `5000` (or configured via environment arguments) and support CORS headers allowing external frontend connections.

### 2.1. GET `/`
- **Purpose:** Base discovery check.
- **Response Schema (`application/json`):**
  ```json
  {
    "status": "online",
    "service": "SplatMesh Core Cloud Orchestrator",
    "version": "1.0.0",
    "endpoints": [
      "/health",
      "/api/status",
      "/api/logs",
      "/api/splat.ply",
      "/api/describe"
    ]
  }
  ```

### 2.2. GET `/health`
- **Purpose:** Diagnoses server subsystem file-presence and execution paths.
- **Response Schema (`application/json`):**
  ```json
  {
    "status": "healthy",
    "cuda_available": true,
    "llama_loaded": true,
    "directories": {
      "processed_data": "exists",
      "outputs": "exists",
      "splat_export": "exists"
    },
    "point_cloud_exists": true
  }
  ```

### 2.3. POST `/api/upload`
- **Purpose:** Ingests the zip file payload containing the COLMAP output frames and the `transforms.json` file.
- **Request Parameters:**
  - `multipart/form-data` with key `dataset` holding the binary `.zip` package.
- **Processing Logic:** 
  1. Receives file and saves to temporary directory.
  2. Extracts contents into `data/processed_data/` (clears existing contents first).
  3. Validates that `transforms.json` exists in root of target directory.
- **Response Schema (`application/json`):**
  - *Success (200):*
    ```json
    {
      "status": "success",
      "message": "Dataset extracted successfully",
      "files_extracted": 142
    }
    ```
  - *Failure (400/500):*
    ```json
    {
      "status": "error",
      "message": "Invalid zip format or missing transforms.json inside package"
    }
    ```

### 2.4. POST `/api/train/start`
- **Purpose:** Launches the nerfstudio training run in the background.
- **Processing Logic:**
  1. Checks if training is already running by reading `outputs/train.pid` and validating the process.
  2. Spawns `ns-train splatfacto --data /home/ubuntu/splatmesh-core/data/processed_data --output-dir /home/ubuntu/splatmesh-core/outputs --max-num-iterations 30000` via `subprocess.Popen`.
  3. Redirects stdout/stderr to `outputs/train_log.txt`.
  4. Writes the spawned subprocess PID to `outputs/train.pid`.
- **Response Schema (`application/json`):**
  ```json
  {
    "status": "started",
    "pid": 28491,
    "log_file": "outputs/train_log.txt"
  }
  ```

### 2.5. POST `/api/train/stop`
- **Purpose:** Safely kills the active training run.
- **Processing Logic:**
  1. Reads `outputs/train.pid`.
  2. Kills the PID using `os.kill(pid, signal.SIGTERM)`. If the process resists, falls back to `SIGKILL`.
  3. Deletes `outputs/train.pid`.
- **Response Schema (`application/json`):**
  ```json
  {
    "status": "stopped",
    "message": "Process 28491 terminated successfully"
  }
  ```

### 2.6. GET `/api/status`
- **Purpose:** Fetches current training metrics and progress telemetry.
- **Processing Logic:**
  1. Parses the tail of `outputs/train_log.txt` using regex to find the last occurrence of:
     - `Step (\d+)` to read the iteration.
     - `loss=(\d+\.\d+)` to record training loss.
  2. Checks for checkpoints in `outputs/splatfacto/.../nerfstudio_models/*.ckpt`.
  3. Resolves current progress percentage: `(current_iteration / total_iterations) * 100`.
- **Response Schema (`application/json`):**
  ```json
  {
    "training_done": false,
    "progress": 42.5,
    "iteration": 12750,
    "total": 30000,
    "current_loss": 0.0412,
    "eta_seconds": 920.4,
    "ply_available": false
  }
  ```

### 2.7. GET `/api/logs`
- **Purpose:** Retrieves the tail of the log file for streaming console updates.
- **Query Params:** `lines` (default 50) - number of log lines to return.
- **Response Schema (`application/json`):**
  ```json
  {
    "lines": [
      "[11:15:32] Iteration 12700 | Loss: 0.0423 | Stats...",
      "[11:15:45] Iteration 12750 | Loss: 0.0412 | Stats..."
    ]
  }
  ```

### 2.8. GET `/api/splat.ply`
- **Purpose:** Streams the exported binary 3D Gaussian Splat PLY model.
- **Processing Logic:**
  1. Validates file existence at `outputs/splat_export/point_cloud.ply`.
  2. Returns a Flask `Response` with a chunked file streaming generator.
  3. Includes exact HTTP headers:
     - `Content-Type: application/octet-stream`
     - `Content-Disposition: inline; filename=point_cloud.ply`
     - `Access-Control-Allow-Origin: *` (Allows frontend direct fetch)
     - `Content-Length: <file_size>` (Allows the frontend to render a progressive download percentage bar)

### 2.9. GET `/api/describe`
- **Purpose:** Invokes Llama 3.1-8B-Instruct to describe the scanned scene based on a middle keyframe image.
- **Processing Logic:**
  1. Scans `data/colmap_keyframes/` and selects the middle frame file (e.g. index `total / 2`).
  2. Formulates the prompt with tags:
     `"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nAnalyze this indoor space keyframe image and describe it in 2-3 sentences. Focus on layout, room type, objects, and lighting.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"`
  3. Feeds the prompt to the Hugging Face Pipeline loaded in memory.
  4. Caches the description so subsequent calls within 10 seconds bypass inference to protect GPU/NPU compute cycles.
- **Response Schema (`application/json`):**
  ```json
  {
    "status": "success",
    "text": "The image reveals a spacious modern conference room featuring a central polished wooden table surrounded by high-back ergonomic chairs. A large flat-screen display is mounted on the front grey accent wall, and natural daylight floods the space through floor-to-ceiling windows on the right side.",
    "source_frame": "frame_00125.jpg"
  }
  ```

---

## 3. Frontend Architecture (React + TS + Vite + Tailwind)

Inside `frontend/`, the structure will look as follows:

```text
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── index.css
│   ├── App.tsx
│   ├── types.ts                     ← Shared TS Interfaces
│   └── components/
│       ├── SplatViewer.tsx          ← WebGL renderer with gsplat.js
│       ├── TelemetryHUD.tsx         ← Graphic widgets (loss, FPS, progress)
│       ├── LogTerminal.tsx          ← Monospace scrolling terminal panel
│       └── SceneDescription.tsx     ← Llama description text display
```

### 3.1. Telemetry State & Global Context (`src/types.ts`)
```typescript
export interface ServerStatus {
  training_done: boolean;
  progress: number;
  iteration: number;
  total: number;
  current_loss: number;
  eta_seconds: number;
  ply_available: boolean;
}

export interface SystemConfig {
  backendUrl: string;
  pollIntervalMs: number;
}
```

### 3.2. SplatViewer Component details (`src/components/SplatViewer.tsx`)
- Loads `gsplat` package dynamically or using standard imports.
- Establishes a WebGL scene inside a `useEffect` hook referencing a `<canvas>` element.
- Exposes controls: Left-drag to orbit, right-drag to pan, mouse-wheel to zoom.
- Handles progressive download loading loops:
  ```typescript
  import React, { useEffect, useRef, useState } from 'react';
  // Assume global GSPLAT loaded from script or package
  
  export const SplatViewer: React.FC<{ plyUrl: string | null }> = ({ plyUrl }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [downloadPct, setDownloadPct] = useState(0);
    const [renderingStats, setRenderingStats] = useState({ fps: 0, splats: 0 });

    useEffect(() => {
      if (!canvasRef.current || !plyUrl) return;
      
      // Initialize GSPLAT Scene, Camera, WebGLRenderer, and OrbitControls
      // Load PLY from URL using callbacks to setDownloadPct
      // Setup requestAnimationFrame loop to render scene and calculate FPS
      
      return () => {
        // Cleanup WebGL context, dispose buffers, and remove event listeners
      };
    }, [plyUrl]);

    return (
      <div className="relative w-full h-full rounded-2xl overflow-hidden border border-white/10 bg-slate-950">
        <canvas ref={canvasRef} className="w-full h-full block" />
        {downloadPct < 100 && plyUrl && (
          <div className="absolute inset-0 flex flex-col justify-center items-center bg-slate-950/80 z-20">
            <span className="text-sm font-semibold text-indigo-400 mb-2">Streaming 3D Gaussians...</span>
            <div className="w-64 h-2 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-150" style={{ width: `${downloadPct}%` }} />
            </div>
            <span className="text-xs text-white/50 mt-1">{downloadPct}%</span>
          </div>
        )}
      </div>
    );
  };
  ```

---

## 4. Flutter Mobile App (Frontend Ingestion)

Instead of the static HTML `phone_stream.html` page, we will build a native Flutter mobile app (`splatmesh_mobile`) for Android and iOS. This app connects to the Snapdragon X Elite Laptop receiver to stream video frames and device-native sensor IMU data.

### 4.1. Key Benefits
1. **Hardware Camera Controls**: Access advanced API features (focus lock, manual exposure, auto-white-balance lock) to reduce motion blur and exposure variations, which are critical for high-quality COLMAP reconstruction.
2. **Built-in IMU Streaming**: Uses the phone's native Gyroscope and Accelerometer sensors via the `sensors_plus` package, calculating pitch and roll in real-time. This transmits IMU data directly over the WebSocket connection, eliminating the external Arduino MPU6050 hardware dependency.
3. **Robust Connection Management**: Auto-reconnect, custom frame-rate throttling, and visual HUD for streaming FPS, latency, and recording status.

### 4.2. Project Structure & Dependency Configurations
In `splatmesh_mobile/pubspec.yaml`:
```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.10.5+9
  web_socket_channel: ^2.4.1
  sensors_plus: ^5.0.1
  path_provider: ^2.1.1
  image: ^4.1.3
```

### 4.3. High-Level Ingestion Flow
```
[Flutter Mobile App]
  |-- (Start Camera stream at 30 FPS, lock exposure/focus)
  |-- (Subscribe to gyroscope/accelerometer streams)
  |-- (Compress frame as JPEG at quality 0.8)
  |
  +-- Websocket Binary Message ----> [laptop_receiver.py] (Saves RAW images)
  +-- Websocket Text (JSON) --------> [laptop_receiver.py] (Parses pitch/roll)
```

### 4.4. Core Streaming Service (`lib/streaming_service.dart`)
```dart
import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:sensors_plus/sensors_plus.dart';

class StreamingService {
  WebSocketChannel? _channel;
  bool _isStreaming = false;
  StreamSubscription? _imuSubscription;
  double _pitch = 0.0;
  double _roll = 0.0;

  void connect(String url) {
    _channel = WebSocketChannel.connect(Uri.parse(url));
    _startIMUTracking();
  }

  void _startIMUTracking() {
    _imuSubscription = accelerometerEvents.listen((AccelerometerEvent event) {
      // Basic pitch and roll calculation from gravity vector
      _pitch = event.y;
      _roll = event.x;
      
      // Stream IMU data as a JSON text packet over WebSocket
      if (_isStreaming && _channel != null) {
        _channel!.sink.add(jsonEncode({
          "type": "imu",
          "pitch": _pitch,
          "roll": _roll,
          "timestamp": DateTime.now().millisecondsSinceEpoch,
        }));
      }
    });
  }

  void sendFrame(List<int> jpegBytes) {
    if (_isStreaming && _channel != null) {
      // Send binary frame bytes
      _channel!.sink.add(jpegBytes);
    }
  }

  void startStreaming() { _isStreaming = true; }
  void stopStreaming() { _isStreaming = false; }

  void dispose() {
    _imuSubscription?.cancel();
    _channel?.sink.close();
  }
}
```

---

## 5. 3D Splat Streaming Strategies to Laptop

Serving raw Gaussian Splat models (`.ply` format) is resource-intensive due to large file sizes (typically 200MB - 1.5GB). Below are four strategies to stream 3D Splatting data efficiently to a client laptop:

### Option A: Format Conversion & Weight Quantization (SPLAT / KSPLAT)
- **Concept:** Convert the exported PLY file to Luma `.splat` or `.ksplat` format.
- **Why it works:** Raw PLY stores covariance matrices, scales, rotation quaternions, color coefficients (spherical harmonics), and opacities as full 32-bit floats. By converting to half-floats (16-bit) and quantizing color/rotation parameters to 8-bit integers, the file size shrinks by **10x to 15x** (e.g., from 300MB to ~20MB) with zero perceptual degradation.
- **Implementation:**
  1. Add a post-export step to the cloud orchestrator to convert `point_cloud.ply` to `point_cloud.splat` using a lightweight Python script or Node package.
  2. Serve the `.splat` file via the endpoint `/api/splat.splat`.
  3. Load and render directly using native `.splat` support in `gsplat.js`.

### Option B: Remote Headless WebGL / WebGPU Rendering (WebRTC Streaming)
- **Concept:** Keep the 3D model on the high-performance Cloud AI 100/GPU server. The server runs a headless renderer and streams the viewport as a H.264/VP9/AV1 video feed using WebRTC.
- **Why it works:** The laptop does not download or render any 3D data. This results in **sub-second initial load times** regardless of model size, and ensures high frame rates (60 FPS) on low-spec client laptops.
- **Implementation:**
  1. Spawns a headless Python rasterizer (e.g., `diff-gaussian-rasterization` or a headless PyTorch window) on the server.
  2. Uses `aiortc` (Python WebRTC library) to set up a media stream channel.
  3. The laptop React client captures user mouse/drag interactions on a dummy canvas and sends rotation/pan/zoom delta packets to the server via WebRTC Data Channels.
  4. The server updates the camera matrix, renders the frame, and returns it as a low-latency video track.

### Option C: Level of Detail (LoD) Octree-based Streaming
- **Concept:** Partition the Gaussian Splat point cloud into a spatial Octree structure on the server.
- **Why it works:** Instead of downloading the full point cloud at once, the client fetches the 3D scene hierarchically.
- **Implementation:**
  1. The server pre-processes the PLY file into an octree folder structure containing spatial chunks (e.g., root node, child nodes).
  2. The frontend client downloads only the low-resolution coarse splats (top level of the octree) to render the initial scene.
  3. As the user zooms in, the client sends bounding-box queries to the server to fetch high-resolution detail chunks for the visible viewport.

### Option D: Opacity-Sorted Progressive HTTP Range Requests
- **Concept:** Order the Gaussians in the PLY file so that the most visually significant ones (largest scales, highest opacities, and closest to the scene center) are saved first.
- **Why it works:** Allows progressive streaming using standard HTTP range requests.
- **Implementation:**
  1. Sort the Gaussian array by a weight metric: $W = \text{opacity} \times \text{scale}$.
  2. The frontend client sends an initial HTTP request with a `Range: bytes=0-10485760` header (fetching the first 10MB).
  3. The client parses and renders these vital 10MB of splats immediately, showing a recognizable 3D model, while continuing to download the remaining chunks in the background.

---

## 6. Implementation Steps

```mermaid
grid
  title Implementation Steps Timeline
```

1. **Phase 1: Backend Server Deployment**
   - Create directories under `splatmesh-core/5_cloud_server/`.
   - Implement `cloud_server.py` with multi-threaded loops for log parsing and checkpoint monitoring.
   - Configure Python dependencies (Flask, Flask-CORS, transformers).
   - Verify server endpoints manually using local mock files.
2. **Phase 2: Mobile Flutter Ingestion App Setup**
   - Initialize the Flutter project `flutter create splatmesh_mobile`.
   - Implement camera controls (focus lock, manual exposure) and the `web_socket_channel` interface.
   - Integrate built-in IMU streaming using the device gyroscope/accelerometer sensors via `sensors_plus`.
3. **Phase 3: Frontend Setup**
   - Initialize Vite template: `npm create vite@latest frontend -- --template react-ts`.
   - Install Tailwind CSS and verify postcss configurations.
   - Add dashboard design patterns (glassmorphism CSS utility classes, custom scrollbars).
4. **Phase 4: Component Assembly**
   - Implement telemetry hook polling backend configurations.
   - Build custom Canvas component binding the `gsplat.js` library logic.
   - Implement scrolling logs window and active loss graph elements.
5. **Phase 5: End-to-End Testing**
   - Mount actual datasets, run training loop, and verify UI telemetry reacts correctly.
   - Run Llama scene analysis and render the points.

---

## 7. User Review Required

> [!IMPORTANT]
> **NPU Inference vs. GPU Inference for Llama 3.1-8B:**
> Using Llama 3.1-8B-Instruct via the ONNX Runtime execution provider optimized for Cloud AI 100 is highly recommended. However, to ensure rapid deployment during the hackathon, we will configure a fallback path using `transformers` to load via CUDA GPU in PyTorch if the NPU QAIRT model package is missing.

---

## 8. Open Questions

> [!WARNING]
> Please confirm if you would like us to proceed with:
> 1. Setting up mock JSON server parameters inside the Flask app so you can test the frontend dashboard without having a running nerfstudio training process active.
> 2. Installing tailwindcss version 3 or version 4. (We recommend version 3 for simple configuration).

---

## 9. Proposed Changes

### [NEW] [splatmesh_mobile](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/splatmesh_mobile)
Flutter application code for Android/iOS frame capture and IMU data streaming.

### [NEW] [cloud_server.py](file:///C:/Users/Darshan%20V%20G/.gemini/antigravity-ide/brain/fe86e064-4dad-4cb7-bfe3-947e7600c2c4/scratch/cloud_server.py)
A complete Flask app code mockup designed to run on the Cloud AI 100 host machine.

### [NEW] [App.tsx](file:///C:/Users/Darshan%20V%20G/.gemini/antigravity-ide/brain/fe86e064-4dad-4cb7-bfe3-947e7600c2c4/scratch/App.tsx)
Vite React core component structure coordinating the UI dashboard panels.

---

## 10. Verification Plan

### Automated Tests
- Syntax compile verification:
  `python -m py_compile 5_cloud_server/cloud_server.py`
- TypeScript bundler validation:
  `cd frontend && npm run build`
- Flutter syntax/build validation:
  `cd splatmesh_mobile && flutter build apk --debug`

### Manual Verification
1. Run server: `python cloud_server.py` and query `http://localhost:5000/status` to ensure JSON fields load correctly.
2. Verify that mock logs tail properly updates when appending mock lines to `train_log.txt`.
3. Open dashboard in Google Chrome, select orbit controls, and ensure there are no WebGL context loop errors.
4. Launch the Flutter mobile app, input the laptop's IP, check that camera frames are successfully received on the laptop UI, and verify pitch/roll angles are updating in the overlay console.
