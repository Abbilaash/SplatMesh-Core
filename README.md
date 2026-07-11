# SplatMesh-Core — 3D Gaussian Splatting Edge & Cloud Pipeline

SplatMesh-Core is an NPU-accelerated hybrid edge-cloud pipeline designed for real-time 3D Gaussian Splatting and scene narration.

---

## 1. System Components

The repository is organized into three primary subdirectories:

1.  **[flask_app/](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/flask_app)**: The Cloud Orchestrator backend running on the **Qualcomm Cloud AI 100 Linux machine**. It receives telemetry, manages background nerfstudio `splatfacto` training loops, exports progressive checkpoints, runs **Llama 3.1-8B-Instruct** for room descriptions, and serves WebGL-compatible 3D files.
2.  **[frontend/](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/frontend)**: The Web Dashboard running in a **web browser on the Laptop**. It is built with React + Vite + TypeScript + Tailwind CSS and uses `gsplat.js` to render the 3D models progressively in real-time as they train.
3.  **[splatmesh_mobile/](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/splatmesh_mobile)**: The edge Ingestion App running on an **Android or iOS phone**. It captures camera frame streams and built-in accelerometer IMU data, and transmits them over a local WebSocket to the laptop receiver.

---

## 2. Ingestion & Processing Flow

$$\text{Phone} \xrightarrow{\text{Wi-Fi WebSocket}} \text{NPU Laptop} \xrightarrow{\text{WAN HTTP Upload}} \text{Cloud Flask Server} \xleftarrow{\text{HTTP API / JSON Polling}} \text{Web Browser}$$

1.  **Capture**: Flutter mobile app streams locked-focus JPEG frames and device pitch/roll coordinates over WebSocket to the local laptop.
2.  **Edge Processing**: The Snapdragon X Elite laptop runs **YOLOv8** (segmentation masks) and **DepthAnything-V2** (relative depth maps) on the **Hexagon NPU** via QNN execution provider.
3.  **Pose Reconstruction**: The laptop runs **COLMAP** to compute camera path matrices and outputs the nerfstudio-ready `transforms.json` mapping.
4.  **Cloud Training**: Laptop zips the dataset, uploads it to the Cloud AI 100 Flask server, and triggers the `nerfstudio` training loop.
5.  **Interactive Rendering**: The laptop's web browser polls training logs and progress metrics, progressively downloading intermediate checkpoint models for live rendering. Llama 3.1 describes the space when finished.

---

## 3. Getting Started & Documentation

For detailed installation and runtime scripts, refer to:

*   **[Deployment Guide (readme3.md)](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/readme3.md)**: Full step-by-step production run ordering for hardware setups.
*   **[Implementation Plan (implementation_plan.md)](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/implementation_plan.md)**: Architectural and design specification parameters.
*   **[Flask API Docs (flask_app/README.md)](file:///e:/Repositories/Qualcomm-SplatMesh/SplatMesh-Core/flask_app/README.md)**: Request and response schema documentation for server routes.