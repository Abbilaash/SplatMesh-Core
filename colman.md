
# 3D Reconstruction & Splatting Pipeline

This document outlines the workflow for extracting frames, running COLMAP, and verifying point cloud density before training a Gaussian Splat. 

If the resulting point cloud is too sparse (e.g., featureless objects like a mouse), increase the frame count to reduce the baseline distance between camera shots, which helps the SIFT algorithm track micro-textures.

## Step 1: Clean the Workspace
Before extracting a new batch of frames, you must delete the old `images` directory. If you do not delete it, the extraction script will fail or mix the new frames with the old ones.

*Note: You do not need to delete `database.db` or the `sparse` folder. The COLMAP wrapper will automatically wipe and recreate those when executed.*

**Command:**
Delete the folder manually in File Explorer, or run:
```bash
rmdir /s /q E:\qhack\SplatMesh-Core\test\images

```

## Step 2: Extract Keyframes

Run the extraction script to pull a specific number of high-quality, evenly spaced frames from your raw dataset.

* **Input:** `images_raw` (Contains all your original, unfiltered frames)
* **Output:** `images` (The exact subset of frames COLMAP will use)
* **Tuning:** Change `--target-count` depending on the complexity of the object. Start with 120, push to 300+ for featureless objects.

**Command:**

```bash
python extract_keyframes.py --input-dir E:\qhack\SplatMesh-Core\test\images_raw --output-dir E:\qhack\SplatMesh-Core\test\images --target-count 300

```

## Step 3: Run COLMAP (Sparse Reconstruction)

Feed the extracted images into the COLMAP wrapper. This script leverages the GPU to run feature extraction (SIFT) and exhaustive matching without crashing the system memory.

**What happens here:**

1. **Feature Extraction:** Finds anchor points in every image using CUDA.
2. **Sequential Matching:** Tracks those points across the image sequence.
3. **Mapping:** Runs bundle adjustment to calculate 3D coordinates and camera poses.

**Command:**

```bash
python colmap_wrapper.py E:\qhack\SplatMesh-Core\test

```

*Success criteria: The script finishes without error and populates the `E:\qhack\SplatMesh-Core\test\sparse\0` directory with `cameras.bin`, `images.bin`, and `points3D.bin`.*

## Step 4: Convert and Verify (Optional but Recommended)

Gaussian Splatting scripts read the `.bin` files natively, but you cannot visualize them easily. Convert the sparse model to a `.ply` file to inspect it in Blender or MeshLab.

If the point cloud loosely resembles your target object (even just an outline), the mathematical foundation is solid, and you are ready to train the Gaussian Splat.

**Command:**

```bash
"C:\colmap\COLMAP.bat" model_converter --input_path E:\qhack\SplatMesh-Core\test\sparse\0 --output_path E:\qhack\SplatMesh-Core\test\sparse\0\points.ply --output_type PLY

```