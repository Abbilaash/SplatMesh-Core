#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path

def get_colmap_exe():
    """Find colmap executable. On Windows, we must use COLMAP.bat to load Qt plugins for the GPU."""
    # Always prefer the batch file on Windows to prevent Qt environment errors
    bat_path = Path("C:/colmap/COLMAP.bat")
    if bat_path.exists():
        return bat_path
        
    colmap_path = shutil.which("colmap")
    if not colmap_path:
        fallback_exe = Path("C:/colmap/bin/colmap.exe")
        if fallback_exe.exists():
            # Temporarily add fallback to environment PATH
            os.environ["PATH"] = str(fallback_exe.parent) + os.path.pathsep + os.environ.get("PATH", "")
            colmap_path = shutil.which("colmap")
    return colmap_path

def run_command(cmd, log_file):
    """Run a subprocess command, logging output to a log file."""
    log_file.write(f"\n{'='*60}\nRunning: {' '.join(str(c) for c in cmd)}\n{'='*60}\n")
    log_file.flush()
    
    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        log_file.write(f"\n[ERROR] Command failed with code {result.returncode}\n")
        return False
    return True

def run_colmap_reconstruction(session_dir: Path) -> bool:
    """
    Runs the complete COLMAP sparse reconstruction pipeline on the images in session_dir.
    Outputs database.db and sparse/0/ directory containing the model binaries inside session_dir.
    """
    colmap_exe = get_colmap_exe()
    if not colmap_exe:
        print("[COLMAP WRAPPER] [ERROR] colmap executable not found in PATH or standard fallbacks.")
        return False

    images_dir = session_dir / "images"
    db_path = session_dir / "database.db"
    sparse_dir = session_dir / "sparse"
    log_path = session_dir / "colmap_reconstruction.log"

    print(f"[COLMAP WRAPPER] Starting COLMAP reconstruction for session: {session_dir.name}")
    print(f"[COLMAP WRAPPER] Logging detailed output to: {log_path}")

    # Ensure output directories are clean
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        # 1. Feature Extraction (GPU Enabled)
        cmd_extract = [
            str(colmap_exe), "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1"
        ]
        print("[COLMAP WRAPPER] Running Feature Extractor (GPU)...")
        if not run_command(cmd_extract, log_file):
            print("[COLMAP WRAPPER] [ERROR] Feature extraction failed. Check log for details.")
            return False

        # 2. Sequential Feature Matching (GPU Enabled)
        cmd_match = [
            str(colmap_exe), "sequential_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "1"
        ]
        print("[COLMAP WRAPPER] Running Feature Matcher (GPU)...")
        if not run_command(cmd_match, log_file):
            print("[COLMAP WRAPPER] [ERROR] Feature matching failed. Check log for details.")
            return False

        # 3. Sparse Reconstruction / Mapping
        cmd_map = [
            str(colmap_exe), "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        print("[COLMAP WRAPPER] Running Incremental Mapper...")
        if not run_command(cmd_map, log_file):
            print("[COLMAP WRAPPER] [ERROR] Mapping failed. Check log for details.")
            return False

        # 4. Verify Reconstruction
        subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("[COLMAP WRAPPER] [ERROR] No sparse models generated. Mapper likely failed to initialize or reconstruct.")
            return False

        model_dir = subdirs[0]
        bin_files = ["cameras.bin", "images.bin", "points3D.bin"]
        txt_files = ["cameras.txt", "images.txt", "points3D.txt"]
        
        has_bin = all((model_dir / f).exists() for f in bin_files)
        has_txt = all((model_dir / f).exists() for f in txt_files)

        if not (has_bin or has_txt):
            print("[COLMAP WRAPPER] [ERROR] Sparse model files missing from mapping output.")
            return False

        # 5. Analyze Model quality
        cmd_analyze = [
            str(colmap_exe), "model_analyzer",
            "--path", str(model_dir)
        ]
        run_command(cmd_analyze, log_file)

    print(f"[COLMAP WRAPPER] [SUCCESS] Reconstruction complete! Model stored at: {model_dir}")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python colmap_wrapper.py <session_dir>")
        sys.exit(1)
    
    run_colmap_reconstruction(Path(sys.argv[1]))