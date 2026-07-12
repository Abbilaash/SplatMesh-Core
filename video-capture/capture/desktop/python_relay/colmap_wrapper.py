#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path

def get_colmap_exe():
    bat_path = Path("C:/colmap/COLMAP.bat")
    if bat_path.exists(): return bat_path
    return shutil.which("colmap")

def run_command(cmd, log_file):
    # Log the command being run
    log_file.write(f"\n{'='*60}\nRunning: {' '.join(str(c) for c in cmd)}\n{'='*60}\n")
    log_file.flush()
    # Run the process
    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0

def run_colmap_reconstruction(session_dir: Path) -> bool:
    colmap_exe = get_colmap_exe()
    if not colmap_exe: 
        print("COLMAP executable not found.")
        return False

    images_dir = session_dir / "images"
    db_path = session_dir / "database.db"
    sparse_dir = session_dir / "sparse"
    log_path = session_dir / "colmap_reconstruction.log"

    # Reset workspace
    if db_path.exists(): db_path.unlink()
    if sparse_dir.exists(): shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        # 1. Feature Extraction: Forced CPU mode
        cmd_extract = [
            str(colmap_exe), "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "0"
        ]
        if not run_command(cmd_extract, log_file): return False

        # 2. Sequential Matching: Forced CPU mode
        cmd_match = [
            str(colmap_exe), "sequential_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0"
        ]
        if not run_command(cmd_match, log_file): return False

        # 3. Mapping: Incremental CPU mode
        cmd_map = [
            str(colmap_exe), "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        if not run_command(cmd_map, log_file): return False

        # 4. Conversion: Output to PLY
        subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not subdirs: return False
        
        model_dir = subdirs[0]
        ply_path = model_dir / "model.ply"
        cmd_convert = [
            str(colmap_exe), "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(ply_path),
            "--output_type", "PLY"
        ]
        run_command(cmd_convert, log_file)
        
    return True