#!/usr/bin/env python3
"""Extract keyframes from an image directory using blur and uniform temporal filtering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    path: Path
    blur_variance: float
    motion_score: float
    similarity: float
    selected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a targeted number of high-quality keyframes from a directory for 3D reconstruction."
    )
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Path to the directory containing raw input images (e.g., images_raw)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where selected keyframes will be written (e.g., images).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=120,
        help="Target number of keyframes to extract for COLMAP.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum Laplacian variance required to keep a frame.",
    )
    return parser.parse_args()


def laplacian_variance(image: np.ndarray) -> float:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def frame_similarity(previous: np.ndarray, current: np.ndarray) -> float:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(current_gray, (previous_gray.shape[1], previous_gray.shape[0]))

    difference = cv2.absdiff(previous_gray, current_gray)
    normalized_difference = float(np.mean(difference) / 255.0)
    return max(0.0, 1.0 - normalized_difference)


def select_keyframes(
    frame_paths: list[Path],
    blur_threshold: float,
    target_count: int,
) -> list[FrameMetrics]:
    # Pass 1: Filter out blurry frames completely and record variances
    sharp_candidates: list[Path] = []
    blur_values: dict[Path, float] = {}
    
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        bv = laplacian_variance(image)
        blur_values[path] = bv
        if bv >= blur_threshold:
            sharp_candidates.append(path)

    # Pass 2: Select target_count frames uniformly distributed across sharp frames
    n_sharp = len(sharp_candidates)
    if n_sharp <= target_count:
        selected_paths = set(sharp_candidates)
        if n_sharp < target_count:
            print(f"Warning: Only found {n_sharp} sharp frames, which is less than target {target_count}.", file=sys.stderr)
    else:
        indices = [int(i * n_sharp / target_count) for i in range(target_count)]
        selected_paths = set(sharp_candidates[idx] for idx in indices)

    # Pass 3: Construct the final metrics mapping matched to original file order
    selected_frames: list[FrameMetrics] = []
    previous_selected_image = None
    
    for path in frame_paths:
        if path not in blur_values:
            continue
            
        bv = blur_values[path]
        if path in selected_paths:
            image = cv2.imread(str(path))
            similarity = 0.0
            if previous_selected_image is not None:
                similarity = frame_similarity(previous_selected_image, image)
            motion_score = 1.0 - similarity
            
            selected_frames.append(FrameMetrics(path, bv, motion_score, similarity, True))
            previous_selected_image = image
        else:
            selected_frames.append(FrameMetrics(path, bv, 0.0, 0.0, False))
            
    return selected_frames


def copy_selected_frames(metrics: Iterable[FrameMetrics], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    selected_index = 1

    for metric in metrics:
        if not metric.selected:
            continue

        destination = output_dir / f"frame_{selected_index:04d}{metric.path.suffix}"
        shutil.copy2(metric.path, destination)
        written_files.append(destination)
        selected_index += 1

    return written_files


def write_manifest(output_dir: Path, metrics: list[FrameMetrics], selected_files: list[Path]) -> None:
    manifest = {
        "selected_count": len(selected_files),
        "frames": [
            {
                "source": str(metric.path),
                "blur_variance": metric.blur_variance,
                "motion_score": metric.motion_score,
                "similarity": metric.similarity,
                "selected": metric.selected,
            }
            for metric in metrics
        ],
        "selected_files": [str(path) for path in selected_files],
    }
    (output_dir / "keyframes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Grab all image files and sort them to maintain temporal sequence
    valid_extensions = {".png", ".jpg", ".jpeg"}
    frame_paths = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
    )

    if not frame_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(frame_paths)} images in {input_dir}. Filtering...")

    metrics = select_keyframes(
        frame_paths,
        blur_threshold=args.blur_threshold,
        target_count=args.target_count,
    )
    
    selected_files = copy_selected_frames(metrics, output_dir)
    write_manifest(output_dir, metrics, selected_files)

    print(
        f"Extracted {len(selected_files)} uniform keyframes to {output_dir} "
        f"from {len(frame_paths)} raw candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())