#!/usr/bin/env python3
"""Extract keyframes from a video using blur and motion filtering."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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
        description="Extract keyframes from a video using FFmpeg frame sampling."
    )
    parser.add_argument("--video-path", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where selected keyframes will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Sampling rate used by FFmpeg before keyframe filtering.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum Laplacian variance required to keep a frame.",
    )
    parser.add_argument(
        "--keyframe-similarity-threshold",
        type=float,
        default=0.92,
        help="Maximum similarity to the previous accepted keyframe before skipping a frame.",
    )
    return parser.parse_args()


def run_ffmpeg_frame_extraction(video_path: Path, frames_dir: Path, fps: float) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame_%06d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "vfr",
        str(output_pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg frame extraction failed:\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{completed.stderr.strip()}"
        )
    return sorted(frames_dir.glob("frame_*.png"))


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
    frame_paths: Iterable[Path],
    blur_threshold: float,
    keyframe_similarity_threshold: float,
) -> list[FrameMetrics]:
    selected_frames: list[FrameMetrics] = []
    previous_selected_image: np.ndarray | None = None

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        blur_variance = laplacian_variance(image)
        if blur_variance < blur_threshold:
            selected_frames.append(
                FrameMetrics(frame_path, blur_variance, 0.0, 0.0, False)
            )
            continue

        if previous_selected_image is None:
            selected_frames.append(
                FrameMetrics(frame_path, blur_variance, 1.0, 0.0, True)
            )
            previous_selected_image = image
            continue

        similarity = frame_similarity(previous_selected_image, image)
        motion_score = 1.0 - similarity
        selected = similarity <= keyframe_similarity_threshold

        selected_frames.append(
            FrameMetrics(frame_path, blur_variance, motion_score, similarity, selected)
        )

        if selected:
            previous_selected_image = image

    return selected_frames


def copy_selected_frames(metrics: Iterable[FrameMetrics], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    selected_index = 1

    for metric in metrics:
        if not metric.selected:
            continue

        destination = output_dir / f"keyframe_{selected_index:04d}{metric.path.suffix}"
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
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1

    if not shutil.which("ffmpeg"):
        print("FFmpeg is not available on PATH.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="extract_keyframes_") as temp_dir:
        frames_dir = Path(temp_dir)
        frame_paths = run_ffmpeg_frame_extraction(video_path, frames_dir, args.fps)
        if not frame_paths:
            print("No frames were extracted from the input video.", file=sys.stderr)
            return 1

        metrics = select_keyframes(
            frame_paths,
            blur_threshold=args.blur_threshold,
            keyframe_similarity_threshold=args.keyframe_similarity_threshold,
        )
        selected_files = copy_selected_frames(metrics, output_dir)
        write_manifest(output_dir, metrics, selected_files)

    print(
        f"Extracted {len(selected_files)} keyframes to {output_dir} "
        f"from {len(metrics)} sampled frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())