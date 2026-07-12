#!/usr/bin/env python3
import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {video_path}. Check the path.")
        return

    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames to: {output_dir} ...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break  # End of video
            
        frame_count += 1
        
        # Format the filename with leading zeros (e.g., frame_0001.png)
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        
        # Save the frame losslessly as a PNG
        cv2.imwrite(filename, frame)
        saved_count += 1
        
        if saved_count % 100 == 0:
            print(f"Extracted {saved_count} frames...")

    cap.release()
    print(f"\n[SUCCESS] Finished extracting {saved_count} total frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rip frames from an MP4 video.")
    parser.add_argument("video_path", help="Path to the input .mp4 file")
    parser.add_argument("output_dir", help="Directory to save the extracted frames")
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir)