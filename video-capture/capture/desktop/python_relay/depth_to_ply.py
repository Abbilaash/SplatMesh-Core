import os
import glob
import numpy as np
import open3d as o3d
import cv2

def process_all_depths(npy_dir, img_dir, out_dir, downsample_factor=4):
    os.makedirs(out_dir, exist_ok=True)
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    
    print(f"Found {len(npy_files)} depth maps. Starting conversion...")
    
    for npy_path in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # Match the depth map to the PNG image
        img_path = os.path.join(img_dir, f"{base_name}.png")
        if not os.path.exists(img_path):
            print(f"Skipping {base_name}: Matching image not found.")
            continue
            
        # Load arrays
        depth = np.load(npy_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # DOWNSAMPLE: This is critical to prevent RAM crashes
        depth = depth[::downsample_factor, ::downsample_factor]
        img = img[::downsample_factor, ::downsample_factor]
        
        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Flatten and stack. Multiplying depth by 100 scales it up slightly for the viewport
        points = np.stack((x.flatten(), y.flatten(), -depth.flatten() * 100), axis=-1)
        colors = img.reshape(-1, 3) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        out_path = os.path.join(out_dir, f"{base_name}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"Saved: {out_path}")

# Run the batch processor
process_all_depths(
    npy_dir=r"E:\depth_raw",
    img_dir=r"E:\qhack\SplatMesh-Core\test\images",
    out_dir=r"E:\depth_plys",
    downsample_factor=4  # Takes every 4th pixel. Increase to 8 if the files are still too heavy.
)