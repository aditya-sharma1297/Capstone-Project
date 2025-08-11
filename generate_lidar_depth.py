import os
import numpy as np
import cv2
import glob

# Camera intrinsics (from RELLIS Basler camera intrinsic)
fx = 2813.643275
fy = 2808.326079
cx = 969.285772
cy = 624.049972

# Original image resolution (used for filtering projections)
IMG_W = 1920
IMG_H = 1200

# Camera To LiDAR Extrinsics, rotation stored as a quaternion
q = {
    'w': -0.50507811,
    'x': 0.51206185,
    'y': 0.49024953,
    'z': -0.49228464
}

# Translation vector (in meters)
t = np.array([-0.13165462, 0.03870398, -0.17253834])

# Converting quaternion to rotation matrix
def quat_to_rot_matrix(q):
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

# Computing the rotation matrix
R = quat_to_rot_matrix(q)

# Loading bin file and extracting 3D point cloud
def load_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]

# Projecting 3D LiDAR points into 2D camera image
def project_points(points):
    # Apply rotation and translation to move points from LiDAR frame to camera frame
    points_cam = (R @ points.T).T + t 

    # Keeping only points that are in front of the camera
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    # Applying perspective projection using camera intrinsics
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # Filtering out projected points that fall outside image bounds
    valid_mask = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]

    return u, v, z

# Creating a depth map from projected LiDAR points
def create_depth_map(u, v, z):
    depth_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    # Filling in depth values (for overlapping points, keep the closest one)
    for px, py, pz in zip(u, v, z):
        current_depth = depth_map[py, px]
        if current_depth == 0 or pz < current_depth:
            depth_map[py, px] = pz
    return depth_map

# Saving the depth map as a colorized PNG image
def save_depth_png(depth_map, save_path):
    # Clip and Normalize depth values for visualization
    max_depth = 80.0
    depth_map = np.clip(depth_map, 0, max_depth)
    depth_norm = (depth_map / max_depth * 255).astype(np.uint8)

    # Colorize depth map for easier viewing
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, depth_color)

# Converting all bin files to depth images
def main():
    bin_root = '/Users/adityasharma/Desktop/Fine Tuning Set/lidar_bin'
    depth_root = '/Users/adityasharma/Desktop/Fine Tuning Set/lidar_depth'

    os.makedirs(depth_root, exist_ok=True)

    scene_folders = [d for d in os.listdir(bin_root) if not d.startswith('.')]
    for scene in scene_folders:
        print(f"Processing scene: {scene}")
        scene_bin_dir = os.path.join(bin_root, scene)
        scene_depth_dir = os.path.join(depth_root, scene)
        os.makedirs(scene_depth_dir, exist_ok=True)

        bin_files = glob.glob(os.path.join(scene_bin_dir, '*.bin'))
        for bin_file in bin_files:
            fname = os.path.basename(bin_file).replace('.bin', '.png')
            save_path = os.path.join(scene_depth_dir, fname)

            points = load_bin(bin_file)
            u, v, z = project_points(points)
            depth_map = create_depth_map(u, v, z)
            save_depth_png(depth_map, save_path)

if __name__ == '__main__':
    main()
