import os
import shutil
import imagehash
from PIL import Image
from tqdm import tqdm
import re

# Utility function to extract frame ID like 000123 from image name
def extract_frame_id(filename):
    match = re.search(r'frame(\d{6})-', filename)
    return match.group(1) if match else None

# Paths to existing sets, used to avoid overlap
test_rgb_path = "Fine_Tuning_Set/rgb"
test_rgb_path_2 = "Validation_Set/rgb"

# Output path for new folder
val_output_root = "Pretraining_Set"

# Creating subfolders for RGB images, labels, LiDAR data and LiDAR labels
for sub in ["rgb", "rgb_labels", "lidar_bin", "lidar_label"]:
    os.makedirs(os.path.join(val_output_root, sub), exist_ok=True)

# Defining source paths 
rgb_root = "Rellis-3D Images"
rgb_label_root = "Rellis-3D Labels"
lidar_bin_root = "Rellis-3D Ouster Bin"
lidar_label_root = "Rellis-3D Ouster Annotation"

# Subfolders inside each diretory
rgb_sub = "pylon_camera_node"
rgb_label_sub = "pylon_camera_node_label_id"
lidar_bin_sub = "os1_cloud_node_kitti_bin"
lidar_label_sub = "os1_cloud_node_semantickitti_label_id"

# Parameters
SCENES = ["00000", "00001", "00002", "00003", "00004"]
HASH_THRESHOLD = 8
IMAGES_PER_FOLDER = 20
total_copied = 0

# Collecting data that is already used in test_path sets to avoid overlap
test_frame_ids = {scene: set() for scene in SCENES}

for test_path in [test_rgb_path, test_rgb_path_2]:
    for scene in SCENES:
        test_scene_path = os.path.join(test_path, scene)
        if not os.path.exists(test_scene_path):
            continue
        for fname in os.listdir(test_scene_path):
            if fname.endswith(".jpg"):
                frame_id = fname.replace(".jpg", "")
                test_frame_ids[scene].add(frame_id)

# Process to select diverse images for the new output folder
for scene in SCENES:
    print(f"\n📁 Processing scene {scene} for validation set...")

    rgb_path = os.path.join(rgb_root, scene, rgb_sub)
    rgb_label_path = os.path.join(rgb_label_root, scene, rgb_label_sub)
    lidar_bin_path = os.path.join(lidar_bin_root, scene, lidar_bin_sub)
    lidar_label_path = os.path.join(lidar_label_root, scene, lidar_label_sub)

    # Getting list of all RGB image files
    image_files = [f for f in os.listdir(rgb_path) if f.endswith(".jpg")]
    selected_hashes = []
    selected_files = []

    # To select images from the start (reverse=False) or from the end of the folder
    for fname in sorted(image_files, reverse=True):
        if len(selected_files) >= IMAGES_PER_FOLDER:
            break

        frame_id = extract_frame_id(fname)
        if not frame_id or frame_id in test_frame_ids[scene]:
            continue

        # Derives corresponding filenames for label, LiDAR point cloud, and LiDAR label
        # based on the current RGB image's filename and frame ID
        label_name = fname.replace(".jpg", ".png")
        bin_name = f"{frame_id}.bin"
        lidar_label_name = f"{frame_id}.label"

        src_rgb = os.path.join(rgb_path, fname)
        src_label = os.path.join(rgb_label_path, label_name)
        src_bin = os.path.join(lidar_bin_path, bin_name)
        src_lidar_label = os.path.join(lidar_label_path, lidar_label_name)

        # Ensuring all corresponding files exist
        if not (os.path.exists(src_rgb) and os.path.exists(src_label) and
                os.path.exists(src_bin) and os.path.exists(src_lidar_label)):
            continue

        # Visual diversity check
        try:
            img = Image.open(src_rgb).convert("RGB")
            img_hash = imagehash.phash(img)

            # Add if the image is visually distinct enough
            if all(img_hash - h > HASH_THRESHOLD for h in selected_hashes):
                selected_hashes.append(img_hash)
                selected_files.append((fname, frame_id))
        except:
            continue

    print(f"✅ Selected {len(selected_files)} validation frames for scene {scene}")

    # Copying files to output directory
    for fname, frame_id in tqdm(selected_files):
        label_name = fname.replace(".jpg", ".png")
        bin_name = f"{frame_id}.bin"
        lidar_label_name = f"{frame_id}.label"

        src_rgb = os.path.join(rgb_path, fname)
        src_label = os.path.join(rgb_label_path, label_name)
        src_bin = os.path.join(lidar_bin_path, bin_name)
        src_lidar_label = os.path.join(lidar_label_path, lidar_label_name)

        dst_rgb = os.path.join(val_output_root, "rgb", scene, f"{frame_id}.jpg")
        dst_label = os.path.join(val_output_root, "rgb_labels", scene, f"{frame_id}.png")
        dst_bin = os.path.join(val_output_root, "lidar_bin", scene, f"{frame_id}.bin")
        dst_lidar_label = os.path.join(val_output_root, "lidar_label", scene, f"{frame_id}.label")

        for path in [dst_rgb, dst_label, dst_bin, dst_lidar_label]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            shutil.copy(src_rgb, dst_rgb)
            shutil.copy(src_label, dst_label)
            shutil.copy(src_bin, dst_bin)
            shutil.copy(src_lidar_label, dst_lidar_label)
            total_copied += 1
        except Exception as e:
            print(f"Copy failed for frame {frame_id}: {e}")

print(f"\nDone! Total validation samples copied: {total_copied}")
