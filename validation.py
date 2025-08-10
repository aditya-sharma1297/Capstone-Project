import os
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from model import get_multimodal_deeplabv3plus
from dataset import RGBLiDARDataset
from utils import decode_segmap, pixel_accuracy, mean_iou


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 35
BATCH_SIZE = 1
MODEL_PATH = "finetuned_multimodal_deeplabv3.pth"
DATA_ROOT_RGB = os.path.join("/Users/adityasharma/Desktop/Validation_Set/rgb")
DATA_ROOT_LIDAR = os.path.join("/Users/adityasharma/Desktop/Validation_Set/lidar_depth")
DATA_ROOT_LABEL = os.path.join("/Users/adityasharma/Desktop/Validation_Set/rgb_labels")

# Saving predictions to a directory
SAVE_DIR = "Final Predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# Defining Loss Function for evaluation
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Image pre-processing transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Building a file list of RGB, LiDAR and Label
file_list = []
for scene in sorted(os.listdir(DATA_ROOT_RGB)):
    rgb_dir = os.path.join(DATA_ROOT_RGB, scene)
    lidar_dir = os.path.join(DATA_ROOT_LIDAR, scene)
    label_dir = os.path.join(DATA_ROOT_LABEL, scene)
    if not all(os.path.isdir(p) for p in [rgb_dir, lidar_dir, label_dir]):
        continue
    for f in os.listdir(rgb_dir):
        name = os.path.splitext(f)[0]
        paths = (
            os.path.join(rgb_dir, f),
            os.path.join(lidar_dir, name + ".png"),
            os.path.join(label_dir, name + ".png")
        )
        if all(os.path.exists(p) for p in paths):
            file_list.append(paths)

# Preparing dataset and data loader
dataset = RGBLiDARDataset(file_list, transform_rgb=transform, transform_lidar=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Loading the model
model = get_multimodal_deeplabv3plus(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluation and visualization loop
total_acc, total_miou, total_loss, count = 0, 0, 0, 0

for idx, (rgb, lidar, label) in enumerate(loader):
    rgb, lidar, label = rgb.to(DEVICE), lidar.to(DEVICE), label.to(DEVICE)
    
    # Formatting label correctly and ignoring invalid class values
    label = label.squeeze(1)
    label[label >= NUM_CLASSES] = 255

    # Merging RGB and LiDAR into one 6 channel input
    fused_input = torch.cat([rgb, lidar], dim=1)  # (B, 6, H, W)

    with torch.no_grad():
        # Running the model and getting predicted class per pixel
        out = model(fused_input)
        pred = out.argmax(1)
        
        # Computing and accumulating loss
        loss = criterion(out, label)
        total_loss += loss.item()

    total_acc += pixel_accuracy(pred, label)
    total_miou += mean_iou(pred, label, NUM_CLASSES)
    count += 1

    # Visualization
    # Reversing Normalization for display
    rgb_image = rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    rgb_image = np.clip(rgb_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    
    # Decoding segmentation maps into color images
    gt_img = decode_segmap(label.squeeze(0).cpu().numpy())
    pred_img = decode_segmap(pred.squeeze(0).cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_image); axes[0].set_title("RGB")
    axes[1].imshow(gt_img); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_img); axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"result_{idx:03d}.png"))
    plt.close(fig)

# Computing final metrics summary
avg_loss = total_loss / count
avg_acc = total_acc / count
avg_miou = total_miou / count

print(f"Validation Results — Loss: {avg_loss:.4f}, Mean IoU: {avg_miou:.4f}, Pixel Accuracy: {avg_acc:.4f}")
print(f"Visualization saved to: {SAVE_DIR}")
