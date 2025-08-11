import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from model import get_multimodal_deeplabv3plus
from dataset import RGBLiDARDataset
from utils import pixel_accuracy, mean_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 35
BATCH_SIZE = 4
NUM_EPOCHS = 60
FREEZE_ENCODER_EPOCHS = 5
PRETRAINED_ENCODER_PATH = "encoder_epoch_60.pth"
SAVE_PATH = "finetuned_multimodal_deeplabv3.pth"

# Defining Dataset Paths for RGB images, LiDAR Depth images and Segmentation Labels
DATA_ROOT_RGB = "Fine_Tuning_Set/rgb"
DATA_ROOT_LIDAR = "Fine_Tuning_Set/lidar_depth"
DATA_ROOT_LABEL = "Fine_Tuning_Set/rgb_labels"

# Image Pre-processing transforms
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
        rgb_path = os.path.join(rgb_dir, f)
        lidar_path = os.path.join(lidar_dir, name + ".png")
        label_path = os.path.join(label_dir, name + ".png")
        if os.path.exists(rgb_path) and os.path.exists(lidar_path) and os.path.exists(label_path):
            file_list.append((rgb_path, lidar_path, label_path))

print(f"[INFO] Found {len(file_list)} samples.")

# Creating dataset and data loaders
dataset = RGBLiDARDataset(file_list, transform_rgb=transform, transform_lidar=transform)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Initializing model
model = get_multimodal_deeplabv3plus(num_classes=NUM_CLASSES).to(DEVICE)

# Loading pretrained encoder
if os.path.exists(PRETRAINED_ENCODER_PATH):
    encoder_ckpt = torch.load(PRETRAINED_ENCODER_PATH, map_location=DEVICE)
    model.model.encoder.load_state_dict(encoder_ckpt, strict=False)
    print("Loaded pretrained encoder.")

# Freezing encoder for warm-up
def freeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = False

# Unfreezing encoder after warm-up
def unfreeze_encoder(model):
    for param in model.model.encoder.parameters():
        param.requires_grad = True

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Training loop
best_miou = 0.0
for epoch in range(NUM_EPOCHS):
    if epoch == 0:
        freeze_encoder(model)
        print("Encoder frozen for warm-up.")
    elif epoch == FREEZE_ENCODER_EPOCHS:
        unfreeze_encoder(model)
        print("Encoder unfrozen for fine-tuning.")

    model.train()
    total_loss = 0.0
    for rgb, lidar, label in train_loader:
        rgb, lidar, label = rgb.to(DEVICE), lidar.to(DEVICE), label.to(DEVICE)
        label[label >= NUM_CLASSES] = 255

        # Fusing RGB and LiDAR inputs into 6 channel input
        fused_input = torch.cat([rgb, lidar], dim=1)  # (B, 6, H, W)
        
        # Forward Pass
        out = model(fused_input)
        loss = criterion(out, label)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    # Validation Loop
    model.eval()
    total_miou = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for rgb, lidar, label in val_loader:
            rgb, lidar, label = rgb.to(DEVICE), lidar.to(DEVICE), label.to(DEVICE)
            label[label >= NUM_CLASSES] = 255

            fused_input = torch.cat([rgb, lidar], dim=1)
            out = model(fused_input)
            # Getting predicted pixel per class
            pred = out.argmax(1)

            total_acc += pixel_accuracy(pred, label)
            total_miou += mean_iou(pred, label, NUM_CLASSES)
            count += 1

    avg_miou = total_miou / count
    avg_acc = total_acc / count

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
          f"Loss: {avg_loss:.4f} | mIoU: {avg_miou:.4f} | Acc: {avg_acc:.4f}")

    # Saveing the best model
    if avg_miou > best_miou:
        best_miou = avg_miou
        torch.save(model.state_dict(), SAVE_PATH)
        print("Saved best fine-tuned model.")

print("Fine-tuning complete.")

