import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from model import get_multimodal_deeplabv3plus


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 60
DATA_ROOT_RGB = "Pretraining_Set/rgb"
DATA_ROOT_LIDAR = "Pretraining_Set/lidar_depth"

# Base dataset, loading raw RGB and LiDAR image pairs
class RGBLiDARDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        rgb_path, lidar_path = self.file_list[idx]
        rgb_img = Image.open(rgb_path).convert("RGB")
        lidar_img = Image.open(lidar_path).convert("RGB")
        return rgb_img, lidar_img

# Dataset for contrastive learning, returns two augmented views of each image pair
class MultiViewRGBLiDARDataset(Dataset):
    def __init__(self, base_dataset, transform_rgb, transform_lidar):
        self.base_dataset = base_dataset
        self.transform_rgb = transform_rgb
        self.transform_lidar = transform_lidar
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        rgb_img, lidar_img = self.base_dataset[idx]
        rgb1 = self.transform_rgb(rgb_img)
        lidar1 = self.transform_lidar(lidar_img)
        rgb2 = self.transform_rgb(rgb_img)
        lidar2 = self.transform_lidar(lidar_img)
        return (rgb1, lidar1), (rgb2, lidar2)

# SimCLR style image transformations
simclr_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Preparing list of RGB and LiDAR image pairs
file_list = []
scenes = sorted(os.listdir(DATA_ROOT_RGB))
for scene in scenes:
    rgb_dir = os.path.join(DATA_ROOT_RGB, scene)
    lidar_dir = os.path.join(DATA_ROOT_LIDAR, scene)
    if not os.path.isdir(rgb_dir) or not os.path.isdir(lidar_dir):
        continue
    rgb_files = sorted(os.listdir(rgb_dir))
    for f in rgb_files:
        name = os.path.splitext(f)[0]
        rgb_path = os.path.join(rgb_dir, f)
        lidar_path = os.path.join(lidar_dir, name + ".png")
        if os.path.exists(rgb_path) and os.path.exists(lidar_path):
            file_list.append((rgb_path, lidar_path))
print(f"[INFO] Found {len(file_list)} image pairs.")

# Loading base and SSL dataset
base_dataset = RGBLiDARDataset(file_list)
ssl_dataset = MultiViewRGBLiDARDataset(base_dataset, simclr_transform, simclr_transform)
train_loader = DataLoader(ssl_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Encoder wrapper to extract encoder's output (segmentation head removed)
class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, lidar):
        # Concatenating RGB and LiDAR along channel dimension (6-channel input)
        x = torch.cat([rgb, lidar], dim=1)
        
        # Extracting encoder features
        features = self.model.forward_features(x)

        # Use last layer’s output if multiple feature maps are returned
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # Applying global average pooling to get a 1D feature vector
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return pooled


# SimCLR Projection head, maps encoder features to contrastive space
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# NT-Xent loss
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenating positive pairs for similarity matrix
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).to(z1.device)
    positives = torch.cat([torch.diag(similarity_matrix, batch_size),
                           torch.diag(similarity_matrix, -batch_size)]).view(2 * batch_size, 1)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)
    
    # Combining positives and negatives into logits
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# Intializing full model
full_model = get_multimodal_deeplabv3plus(num_classes=35, return_features=True).to(DEVICE)
encoder = EncoderWrapper(full_model).to(DEVICE)

# Getting encoder output dimension by forwarding a dummy batch
dummy_rgb = torch.randn(1, 3, 224, 224).to(DEVICE)
dummy_lidar = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    feat = encoder(dummy_rgb, dummy_lidar)
feature_dim = feat.shape[1]

projection_head = ProjectionHead(in_dim=feature_dim).to(DEVICE)

# Optimizer for encoder and projection head
optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=3e-4)

# Training loop
encoder.train()
projection_head.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for (rgb1, lidar1), (rgb2, lidar2) in train_loader:
        rgb1, lidar1 = rgb1.to(DEVICE), lidar1.to(DEVICE)
        rgb2, lidar2 = rgb2.to(DEVICE), lidar2.to(DEVICE)

        # Getting encoder outputs for both views
        h1 = encoder(rgb1, lidar1)
        h2 = encoder(rgb2, lidar2)

        # Projecting features into contrastive space
        z1 = projection_head(h1)
        z2 = projection_head(h2)

        # Computing contrastive loss
        loss = nt_xent_loss(z1, z2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculating average loss per epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] SSL Loss: {avg_loss:.4f}")

    # Saving encoder and projection head every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
        torch.save(projection_head.state_dict(), f"projhead_epoch_{epoch+1}.pth")

print("Self-supervised pretraining complete!")


