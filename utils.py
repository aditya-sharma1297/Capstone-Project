import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Calculating pixel accuracy between predicted and ground truth segmentation
# Ignoring pixels with the 'ignore_index' label (default is 255)
def pixel_accuracy(pred, label, ignore_index=255):
    valid = (label != ignore_index)
    correct = (pred == label) & valid
    return correct.sum().item() / valid.sum().item()

# Computes mean Intersection over Union (IoU) across all classes
def mean_iou(pred, label, num_classes, ignore_index=255):
    ious = []
    pred = pred.view(-1)
    label = label.view(-1)

    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]

    # Computing IoU for each class
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        intersection = (pred_inds & label_inds).sum().item() # Pixels correctly predicted
        union = (pred_inds | label_inds).sum().item()        # All pixels predicted

        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

# Defining colour map to visualise each class in the segmentation output
RELLIS_COLORMAP = [
    (0, 0, 0),           # 0 - void
    (108, 64, 20),       # 1 - dirt
    (255, 229, 204),     # 2 - sand
    (0, 102, 0),         # 3 - grass
    (0, 255, 0),         # 4 - tree
    (0, 153, 153),       # 5 - pole
    (0, 128, 255),       # 6 - water
    (0, 0, 255),         # 7 - sky
    (255, 255, 0),       # 8 - vehicle
    (255, 0, 127),       # 9 - object
    (64, 64, 64),        # 10 - asphalt
    (229, 229, 229),     # 11 - building windows
    (255, 0, 0),         # 12 - building
    (102, 102, 255),     # 13 - bridge
    (102, 0, 204),       # 14 - rail track
    (102, 0, 0),         # 15 - log
    (153, 0, 0),         # 16 - sign
    (204, 153, 255),     # 17 - person
    (102, 0, 204),       # 18 - fence
    (255, 153, 204),     # 19 - bush
    (170, 170, 255),     # 20 - traffic cone
    (41, 121, 255),      # 21 - barrier (type 1)
    (134, 255, 239),     # 22 - barrier (type 2)
    (170, 170, 170),     # 23 - concrete
    (255, 255, 255),     # 24 - container
    (140, 140, 200),     # 25 - edge of building
    (255, 255, 199),     # 26 - tire
    (255, 128, 0),       # 27 - pushable pullable
    (191, 255, 128),     # 28 - removable
    (255, 255, 128),     # 29 - obstacle
    (224, 224, 224),     # 30 - trash
    (41, 121, 255),      # 31 - puddle
    (134, 255, 239),     # 32 - manhole
    (99, 66, 34),        # 33 - mud
    (110, 22, 138),      # 34 - rubble
]

# Converting a label mask (2D array of class indices) into an RGB image
def decode_segmap(label_mask):
    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    
    for l in range(0, len(RELLIS_COLORMAP)):
        idx = label_mask == l
        r[idx], g[idx], b[idx] = RELLIS_COLORMAP[l]
    
    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb)

