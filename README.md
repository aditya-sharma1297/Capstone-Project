# Project Title - Self-Supervised Learning for Unmanned Ground Vehicle Using Multi-Modal Data

About Project - Self-Supervised Learning for Unmanned Ground Vehicles Using Multi-Modal Data Unmanned Ground Vehicles (UGVs) have garnered increasing attention due to their
ability to autonomously navigate dynamic, unstructured environments. A key component for this autonomy is **semantic segmentation**, which allows the vehicle to understand its
surroundings.
This project addresses the challenge of limited annotated data in off-road environments by introducing a self-supervised learning pipeline. Using **RGB** and **LiDAR depth**
data as input, we fine-tune the **DeepLabV3+** segmentation model to improve perception and generalization in low-label regimes.
Our work leverages the **RELLIS-3D** dataset, a benchmark for UGV research and demonstrates improved segmentation performance under label-scarce conditions.

## Dataset
From [Rellis-3D Dataset](https://github.com/unmannedlab/RELLIS-3D), download Full Images (RGB), Full Image Annotations ID Format and Ouster LiDAR SemanticKITTI Format (.bin format) data. Once downloaded, all the folders will have a similar structure as follows:

```
RELLIS-3D LiDAR
|--Rellis Ouster Bin
   |-- 00000
       |-- os1_cloud_node_kitti_bin
           |-- .bin format files
   |-- 00001
       |-- os1_cloud_node_kitti_bin
           |-- .bin format files
   |-- 00002
       |-- os1_cloud_node_kitti_bin
           |-- .bin format files
   |-- 00003
       |-- os1_cloud_node_kitti_bin
           |-- .bin format files
   |-- 00004
       |-- os1_cloud_node_kitti_bin
           |-- .bin format files

RELLIS-3D Images/Label IDs
|-- 00000
    |-- pylon_camera_node/pylon_camera_node_label_id
        |-- images
|-- 00001
    |-- pylon_camera_node/pylon_camera_node_label_id
        |-- images
|-- 00002
    |-- pylon_camera_node/pylon_camera_node_label_id
        |-- images
|-- 00003
    |-- pylon_camera_node/pylon_camera_node_label_id
        |-- images
|-- 00004
    |-- pylon_camera_node/pylon_camera_node_label_id
        |-- images        
```

## Preparing the Data
### Step 1: Generating LiDAR Depth Images
The first step is to generate LiDAR depth images from the raw Ouster LiDAR .bin files using the script generate_lidar_depth.py. These depth images are essential and will later be used alongside the RGB images during training

### Step 2: Creating Dataset Split
The next step is to create a dataset split for Pre-training, Fine Tuning and Validation. For this, the script prepare_dataset_split.py can be used. It copies RGB images, labels, and LiDAR data from the main dataset into separate folders for each split. The script ensures that there are no duplicate images across splits by checking existing files before copying.

## Requirements
* Python ≥ 3.8
* PyTorch ≥ 1.10
* segmentation_models_pytorch
* torchvision
* imagehash
* matplotlib, tqdm, opencv-python, Pillow, etc.

## Script Execution Order
1. generate_lidar_depth.py – Generate LiDAR-based depth maps from point clouds.
2. prepare_dataset_split.py – Split the dataset into pre-training, fine tuning and validation sets.
3. pre_training.py – Run SimCLR-style self-supervised learning on RGB and depth data.
4. fine_tuning.py – Fine-tune the segmentation model using labeled data.
5. validation.py – Evaluate model performance on the validation set.

Following modules are imported and used by the main scripts:
1. utils.py - Utility functions for data processing, visualization, and general helpers.
2. dataset.py - Defines dataset classes and PyTorch data loaders.
3. model.py - Implements the DeepLabV3+ segmentation model and related components.
