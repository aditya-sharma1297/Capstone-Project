# Project Title - Self-Supervised Learning for Unmanned Ground Vehicle Using Multi-Modal Data

About Project - Self-Supervised Learning for Unmanned Ground Vehicles Using Multi-Modal Data Unmanned Ground Vehicles (UGVs) have garnered increasing attention due to their
ability to autonomously navigate dynamic, unstructured environments. A key component for this autonomy is **semantic segmentation**, which allows the vehicle to understand its
surroundings.
This project addresses the challenge of limited annotated data in off-road environments by introducing a self-supervised learning pipeline. Using **RGB** and **LiDAR depth**
data as input, we fine-tune the **DeepLabV3+** segmentation model to improve perception and generalization in low-label regimes.
Our work leverages the **RELLIS-3D** dataset, a benchmark for UGV research and demonstrates improved segmentation performance under label-scarce conditions.

# Dataset
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

# Preparing the Data
   
