
# CasA: A Cascade Attention Network for 3D Object Detection from LiDAR point clouds 

`CasA` is a simple multi-stage 3D object detection framework based on a Cascade Attention design.
`CasA` can be integrated into many SoTA 3D detectors and greatly improve their detection performance. 
The paper of "CasA: A Cascade Attention Network for 3D Object Detection from LiDAR point clouds" can be found [here](https://ieeexplore.ieee.org/abstract/document/9870747).  
This code is mostly built upon [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Note that, the CasA++ is based on a transfer learning framework: pre-training on Waymo and fine-tuning on KITTI. Since additional data has been included, we did not release the CasA++ codes. 

## Overview
- [Cascade Attention Design](#cascade-attention-design)
- [Model Zoo](#model-zoo)
- [Getting Started](#getting-started)
- [Citation](#citation)


## Cascade Attention Design
Cascade frameworks have been widely studied in
2D object detection but less investigated in 3D
space. Conventional cascade structures use multiple separate sub-networks to sequentially refine
region proposals. Such methods, however, have
limited ability to measure proposal quality in all
stages, and hard to achieve a desirable detection
performance improvement in 3D space. We
propose a new cascade framework, termed CasA,
for 3D object detection from point clouds. CasA
consists of a Region Proposal Network (RPN) and
a Cascade Refinement Network (CRN). In this
CRN, we designed a new Cascade Attention Module that uses multiple sub-networks and attention
modules to aggregate the object features from different stages and progressively refine region proposals.
CasA can be integrated into various two-stage 3D detectors and greatly improve their detection performance. 
Extensive experimental results
on KITTI and Waymo datasets with various baseline detectors demonstrate the universality and superiority 
of our CasA. In particular, based on one
variant of Voxel-RCNN, we achieve state-of-the-art
results on KITTI 3D object detection benchmark.

![framework](./docs/framework.png)

## Update Log

* 2022/10/15 Update a 3D multi-object tracker [CasTrack](https://github.com/hailanyi/3D-Multi-Object-Tracker) based on the CasA detections, currently **rank first** on the KITTI tracking leader-board :fire:!

* 2022/9/30 Update details of [installation](#installation). Update [environment](#environment-we-tested) we tested. Update [Spconv2.X](https://github.com/traveller59/spconv) support :rocket:!

* 2022/3/3 Initial update, achieve SOTA performance on the KITTI 3D detection leader-board

## Model Zoo

### KITTI 3D Object Detection Results
The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.
Currently, this repo supports CasA-PV, CasA-V, CasA-T and CasA-PV2. The base detectors are 
PV-RCNN, Voxel-RCNN, CT3D and PV-RCNN++, respectively.
* All released models are trained with 2 3090 GPUs and are available for download. 
* These models are not suitable to directly report results on KITTI *test* set, please use slightly lower score threshold and 
train the models on all or 80% training data to achieve a desirable performance on KITTI *test* set.

#### PV-RCNN VS. CasA-PV
|               Detectors               | Car(R11/R40) | Pedestrian(R11/R40) | Cyclist(R11/R40)  | download |
|:---------------------------------------------:|:-------:|:-------:|:-------:|:---------:|
| [PV-RCNN baseline](https://github.com/open-mmlab/OpenPCDet) | 83.90/84.83 | 57.90/56.67 | 70.47/71.95 |   | 
| [CasA-PV](tools/cfgs/kitti_models/CasA-PV.yaml) | **86.18/85.86** | **58.90/59.17** | 66.01/69.09 | [model-44M](https://drive.google.com/file/d/1QolF8lkGwlJDpN3MV7-Y5MdhBCROJnfC/view?usp=sharing) | 

#### Voxel-RCNN VS. CasA-V
|               Detectors               | Car(R11/R40) | Pedestrian(R11/R40) | Cyclist(R11/R40)  | download |
|:---------------------------------------------:|:-------:|:-------:|:-------:|:---------:|
| [Voxel-RCNN baseline](https://github.com/open-mmlab/OpenPCDet) | 84.52/85.29 | 61.72/60.97 | 71.48/72.54 |   | 
| [CasA-V](tools/cfgs/kitti_models/CasA-V.yaml)   | **86.54/86.30** | **67.93/66.54** | **74.27/73.08** | [model-44M](https://drive.google.com/file/d/13LO8BAz0h1MbXg97i8k18pHfWGxXEjFP/view?usp=sharing) |

#### CT3D VS. CasA-T
|               Detectors               | Car(R11/R40) | Pedestrian(R11/R40) | Cyclist(R11/R40)  | download |
|:---------------------------------------------:|:-------:|:-------:|:-------:|:---------:|
| [CT3D3cat baseline](https://github.com/hlsheng1/CT3D) | 84.97/85.04 | 56.28/55.58 | 71.71/71.88 |   | 
| [CasA-T](tools/cfgs/kitti_models/CasA-T.yaml)   | **86.76/86.44** | **60.91/62.53** | **73.36**/71.83 | [model-22M](https://drive.google.com/file/d/1pZ4xIa7aTPwAgxUDcbE7b_edctLVXQbb/view?usp=sharing)| 

#### PV-RCNN++ VS. CasA-PV2
|               Detectors               | Car(R11/R40) | Pedestrian(R11/R40) | Cyclist(R11/R40)  | download |
|:---------------------------------------------:|:-------:|:-------:|:-------:|:---------:|
| *[PV-RCNN++ baseline](https://github.com/open-mmlab/OpenPCDet) | 85.36/85.50 | 57.43/57.15 | 71.30/71.85 |   | 
| [CasA-PV2](tools/cfgs/kitti_models/CasA-PV2.yaml)   | **86.32/86.10** | **59.50/60.54** | **72.74/73.16** | [model-47M](https://drive.google.com/file/d/1POWX2ruds3t0XOSvBz5-VmG67c4F9mfE/view?usp=sharing) | 

Where * denodes reproduced results of a simplified version using their open-source codes. 

### Waymo Open Dataset Results
Here we provided two models on WOD, where CasA-V-center denotes that the center-based RPN are used.
All models are trained with **a single frame**  on 8 V100 GPUs, and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).    

|    100\% Data, 2 returns        | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|:---------------------------------------------:|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| *[Voxel-RCNN baseline](https://github.com/open-mmlab/OpenPCDet)|77.43/76.71| 68.73/68.24 | 76.37/68.21 | 67.92/60.40 | 68.74/67.56 | 66.46/65.35 |
| [CasA-V](tools/cfgs/waymo_models/CasA-V.yaml)|78.54/78.00| 69.91/69.42 | 80.88/73.10 | 71.87/64.78 | 69.66/68.38 | 67.07/66.83 |
| [CasA-V-Center](tools/cfgs/waymo_models/CasA-V-Center.yaml) |**78.62/78.04** | **69.94/69.47** | **81.76/75.69** | **72.75/67.21** | **72.47/71.18** | **70.20/68.94**|

Where * denodes reproduced results using their open-source codes.

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you could easily achieve similar performance by training with the default configs.

## Getting Started

```
conda create -n spconv2 python=3.9
conda activate spconv2
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Environment we tested

Our released implementation is tested on.
+ Ubuntu 18.04
+ Python 3.6.9 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ [Spconv 1.2.1](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)
+ NVIDIA CUDA 11.1
+ 8x Tesla V100 GPUs

We also tested on.
+ Ubuntu 18.04
+ Python 3.9.13 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ [Spconv 2.1.22](https://github.com/traveller59/spconv) # pip install spconv-cu111
+ NVIDIA CUDA 11.1 
+ 2x 3090 GPUs

### Prepare Dataset 

#### KITTI Dataset

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
CasA
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```



#### Waymo Dataset

```
CasA
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_train_val_test
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_waymo_track_dbinfos_train_cp.pkl
│   │   │── waymo_infos_test.pkl
│   │   │── waymo_infos_train.pkl
│   │   │── waymo_infos_val.pkl
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.waymo.waymo_tracking_dataset --cfg_file tools/cfgs/dataset_configs/waymo_tracking_dataset.yaml 
```

#### Installation

```
git clone https://github.com/hailanyi/CasA.git
cd CasA
python3 setup.py develop
```

### Training and Evaluation

#### Evaluation

```
cd tools
python3 test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

For example, if you test the CasA-V model:

```
cd tools
python3 test.py --cfg_file cfgs/kitti_models/CasA-V.yaml --ckpt CasA-V.pth
```

Multiple GPU test: you need modify the gpu number in the dist_test.sh and run
```
sh dist_test.sh 
```
The log infos are saved into log-test.txt
You can run ```cat log-test.txt``` to view the test results.

#### Training

```
cd tools
python3 train.py --cfg_file ${CONFIG_FILE}
```

For example, if you train the CasA-V model:

```
cd tools
python3 train.py --cfg_file cfgs/kitti_models/CasA-V.yaml
```

Multiple GPU train: you can modify the gpu number in the dist_train.sh and run
```
sh dist_train.sh
```
The log infos are saved into log.txt
You can run ```cat log.txt``` to view the training process.

## Acknowledgement
This repo is developed from `OpenPCDet 0.3`, we thank shaoshuai shi for his implementation of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).   

## Citation 
If you find this project useful in your research, please consider cite:

```
@article{casa2022,
    title={CasA: A Cascade Attention Network for 3D Object Detection from LiDAR point clouds},
    author={Wu, Hai and Deng, Jinhao and Wen, Chenglu and Li, Xin and Wang, Cheng},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    year={2022}
}
```
