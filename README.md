# Pytorch Implementation of PointNet and PointNet++

This repo is a fork of
[yanx27's torch implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
of [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf).

This fork features:

- Jupyter Notebook compatible
- Flexible configuration via arguments

## Update

**2024/08/13:**

Created fork.

(1) Add Jupyter Notebook Example.

(2) Config passed as arguments, which was hard-coded previously.

(3) Removed deprecated code / platform specific code.

**2021/03/27:**

(1) Release pre-trained models for semantic segmentation, where PointNet++ can achieve **53.5\%** mIoU.

(2) Release pre-trained models for classification and part segmentation in `log/`.

**2021/03/20:** Update codes for classification, including:

(1) Add codes for training **ModelNet10** dataset. Using setting of ``--num_category 10``.

(2) Add codes for running on CPU only. Using setting of ``--use_cpu``.

(3) Add codes for offline data preprocessing to accelerate training. Using setting of ``--process_data``.

(4) Add codes for training with uniform sampling. Using setting of ``--use_uniform_sample``.

**2019/11/26:**

(1) Fixed some errors in previous codes and added data augmentation tricks. Now classification by only 1024 points can achieve **92.8\%**!

(2) Added testing codes, including classification and segmentation, and semantic segmentation with visualization.

(3) Organized all models into `./models` files for easy using.

## Environment

The latest codes are tested on:

- WSL Ubuntu 24.04 on Windows 11 pro 24H2
- Intel 14700KF/64GB RAM
- Zotac RTX 4070 Super 12GB
  - Studio Driver 572.60
  - CUDA 12.4
- Python 3.12.3
  - packages: see `requrements.txt`

## Classification (ModelNet10/40)

### Data Preparation

~~Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.~~

The official website is down.
As of Mar.2025, data is available from:

- [Kaggle](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled)
- [Pointcept Huggingface](https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed)

Data tree:

- `<data directory>/modelnet40_normal_resampled/`
  - `airplane/`
    - `airplane_0001.txt`
    - `airplane_0002.txt`
    - ...
  - `bathtub/*`
  - `bed/*`
  - ...

### Run

You can run different modes with following codes.

- If you want to use offline processing of data, you can use `--process_data` in the first run. You can download pre-processd data [here](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing) and save it in `data/modelnet40_normal_resampled/`.
- If you want to train on ModelNet10, you can use `--num_category 10`.

```shell
# ModelNet40
## Select different models in ./models

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps

# ModelNet10
## Similar setting like ModelNet40, just using --num_category 10

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### Performance

| Model | Accuracy |
|--|--|
| PointNet (Official)                    |   89.2   |
| PointNet2 (Official)                   |   91.9   |
| PointNet (Pytorch without normal)      |   90.6   |
| PointNet (Pytorch with normal)         |   91.4   |
| PointNet2_SSG (Pytorch without normal) |   92.2   |
| PointNet2_SSG (Pytorch with normal)    |   92.4   |
| PointNet2_MSG (Pytorch with normal)    | **92.8** |

## Part Segmentation (ShapeNet)

### Data Preparation

~~Download alignment **ShapeNet**
[here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)
and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.~~

The official website is down.
As of Mar. 2025, data is available from:

- [Kaggle](https://www.kaggle.com/datasets/mitkir/shapenet?resource=download)

### Run

```shell
## Check model in ./models
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```

### Performance

| Model | Inctance avg IoU| Class avg IoU |
|--|--|--|
| PointNet (Official)     |   83.7   |   80.4   |
| PointNet2 (Official)    |   85.1   |   81.9   |
| PointNet (Pytorch)      |   84.3   |   81.1   |
| PointNet2_SSG (Pytorch) |   84.9   |   81.8   |
| PointNet2_MSG (Pytorch) | **85.4** | **82.5** |

## Semantic Segmentation (S3DIS)

### Data Preparation

~~Download 3D indoor parsing dataset (**S3DIS**)
[here](http://buildingparser.stanford.edu/dataset.html)
and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.~~

**NOTE (Aug. 2024)**: The website is down. Need to contact author to access data.

```shell
cd data_utils
python collect_indoor3d_data.py
```

Processed data will save in `data/stanford_indoor3d/`.

### Run

```shell
## Check model in ./models
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```

Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).

### Performance

| Model  | Overall Acc |Class avg IoU | Checkpoint |
|--|--|--|--|
| PointNet (Pytorch)      |   78.9   |   43.7   | [40.7MB](log/sem_seg/pointnet_sem_seg) |
| PointNet2_ssg (Pytorch) | **83.0** | **53.5** | [11.2MB](log/sem_seg/pointnet2_sem_seg) |

## Visualization

### Using show3d_balls.py

```shell
## build C++ code for visualization
cd visualizer
bash build.sh
## run one example
python show3d_balls.py
```

![](/visualizer/pic.png)

### Using MeshLab

![](/visualizer/pic2.png)

## Reference By

- [halimacc/pointnet3](https://github.com/halimacc/pointnet3)
- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [charlesq34/PointNet](https://github.com/charlesq34/pointnet): The official code for PointNet paper.
- [charlesq34/PointNet++](https://github.com/charlesq34/pointnet2): The official code for PointNet++ paper.
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch): Upstream repo.

## Citation

If you find this repo useful in your research, please consider citing it and our other works:

```bibTeX
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```

```bibTeX
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

```bibTeX
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```

```bibTeX
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds},
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```

## Selected Projects using The original Codebase (by yanx27)

- [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
- [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
- [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
- [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
- [PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud](https://github.com/lly007/PointStructuringNet)
- [Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22](https://github.com/dvlab-research/stratified-transformer)
