#  Solution of team HIASBUAA of FLARE2023 Challenge

## [Coarse to Fine Segmentation Method Enables Accurate and Efficient Segmentation of Organs and Tumor in Abdominal CT](https://openreview.net/forum?id=MnPJnCPyKY&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMICCAI.org%2F2023%2FFLARE%2FAuthors%23your-submissions))

*Hui Meng, Haochen Zhao, Deqian Yang, wang ping, Zhenpeng Li*


# this is the official code of Team hiasbuaa for [FLARE23 Challenge](https://codalab.lisn.upsaclay.fr/competitions/12239)


### Overview of our work.

![image](https://github.com/zhuji423/MICCAI2023_Flare2023/blob/main/images/architecture.jpg?raw=true)

we use coarse to fine three stage method to segment organs and tumors. 
1. we train a coarse model to make the focus of ct  is on the abdomial area.
    - coarse model was trained on *Group6* 219 CTs
2. we train a tumor model to segment tumors.
    - using coarse model to inference the *Group4* 1497 CTs to make a dataset with full organ and tumor.
    - tumor model was trained on the above dataset
3. we train a organ model to segment organs.
    - organ model was trained on *Group1* and *Group2* 4000 CTs ,which pseudo-label was obtained from flare committee ([blackbean](https://drive.google.com/drive/folders/1sQ89xJsTeplXF6FFVwT7E5p8w0FUiyeP?usp=sharing))


## Dataset 
![datagroup](https://github.com/zhuji423/MICCAI2023_Flare2023/blob/main/images/datagroup.jpg?raw=true)
we divide the data in to 6 groups

## Preprocessing
A brief description of the preprocessing method

- cropping: Before model training, the training CT scans are cropped along the z-axis direction based on ground truth or pseudo labels. Specifically, the indices of start slice and the end slice of region containing targets are first calculated based on labels. To reserve context information of segmentation targets, we reduce the index of the start slice by 10 and add the index of the end slice by 10 During model training, the cropped CT scans are further cropped based on non-zero region introduced by nnU-Net.

- intensity normalization: We gather pixel values in the cropped CT scans and subsequently truncateall data to fall within [0.5, 99.5] of foreground voxel values .Following that,z-score normalization is applied.

- Resampling method for anisotropic data:: We perform image redirection to the desired orientation, followed by resampling all CT scans to match the median voxel spacing of the training dataset. Specifically, third-order spline interpolation is used for image resampling, and the nearest neighbor interpolation is employed for label resampling.

## Environments and Installation
- Ubuntu 20.04 LTS
- CPU 13th Gen Intel(R) Core(TM) i7-13700KF 3.40 GHz
- RAM 32GB;
- GPU 1 NVIDIA RTX 4090 12G
- CUDA version 12.1
- python version 3.10


## Training

```
cd nnunet/experiment_planning/
python nnUNet_convert_decathlon_task.py
python nnUNet_plan_and_preprocess.py
python run_training.py 3d_lowres nnUNetTrainerV2
```

## Inference
we use three stage framework to inference the online 100 validation set. we provide 4 pretrained weights to get the  results,which is provided on the [zenodo]()

```
python predict_final.py
```

## Reference

MACCAI FLARE2023 https://codalab.lisn.upsaclay.fr/competitions/12239

MACCAI FLARE2022 Team balackbean https://github.com/Ziyan-Huang/FLARE22

## Citations


