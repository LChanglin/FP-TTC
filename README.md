# FP-TTC: Fast Prediction of Time-to-Collision using Monocular Images

Official implementation of "FP-TTC: Fast Prediction of Time-to-Collision using Monocular Images". Submitted to T-CSVT on May **, 2024.

![pipeline](./images/pipeline.png)



## Abstract

**Time-to-Collision (TTC)** is a measure of the time until an object collides with the observation plane which is a critical input indicator for obstacle avoidance and other downstream modules. Previous works have utilized deep neural networks to estimate TTC with monocular cameras in an end-toend manner, which obtain the state-of-the-art (SOTA) accuracy performance. However, these models usually have deep layers and numerous parameters, resulting in long inference time and high computational overhead. Moreover, existing methods use two frames which are the current and future moments as input to calculate the TTC resulting in a delay during the calculation process. To solve these issues, we propose a novel fast TTC prediction model: FP-TTC. We first use an attention-based scale encoder to model the scale-matching process between images, which significantly reduces the computational overhead as well as improves the model’s accuracy. Meanwhile, a simple but powerful trick is introduced to the model, where we built a time-series decoder and predict the current TTC from RGB images in the past, avoiding the computational delay caused by the system time step interval, and further improved the TTC prediction speed. Compared to the previous SOTA work, our model achieves a parameter reduction of 89.1%, a 6-fold increase in inference speed, a 19.3% improvement in accuracy.

## Setup

### Environment

Our experiments are conducted in Ubuntu20.04 with Anaconda3, Pytorch 1.12.0, CUDA 11.3, 3090 GPU.

1. create conda environment:

```shell
conda create -n fpttc python=3.8 -y
conda activate fpttc
```

2. install dependencies:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt 
```

3. clone our code:

```
git clone https://github.com/LChanglin/FP-TTC.git
```

4. download our pretrained weights from [link](https://drive.google.com/drive/folders/1WL2cuKDt2YPERB8WaAScX9qbO4x4p4hI?usp=sharing).



### Datasets

Download Driving and KITTI for training. 

```bash
Datasets
|-- Driving
|   |-- camera_data
|   |-- disparity
|   |-- disparity_change
|   |-- frames_cleanpass
|   `-- optical_flow
`-- kitti
    |-- data_scene_flow
    |   |-- testing
    |   `-- training
    |-- data_scene_flow_calib
    |   |-- testing
    |   `-- training
    `-- data_scene_flow_multi
    |-- testing
    `--training
```



## Usage

We use 3090 GPUs for training and testing.

### training

```bash
# train with our settings
# --resume: load with pretrained weights, used for finetuning (default:./pretrained/fpttc_mix.pth.tar)
# --epoch: training epoches
# --lr: learning rate: set as mentioned in out paper
# --image_size: resolution
sh train.sh
```



### inference with your own data

```bash
# test with our settings
# --resume: load with pretrained weights (default:./pretrained/fpttc_mix.pth.tar)
# --inference_dir: tested images
sh train.sh
```



### evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
--resume ./pretrained/fpttc_mix.pth.tar \
--inference_dir [PATH TO KITTI]/testing/image_2/
```

The evaluation results will be saved as .npy.

| Pretrained Weights       | Mid Error |
| ------------------------ | --------- |
| fintuned on kitti        | **59.35** |
| trained on mixed dataset | 62.30     |



## Visualization

KITTI:

![viz](./images/viz.png)