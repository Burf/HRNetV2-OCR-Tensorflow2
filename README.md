# HRNetV2 + OCR for Tensorflow2

This is an implementation of "HRNetV2 + OCR" on Keras and Tensorflow.

The implementation is based on papers[[1](https://arxiv.org/abs/1904.04514), [2](https://arxiv.org/abs/1909.11065v6)] and official implementations[[3](https://github.com/HRNet/HRNet-Semantic-Segmentation), [4](https://github.com/HRNet/HRNet-Image-Classification)].

## Model

- Model
  * HRNetV2 + Semantic Segmentation
  * HRNetV2 + Classifier
  * HRNetV2 + OCR + Semantic Segmentation
  * Backbone + HRNetV2 + @ (Custom Development)
- Pre-trained weight
  * X

## Requirements

- Python 3
- tensorflow 2

## Reference
 1. High-Resolution Representations for Labeling Pixels and Regions,
    Ke Sun, Yang Zhao, Borui Jiang, Tianheng Cheng, Bin Xiao, Dong Liu, Yadong Mu, Xinggang Wang, Wenyu Liu, Jingdong Wang,
	https://arxiv.org/abs/1904.04514

 2. Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation,
    Yuhui Yuan, Xiaokang Chen, Xilin Chen, Jingdong Wang,
    https://arxiv.org/abs/1909.11065v6
    
 3. High-resolution networks and Segmentation Transformer for Semantic Segmentation,
    HRNet,
    https://github.com/HRNet/HRNet-Semantic-Segmentation

 4. High-resolution networks (HRNets) for Image classification,
    HRNet,
	https://github.com/HRNet/HRNet-Image-Classification
   
## Contributor

 * Hyungjin Kim(flslzk@gmail.com)