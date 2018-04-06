# LSTM-water-table-depth-prediction

## Introduction
This is an implementation of [DeepLab-Xception](https://arxiv.org/pdf/1802.02611) on Python 3, TFSlim and TensorFlow for semantic segmentation on the [Penn-Fudan Pedestrian Detection and Segmentation Database](http://www.cis.upenn.edu/~jshi/ped_html/).The model generates segmentation masks for the image. It's based on a modified Xception backbone.
## Status
In progress
## Requirements
```
python>=3.5.2
numpy>=1.14.0
scipy>=1.0.0
opencv-python>=3.3.0
tensorflow>=1.2.0
```
## Training on  Penn-Fudan Pedestrian
```
python train.py --data-dir path/to/your/data --data-list path/to/your/list
```
To see the documentation on each of the training settings run the following:
```
python train.py --help
```

## Evaluation
The following command provides the description of each of the evaluation settings:
```
python evaluate.py --help
```
## Inference
```
python inference.py --data-dir path/to/your/data --data-list path/to/your/list --model-dir path/to/your/model
```
To see the documentation on each of the inference settings run the following:
```
python inference.py --help
```
## Citation
```
@article{deeplabv3+,
  author = {
Chen, Liang-Chieh; Zhu, Yukun; Papandreou, George; Schroff, Florian; Adam, Hartwig},
  title = {Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
## Acknowledgment
This repo borrows tons of code from
1. [DrSleep/tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet)
2. [hellochick/PSPNet-tensorflow](https://github.com/hellochick/PSPNet-tensorflow)
3. [kwotsin/TensorFlow-Xception](https://github.com/kwotsin/TensorFlow-Xception)


