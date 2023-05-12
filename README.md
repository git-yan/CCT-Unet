# CCT-Unet
This repo is the official implementation of "A U-shaped Network based on Convolution Coupled Transformer for Segmentation of Peripheral and Transition Zones in Prostate MRI".

## overview
In this work, a U-shaped network based on the convolution coupled Transformer is proposed for segmentation of peripheral and transition zones in prostate MRI, named the convolution coupled Transformer U-Net (CCT-Unet). The convolutional embedding block is first designed for encoding high-resolution input to retain the edge detail of the image. Then the convolution coupled Transformer block is proposed to enhance the ability of local feature extraction and capture long-term correlation that encompass anatomical information. The feature conversion module is also proposed to alleviate the semantic gap in the process of jumping connection.

![](https://github.com/git-yan/CCT-Unet/blob/main/CCT-Unet%20framework.jpg?raw=true)

## Pretrain model on ProstateX
| model       | pretrain    | resolution  | #params     | FLOPs       | Pretrain model |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CCT-Unet    | ProstateX   | 224x224     | ProstateX   | 224x224     | [model](https://pan.baidu.com/s/11JSZz1Mr4C9pYrBEYGsiEA?pwd=0000 )   |



## Requirements
einops==0.6.1 \
numpy==1.23.5 \
timm==0.6.13 \
torch==1.12.1



## Reference
A U-shaped Network based on Convolution Coupled Transformer for Segmentation of Peripheral and Transition Zones in Prostate MRI, Yifei Yan, Rongzong Liu, Haobo Chen, Limin Zhang, Qi Zhang \
G. Litjens, O. Debats, J. Barentsz, et al., “Computer-aided detection of prostate cancer in MRI,” IEEE Trans. Med. Imaging, vol. 33, no. 5, pp. 1083–1092, 2014. \
Y. Liu, K. Sung, G. Yang, et al., “Automatic Prostate Zonal Segmentation Using Fully Convolutional Network with Feature Pyramid Attention,” IEEE Access, vol. 7, pp. 163626–163632, 2019.
