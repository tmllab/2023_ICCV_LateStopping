# Late Stopping

This is the code for the paper:

[Late Stopping: Avoiding Confidently Learning from Mislabeled Examples.](https://openaccess.thecvf.com/content/ICCV2023/papers/Yuan_Late_Stopping_Avoiding_Confidently_Learning_from_Mislabeled_Examples_ICCV_2023_paper.pdf)

ICCV 2023 Poster.

# BibTex
@inproceedings{yuan2023late,
  title={Late stopping: Avoiding confidently learning from mislabeled examples},
  author={Yuan, Suqin and Feng, Lei and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16079--16088},
  year={2023}
}

## Dependencies
We implement our methods by PyTorch on NVIDIA RTX 3090&4090 GPU. The environment is as bellow:
- [PyTorch](https://PyTorch.org/), version = 1.11.0
- [Ubuntu20.04](https://ubuntu.com/download)
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 11.3

## Experiments
We verify the effectiveness of the proposed method on noisy datasets. In this repository, we provide the used [datasets](https://www.cs.toronto.edu/~kriz/cifar.html). 
You should put the datasets in the folder ''cifar-10'' and ''cifar-100'' when you have downloaded them. 
Label files for CIFAR-10N is already in the "data" folder.


Here is a training example: 
```bash
python3 main_cifar10.py
 
