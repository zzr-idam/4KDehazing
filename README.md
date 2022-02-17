# 4KDehazing
This is the PyTorch implementation for our CVPR'21 paper. 
The model can removal hazy, smoke or even water impurities.


The repository includes:
1. Source code of our model.
2. Training code for O-hazy dataset.
3. Testing code for O-hazy dataset
4. Pre-trained model for O-hazy dataset.

Setup:
依赖的库
torch, numpy, tqdm, torchvision, kornia, opencv-python


Training
将带雾训练数据集放在./hazy 文件夹下 对应的清晰数据集放在./gt文件夹下。
运行命令 python train.py。 
训练过程可在./result文件夹下找到。
模型保存在./model文件夹下。

Test model
将需要测试的数据集放在./OHAZE_test文件下。
运行命令 python test_model.py。
测试结果可在./test_result文件夹下找到。

Requirements：
Python 3.7
PyTorch 1.6.0
CUDA 10.0
Ubuntu 16.04


Dataset (Daytime)
Link：https://pan.baidu.com/s/1sqJpxvt1-ONqcuG7RLTq0A
Password：vodp

Nighttime with Haze 4K dataset
Link： https://pan.baidu.com/s/1pxmsFOU-3ELNgR8KtvWW1g
Password：5h6A

Pre-model
Link：https://pan.baidu.com/s/1UwDL8rzTVFYFDwOsU9TBlA 
Password：u5pk 

4K real-world video with hazy
Link：https://pan.baidu.com/s/1wqQKEPLnzTPqANAr-D5Mxw 
Password：p83y 

4K real-world images
Link：https://pan.baidu.com/s/1u5Wq3aKBVxZhAkVdHsdMVQ 
Password：776z 

4K nighttime videos
Link：https://pan.baidu.com/s/1lCPh70aKfOd2Zg6Ah37jGg 
Password：h1in 

Cite:
{
  title     = {Ultra-High-Definition Image Dehazing via Multi-Guided Bilateral Learning},
  booktitle = {CVPR},
  year      = {2021}
}


# Other work
<font color="Hotpink">Our team is moving towards under-screen cameras, and we hope you will support us more.</font>
https://github.com/zzr-idam/under_camera




