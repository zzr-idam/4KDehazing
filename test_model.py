import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import network
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch 
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from tqdm import tqdm
import kornia
import dataset
from torch.nn import functional as F
from torchvision.utils import save_image
import network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)


my_model.load_state_dict(torch.load("/home/dell/4Kdehaze/model/4K_ohaze.pth")) 
#GAN.load_state_dict(torch.load("/home/dell/IJCAI/JBL/JBPSC/model/model_g_epoch69.pth"))
to_pil_image = transforms.ToPILImage()


tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])



def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' in name]
    name_list.sort()
    return name_list
   
list_s = load_simple_list('/home/dell/4Kdehaze/OHAZE_test')


i = 0

for idx in range(1):
     image_in = Image.open('/home/dell/4Kdehaze/OHAZE_test/27_outdoor_hazy.jpg').convert('RGB')

     full = tfs_full(image_in).unsqueeze(0).to(device)
     

     output = my_model(full)

     save_image(output[0], 'test_result/{}.jpg'.format('27_outdoor_hazy'))


