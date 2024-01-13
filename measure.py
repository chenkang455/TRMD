from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
import torch
import lpips
loss_fn_vgg = lpips.LPIPS(net='alex').to()

def lpips_cal(img1,img2):
    re = 0
    img1 = (torch.tensor(img1)).to(torch.float32)
    img2 = (torch.tensor(img2)).to(torch.float32)
    if img1.ndim == 2:
        return loss_fn_vgg(img1, img2)[0,0,0,0].detach().numpy()
    for i in range(len(img1)):
        re += loss_fn_vgg(img1[i, :, :], img2[i, :, :])[0,0,0,0].detach().numpy()
    re = re / len(img1)
    return re

def psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    re = 0
    for i in range(len(img1)):
        re += peak_signal_noise_ratio(img1[i, :, :], img2[i, :, :])
    re = re / len(img1)
    return re

def ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    re = 0
    for i in range(len(img1)):
        re += structural_similarity(img1[i, :, :], img2[i, :, :])
    re = re / len(img1)
    return re

def ssim_color(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    re = 0
    for i in range(len(img1)):
        re += structural_similarity(img1[i, :, :], img2[i, :, :],
                                    channel_axis=0,data_range=1.0)
    re = re / len(img1)
    return re

def mse(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)   
    re = 0
    for i in range(len(img1)):
        re += mean_squared_error(img1[i, :, :], img2[i, :, :])
    re = re / len(img1)
    return re


