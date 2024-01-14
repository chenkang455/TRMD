import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from network import RE_Net
from utils import *
import torch.nn as nn
from dataset_h5 import REBlur
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import yaml

set_random_seed(1)
# create dataset
def create_dataset(opt):
    test_dataset = concatenate_h5_datasets(
        REBlur,
        opt.data_path_test,
        num_bin=6,
        )
    return test_dataset

# create dataloader
def create_dataloader(test_dataset, opt):
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False)
    return test_loader

# output metrics information
def log_metrics(metrics):
    info = 'MSE: {:.6f} PSNR: {:.3f} SSIM: {:.3f}'.format(
        metrics['MSE'].avg, metrics['PSNR'].avg, metrics['SSIM'].avg)
    return info


def prepare():
    global test_loader,unet,integral_net,criterion,device
    # basic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    test_dataset = create_dataset(opt)
    # dataloader
    test_loader = create_dataloader(test_dataset, opt)
    # model setting
    unet = nn.DataParallel(RE_Net(out_channels=6, event_channels=6)).to(device)
    # load net 
    unet.load_state_dict(torch.load(opt.load_path)['state_dict'])

def integral_cal_sharp(blur_image,res_pre):
    res_sum_ori = 0
    res_sum_ori = torch.sum(res_pre,dim = 1)
    L_f = blur_image - res_sum_ori / 7 
    L_t = L_f.repeat(1,6,1,1) + res_pre
    L_t = L_t
    return L_f,L_t


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

from measure import mse,psnr,ssim

def detect(epoch,loader):
    with torch.no_grad():
        unet.eval()
        pbar = tqdm(total=len(loader))
        for i,item in enumerate(loader):
            # load data
            blur_image = item['blur_image'].float().to(device)
            voxel = item['voxel'].float().to(device)
            sharp = item['sharp_image'].float().to(device)
            # calculation
            res_pre = unet(blur_image,voxel)
            L_f,L_t = integral_cal_sharp(blur_image,res_pre)
            # visualization results
            os.makedirs(f'Result/{epoch}', exist_ok=True)
            for j in range(len(blur_image)):
                save_image(np.array(blur_image[j,0].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_blur_{j}')
                save_image(np.array(L_f[j,0].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_deblur_{j}')
                save_image(np.array(sharp[j,0].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_sharp_{j}')
            pbar.update(1)
        pbar.close()

def get_parser():
    dic = read_yaml('config.yaml')
    parser = argparse.ArgumentParser()
    # dataset path settings
    parser.add_argument("--data_path_test",default=dic['REBlur']['test'])
    # train & test settings
    parser.add_argument("--test_batch_size", default=dic['test_setting']['batch_size'])
    # model parameter settings
    parser.add_argument("--num_bin", default=dic['num_bin'])
    # model loading path
    parser.add_argument("--load_path",default=dic['unet']['load_path'])
    # load model
    parser.add_argument("--load_unet", default= dic['unet']['load'])
    parser.add_argument("--seed",  default= dic['seed'])
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    global opt
    opt = get_parser()
    prepare()
    detect('test_REBlur',test_loader)
