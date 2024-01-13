import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from network import RE_Net
from utils import *
import torch.nn as nn
from dataset_h5 import GoPro_7,REBlur
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import yaml
from measure import psnr,ssim
from measure import mean_squared_error as mse
from loss import LOSS

# create dataset
def create_dataset(opt):
    test_dataset = concatenate_h5_datasets(
        GoPro_7,
        opt.data_path_test,
        num_bin=opt.num_bin,
        use_roi = False,
        rgb = opt.rgb)
    return test_dataset

# create dataloader
def create_dataloader(test_dataset, opt):
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle = False)
    return test_loader

# output metrics information
def log_metrics(metrics):
    info = 'MSE: {:.6f} PSNR: {:.3f} SSIM: {:.4f}'.format(
        metrics['MSE'].avg, metrics['PSNR'].avg, metrics['SSIM'].avg)
    return info

# Deblur & HFR modules
def cal_res(blur_image,res_pre):
    if opt.rgb == False:
        output_image = blur_image - (torch.sum(res_pre,axis = 1,keepdim=True))/7
    else:
        bs,channels,w,h = res_pre.shape
        res_pre = res_pre.reshape(bs,channels//3,3,w,h)
        output_image = blur_image - (torch.sum(res_pre,axis = 1,keepdim=False))/7
    return output_image

def prepare():
    global test_loader,unet,device,test_dataset
    # basic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    test_dataset = create_dataset(opt)
    # dataloader
    test_loader = create_dataloader(test_dataset, opt)
    # model setting
    unet = nn.DataParallel(RE_Net(out_channels=opt.num_bin - 1, event_channels=opt.num_bin - 1,rgb = opt.rgb)).to(device)
    # load net 
    unet.load_state_dict(torch.load(opt.load_path)['state_dict'])

def test(test_loader):
    global unet
    metrics = {}
    metrics_HFR = {}
    metrics_name_list = ['MSE', 'PSNR', 'SSIM']
    metrics_method_list = [mse, psnr, ssim]
    for metric_name in metrics_name_list:
        metrics[metric_name] = AverageMeter()
        metrics_HFR[metric_name] = AverageMeter()
    pbar = tqdm(total=len(test_loader))
    with torch.no_grad():
        unet.eval()
        for item in test_loader:
            blur_image = item['blur_image'].float().to(device)
            sharp_image = item['sharp_image'].float().to(device)
            voxel = item['voxel'].float().to(device)
            res_pre = unet(blur_image,voxel).clip(-1,1)
            output_image = cal_res(blur_image,res_pre)
            HFR = output_image.repeat(1,6,1,1) + res_pre
            HFR = HFR.clip(0,1)
            res_sharp = item['res_sharp'].float().to(device)
            HFR_gt = sharp_image.repeat(1,6,1,1) + res_sharp
            # calculate metric
            if opt.rgb == True:
                output_image = output_image.detach().cpu().numpy().squeeze(axis=0)
                output_image = output_image.clip(0,1)
                sharp_image = sharp_image.cpu().detach().numpy().squeeze(axis=0)
            else:
                output_image = output_image.detach().cpu().numpy().squeeze(axis=1)
                output_image = output_image.clip(0,1)
                sharp_image = sharp_image.cpu().detach().numpy().squeeze(axis=1)
            HFR = HFR.cpu().detach().numpy().squeeze(axis=0)
            HFR_gt = HFR_gt.cpu().detach().numpy().squeeze(axis=0)
            
            # deblurring and HFR metric result
            for metric_name, metric_method in zip(metrics_name_list, metrics_method_list):
                metrics[metric_name].update(
                    metric_method(output_image, sharp_image))
                metrics_HFR[metric_name].update(
                    metric_method(HFR, HFR_gt))
            pbar.update(1)
        pbar.close()
    # del output_image
    torch.cuda.empty_cache()
    print("deblurring metric results:")
    print(log_metrics(metrics))
    print("HFR metric results:")
    print(log_metrics(metrics_HFR))
    return metrics

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict
        
def detect(epoch,loader):
    with torch.no_grad():
        unet.eval()
        os.makedirs(f'Result/{epoch}', exist_ok=True)
        print("Data Loaded")
        pbar = tqdm(total=len(loader))
        for i,item in enumerate(loader):
            # load data
            blur_image = item['blur_image'].float().to(device)
            sharp_image = item['sharp_image'].float().to(device)
            voxel = item['voxel'].float().to(device)
            res_pre = unet(blur_image,voxel).clip(-1,1)
            output_image = cal_res(blur_image,res_pre)
            output_image = output_image.clip(0,1)
            pbar.update(1)
            os.makedirs(f'Result/{epoch}', exist_ok=True)
            for j in range(len(voxel)):
                save_image(np.array(output_image[j,:].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_output_{j}',rgb = opt.rgb)
                save_image(np.array(sharp_image[j,:].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_sharp_{j}',rgb = opt.rgb)
                save_image(np.array(blur_image[j,:].detach().cpu()),f'{epoch}/{str(i).zfill(4)}_blur_{j}',rgb = opt.rgb)
        pbar.close()

def get_parser():
    dic = read_yaml('config.yaml')
    parser = argparse.ArgumentParser()
    # dataset path settings
    parser.add_argument("--data_path_test",default=dic['GOPRO']['test'])
    # train & test settings
    parser.add_argument("--test_batch_size", default=dic['test_setting']['batch_size'])
    # model parameter settings
    parser.add_argument("--num_bin", default=dic['num_bin'])
    # model loading path
    parser.add_argument("--load_path",default=dic['unet']['load_path'])
    # load model
    parser.add_argument("--load_unet", default= dic['unet']['load'])
    parser.add_argument("--seed",  default= dic['seed'])
    # rgb
    parser.add_argument("--rgb",  default = dic['rgb'])
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    global opt
    opt = get_parser()
    prepare()
    # test-metric
    metrics = test(test_loader)
    # visualize the final result
    detect('test_GoPro',test_loader)

