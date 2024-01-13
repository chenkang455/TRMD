import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from network import RE_Net
from utils import *
import torch.nn as nn
from dataset_h5 import GoPro_7
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import argparse
import yaml
import matplotlib.pyplot as plt
from measure import psnr,mse
from loss import LOSS


# create dataset
def create_dataset(opt):
    train_dataset = concatenate_h5_datasets(
        GoPro_7,
        opt.data_path_train,
        num_bin=opt.num_bin,
        use_roi = True,
        rgb = opt.rgb)

    test_dataset = concatenate_h5_datasets(
        GoPro_7,
        opt.data_path_test,
        num_bin=opt.num_bin,
        use_roi = False,
        rgb = opt.rgb)
    return train_dataset, test_dataset


# create dataloader
def create_dataloader(train_dataset, test_dataset, opt):
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,)
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    return train_loader, test_loader

# output metrics information
def log_metrics(metrics):
    info = 'MSE: {:.6f} PSNR: {:.3f} SSIM: {:.3f}'.format(
        metrics['MSE'].avg, metrics['PSNR'].avg, metrics['SSIM'].avg)
    return info


def prepare(opt):
    global train_loader,test_loader,unet,criterion,device,optimizer,scheduler
    # basic settings
    set_random_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    train_dataset, test_dataset = create_dataset(opt)
    # dataloader
    train_loader, test_loader = create_dataloader(train_dataset, test_dataset, opt)
    # model setting
    unet = nn.DataParallel(RE_Net(rgb = opt.rgb,out_channels=opt.num_bin - 1, event_channels=opt.num_bin - 1)).to(device)
    # train setting
    optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 ,1e-7)
    # load net and optimizer parameter
    if opt.load_unet:
        unet.load_state_dict(torch.load(opt.load_path)['state_dict'])
        optimizer.load_state_dict(torch.load(opt.load_path)['optimizer'])
        scheduler.load_state_dict(torch.load(opt.load_path)['scheduler']) 
    criterion = LOSS()


def cal_res(blur_image,res_pre):
    if opt.rgb == False:
        output_image = blur_image - (torch.sum(res_pre,axis = 1,keepdim=True))/7
    else:
        bs,channels,w,h = res_pre.shape
        res_pre = res_pre.reshape(bs,channels//3,3,w,h)
        output_image = blur_image - (torch.sum(res_pre,axis = 1,keepdim=False))/7
    return output_image


def train(opt):
    global train_loader,test_loader,unet,optimizer,scheduler,criterion,device,epoch
    # train and test loss save
    train_loss_plot = []
    test_loss_plot = []
    # -------------------train part-------------------
    for epoch in range(opt.num_epoch):
        unet = unet.train()
        pbar = tqdm(total=len(train_loader))
        train_loss = []
        for item in (train_loader):
            # load data
            blur_image = item['blur_image'].float().to(device)
            voxel = item['voxel'].float().to(device)
            sharp_image = item['sharp_image'].float().to(device)
            res_sharp = item['res_sharp'].float().to(device)
            res_pre = unet(blur_image,voxel)
            output_image = cal_res(blur_image,res_pre)
            loss = criterion(res_sharp,res_pre,sharp_image,output_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            pbar.update(1)
        del loss
        # scheduler.step()
        train_loss_plot.append(sum(train_loss)/len(train_loss))
        pbar.write(f"iter:{epoch},loss:{sum(train_loss)/len(train_loss)}")
        torch.cuda.empty_cache()
        # -------------------test part-------------------
        if epoch % 10 == 0:
            metrics = test(test_loader)
            print(log_metrics(metrics))
            test_loss_plot.append(metrics['MSE'].avg)
            # save the last checkpoint
            checkpoint_unet = {'state_dict': unet.state_dict(), 'optimizer': optimizer.state_dict() , 'scheduler': scheduler.state_dict()}
            torch.save(checkpoint_unet, opt.save_unet_path)
        pbar.close()
    # plot and save result
    save_plot('train_loss',range(len(train_loss_plot)),train_loss_plot)
    save_plot('test_loss',range(len(test_loss_plot)),test_loss_plot)

def test(test_loader):
    global unet
    # rgb measure metric 
    if opt.rgb == True:
        from measure import ssim_color as ssim
    else:
        from measure import ssim
    metrics = {}
    metrics_name_list = ['MSE', 'PSNR', 'SSIM']
    metrics_method_list = [mse, psnr, ssim]
    for metric_name in metrics_name_list:
        metrics[metric_name] = AverageMeter()
    pbar = tqdm(total=len(test_loader))
    unet.eval()
    with torch.no_grad():
        for item in test_loader:
            blur_image = item['blur_image'].float().to(device)
            sharp_image = item['sharp_image'].float()
            voxel = item['voxel'].float().to(device)
            res_pre = unet(blur_image,voxel)
            output_image = cal_res(blur_image,res_pre)
            # calculate metric
            if opt.rgb == False:
                output_image = output_image.detach().cpu().numpy().squeeze(axis=1)
                output_image = output_image.clip(0,1)
                sharp_image = sharp_image.numpy().squeeze(axis=1)
            else:
                output_image = output_image.detach().cpu().numpy()
                output_image = output_image.clip(0,1)
                sharp_image = sharp_image.numpy()
            for metric_name, metric_method in zip(metrics_name_list, metrics_method_list):
                metrics[metric_name].update(
                    metric_method(output_image, sharp_image))
            pbar.update(1)
        pbar.close()
        detect(epoch,test_loader)
    del output_image
    torch.cuda.empty_cache()
    return metrics

# read yaml from config.yaml
def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

# visualize the deblurred result
def detect(epoch,loader):
    with torch.no_grad():
        unet.eval()
        os.makedirs(f'Result/train/{epoch}', exist_ok=True)
        for item in loader:
            # load data
            blur_image = item['blur_image'].float().to(device)
            sharp_image = item['sharp_image'].float().to(device)
            voxel = item['voxel'].float().to(device)
            res_pre = unet(blur_image,voxel)
            output_image = cal_res(blur_image,res_pre)
            output_image = output_image.clip(0,1)
            for j in range(opt.test_batch_size):
                save_image(np.array(output_image[j].detach().cpu()),f'train/{epoch}/output_{j}',rgb = opt.rgb)
                save_image(np.array(sharp_image[j].detach().cpu()),f'train/{epoch}/sharp_{j}',rgb = opt.rgb)
                save_image(np.array(blur_image[j].detach().cpu()),f'train/{epoch}/blur_{j}',rgb = opt.rgb)
            break
        
# parser reading
def get_parser():
    dic = read_yaml('config.yaml')
    parser = argparse.ArgumentParser()
    # dataset path settings
    parser.add_argument("--data_path_train",default=dic['GOPRO']['train'])
    parser.add_argument("--data_path_test",default=dic['GOPRO']['test'])
    # train & test settings
    parser.add_argument("--train_batch_size",default=dic['train_setting']['batch_size'])
    parser.add_argument("--test_batch_size", default=dic['test_setting']['batch_size'])
    parser.add_argument("--num_workers", default=dic['train_setting']['num_workers'])
    parser.add_argument("--num_epoch", default=dic['train_setting']['num_epoch'])
    # model parameter settings
    parser.add_argument("--num_bin", default=dic['num_bin'])
    # load model or not
    parser.add_argument("--load_unet", default= dic['unet']['load'])
    # model loading path
    parser.add_argument("--load_path",default=dic['unet']['load_path'])
    # model saving path --last
    parser.add_argument("--save_unet_path",default=dic['unet']['save_path'])
    # lr
    parser.add_argument("--lr",  default= dic['train_setting']['lr'])
    parser.add_argument("--seed",  default= dic['seed'])
    # rgb
    parser.add_argument("--rgb",  default = dic['rgb'])
    opt = parser.parse_args()
    return opt

# main function
if __name__ == "__main__":
    global opt
    opt = get_parser()
    prepare(opt)
    train(opt)

        
