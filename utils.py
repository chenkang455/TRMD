import cv2
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import random
import torch
from torch.utils.data import ConcatDataset
import os
import argparse

class Timer:
    """
    count running time
    """

    def __init__(self, msg='Time elapsed'):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f'{self.msg}: {duration:.2f}s')

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The HDF5 dataset
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def filter_events_time(events, start_time, end_time, voxel_type='gopro_7'):
    """
    select time range
    type1 : dataset from cross attention model
    type2 : dataset from zhangxiang 
    """
    if voxel_type == 'EFNet':
        start_index = binary_search_h5_dset(events['ts'],start_time)
        end_index = binary_search_h5_dset(events['ts'],end_time)
        x = events['xs'][start_index:end_index]
        y = events['ys'][start_index:end_index]
        p = events['ps'][start_index:end_index].astype(int)
        p[p == 0] = -1
        t = events['ts'][start_index:end_index]
    elif voxel_type == 'gopro_7' :
        start_index = binary_search_h5_dset(events[3],start_time)
        end_index = binary_search_h5_dset(events[3],end_time)
        x = events[0][start_index:end_index].astype(np.int16)
        y = events[1][start_index:end_index].astype(np.int16)
        p = events[2][start_index:end_index].astype(int)
        t = events[3][start_index:end_index]
    elif voxel_type == 'EDI':
        start_index = binary_search_h5_dset(events[3],start_time)
        end_index = binary_search_h5_dset(events[3],end_time)
        x = events[0][start_index:end_index].astype(np.int16)
        y = events[1][start_index:end_index].astype(np.int16)
        p = events[2][start_index:end_index].astype(int)
        t = events[3][start_index:end_index]
        p[p==0] = -1
    return x, y, p, t 

def inv_normalize_time(t,start_time,end_time):
    inv_t = start_time + t * (end_time - start_time)
    return np.clip(inv_t,start_time,end_time)

def event2tensor_key(events, img_size, start_time, end_time, num_bin, keypoints):
    """
    event streams to [num_bin, C, H, W] event tensor, C=2 indicates polarity
    split T into N parts.Each part has the same time length.
    Return:
    E -> tensor [num_bin, C, H, W]
    events_bedding -> array [num_bedding] stores the events number information
    keypoints: shape:[num_bin] normalized to [0,...,1]
    """
    keypoints = inv_normalize_time(keypoints,start_time,end_time)
    E = np.zeros((num_bin, 2, img_size[0], img_size[1]))
    x, y, p, t = filter_events_time(events, start_time, end_time)
    events_num = len(t)
    events_bedding = np.zeros(num_bin)
    idx = np.zeros_like(t).astype(np.int8)
    for i in range(len(keypoints)-1):
        start_index = binary_search_h5_dset(t,keypoints[i])
        end_index = binary_search_h5_dset(t,keypoints[i+1])
        idx[start_index:end_index] = i
    index = (t == keypoints[i+1])
    idx[index] = i
    # idx范围为[0,num_bin]，因此需要限制最大值的idx减1
    # # p[(T0_p,T0_n),(T1_p,T1_n),...,(T15_p,T15_n)] 总共32长
    p[p == 1] = 0  # 正极性在第一个通道上
    p[p == -1] = 1  # 负极性在第二个通道上
    np.add.at(E, (idx, p, y, x), 1)
    events_bedding[:] = np.sum(E, axis=(1, 2, 3))
    return E, events_bedding

def event2tensor_time(events, img_size, start_time, end_time, num_bin,voxel_type ='gopro_7'):
    """
    event streams to [num_bin, C, H, W] event tensor, C=2 indicates polarity
    split T into N parts.Each part has the same time length.
    Return:
    E -> tensor [num_bin, C, H, W]
    events_bedding -> array [num_bedding] stores the events number information
    """
    interval_time = (end_time - start_time) / num_bin
    E = np.zeros((num_bin, 2, img_size[0], img_size[1]))
    x, y, p, t = filter_events_time(events, start_time, end_time,voxel_type)
    events_num = len(t)
    events_bedding = np.zeros(num_bin)
    new_t = t - start_time
    idx = np.floor(new_t/interval_time).astype(np.int8)
    idx[idx == num_bin] -= 1  # idx范围为[0,num_bin]，因此需要限制最大值的idx减1
    # p[(T0_p,T0_n),(T1_p,T1_n),...,(T15_p,T15_n)] 总共32长
    p[p == 1] = 0  # 正极性在第一个通道上
    p[p == -1] = 1  # 负极性在第二个通道上
    np.add.at(E, (idx, p, y, x), 1)
    events_bedding[:] = np.sum(E, axis=(1, 2, 3))
    return E, events_bedding

def event2tensor_space(events, img_size, start_time, end_time, num_bin,voxel_type ='gopro_7'):
    """
    datasource: event data from start_time to end_time
    event streams to [T, C, H, W] event tensor, C=2 indicates polarity
    split events' number into N parts.Each part has the same events number.
    Return:
    E -> tensor [num_bin, C, H, W]
    time_bedding -> array [num_bedding] : stores the time information in the middle of events stream
    """
    E = np.zeros((num_bin, 2, img_size[0], img_size[1]))
    x, y, p, t = filter_events_time(events, start_time, end_time,voxel_type)
    events_num = len(t)
    interval_space = events_num//num_bin
    time_bedding = np.zeros(num_bin)
    for idx in range(num_bin):
        if idx == num_bin - 1:
            x_ = x[idx*interval_space:]
            y_ = y[idx*interval_space:]
            p_ = p[idx*interval_space:]
            t_ = t[idx*interval_space:]
        else:
            x_ = x[idx*interval_space:(idx+1)*interval_space]
            y_ = y[idx*interval_space:(idx+1)*interval_space]
            p_ = p[idx*interval_space:(idx+1)*interval_space]
            t_ = t[idx*interval_space:(idx+1)*interval_space]
        p_[p_ == 1] = 0  # 正极性在第一个通道上
        p_[p_ == -1] = 1  # 负极性在第二个通道上
        np.add.at(E, (idx, p_, y_, x_), 1)
        time_bedding[idx] = t_[-1]
    return E, time_bedding

def normalize_image(image, percentile_lower=0, percentile_upper=100):
    """
    normalize image to [0,1]
    """
    mini, maxi = np.percentile(
        image, (percentile_lower, percentile_upper))  # mini,maxi = 1 if all pixes are the same
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)

def animate(images, fig_title=' ', fps=30):
    fig = plt.figure(figsize=(0.1, 0.1))  # don't take up room initially
    fig.suptitle(fig_title)
    fig.set_size_inches(7.2, 5.4, forward=False)  # resize but don't update gui
    ims = []
    for image in tqdm(images):
        im = plt.imshow(normalize_image(image), cmap='gray',
                        vmin=0, vmax=1, animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    ani.save(f'result/{fig_title}.gif', fps=fps)
    plt.close(ani._fig)

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

def concatenate_h5_datasets(dataset, path,select_path = None, **args):
    """concat all h5 file under path
    Args:
        dataset (class): dataset class eg: REBlur
        path (str): path that contains the h5 file
    """
    file_folder_path = path
    h5_file_path = [os.path.join(file_folder_path, s)
                    for s in os.listdir(file_folder_path)]
    print('Found {} h5 files in {}'.format(
        len(h5_file_path), file_folder_path))
    datasets = []
    for h5_file in h5_file_path:
        print(h5_file)
        if select_path == None:
            datasets.append(dataset(h5_file, **args))
        else:
            if h5_file == select_path:
                datasets.append(dataset(h5_file, **args))
    return ConcatDataset(datasets)

def save_plot(title, x_data, y_data):
    plt.clf()
    plt.title(title)
    plt.grid(linestyle=":")
    plt.plot(x_data, y_data)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.tight_layout()
    plt.savefig('Result/'+title+'.png')

def save_image(img,title = 'test',nor = False,rgb = False):
    # nor or save the result directly
    if nor:
        img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    else:
        img = img.clip(0,1)
        img = np.array(img * 255).astype(np.uint8)
    # rgb save or not
    if rgb == True:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img[0]
    cv2.imwrite('Result/' + title + '.png', img)

def save_res_image(img,title = 'test',nor = False):
    if nor:
        img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img = np.transpose(img, (1, 2, 0))
    else:
        img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imwrite('Result/' + title + '.png', img)

def save_event_image(voxel, title,mode='red-blue'):
    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[voxel > 0] = 255
        r[voxel < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = voxel.min(), voxel.max()
        event_preview = np.clip((255.0 * (voxel - m) / (M - m)).astype(np.uint8), 0, 255)
    cv2.imwrite('Result/' + title + '.png', event_preview)
    
    
def show_image(img, title='test', wait=True):
    image = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count
        
        
def save_event_image(voxel, title,mode='red-blue'):
    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((voxel.shape[0], voxel.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        g = event_preview[:, :, 1]
        r = event_preview[:, :, 2]
        event_preview.fill(255)
        b[voxel > 0] = 255
        g[voxel > 0] = 0
        r[voxel > 0] = 0
        
        b[voxel < 0] = 0
        r[voxel < 0] = 255
        g[voxel < 0] = 0

    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = voxel.min(), voxel.max()
        event_preview = np.clip((255.0 * (voxel - m) / (M - m)).astype(np.uint8), 0, 255)
    cv2.imwrite('Result/' + title + '.png', event_preview)