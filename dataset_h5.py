from torch.utils.data import Dataset
from utils import event2tensor_time,event2tensor_space
import h5py
import numpy as np
from skimage import color

# GOPRO dataset
class GoPro_7(Dataset):
    def __init__(self, path, num_bin = 7,mode = 'slow',
                 roi_size=(200, 256),threhold = 0,
                 use_roi = True,rgb = False, fraction_residual = False):
        self.num_bin = num_bin
        self.path = path
        self.mode = mode
        self.roi_size = roi_size
        self.use_roi = use_roi
        self.threhold = threhold
        self.fraction_residual = fraction_residual
        self.img_size = (360,640)
        self.rgb = rgb
        with h5py.File(self.path, 'r') as file:
            #! bug: h5py concide with num_worker; can't assign value to variable like self.events/blur_images/sharp_images
            self.dataset_len = file['blur_images'].shape[0]
        
    def __len__(self):
        return self.dataset_len
    
    def open_hdf5(self):
        """ 
        sharp_images:[index,w,h,3]
        blur_images:[index,w,h,3]
        events:[(x,y,p,t),events_length]
        """
        # slow mode : low ram low speed
        if self.mode == 'slow':
            self.data_h5 = h5py.File(self.path, 'r')
            self.blur_images = self.data_h5['blur_images']
            self.sharp_images = self.data_h5['sharp_images']
            self.voxel = self.data_h5['voxel']
        # fast mode : high ram high speed
        elif self.mode == 'fast':
            self.data_h5 = h5py.File(self.path, 'r')
            self.blur_images = self.data_h5['blur_images'][:].astype(np.uint8)
            self.sharp_images = self.data_h5['sharp_images'][:].astype(np.uint8)
            self.voxel = self.data_h5['voxel'][:]
        
    def __del__(self):
        if hasattr(self, 'data_h5'):
            self.data_h5.close()
            
    def get_blur_image(self,index):
        """ 
        return blurry image:[1,w,h]
        """
        if self.mode == 'slow':
            blur_img = self.blur_images[index][:].astype(np.uint8)
        elif self.mode == 'fast':
            blur_img = self.blur_images[index]
        blur_img_g = color.rgb2gray(blur_img)
        return np.expand_dims(blur_img_g,axis = 0)

    def get_blur_image_color(self,index):
        """
        rgb version
        """
        if self.mode == 'slow':
            blur_img = self.blur_images[index][:].astype(np.uint8)
        elif self.mode == 'fast':
            blur_img = self.blur_images[index]
        blur_img_rgb = blur_img.transpose((2, 0, 1))
        blur_img_rgb = blur_img_rgb.astype(np.float32) / 255.0
        return blur_img_rgb
    
    def get_sharp_image(self,index):
        """ 
        return sharp_images:[7,w,h]
        """
        if self.mode == 'slow':
            sharp_image = self.sharp_images[index][:].astype(np.uint8)
        elif self.mode == 'fast':
            sharp_image = self.sharp_images[index]
        sharp_image_g = []
        for i in range(len(sharp_image)):
            sharp_image_g.append(color.rgb2gray(sharp_image[i]))
            # sharp_image_g.append(cv2.cvtColor(sharp_image[i],cv2.COLOR_BGR2GRAY))
        return np.array(sharp_image_g)
    
    def get_sharp_image_color(self,index):
        """
        rgb version
        """
        if self.mode == 'slow':
            sharp_image = self.sharp_images[index][:].astype(np.uint8)
        elif self.mode == 'fast':
            sharp_image = self.sharp_images[index]
        sharp_image_g = []
        for i in range(len(sharp_image)):
            sharp_image_g.append(sharp_image[i].transpose((2, 0, 1)).astype(np.float32) / 255.0)
        return  np.array(sharp_image_g)
    
    def get_voxel_SCER(self, index):
        """
        get SCER voxel Ref: Event-Based Fusion for Motion Deblurring with Cross-modal Attention
        convert event streams to [num_bin, H, W] event tensor  
        **args: keypoints [num_bim]
        """
        if self.mode == 'slow':
            return self.voxel[index][:]
        elif self.mode == 'fast':
            return self.voxel[index]
    
    # Residual format proposed in this paper.
    def get_res_sharp(self,index):
        sharp_images = self.get_sharp_image(index)
        res_sharp = np.zeros((6,360,640))
        indices = [0,1,2,4,5,6]
        res_sharp = sharp_images[indices] - (sharp_images[3:4].repeat(6,axis = 0))
        return res_sharp
    
    # Fraction Residual format proposed in previous papers.
    def get_res_fraction(self,index,threhold):
        sharp_images = self.get_sharp_image(index)
        res_sharp = np.zeros((6,360,640))
        indices = [0,1,2,4,5,6]
        res_sharp = sharp_images[indices] / (sharp_images[3:4].repeat(6,axis = 0) + 1e-6)
        res_sharp = res_sharp.clip(0,threhold)
        return res_sharp
    
    # RGB-version Residual format proposed in this paper.
    def get_res_sharp_color(self,index):
        sharp_images = self.get_sharp_image_color(index)
        res_sharp = np.zeros((6,3,360,640))
        indices = [0,1,2,4,5,6]
        res_sharp = sharp_images[indices] - (sharp_images[3:4].repeat(6,axis = 0))
        return res_sharp
    
    def __getitem__(self, index):
        item = {}
        if not hasattr(self, 'data_h5'):
            self.open_hdf5()
        item['voxel'] = self.get_voxel_SCER(index)
        # rgb
        if self.rgb == True:
            item['blur_image'] = self.get_blur_image_color(index)
            item['res_sharp'] = self.get_res_sharp_color(index)
            item['res_sharp'] = item['res_sharp'].reshape(-1,item['res_sharp'].shape[-2],item['res_sharp'].shape[-1])
            item['sharp_image'] = self.get_sharp_image_color(index)[3]
        # Fraction residual format proposed in previous papers or direct residual in this paper. 
        elif self.fraction_residual == True:
            item['blur_image'] = self.get_blur_image(index)
            item['res_sharp'] = self.get_res_fraction(index,threhold = self.threhold)
            item['sharp_image'] = self.get_sharp_image(index)[3:4]
        else:
            item['blur_image'] = self.get_blur_image(index)
            item['res_sharp'] = self.get_res_sharp(index)
            item['sharp_image'] = self.get_sharp_image(index)[3:4]
        # roi for faster training
        if self.use_roi == True:
            roiTL = (np.random.randint(0, self.img_size[0]-self.roi_size[0]+1), np.random.randint(0, self.img_size[1]-self.roi_size[1]+1)) # top-left coordinate
            roiBR = (roiTL[0]+self.roi_size[0],roiTL[1]+self.roi_size[1])
            roiList = ['blur_image','res_sharp','voxel','sharp_image']
            for key in roiList:
                item[key] = item[key][:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]].astype(np.float32)
        return item

# Base Dataset
class BaseBlur(Dataset):
    def __init__(self, path, num_bin, voxel_type , use_raw_points , mode = 'fast'):
        self.num_bin = num_bin
        self.type = voxel_type
        self.path = path
        self.use_raw_points = use_raw_points
            
    def open_hdf5(self):
        self.data_h5 = h5py.File(self.path, 'r')
        self.events = self.data_h5['events']
        self.blur_images = self.data_h5['images']
        self.sharp_images = self.data_h5['sharp_images']

    def __del__(self):
        if hasattr(self, 'data_h5'):
            self.data_h5.close()

    def __len__(self):
        # bug: can't call open_hdf5 function
        return self.dataset_len

    def get_blur_frame(self, index):
        """
        return the frame at index [1,w,h]
        """
        if not hasattr(self, 'data_h5'):
            self.open_hdf5()
        blur_img = self.blur_images['image{:09d}'.format(
            index)]
        blur_img_g = color.rgb2gray(blur_img)
        return np.expand_dims(blur_img_g, axis=0)

    def get_gt_frame(self, index):
        """
        return the gt_frame at index [1,w,h]
        """
        if not hasattr(self, 'data_h5'):
            self.open_hdf5()
        sharp_img = self.sharp_images['image{:09d}'.format(
            index)]
        sharp_img_g = color.rgb2gray(sharp_img)
        return np.expand_dims(sharp_img_g, axis=0)

    def extract_time_from_index(self, index):
        raise NotImplementedError

    def get_voxel(self,index, num_bin,keypoints = None):
        if self.voxel_type == 'TIME':
            voxel = self.get_voxel_TIME(index, num_bin)
        elif self.voxel_type == 'SCER':
            voxel = self.get_voxel_SCER(index, num_bin)
        elif self.voxel_type == 'SCER_dynamic':
            voxel = self.get_voxel_SCER(index,num_bin,dynamic = True,keypoints = keypoints)
        else:
            print('No event type found')
        return voxel # 1,262,320 -> 1,260,320

    def get_voxel_SCER(self, index, num_bin,dynamic = False,keypoints = None):
        """
        get SCER voxel Ref: Event-Based Fusion for Motion Deblurring with Cross-modal Attention
        convert event streams to [num_bin, H, W] event tensor  
        """
        if not hasattr(self, 'data_h5'):
            self.open_hdf5()
        if dynamic == True and keypoints == None:
            raise ValueError("keypoints are none")
        img_size, start_time, end_time = self.extract_time_from_index(index)
        E, _ = event2tensor_time(self.events, img_size, start_time, end_time, num_bin,voxel_type ='EFNet')
        voxel = E[:, 0, :, :] - E[:, 1, :, :]
        re_voxel = np.zeros_like(voxel)
        left_voxel = voxel[:num_bin//2, :, :]
        right_voxel = voxel[num_bin//2:, :, :]
        right_voxel_sum = np.cumsum(right_voxel, axis=0)
        left_voxel = left_voxel[::-1]
        left_voxel_sum = np.cumsum(left_voxel, axis=0)
        left_voxel_sum = left_voxel_sum[::-1]
        re_voxel[:num_bin//2, :, :] = -left_voxel_sum
        re_voxel[num_bin//2:, :, :] = right_voxel_sum
        return re_voxel

    def __getitem__(self, index):
        """
        return : blur_image | sharp_image | voxel
        events: used for split the voxel
        """
        item = {}
        if not hasattr(self, 'data_h5'):
            self.open_hdf5()
        item['blur_image'] = self.get_blur_frame(index)
        item['sharp_image'] = self.get_gt_frame(index)
        item['voxel'] = self.get_voxel(index, self.num_bin)
        # roi
        roiTL = (0, 0) # top-left coordinate
        roiBR = (self.roi_size[0],self.roi_size[1]) # bottom-right coordinate
        roiList = ['blur_image','voxel','sharp_image']
        for key in roiList:
            item[key] = item[key][:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]].astype(np.float32)
        return item

# REBlur Dataset
class REBlur(BaseBlur):
    def __init__(self, path, num_bin = 6,voxel_type='SCER',use_raw_points = False,bin_type ='time'):
        self.voxel_type = voxel_type
        self.img_size = (260,320)
        self.roi_size = (248,320) # 168 240
        self.bin_type = bin_type # space or time
        with h5py.File(path, 'r') as file:
            self.dataset_len = len(file['images'].keys())
        super().__init__(path=path, num_bin=num_bin , voxel_type=voxel_type,use_raw_points = use_raw_points)

    def get_blur_frame(self, index):
        """
        return the frame at index [1,w,h]
        """
        img = super().get_blur_frame(index)
        # 1,262,320 -> 1,260,320
        return img[:, :-2, :]

    def get_gt_frame(self, index):
        img = super().get_gt_frame(index)
        # 1,262,320 -> 1,260,320
        return img[:, :-2, :]

    def extract_time_from_index(self, index):
        """
        get img_size,start_time,end_time according to index  
        """
        img_size = self.blur_images['image{:09d}'.format(index)].attrs['size']
        start_time = self.blur_images['image{:09d}'.format(
            index)].attrs['exposure_start']
        end_time = self.blur_images['image{:09d}'.format(
            index)].attrs['exposure_end']
        return img_size, start_time, end_time

    def get_voxel(self, index, num_bin):
        voxel = super().get_voxel(index,num_bin)
        return voxel[:, :-2, :]  # 1,262,320 -> 1,260,320
    