# Motion-Deblur-by-Learning-Residual-from-Events
This is an official PyTorch implementation of "Motion Deblur by Learning Residual from Events"  (TMM 2024).

In this paper, we propose a **Two-stage Residual-based Motion Deblurring (TRMD)** framework for an event camera, which converts a blurry image into a sequence of sharp images, leveraging the abundant motion features encoded in events.

## Method overview
![img](https://github.com/chenkang455/Motion-Deblur-by-Learning-Residual-from-Events/assets/72788314/5feb49ae-f32d-4710-a249-e2b60c7ae842)

## Quickstart
### 1. Setup environment

```
git clone https://github.com/chenkang455/TRMD
cd TRMD
pip install -r requirements.txt
```
### 2. Download datasets
You can download our trained models, synthesized dataset GOPRO and real event dataset REBlur (from [EFNet](https://github.com/AHupuJR/EFNet)) from [Google Drive](). 

Place the downloaded models and datasets (path defined in config.yaml) according to the following directory structure:
```                                                                                            
├── Data                                                                                                                                                            
│   ├── GOPRO                                                                                              
│   │   └── train                                                                                                                             
│   │   └── test                                                                                    
|   ├── REBlur
|   |   └── trian
|   |   └── test   
|   |   └── addition
|   |   └── README.md 
├── Pretrained_Model
│   ├── RE_Net.pth 
│   ├── RE_Net_rgb.pth 
├── config.yaml
├── ...
```

### 3. Configs
Change the data path and other parameters (if needed) in ```config.yaml```. 

### 4. Test with our pre-trained models
* To test the metric and visualize the deblurred result on **GRAY-GOPRO**:
```
python test_GoPro.py --rgb False --load_path Pretrained_Model/RE_Net_GRAY.pth
```
* To test the metric and visualize the deblurred result on **RGB-GOPRO**:
```
python test_GoPro.py --rgb True --load_path Pretrained_Model/RE_Net_RGB.pth
```
* To visualize the deblurred result on **REBlur**:
```
python test_REBlur.py --load_path Pretrained_Model/RE_Net_GRAY.pth
```
* To test our model size and FLOPs:
```
python network.py 
```


### 5. Training
To train our model from scratch on **GRAY-GOPRO**:
```
python train_GoPro.py --rgb False --save_unet_path Model/RE_Net_GRAY.pth
```
To train our model from scratch on **RGB-GOPRO**:
```
python train_GoPro.py --rgb True --save_unet_path Model/RE_Net_RGB.pth
```

## Acknowledgment

Our event representation (SCER) code and REBlur dataset is derived from [EFNet](https://github.com/AHupuJR/EFNet). We appreciate the effort of the contributors to these repositories.
