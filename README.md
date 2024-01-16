# Motion-Deblur-by-Learning-Residual-from-Events
This is an official PyTorch implementation of [Motion Deblur by Learning Residual from Events]() to be published in **TMM 2024**.

**Authors:** Kang Chen and [Lei Yu✉️](http://eis.whu.edu.cn/index/szdwDetail?rsh=00030713&newskind_id=20160320222026165YIdDsQIbgNtoE) from Wuhan university, Wuhan, China.
## Method overview
In this paper, we propose a **Two-stage Residual-based Motion Deblurring (TRMD)** framework for an event camera, which converts a blurry image into a sequence of sharp images, leveraging the abundant motion features encoded in events.

![img](Img/framework.png)

## Quickstart
### 1. Setup environment
```
git clone https://github.com/chenkang455/TRMD
cd TRMD
pip install -r requirements.txt
```
### 2. Download datasets
You can download our trained models, synthesized dataset GOPRO and real event dataset REBlur (from [EFNet](https://github.com/AHupuJR/EFNet)) from [Baidu Netdisk](https://pan.baidu.com/s/1advngktF3hiHzLO_fs6E0w?pwd=e1uc) with the password ```eluc```. 

Unzip the ```GOPRO.zip``` file before placing the downloaded models and datasets (path defined in [config.yaml](https://github.com/chenkang455/TRMD/blob/main/config.yaml)) according to the following directory structure:
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
Change the data path and other parameters (if needed) in [config.yaml](https://github.com/chenkang455/TRMD/blob/main/config.yaml). 

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


<!-- ### 5. Training
To train our model from scratch on **GRAY-GOPRO**:
```
python train_GoPro.py --rgb False --save_path Model/RE_Net_GRAY.pth
```
To train our model from scratch on **RGB-GOPRO**:
```
python train_GoPro.py --rgb True --save_path Model/RE_Net_RGB.pth
``` -->

### 5. Results
<details><summary>GoPro dataset (Click to expand) </summary>
<img src="Img/gopro.png" alt="gopro_table" style="zoom:100%;" />
</details>

<details><summary>REBlur dataset (Click to expand) </summary>
<img src="Img/reblur.png" alt="reblur_table" style="zoom:100%;" />
</details>

## Contact
Should you have any questions, please feel free to contact [mrchenkang@whu.edu.cn](mailto:mrchenkang@whu.edu.cn) or [ly.wd@whu.edu.cn](mailto:ly.wd@whu.edu.cn).

## Acknowledgment

Our event representation (SCER) code and REBlur dataset are derived from [EFNet](https://github.com/AHupuJR/EFNet). Some of the code for metric testing and module construction is from [E-CIR](https://github.com/chensong1995/E-CIR). We appreciate the effort of the contributors to these repositories.
