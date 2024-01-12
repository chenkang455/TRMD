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
You can download the Event-Version GOPRO Dataset and Real World Blur (REBlur) Dataset from [here].

### 3. Configs

### 4. Test the metric/Visualize the deblurred image with our pre-trained model

### 5. Training


## Acknowledgment

Our event representation (SCER) code is derived from [EFNet](https://github.com/AHupuJR/EFNet). We appreciate the effort of the contributors to these repositories.
