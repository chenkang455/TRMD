import torch.nn as nn
class LOSS:
    def __init__(self):
        self.loss = nn.MSELoss()
    def __call__(self, res_sharp, res_pre,sharp_image,output_image):
        # sharp_image: [bs,num_bin,w,h]
        # pre_image: [bs,num_bin,w,h]
        deblur_loss = self.loss(output_image,sharp_image)
        res_loss = self.loss(res_sharp,res_pre)
        loss = res_loss + deblur_loss
        return loss
    