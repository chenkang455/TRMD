import torch
import torch.nn as nn
from modules import conv,res_block,de_conv,SPCA_Attention
from thop import profile

class RE_Net(nn.Module):
    def __init__(self,
                 blurry_channels=1,
                 event_channels=6,
                 out_channels = 6,
                 rgb = False):
        """  Structure of the Residual Estimation Network (RE-Net).
        Args:
            blurry_channels (int): The channel of the input blurry image. Defaults to 1.
            event_channels (int): The channel of the input events (SCER). Defaults to 6.
            out_channels (int): The channel of the output residual sequence. Defaults to 6.
            rgb (bool): Input image is rgb or grayscale. Defaults to False.
        """
        super(RE_Net, self).__init__()
        # RGB version 
        if rgb:
            out_channels = out_channels * 3
            blurry_channels = blurry_channels * 3
        # ablation study 
        self.use_feature_extraction = True # FEM
        self.use_fusion = True # FFM
        self.use_events = True # use_event
        self.use_blur = True # use_blur
        # network parameters
        in_channels = 0
        num = 64 # for residual estimation module
        s_num = 16 # for blur and event encoder
        # blur_encoder
        self.blur_en_conv1 = conv(blurry_channels,s_num,5,1,2)
        self.blur_en_conv2 = conv(s_num,2 * s_num,5,1,2)
        # event_encoder
        self.event_en_conv1 = conv(event_channels,s_num,5,1,2)
        self.event_en_conv2 = conv(s_num,2 * s_num,5,1,2)

        # fusion_encoder
        if self.use_feature_extraction == False:
            print("Feature Extraction Module (FEM) not used")
            if self.use_events == True:
                print("Events used")
                in_channels += event_channels
            if self.use_blur == True:
                print("Blur used")
                in_channels += blurry_channels
        else:
            print("Feature Extraction Module (FEM) used")
            if self.use_fusion == True:
                # spatial attention and channel attention
                self.sca = SPCA_Attention(2 * s_num)
                print("Feature Fusion Module (FEM) used")
                in_channels = 2 * s_num
            else:
                in_channels = 4 * s_num
        # REM-Encoder
        self.en_conv1 = conv(in_channels , num,5,2,2)
        self.en_conv2 = conv(num,2 * num,5,2,2)
        self.en_conv3 = conv(2 * num,4 * num,5,2,2)
        self.en_conv4 = conv(4 * num,8 * num,5,2,2)
        # REM-ResBlock
        self.res1 = res_block(8 * num)
        self.relu1 = nn.ReLU(inplace=True)
        self.res2 = res_block(8 * num)
        self.relu2 = nn.ReLU(inplace=True)
        # REM-Decoder
        self.de_conv1 = de_conv(8 * num, 4 * num,5,2,2,(0,1))
        self.de_conv2 = de_conv(4 * num + 4 * num, 2 * num,5,2,2,(1,1))
        self.de_conv3 = de_conv(2 * num + 2 * num ,  num,5,2,2,(1,1))
        self.de_conv4 = de_conv(num + num ,  num//2,5,2,2,(1,1))
        # Residual Connection
        if self.use_blur == True and self.use_events == True:
            self.pred = nn.Conv2d(
                num//2 + blurry_channels + event_channels , out_channels, kernel_size=1, stride=1)
        elif self.use_blur == True:
            self.pred = nn.Conv2d(
                num//2 + blurry_channels, out_channels, kernel_size=1, stride=1)
        elif self.use_events == True:
            self.pred = nn.Conv2d(
                num//2 + event_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, blurry_frame, event_map):
        """
        Args:
            blurry_frame: [bs,1,w,h] if rgb == False else [bs,3,w,h]. The input blurry image.
            event_map: [bs,num_bin,w,h]. The input event stream (SCER format in this paper).
            output_frame: [bs,num_bin,w,h]. The output residual sequence.
        """
        # assemble input tensor
        x_in = []
        if self.use_feature_extraction == True:
            blur1 = self.blur_en_conv1(blurry_frame)
            event1 = self.event_en_conv1(event_map)
            blur2 = self.blur_en_conv2(blur1)
            event2 = self.event_en_conv2(event1)
            if self.use_fusion:
                x_in = self.sca(blur2,event2)
            else:
                x_in = torch.cat([blur2,event2],dim = 1)
        else:
            if self.use_blur == True and self.use_events == False:
                x_in = blurry_frame
            elif self.use_blur == False and self.use_events == True:
                x_in = event_map
            elif self.use_blur == True and self.use_events == True:
                x_in = torch.cat([blurry_frame,event_map],dim = 1)
        # pass to the network
        x1 = self.en_conv1(x_in) # 260 - > 130 720、1280
        x2 = self.en_conv2(x1) # 65  360 640
        x3 = self.en_conv3(x2) # 33  180 320
        x4 = self.en_conv4(x3) # 17  90 160
        # x4 = self.attention(x4)
        x5 = self.relu1(self.res1(x4) + x4) # 17 45 80
        x6 = self.relu2(self.res2(x5) + x5) # 17 45 80
        x7 = self.de_conv1(x6) # 33 
        x8 = self.de_conv2(torch.cat([x7, x3], dim=1)) # 65 
        x9 = self.de_conv3(torch.cat([x8, x2], dim=1))
        x10 = self.de_conv4(torch.cat([x9, x1], dim=1))
        if self.use_blur == True and self.use_events == True:
            x_out = self.pred(torch.cat([x10, blurry_frame,event_map], dim=1))
        elif self.use_blur == True:
            x_out = self.pred(torch.cat([x10, blurry_frame], dim=1))
        elif self.use_events == True:
            x_out = self.pred(torch.cat([x10, event_map], dim=1))
        return x_out



if __name__ == "__main__":
    net = RE_Net()
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('Pretrained_Model/RE_Net_GRAY.pth')['state_dict'])
    img = torch.ones(1, 1, 360,640)
    events = torch.ones(1, 6, 360,640)
    flops, params = profile((net.module).cpu(), inputs=(img,events))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.2fM" % (total/1e6))