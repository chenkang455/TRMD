import torch
import torch.nn as nn
from einops import rearrange

def conv(inDim,outDim,ks,s,p):
# inDim,outDim,kernel_size,stride,padding
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    seq = nn.Sequential(conv, relu)
    return seq

def conv_att(in_dim,out_dim,ks,s,p):
    seq = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim,kernel_size=ks, stride=s, padding=p), nn.Dropout(0.5),nn.BatchNorm2d(out_dim), nn.ReLU())
    return seq

def de_conv(inDim,outDim,ks,s,p,op):
# inDim,outDim,kernel_size,stride,padding,output_padding
    conv_t = nn.ConvTranspose2d(inDim,outDim, kernel_size=ks, stride=s,
                               padding=p, output_padding=op)
    relu = nn.ReLU(inplace=True)
    seq = nn.Sequential(conv_t, relu)
    return seq

def res_block(channel):
    seq = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel)
        )  
    return seq

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
import copy

class SPCA_Attention(nn.Module):
    def __init__(self,dim) :
        super().__init__()
        out_dim = dim
        self.dim = dim
        self.out_dim = out_dim
        # CA
        # blur attention layer
        self.blur_q = conv_att(dim,out_dim,1,1,0)
        self.blur_k = conv_att(dim,out_dim,1,1,0)
        self.blur_v = nn.Conv2d(dim, out_dim, kernel_size = 1,stride=1,padding = 0)
        self.blur_project = nn.Conv2d(out_dim,dim,kernel_size = 1,stride = 1,padding=0)
        # event attention layer
        self.event_q = conv_att(dim,out_dim,1,1,0)
        self.event_k = conv_att(dim,out_dim,1,1,0)
        self.event_v = nn.Conv2d(dim, out_dim, kernel_size=1,stride = 1,padding = 0)
        self.event_project = nn.Conv2d(out_dim, dim, kernel_size=1,stride = 1,padding = 0)
        # SA
        self.gate_blur = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self.gate_event = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,blur,event):
        # blur[bs,c,h,w] event[bs,c,h,w]
        bs = blur.size(0)
        c = self.out_dim
        h = blur.size(2)
        w = blur.size(3)
        # CA
        # blur cal K.Q.V: B,C,N 
        blur_q = self.blur_q(blur).view(bs,c,-1)
        blur_k = self.blur_k(blur).view(bs,c,-1)
        blur_v = self.blur_v(blur).view(bs,c,-1)

        # event cal K:B,C',N' Q:B,C,N V:B,C',N'
        event_q = self.event_q(event).view(bs,c,-1)
        event_k = self.event_k(event).view(bs,c,-1)
        event_v = self.event_v(event).view(bs,c,-1)

        # attention fusion for blur q[b,head,c,h*w] blur_att[b,head,c,c]
        blur_att = torch.matmul(blur_k,event_q.transpose(-2,-1))
        blur_att = (c ** -.5) * blur_att
        blur_att = blur_att.softmax(dim = -1)
        blur_out = torch.matmul(blur_att,blur_v).view(bs,c,h,w)
        blur_out = self.blur_project(blur_out)
        
        # attention fusion for event
        event_att = torch.matmul(blur_q,event_k.transpose(-2,-1))
        event_att = (c ** -.5) * event_att
        event_att = event_att.softmax(dim = -1)
        event_out = torch.matmul(event_att,event_v).view(bs,c,h,w)
        event_out = self.blur_project(event_out)
        blur_out += blur
        event_out += event
        # SA
        cat_fea = torch.cat([blur_out,event_out],dim = 1)
        attention_vector_blur = self.gate_blur(cat_fea)
        attention_vector_event = self.gate_event(cat_fea)
        attention_vector = torch.cat([attention_vector_blur, attention_vector_event], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_blur, attention_vector_event = attention_vector[:, :self.dim, :, :], attention_vector[:, self.dim:, :, :]
        fusion = blur_out * attention_vector_blur + event_out * attention_vector_event
        return fusion


if __name__ == "__main__":
    # net = UNet()
    img = torch.ones(1, 32, 360,640)
    events = torch.ones(1, 32, 360,640)
    mul_att = SPCA_Attention(32)
    print(mul_att(img,events))