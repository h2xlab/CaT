import torch
from torch import nn 
from torch.nn import functional as F
from .resnet import resnet18, resnet34
from .segmentation import SegmentationHead
from .unet import UNet
from .attention import Attention
from .erfnet import ERFNet

class Normalize(nn.Module):
    """ ImageNet normalization """
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False).to(device)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False).to(device)

    def forward(self, x):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]

class RGBModel(nn.Module):
    def __init__(self, seg_channels, pretrained=True):
        super().__init__()
        self.num_channels = len(seg_channels)

        self.backbone = resnet18(pretrained=pretrained)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.head = None

    def forward(self, rgb):

        embd = self.backbone(self.normalize(rgb/255.))

        return self.head(embd).squeeze(-1)

class RGBSegmentationModel(nn.Module):
    def __init__(self, seg_channels):

        super().__init__()

        self.erfnet = ERFNet(len(seg_channels)+1)
        self.normalize = lambda x: (x/255.-.5)*2

    def forward(self, rgb):

        return self.erfnet(self.normalize(rgb))

class RGBBrakePredictionModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv_backbone = resnet18(pretrained=pretrained)

        self.attn1 = Attention(512, num_heads=8)
        self.attn2 = Attention(512, num_heads=8)

        self.classifier = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, rgb1, rgb2):

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=rgb1.device)

        x1 = self.conv_backbone(self.normalize(rgb1/255.))
        x2 = self.conv_backbone(self.normalize(rgb2/255.))

        h1 = self.attn1(x1)
        h2 = self.attn2(x2)
        
        pred_bra = self.classifier(torch.cat([h1,h2], dim=1))
        return pred_bra[:,0]
