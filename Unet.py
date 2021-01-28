import torch
import torch.nn as nn
import torch.nn.functional as F
from convrf.convrf.convrf import Conv2dRF


def double_conv_RF(in_channels, out_channels):
    return nn.Sequential(
        Conv2dRF(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, fbank_type="frame",kernel_drop_rate=.4),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        Conv2dRF(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, fbank_type="frame",kernel_drop_rate=.4),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2)
        
    )

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2)

    )
    
class UNet(nn.Module):

    def __init__(self, n_class):
        super(UNet, self).__init__()
        
        self.dconv_down1 = double_conv(1, 32)       # number channels
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv_RF(64, 128)
        self.dconv_down4 = double_conv_RF(128, 256)

        self.maxpool = nn.MaxPool2d((2,2))

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, n_class,1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = F.interpolate(input=x, scale_factor=2, mode='nearest',) 
        x = torch.cat((F.interpolate(input=x, scale_factor = 1, mode ='nearest',), conv3), dim=1) 

        x = self.dconv_up3(x)
        x = F.interpolate(input=x, scale_factor=2, mode='nearest',) 
        x = torch.cat((F.interpolate(input=x, scale_factor = 1, mode ='nearest',), conv2), dim=1)

        x = self.dconv_up2(x)
        x = F.interpolate(input=x, scale_factor=2, mode='nearest',)
        x = torch.cat((F.interpolate(input=x, scale_factor = 1, mode ='nearest',), conv1), dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
