import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), #stabalizer, across the batch, the values for each channel have a mean of 0 and ds of 1
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
# Why BatchNorm then ReLU? ReLU transforms negetive number into 0. The normalization will have to work with partial information if ReLU comes before it
# Why does it not matter that ReLU deletes information? The network quickly learns that ReLU kills negetive numbers. So, if a negetive feature is important
# The network uses a different channel to flip the number by multiplying with a negetive weight to prevent information getting lost.
# ReLU is important as it cancels noises.

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 32) # b,p,3 becomes b,p, 32
        self.pool1 = nn.MaxPool2d(2) # b,p, 32 becomes b,p/2, 32
        self.down2 = DoubleConv(32, 64)  # b,p,32 becomes b,p, 64
        self.pool2 = nn.MaxPool2d(2) # 1 single pool would be enough as max pool don't have learned weight. But I followed conventions.
        self.bottleneck = DoubleConv(64, 128)  # b,p,64 becomes b,p, 128

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # shape b, p*4, 64
        self.up1 = DoubleConv(128, 64) # 128 because skip connection (64+64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # shape b, p*4, 32
        self.up2 = DoubleConv(64, 32) # 64 because of skip connection (32+32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.extract_features(x)
        return features['final_output']

    def extract_features(self, x):
        features = {}

        features['1_down1_conv'] = self.down1(x)
        features['2_pool1'] = self.pool1(features['1_down1_conv'])
        features['3_down2_conv'] = self.down2(features['2_pool1'])
        features['4_pool2'] = self.pool2(features['3_down2_conv'])
        features['5_bottleneck'] = self.bottleneck(features['4_pool2'])

        features['6_up1conv'] = self.upconv1(features['5_bottleneck'])
        # skip (expanded the figure and added the lost details from the from bottleneck downsampling)
        concat1 = torch.cat([features['3_down2_conv'], features['6_up1conv']], dim=1)
        features['7_up1_conv'] = self.up1(concat1)
        features['8_up2conv'] = self.upconv2(features['7_up1_conv'])
        concat2 = torch.cat([features['1_down1_conv'], features['8_up2conv']], dim=1)
        features['9_up2_conv'] = self.up2(concat2)

        out = self.out_conv(features['9_up2_conv'])
        features['final_output'] = self.sigmoid(out)
        return features