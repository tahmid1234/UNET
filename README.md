# Visualizing What U-Net Actually Sees

The best way to learn U-Net is to visualize what each section is doing. For that reason, I used a single image and trained the U-Net architecture to overfit it. Next, I generated images from the output of each operation. However, the images are not exactly what the model sees, as the outputs have channels like 128, 64, 32, etc. Standard image formats and displays are designed around grayscale images (1 channel) or RGB images (3 channels), so I converted the higher-dimensional feature maps into something visualizable - I averaged the high-dimensional value of each pixel to generate an image that both a computer and a human eye can comprehend. Worth being clear about this: the averaged image is only a projection of the feature representation into something viewable — different channels can encode completely different features, and averaging them can cause positive and negative activations to cancel out or hide detail. It's not a literal picture of what the network "sees." (A useful follow-up would be visualizing individual channels separately instead of just the average.)

Before diving into details, let's introduce what a U-Net architecture is.

A U-Net architecture has two sections: the encoder (for downsampling) and the decoder (for upsampling).

Downsampling typically decreases the spatial resolution (number of pixels) while increasing the number of channels (features per pixel). The CNN blocks in this section try to learn edges, corners, and eventually shapes. The decoder then tries to bring back the information that might have been lost during downsampling by using skip connections. In this U-Net, the skip connection concatenates the encoder's feature map with the corresponding upsampled decoder feature map along the channel dimension (skip connections don't have to mean concatenation in general — that's just how this implementation does it). Upsampling decreases the dimensions and increases the pixel count to regenerate the image so it can match the ground truth.

## The Code

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # stabilizer — normalizes each channel using the mean and variance computed over the batch and spatial dimensions during training; at inference it uses running statistics instead
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
# Why BatchNorm then ReLU? ReLU turns negative numbers into 0. The normalization would have to work with partial information if ReLU came before it.
# Does ReLU deleting information matter? Once ReLU zeroes a value, that specific value is gone — there's no guarantee it's recovered elsewhere.
# But the network can learn to build other channels or weights that carry the information it needs, in ways that work despite this loss.
# ReLU's real job is introducing nonlinearity and suppressing negative activations, not "cancelling noise."

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 32) # b,p,3 becomes b,p,32
        self.pool1 = nn.MaxPool2d(2) # b,p,32 becomes b,p/2,32
        self.down2 = DoubleConv(32, 64)  # b,p,32 becomes b,p,64
        self.pool2 = nn.MaxPool2d(2) # two pools because I wanted two levels of downsampling before the bottleneck, not one
        self.bottleneck = DoubleConv(64, 128)  # b,p,64 becomes b,p,128

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # shape b, p*4, 64
        self.up1 = DoubleConv(128, 64) # 128 because of skip connection (64+64)
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
        # skip connection: expands the figure and adds back the details lost during bottleneck downsampling
        concat1 = torch.cat([features['3_down2_conv'], features['6_up1conv']], dim=1)
        features['7_up1_conv'] = self.up1(concat1)
        features['8_up2conv'] = self.upconv2(features['7_up1_conv'])
        concat2 = torch.cat([features['1_down1_conv'], features['8_up2conv']], dim=1)
        features['9_up2_conv'] = self.up2(concat2)

        out = self.out_conv(features['9_up2_conv'])
        features['final_output'] = self.sigmoid(out)
        return features
```

## How I Understand What's Happening Inside

### Conv2d (the "going down" step)

`nn.Conv2d(3, 64, kernel=3)` — let's say the pixel count is 512.

1. For each output channel, the kernel is effectively 3×3×3, where the third 3 comes from the 3 input channels.
2. The 3x3 kernel slides across the spatial dimensions (height and width), while spanning all 3 input channels at each position — it doesn't slide separately through the channel dimension.
3. At a time, it covers 9 pixels, and each pixel has 3 values — that's 27 multiplications at a time.
4. All those 27 products get summed together (plus a bias), and 9 pixels with 3 dimensions each collapse into one unified value for one output channel. This repeats with a different set of learned weights for each of the 64 output channels — it's not just "27 values become 1 value," it's "one 3x3x3 region combined with 64 different weight sets produces 64 separate output values."
5. Here we use stride=1, but stride alone isn't what keeps the pixel count the same — it's the padding doing that job. The kernel moves horizontally: if it processes pixels 0,0 to 2,2 first, next it processes 0,1 to 2,3.
6. Since out_channel = 64, there will be 64 such unified scores per pixel.
7. So the shape would be 64, pixel_count.
8. And after swapping, pixel_count, 64.
9. padding = (kernel - 1) / 2 — this is the specific padding that keeps spatial size unchanged for stride=1. (Conv2d can technically change spatial size in other directions too, depending on padding/stride/dilation, but keeping it the same for stride=1 is what this formula is for.)
10. To keep the pixel count consistent under stride=1, this amount of padding is required.

*(A note on terms: I use "pixel_count" here as a flattened H×W for simplicity, but the real tensor shape PyTorch works with is `[B, C, H, W]` — worth keeping in mind if H and W aren't equal.)*

### ConvTranspose2d (the "going back up" step)

For `nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)`, the weight shape is actually `[128, 64, 2, 2]` — meaning for **each of the 64 output channels**, there are **128 separate 2x2 kernels**, one per input channel. It's not "one kernel per input channel" shared across outputs — every output channel has its own full set of 128 kernels.

1. At one input pixel, take its 128 channel values.
2. For a given output channel, multiply each of those 128 values by its own learned 2x2 kernel, then sum all 128 resulting 2x2 patches together into a single 2x2 patch. That's this input pixel's contribution to this one output channel.
3. Repeat step 2 independently for all 64 output channels, each using its own set of 128 kernels.
4. Because `kernel_size=2` and `stride=2` are equal here, these 2x2 patches from neighboring input pixels don't overlap — they tile the output cleanly, side by side, with no interference between neighbors. (This is a special, non-overlapping case. If stride and kernel size were different, patches from neighboring input pixels would overlap and get added together instead of just placed side by side — that's the general behavior of a transposed convolution.)
5. For one input pixel, we can conceptually visualize its contribution to the output as 64 separate 2×2 patches — one for each output channel.
6. Since CNNs are channel-first, we swap it and it becomes 256, 64 — meaning 256 pixels now have 64 dimensions each. The image has expanded.
7. Pixel count calculation for this specific configuration (kernel=stride=2, padding=0): if there are 64 pixels, meaning 8x8, the side count = (8-1)\*stride + kernel = 7\*2 + 2 = 16. Total = 16x16. (This formula only holds for this exact setup — the general ConvTranspose2d formula also accounts for padding, dilation, and output_padding.)
8. If stride is larger than kernel size, some output positions end up receiving no contribution from any input pixel and stay at zero. That's a property of which positions get written by this scatter-and-accumulate process — not that PyTorch pre-fills a blank canvas first.

### The final 1x1 conv

`self.out_conv = nn.Conv2d(32, 3, kernel_size=1)`

Say there are 256 pixels, and each pixel has 32 values. Kernel = 1 means a 1x1x32 shape, so it only covers 1 pixel at a time. The 32 values get multiplied by their corresponding kernel values and summed together, so 1 pixel now has 1 value. So 256 pixels have 256 values, and for 3 output channels, the final shape is 3, 256.

### Conv2d vs ConvTranspose2d

In typical CNN/U-Net usage, Conv2d is used to maintain or reduce spatial size, while ConvTranspose2d is specifically designed to learn an increase in spatial size. (Technically Conv2d *can* grow spatial size with large enough padding, but that's not how it's normally used — the padding formula above is there to keep size the same, not grow it.) With kernel=3, stride=1, padding=0, the figure loses 2 columns and 2 rows. With stride=2, the kernel moves two pixels at a time, so the output is often roughly half the spatial size. The exact output size also depends on kernel size and padding.

You could use Conv2d as an alternative to ConvTranspose2d by upsampling the image first (nearest-neighbor or bilinear) and then applying Conv2d — this is actually a common alternative in practice. The choice of upsampling method matters though; it's not mathematically equivalent to a transposed convolution. Nearest-neighbor and bilinear upsampling produce meaningfully different results. 

## Visualizing Each Stage

The model was trained on a single image for 200 epochs to overfit it. During inference, I ran the same image through and saved the output at each operation, then averaged the channel values down to something displayable (3 channels or 1), since a screen can't render a 64- or 128-channel image directly.

| Stage | Image |
|---|---|
| Original input | ![original input](images/original.png) |
| Downsample 1 (`down1` output) | ![downsample 1](images/downsample1.png) |
| Pool 1 | ![pool 1](images/pool1.png) |
| Downsample 2 (`down2` output) | ![downsample 2](images/downsample2.png) |
| Pool 2 | ![pool 2](images/pool2.png) |
| Bottleneck | ![bottleneck](images/bottleneck.png) |
| Upsample 1 (`upconv1` output) | ![upsample 1](images/upsample1.png) |
| Up 1 (`up1` output, after skip concat) | ![up 1](images/up1.png) |
| Upsample 2 (`upconv2` output) | ![upsample 2](images/upsample2.png) |
| Up 2 (`up2` output, after skip concat) | ![up 2](images/up2.png) |
| Final output | ![final output](images/final_output_3ch.png) |

## A Note on the Final Sigmoid

`self.out_conv` outputs 3 channels, followed by `nn.Sigmoid()`. This makes sense for a reconstruction task like this one, where the target pixels are normalized to [0,1] — but it wouldn't be the right choice for something like multi-class segmentation, where you'd typically leave the raw logits unsquashed and let a loss like `CrossEntropyLoss` handle the class normalization internally. Worth keeping in mind if you adapt this code for a different task.