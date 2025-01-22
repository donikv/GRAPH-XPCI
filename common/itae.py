from torchvision.models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# Refer to https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        return self.conv(x)


class ITAEModel(nn.Module):
    """ ITAE model """

    def __init__(self, n_channels=3, bilinear=True):
        super(ITAEModel, self).__init__()
        self.in_conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64)
        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x3_3 = self.down3(x2_3)
        x4_3 = self.down4(x3_3)
        x = self.up1(x4_3, x3_3)
        x = self.up2(x, x2_3)
        x = self.up3(x, x1_3)
        x = self.up4(x, x0_2)
        out = torch.sigmoid(self.out_conv(x))
        return out, None, None

class ITAEncoder(nn.Module):
    """ ITAE model """

    def __init__(self, num_classes=3):
        super(ITAEncoder, self).__init__()
        self.in_conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 256)
        # seems that the structure in paper does not contain 'tanh'

        self.average = nn.AdaptiveMaxPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*256, 384)
        self.linear2 = nn.Linear(384, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x3_3 = self.down3(x2_3)
        x4_3 = self.down4(x3_3)

        y = self.average(x4_3)
        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        y = torch.relu(self.linear2(y))
        out = self.out(y)
        out = F.log_softmax(out, dim=1)

        return out
    
    
class ITAEModelSmall(nn.Module):
    """ ITAE model with half the parameters """

    def __init__(self, n_channels=3, bilinear=True):
        super(ITAEModelSmall, self).__init__()
        self.in_conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up3 = Up(256+128, 128, bilinear)
        self.up4 = Up(128+64, 64)
        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x = self.up3(x2_3, x1_3)
        x = self.up4(x, x0_2)
        out = torch.tanh(self.out_conv(x))
        return out

class ITAEModelTiny(nn.Module):
    """ ITAE model with small convolutions"""

    def __init__(self, n_channels=3, bilinear=True):
        super(ITAEModelTiny, self).__init__()
        self.in_conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 64)
        self.up3 = Up(64+64, 64, bilinear)
        self.up4 = Up(64+32, 32)
        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(32, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x = self.up3(x2_3, x1_3)
        x = self.up4(x, x0_2)
        out = torch.tanh(self.out_conv(x))
        return out, None, None

class ITAEModelTiny2(nn.Module):
    """ ITAE model with small convolutions"""

    def __init__(self, n_channels=1, bilinear=True):
        super(ITAEModelTiny2, self).__init__()
        self.in_conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 64)
        self.up3 = Up1(64, 64, bilinear)
        self.up4 = Up1(64, 32)
        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(32, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x = self.up3(x2_3)
        x = self.up4(x)
        out = torch.sigmoid(self.out_conv(x))
        return out, None, None

    def encoder(self):
        return EncoderITAETiny2(self)

class EncoderITAETiny2(nn.Module):
    def __init__(self, model, features=[0,1,2]) -> None:
        super().__init__()
        self.model = model
        self.scales = [2, 4]
        self.channels = [64, 64]

    def forward(self, x):
        x1 = self.model.in_conv(x)
        x2 = self.model.down1(x1)
        x3 = self.model.down2(x2)
        return [x2,x3]

class ITVAEModelTiny(nn.Module):
    """ ITAE model with small convolutions"""

    def __init__(self, n_channels=1, latent_dim=64, img_size=256, bilinear=True):
        super(ITVAEModelTiny, self).__init__()
        self.in_conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.up2 = Up1(64, 64, bilinear)
        self.up3 = Up1(64, 64, bilinear)
        self.up4 = Up1(64, 32)

        s = img_size // 8
        
        self.fc_mu = nn.Linear(64*s*s, latent_dim)
        self.fc_logvar = nn.Linear(64*s*s, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64*s*s)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (64, s, s))

        # seems that the structure in paper does not contain 'tanh'
        self.out_conv = nn.Conv2d(32, n_channels, kernel_size=3, stride=1, padding=1, bias=False)# Unet use 1*1conv to be out_conv

    def forward(self, x):
        x0_2 = self.in_conv(x)
        x1_3 = self.down1(x0_2)
        x2_3 = self.down2(x1_3)
        x3_3 = self.down3(x2_3)

        h = self.flatten(x3_3)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z = self.fc_decode(z)
        z = self.unflatten(z)

        x = self.up2(z)
        x = self.up3(x)
        x = self.up4(x)
        out = torch.sigmoid(self.out_conv(x))
        return out, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std