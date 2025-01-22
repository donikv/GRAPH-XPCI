import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_wavelets import ScatLayer

class TanhBlock(nn.Module):
    def __init__(self, in_kernels, out_kernels, size=3, stride=1, pooling=2) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(in_kernels, out_kernels, size, stride, padding='same')
        self.maxpool = nn.MaxPool2d(pooling, pooling)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.tanh(y)
        y = self.maxpool(y)
        return y

class ReluBlockBN(nn.Module):
    def __init__(self, in_kernels, out_kernels, size=3, stride=1, pooling=2) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_kernels)
        self.conv = nn.Conv2d(in_kernels, out_kernels, size, stride, padding='same')
        self.maxpool = nn.MaxPool2d(pooling, pooling)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

class ReluBlock(nn.Module):
    def __init__(self, in_kernels, out_kernels, size=3, stride=1, pooling=2) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_kernels, out_kernels, size, stride, padding='same')
        self.maxpool = nn.MaxPool2d(pooling, pooling)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

class ReluBlockCBAM(nn.Module):
    def __init__(self, in_kernels, out_kernels, size=3, stride=1, pooling=2, preactivation=False) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_kernels)
        self.conv = nn.Conv2d(in_kernels, out_kernels, size, stride, padding='same')
        self.maxpool = nn.MaxPool2d(pooling, pooling)
        self.cbam = CBAM(out_kernels)
        self.preactivation = preactivation
    
    def forward(self, x):
        y = self.conv(x)

        if self.preactivation:
            y1 = self.cbam(y)
            y1 = self.bn(y1)
            y1 = self.relu(y1)
            y = y1 + y
        else:
            y = self.cbam(y) + y
            y = self.bn(y)
            y = self.relu(y)

        y = self.maxpool(y)
        return y

class TanhBlockBN(nn.Module):
    def __init__(self, in_kernels, out_kernels, size=3, stride=1, pooling=2) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(in_kernels, out_kernels, size, stride, padding='same')
        self.maxpool = nn.MaxPool2d(pooling, pooling)
        self.bn = nn.BatchNorm2d(out_kernels)
    
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.tanh(y)
        y = self.maxpool(y)

        return y

class ScatBlock(nn.Module):

    def __init__(self, upsample=False) -> None:
        super().__init__()
        self.scat = ScatLayer()
        self.relu = nn.ReLU()
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else None
    
    def forward(self, x):
        if self.upscale is not None:
            x = self.upscale(x)
        x = self.scat(x)
        x = self.relu(x)
        return x

class OutBlock(nn.Module):

    def __init__(self, in_dim, num_classes, regression) -> None:
        super().__init__()
        if regression:
            self.width = nn.Parameter(torch.tensor(1.0))
            self.out = nn.Linear(in_dim, 1)
        else:
            self.out = nn.Linear(in_dim, num_classes)
        
        self.regression = regression
        self.num_classes = num_classes
    
    def forward(self, x):
        out = self.out(x)
        if self.regression:
            scale = (self.num_classes)
            out = torch.tanh(out / self.width) * scale/2 + (scale/2) - 0.5
            # out = torch.sigmoid(out / self.width) * scale - 0.5
        else:
            out = out
        return out

class ChannelAttention(nn.Module):

    def __init__(self, in_kernels, reduction_ratio=8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_kernels, in_kernels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_kernels // reduction_ratio, in_kernels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = self.flatten(avg_pool)
        max_pool = self.flatten(max_pool)
        avg_pool = self.fc2(self.relu(self.fc1(avg_pool)))
        max_pool = self.fc2(self.relu(self.fc1(max_pool)))

        out = self.sigmoid(avg_pool + max_pool)
        out = out.unsqueeze(2).unsqueeze(3)
        return out

class SpatialAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            avg_pool = torch.mean(x, dim=1, keepdim=True)
            max_pool, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_pool, max_pool], dim=1)
            out = self.conv(out)
            out = self.sigmoid(out)
            return out

class CBAM(nn.Module):

    def __init__(self, in_kernels, reduction_ratio=8) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_kernels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa
        return x
