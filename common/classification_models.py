import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_wavelets import ScatLayer
from common.modules import *


class ClassificationModelSmall(nn.Module):

    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.tanh1 = TanhBlock(1, 64, 5)
        self.tanh2 = TanhBlock(64, 128, 3)
        self.tanh3 = TanhBlock(128, 256, 1)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*256, 64)
        self.out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)
        out = F.log_softmax(out, dim=1)

        return out


class Encoder(nn.Module):
    def __init__(self, model, features=[0,1,2]) -> None:
        super().__init__()
        self.model = model
        self.scales = [2, 4, 8]
        self.channels = [32, 64, 32]

    def forward(self, x):
        x1 = self.model.tanh1(x)
        x2 = self.model.tanh2(x1)
        x3 = self.model.tanh3(x2)
        return [x1,x2,x3]

from torchvision import models

class VariableEncoderClassificationModelNew(nn.Module):
    def __init__(self, num_classes=4, regression=False, encoder_type='resnet18', rgb=False):
        super().__init__()

        if encoder_type == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            if not rgb:
                self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove the last fc layer
        elif encoder_type == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            if not rgb:
                self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            num_features = self.encoder.classifier[6].in_features
            self.encoder.classifier[6] = nn.Identity()  # Remove the last fc layer
        elif encoder_type == 'efficientnet-b0':
            self.encoder = models.efficientnet_b0(pretrained=True)
            if not rgb:
                self.encoder._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            self.num_features = self.encoder.classifier.in_features
            self.encoder.classifier = self.encoder.classifier[0]  # Remove the last fc layer
        else:
            raise NotImplementedError(f'Encoder type {encoder_type} not implemented')

        self.average = nn.AdaptiveAvgPool2d((32, 32))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_features, 64)
        
        self.out = OutBlock(64, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        x = self.encoder(x)
        # y = self.average(x)        
        # y = self.flatten(y)
        y = torch.relu(self.linear(x))
        out = self.out(y)
        
        return out

class VariableEncoderClassificationModel(nn.Module):
    def __init__(self, num_classes=4, regression=False, encoder_type='resnet18'):
        super().__init__()

        if encoder_type == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove the last fc layer
        elif encoder_type == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            num_features = self.encoder.classifier[6].in_features
            self.encoder.classifier[6] = nn.Identity()  # Remove the last fc layer
        elif encoder_type == 'efficientnet-b0':
            self.encoder = models.efficientnet_b0(pretrained=True)
            self.encoder.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            self.num_features = self.encoder.classifier[1].in_features
            self.encoder.classifier = self.encoder.classifier[0]  # Remove the last fc layer
        else:
            raise NotImplementedError(f'Encoder type {encoder_type} not implemented')

        self.linear = nn.Linear(self.num_features, 64)

        if regression:
            self.width = nn.Parameter(torch.tensor((num_classes - 1) / 2))
            self.out = nn.Linear(64, 1)
        else:
            self.out = nn.Linear(64, num_classes)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        x = self.encoder(x)
        # y = self.average(x)        
        # y = self.flatten(y)
        y = torch.relu(self.linear(x))
        out = self.out(y)

        if self.regression:
            scale = (self.num_classes - 1) / 2
            out = torch.tanh(out / self.width) * scale + (scale)
        else:
            out = F.log_softmax(out, dim=1)
        return out

class ClassificationModelTiny(nn.Module):

    def __init__(self, num_classes=4, regression=False, bn=False) -> None:
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(1, 32, 11)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 32, 3)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*32, 64)

        if regression:
            self.width = nn.Parameter(torch.tensor((num_classes - 1) / 2))
            self.out = nn.Linear(64, 1)
        else:
            self.out = nn.Linear(64, num_classes)

        self.regression = regression
        self.num_classes = num_classes

    def encoder(self):
        return Encoder(self)

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)

        if self.regression:
            scale = (self.num_classes - 1) / 2
            # out = torch.sigmoid(out / self.width * 4) * (self.num_classes-1)
            out = torch.tanh(out / self.width) * scale + (scale)
        else:
            out = F.log_softmax(out, dim=1)
        return out

class ClassificationModelTinyAttentionNewSmallImages(nn.Module):

    def __init__(self, in_channels=1, num_classes=4, regression=False, bn=False) -> None:
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 7)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 512, 3)
        self.average = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 64)

        self.out = OutBlock(64, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def encoder(self):
        return Encoder(self)

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)

        return out

class ClassificationModelTinyNewSmallImages(nn.Module):
#BECAUSE OF A BUG, THIS ONE HAS THE ATTENTION AND THE ATTENTION ONE DOES NOT

    def __init__(self, in_channels=1, num_classes=4, regression=False, bn=False) -> None:
        super().__init__()
        block = ReluBlockBN if bn else ReluBlock
        self.tanh1 = block(in_channels, 32, 7)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 512, 3)
        self.average = nn.AdaptiveAvgPool2d((1, 1))
        
        self.attention = nn.Conv2d(64, 1, 3, stride=1, padding='same')
        self.max2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 64)

        self.out = OutBlock(64, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def encoder(self):
        return Encoder(self)

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        att = F.sigmoid(self.max2(self.attention(y)))
        y = self.tanh3(y)
        y = y * att
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)

        return out
    
class ScatClassificationModelTinyNewSmallImages(ClassificationModelTinyNewSmallImages):
    def __init__(self, in_channels=1, num_classes=4, regression=False, bn=False, upsample=True) -> None:
        super().__init__(in_channels*7, num_classes, regression, bn)
        self.scat = ScatBlock(upsample=upsample)
    
    def forward(self, x):
        x = self.scat(x)
        return super().forward(x)

class ScatClassificationModelTinyAttentionNewSmallImages(ClassificationModelTinyAttentionNewSmallImages):
    def __init__(self, in_channels=1, num_classes=4, regression=False, bn=False, upsample=True) -> None:
        super().__init__(in_channels*7, num_classes, regression, bn)
        self.scat = ScatBlock(upsample=upsample)
    
    def forward(self, x):
        x = self.scat(x)
        return super().forward(x)

class ClassificationModelTinyAttention(nn.Module):

    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1) -> None:
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 11)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 32, 3)
        self.attention = nn.Conv2d(64, 1, 3, stride=1, padding='same')
        self.max2 = nn.MaxPool2d(2, 2)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*32, 64)

        if regression:
            self.width = nn.Parameter(torch.tensor((num_classes - 1) / 2))
            self.out = nn.Linear(64, 1)
        else:
            self.out = nn.Linear(64, num_classes)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        a = self.attention(y)
        y = self.tanh3(y)
        y = y * torch.sigmoid(self.max2(a)) #attention
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)

        if self.regression:
            scale = (self.num_classes) / 2
            # out = torch.sigmoid(out / self.width * 4) * (self.num_classes-1)
            out = torch.tanh(out / self.width) * scale + (scale) - 0.5
        else:
            out = F.log_softmax(out, dim=1)
        return out

class ClassificationModelTinyAttentionNew(nn.Module):

    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1, dropout=0.5) -> None:
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 11)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 32, 3)
        self.attention = nn.Conv2d(64, 1, 3, stride=1, padding='same')
        self.max2 = nn.MaxPool2d(2, 2)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.linear = nn.Linear(32*32*32, 64)

        self.out = OutBlock(64, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        a = self.attention(y)
        y = self.tanh3(y)
        y = y * torch.sigmoid(self.max2(a)) #attention
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        if self.dropout is not None:
            y = self.dropout(y)
        out = self.out(y)
        return out

class ClassificationModelTinyAttention2(nn.Module):

    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1) -> None:
        super().__init__()
        block = ReluBlockCBAM #TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 11)
        self.tanh2 = block(32, 64, 5)
        self.tanh3 = block(64, 64, 3)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*64, 128)

        self.out = OutBlock(128, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        
        y = self.average(y)
        y = self.flatten(y)

        y = torch.relu(self.linear(y))
        out = self.out(y)
        return out

class ClassificationModelTinyAttentionSmallImages(nn.Module):

    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1, dropout=0.5) -> None:
        super().__init__()
        block = ReluBlockBN if bn else ReluBlock
        self.tanh1 = block(in_channels, 32, 7)
        self.tanh2 = block(32, 96, 3)
        self.tanh3 = block(96, 256, 3)
        self.attention = nn.Conv2d(96, 1, 3, stride=1, padding='same')
        self.max2 = nn.MaxPool2d(2, 2)
        self.average = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.linear = nn.Linear(256, 64)

        self.out = OutBlock(64, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        a = self.attention(y)
        y = self.tanh3(y)
        y = y * torch.sigmoid(self.max2(a)) #attention
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        if self.dropout is not None:
            y = self.dropout(y)
        out = self.out(y)
        return out

class ScatClassificationModelTinyAttention2(ClassificationModelTinyAttention2):
    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1, upsample=True) -> None:
        super().__init__(num_classes, regression, bn, 7*in_channels)
        self.scat = ScatBlock(upsample=upsample)
    
    def forward(self, x):
        x = self.scat(x)
        return super().forward(x)   

class ScatClassificationModelTinyAttentionNew(ClassificationModelTinyAttentionNew):
    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1, upsample=True, dropout=0.5) -> None:
        super().__init__(num_classes, regression, bn, 7*in_channels, dropout=dropout)
        self.scat = ScatBlock(upsample=upsample)
    
    def forward(self, x):
        x = self.scat(x)
        return super().forward(x)

class ScatClassificationModelTinyAttention(ClassificationModelTinyAttention):
    def __init__(self, num_classes=4, regression=False, bn=False, in_channels=1, upsample=True) -> None:
        super().__init__(num_classes, regression, bn, 7*in_channels)
        self.scat = ScatBlock(upsample=upsample)
    
    def forward(self, x):
        x = self.scat(x)
        return super().forward(x)

class ClassificationModelSmallBN(nn.Module):

    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.tanh1 = ReluBlockBN(1, 64, 5)
        self.tanh2 = TanhBlock(64, 128, 3)
        self.tanh3 = TanhBlock(128, 256, 1)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*256, 64)
        self.out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        out = self.out(y)
        out = F.log_softmax(out, dim=1)

        return out

class RegressionModelSmall(nn.Module):

    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.tanh1 = TanhBlock(1, 64, 5)
        self.tanh2 = TanhBlock(64, 128, 3)
        self.tanh3 = TanhBlock(128, 256, 1)
        self.average = nn.AdaptiveAvgPool2d((32, 32))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*256, 64)
        self.out = nn.Linear(64, 1)
        
        self.scale = num_classes / 2
    
    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        
        out = self.out(y)
        out = torch.tanh(out / self.scale) * self.scale + (self.scale)

        return out

class ClassificationModelCBAM(nn.Module):

    def __init__(self, in_channels=1, num_classes=4, regression=True, bn=False) -> None:
        super().__init__()
        self.tanh1 = TanhBlockBN(in_channels, 64, 7)
        self.tanh2 = TanhBlockBN(64, 128, 5)
        self.tanh3 = ReluBlockCBAM(128, 256, 3)
        self.tanh4 = ReluBlockCBAM(256, 256, 1)
        self.average = nn.AdaptiveAvgPool2d((16, 16))
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16*16*256, 256)
        self.linear2 = nn.Linear(256, 64)
        self.out = OutBlock(64, num_classes, regression=regression)
    
    def forward(self, x):
        y = self.tanh1(x)
        y = self.tanh2(y)
        y = self.tanh3(y)
        y = self.tanh4(y)
        y = self.average(y)

        y = self.flatten(y)
        y = torch.relu(self.linear(y))
        y = torch.relu(self.linear2(y))
        out = self.out(y)

        return out


class ClassificationModel3D(nn.Module):

    def __init__(self, network_2d) -> None:
        super().__init__()

        self.network_2d = network_2d


class VAEModelTinyAttention(nn.Module):
    def __init__(self, in_channels=1, latent_dim=100, img_size=256, bn=True, checkpoint_path='models/regression/20231123_121507/ClassificationModelTinyAttention_best'):
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 11)
        self.tanh2 = block(32, 64, 5)
        self.max1 = nn.MaxPool2d(4, 4)
        self.tanh3 = block(64, 32, 3)
        self.attention = nn.Conv2d(64, 1, 3, stride=1, padding='same')
        self.max2 = nn.MaxPool2d(2, 2)

        # Load weights from checkpoint for the encoder
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)

        s = img_size // 32
        self.fc_mu = nn.Linear(32*s*s, latent_dim)
        self.fc_logvar = nn.Linear(32*s*s, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 32*s*s)
        self.unflatten = nn.Unflatten(1, (32, s, s))

        self.decoder = nn.Sequential(
            self.fc_decode,
            self.unflatten,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='nearest'),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x64
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='nearest'),  # 128x128
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128x128
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 256x256
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),  # 256x256
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.tanh1(x)
        x = self.tanh2(x)
        x = self.max1(x)
        a = self.attention(x)
        x = self.tanh3(x)
        x = x * self.max2(torch.sigmoid(a)) #attention
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class ClassificationModelFPN(nn.Module):

    def __init__(self, in_channels=1, num_classes=4, regression=False, bn=False) -> None:
        super().__init__()
        block = TanhBlockBN if bn else TanhBlock
        self.tanh1 = block(in_channels, 32, 7)
        self.tanh2 = block(32, 128, 5)
        self.tanh3 = block(128, 256, 3)

        self.average1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten()
        self.average2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten2 = nn.Flatten()       
        self.average3 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten3 = nn.Flatten() 
        
        self.linear1 = nn.Linear(256, 32)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(256, 32)


        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.match_conv1 = nn.Conv2d(128,256,1)
        self.match_conv2 = nn.Conv2d(32,256,1)

        self.out = OutBlock(96, num_classes, regression)

        self.regression = regression
        self.num_classes = num_classes

    def encoder(self):
        return Encoder(self)

    def forward(self, x):
        y1 = self.tanh1(x)
        y2 = self.tanh2(y1)
        y3 = self.tanh3(y2)

        o3 = torch.relu(self.linear1(self.flatten1(self.average1(y3))))
        
        o21 = self.upsample1(y3)
        o22 = torch.relu(self.match_conv1(y2))
        o23 = o21 + o22
        o2 = torch.relu(self.linear2(self.flatten2(self.average2(o23))))

        o11 = self.upsample1(o23)
        o12 = torch.relu(self.match_conv2(y1))
        o1 = torch.relu(self.linear3(self.flatten3(self.average3(o11+o12))))

        y = torch.concat([o1,o2,o3], dim=1)

        out = self.out(y)

        return out