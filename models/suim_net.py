"""
# SUIM-Net model for underwater image segmentation
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vgg16

# Residual Skip Block (RSB)
class RSB(nn.Module):
    def __init__(self, filters, kernel_size, strides=1, skip=True):
        super(RSB, self).__init__()
        f1, f2, f3, f4 = filters
        self.skip = skip
        
        # sub-block1
        self.conv1 = nn.Conv2d(f1, f1, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.8)
        
        # sub-block2
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.8)
        
        # sub-block3
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3, momentum=0.8)

        # optional skip connection
        if not skip:
            self.conv4 = nn.Conv2d(f1, f4, kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm2d(f4, momentum=0.8)

    def forward(self, x):
        shortcut = x

        # Sub-block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Sub-block 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Sub-block 3
        x = self.bn3(self.conv3(x))

        if not self.skip:
            shortcut = F.relu(self.bn4(self.conv4(shortcut)))

        x = F.relu(x + shortcut)
        return x

# SUIM Encoder with RSB blocks
class SuimEncoderRSB(nn.Module):
    def __init__(self, channels=1):
        super(SuimEncoderRSB, self).__init__()
        
        # Encoder block 1
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, stride=1)
        
        # Encoder block 2
        self.bn1 = nn.BatchNorm2d(64, momentum=0.8)
        self.rsb2a = RSB([64, 64, 128, 128], kernel_size=3, strides=2, skip=False)
        self.rsb2b = RSB([64, 64, 128, 128], kernel_size=3, skip=True)
        
        # Encoder block 3
        self.rsb3a = RSB([128, 128, 256, 256], kernel_size=3, strides=2, skip=False)
        self.rsb3b = RSB([128, 128, 256, 256], kernel_size=3, skip=True)

    def forward(self, x):
        # Encoder block 1
        enc_1 = self.conv1(x)

        # Encoder block 2
        x = F.relu(self.bn1(enc_1))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.rsb2a(x)
        x = self.rsb2b(x)
        enc_2 = x

        # Encoder block 3
        x = self.rsb3a(x)
        x = self.rsb3b(x)
        enc_3 = x

        return [enc_1, enc_2, enc_3]

# SUIM Decoder
class SuimDecoderRSB(nn.Module):
    def __init__(self, n_classes):
        super(SuimDecoderRSB, self).__init__()
        
        # Decoder block 1
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.8)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder block 2
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder block 3
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.8)

        # Output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def forward(self, enc_inputs):
        enc_1, enc_2, enc_3 = enc_inputs
        
        # Decoder block 1
        x = F.relu(self.bn1(self.conv1(enc_3)))
        x = self.upsample1(x)
        
        # Decoder block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample2(x)

        # Decoder block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Output layer
        out = torch.sigmoid(self.out_conv(x))
        return out

# Model class to combine encoder and decoder
class SUIMNet(nn.Module):
    def __init__(self, base='RSB', im_res=(320, 240), n_classes=5):
        super(SUIMNet, self).__init__()
        self.base = base
        self.n_classes = n_classes

        if self.base == 'RSB':
            self.encoder = SuimEncoderRSB(channels=3)
            self.decoder = SuimDecoderRSB(n_classes)

        elif self.base == 'VGG':
            self.vgg = vgg16(pretrained=True)
            self.vgg_features = list(self.vgg.features)[:23]  # Use layers up to block4_conv3

            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
            )

    def forward(self, x):
        if self.base == 'RSB':
            enc_outputs = self.encoder(x)
            out = self.decoder(enc_outputs)
        elif self.base == 'VGG':
            x = self.vgg_features(x)
            out = self.decoder(x)

        return out

if __name__ == "__main__":
    model = SUIMNet(base='RSB', im_res=(320, 240), n_classes=5)
    print(model)