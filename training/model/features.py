import torch
from torch import nn
from pretrainedmodels import xception
from torchvision.models import efficientnet_b4, resnet50
# from timm.models import swin_base_patch4_window12_384, swin_base_patch4_window7_224

clip_len = 8

class Xception_Fea(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = xception(num_classes=2, pretrained=False)
        pretrained_path = '/irip/tanlingfeng_2020/v2g/init_weights/xception_HQ_299.ckpt'
        net.load_state_dict(torch.load(pretrained_path))
        print("load xception from:" + pretrained_path)
        self.conv1 = nn.Sequential(
            net.conv1, net.bn1, net.relu,
            net.conv2, net.bn2, net.relu
        )

        self.features = nn.Sequential(
            # entry flow -> 728x19x19
            net.block1, net.block2, net.block3,
            
            # middle flow -> 728x19x19
            net.block4, 
            net.block5,
            net.block6,
            net.block7, net.block8,
            net.block9, net.block10, net.block11,
            
            # exit flow -> 1024x10x10
            net.block12
        )

        self.conv2 = nn.Sequential(
            net.conv3, net.bn3, net.relu,
            net.conv4, net.bn4
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        return x


class ResNet_Fea(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = resnet50(num_classes=2, pretrained=False)
        net.load_state_dict(torch.load('init_weights/resnet-50-HQ.ckpt'))
        self.conv1 = nn.Sequential(
            net.conv1, net.bn1,
            net.relu, net.maxpool
        )

        self.features = nn.Sequential(
            net.layer1, net.layer2,
            net.layer3, net.layer4
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        return x


class EffNet_Fea(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = efficientnet_b4(num_classes=2)
        net.load_state_dict(torch.load('init_weights/effnet_b4_320_HQ.ckpt'))
        # net = efficientnet_b4(True)
        self.features = net.features
        # self.features[0].requires_grad_=False
        # self.features[1].requires_grad_=False

    def forward(self, x):
        x = self.features(x)

        return x


class Swin_Fea(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = swin_base_patch4_window7_224(num_classes=2)
        net.load_state_dict(torch.load('init_weights/swin224-HQ.ckpt'))
        self.features = net

    def forward(self, x):
        x = self.features.patch_embed(x)
        if self.features.absolute_pos_embed is not None:
            x = x + self.features.absolute_pos_embed
        x = self.features.pos_drop(x)
        x = self.features.layers(x)
        x = self.features.norm(x)  # B L C

        B, N, C = x.shape
        
        x = x.transpose(1, 2)
        x = x.view(B, C, 7, 7)

        return x
