import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock
from .vision_transformer import vit_base

import sys
sys.path.append('..')
from CLIP import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        y = self.avgpool(x)
        x = torch.cat((y.reshape(y.shape[0], 512, -1), x.reshape(x.shape[0], 512, -1)), dim=2)
        
        x = x.transpose(1,2)

        return x


def get_image_extractor(arch = 'dino'):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''

    if arch == 'resnet18_conv':
        model = ResNet18_conv()
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())

    elif arch == 'dino':
        model = vit_base()
        state_dict = torch.load('./pretrain/dino_vitbase16_pretrain.pth')
        model.load_state_dict(state_dict, strict=True)

    return model

