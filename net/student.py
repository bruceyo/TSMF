import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .resnet import resnet18 as ResNet
#from .resnet import ResNet34 as ResNet
import numpy as np

import sys

class Model(nn.Module):
    r"""Teacher-Student networks.

    Args:
        num_class (int): Number of classes for the classification task
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, num_class, **kwargs):
        super().__init__()

        self.resnet = ResNet(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_class)

        self.stgcn = ''

    def forward(self, x_, x_rgb):

        predict, feature = self.stgcn.extract_feature(x_)
        intensity_s = (feature*feature).sum(dim=1)**0.5
        intensity_s = intensity_s.cpu().detach().numpy()
        #feature = np.abs(intensity)
        feature_s = np.abs(intensity_s)
        #feature = feature / feature.mean()
        feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
        N, C, T, V, M = x_.size()

        weight = np.full((N, 1, 225, 225),0) # full_rgb_crop_sklweight_auto_1
        for n in range(N):
            if True:#feature_s[n, :, :, 0].mean(1).mean(0) > feature_s[n, :, :, 1].mean(1).mean(0):
                for j, v in enumerate([3, 11, 7, 18, 14]):
                    feature = feature_s[n, :, v, 0]
                    temp = np.partition(-feature, 15)
                    #print('feature ', v, ' ', feature, -temp[:15].mean())
                    feature = -temp[:15].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]
            else:
                for j, v in enumerate([3, 11, 7, 18, 14]):

                    feature = feature_s[n, :, v, 1]
                    temp = np.partition(-feature, 15)
                    #print('feature ', v, ' ', feature, -temp[:15].mean())
                    feature = -temp[:15].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]

        weight_cuda = torch.from_numpy(weight).float().cuda()
        weight_cuda = weight_cuda / 127

        # fused representation
        rgb_weighted = x_rgb * weight_cuda

        out = self.resnet(rgb_weighted)

        return out#, weight_cuda, rgb_weighted


    def extract_feature(self, x_rgb):

        x = self.resnet.conv1(x_rgb)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        out = self.classifier_(x)

        return out, x
