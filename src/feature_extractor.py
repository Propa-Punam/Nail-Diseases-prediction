# -*- coding: utf-8 -*-
"""feature_extractor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HJhrCWhOzjM1kWTYAC1xQiSishDDB3bK
"""

import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove fully connected layers

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x