# -*- coding: utf-8 -*-
"""hybrid_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HJhrCWhOzjM1kWTYAC1xQiSishDDB3bK
"""

import torch
import torch.nn as nn
from src.feature_extractor import FeatureExtractor
from src.vit_classifier import VisionTransformerClassifier

class HybridModel(nn.Module):
    def __init__(self, feature_extractor, vit_model):
        super(HybridModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.vit_model = vit_model

    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.vit_model(features)
        return x