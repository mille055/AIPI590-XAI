import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

## takes 3 tensors of length num_classes, concatenates, and uses that for a fully connected layer
class FusionModel(nn.Module):
    def __init__(self, model1, model2, model3, num_classes):
        super(FusionModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.fusion_layer = nn.Linear(num_classes * 3, num_classes)

    def forward(self, x1, x2, x3):
        
        x = torch.cat((x1, x2, x3), dim=0)
        x = self.fusion_layer(x)

        return x

## takes 2 tensors of length num_classes and uses that for a fully connected layer
class PartialFusionModel(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(PartialFusionModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        #self.model3 = model3
        self.fusion_layer = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x1, x2):
        
        x = torch.cat((x1, x2), dim=0)
        x = self.fusion_layer(x)

        return x